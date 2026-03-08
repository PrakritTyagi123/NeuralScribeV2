"""
NeuralScribe v2 — Dataset service (language-aware).
Handles EMNIST loading, synthetic symbol generation, preprocessing,
augmentation caching, and dataset splitting.
Each language has its own config, registry, cache, and dataset directory.
"""

import os
import time
import numpy as np
import torch
from torch.utils.data import TensorDataset
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Tuple
import asyncio

from ..utils.config import (
    Config, ClassRegistry, ProjectConfig, LanguagePaths,
    PROJECT_ROOT, DEFAULT_LANGUAGE, SUPPORTED_LANGUAGES,
    get_language_paths, ensure_all_language_dirs,
)
from ..utils.logging import get_logger
from ..utils.helpers import human_readable_size, eta_string
from ..ml.preprocess import (
    bbox_crop_pad, center_of_mass_align, apply_emnist_orientation,
    smooth_image, DEFAULT_MEAN, DEFAULT_STD,
)
from ..ml.synthetic import SyntheticSymbolGenerator
from ..ml.augmentation import AugmentationPipeline

log = get_logger(__name__)


def _build_emnist_label_map(registry: ClassRegistry) -> Dict[int, int]:
    """
    Build mapping from EMNIST balanced class index to our registry class IDs.
    Only maps classes that exist in the registry (language-dependent).
    """
    mapping = {}

    # Digits 0-9: EMNIST idx 0-9
    for i in range(10):
        our_id = registry.label_to_id(str(i))
        if our_id is not None:
            mapping[i] = our_id

    # Uppercase A-Z: EMNIST idx 10-35
    for i in range(26):
        ch = chr(65 + i)  # A-Z
        our_id = registry.label_to_id(ch)
        if our_id is not None:
            mapping[10 + i] = our_id

    # Lowercase subset in balanced: EMNIST idx 36-46
    emnist_lower_chars = ['a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']
    for i, ch in enumerate(emnist_lower_chars):
        our_id = registry.label_to_id(ch)
        if our_id is not None:
            mapping[36 + i] = our_id

    return mapping


class DatasetService:
    """
    Manages dataset preparation per language.
    Call set_language() to switch context.
    """

    def __init__(self):
        self._project_config = ProjectConfig()
        self._language = self._project_config.selected_language
        self._paths = get_language_paths(self._language)
        self._prep_config = self._load_prep_config()
        self._registry = ClassRegistry(language=self._language)
        self._is_preparing = False
        self._cancel_requested = False
        self._progress: Dict[str, Any] = {}
        self._cached_dataset: Optional[Dict[str, torch.Tensor]] = None

        ensure_all_language_dirs()

    def _load_prep_config(self) -> Config:
        """Load prep config for current language, fallback to empty."""
        config_path = self._paths.prep_config
        if config_path.exists():
            return Config(str(config_path.relative_to(PROJECT_ROOT)))
        return Config()

    # ── Language switching ──

    @property
    def language(self) -> str:
        return self._language

    def set_language(self, language: str) -> Dict[str, Any]:
        """Switch to a different language context."""
        if language not in SUPPORTED_LANGUAGES:
            return {"error": f"Unsupported language: {language}"}
        if language == self._language and self._registry.num_classes > 0:
            return {"status": "already_set", "language": language}

        self._language = language
        self._paths = get_language_paths(language)
        self._paths.ensure_dirs()
        self._prep_config = self._load_prep_config()
        self._registry = ClassRegistry(language=language)
        self._cached_dataset = None  # Clear cached data for old language

        # Update project config
        self._project_config.set_ui_state("prep_language", language)

        log.info(f"Dataset service switched to language: {language} ({self._registry.num_classes} classes)")
        return {
            "status": "switched",
            "language": language,
            "num_classes": self._registry.num_classes,
            "registry_status": self._registry.status,
        }

    # ── Properties ──

    @property
    def is_preparing(self) -> bool:
        return self._is_preparing

    @property
    def progress(self) -> Dict[str, Any]:
        return self._progress

    @property
    def registry(self) -> ClassRegistry:
        return self._registry

    def cancel(self):
        self._cancel_requested = True

    def get_status(self) -> Dict[str, Any]:
        cache_path = self._paths.dataset_cache
        cache_exists = cache_path.exists()
        cache_size = human_readable_size(cache_path.stat().st_size) if cache_exists else "N/A"

        return {
            "language": self._language,
            "cache_exists": cache_exists,
            "cache_path": str(cache_path),
            "cache_size": cache_size,
            "is_preparing": self._is_preparing,
            "progress": self._progress,
            "num_classes": self._registry.num_classes,
            "registry_status": self._registry.status,
        }

    def get_config(self) -> Dict[str, Any]:
        return self._prep_config.to_dict()

    def update_config(self, overrides: Dict[str, Any]) -> None:
        self._prep_config.update(overrides)

    # ── Dataset preparation ──

    async def prepare_dataset(self, ws_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Run the full dataset preparation pipeline for the current language."""
        if self._is_preparing:
            return {"error": "Dataset preparation already in progress"}

        # Check if language is placeholder
        if self._registry.status == "placeholder":
            return {"error": f"Language '{self._language}' is a placeholder. Dataset not available yet."}

        self._is_preparing = True
        self._cancel_requested = False
        self._progress = {"stage": "starting", "processed": 0, "total": 0, "language": self._language}

        main_loop = asyncio.get_event_loop()

        try:
            result = await main_loop.run_in_executor(
                None, self._prepare_sync, ws_callback, main_loop
            )
            # Update project config on success
            if result.get("status") == "complete":
                self._project_config.set_language_config(self._language, "dataset_prepared", True)
            return result
        except Exception as e:
            log.error(f"Dataset preparation failed for {self._language}: {e}", exc_info=True)
            return {"error": str(e)}
        finally:
            self._is_preparing = False

    def _prepare_sync(self, ws_callback: Optional[Callable] = None, main_loop=None) -> Dict[str, Any]:
        """Synchronous preparation pipeline."""
        start_time = time.time()
        config = self._prep_config

        def emit(data: Dict):
            data["language"] = self._language
            self._progress = data
            if ws_callback and main_loop:
                try:
                    asyncio.run_coroutine_threadsafe(ws_callback(data), main_loop)
                except Exception:
                    pass

        emit({"stage": "loading_emnist", "processed": 0, "total": 0, "message": f"Loading EMNIST for {self._language}..."})

        # ── Step 1: Load EMNIST ──
        emnist_images, emnist_labels = self._load_emnist(config, emit)
        if self._cancel_requested:
            return {"status": "cancelled"}

        log.info(f"[{self._language}] EMNIST loaded: {len(emnist_images)} samples")

        # ── Step 2: Generate synthetic symbols (if enabled) ──
        synth_images, synth_labels = self._generate_synthetic(config, emit)
        if self._cancel_requested:
            return {"status": "cancelled"}

        if len(synth_images) > 0:
            log.info(f"[{self._language}] Synthetic generated: {len(synth_images)} samples")

        # ── Step 3: Merge datasets ──
        emit({"stage": "merging", "message": "Merging datasets..."})
        if len(synth_images) > 0:
            all_images = np.concatenate([emnist_images, synth_images], axis=0)
            all_labels = np.concatenate([emnist_labels, synth_labels], axis=0)
        else:
            all_images = emnist_images
            all_labels = emnist_labels

        total_samples = len(all_images)
        log.info(f"[{self._language}] Merged dataset: {total_samples} samples")

        # ── Step 4: Preprocess ──
        emit({"stage": "preprocessing", "processed": 0, "total": total_samples, "message": "Preprocessing..."})
        all_images = self._preprocess_all(all_images, config, emit)
        if self._cancel_requested:
            return {"status": "cancelled"}

        # ── Step 5: Augmentation (precompute if enabled) ──
        precompute_enabled = config.get("augmentation.precompute", False)
        precompute_factor = config.get("augmentation.precompute_factor", 3)
        if precompute_enabled:
            emit({"stage": "augmenting", "processed": 0, "total": total_samples, "message": "Augmenting..."})
            all_images, all_labels = self._precompute_augmentations(
                all_images, all_labels, config, emit
            )
            if self._cancel_requested:
                return {"status": "cancelled"}

        # ── Step 6: Split ──
        emit({"stage": "splitting", "message": "Creating train/val/test splits..."})
        splits = self._split_dataset(all_images, all_labels, config)

        # ── Step 7: Cache ──
        emit({"stage": "caching", "message": "Saving cached dataset..."})
        cache_path = self._paths.dataset_cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        cache_data = {
            "train_images": torch.from_numpy(splits["train_images"]).float() / 255.0,
            "train_labels": torch.from_numpy(splits["train_labels"]).long(),
            "val_images": torch.from_numpy(splits["val_images"]).float() / 255.0,
            "val_labels": torch.from_numpy(splits["val_labels"]).long(),
            "test_images": torch.from_numpy(splits["test_images"]).float() / 255.0,
            "test_labels": torch.from_numpy(splits["test_labels"]).long(),
            "mean": DEFAULT_MEAN,
            "std": DEFAULT_STD,
            "num_classes": self._registry.num_classes,
            "language": self._language,
            "class_registry": self._registry.to_dict(),
        }

        torch.save(cache_data, cache_path)
        elapsed = time.time() - start_time

        result = {
            "status": "complete",
            "language": self._language,
            "total_samples": len(splits["train_images"]) + len(splits["val_images"]) + len(splits["test_images"]),
            "train_samples": len(splits["train_images"]),
            "val_samples": len(splits["val_images"]),
            "test_samples": len(splits["test_images"]),
            "cache_path": str(cache_path),
            "cache_size": human_readable_size(cache_path.stat().st_size),
            "elapsed_seconds": round(elapsed, 1),
            "num_classes": self._registry.num_classes,
        }

        emit({"stage": "complete", **result})
        return result

    def _load_emnist(self, config: Config, emit: Callable) -> Tuple[np.ndarray, np.ndarray]:
        """Load EMNIST balanced dataset, filtered to current language's classes."""
        import torchvision

        emnist_root = str(PROJECT_ROOT / config.get("dataset.emnist_root", "data/raw/emnist"))
        auto_download = config.get("dataset.auto_download", True)

        label_map = _build_emnist_label_map(self._registry)

        all_images = []
        all_labels = []

        for split_name in ["train", "test"]:
            dataset = torchvision.datasets.EMNIST(
                root=emnist_root,
                split="balanced",
                train=(split_name == "train"),
                download=auto_download,
            )

            for i in range(len(dataset)):
                if self._cancel_requested:
                    return np.array([]), np.array([])

                img, label = dataset[i]
                img_array = np.array(img, dtype=np.float32).squeeze()

                # Only keep classes mapped in our registry
                if label in label_map:
                    our_label = label_map[label]
                    img_array = apply_emnist_orientation(img_array)
                    all_images.append(img_array)
                    all_labels.append(our_label)

                if i % 10000 == 0:
                    emit({
                        "stage": "loading_emnist",
                        "processed": len(all_images),
                        "total": len(dataset),
                        "split": split_name,
                        "message": f"Loading EMNIST {split_name}: {len(all_images)} samples",
                    })

        if len(all_images) == 0:
            return np.zeros((0, 28, 28), dtype=np.float32), np.zeros(0, dtype=np.int64)

        return np.stack(all_images, axis=0), np.array(all_labels, dtype=np.int64)

    def _generate_synthetic(self, config: Config, emit: Callable) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic images for classes not covered by EMNIST."""
        if not config.get("synthetic.enabled", False):
            return np.zeros((0, 28, 28), dtype=np.float32), np.zeros(0, dtype=np.int64)

        # Determine EMNIST-covered IDs
        emnist_covered_ids = set()
        for i in range(10):
            cid = self._registry.label_to_id(str(i))
            if cid is not None:
                emnist_covered_ids.add(cid)
        for i in range(26):
            cid = self._registry.label_to_id(chr(65 + i))
            if cid is not None:
                emnist_covered_ids.add(cid)
        emnist_lower_chars = ['a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']
        for ch in emnist_lower_chars:
            cid = self._registry.label_to_id(ch)
            if cid is not None:
                emnist_covered_ids.add(cid)

        synthetic_classes = [
            c for c in self._registry.classes if c["id"] not in emnist_covered_ids
        ]

        if not synthetic_classes:
            return np.zeros((0, 28, 28), dtype=np.float32), np.zeros(0, dtype=np.int64)

        generator = SyntheticSymbolGenerator(
            fonts_dir=str(PROJECT_ROOT / config.get("synthetic.fonts_dir", "data/fonts")),
            font_sizes=config.get("synthetic.font_sizes", [18, 20, 22, 24]),
            stroke_widths=config.get("synthetic.stroke_widths", [1, 2, 3]),
            samples_per_symbol=config.get("synthetic.samples_per_symbol", 1500),
            noise_level=config.get("synthetic.noise_level", 0.02),
            blur_sigma_range=config.get("synthetic.blur_sigma_range", [0.3, 0.8]),
        )

        def progress_cb(done, total, label):
            emit({
                "stage": "generating_synthetic",
                "processed": done,
                "total": total,
                "message": f"Generating: {label} ({done}/{total})",
            })

        images, labels = generator.generate_all(synthetic_classes, progress_callback=progress_cb)
        return images, labels

    def _preprocess_all(self, images: np.ndarray, config: Config, emit: Callable) -> np.ndarray:
        """Apply preprocessing pipeline to all images."""
        total = len(images)
        target_size = config.get("preprocessing.image_size", 28)
        do_com = config.get("preprocessing.center_of_mass", True)
        do_smooth = config.get("preprocessing.smoothing.enabled", True)
        sigma = config.get("preprocessing.smoothing.sigma", 0.5)

        processed = []
        start_time = time.time()

        for i in range(total):
            if self._cancel_requested:
                return np.array(processed) if processed else np.zeros((0, 28, 28), dtype=np.float32)

            img = images[i]
            img = bbox_crop_pad(img)
            if do_com:
                img = center_of_mass_align(img, target_size)
            if do_smooth:
                img = smooth_image(img, sigma)
            processed.append(img)

            if i % 5000 == 0 and i > 0:
                elapsed = time.time() - start_time
                rate = i / elapsed
                eta = (total - i) / rate if rate > 0 else 0
                emit({
                    "stage": "preprocessing",
                    "processed": i,
                    "total": total,
                    "samples_per_sec": round(rate),
                    "eta_seconds": round(eta),
                    "message": f"Preprocessing: {i}/{total} ({rate:.0f}/s)",
                })

        return np.stack(processed, axis=0)

    def _precompute_augmentations(
        self, images: np.ndarray, labels: np.ndarray, config: Config, emit: Callable
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Precompute augmented copies of the dataset."""
        factor = config.get("augmentation.precompute_factor", 3)
        aug = AugmentationPipeline(
            rotation_range=tuple(config.get("augmentation.rotation_range", [-12, 12])),
            translation_range=tuple(config.get("augmentation.translation_range", [-2, 2])),
            scale_range=tuple(config.get("augmentation.scale_range", [0.9, 1.1])),
            shear_range=tuple(config.get("augmentation.shear_range", [-5, 5])),
        )

        aug_images = [images]
        aug_labels = [labels]
        total = len(images) * factor
        done = 0
        start_time = time.time()

        for f in range(factor):
            batch_imgs = []
            for i in range(len(images)):
                if self._cancel_requested:
                    break
                batch_imgs.append(aug(images[i]))
                done += 1
                if done % 10000 == 0:
                    elapsed = time.time() - start_time
                    rate = done / elapsed if elapsed > 0 else 0
                    emit({
                        "stage": "augmenting",
                        "processed": done,
                        "total": total,
                        "samples_per_sec": round(rate),
                        "message": f"Augmenting pass {f+1}/{factor}: {done}/{total}",
                    })
            aug_images.append(np.stack(batch_imgs, axis=0))
            aug_labels.append(labels.copy())

        return np.concatenate(aug_images, axis=0), np.concatenate(aug_labels, axis=0)

    def _split_dataset(
        self, images: np.ndarray, labels: np.ndarray, config: Config
    ) -> Dict[str, np.ndarray]:
        """Split into train/val/test with optional stratification."""
        test_ratio = config.get("split.test_ratio", 0.15)
        val_ratio = config.get("split.val_ratio", 0.10)
        seed = config.get("split.seed", 42)
        stratified = config.get("split.stratified", False)

        rng = np.random.RandomState(seed)

        if not stratified:
            n = len(images)
            indices = rng.permutation(n)
            test_size = int(n * test_ratio)
            val_size = int(n * val_ratio)
            test_idx = indices[:test_size]
            val_idx = indices[test_size:test_size + val_size]
            train_idx = indices[test_size + val_size:]
        else:
            unique_labels = np.unique(labels)
            train_idx_list, val_idx_list, test_idx_list = [], [], []
            for cls in unique_labels:
                cls_indices = np.where(labels == cls)[0]
                if cls_indices.size == 0:
                    continue
                cls_perm = rng.permutation(cls_indices)
                n_cls = cls_perm.size
                cls_test_size = int(n_cls * test_ratio)
                cls_val_size = int(n_cls * val_ratio)
                test_idx_list.append(cls_perm[:cls_test_size])
                val_idx_list.append(cls_perm[cls_test_size:cls_test_size + cls_val_size])
                train_idx_list.append(cls_perm[cls_test_size + cls_val_size:])
            train_idx = np.concatenate(train_idx_list) if train_idx_list else np.array([], dtype=int)
            val_idx = np.concatenate(val_idx_list) if val_idx_list else np.array([], dtype=int)
            test_idx = np.concatenate(test_idx_list) if test_idx_list else np.array([], dtype=int)

        return {
            "train_images": images[train_idx],
            "train_labels": labels[train_idx],
            "val_images": images[val_idx],
            "val_labels": labels[val_idx],
            "test_images": images[test_idx],
            "test_labels": labels[test_idx],
        }

    # ── Cache / DataLoader ──

    def load_cached_dataset(self) -> Optional[Dict[str, torch.Tensor]]:
        """Load the cached dataset for current language from disk."""
        cache_path = self._paths.dataset_cache
        if not cache_path.exists():
            log.warning(f"[{self._language}] Cache not found: {cache_path}")
            return None
        log.info(f"[{self._language}] Loading cached dataset from {cache_path}")
        data = torch.load(cache_path, map_location="cpu", weights_only=False)
        self._cached_dataset = data
        return data

    def get_cached_dataset(self) -> Optional[Dict[str, torch.Tensor]]:
        if self._cached_dataset is None:
            return self.load_cached_dataset()
        return self._cached_dataset

    def get_dataloader(
        self,
        split: str = "train",
        batch_size: int = 256,
        num_workers: int = 4,
        pin_memory: bool = True,
        shuffle: bool = True,
    ) -> Optional[torch.utils.data.DataLoader]:
        """Create a DataLoader for the given split."""
        data = self.get_cached_dataset()
        if data is None:
            return None
        images = data[f"{split}_images"]
        labels = data[f"{split}_labels"]
        mean = data.get("mean", DEFAULT_MEAN)
        std = data.get("std", DEFAULT_STD)
        images = (images.unsqueeze(1) - mean) / std
        dataset = TensorDataset(images, labels)
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory,
            persistent_workers=num_workers > 0, drop_last=(split == "train"),
        )