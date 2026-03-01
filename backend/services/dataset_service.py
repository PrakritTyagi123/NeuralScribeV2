"""
Dataset service for NeuralScribe v2.
Handles EMNIST loading, synthetic symbol generation, preprocessing,
augmentation caching, and dataset splitting.
Emits progress via WebSocket callback.
"""

import os
import time
import numpy as np
import torch
from torch.utils.data import TensorDataset
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Tuple
import asyncio

from ..utils.config import Config, ClassRegistry, PROJECT_ROOT
from ..utils.logging import get_logger
from ..utils.helpers import human_readable_size, eta_string
from ..ml.preprocess import (
    bbox_crop_pad, center_of_mass_align, apply_emnist_orientation,
    smooth_image, DEFAULT_MEAN, DEFAULT_STD,
)
from ..ml.synthetic import SyntheticSymbolGenerator
from ..ml.augmentation import AugmentationPipeline

log = get_logger(__name__)


# EMNIST Balanced class mapping (47 classes)
# EMNIST balanced merges some confusable pairs.
# Mapping from EMNIST class index to our class registry IDs.
# EMNIST balanced: 0-9 = digits, 10-35 = uppercase A-Z, 36-46 = lowercase (subset)
# The "balanced" split merges these lowercase with uppercase:
# c, i, j, k, l, m, o, p, s, u, v, w, x, y, z → not separate classes
# So EMNIST balanced has only: a, b, d, e, f, g, h, n, q, r, t
EMNIST_BALANCED_LABELS = (
    list(range(10)) +                    # 0-9 → digits
    [chr(c) for c in range(65, 91)] +    # 10-35 → A-Z
    list("abdeghnqrt")                   # 36-46 → lowercase subset (11 chars)
    # Note: actual EMNIST balanced has specific merged mapping
)


def _build_emnist_label_map(registry: ClassRegistry) -> Dict[int, int]:
    """
    Build mapping from EMNIST balanced class index to our registry class IDs.
    EMNIST balanced split has 47 classes.
    """
    mapping = {}

    # Digits 0-9: EMNIST idx 0-9 → our 0-9
    for i in range(10):
        mapping[i] = i

    # Uppercase A-Z: EMNIST idx 10-35 → our 10-35
    for i in range(26):
        mapping[10 + i] = 10 + i

    # Lowercase subset in balanced: EMNIST idx 36-46
    # EMNIST balanced lowercase order: a, b, d, e, f, g, h, n, q, r, t
    emnist_lower_chars = ['a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']
    for i, ch in enumerate(emnist_lower_chars):
        our_id = registry.label_to_id(ch)
        if our_id is not None:
            mapping[36 + i] = our_id

    return mapping


class DatasetService:
    """
    Manages dataset preparation: loading, preprocessing, caching.
    """

    def __init__(self):
        self._prep_config = Config("configs/prep_v2.yaml")
        self._registry = ClassRegistry()
        self._is_preparing = False
        self._cancel_requested = False
        self._progress: Dict[str, Any] = {}
        self._cached_dataset: Optional[Dict[str, torch.Tensor]] = None

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
        cache_path = PROJECT_ROOT / self._prep_config.get("cache.path", "data/cache/cached_dataset_v2.pt")
        cache_exists = cache_path.exists()
        cache_size = human_readable_size(cache_path.stat().st_size) if cache_exists else "N/A"

        return {
            "cache_exists": cache_exists,
            "cache_path": str(cache_path),
            "cache_size": cache_size,
            "is_preparing": self._is_preparing,
            "progress": self._progress,
            "num_classes": self._registry.num_classes,
        }

    def get_config(self) -> Dict[str, Any]:
        return self._prep_config.to_dict()

    def update_config(self, overrides: Dict[str, Any]) -> None:
        self._prep_config.update(overrides)

    async def prepare_dataset(self, ws_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Run the full dataset preparation pipeline.
        Emits progress via ws_callback(event_dict).
        """
        if self._is_preparing:
            return {"error": "Dataset preparation already in progress"}

        self._is_preparing = True
        self._cancel_requested = False
        self._progress = {"stage": "starting", "processed": 0, "total": 0}

        # Capture main event loop before entering thread
        main_loop = asyncio.get_event_loop()

        try:
            result = await main_loop.run_in_executor(
                None, self._prepare_sync, ws_callback, main_loop
            )
            return result
        except Exception as e:
            log.error(f"Dataset preparation failed: {e}", exc_info=True)
            return {"error": str(e)}
        finally:
            self._is_preparing = False

    def _prepare_sync(self, ws_callback: Optional[Callable] = None, main_loop=None) -> Dict[str, Any]:
        """Synchronous preparation pipeline."""
        start_time = time.time()
        config = self._prep_config

        def emit(data: Dict):
            self._progress = data
            if ws_callback and main_loop:
                try:
                    asyncio.run_coroutine_threadsafe(ws_callback(data), main_loop)
                except Exception:
                    pass

        emit({"stage": "loading_emnist", "processed": 0, "total": 0, "message": "Loading EMNIST..."})

        # ── Step 1: Load EMNIST ──
        emnist_images, emnist_labels = self._load_emnist(config, emit)
        if self._cancel_requested:
            return {"status": "cancelled"}

        log.info(f"EMNIST loaded: {len(emnist_images)} samples")

        # ── Step 2: Generate synthetic symbols ──
        synth_images, synth_labels = self._generate_synthetic(config, emit)
        if self._cancel_requested:
            return {"status": "cancelled"}

        log.info(f"Synthetic generated: {len(synth_images)} samples")

        # ── Step 3: Merge datasets ──
        emit({"stage": "merging", "message": "Merging datasets..."})
        all_images = np.concatenate([emnist_images, synth_images], axis=0)
        all_labels = np.concatenate([emnist_labels, synth_labels], axis=0)

        total_samples = len(all_images)
        log.info(f"Merged dataset: {total_samples} samples")

        # ── Step 4: Preprocess ──
        emit({"stage": "preprocessing", "processed": 0, "total": total_samples, "message": "Preprocessing..."})
        all_images = self._preprocess_all(all_images, config, emit)
        if self._cancel_requested:
            return {"status": "cancelled"}

        # ── Step 5: Augmentation (precompute if enabled) ──
        if config.get("augmentation.precompute", False):
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
        cache_path = PROJECT_ROOT / config.get("cache.path", "data/cache/cached_dataset_v2.pt")
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
            "class_registry": self._registry.to_dict(),
        }

        torch.save(cache_data, cache_path)
        elapsed = time.time() - start_time

        result = {
            "status": "complete",
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
        """Load EMNIST balanced dataset."""
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

                # Map EMNIST label to our registry
                if label in label_map:
                    our_label = label_map[label]
                    # Apply EMNIST orientation correction
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

        return np.stack(all_images, axis=0), np.array(all_labels, dtype=np.int64)

    def _generate_synthetic(self, config: Config, emit: Callable) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic images for classes not covered by EMNIST."""
        if not config.get("synthetic.enabled", True):
            return np.zeros((0, 28, 28), dtype=np.float32), np.zeros(0, dtype=np.int64)

        # Determine which classes need synthetic data
        # EMNIST covers: digits (0-9), uppercase (10-35), and some lowercase
        # We need synthetic for: remaining lowercase, all Greek, all symbols
        emnist_covered_ids = set(range(10))  # digits
        emnist_covered_ids.update(range(10, 36))  # uppercase

        # EMNIST lowercase subset
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
                return np.array(processed)

            img = images[i]

            # Bbox crop + center of mass (only if not already 28x28 and centered)
            if img.shape != (target_size, target_size):
                img = bbox_crop_pad(img)
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
        """Split into train/val/test with stratification."""
        test_ratio = config.get("split.test_ratio", 0.15)
        val_ratio = config.get("split.val_ratio", 0.10)
        seed = config.get("split.seed", 42)

        rng = np.random.RandomState(seed)
        n = len(images)
        indices = rng.permutation(n)

        test_size = int(n * test_ratio)
        val_size = int(n * val_ratio)

        test_idx = indices[:test_size]
        val_idx = indices[test_size:test_size + val_size]
        train_idx = indices[test_size + val_size:]

        return {
            "train_images": images[train_idx],
            "train_labels": labels[train_idx],
            "val_images": images[val_idx],
            "val_labels": labels[val_idx],
            "test_images": images[test_idx],
            "test_labels": labels[test_idx],
        }

    def load_cached_dataset(self) -> Optional[Dict[str, torch.Tensor]]:
        """Load the cached dataset from disk."""
        cache_path = PROJECT_ROOT / self._prep_config.get("cache.path", "data/cache/cached_dataset_v2.pt")
        if not cache_path.exists():
            log.warning(f"Cache not found: {cache_path}")
            return None

        log.info(f"Loading cached dataset from {cache_path}")
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

        # Normalize: images are stored as 0-1 floats
        images = (images.unsqueeze(1) - mean) / std  # (N, 1, 28, 28)

        dataset = TensorDataset(images, labels)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            drop_last=(split == "train"),
        )