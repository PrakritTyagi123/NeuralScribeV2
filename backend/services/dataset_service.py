"""
NeuralScribe v2 — Dataset service.
Reads EMNIST directly via torchvision, preprocesses, augments, caches.
No folder conversion step — reads the torchvision format directly.
"""

import time, asyncio
import numpy as np
import torch
from torch.utils.data import TensorDataset
from typing import Dict, Any, Optional, Callable

from ..utils.config import (Config, ClassRegistry, ProjectConfig, PROJECT_ROOT, get_language_paths, ensure_all_language_dirs)
from ..utils.logging import get_logger
from ..utils.helpers import human_readable_size
from ..ml.preprocess import (bbox_crop_pad, center_of_mass_align, apply_emnist_orientation, smooth_image, DEFAULT_MEAN, DEFAULT_STD)
from ..ml.augmentation import AugmentationPipeline

log = get_logger(__name__)

EMNIST_ROOT = "datasets/english/raw/emnist"


class DatasetService:
    def __init__(self):
        self._project_config = ProjectConfig()
        self._paths = get_language_paths("english")
        self._prep_config = self._load_prep_config()
        self._registry = ClassRegistry(language="english")
        self._is_preparing = False
        self._cancel_requested = False
        self._progress: Dict[str, Any] = {}
        self._cached_dataset = None
        ensure_all_language_dirs()

    def _load_prep_config(self) -> Config:
        p = self._paths.prep_config
        return Config(str(p.relative_to(PROJECT_ROOT))) if p.exists() else Config()

    @property
    def language(self): return "english"
    @property
    def is_preparing(self): return self._is_preparing
    @property
    def progress(self): return self._progress
    @property
    def registry(self): return self._registry

    def cancel(self): self._cancel_requested = True

    def set_language(self, lang): return {"status": "ok", "language": "english", "num_classes": self._registry.num_classes}

    def get_status(self) -> Dict[str, Any]:
        cp = self._paths.dataset_cache
        exists = cp.exists()
        from ..ml.dataset_downloader import get_download_status
        dl = get_download_status()
        return {
            "language": "english",
            "emnist_downloaded": dl["downloaded"],
            "cache_exists": exists,
            "cache_path": str(cp),
            "cache_size": human_readable_size(cp.stat().st_size) if exists else "N/A",
            "is_preparing": self._is_preparing,
            "progress": self._progress,
            "num_classes": self._registry.num_classes,
        }

    def get_config(self): return self._prep_config.to_dict()
    def update_config(self, overrides): self._prep_config.update(overrides)

    async def prepare_dataset(self, ws_callback=None) -> Dict[str, Any]:
        if self._is_preparing: return {"error": "Already in progress"}
        self._is_preparing = True
        self._cancel_requested = False
        self._progress = {"stage": "starting"}
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(None, self._prepare_sync, ws_callback, loop)
            if result.get("status") == "complete":
                self._project_config.set_language_config("english", "dataset_prepared", True)
            return result
        except Exception as e:
            log.error(f"Prep failed: {e}", exc_info=True)
            return {"error": str(e)}
        finally:
            self._is_preparing = False

    def _prepare_sync(self, ws_callback=None, main_loop=None) -> Dict[str, Any]:
        t0 = time.time()
        config = self._prep_config

        def emit(data):
            data["language"] = "english"
            self._progress = data
            if ws_callback and main_loop:
                try: asyncio.run_coroutine_threadsafe(ws_callback(data), main_loop)
                except: pass

        # ── Step 1: Load EMNIST ──
        emit({"stage": "loading_emnist", "processed": 0, "total": 0, "message": "Loading EMNIST..."})
        images, labels = self._load_emnist(config, emit)
        if self._cancel_requested: return {"status": "cancelled"}
        if len(images) == 0:
            return {"error": "No images loaded. Download EMNIST first (go to Download page)."}
        log.info(f"EMNIST loaded: {len(images)} samples, {len(np.unique(labels))} classes")

        # ── Step 2: Preprocess ──
        total = len(images)
        emit({"stage": "preprocessing", "processed": 0, "total": total, "message": "Preprocessing..."})
        images = self._preprocess_all(images, config, emit)
        if self._cancel_requested: return {"status": "cancelled"}

        # ── Step 3: Augment ──
        aug_factor = config.get("augmentation.precompute_factor", 3)
        if aug_factor > 0:
            emit({"stage": "augmenting", "processed": 0, "total": len(images) * aug_factor, "message": f"Augmenting ×{aug_factor}..."})
            images, labels = self._augment(images, labels, config, emit)
            if self._cancel_requested: return {"status": "cancelled"}

        # ── Step 4: Split ──
        emit({"stage": "splitting", "message": "Splitting train/val/test..."})
        splits = self._split(images, labels, config)

        # ── Step 5: Cache ──
        emit({"stage": "caching", "message": "Saving cache..."})
        cache_path = self._paths.dataset_cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "train_images": torch.from_numpy(splits["train_images"]).float() / 255.0,
            "train_labels": torch.from_numpy(splits["train_labels"]).long(),
            "val_images": torch.from_numpy(splits["val_images"]).float() / 255.0,
            "val_labels": torch.from_numpy(splits["val_labels"]).long(),
            "test_images": torch.from_numpy(splits["test_images"]).float() / 255.0,
            "test_labels": torch.from_numpy(splits["test_labels"]).long(),
            "mean": DEFAULT_MEAN, "std": DEFAULT_STD, "num_classes": self._registry.num_classes,
        }, cache_path)

        elapsed = time.time() - t0
        n = sum(len(splits[k]) for k in ["train_images", "val_images", "test_images"])
        result = {
            "status": "complete", "language": "english", "total_samples": n,
            "train_samples": len(splits["train_images"]),
            "val_samples": len(splits["val_images"]),
            "test_samples": len(splits["test_images"]),
            "cache_size": human_readable_size(cache_path.stat().st_size),
            "elapsed_seconds": round(elapsed, 1),
            "num_classes": self._registry.num_classes,
        }
        emit({"stage": "complete", **result})
        return result

    def _load_emnist(self, config, emit):
        """Load EMNIST balanced via torchvision, filter to 36 classes."""
        import torchvision

        emnist_root = str(PROJECT_ROOT / config.get("dataset.emnist_root", EMNIST_ROOT))

        # Build EMNIST label index → our class ID mapping
        # EMNIST balanced: 0-9 = digits, 10-35 = uppercase A-Z, 36-46 = lowercase subset
        label_map = {}
        # Digits 0-9
        for i in range(10):
            cid = self._registry.label_to_id(str(i))
            if cid is not None: label_map[i] = cid
        # Uppercase A-Z
        for i in range(26):
            cid = self._registry.label_to_id(chr(65 + i))
            if cid is not None: label_map[10 + i] = cid
        # Lowercase subset: EMNIST idx 36-46 = a, b, d, e, f, g, h, n, q, r, t
        emnist_lower = ['a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']
        for i, ch in enumerate(emnist_lower):
            cid = self._registry.label_to_id(ch)
            if cid is not None: label_map[36 + i] = cid

        log.info(f"EMNIST label map: {len(label_map)} classes mapped")

        all_images, all_labels = [], []
        for is_train in [True, False]:
            try:
                ds = torchvision.datasets.EMNIST(root=emnist_root, split="balanced",
                                                  train=is_train, download=True)
            except Exception as e:
                log.error(f"Cannot load EMNIST: {e}")
                return np.array([]), np.array([])

            for i in range(len(ds)):
                if self._cancel_requested:
                    return np.array([]), np.array([])
                img, label = ds[i]
                if label in label_map:
                    arr = np.array(img, dtype=np.float32).squeeze()
                    arr = apply_emnist_orientation(arr)
                    all_images.append(arr)
                    all_labels.append(label_map[label])
                if i % 10000 == 0:
                    emit({"stage": "loading_emnist", "processed": len(all_images),
                          "total": len(ds), "split": "train" if is_train else "test",
                          "message": f"Loading EMNIST: {len(all_images):,} samples"})

        if not all_images:
            return np.zeros((0, 28, 28), dtype=np.float32), np.zeros(0, dtype=np.int64)
        return np.stack(all_images), np.array(all_labels, dtype=np.int64)

    def _preprocess_all(self, images, config, emit):
        total = len(images)
        do_com = config.get("preprocessing.center_of_mass", True)
        do_smooth = config.get("preprocessing.smoothing.enabled", True)
        sigma = config.get("preprocessing.smoothing.sigma", 0.5)
        target = config.get("preprocessing.image_size", 28)
        processed = []
        t0 = time.time()
        for i in range(total):
            if self._cancel_requested: break
            img = bbox_crop_pad(images[i])
            if do_com: img = center_of_mass_align(img, target)
            if do_smooth: img = smooth_image(img, sigma)
            processed.append(img)
            if i % 5000 == 0 and i > 0:
                rate = i / (time.time() - t0)
                eta = (total - i) / rate if rate > 0 else 0
                emit({"stage": "preprocessing", "processed": i, "total": total,
                      "samples_per_sec": round(rate),
                      "message": f"Preprocessing: {i:,}/{total:,} ({rate:.0f}/s, ETA {int(eta)}s)"})
        return np.stack(processed) if processed else np.zeros((0, 28, 28))

    def _augment(self, images, labels, config, emit):
        factor = config.get("augmentation.precompute_factor", 3)
        aug = AugmentationPipeline(
            rotation_range=tuple(config.get("augmentation.rotation_range", [-12, 12])),
            translation_range=tuple(config.get("augmentation.translation_range", [-2, 2])),
            scale_range=tuple(config.get("augmentation.scale_range", [0.9, 1.1])),
            shear_range=tuple(config.get("augmentation.shear_range", [-5, 5])))
        aug_imgs, aug_lbls = [images], [labels]
        done, total = 0, len(images) * factor
        t0 = time.time()
        for f in range(factor):
            batch = []
            for i in range(len(images)):
                if self._cancel_requested: break
                batch.append(aug(images[i]))
                done += 1
                if done % 10000 == 0:
                    rate = done / max(time.time() - t0, 0.1)
                    eta = (total - done) / rate if rate > 0 else 0
                    emit({"stage": "augmenting", "processed": done, "total": total,
                          "samples_per_sec": round(rate),
                          "message": f"Augmenting {f+1}/{factor}: {done:,}/{total:,} ({rate:.0f}/s, ETA {int(eta)}s)"})
            aug_imgs.append(np.stack(batch))
            aug_lbls.append(labels.copy())
        return np.concatenate(aug_imgs), np.concatenate(aug_lbls)

    def _split(self, images, labels, config):
        test_r = config.get("split.test_ratio", 0.15)
        val_r = config.get("split.val_ratio", 0.1)
        seed = config.get("split.seed", 42)
        rng = np.random.RandomState(seed)
        if config.get("split.stratified", True):
            tr, va, te = [], [], []
            for cls in np.unique(labels):
                idx = rng.permutation(np.where(labels == cls)[0])
                nt, nv = int(len(idx) * test_r), int(len(idx) * val_r)
                te.append(idx[:nt]); va.append(idx[nt:nt+nv]); tr.append(idx[nt+nv:])
            ti, vi, tei = np.concatenate(tr), np.concatenate(va), np.concatenate(te)
        else:
            idx = rng.permutation(len(images))
            nt, nv = int(len(idx) * test_r), int(len(idx) * val_r)
            tei, vi, ti = idx[:nt], idx[nt:nt+nv], idx[nt+nv:]
        return {"train_images": images[ti], "train_labels": labels[ti],
                "val_images": images[vi], "val_labels": labels[vi],
                "test_images": images[tei], "test_labels": labels[tei]}

    # ── Cache / DataLoader ──

    def load_cached_dataset(self):
        cp = self._paths.dataset_cache
        if not cp.exists(): return None
        log.info(f"Loading cache: {cp}")
        data = torch.load(cp, map_location="cpu", weights_only=False)
        self._cached_dataset = data
        return data

    def get_cached_dataset(self):
        return self._cached_dataset or self.load_cached_dataset()

    def get_dataloader(self, split="train", batch_size=256, num_workers=4, pin_memory=True, shuffle=True):
        data = self.get_cached_dataset()
        if not data: return None
        images = (data[f"{split}_images"].unsqueeze(1) - data.get("mean", DEFAULT_MEAN)) / data.get("std", DEFAULT_STD)
        ds = TensorDataset(images, data[f"{split}_labels"])
        return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory,
            persistent_workers=num_workers > 0, drop_last=(split == "train"))
