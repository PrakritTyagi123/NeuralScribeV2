"""
Training service for NeuralScribe v2.
Runs the training loop with mixed precision, cosine annealing warm restarts,
mixup, and real-time WebSocket progress reporting.
Single training job at a time.
"""

import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
import asyncio

from ..utils.config import Config, ClassRegistry, PROJECT_ROOT
from ..utils.logging import get_logger
from ..utils.helpers import timestamp_str, human_readable_size
from ..utils.metrics import (
    compute_per_class_metrics, compute_confusion_matrix,
    top_confusion_pairs, compute_overall_accuracy, compute_category_accuracy,
)
from ..ml.model import NeuralScribeNet
from ..ml.losses import CombinedLoss, mixup_data, mixup_criterion

log = get_logger(__name__)


def _emit_ws(ws_callback, data):
    """Helper to call async ws_callback from sync code."""
    if ws_callback is None:
        return
    try:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(ws_callback(data))
        loop.close()
    except Exception:
        pass


class TrainingService:
    """Manages model training lifecycle."""

    def __init__(self):
        self._config = Config("configs/train_v2.yaml")
        self._registry = ClassRegistry()
        self._is_training = False
        self._is_paused = False
        self._stop_requested = False
        self._model: Optional[NeuralScribeNet] = None
        self._current_epoch = 0
        self._best_val_acc = 0.0
        self._history: List[Dict[str, Any]] = []
        self._training_state: Dict[str, Any] = {}

        # Device detection
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            self._device = torch.device("cuda")
            log.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            self._device = torch.device("cpu")
            log.warning("CUDA not available — training will use CPU (slow!)")

    # ── Properties ──

    @property
    def is_training(self) -> bool:
        return self._is_training

    @property
    def training_state(self) -> Dict[str, Any]:
        return self._training_state

    @property
    def model(self) -> Optional[NeuralScribeNet]:
        return self._model

    @property
    def history(self) -> List[Dict[str, Any]]:
        return self._history

    # ── Config ──

    def get_config(self) -> Dict[str, Any]:
        return self._config.to_dict()

    def update_config(self, overrides: Dict[str, Any]) -> None:
        self._config.update(overrides)

    # ── Controls ──

    def pause(self):
        self._is_paused = True

    def resume(self):
        self._is_paused = False

    def stop(self):
        self._stop_requested = True

    def get_status(self) -> Dict[str, Any]:
        return {
            "is_training": self._is_training,
            "is_paused": self._is_paused,
            "current_epoch": self._current_epoch,
            "best_val_acc": round(self._best_val_acc, 4),
            "device": str(self._device),
            "history": self._history[-20:],
            "state": self._training_state,
        }

    # ── Training entry point ──

    async def start_training(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        ws_callback: Optional[Callable] = None,
        resume_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self._is_training:
            return {"error": "Training already in progress"}

        self._is_training = True
        self._stop_requested = False
        self._is_paused = False

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._train_sync, train_loader, val_loader, ws_callback, resume_path
            )
            return result
        except Exception as e:
            log.error(f"Training failed: {e}", exc_info=True)
            return {"error": str(e)}
        finally:
            self._is_training = False

    # ── Model builder ──

    def _build_model(self) -> NeuralScribeNet:
        c = self._config
        return NeuralScribeNet(
            num_classes=c.get("model.num_classes", self._registry.num_classes),
            stem_channels=c.get("model.stem_channels", 32),
            block_channels=c.get("model.block_channels", [64, 128, 256, 320]),
            se_reduction=c.get("model.se_reduction", 16),
            dropout=c.get("model.dropout", 0.4),
            drop_path_rate=c.get("model.drop_path_rate", 0.1),
        ).to(self._device)

    # ── Core training loop ──

    def _train_sync(self, train_loader, val_loader, ws_callback, resume_path) -> Dict[str, Any]:
        cfg = self._config
        device = self._device
        total_epochs = cfg.get("training.epochs", 100)

        # Build model
        model = self._build_model()
        self._model = model
        log.info(f"Model: {model.count_parameters():,} params on {device}")

        # Loss
        criterion = CombinedLoss(
            label_smoothing=cfg.get("loss.label_smoothing", 0.08),
            focal_gamma=cfg.get("loss.focal_gamma", 1.5),
            focal_weight=cfg.get("loss.focal_weight", 0.3),
        ).to(device)

        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.get("optimizer.lr", 0.003),
            weight_decay=cfg.get("optimizer.weight_decay", 1e-4),
            betas=tuple(cfg.get("optimizer.betas", [0.9, 0.999])),
        )

        # Scheduler
        warmup_epochs = cfg.get("scheduler.warmup_epochs", 3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cfg.get("scheduler.T_0", 10),
            T_mult=cfg.get("scheduler.T_mult", 2),
            eta_min=cfg.get("scheduler.eta_min", 1e-5),
        )

        # Mixed precision
        use_amp = cfg.get("mixed_precision.enabled", True) and device.type == "cuda"
        scaler = GradScaler('cuda', enabled=use_amp)

        # Mixup
        mixup_alpha = cfg.get("regularization.mixup_alpha", 0.2)
        mixup_prob = cfg.get("regularization.mixup_probability", 0.5)

        # Resume
        start_epoch = 0
        if resume_path:
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            if "scaler_state_dict" in ckpt and use_amp:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            self._best_val_acc = ckpt.get("best_val_acc", 0.0)
            self._history = ckpt.get("history", [])
            log.info(f"Resumed from epoch {start_epoch}, best={self._best_val_acc:.4f}")

        # WS throttle
        last_ws = [0.0]
        throttle = cfg.get("logging.ws_throttle_seconds", 1.0)

        def emit(data):
            self._training_state = data
            now = time.time()
            if now - last_ws[0] >= throttle:
                last_ws[0] = now
                _emit_ws(ws_callback, data)

        def emit_force(data):
            self._training_state = data
            _emit_ws(ws_callback, data)

        # Checkpoint dir
        models_dir = PROJECT_ROOT / cfg.get("checkpointing.models_dir", "backend/models")
        models_dir.mkdir(parents=True, exist_ok=True)

        # ── Epoch loop ──
        training_start = time.time()

        for epoch in range(start_epoch, total_epochs):
            if self._stop_requested:
                break

            while self._is_paused:
                time.sleep(0.5)
                if self._stop_requested:
                    break
            if self._stop_requested:
                break

            self._current_epoch = epoch
            epoch_start = time.time()

            # Warmup LR
            if epoch < warmup_epochs:
                w0 = cfg.get("scheduler.warmup_start_lr", 1e-4)
                w1 = cfg.get("optimizer.lr", 0.003)
                lr = w0 + (w1 - w0) * (epoch / max(warmup_epochs, 1))
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

            # ── Train one epoch ──
            model.train()
            loss_sum = 0.0
            correct = 0
            total = 0
            n_batches = len(train_loader)

            for bi, (images, targets) in enumerate(train_loader):
                if self._stop_requested:
                    break

                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                do_mixup = mixup_alpha > 0 and np.random.random() < mixup_prob
                if do_mixup:
                    images, ya, yb, lam = mixup_data(images, targets, mixup_alpha)

                optimizer.zero_grad(set_to_none=True)

                with autocast('cuda', enabled=use_amp):
                    logits = model(images)
                    loss = mixup_criterion(criterion, logits, ya, yb, lam) if do_mixup else criterion(logits, targets)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                bs = images.size(0)
                loss_sum += loss.item() * bs
                _, pred = logits.max(1)
                if do_mixup:
                    correct += (lam * pred.eq(ya).float() + (1 - lam) * pred.eq(yb).float()).sum().item()
                else:
                    correct += pred.eq(targets).sum().item()
                total += bs

                log_every = cfg.get("logging.log_every_n_batches", 50)
                if bi % log_every == 0:
                    emit({
                        "type": "training_batch",
                        "epoch": epoch, "batch": bi, "total_batches": n_batches,
                        "loss": round(loss.item(), 4),
                        "lr": round(optimizer.param_groups[0]["lr"], 8),
                    })

            if self._stop_requested:
                break

            if epoch >= warmup_epochs:
                scheduler.step()

            train_loss = loss_sum / max(total, 1)
            train_acc = correct / max(total, 1)

            # ── Validate ──
            val_loss, val_acc, val_per_class = self._validate(model, val_loader, criterion, device, use_amp)

            epoch_time = time.time() - epoch_start
            elapsed = time.time() - training_start
            done = epoch - start_epoch + 1
            eta = (elapsed / done) * (total_epochs - epoch - 1)

            is_best = val_acc > self._best_val_acc
            if is_best:
                self._best_val_acc = val_acc

            rec = {
                "type": "training_epoch",
                "epoch": epoch, "total_epochs": total_epochs,
                "train_loss": round(train_loss, 4), "train_acc": round(train_acc, 4),
                "val_loss": round(val_loss, 4), "val_acc": round(val_acc, 4),
                "lr": round(optimizer.param_groups[0]["lr"], 8),
                "epoch_time": round(epoch_time, 1), "eta_seconds": round(eta),
                "best_val_acc": round(self._best_val_acc, 4), "is_best": is_best,
            }
            self._history.append(rec)
            emit_force(rec)

            log.info(
                f"Epoch {epoch}/{total_epochs} — "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.6f}"
                f"{' ★ BEST' if is_best else ''}"
            )

            # Checkpoint
            self._save_checkpoint(
                model, optimizer, scheduler, scaler,
                epoch, train_loss, val_loss, val_acc,
                is_best, models_dir, cfg, val_per_class,
            )

        # ── Done ──
        total_time = time.time() - training_start
        result = {
            "type": "training_complete",
            "status": "stopped" if self._stop_requested else "complete",
            "epochs_run": self._current_epoch - start_epoch + 1,
            "best_val_acc": round(self._best_val_acc, 4),
            "total_time_seconds": round(total_time, 1),
        }
        emit_force(result)
        return result

    # ── Validation ──

    def _validate(self, model, val_loader, criterion, device, use_amp):
        model.eval()
        loss_sum = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with autocast('cuda', enabled=use_amp):
                    logits = model(images)
                    loss = F.cross_entropy(logits, targets)
                loss_sum += loss.item() * images.size(0)
                _, pred = logits.max(1)
                all_preds.append(pred.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        val_loss = loss_sum / max(len(all_targets), 1)
        val_acc = compute_overall_accuracy(all_targets, all_preds)

        per_class = {}
        if self._config.get("validation.per_class_metrics", True):
            per_class = compute_per_class_metrics(all_targets, all_preds, self._registry.num_classes)

        return val_loss, val_acc, per_class

    # ── Checkpointing ──

    def _save_checkpoint(
        self, model, optimizer, scheduler, scaler,
        epoch, train_loss, val_loss, val_acc,
        is_best, models_dir, cfg, val_per_class,
    ):
        save_interval = cfg.get("checkpointing.save_interval", 10)
        save_best = cfg.get("checkpointing.save_best", True)
        max_ckpts = cfg.get("checkpointing.max_checkpoints", 5)

        should_save_periodic = (epoch + 1) % save_interval == 0
        if not should_save_periodic and not (is_best and save_best):
            return

        ts = timestamp_str()

        ckpt_data = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best_val_acc": self._best_val_acc,
            "history": self._history,
            "config": cfg.to_dict(),
            "num_classes": self._registry.num_classes,
        }

        # Save best
        if is_best and save_best:
            best_path = models_dir / "best_model.pth"
            torch.save(ckpt_data, best_path)
            log.info(f"Saved best model: val_acc={val_acc:.4f}")

            meta = {
                "name": f"best_v2_{ts}", "epoch": epoch,
                "train_loss": round(train_loss, 4), "val_loss": round(val_loss, 4),
                "val_acc": round(val_acc, 4), "n_params": model.count_parameters(),
                "num_classes": self._registry.num_classes, "timestamp": ts,
                "per_class_metrics": {str(k): v for k, v in val_per_class.items()},
            }
            with open(models_dir / "metadata_best_model.json", "w") as f:
                json.dump(meta, f, indent=2, default=str)

        # Save periodic
        if should_save_periodic:
            name = f"model_v2_{ts}_epoch_{epoch}"
            ckpt_path = models_dir / f"{name}.pth"
            torch.save(ckpt_data, ckpt_path)
            log.info(f"Saved checkpoint: {name}")

            meta = {
                "name": name, "epoch": epoch,
                "train_loss": round(train_loss, 4), "val_loss": round(val_loss, 4),
                "val_acc": round(val_acc, 4), "n_params": model.count_parameters(),
                "num_classes": self._registry.num_classes, "timestamp": ts,
                "file": f"{name}.pth",
                "size": human_readable_size(ckpt_path.stat().st_size),
            }
            with open(models_dir / f"metadata_{name}.json", "w") as f:
                json.dump(meta, f, indent=2, default=str)

            self._update_model_index(models_dir, meta)
            self._prune_checkpoints(models_dir, max_ckpts)

    def _update_model_index(self, models_dir: Path, meta: Dict):
        index_path = models_dir / "index.json"
        index = []
        if index_path.exists():
            try:
                with open(index_path) as f:
                    index = json.load(f)
            except Exception:
                index = []

        index.append({
            "name": meta["name"], "epoch": meta["epoch"],
            "val_acc": meta["val_acc"], "timestamp": meta["timestamp"],
            "file": meta.get("file", ""), "size": meta.get("size", ""),
        })

        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

    def _prune_checkpoints(self, models_dir: Path, max_keep: int):
        ckpts = sorted(models_dir.glob("model_v2_*.pth"), key=lambda p: p.stat().st_mtime)
        while len(ckpts) > max_keep:
            oldest = ckpts.pop(0)
            oldest.unlink(missing_ok=True)
            meta = models_dir / f"metadata_{oldest.stem}.json"
            meta.unlink(missing_ok=True)
            log.info(f"Pruned old checkpoint: {oldest.name}")

    # ── Evaluation (standalone) ──

    def evaluate(self, test_loader, model_path: Optional[str] = None) -> Dict[str, Any]:
        device = self._device

        if model_path:
            model = self._build_model()
            ckpt = torch.load(model_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
        elif self._model is not None:
            model = self._model
        else:
            return {"error": "No model loaded"}

        model.eval()
        all_preds, all_targets = [], []

        with torch.no_grad():
            for images, targets in test_loader:
                images = images.to(device, non_blocking=True)
                logits = model(images)
                _, pred = logits.max(1)
                all_preds.append(pred.cpu().numpy())
                all_targets.append(targets.numpy())

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        nc = self._registry.num_classes
        cm = compute_confusion_matrix(all_targets, all_preds, nc)
        display_map = self._registry.get_display_map()

        return {
            "overall_accuracy": round(compute_overall_accuracy(all_targets, all_preds), 4),
            "per_class_metrics": compute_per_class_metrics(all_targets, all_preds, nc),
            "confusion_matrix": cm.tolist(),
            "top_confusions": top_confusion_pairs(
                cm, k=self._config.get("validation.top_confusions", 10),
                id_to_display=display_map,
            ),
            "category_accuracy": compute_category_accuracy(
                all_targets, all_preds,
                {c["id"]: c["category"] for c in self._registry.classes},
            ),
            "total_samples": len(all_targets),
        }