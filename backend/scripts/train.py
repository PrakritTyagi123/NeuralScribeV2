"""
NeuralScribe v2 — CLI training script.
Standalone alternative to the Training UI tab.

Usage:
    python backend/scripts/train.py --config configs/train_v2.yaml
    python backend/scripts/train.py --resume backend/models/best_model.pth
"""

import sys
import os
import argparse
import time
import asyncio

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.utils.logging import setup_logging, get_logger
from backend.utils.helpers import eta_string
from backend.services.dataset_service import DatasetService
from backend.services.training_service import TrainingService

setup_logging()
log = get_logger("train")


def print_epoch(data: dict):
    """Pretty-print epoch results."""
    if data.get("type") == "training_epoch":
        epoch = data["epoch"]
        total = data["total_epochs"]
        t_loss = data["train_loss"]
        t_acc = data["train_acc"]
        v_loss = data["val_loss"]
        v_acc = data["val_acc"]
        lr = data["lr"]
        etime = data.get("epoch_time", 0)
        eta = data.get("eta_seconds", 0)
        best = data.get("best_val_acc", 0)
        is_best = data.get("is_best", False)

        star = " ★ BEST" if is_best else ""
        print(
            f"  Epoch {epoch:3d}/{total} | "
            f"train_loss={t_loss:.4f} train_acc={t_acc:.4f} | "
            f"val_loss={v_loss:.4f} val_acc={v_acc:.4f} | "
            f"lr={lr:.6f} | {etime:.1f}s | "
            f"ETA {eta_string(eta)} | best={best:.4f}{star}"
        )

    elif data.get("type") == "training_batch":
        epoch = data["epoch"]
        batch = data["batch"]
        total_b = data["total_batches"]
        loss = data["loss"]
        pct = batch / total_b * 100 if total_b > 0 else 0
        print(f"\r    batch {batch}/{total_b} ({pct:.0f}%) loss={loss:.4f}", end="", flush=True)
        if batch > 0 and batch % 200 == 0:
            print()  # newline periodically


async def run_training(config_path: str, resume_path: str = None):
    """Run the training pipeline."""
    log.info("=" * 60)
    log.info("NeuralScribe v2 — Training")
    log.info("=" * 60)

    # Load dataset
    ds = DatasetService()
    status = ds.get_status()

    if not status["cache_exists"]:
        log.error("Dataset not prepared! Run prepare_dataset.py first.")
        log.error(f"  Expected: {status['cache_path']}")
        return

    log.info(f"Dataset cache: {status['cache_size']}")

    # Create service
    ts = TrainingService()
    cfg = ts.get_config()

    batch_size = cfg.get("training", {}).get("batch_size", 256)
    num_workers = cfg.get("training", {}).get("num_workers", 4)
    pin_memory = cfg.get("training", {}).get("pin_memory", True)
    epochs = cfg.get("training", {}).get("epochs", 100)

    log.info(f"Config: batch_size={batch_size}, epochs={epochs}, workers={num_workers}")
    log.info(f"Device: {ts.get_status()['device']}")

    # Create dataloaders
    train_loader = ds.get_dataloader("train", batch_size=batch_size,
                                      num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
    val_loader = ds.get_dataloader("val", batch_size=batch_size,
                                    num_workers=num_workers, pin_memory=pin_memory, shuffle=False)

    if train_loader is None or val_loader is None:
        log.error("Failed to create dataloaders.")
        return

    log.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    if resume_path:
        log.info(f"Resuming from: {resume_path}")

    start = time.time()
    print()

    async def progress_callback(data: dict):
        print_epoch(data)

    result = await ts.start_training(
        train_loader, val_loader,
        ws_callback=progress_callback,
        resume_path=resume_path,
    )

    elapsed = time.time() - start
    print()
    log.info("=" * 60)

    if "error" in result:
        log.error(f"Training failed: {result['error']}")
        return

    log.info(f"Training {result.get('status', 'complete')}!")
    log.info(f"  Epochs run:     {result.get('epochs_run', 0)}")
    log.info(f"  Best val acc:   {result.get('best_val_acc', 0):.4f}")
    log.info(f"  Total time:     {elapsed:.1f}s")
    log.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="NeuralScribe v2 — Train Model")
    parser.add_argument(
        "--config", type=str, default="configs/train_v2.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint .pth file to resume from",
    )
    args = parser.parse_args()

    asyncio.run(run_training(args.config, args.resume))


if __name__ == "__main__":
    main()