"""
NeuralScribe v2 — CLI dataset preparation script.
Standalone alternative to the Data Preparation UI tab.

Usage:
    python backend/scripts/prepare_dataset.py --config configs/prep_v2.yaml
"""

import sys
import os
import argparse
import time
import asyncio

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.utils.logging import setup_logging, get_logger
from backend.utils.helpers import eta_string, human_readable_size
from backend.services.dataset_service import DatasetService

setup_logging()
log = get_logger("prepare_dataset")


def print_progress(data: dict):
    """Pretty-print progress updates."""
    stage = data.get("stage", "")
    message = data.get("message", "")
    processed = data.get("processed", 0)
    total = data.get("total", 0)
    rate = data.get("samples_per_sec", 0)
    eta = data.get("eta_seconds", 0)

    if total > 0:
        pct = processed / total * 100
        bar_len = 40
        filled = int(bar_len * processed / total)
        bar = "█" * filled + "░" * (bar_len - filled)
        line = f"\r  [{bar}] {pct:5.1f}% | {processed:,}/{total:,}"
        if rate > 0:
            line += f" | {rate:,.0f}/s"
        if eta > 0:
            line += f" | ETA {eta_string(eta)}"
        print(line, end="", flush=True)
    elif message:
        print(f"  {message}")

    if stage == "complete":
        print()  # newline after progress bar


async def run_preparation(config_path: str):
    """Run the dataset preparation pipeline."""
    log.info("=" * 60)
    log.info("NeuralScribe v2 — Dataset Preparation")
    log.info("=" * 60)

    service = DatasetService()

    # Show config
    status = service.get_status()
    log.info(f"Num classes: {status['num_classes']}")
    log.info(f"Cache path: {status['cache_path']}")

    if status["cache_exists"]:
        log.info(f"Existing cache found: {status['cache_size']}")
        response = input("  Cache exists. Rebuild? [y/N]: ").strip().lower()
        if response != "y":
            log.info("Skipping preparation — using existing cache.")
            return

    start = time.time()

    async def progress_callback(data: dict):
        print_progress(data)

    log.info("Starting dataset preparation...")
    print()

    result = await service.prepare_dataset(ws_callback=progress_callback)

    elapsed = time.time() - start
    print()
    log.info("=" * 60)

    if result.get("status") == "cancelled":
        log.info("Preparation cancelled.")
        return

    if "error" in result:
        log.error(f"Preparation failed: {result['error']}")
        return

    log.info("Dataset preparation complete!")
    log.info(f"  Total samples:  {result.get('total_samples', 0):,}")
    log.info(f"  Train samples:  {result.get('train_samples', 0):,}")
    log.info(f"  Val samples:    {result.get('val_samples', 0):,}")
    log.info(f"  Test samples:   {result.get('test_samples', 0):,}")
    log.info(f"  Cache size:     {result.get('cache_size', 'N/A')}")
    log.info(f"  Num classes:    {result.get('num_classes', 0)}")
    log.info(f"  Total time:     {elapsed:.1f}s")
    log.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="NeuralScribe v2 — Prepare Dataset")
    parser.add_argument(
        "--config", type=str, default="configs/prep_v2.yaml",
        help="Path to preparation config YAML",
    )
    args = parser.parse_args()

    asyncio.run(run_preparation(args.config))


if __name__ == "__main__":
    main()