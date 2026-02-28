"""
NeuralScribe v2 — CLI evaluation script.
Runs full evaluation on the test set and prints detailed metrics.

Usage:
    python backend/scripts/evaluate.py
    python backend/scripts/evaluate.py --model backend/models/best_model.pth
"""

import sys
import os
import argparse
import json

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.utils.logging import setup_logging, get_logger
from backend.utils.config import ClassRegistry
from backend.services.dataset_service import DatasetService
from backend.services.training_service import TrainingService

setup_logging()
log = get_logger("evaluate")


def print_category_accuracy(cat_acc: dict):
    """Pretty-print category accuracy table."""
    print()
    print(f"  {'Category':<20} {'Accuracy':>10} {'Correct':>10} {'Total':>10}")
    print(f"  {'─' * 20} {'─' * 10} {'─' * 10} {'─' * 10}")
    for cat, data in sorted(cat_acc.items()):
        acc = data["accuracy"] * 100
        print(f"  {cat:<20} {acc:>9.2f}% {data['correct']:>10,} {data['total']:>10,}")


def print_top_confusions(confusions: list, n: int = 10):
    """Pretty-print top confusion pairs."""
    print()
    print(f"  {'True':<10} {'Predicted':<10} {'Count':>8}")
    print(f"  {'─' * 10} {'─' * 10} {'─' * 8}")
    for c in confusions[:n]:
        print(f"  {c['true_label']:<10} {c['pred_label']:<10} {c['count']:>8,}")


def print_worst_classes(per_class: dict, registry: ClassRegistry, n: int = 10):
    """Print the N worst-performing classes by F1."""
    items = []
    for class_id_str, metrics in per_class.items():
        class_id = int(class_id_str)
        if metrics["support"] > 0:
            items.append({
                "class_id": class_id,
                "display": registry.id_to_display(class_id),
                "label": registry.id_to_label(class_id),
                "f1": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "support": metrics["support"],
            })

    items.sort(key=lambda x: x["f1"])

    print()
    print(f"  {'Class':<12} {'Display':<8} {'F1':>8} {'Prec':>8} {'Recall':>8} {'Support':>8}")
    print(f"  {'─' * 12} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8} {'─' * 8}")
    for item in items[:n]:
        print(
            f"  {item['label']:<12} {item['display']:<8} "
            f"{item['f1']:>8.4f} {item['precision']:>8.4f} "
            f"{item['recall']:>8.4f} {item['support']:>8,}"
        )


def main():
    parser = argparse.ArgumentParser(description="NeuralScribe v2 — Evaluate Model")
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to model .pth file (default: uses currently loaded or best_model)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--save-report", type=str, default=None,
        help="Save evaluation report as JSON to this path",
    )
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("NeuralScribe v2 — Evaluation")
    log.info("=" * 60)

    # Load dataset
    ds = DatasetService()
    status = ds.get_status()

    if not status["cache_exists"]:
        log.error("Dataset not prepared! Run prepare_dataset.py first.")
        return

    registry = ds.registry

    # Create test loader
    test_loader = ds.get_dataloader(
        "test", batch_size=args.batch_size,
        num_workers=2, pin_memory=True, shuffle=False,
    )
    if test_loader is None:
        log.error("Failed to create test dataloader.")
        return

    log.info(f"Test batches: {len(test_loader)}")

    # Evaluate
    ts = TrainingService()
    model_path = args.model or str(
        ds._prep_config.resolve_path("backend/models/best_model.pth")
    )

    log.info(f"Evaluating model: {model_path}")
    print()

    result = ts.evaluate(test_loader, model_path=model_path)

    if "error" in result:
        log.error(f"Evaluation failed: {result['error']}")
        return

    # ── Print results ──
    print("=" * 60)
    print(f"  Overall Accuracy: {result['overall_accuracy'] * 100:.2f}%")
    print(f"  Total Samples:    {result['total_samples']:,}")

    # Category accuracy
    print("\n  ── Accuracy by Category ──")
    print_category_accuracy(result["category_accuracy"])

    # Top confusions
    print("\n  ── Top Confusion Pairs ──")
    print_top_confusions(result["top_confusions"])

    # Worst classes
    print("\n  ── Worst Performing Classes (by F1) ──")
    print_worst_classes(result["per_class_metrics"], registry)

    print()
    print("=" * 60)

    # Save report
    if args.save_report:
        # Convert numpy types for JSON serialization
        report = {
            "overall_accuracy": result["overall_accuracy"],
            "total_samples": result["total_samples"],
            "category_accuracy": result["category_accuracy"],
            "top_confusions": result["top_confusions"],
            "per_class_metrics": {str(k): v for k, v in result["per_class_metrics"].items()},
        }
        with open(args.save_report, "w") as f:
            json.dump(report, f, indent=2, default=str)
        log.info(f"Report saved to: {args.save_report}")


if __name__ == "__main__":
    main()