"""
NeuralScribe v2 — Phase 3 Verification Script
Validates the English pipeline end-to-end without downloading EMNIST.
Checks: directory structure, configs, registry, model creation, forward pass.

Usage:
    python verify_english.py
"""

import sys
import os
import json

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  ✓ {name}")
        passed += 1
    else:
        print(f"  ✗ {name} — {detail}")
        failed += 1


def main():
    global passed, failed

    print("=" * 60)
    print("NeuralScribe v2 — English Pipeline Verification")
    print("=" * 60)

    # ── 1. Check directory structure ──
    print("\n── Directory Structure ──")
    from pathlib import Path

    try:
        from backend.utils.config import PROJECT_ROOT, get_language_paths, SUPPORTED_LANGUAGES, ensure_all_language_dirs
        check("config.py imports", True)
    except Exception as e:
        check("config.py imports", False, str(e))
        print("\nCannot continue — config.py import failed.")
        return

    ensure_all_language_dirs()
    check("ensure_all_language_dirs()", True)

    paths = get_language_paths("english")
    check("English paths created", True)

    for name, path in [
        ("configs/languages/english/", paths.class_registry.parent),
        ("datasets/english/cache/", paths.dataset_cache.parent),
        ("datasets/english/raw/", paths.dataset_raw),
        ("models/english/", paths.models_dir),
        ("models/english/exports/", paths.exports_dir),
        ("training_logs/english/", paths.training_logs_dir),
    ]:
        check(f"Dir exists: {name}", path.exists(), f"Missing: {path}")

    # ── 2. Check class registry ──
    print("\n── English Class Registry ──")
    try:
        from backend.utils.config import ClassRegistry
        registry = ClassRegistry(language="english")
        check("Registry loads", True)
        check(f"Num classes = 36", registry.num_classes == 36, f"Got {registry.num_classes}")
        check(f"Language = english", registry.language == "english", f"Got {registry.language}")

        # Check digit labels
        for i in range(10):
            label = registry.id_to_label(i)
            check(f"Class {i} = '{label}'", label == str(i), f"Expected '{i}', got '{label}'")

        # Check uppercase labels
        for i in range(26):
            expected = chr(65 + i)
            label = registry.id_to_label(10 + i)
            check(f"Class {10+i} = '{label}'", label == expected, f"Expected '{expected}', got '{label}'")

        # Check no Greek/math classes
        check("No class id=62 (was Gamma)", registry.id_to_label(62).startswith("unk"), f"Got '{registry.id_to_label(62)}'")
        check("No class id=88 (was plus)", registry.id_to_label(88).startswith("unk"), f"Got '{registry.id_to_label(88)}'")

        # Check category order
        check("Categories = [digit, uppercase]", registry.category_order == ["digit", "uppercase"],
              f"Got {registry.category_order}")

    except Exception as e:
        check("Registry loads", False, str(e))

    # ── 3. Check prep config ──
    print("\n── English Prep Config ──")
    try:
        from backend.utils.config import Config
        prep_path = paths.prep_config
        check(f"Prep config exists", prep_path.exists(), f"Missing: {prep_path}")

        if prep_path.exists():
            prep = Config(str(prep_path.relative_to(PROJECT_ROOT)))
            check("Synthetic disabled", prep.get("synthetic.enabled") == False, f"Got {prep.get('synthetic.enabled')}")
            check("Cache path correct", "datasets/english/" in prep.get("cache.path", ""), f"Got {prep.get('cache.path')}")
            check("Stratified split", prep.get("split.stratified") == True)
    except Exception as e:
        check("Prep config", False, str(e))

    # ── 4. Check train config ──
    print("\n── English Train Config ──")
    try:
        train_path = paths.train_config
        check(f"Train config exists", train_path.exists(), f"Missing: {train_path}")

        if train_path.exists():
            train_cfg = Config(str(train_path.relative_to(PROJECT_ROOT)))
            num_classes = train_cfg.get("model.num_classes")
            check(f"num_classes = 36", num_classes == 36, f"Got {num_classes}")
            check("Models dir = models/english", "models/english" in train_cfg.get("checkpointing.models_dir", ""),
                  f"Got {train_cfg.get('checkpointing.models_dir')}")
    except Exception as e:
        check("Train config", False, str(e))

    # ── 5. Check model creation ──
    print("\n── Model Creation (36 classes) ──")
    try:
        import torch
        from backend.ml.model import NeuralScribeNet

        model = NeuralScribeNet(num_classes=36)
        check("Model created (36 classes)", True)
        check(f"Classifier output = 36", model.classifier.out_features == 36,
              f"Got {model.classifier.out_features}")

        n_params = model.count_parameters()
        check(f"Param count reasonable ({n_params:,})", 100_000 < n_params < 10_000_000)

        # Forward pass
        dummy = torch.randn(2, 1, 28, 28)
        with torch.no_grad():
            output = model(dummy)
        check(f"Forward pass shape = (2, 36)", output.shape == (2, 36), f"Got {output.shape}")

        # Forward with intermediates
        logits, activations = model.forward_with_intermediates(dummy)
        check(f"Intermediates captured", len(activations) > 0, f"Got {len(activations)} layers")
        check(f"Has stem activation", "stem" in activations)
        check(f"Has block_3 activation", "block_3" in activations)

    except Exception as e:
        check("Model creation", False, str(e))

    # ── 6. Check EMNIST label mapping ──
    print("\n── EMNIST Label Mapping ──")
    try:
        from backend.services.dataset_service import _build_emnist_label_map

        label_map = _build_emnist_label_map(registry)
        check(f"Label map has 36 entries", len(label_map) == 36, f"Got {len(label_map)}")

        # Digits
        for i in range(10):
            check(f"EMNIST {i} → our {i}", label_map.get(i) == i, f"Got {label_map.get(i)}")

        # Uppercase
        for i in range(26):
            check(f"EMNIST {10+i} → our {10+i}", label_map.get(10 + i) == 10 + i,
                  f"Got {label_map.get(10+i)}")

        # Lowercase should NOT be mapped
        for i in range(36, 47):
            check(f"EMNIST {i} not mapped", i not in label_map, f"Found mapping: {label_map.get(i)}")

    except Exception as e:
        check("Label mapping", False, str(e))

    # ── 7. Check project config ──
    print("\n── Project Config ──")
    try:
        from backend.utils.config import ProjectConfig
        pc = ProjectConfig()
        check("ProjectConfig loads", True)
        check(f"Default language = english", pc.selected_language == "english",
              f"Got {pc.selected_language}")

        langs = pc.get_available_languages()
        check(f"5 languages listed", len(langs) == 5, f"Got {len(langs)}")

        eng = next((l for l in langs if l["id"] == "english"), None)
        check("English is enabled", eng and eng["enabled"] == True)
        check("English status = ready", eng and eng["status"] == "ready")

        dev = next((l for l in langs if l["id"] == "devanagari"), None)
        check("Devanagari is placeholder", dev and dev["status"] == "placeholder")

    except Exception as e:
        check("Project config", False, str(e))

    # ── 8. Check placeholder languages ──
    print("\n── Placeholder Languages ──")
    for lang in ["devanagari", "arabic", "cyrillic", "chinese"]:
        try:
            reg = ClassRegistry(language=lang)
            check(f"{lang}: registry loads ({reg.num_classes} classes)", reg.num_classes > 0)
            check(f"{lang}: status = placeholder", reg.status == "placeholder", f"Got {reg.status}")
        except Exception as e:
            check(f"{lang}: registry", False, str(e))

    # ── Summary ──
    print()
    print("=" * 60)
    total = passed + failed
    print(f"  Results: {passed}/{total} passed, {failed} failed")
    if failed == 0:
        print("  ✓ All checks passed — English pipeline is ready!")
        print()
        print("  Next steps:")
        print("    1. python backend/scripts/prepare_dataset.py --language english")
        print("    2. python backend/scripts/train.py --language english")
        print("    3. python backend/scripts/evaluate.py --language english")
    else:
        print(f"  ✗ {failed} checks failed — review errors above")
    print("=" * 60)


if __name__ == "__main__":
    main()
