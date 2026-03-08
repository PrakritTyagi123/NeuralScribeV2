"""
Configuration manager for NeuralScribe v2.
Loads YAML configs, supports per-language path resolution,
and manages persistent project state (project_config.json).
"""

import os
import copy
import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional


def _get_project_root() -> Path:
    """Walk up from this file to find the project root (contains run_backend.py)."""
    current = Path(__file__).resolve().parent
    for _ in range(5):
        if (current / "run_backend.py").exists():
            return current
        current = current.parent
    return Path(__file__).resolve().parent.parent.parent


PROJECT_ROOT = _get_project_root()

# ── Supported languages ──
SUPPORTED_LANGUAGES = ["english", "devanagari", "arabic", "cyrillic", "chinese"]
DEFAULT_LANGUAGE = "english"


# ═══════════════════════════════════════
# LANGUAGE PATH RESOLVER
# ═══════════════════════════════════════

class LanguagePaths:
    """Resolves all paths for a given language."""

    def __init__(self, language: str):
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported language: {language}. Must be one of {SUPPORTED_LANGUAGES}")
        self.language = language

    @property
    def class_registry(self) -> Path:
        return PROJECT_ROOT / "configs" / "languages" / self.language / "class_registry.yaml"

    @property
    def prep_config(self) -> Path:
        return PROJECT_ROOT / "configs" / "languages" / self.language / "prep.yaml"

    @property
    def train_config(self) -> Path:
        return PROJECT_ROOT / "configs" / "languages" / self.language / "train.yaml"

    @property
    def dataset_dir(self) -> Path:
        return PROJECT_ROOT / "datasets" / self.language

    @property
    def dataset_cache(self) -> Path:
        return self.dataset_dir / "cache" / "cached_dataset.pt"

    @property
    def dataset_raw(self) -> Path:
        return self.dataset_dir / "raw"

    @property
    def models_dir(self) -> Path:
        return PROJECT_ROOT / "models" / self.language

    @property
    def exports_dir(self) -> Path:
        return self.models_dir / "exports"

    @property
    def training_logs_dir(self) -> Path:
        return PROJECT_ROOT / "training_logs" / self.language

    @property
    def training_history(self) -> Path:
        return self.training_logs_dir / "training_history.json"

    @property
    def loss_graph(self) -> Path:
        return self.training_logs_dir / "loss_graph.json"

    @property
    def accuracy_graph(self) -> Path:
        return self.training_logs_dir / "accuracy_graph.json"

    @property
    def training_table(self) -> Path:
        return self.training_logs_dir / "training_table.json"

    def ensure_dirs(self):
        """Create all directories for this language if they don't exist."""
        for d in [
            self.dataset_dir / "cache",
            self.dataset_dir / "raw",
            self.models_dir,
            self.exports_dir,
            self.training_logs_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, str]:
        return {
            "language": self.language,
            "class_registry": str(self.class_registry),
            "prep_config": str(self.prep_config),
            "train_config": str(self.train_config),
            "dataset_dir": str(self.dataset_dir),
            "dataset_cache": str(self.dataset_cache),
            "models_dir": str(self.models_dir),
            "training_logs_dir": str(self.training_logs_dir),
        }


def get_language_paths(language: str) -> LanguagePaths:
    """Get a LanguagePaths instance for the given language."""
    return LanguagePaths(language)


def ensure_all_language_dirs():
    """Create directory structure for all supported languages."""
    for lang in SUPPORTED_LANGUAGES:
        LanguagePaths(lang).ensure_dirs()


# ═══════════════════════════════════════
# PROJECT CONFIG (persistent state)
# ═══════════════════════════════════════

class ProjectConfig:
    """
    Manages project_config.json — persistent state across sessions.
    Auto-saves on every mutation.
    """

    def __init__(self, path: str = "project_config.json"):
        self._path = PROJECT_ROOT / path
        self._data: Dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._data = self._default()
                self.save()
        else:
            self._data = self._default()
            self.save()

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=4, ensure_ascii=False)

    def _default(self) -> Dict[str, Any]:
        return {
            "version": 2,
            "last_selected_language": DEFAULT_LANGUAGE,
            "last_selected_model": None,
            "languages": {
                lang: {
                    "enabled": lang == "english",
                    "status": "ready" if lang == "english" else "placeholder",
                    "dataset_prepared": False,
                    "last_model": None,
                }
                for lang in SUPPORTED_LANGUAGES
            },
            "ui_state": {
                "active_view": "dashboard",
                "prep_language": DEFAULT_LANGUAGE,
                "training_language": DEFAULT_LANGUAGE,
                "inference_language": DEFAULT_LANGUAGE,
            },
        }

    # ── Accessors ──

    @property
    def selected_language(self) -> str:
        return self._data.get("last_selected_language", DEFAULT_LANGUAGE)

    @selected_language.setter
    def selected_language(self, language: str):
        self._data["last_selected_language"] = language
        self.save()

    @property
    def selected_model(self) -> Optional[str]:
        return self._data.get("last_selected_model")

    @selected_model.setter
    def selected_model(self, model: Optional[str]):
        self._data["last_selected_model"] = model
        self.save()

    def get_language_config(self, language: str) -> Dict[str, Any]:
        return self._data.get("languages", {}).get(language, {})

    def set_language_config(self, language: str, key: str, value: Any):
        if "languages" not in self._data:
            self._data["languages"] = {}
        if language not in self._data["languages"]:
            self._data["languages"][language] = {}
        self._data["languages"][language][key] = value
        self.save()

    def get_ui_state(self) -> Dict[str, Any]:
        return self._data.get("ui_state", {})

    def set_ui_state(self, key: str, value: Any):
        if "ui_state" not in self._data:
            self._data["ui_state"] = {}
        self._data["ui_state"][key] = value
        self.save()

    def get_available_languages(self) -> List[Dict[str, Any]]:
        """Return list of languages with their status for the frontend."""
        result = []
        for lang in SUPPORTED_LANGUAGES:
            lang_cfg = self.get_language_config(lang)
            result.append({
                "id": lang,
                "name": lang.capitalize(),
                "enabled": lang_cfg.get("enabled", False),
                "status": lang_cfg.get("status", "placeholder"),
                "dataset_prepared": lang_cfg.get("dataset_prepared", False),
                "last_model": lang_cfg.get("last_model"),
            })
        return result

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self._data)


# ═══════════════════════════════════════
# YAML CONFIG (per-language)
# ═══════════════════════════════════════

class Config:
    """
    Hierarchical config backed by YAML files.

    - Load from disk on init
    - Access via dot-path: config.get("training.batch_size")
    - Override from UI: config.set("training.batch_size", 512)
    - Save overrides back: config.save()
    """

    def __init__(self, yaml_path: str = None):
        self._data: Dict[str, Any] = {}
        if yaml_path:
            self._path = PROJECT_ROOT / yaml_path
            self.load()
        else:
            self._path = None

    def load(self) -> None:
        if self._path is None or not self._path.exists():
            self._data = {}
            return
        with open(self._path, "r", encoding="utf-8") as f:
            self._data = yaml.safe_load(f) or {}

    def save(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            yaml.dump(self._data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    def get(self, dot_path: str, default: Any = None) -> Any:
        keys = dot_path.split(".")
        node = self._data
        for key in keys:
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                return default
        return node

    def set(self, dot_path: str, value: Any, persist: bool = True) -> None:
        keys = dot_path.split(".")
        node = self._data
        for key in keys[:-1]:
            if key not in node or not isinstance(node[key], dict):
                node[key] = {}
            node = node[key]
        node[keys[-1]] = value
        if persist:
            self.save()

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self._data)

    def update(self, overrides: Dict[str, Any], persist: bool = True) -> None:
        """Merge a flat dict of dot-path overrides."""
        for key, value in overrides.items():
            self.set(key, value, persist=False)
        if persist:
            self.save()

    @staticmethod
    def resolve_path(relative: str) -> Path:
        return PROJECT_ROOT / relative


# ═══════════════════════════════════════
# CLASS REGISTRY (per-language)
# ═══════════════════════════════════════

class ClassRegistry:
    """Loads and exposes the class registry from YAML. Language-aware."""

    def __init__(self, yaml_path: str = None, language: str = None):
        if yaml_path:
            self._path = PROJECT_ROOT / yaml_path
        elif language:
            self._path = get_language_paths(language).class_registry
        else:
            # Default fallback to english
            self._path = get_language_paths(DEFAULT_LANGUAGE).class_registry
        self._data: Dict[str, Any] = {}
        self._classes: list = []
        self._id_to_info: Dict[int, Dict] = {}
        self._label_to_id: Dict[str, int] = {}
        self.load()

    def load(self) -> None:
        if not self._path.exists():
            self._data = {}
            self._classes = []
            self._id_to_info = {}
            self._label_to_id = {}
            return
        with open(self._path, "r", encoding="utf-8") as f:
            self._data = yaml.safe_load(f)
        self._classes = self._data.get("classes", [])
        self._id_to_info = {c["id"]: c for c in self._classes}
        self._label_to_id = {c["label"]: c["id"] for c in self._classes}

    @property
    def language(self) -> str:
        return self._data.get("language", "unknown")

    @property
    def status(self) -> str:
        return self._data.get("status", "ready")

    @property
    def num_classes(self) -> int:
        return len(self._classes)

    @property
    def classes(self) -> list:
        return self._classes

    @property
    def category_order(self) -> list:
        return self._data.get("category_order", [])

    def id_to_label(self, class_id: int) -> str:
        return self._id_to_info.get(class_id, {}).get("label", f"unk_{class_id}")

    def id_to_display(self, class_id: int) -> str:
        return self._id_to_info.get(class_id, {}).get("display", "?")

    def label_to_id(self, label: str) -> Optional[int]:
        return self._label_to_id.get(label)

    def id_to_category(self, class_id: int) -> str:
        return self._id_to_info.get(class_id, {}).get("category", "unknown")

    def get_classes_by_category(self, category: str) -> list:
        return [c for c in self._classes if c["category"] == category]

    def get_display_map(self) -> Dict[int, str]:
        return {c["id"]: c["display"] for c in self._classes}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "language": self.language,
            "num_classes": self.num_classes,
            "classes": self._classes,
            "category_order": self.category_order,
        }