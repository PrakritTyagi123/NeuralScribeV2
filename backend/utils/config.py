"""
Configuration manager for NeuralScribe v2.
Loads YAML configs and supports saving UI overrides back to disk.
"""

import os
import copy
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


def _get_project_root() -> Path:
    """Walk up from this file to find the project root (contains run_backend.py)."""
    current = Path(__file__).resolve().parent
    for _ in range(5):
        if (current / "run_backend.py").exists():
            return current
        current = current.parent
    return Path(__file__).resolve().parent.parent.parent


PROJECT_ROOT = _get_project_root()


class Config:
    """
    Hierarchical config backed by YAML files.

    - Load from disk on init
    - Access via dot-path: config.get("training.batch_size")
    - Override from UI: config.set("training.batch_size", 512)
    - Save overrides back: config.save()
    """

    def __init__(self, yaml_path: str):
        self._path = PROJECT_ROOT / yaml_path
        self._data: Dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        if self._path.exists():
            with open(self._path, "r", encoding="utf-8") as f:
                self._data = yaml.safe_load(f) or {}
        else:
            self._data = {}

    def save(self) -> None:
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


class ClassRegistry:
    """Loads and exposes the class registry from YAML."""

    def __init__(self, yaml_path: str = "configs/class_registry.yaml"):
        self._path = PROJECT_ROOT / yaml_path
        self._data: Dict[str, Any] = {}
        self._classes: list = []
        self._id_to_info: Dict[int, Dict] = {}
        self._label_to_id: Dict[str, int] = {}
        self.load()

    def load(self) -> None:
        with open(self._path, "r", encoding="utf-8") as f:
            self._data = yaml.safe_load(f)
        self._classes = self._data.get("classes", [])
        self._id_to_info = {c["id"]: c for c in self._classes}
        self._label_to_id = {c["label"]: c["id"] for c in self._classes}

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
            "num_classes": self.num_classes,
            "classes": self._classes,
            "category_order": self.category_order,
        }