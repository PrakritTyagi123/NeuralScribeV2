"""
NeuralScribe v2 — Configuration (English only).
"""
import os, copy, json, yaml
from pathlib import Path
from typing import Any, Dict, List, Optional


def _get_project_root() -> Path:
    current = Path(__file__).resolve().parent
    for _ in range(5):
        if (current / "run_backend.py").exists():
            return current
        current = current.parent
    return Path(__file__).resolve().parent.parent.parent


PROJECT_ROOT = _get_project_root()
SUPPORTED_LANGUAGES = ["english"]
DEFAULT_LANGUAGE = "english"

# ── Paths ──

class LanguagePaths:
    def __init__(self, language: str = "english"):
        self.language = language
    @property
    def class_registry(self): return PROJECT_ROOT / "configs/languages" / self.language / "class_registry.yaml"
    @property
    def prep_config(self): return PROJECT_ROOT / "configs/languages" / self.language / "prep.yaml"
    @property
    def train_config(self): return PROJECT_ROOT / "configs/languages" / self.language / "train.yaml"
    @property
    def dataset_dir(self): return PROJECT_ROOT / "datasets" / self.language
    @property
    def dataset_cache(self): return self.dataset_dir / "cache" / "cached_dataset.pt"
    @property
    def dataset_raw(self): return self.dataset_dir / "raw"
    @property
    def models_dir(self): return PROJECT_ROOT / "models" / self.language
    @property
    def exports_dir(self): return self.models_dir / "exports"
    @property
    def training_logs_dir(self): return PROJECT_ROOT / "training_logs" / self.language
    @property
    def training_history(self): return self.training_logs_dir / "training_history.json"
    @property
    def loss_graph(self): return self.training_logs_dir / "loss_graph.json"
    @property
    def accuracy_graph(self): return self.training_logs_dir / "accuracy_graph.json"
    @property
    def training_table(self): return self.training_logs_dir / "training_table.json"
    def ensure_dirs(self):
        for d in [self.dataset_dir / "cache", self.dataset_dir / "raw",
                  self.models_dir, self.exports_dir, self.training_logs_dir]:
            d.mkdir(parents=True, exist_ok=True)

def get_language_paths(language: str = "english"): return LanguagePaths(language)
def ensure_all_language_dirs():
    for lang in SUPPORTED_LANGUAGES: LanguagePaths(lang).ensure_dirs()

# ── Project Config ──

class ProjectConfig:
    def __init__(self):
        self._path = PROJECT_ROOT / "project_config.json"
        self._data = {}
        self.load()
    def load(self):
        if self._path.exists():
            try:
                with open(self._path, "r") as f: self._data = json.load(f)
            except: self._data = self._default(); self.save()
        else: self._data = self._default(); self.save()
    def save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f: json.dump(self._data, f, indent=4)
    def _default(self):
        return {"version": 2, "last_selected_language": "english", "last_selected_model": None,
                "languages": {"english": {"enabled": True, "status": "ready", "dataset_prepared": False, "last_model": None}},
                "ui_state": {"active_view": "intro"}}
    @property
    def selected_language(self): return self._data.get("last_selected_language", "english")
    @selected_language.setter
    def selected_language(self, v): self._data["last_selected_language"] = v; self.save()
    @property
    def selected_model(self): return self._data.get("last_selected_model")
    @selected_model.setter
    def selected_model(self, v): self._data["last_selected_model"] = v; self.save()
    def get_language_config(self, lang): return self._data.get("languages", {}).get(lang, {})
    def set_language_config(self, lang, key, val):
        self._data.setdefault("languages", {}).setdefault(lang, {})[key] = val; self.save()
    def set_ui_state(self, key, val): self._data.setdefault("ui_state", {})[key] = val; self.save()
    def to_dict(self): return copy.deepcopy(self._data)

# ── YAML Config ──

class Config:
    def __init__(self, yaml_path: str = None):
        self._data = {}
        self._path = (PROJECT_ROOT / yaml_path) if yaml_path else None
        if self._path: self.load()
    def load(self):
        if self._path and self._path.exists():
            with open(self._path, "r", encoding="utf-8") as f: self._data = yaml.safe_load(f) or {}
    def save(self):
        if not self._path: return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            yaml.dump(self._data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    def get(self, dot_path: str, default=None):
        keys = dot_path.split("."); node = self._data
        for k in keys:
            if isinstance(node, dict) and k in node: node = node[k]
            else: return default
        return node
    def set(self, dot_path: str, value, persist=True):
        keys = dot_path.split("."); node = self._data
        for k in keys[:-1]: node = node.setdefault(k, {})
        node[keys[-1]] = value
        if persist: self.save()
    def to_dict(self): return copy.deepcopy(self._data)
    def update(self, overrides, persist=True):
        for k, v in overrides.items(): self.set(k, v, persist=False)
        if persist: self.save()
    @staticmethod
    def resolve_path(rel): return PROJECT_ROOT / rel

# ── Class Registry ──

class ClassRegistry:
    def __init__(self, yaml_path: str = None, language: str = "english"):
        self._path = (PROJECT_ROOT / yaml_path) if yaml_path else get_language_paths(language).class_registry
        self._data, self._classes = {}, []
        self._id_map, self._label_map = {}, {}
        self.load()
    def load(self):
        if not self._path or not self._path.exists(): return
        with open(self._path, "r", encoding="utf-8") as f: self._data = yaml.safe_load(f) or {}
        self._classes = self._data.get("classes", [])
        self._id_map = {c["id"]: c for c in self._classes}
        self._label_map = {c["label"]: c["id"] for c in self._classes}
    @property
    def language(self): return self._data.get("language", "english")
    @property
    def num_classes(self): return len(self._classes)
    @property
    def classes(self): return self._classes
    @property
    def category_order(self): return self._data.get("category_order", [])
    @property
    def status(self): return self._data.get("status", "ready")
    def id_to_label(self, cid): return self._id_map.get(cid, {}).get("label", f"unk_{cid}")
    def id_to_display(self, cid): return self._id_map.get(cid, {}).get("display", "?")
    def label_to_id(self, label): return self._label_map.get(label)
    def id_to_category(self, cid): return self._id_map.get(cid, {}).get("category", "unknown")
    def get_display_map(self): return {c["id"]: c["display"] for c in self._classes}
    def to_dict(self): return {"language": self.language, "num_classes": self.num_classes,
                                "classes": self._classes, "category_order": self.category_order}
