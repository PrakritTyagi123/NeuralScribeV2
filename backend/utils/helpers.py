"""
General helper functions for NeuralScribe v2.
"""

import time
import json
import hashlib
from pathlib import Path
from typing import Any, Dict
from datetime import datetime


def get_project_root() -> Path:
    from .config import PROJECT_ROOT
    return PROJECT_ROOT


def timestamp_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def human_readable_size(size_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def compute_file_hash(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def eta_string(seconds: float) -> str:
    if seconds < 0:
        return "--:--:--"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


class Timer:
    """Simple context-manager timer."""

    def __init__(self):
        self.start_time = None
        self.elapsed = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start_time

    @property
    def elapsed_ms(self) -> float:
        return self.elapsed * 1000


def safe_json_dump(obj: Any) -> str:
    """JSON serialize with numpy/tensor handling."""
    import numpy as np

    class _Encoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, Path):
                return str(o)
            if isinstance(o, datetime):
                return o.isoformat()
            return super().default(o)

    return json.dumps(obj, cls=_Encoder)