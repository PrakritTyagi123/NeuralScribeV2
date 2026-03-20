"""
NeuralScribe v2 — EMNIST downloader.
Downloads via torchvision with patched progress reporting.
"""

import urllib.request, time
from pathlib import Path
from typing import Dict, Optional, Callable

from ..utils.config import PROJECT_ROOT, get_language_paths
from ..utils.logging import get_logger

log = get_logger(__name__)

EMNIST_DIR = "datasets/english/raw/emnist"


def get_download_status() -> Dict:
    """Check if EMNIST is downloaded."""
    emnist_path = PROJECT_ROOT / EMNIST_DIR
    if not emnist_path.exists():
        return {"downloaded": False, "message": "Not downloaded"}

    # torchvision may create different structures depending on version:
    # root/EMNIST/processed/*.pt  OR  root/EMNIST/raw/*.gz
    # Search recursively for either
    for pt in emnist_path.rglob("*.pt"):
        return {"downloaded": True, "message": "EMNIST ready"}
    for gz in emnist_path.rglob("*.gz"):
        return {"downloaded": True, "message": "EMNIST ready"}
    # Check if any EMNIST directory exists with content
    for d in emnist_path.rglob("EMNIST"):
        if d.is_dir():
            return {"downloaded": True, "message": "EMNIST ready"}
    # Fallback — check if directory has any substantial content
    files = list(emnist_path.rglob("*"))
    if len(files) > 5:
        return {"downloaded": True, "message": "EMNIST data found"}
    return {"downloaded": False, "message": "Not downloaded"}


def download_emnist(emit: Optional[Callable] = None) -> Dict:
    """Download EMNIST balanced via torchvision with progress."""
    status = get_download_status()
    if status["downloaded"]:
        return {"status": "exists", "message": "Already downloaded"}

    emnist_path = PROJECT_ROOT / EMNIST_DIR
    emnist_path.mkdir(parents=True, exist_ok=True)

    if emit:
        emit({"stage": "download", "pct": 0, "message": "Starting EMNIST download (~550 MB)..."})

    import torchvision

    # Patch urlretrieve to capture download progress
    original = urllib.request.urlretrieve
    last_emit = [0.0]

    def patched(url, filename=None, reporthook=None, data=None):
        def hook(count, block_size, total_size):
            now = time.time()
            if emit and now - last_emit[0] > 0.5:
                last_emit[0] = now
                dl = count * block_size
                if total_size > 0:
                    pct = min(95, dl / total_size * 95)
                    emit({"stage": "download", "pct": round(pct, 1),
                          "message": f"Downloading: {dl // 1048576}/{total_size // 1048576} MB ({pct:.0f}%)"})
                else:
                    emit({"stage": "download", "pct": 0,
                          "message": f"Downloading: {dl // 1048576} MB"})
            if reporthook:
                reporthook(count, block_size, total_size)
        return original(url, filename, hook, data)

    try:
        urllib.request.urlretrieve = patched
        for idx, is_train in enumerate([True, False]):
            name = "train" if is_train else "test"
            if emit:
                emit({"stage": "download", "pct": idx * 48,
                      "message": f"Downloading EMNIST {name} split..."})
            torchvision.datasets.EMNIST(root=str(emnist_path), split="balanced",
                                         train=is_train, download=True)
    except Exception as e:
        log.error(f"EMNIST download failed: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}
    finally:
        urllib.request.urlretrieve = original

    if emit:
        emit({"stage": "download", "pct": 100, "message": "EMNIST downloaded successfully"})

    log.info("EMNIST downloaded")
    return {"status": "success", "message": "EMNIST downloaded"}
