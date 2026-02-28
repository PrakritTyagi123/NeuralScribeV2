"""
GPU monitoring for NeuralScribe v2.
Primary: pynvml | Fallback: nvidia-smi subprocess | Graceful: CPU-only mode
"""

import subprocess
import json
from typing import Dict, Any, Optional
from .logging import get_logger

log = get_logger(__name__)

# Try pynvml
_pynvml_available = False
try:
    import pynvml
    pynvml.nvmlInit()
    _pynvml_available = True
    log.info("GPU monitoring: pynvml initialized")
except Exception:
    log.info("pynvml not available, trying nvidia-smi fallback")


def _query_pynvml() -> Optional[Dict[str, Any]]:
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        return {
            "available": True,
            "name": name,
            "gpu_util_percent": util.gpu,
            "memory_used_mb": round(info.used / 1024 / 1024),
            "memory_total_mb": round(info.total / 1024 / 1024),
            "memory_percent": round(info.used / info.total * 100, 1),
            "temperature_c": temp,
        }
    except Exception as e:
        log.warning(f"pynvml query failed: {e}")
        return None


def _query_nvidia_smi() -> Optional[Dict[str, Any]]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None
        parts = [p.strip() for p in result.stdout.strip().split(",")]
        if len(parts) < 5:
            return None
        mem_used = float(parts[2])
        mem_total = float(parts[3])
        return {
            "available": True,
            "name": parts[0],
            "gpu_util_percent": int(parts[1]),
            "memory_used_mb": int(mem_used),
            "memory_total_mb": int(mem_total),
            "memory_percent": round(mem_used / mem_total * 100, 1) if mem_total > 0 else 0,
            "temperature_c": int(parts[4]),
        }
    except Exception as e:
        log.debug(f"nvidia-smi fallback failed: {e}")
        return None


def get_gpu_info() -> Dict[str, Any]:
    """Return GPU stats. Falls back gracefully."""
    if _pynvml_available:
        result = _query_pynvml()
        if result:
            return result

    result = _query_nvidia_smi()
    if result:
        return result

    return {
        "available": False,
        "name": "No GPU detected",
        "gpu_util_percent": 0,
        "memory_used_mb": 0,
        "memory_total_mb": 0,
        "memory_percent": 0,
        "temperature_c": 0,
    }


def get_cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_device_string() -> str:
    if get_cuda_available():
        return "cuda"
    return "cpu"