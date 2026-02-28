"""
System service for NeuralScribe v2.
Provides GPU monitoring, system resource stats, and dashboard overview data.
"""

import psutil
import platform
from typing import Dict, Any
from pathlib import Path

from ..utils.config import Config, ClassRegistry, PROJECT_ROOT
from ..utils.logging import get_logger
from ..utils.gpu_monitor import get_gpu_info, get_cuda_available, get_device_string
from ..utils.helpers import human_readable_size

log = get_logger(__name__)


class SystemService:
    """System monitoring and dashboard data provider."""

    def __init__(self):
        self._registry = ClassRegistry()

    def get_gpu_status(self) -> Dict[str, Any]:
        """Real-time GPU stats."""
        info = get_gpu_info()
        info["cuda_available"] = get_cuda_available()
        info["device"] = get_device_string()
        return info

    def get_system_stats(self) -> Dict[str, Any]:
        """CPU, RAM, disk stats."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage(str(PROJECT_ROOT))

        return {
            "cpu_percent": cpu_percent,
            "cpu_count": psutil.cpu_count(),
            "ram_used_gb": round(mem.used / 1024**3, 1),
            "ram_total_gb": round(mem.total / 1024**3, 1),
            "ram_percent": mem.percent,
            "disk_used_gb": round(disk.used / 1024**3, 1),
            "disk_total_gb": round(disk.total / 1024**3, 1),
            "disk_percent": round(disk.percent, 1),
            "platform": platform.system(),
            "python_version": platform.python_version(),
        }

    def get_dashboard_overview(
        self,
        dataset_service=None,
        training_service=None,
        model_service=None,
    ) -> Dict[str, Any]:
        """
        Aggregate overview for the dashboard view.
        Combines GPU, dataset, model, and training status.
        """
        gpu = self.get_gpu_status()
        system = self.get_system_stats()

        # Dataset status
        dataset_status = {}
        if dataset_service:
            dataset_status = dataset_service.get_status()

        # Training status
        training_status = {}
        if training_service:
            training_status = training_service.get_status()

        # Model status
        model_list = []
        best_model = None
        if model_service:
            model_list = model_service.list_models()
            for m in model_list:
                if m.get("is_best"):
                    best_model = m
                    break

        return {
            "gpu": gpu,
            "system": system,
            "dataset": {
                "prepared": dataset_status.get("cache_exists", False),
                "num_classes": dataset_status.get("num_classes", 0),
                "cache_size": dataset_status.get("cache_size", "N/A"),
                "is_preparing": dataset_status.get("is_preparing", False),
            },
            "training": {
                "is_training": training_status.get("is_training", False),
                "is_paused": training_status.get("is_paused", False),
                "current_epoch": training_status.get("current_epoch", 0),
                "best_val_acc": training_status.get("best_val_acc", 0),
            },
            "model": {
                "total_saved": len(model_list),
                "best_accuracy": best_model.get("val_acc") if best_model else None,
                "best_epoch": best_model.get("epoch") if best_model else None,
            },
        }

    def get_torch_info(self) -> Dict[str, Any]:
        """PyTorch and CUDA version info."""
        import torch
        info = {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": None,
            "cudnn_version": None,
            "device_count": 0,
            "devices": [],
        }

        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["cudnn_version"] = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None
            info["device_count"] = torch.cuda.device_count()
            for i in range(torch.cuda.device_count()):
                info["devices"].append({
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "capability": list(torch.cuda.get_device_capability(i)),
                })

        return info