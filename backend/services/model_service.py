"""
Model management service for NeuralScribe v2.
Handles model registry (index.json), loading checkpoints, ONNX export,
model comparison, and artifact management.
"""

import json
import shutil
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Any, Optional, List

from ..utils.config import Config, ClassRegistry, PROJECT_ROOT
from ..utils.logging import get_logger
from ..utils.helpers import human_readable_size, timestamp_str
from ..ml.model import NeuralScribeNet

log = get_logger(__name__)


class ModelService:
    """Manages saved model artifacts."""

    def __init__(self):
        self._config = Config("configs/train_v2.yaml")
        self._registry = ClassRegistry()
        self._models_dir = PROJECT_ROOT / self._config.get("checkpointing.models_dir", "backend/models")
        self._models_dir.mkdir(parents=True, exist_ok=True)
        self._exports_dir = self._models_dir / "exports"
        self._exports_dir.mkdir(parents=True, exist_ok=True)
        self._loaded_model: Optional[NeuralScribeNet] = None
        self._loaded_model_name: Optional[str] = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Registry ──

    def list_models(self) -> List[Dict[str, Any]]:
        """Return all models from index.json + best_model if it exists."""
        models = []

        # Load index
        index_path = self._models_dir / "index.json"
        if index_path.exists():
            try:
                with open(index_path) as f:
                    models = json.load(f)
            except Exception:
                models = []

        # Add best_model entry if present
        best_path = self._models_dir / "best_model.pth"
        best_meta_path = self._models_dir / "metadata_best_model.json"
        if best_path.exists():
            best_entry = {
                "name": "best_model",
                "file": "best_model.pth",
                "size": human_readable_size(best_path.stat().st_size),
                "is_best": True,
            }
            if best_meta_path.exists():
                try:
                    with open(best_meta_path) as f:
                        meta = json.load(f)
                    best_entry.update({
                        "epoch": meta.get("epoch"),
                        "val_acc": meta.get("val_acc"),
                        "timestamp": meta.get("timestamp"),
                        "n_params": meta.get("n_params"),
                    })
                except Exception:
                    pass

            # Prepend best model
            models = [best_entry] + models

        return models

    def get_model_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific model."""
        meta_path = self._models_dir / f"metadata_{name}.json"
        if meta_path.exists():
            with open(meta_path) as f:
                return json.load(f)

        # Try without prefix
        for p in self._models_dir.glob(f"metadata_*{name}*.json"):
            with open(p) as f:
                return json.load(f)

        return None

    # ── Loading ──

    def load_model(self, name: str) -> Dict[str, Any]:
        """Load a model checkpoint into memory for inference."""
        # Find the .pth file
        pth_path = self._models_dir / f"{name}.pth"
        if not pth_path.exists():
            pth_path = self._models_dir / name
        if not pth_path.exists():
            # Search
            matches = list(self._models_dir.glob(f"*{name}*.pth"))
            if matches:
                pth_path = matches[0]
            else:
                return {"error": f"Model not found: {name}"}

        try:
            ckpt = torch.load(pth_path, map_location=self._device, weights_only=False)

            # Build model
            ckpt_config = ckpt.get("config", {})
            model_cfg = ckpt_config.get("model", {})

            model = NeuralScribeNet(
                num_classes=ckpt.get("num_classes", self._registry.num_classes),
                stem_channels=model_cfg.get("stem_channels", 32),
                block_channels=model_cfg.get("block_channels", [64, 128, 256, 320]),
                se_reduction=model_cfg.get("se_reduction", 16),
                dropout=model_cfg.get("dropout", 0.4),
                drop_path_rate=model_cfg.get("drop_path_rate", 0.1),
            ).to(self._device)

            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            self._loaded_model = model
            self._loaded_model_name = name

            log.info(f"Loaded model: {name} ({model.count_parameters():,} params)")

            return {
                "status": "loaded",
                "name": name,
                "epoch": ckpt.get("epoch"),
                "val_acc": ckpt.get("val_acc"),
                "n_params": model.count_parameters(),
            }
        except Exception as e:
            log.error(f"Failed to load model {name}: {e}", exc_info=True)
            return {"error": str(e)}

    def get_loaded_model(self) -> Optional[NeuralScribeNet]:
        return self._loaded_model

    def get_loaded_model_name(self) -> Optional[str]:
        return self._loaded_model_name

    def unload_model(self) -> None:
        """Unload the currently loaded model from memory."""
        if self._loaded_model is not None:
            log.info(f"Unloading model: {self._loaded_model_name}")
        self._loaded_model = None
        self._loaded_model_name = None

    # ── ONNX Export ──

    def export_onnx(self, name: str) -> Dict[str, Any]:
        """Export a model to ONNX format."""
        # Load the model first if needed
        if self._loaded_model is None or self._loaded_model_name != name:
            result = self.load_model(name)
            if "error" in result:
                return result

        model = self._loaded_model
        if model is None:
            return {"error": "No model loaded"}

        model.eval()
        onnx_name = f"{name}.onnx"
        onnx_path = self._exports_dir / onnx_name

        try:
            dummy_input = torch.randn(1, 1, 28, 28, device=self._device)

            torch.onnx.export(
                model,
                dummy_input,
                str(onnx_path),
                export_params=True,
                opset_version=13,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["logits"],
                dynamic_axes={
                    "input": {0: "batch_size"},
                    "logits": {0: "batch_size"},
                },
            )

            # Validate
            import onnx
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)

            log.info(f"Exported ONNX: {onnx_path}")

            return {
                "status": "exported",
                "path": str(onnx_path),
                "name": onnx_name,
                "size": human_readable_size(onnx_path.stat().st_size),
            }
        except Exception as e:
            log.error(f"ONNX export failed: {e}", exc_info=True)
            return {"error": str(e)}

    # ── Deletion ──

    def delete_model(self, name: str) -> Dict[str, Any]:
        """Delete a model and its metadata."""
        deleted = []

        # .pth file
        pth = self._models_dir / f"{name}.pth"
        if pth.exists():
            pth.unlink()
            deleted.append(str(pth))

        # metadata
        meta = self._models_dir / f"metadata_{name}.json"
        if meta.exists():
            meta.unlink()
            deleted.append(str(meta))

        # ONNX export
        onnx = self._exports_dir / f"{name}.onnx"
        if onnx.exists():
            onnx.unlink()
            deleted.append(str(onnx))

        # Update index
        self._remove_from_index(name)

        if self._loaded_model_name == name:
            self._loaded_model = None
            self._loaded_model_name = None

        if deleted:
            log.info(f"Deleted model: {name}")
            return {"status": "deleted", "files": deleted}
        return {"error": f"Model not found: {name}"}

    def _remove_from_index(self, name: str):
        index_path = self._models_dir / "index.json"
        if not index_path.exists():
            return
        try:
            with open(index_path) as f:
                index = json.load(f)
            index = [e for e in index if e.get("name") != name]
            with open(index_path, "w") as f:
                json.dump(index, f, indent=2)
        except Exception:
            pass

    # ── Comparison ──

    def compare_models(self, name_a: str, name_b: str) -> Dict[str, Any]:
        """Compare two models side-by-side (metadata + training history)."""
        meta_a = self.get_model_metadata(name_a)
        meta_b = self.get_model_metadata(name_b)

        if not meta_a:
            return {"error": f"Metadata not found for {name_a}"}
        if not meta_b:
            return {"error": f"Metadata not found for {name_b}"}

        # Load histories from checkpoints if available
        history_a = self._load_history(name_a)
        history_b = self._load_history(name_b)

        return {
            "model_a": {
                "name": name_a,
                "metadata": meta_a,
                "history": history_a,
            },
            "model_b": {
                "name": name_b,
                "metadata": meta_b,
                "history": history_b,
            },
        }

    def _load_history(self, name: str) -> List[Dict]:
        """Try to load training history from a checkpoint."""
        pth_path = self._models_dir / f"{name}.pth"
        if not pth_path.exists():
            return []
        try:
            ckpt = torch.load(pth_path, map_location="cpu", weights_only=False)
            return ckpt.get("history", [])
        except Exception:
            return []

    # ── Download path ──

    def get_model_path(self, name: str) -> Optional[Path]:
        """Get the file path for downloading a model."""
        pth = self._models_dir / f"{name}.pth"
        if pth.exists():
            return pth
        pth = self._models_dir / name
        if pth.exists():
            return pth
        return None

    def get_export_path(self, name: str) -> Optional[Path]:
        """Get the file path for downloading an ONNX export."""
        onnx = self._exports_dir / f"{name}.onnx"
        if onnx.exists():
            return onnx
        return None