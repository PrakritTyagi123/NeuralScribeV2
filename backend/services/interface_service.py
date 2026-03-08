"""
NeuralScribe v2 — Interface service (language-aware).
Handles inference, TTA, and explainability per language.
"""

import io
import base64
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
from typing import Dict, Any, Optional, List

from ..utils.config import ClassRegistry, ProjectConfig, SUPPORTED_LANGUAGES, get_language_paths
from ..utils.logging import get_logger
from ..ml.model import NeuralScribeNet
from ..ml.preprocess import preprocess_canvas_data, DEFAULT_MEAN, DEFAULT_STD
from ..ml.augmentation import TTAAugmentor

log = get_logger(__name__)


class InterfaceService:
    """Inference and explainability service, per language."""

    def __init__(self):
        self._project_config = ProjectConfig()
        self._language = self._project_config.selected_language
        self._registry = ClassRegistry(language=self._language)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model: Optional[NeuralScribeNet] = None
        self._tta = TTAAugmentor()

    # ── Language switching ──

    @property
    def language(self) -> str:
        return self._language

    def set_language(self, language: str) -> Dict[str, Any]:
        if language not in SUPPORTED_LANGUAGES:
            return {"error": f"Unsupported language: {language}"}
        if language == self._language:
            return {"status": "already_set", "language": language}

        self._language = language
        self._registry = ClassRegistry(language=language)
        self._model = None  # Must reload model for new language

        self._project_config.set_ui_state("inference_language", language)
        log.info(f"Interface service switched to language: {language}")
        return {"status": "switched", "language": language}

    def set_model(self, model: NeuralScribeNet, registry: ClassRegistry = None):
        """Set the model for inference."""
        self._model = model
        self._model.eval()
        if registry:
            self._registry = registry

    def clear_model(self) -> None:
        self._model = None

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    # ── Prediction ──

    def predict(self, pixel_data: list, top_k: int = 5, use_tta: bool = False) -> Dict[str, Any]:
        if self._model is None:
            return {"error": "No model loaded"}

        start = time.perf_counter()
        tensor = preprocess_canvas_data(pixel_data)
        tensor = tensor.to(self._device)
        self._model.eval()

        with torch.no_grad():
            if use_tta:
                probs = self._predict_tta(tensor)
            else:
                logits = self._model(tensor)
                probs = F.softmax(logits, dim=-1).squeeze(0)

        elapsed_ms = (time.perf_counter() - start) * 1000

        top_probs, top_indices = probs.topk(top_k)
        top_probs = top_probs.cpu().numpy()
        top_indices = top_indices.cpu().numpy()

        predictions = []
        for i in range(top_k):
            class_id = int(top_indices[i])
            predictions.append({
                "class_id": class_id,
                "label": self._registry.id_to_label(class_id),
                "display": self._registry.id_to_display(class_id),
                "category": self._registry.id_to_category(class_id),
                "confidence": round(float(top_probs[i]), 4),
            })

        predicted = predictions[0]
        all_probs = probs.cpu().numpy()
        category_probs = self._group_probabilities_by_category(all_probs)

        return {
            "language": self._language,
            "predicted_class": predicted["class_id"],
            "predicted_label": predicted["display"],
            "confidence": predicted["confidence"],
            "top_k": predictions,
            "category_probabilities": category_probs,
            "inference_time_ms": round(elapsed_ms, 2),
            "used_tta": use_tta,
        }

    def _predict_tta(self, tensor: torch.Tensor) -> torch.Tensor:
        variants = self._tta.generate_variants(tensor)
        variants = variants.to(self._device)
        with torch.no_grad():
            logits = self._model(variants)
            probs = F.softmax(logits, dim=-1)
        return probs.mean(dim=0)

    def _group_probabilities_by_category(self, all_probs: np.ndarray) -> Dict[str, List[Dict]]:
        grouped = {}
        for cls in self._registry.classes:
            cat = cls["category"]
            if cat not in grouped:
                grouped[cat] = []
            prob_val = float(all_probs[cls["id"]]) if cls["id"] < len(all_probs) else 0.0
            grouped[cat].append({
                "class_id": cls["id"],
                "display": cls["display"],
                "label": cls["label"],
                "probability": round(prob_val, 4),
            })
        for cat in grouped:
            grouped[cat].sort(key=lambda x: x["probability"], reverse=True)
        return grouped

    # ── Explainability ──

    def explain(self, pixel_data: list) -> Dict[str, Any]:
        if self._model is None:
            return {"error": "No model loaded"}

        start = time.perf_counter()
        tensor = preprocess_canvas_data(pixel_data)
        tensor = tensor.to(self._device)
        self._model.eval()

        with torch.no_grad():
            logits, activations = self._model.forward_with_intermediates(tensor)
            probs = F.softmax(logits, dim=-1).squeeze(0)

        prob_evolution = self._compute_probability_evolution(tensor)

        feature_maps = {}
        for layer_name, act in activations.items():
            if layer_name == "logits":
                continue
            fm = self._render_feature_maps(act, max_channels=16)
            if fm is not None:
                feature_maps[layer_name] = fm

        top_probs, top_indices = probs.topk(5)
        predictions = []
        for i in range(5):
            cid = int(top_indices[i])
            predictions.append({
                "class_id": cid,
                "display": self._registry.id_to_display(cid),
                "confidence": round(float(top_probs[i]), 4),
            })

        input_image = self._tensor_to_base64(tensor.squeeze())
        elapsed_ms = (time.perf_counter() - start) * 1000

        return {
            "language": self._language,
            "predictions": predictions,
            "input_image": input_image,
            "feature_maps": feature_maps,
            "probability_evolution": prob_evolution,
            "inference_time_ms": round(elapsed_ms, 2),
            "layers": list(activations.keys()),
        }

    def _compute_probability_evolution(self, tensor: torch.Tensor) -> List[Dict]:
        evolution_raw = self._model.get_probability_evolution(tensor)
        evolution = []
        for step in evolution_raw:
            probs = step["probabilities"]
            top_indices = np.argsort(probs)[::-1][:5]
            top_entries = []
            for idx in top_indices:
                idx = int(idx)
                top_entries.append({
                    "class_id": idx,
                    "display": self._registry.id_to_display(idx),
                    "probability": round(float(probs[idx]), 4),
                })
            evolution.append({"layer": step["layer"], "top_5": top_entries})
        return evolution

    def _render_feature_maps(self, act, max_channels=16):
        if act.dim() == 4:
            act = act.squeeze(0)
            C = act.shape[0]
            channel_importance = act.mean(dim=(1, 2)).cpu().numpy()
            top_channels = np.argsort(channel_importance)[::-1][:max_channels]
            heatmaps = []
            importance_list = []
            for ch_idx in top_channels:
                ch_data = act[int(ch_idx)].cpu().numpy()
                importance_val = float(channel_importance[int(ch_idx)])
                b64 = self._array_to_heatmap_base64(ch_data)
                heatmaps.append({"channel": int(ch_idx), "importance": round(importance_val, 4), "heatmap": b64})
                importance_list.append({"channel": int(ch_idx), "importance": round(importance_val, 4)})
            return {
                "heatmaps": heatmaps,
                "importance": importance_list,
                "total_channels": C,
                "spatial_size": [int(act.shape[1]), int(act.shape[2])],
            }
        if act.dim() == 2:
            vec = act.squeeze(0).cpu().numpy()
        elif act.dim() == 1:
            vec = act.cpu().numpy()
        else:
            return None
        if vec.ndim != 1:
            return None
        stripe = vec[None, :]
        importance_val = float(np.mean(np.abs(vec)))
        b64 = self._array_to_heatmap_base64(stripe)
        return {
            "heatmaps": [{"channel": 0, "importance": round(importance_val, 4), "heatmap": b64}],
            "importance": [{"channel": 0, "importance": round(importance_val, 4)}],
            "total_channels": int(vec.shape[0]),
            "spatial_size": [stripe.shape[0], stripe.shape[1]],
        }

    def _array_to_heatmap_base64(self, arr: np.ndarray, size: int = 56) -> str:
        vmin, vmax = arr.min(), arr.max()
        normalized = (arr - vmin) / (vmax - vmin) if vmax - vmin > 1e-8 else np.zeros_like(arr)
        colored = cm.viridis(normalized)
        colored = (colored[:, :, :3] * 255).astype(np.uint8)
        img = Image.fromarray(colored)
        img = img.resize((size, size), Image.NEAREST)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _tensor_to_base64(self, tensor: torch.Tensor, size: int = 112) -> str:
        if tensor.dim() == 3:
            tensor = tensor.squeeze(0)
        arr = (tensor.cpu().numpy() * DEFAULT_STD + DEFAULT_MEAN)
        arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr, mode="L")
        img = img.resize((size, size), Image.NEAREST)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # ── Layer feature maps on demand ──

    def get_layer_feature_maps(self, pixel_data: list, layer_name: str, max_channels: int = 16) -> Dict[str, Any]:
        if self._model is None:
            return {"error": "No model loaded"}
        tensor = preprocess_canvas_data(pixel_data)
        tensor = tensor.to(self._device)
        self._model.eval()
        with torch.no_grad():
            _, activations = self._model.forward_with_intermediates(tensor)
        if layer_name not in activations:
            return {"error": f"Layer not found: {layer_name}"}
        act = activations[layer_name]
        if act.dim() != 4:
            return {"error": f"Layer {layer_name} is not a conv layer"}
        return {"layer": layer_name, **self._render_feature_maps(act, max_channels)}

    # ── Live explainability ──

    def explain_live(self, pixel_data: list) -> Dict[str, Any]:
        if self._model is None:
            return {"error": "No model loaded"}

        start = time.perf_counter()
        tensor = preprocess_canvas_data(pixel_data)
        tensor = tensor.to(self._device)
        self._model.eval()
        with torch.no_grad():
            logits, activations = self._model.forward_with_intermediates(tensor)
            probs = F.softmax(logits, dim=-1).squeeze(0)

        top_probs, top_indices = probs.topk(5)
        predictions = []
        for i in range(5):
            cid = int(top_indices[i])
            predictions.append({
                "class_id": cid,
                "display": self._registry.id_to_display(cid),
                "confidence": round(float(top_probs[i]), 4),
            })

        layer_stats = []
        input_flat = tensor.squeeze(0).squeeze(0).cpu()
        input_nodes = []
        step = 28 // 4
        for r in range(4):
            for c in range(4):
                patch = input_flat[r*step:(r+1)*step, c*step:(c+1)*step]
                input_nodes.append(round(float(patch.mean()), 4))
        layer_stats.append({
            "name": "input", "type": "input", "channels": 1,
            "spatial": [28, 28], "mean_activation": round(float(input_flat.mean()), 4),
            "node_values": input_nodes,
        })

        for layer_name, act in activations.items():
            act_cpu = act.squeeze(0).cpu()
            if act_cpu.dim() == 3:
                channel_means = act_cpu.mean(dim=(1, 2)).numpy()
                cmin, cmax = channel_means.min(), channel_means.max()
                channel_norms = ((channel_means - cmin) / (cmax - cmin)).tolist() if cmax - cmin > 1e-8 else [0.0] * len(channel_means)
                layer_stats.append({
                    "name": layer_name, "type": "conv",
                    "channels": int(act_cpu.shape[0]),
                    "spatial": [int(act_cpu.shape[1]), int(act_cpu.shape[2])],
                    "mean_activation": round(float(act_cpu.mean()), 4),
                    "max_activation": round(float(act_cpu.max()), 4),
                    "node_values": channel_norms[:32],
                })
            elif act_cpu.dim() in (1, 2):
                vals = act_cpu.flatten().numpy() if act_cpu.dim() == 2 else act_cpu.numpy()
                vmin, vmax = vals.min(), vals.max()
                norms = ((vals - vmin) / (vmax - vmin)).tolist() if vmax - vmin > 1e-8 else [0.0] * len(vals)
                layer_stats.append({
                    "name": layer_name, "type": "fc",
                    "channels": int(len(vals)),
                    "mean_activation": round(float(vals.mean()), 4),
                    "node_values": norms[:32],
                })

        output_probs = probs.cpu().numpy()
        layer_stats.append({
            "name": "output", "type": "output",
            "channels": int(len(output_probs)),
            "node_values": output_probs.tolist(),
            "winner": int(top_indices[0]),
        })

        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "language": self._language,
            "predictions": predictions,
            "layers": layer_stats,
            "inference_time_ms": round(elapsed_ms, 2),
        }
