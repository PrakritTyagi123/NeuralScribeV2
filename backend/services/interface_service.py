"""
Interface service for NeuralScribe v2.
Handles inference requests, test-time augmentation, and explainability
(feature map rendering, probability evolution).
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
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Dict, Any, Optional, List

from ..utils.config import ClassRegistry, PROJECT_ROOT
from ..utils.logging import get_logger
from ..ml.model import NeuralScribeNet
from ..ml.preprocess import preprocess_canvas_data, DEFAULT_MEAN, DEFAULT_STD
from ..ml.augmentation import TTAAugmentor

log = get_logger(__name__)


class InterfaceService:
    """
    Inference and explainability service.
    Provides prediction, TTA, feature map visualization, and probability evolution.
    """

    def __init__(self):
        self._registry = ClassRegistry()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model: Optional[NeuralScribeNet] = None
        self._tta = TTAAugmentor()

    def set_model(self, model: NeuralScribeNet):
        """Set the model for inference (called after loading or training)."""
        self._model = model
        self._model.eval()

    def clear_model(self) -> None:
        """Clear the currently loaded model (used by maintenance endpoints)."""
        self._model = None

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    # ── Prediction ──

    def predict(
        self,
        pixel_data: list,
        top_k: int = 5,
        use_tta: bool = False,
    ) -> Dict[str, Any]:
        """
        Run inference on canvas pixel data.
        Returns top-k predictions with confidence and timing.
        """
        if self._model is None:
            return {"error": "No model loaded"}

        start = time.perf_counter()

        # Preprocess
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

        # Top-k
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

        # Full probability breakdown by category
        all_probs = probs.cpu().numpy()
        category_probs = self._group_probabilities_by_category(all_probs)

        return {
            "predicted_class": predicted["class_id"],
            "predicted_label": predicted["display"],
            "confidence": predicted["confidence"],
            "top_k": predictions,
            "category_probabilities": category_probs,
            "inference_time_ms": round(elapsed_ms, 2),
            "used_tta": use_tta,
        }

    def _predict_tta(self, tensor: torch.Tensor) -> torch.Tensor:
        """Run TTA and average probabilities."""
        variants = self._tta.generate_variants(tensor)
        variants = variants.to(self._device)

        with torch.no_grad():
            logits = self._model(variants)
            probs = F.softmax(logits, dim=-1)

        # Average
        return probs.mean(dim=0)

    def _group_probabilities_by_category(self, all_probs: np.ndarray) -> Dict[str, List[Dict]]:
        """Group all class probabilities by category for the UI grid display."""
        grouped = {}
        for cls in self._registry.classes:
            cat = cls["category"]
            if cat not in grouped:
                grouped[cat] = []
            grouped[cat].append({
                "class_id": cls["id"],
                "display": cls["display"],
                "label": cls["label"],
                "probability": round(float(all_probs[cls["id"]]), 4),
            })

        # Sort each category by probability descending
        for cat in grouped:
            grouped[cat].sort(key=lambda x: x["probability"], reverse=True)

        return grouped

    # ── Explainability ──

    def explain(self, pixel_data: list) -> Dict[str, Any]:
        """
        Full explainability pass: activations, feature maps as base64 heatmaps,
        and probability evolution through layers.
        """
        if self._model is None:
            return {"error": "No model loaded"}

        start = time.perf_counter()

        tensor = preprocess_canvas_data(pixel_data)
        tensor = tensor.to(self._device)

        self._model.eval()

        # Forward with intermediates
        with torch.no_grad():
            logits, activations = self._model.forward_with_intermediates(tensor)
            probs = F.softmax(logits, dim=-1).squeeze(0)

        # Probability evolution
        prob_evolution = self._compute_probability_evolution(tensor)

        # Feature map heatmaps (conv + pooled / output vectors)
        feature_maps = {}
        for layer_name, act in activations.items():
            if layer_name == "logits":
                continue
            fm = self._render_feature_maps(act, max_channels=16)
            if fm is not None:
                feature_maps[layer_name] = fm

        # Top prediction
        top_probs, top_indices = probs.topk(5)
        predictions = []
        for i in range(5):
            cid = int(top_indices[i])
            predictions.append({
                "class_id": cid,
                "display": self._registry.id_to_display(cid),
                "confidence": round(float(top_probs[i]), 4),
            })

        # Processed input image as base64
        input_image = self._tensor_to_base64(tensor.squeeze())

        elapsed_ms = (time.perf_counter() - start) * 1000

        return {
            "predictions": predictions,
            "input_image": input_image,
            "feature_maps": feature_maps,
            "probability_evolution": prob_evolution,
            "inference_time_ms": round(elapsed_ms, 2),
            "layers": list(activations.keys()),
        }

    def _compute_probability_evolution(self, tensor: torch.Tensor) -> List[Dict]:
        """
        Get probability evolution through the network using
        forward hooks + final head projection at each block output.
        """
        evolution_raw = self._model.get_probability_evolution(tensor)
        evolution = []

        for step in evolution_raw:
            probs = step["probabilities"]
            # Get top 5 at this layer
            top_indices = np.argsort(probs)[::-1][:5]
            top_entries = []
            for idx in top_indices:
                idx = int(idx)
                top_entries.append({
                    "class_id": idx,
                    "display": self._registry.id_to_display(idx),
                    "probability": round(float(probs[idx]), 4),
                })

            evolution.append({
                "layer": step["layer"],
                "top_5": top_entries,
            })

        return evolution

    def _render_feature_maps(
        self,
        activations: torch.Tensor,
        max_channels: int = 16,
    ) -> Optional[Dict[str, Any]]:
        """
        Render feature map activations as base64 heatmap images.

        Supports both convolutional feature maps (B, C, H, W) and 1D / 2D
        vectors such as pooled features or logits by treating each channel
        as a 1×1 spatial map.
        """
        act = activations

        # Standard conv feature maps: (B, C, H, W)
        if act.dim() == 4:
            act_np = act.squeeze(0).cpu().numpy()  # (C, H, W)

            n_channels = min(act_np.shape[0], max_channels)

            # Channel importance: mean absolute activation
            importance = np.mean(np.abs(act_np), axis=(1, 2))  # (C,)
            top_channels = np.argsort(importance)[::-1][:n_channels]

            heatmaps = []
            for ch_idx in top_channels:
                ch_map = act_np[ch_idx]
                b64 = self._array_to_heatmap_base64(ch_map)
                heatmaps.append({
                    "channel": int(ch_idx),
                    "importance": round(float(importance[ch_idx]), 4),
                    "heatmap": b64,
                })

            importance_bars = [
                {"channel": int(i), "importance": round(float(importance[i]), 4)}
                for i in np.argsort(importance)[::-1][:32]
            ]

            return {
                "heatmaps": heatmaps,
                "importance": importance_bars,
                "total_channels": act_np.shape[0],
                "spatial_size": [act_np.shape[1], act_np.shape[2]],
            }

        # Pooled / output vectors — render as a single stripe heatmap
        # so they are visible instead of tiny 1×1 tiles.
        if act.dim() == 2:
            vec = act.squeeze(0).cpu().numpy()  # (C,)
        elif act.dim() == 1:
            vec = act.cpu().numpy()  # (C,)
        else:
            return None

        if vec.ndim != 1:
            return None

        stripe = vec[None, :]  # (1, C)

        importance_val = float(np.mean(np.abs(vec)))
        b64 = self._array_to_heatmap_base64(stripe)

        return {
            "heatmaps": [{
                "channel": 0,
                "importance": round(importance_val, 4),
                "heatmap": b64,
            }],
            "importance": [{
                "channel": 0,
                "importance": round(importance_val, 4),
            }],
            "total_channels": int(vec.shape[0]),
            "spatial_size": [stripe.shape[0], stripe.shape[1]],
        }

    def _array_to_heatmap_base64(self, arr: np.ndarray, size: int = 56) -> str:
        """Convert a 2D numpy array to a base64-encoded heatmap PNG."""
        # Mirror horizontally so orientation matches drawing canvas
        arr = np.fliplr(arr)

        # Normalize to 0-1
        vmin, vmax = arr.min(), arr.max()
        if vmax - vmin > 1e-8:
            normalized = (arr - vmin) / (vmax - vmin)
        else:
            normalized = np.zeros_like(arr)

        # Apply colormap
        colored = cm.viridis(normalized)  # RGBA
        colored = (colored[:, :, :3] * 255).astype(np.uint8)  # RGB

        # Resize
        img = Image.fromarray(colored)
        img = img.resize((size, size), Image.NEAREST)

        # Encode
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _tensor_to_base64(self, tensor: torch.Tensor, size: int = 112) -> str:
        """Convert a (H, W) or (1, H, W) tensor to base64 grayscale PNG."""
        if tensor.dim() == 3:
            tensor = tensor.squeeze(0)

        # Denormalize
        arr = (tensor.cpu().numpy() * DEFAULT_STD + DEFAULT_MEAN)
        arr = np.clip(arr * 255, 0, 255).astype(np.uint8)

        img = Image.fromarray(arr, mode="L")
        img = img.resize((size, size), Image.NEAREST)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # ── Feature map for a specific layer (on demand) ──

    def get_layer_feature_maps(
        self,
        pixel_data: list,
        layer_name: str,
        max_channels: int = 16,
    ) -> Dict[str, Any]:
        """Get feature maps for a specific layer (clicked in UI)."""
        if self._model is None:
            return {"error": "No model loaded"}

        tensor = preprocess_canvas_data(pixel_data)
        tensor = tensor.to(self._device)

        self._model.eval()
        with torch.no_grad():
            _, activations = self._model.forward_with_intermediates(tensor)

        if layer_name not in activations:
            return {"error": f"Layer not found: {layer_name}", "available": list(activations.keys())}

        act = activations[layer_name]
        if act.dim() != 4:
            return {"error": f"Layer {layer_name} is not a conv layer (shape: {act.shape})"}

        return {
            "layer": layer_name,
            **self._render_feature_maps(act, max_channels),
        }

    # ── Lightweight live explainability ──

    def explain_live(self, pixel_data: list) -> Dict[str, Any]:
        """
        Fast explainability for real-time NN diagram.
        Returns per-layer activation stats (no base64), top predictions,
        and per-channel mean activations for node coloring.
        """
        if self._model is None:
            return {"error": "No model loaded"}

        start = time.perf_counter()

        tensor = preprocess_canvas_data(pixel_data)
        tensor = tensor.to(self._device)

        self._model.eval()
        with torch.no_grad():
            logits, activations = self._model.forward_with_intermediates(tensor)
            probs = F.softmax(logits, dim=-1).squeeze(0)

        # Top-5 predictions
        top_probs, top_indices = probs.topk(5)
        predictions = []
        for i in range(5):
            cid = int(top_indices[i])
            predictions.append({
                "class_id": cid,
                "display": self._registry.id_to_display(cid),
                "confidence": round(float(top_probs[i]), 4),
            })

        # Per-layer activation stats for NN diagram nodes
        layer_stats = []

        # Add input as first layer
        input_flat = tensor.squeeze(0).squeeze(0).cpu()  # (28, 28)
        input_mean = float(input_flat.mean())
        # Sample 16 patches from the input as node values
        input_nodes = []
        step = 28 // 4
        for r in range(4):
            for c in range(4):
                patch = input_flat[r*step:(r+1)*step, c*step:(c+1)*step]
                input_nodes.append(round(float(patch.mean()), 4))
        layer_stats.append({
            "name": "input",
            "type": "input",
            "channels": 1,
            "spatial": [28, 28],
            "mean_activation": round(input_mean, 4),
            "node_values": input_nodes,
        })

        for layer_name, act in activations.items():
            act_cpu = act.squeeze(0).cpu()

            if act_cpu.dim() == 3:  # Conv: (C, H, W)
                # Per-channel mean activation (for individual nodes)
                channel_means = act_cpu.mean(dim=(1, 2)).numpy()
                # Normalize to 0-1 range
                cmin, cmax = channel_means.min(), channel_means.max()
                if cmax - cmin > 1e-8:
                    channel_norms = ((channel_means - cmin) / (cmax - cmin)).tolist()
                else:
                    channel_norms = [0.0] * len(channel_means)

                layer_stats.append({
                    "name": layer_name,
                    "type": "conv",
                    "channels": int(act_cpu.shape[0]),
                    "spatial": [int(act_cpu.shape[1]), int(act_cpu.shape[2])],
                    "mean_activation": round(float(act_cpu.mean()), 4),
                    "max_activation": round(float(act_cpu.max()), 4),
                    "node_values": channel_norms[:32],  # Top 32 for diagram
                })
            elif act_cpu.dim() == 1:  # FC: (N,)
                vals = act_cpu.numpy()
                vmin, vmax = vals.min(), vals.max()
                if vmax - vmin > 1e-8:
                    norms = ((vals - vmin) / (vmax - vmin)).tolist()
                else:
                    norms = [0.0] * len(vals)
                layer_stats.append({
                    "name": layer_name,
                    "type": "fc",
                    "channels": int(len(vals)),
                    "mean_activation": round(float(vals.mean()), 4),
                    "node_values": norms[:32],
                })
            elif act_cpu.dim() == 2:  # Batch wasn't squeezed or 2D FC
                vals = act_cpu[0].numpy() if act_cpu.shape[0] == 1 else act_cpu.flatten().numpy()
                vmin, vmax = vals.min(), vals.max()
                if vmax - vmin > 1e-8:
                    norms = ((vals - vmin) / (vmax - vmin)).tolist()
                else:
                    norms = [0.0] * len(vals)
                layer_stats.append({
                    "name": layer_name,
                    "type": "fc",
                    "channels": int(len(vals)),
                    "mean_activation": round(float(vals.mean()), 4),
                    "node_values": norms[:32],
                })

        # Output layer probabilities as node values
        output_probs = probs.cpu().numpy()
        layer_stats.append({
            "name": "output",
            "type": "output",
            "channels": int(len(output_probs)),
            "node_values": output_probs.tolist(),
            "winner": int(top_indices[0]),
        })

        elapsed_ms = (time.perf_counter() - start) * 1000

        return {
            "predictions": predictions,
            "layers": layer_stats,
            "inference_time_ms": round(elapsed_ms, 2),
        }