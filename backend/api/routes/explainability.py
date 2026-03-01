"""
Explainability API routes — feature maps, probability evolution, layer inspection.
"""

from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import List

router = APIRouter()


def _services(request: Request):
    return request.app.state.services


class ExplainRequest(BaseModel):
    """Canvas pixel data for explainability."""
    pixels: List[float]


class LayerRequest(BaseModel):
    """Request feature maps for a specific layer."""
    pixels: List[float]
    layer_name: str
    max_channels: int = 16


@router.post("/full")
async def full_explain(request: Request, body: ExplainRequest):
    """Full explainability pass with feature maps and probability evolution."""
    svc = _services(request).interface_service
    if not svc.model_loaded:
        return {"error": "No model loaded"}
    return svc.explain(pixel_data=body.pixels)


@router.post("/live")
async def live_explain(request: Request, body: ExplainRequest):
    """
    Lightweight real-time endpoint for NN diagram.
    Returns per-layer activation stats + top predictions, no base64 images.
    """
    svc = _services(request).interface_service
    if not svc.model_loaded:
        return {"error": "No model loaded"}
    return svc.explain_live(pixel_data=body.pixels)


@router.post("/layer")
async def layer_feature_maps(request: Request, body: LayerRequest):
    """
    Get feature maps for a specific layer (on-demand when user clicks a block).
    Returns heatmaps + channel importance for that layer.
    """
    svc = _services(request).interface_service

    if not svc.model_loaded:
        return {"error": "No model loaded"}

    return svc.get_layer_feature_maps(
        pixel_data=body.pixels,
        layer_name=body.layer_name,
        max_channels=body.max_channels,
    )