"""
Explainability API routes — language-aware feature maps, probability evolution.
"""

from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import List

router = APIRouter()


def _services(request: Request):
    return request.app.state.services


class ExplainRequest(BaseModel):
    pixels: List[float]


class LayerRequest(BaseModel):
    pixels: List[float]
    layer_name: str
    max_channels: int = 16


@router.post("/full")
async def full_explain(request: Request, body: ExplainRequest):
    svc = _services(request).interface_service
    if not svc.model_loaded:
        return {"error": "No model loaded"}
    return svc.explain(pixel_data=body.pixels)


@router.post("/live")
async def live_explain(request: Request, body: ExplainRequest):
    svc = _services(request).interface_service
    if not svc.model_loaded:
        return {"error": "No model loaded"}
    return svc.explain_live(pixel_data=body.pixels)


@router.post("/layer")
async def layer_feature_maps(request: Request, body: LayerRequest):
    svc = _services(request).interface_service
    if not svc.model_loaded:
        return {"error": "No model loaded"}
    return svc.get_layer_feature_maps(
        pixel_data=body.pixels,
        layer_name=body.layer_name,
        max_channels=body.max_channels,
    )
