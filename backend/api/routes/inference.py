"""
Inference API routes — predict from canvas pixel data.
"""

from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter()


def _services(request: Request):
    return request.app.state.services


class PredictRequest(BaseModel):
    """Canvas pixel data for inference."""
    pixels: List[float]        # flat array of pixel values (e.g., 784 floats for 28x28)
    top_k: int = 5
    use_tta: bool = False


@router.post("/predict")
async def predict(request: Request, body: PredictRequest):
    """
    Run inference on canvas pixel data.
    Returns top-k predictions, confidences, category probabilities, and timing.
    """
    svc = _services(request).interface_service

    if not svc.model_loaded:
        return {"error": "No model loaded. Load a model first via Model Manager."}

    try:
        result = svc.predict(
            pixel_data=body.pixels,
            top_k=body.top_k,
            use_tta=body.use_tta,
        )
        return result
    except Exception as e:
        return {"error": str(e)}


@router.post("/debug-preview")
async def debug_preview(request: Request, body: PredictRequest):
    """Return the processed 28x28 image as base64 for debugging."""
    import numpy as np
    import base64
    import io
    from PIL import Image as PILImage
    from ...ml.preprocess import preprocess_canvas_data, DEFAULT_MEAN, DEFAULT_STD

    tensor = preprocess_canvas_data(body.pixels)
    # Denormalize
    img = tensor.squeeze().numpy() * DEFAULT_STD + DEFAULT_MEAN
    img = (img * 255).clip(0, 255).astype(np.uint8)

    pil = PILImage.fromarray(img, mode='L')
    buf = io.BytesIO()
    pil.save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode()

    return {"image_b64": b64, "shape": list(img.shape)}


@router.get("/status")
async def inference_status(request: Request):
    """Check if inference is ready (model loaded)."""
    svc = _services(request).interface_service
    s = _services(request)
    name = s.model_service.get_loaded_model_name()

    return {
        "ready": svc.model_loaded,
        "loaded_model": name,
    }