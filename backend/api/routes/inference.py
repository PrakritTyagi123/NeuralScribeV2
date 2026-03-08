"""
Inference API routes — language-aware prediction from canvas pixel data.
"""

from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter()


def _services(request: Request):
    return request.app.state.services


class PredictRequest(BaseModel):
    pixels: List[float]
    top_k: int = 5
    use_tta: bool = False

    def validate_pixels(self):
        import math
        size = int(math.sqrt(len(self.pixels)))
        if size * size != len(self.pixels) or size < 1:
            raise ValueError(f"pixels must be a perfect square length, got {len(self.pixels)}")


@router.post("/predict")
async def predict(request: Request, body: PredictRequest):
    try:
        body.validate_pixels()
    except ValueError as e:
        return {"error": str(e)}

    svc = _services(request).interface_service
    if not svc.model_loaded:
        return {"error": "No model loaded. Load a model first via Model Manager."}

    try:
        return svc.predict(pixel_data=body.pixels, top_k=body.top_k, use_tta=body.use_tta)
    except Exception as e:
        return {"error": str(e)}


@router.post("/debug-preview")
async def debug_preview(request: Request, body: PredictRequest):
    import numpy as np
    import base64
    import io
    from PIL import Image as PILImage
    from ...ml.preprocess import preprocess_canvas_data, DEFAULT_MEAN, DEFAULT_STD

    tensor = preprocess_canvas_data(body.pixels)
    img = tensor.squeeze().numpy() * DEFAULT_STD + DEFAULT_MEAN
    img = (img * 255).clip(0, 255).astype(np.uint8)
    pil = PILImage.fromarray(img, mode='L')
    buf = io.BytesIO()
    pil.save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode()
    return {"image_b64": b64, "shape": list(img.shape)}


@router.get("/status")
async def inference_status(request: Request):
    svc = _services(request).interface_service
    s = _services(request)
    name = s.model_service.get_loaded_model_name()
    return {
        "ready": svc.model_loaded,
        "loaded_model": name,
        "language": svc.language,
    }
