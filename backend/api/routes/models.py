"""
Model management API routes — list, load, delete, export, compare, download.
"""

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

router = APIRouter()


def _services(request: Request):
    return request.app.state.services


class LoadRequest(BaseModel):
    name: str


class CompareRequest(BaseModel):
    model_a: str
    model_b: str


@router.get("/list")
async def list_models(request: Request):
    """List all saved models."""
    return {"models": _services(request).model_service.list_models()}


@router.get("/metadata/{name}")
async def model_metadata(request: Request, name: str):
    """Get metadata for a specific model."""
    meta = _services(request).model_service.get_model_metadata(name)
    if meta is None:
        return {"error": f"Metadata not found for {name}"}
    return meta


@router.post("/load")
async def load_model(request: Request, body: LoadRequest):
    """Load a model checkpoint for inference."""
    s = _services(request)
    result = s.model_service.load_model(body.name)

    if "error" not in result:
        # Set model in interface service for inference
        model = s.model_service.get_loaded_model()
        if model:
            s.interface_service.set_model(model)

    return result


@router.post("/export-onnx/{name}")
async def export_onnx(request: Request, name: str):
    """Export a model to ONNX format."""
    return _services(request).model_service.export_onnx(name)


@router.delete("/{name}")
async def delete_model(request: Request, name: str):
    """Delete a model and its metadata."""
    return _services(request).model_service.delete_model(name)


@router.post("/compare")
async def compare_models(request: Request, body: CompareRequest):
    """Compare two models side-by-side."""
    return _services(request).model_service.compare_models(body.model_a, body.model_b)


@router.get("/download/{name}")
async def download_model(request: Request, name: str):
    """Download a model .pth file."""
    path = _services(request).model_service.get_model_path(name)
    if path is None:
        return {"error": f"Model not found: {name}"}
    return FileResponse(
        str(path),
        filename=path.name,
        media_type="application/octet-stream",
    )


@router.get("/download-onnx/{name}")
async def download_onnx(request: Request, name: str):
    """Download an ONNX export."""
    path = _services(request).model_service.get_export_path(name)
    if path is None:
        return {"error": f"ONNX export not found: {name}"}
    return FileResponse(
        str(path),
        filename=f"{name}.onnx",
        media_type="application/octet-stream",
    )


@router.get("/loaded")
async def loaded_model_info(request: Request):
    """Get info about the currently loaded model."""
    s = _services(request)
    name = s.model_service.get_loaded_model_name()
    model = s.model_service.get_loaded_model()

    if model is None:
        return {"loaded": False}

    return {
        "loaded": True,
        "name": name,
        "n_params": model.count_parameters(),
        "num_classes": model.num_classes,
    }


@router.post("/unload")
async def unload_model(request: Request):
    """Unload the currently loaded model from memory."""
    s = _services(request)
    s.interface_service.clear_model()
    s.model_service.unload_model()
    return {"message": "Model unloaded"}