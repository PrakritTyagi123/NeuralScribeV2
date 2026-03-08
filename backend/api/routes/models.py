"""
Model management API routes — language-aware list, load, delete, export.
"""

from fastapi import APIRouter, Request, Query, Body
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Dict

router = APIRouter()


def _services(request: Request):
    return request.app.state.services


class LoadRequest(BaseModel):
    name: str
    language: Optional[str] = None


class CompareRequest(BaseModel):
    model_a: str
    model_b: str


@router.get("/list")
async def list_models(request: Request, language: Optional[str] = Query(None)):
    ms = _services(request).model_service
    if language:
        ms.set_language(language)
    return {"language": ms.language, "models": ms.list_models()}


@router.get("/metadata/{name}")
async def model_metadata(request: Request, name: str, language: Optional[str] = Query(None)):
    ms = _services(request).model_service
    if language:
        ms.set_language(language)
    meta = ms.get_model_metadata(name)
    if meta is None:
        return {"error": f"Metadata not found for {name}"}
    return meta


@router.post("/load")
async def load_model(request: Request, body: LoadRequest):
    s = _services(request)
    if body.language:
        s.model_service.set_language(body.language)
        s.interface_service.set_language(body.language)

    result = s.model_service.load_model(body.name)
    if "error" not in result:
        model = s.model_service.get_loaded_model()
        if model:
            s.interface_service.set_model(model)
    return result


@router.post("/export-onnx/{name}")
async def export_onnx(request: Request, name: str, language: Optional[str] = Query(None)):
    ms = _services(request).model_service
    if language:
        ms.set_language(language)
    return ms.export_onnx(name)


@router.delete("/{name}")
async def delete_model(request: Request, name: str, language: Optional[str] = Query(None)):
    ms = _services(request).model_service
    if language:
        ms.set_language(language)
    return ms.delete_model(name)


@router.post("/compare")
async def compare_models(request: Request, body: CompareRequest):
    return _services(request).model_service.compare_models(body.model_a, body.model_b)


@router.get("/download/{name}")
async def download_model(request: Request, name: str):
    path = _services(request).model_service.get_model_path(name)
    if path is None:
        return {"error": f"Model not found: {name}"}
    return FileResponse(str(path), filename=path.name, media_type="application/octet-stream")


@router.get("/download-onnx/{name}")
async def download_onnx(request: Request, name: str):
    path = _services(request).model_service.get_export_path(name)
    if path is None:
        return {"error": f"ONNX export not found: {name}"}
    return FileResponse(str(path), filename=f"{name}.onnx", media_type="application/octet-stream")


@router.get("/loaded")
async def loaded_model_info(request: Request):
    s = _services(request)
    name = s.model_service.get_loaded_model_name()
    model = s.model_service.get_loaded_model()
    if model is None:
        return {"loaded": False, "language": s.model_service.language}
    return {
        "loaded": True,
        "name": name,
        "language": s.model_service.language,
        "n_params": model.count_parameters(),
        "num_classes": model.num_classes,
    }


@router.post("/unload")
async def unload_model(request: Request):
    s = _services(request)
    s.interface_service.clear_model()
    s.model_service.unload_model()
    return {"message": "Model unloaded"}


@router.post("/set-language")
async def set_model_language(request: Request, body: Dict[str, str] = Body(...)):
    language = body.get("language")
    if not language:
        return {"error": "language is required"}
    return _services(request).model_service.set_language(language)
