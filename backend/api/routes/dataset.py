"""
Dataset API routes — language-aware data preparation, status, config.
"""

from fastapi import APIRouter, Request, Query, Body
from pydantic import BaseModel
from typing import Dict, Any, Optional

from ..ws import broadcast_event

router = APIRouter()


def _services(request: Request):
    return request.app.state.services


class PrepareRequest(BaseModel):
    language: Optional[str] = None
    overrides: Optional[Dict[str, Any]] = None


@router.get("/status")
async def dataset_status(request: Request, language: Optional[str] = Query(None)):
    svc = _services(request).dataset_service
    if language:
        svc.set_language(language)
    return svc.get_status()


@router.get("/config")
async def dataset_config(request: Request, language: Optional[str] = Query(None)):
    svc = _services(request).dataset_service
    if language:
        svc.set_language(language)
    return svc.get_config()


@router.post("/config")
async def update_dataset_config(request: Request, body: Dict[str, Any] = Body(...)):
    svc = _services(request).dataset_service
    language = body.pop("language", None)
    if language:
        svc.set_language(language)
    svc.update_config(body)
    return {"status": "updated", "language": svc.language, "applied": body}


@router.post("/prepare")
async def prepare_dataset(request: Request, body: PrepareRequest = PrepareRequest()):
    svc = _services(request).dataset_service

    if body.language:
        result = svc.set_language(body.language)
        if "error" in result:
            return result

    if svc.is_preparing:
        return {"error": "Dataset preparation already in progress"}

    if body.overrides:
        svc.update_config(body.overrides)

    async def ws_progress(data: Dict[str, Any]):
        data["type"] = "dataset_progress"
        await broadcast_event(data)

    import asyncio

    async def run_prep():
        result = await svc.prepare_dataset(ws_callback=ws_progress)
        await broadcast_event({"type": "dataset_complete", **result})

    asyncio.create_task(run_prep())

    return {"status": "started", "language": svc.language, "message": f"Dataset preparation started for {svc.language}"}


@router.post("/cancel")
async def cancel_preparation(request: Request):
    svc = _services(request).dataset_service
    if not svc.is_preparing:
        return {"error": "No preparation in progress"}
    svc.cancel()
    return {"status": "cancelling"}


@router.get("/progress")
async def preparation_progress(request: Request):
    return _services(request).dataset_service.progress


@router.post("/set-language")
async def set_dataset_language(request: Request, body: Dict[str, str] = Body(...)):
    language = body.get("language")
    if not language:
        return {"error": "language is required"}
    return _services(request).dataset_service.set_language(language)
