"""
Dataset API routes — data preparation, status, config.
"""

from fastapi import APIRouter, Request, BackgroundTasks, Body
from pydantic import BaseModel
from typing import Dict, Any, Optional

from ..ws import broadcast_event

router = APIRouter()


def _services(request: Request):
    return request.app.state.services


class PrepareRequest(BaseModel):
    """Optional overrides for data preparation."""
    overrides: Optional[Dict[str, Any]] = None


@router.get("/status")
async def dataset_status(request: Request):
    """Get dataset preparation status and cache info."""
    return _services(request).dataset_service.get_status()


@router.get("/config")
async def dataset_config(request: Request):
    """Get current data preparation config."""
    return _services(request).dataset_service.get_config()


@router.post("/config")
async def update_dataset_config(request: Request, body: Dict[str, Any] = Body(...)):
    """Update data preparation config (saved to YAML)."""
    from ...utils.logging import get_logger
    log = get_logger(__name__)
    log.info(f"Updating dataset config: {body}")
    _services(request).dataset_service.update_config(body)
    return {"status": "updated", "applied": body}


@router.post("/prepare")
async def prepare_dataset(request: Request, body: PrepareRequest = PrepareRequest()):
    """
    Trigger dataset preparation (EMNIST + synthetic generation + caching).
    Runs as background task, streams progress via WebSocket.
    """
    svc = _services(request).dataset_service

    if svc.is_preparing:
        return {"error": "Dataset preparation already in progress"}

    # Apply overrides if any
    if body.overrides:
        svc.update_config(body.overrides)

    async def ws_progress(data: Dict[str, Any]):
        data["type"] = "dataset_progress"
        await broadcast_event(data)

    # Run in background
    import asyncio

    async def run_prep():
        result = await svc.prepare_dataset(ws_callback=ws_progress)
        await broadcast_event({"type": "dataset_complete", **result})

    asyncio.create_task(run_prep())

    return {"status": "started", "message": "Dataset preparation started"}


@router.post("/cancel")
async def cancel_preparation(request: Request):
    """Cancel an in-progress dataset preparation."""
    svc = _services(request).dataset_service
    if not svc.is_preparing:
        return {"error": "No preparation in progress"}
    svc.cancel()
    return {"status": "cancelling"}


@router.get("/progress")
async def preparation_progress(request: Request):
    """Get current preparation progress (poll fallback if WS unavailable)."""
    return _services(request).dataset_service.progress