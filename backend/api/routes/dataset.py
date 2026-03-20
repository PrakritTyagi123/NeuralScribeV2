"""Dataset API routes — download EMNIST + prepare for training."""

from fastapi import APIRouter, Request, Body
from pydantic import BaseModel
from typing import Dict, Any, Optional
import asyncio

from ..ws import broadcast_event

router = APIRouter()

def _svc(r): return r.app.state.services

class PrepareRequest(BaseModel):
    overrides: Optional[Dict[str, Any]] = None

@router.get("/status")
async def status(r: Request):
    return _svc(r).dataset_service.get_status()

@router.get("/config")
async def config(r: Request):
    return _svc(r).dataset_service.get_config()

@router.get("/download-status")
async def download_status(r: Request):
    from ...ml.dataset_downloader import get_download_status
    return get_download_status()

@router.post("/download")
async def trigger_download(r: Request):
    from ...ml.dataset_downloader import download_emnist
    async def ws(d):
        d["type"] = "download_progress"
        await broadcast_event(d)
    async def run():
        loop = asyncio.get_event_loop()
        def sync_emit(d): asyncio.run_coroutine_threadsafe(ws(d), loop)
        result = await loop.run_in_executor(None, download_emnist, sync_emit)
        await broadcast_event({"type": "download_complete", **result})
    asyncio.create_task(run())
    return {"status": "started"}

@router.post("/prepare")
async def prepare(r: Request, body: PrepareRequest = PrepareRequest()):
    svc = _svc(r).dataset_service
    if svc.is_preparing: return {"error": "Already in progress"}
    if body.overrides: svc.update_config(body.overrides)
    async def ws(d): d["type"] = "dataset_progress"; await broadcast_event(d)
    async def run():
        result = await svc.prepare_dataset(ws_callback=ws)
        await broadcast_event({"type": "dataset_complete", **result})
    asyncio.create_task(run())
    return {"status": "started"}

@router.post("/cancel")
async def cancel(r: Request):
    svc = _svc(r).dataset_service
    if not svc.is_preparing: return {"error": "Not in progress"}
    svc.cancel()
    return {"status": "cancelling"}

@router.get("/progress")
async def progress(r: Request):
    return _svc(r).dataset_service.progress
