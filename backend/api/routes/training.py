"""
Training API routes — language-aware start, stop, pause, resume, status, evaluate.
"""

from fastapi import APIRouter, Request, Query, Body
from pydantic import BaseModel
from typing import Dict, Any, Optional
import asyncio

from ..ws import broadcast_event

router = APIRouter()


def _services(request: Request):
    return request.app.state.services


class TrainRequest(BaseModel):
    language: Optional[str] = None
    overrides: Optional[Dict[str, Any]] = None
    resume_from: Optional[str] = None


class EvalRequest(BaseModel):
    language: Optional[str] = None
    model_path: Optional[str] = None


@router.get("/status")
async def training_status(request: Request, language: Optional[str] = Query(None)):
    ts = _services(request).training_service
    if language:
        ts.set_language(language)
    return ts.get_status()


@router.get("/config")
async def training_config(request: Request, language: Optional[str] = Query(None)):
    ts = _services(request).training_service
    if language:
        ts.set_language(language)
    return ts.get_config()


@router.post("/config")
async def update_training_config(request: Request, body: Dict[str, Any] = Body(...)):
    ts = _services(request).training_service
    language = body.pop("language", None)
    if language:
        ts.set_language(language)
    ts.update_config(body)
    return {"status": "updated", "language": ts.language}


@router.post("/start")
async def start_training(request: Request, body: TrainRequest = TrainRequest()):
    s = _services(request)
    ts = s.training_service
    ds = s.dataset_service

    if body.language:
        ts.set_language(body.language)
        ds.set_language(body.language)
        s.model_service.set_language(body.language)

    if ts.is_training:
        return {"error": "Training already in progress"}

    if body.overrides:
        ts.update_config(body.overrides)

    cfg = ts.get_config()
    batch_size = cfg.get("training", {}).get("batch_size", 256)
    num_workers = cfg.get("training", {}).get("num_workers", 4)
    pin_memory = cfg.get("training", {}).get("pin_memory", True)

    train_loader = ds.get_dataloader("train", batch_size=batch_size,
                                      num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
    val_loader = ds.get_dataloader("val", batch_size=batch_size,
                                    num_workers=num_workers, pin_memory=pin_memory, shuffle=False)

    if train_loader is None or val_loader is None:
        return {"error": f"Dataset not prepared for {ts.language}. Run data preparation first."}

    async def ws_progress(data: Dict[str, Any]):
        await broadcast_event(data)

    resume_path = None
    if body.resume_from:
        model_path = s.model_service.get_model_path(body.resume_from)
        if model_path:
            resume_path = str(model_path)
        else:
            return {"error": f"Checkpoint not found: {body.resume_from}"}

    async def run_training():
        result = await ts.start_training(
            train_loader, val_loader,
            ws_callback=ws_progress,
            resume_path=resume_path,
        )
        # After training, reload best model for inference
        load_result = s.model_service.load_model("best_model")
        if "error" not in load_result:
            model = s.model_service.get_loaded_model()
            if model:
                s.interface_service.set_model(model)
        await broadcast_event({"type": "training_complete", **result})

    asyncio.create_task(run_training())

    return {"status": "started", "language": ts.language, "message": f"Training started for {ts.language}"}


@router.post("/stop")
async def stop_training(request: Request):
    ts = _services(request).training_service
    if not ts.is_training:
        return {"error": "No training in progress"}
    ts.stop()
    return {"status": "stopping"}


@router.post("/pause")
async def pause_training(request: Request):
    ts = _services(request).training_service
    if not ts.is_training:
        return {"error": "No training in progress"}
    ts.pause()
    return {"status": "paused"}


@router.post("/resume")
async def resume_training(request: Request):
    ts = _services(request).training_service
    if not ts.is_training:
        return {"error": "No training in progress"}
    ts.resume()
    return {"status": "resumed"}


@router.get("/history")
async def training_history(request: Request, language: Optional[str] = Query(None)):
    ts = _services(request).training_service
    if language:
        ts.set_language(language)
    return {"language": ts.language, "history": ts.history}


@router.post("/evaluate")
async def evaluate_model(request: Request, body: EvalRequest = EvalRequest()):
    s = _services(request)
    ts = s.training_service
    ds = s.dataset_service

    if body.language:
        ts.set_language(body.language)
        ds.set_language(body.language)

    cfg = ts.get_config()
    batch_size = cfg.get("training", {}).get("batch_size", 256)

    test_loader = ds.get_dataloader("test", batch_size=batch_size,
                                     num_workers=2, pin_memory=True, shuffle=False)
    if test_loader is None:
        return {"error": f"Dataset not prepared for {ts.language}"}

    return ts.evaluate(test_loader, model_path=body.model_path)


@router.post("/set-language")
async def set_training_language(request: Request, body: Dict[str, str] = Body(...)):
    language = body.get("language")
    if not language:
        return {"error": "language is required"}
    return _services(request).training_service.set_language(language)
