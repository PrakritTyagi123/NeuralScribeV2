"""
Training API routes — start, stop, pause, resume, status, evaluate.
"""

from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import Dict, Any, Optional
import asyncio

from ..ws import broadcast_event

router = APIRouter()


def _services(request: Request):
    return request.app.state.services


class TrainRequest(BaseModel):
    """Training start request with optional overrides."""
    overrides: Optional[Dict[str, Any]] = None
    resume_from: Optional[str] = None


class EvalRequest(BaseModel):
    """Evaluation request."""
    model_path: Optional[str] = None


@router.get("/status")
async def training_status(request: Request):
    """Get current training status."""
    return _services(request).training_service.get_status()


@router.get("/config")
async def training_config(request: Request):
    """Get current training config."""
    return _services(request).training_service.get_config()


@router.post("/config")
async def update_training_config(request: Request, body: Dict[str, Any]):
    """Update training config (saved to YAML)."""
    _services(request).training_service.update_config(body)
    return {"status": "updated"}


@router.post("/start")
async def start_training(request: Request, body: TrainRequest = TrainRequest()):
    """
    Start training. Requires dataset to be prepared first.
    Streams progress via WebSocket.
    """
    s = _services(request)
    ts = s.training_service
    ds = s.dataset_service

    if ts.is_training:
        return {"error": "Training already in progress"}

    # Apply overrides
    if body.overrides:
        ts.update_config(body.overrides)

    # Get config for dataloader params
    cfg = ts.get_config()
    batch_size = cfg.get("training", {}).get("batch_size", 256)
    num_workers = cfg.get("training", {}).get("num_workers", 4)
    pin_memory = cfg.get("training", {}).get("pin_memory", True)

    # Create dataloaders
    train_loader = ds.get_dataloader("train", batch_size=batch_size,
                                      num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
    val_loader = ds.get_dataloader("val", batch_size=batch_size,
                                    num_workers=num_workers, pin_memory=pin_memory, shuffle=False)

    if train_loader is None or val_loader is None:
        return {"error": "Dataset not prepared. Run data preparation first."}

    async def ws_progress(data: Dict[str, Any]):
        await broadcast_event(data)

    # Resolve resume path
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

    return {"status": "started", "message": "Training started"}


@router.post("/stop")
async def stop_training(request: Request):
    """Stop training after current batch."""
    ts = _services(request).training_service
    if not ts.is_training:
        return {"error": "No training in progress"}
    ts.stop()
    return {"status": "stopping"}


@router.post("/pause")
async def pause_training(request: Request):
    """Pause training."""
    ts = _services(request).training_service
    if not ts.is_training:
        return {"error": "No training in progress"}
    ts.pause()
    return {"status": "paused"}


@router.post("/resume")
async def resume_training(request: Request):
    """Resume paused training."""
    ts = _services(request).training_service
    if not ts.is_training:
        return {"error": "No training in progress"}
    ts.resume()
    return {"status": "resumed"}


@router.get("/history")
async def training_history(request: Request):
    """Get full training history."""
    return {"history": _services(request).training_service.history}


@router.post("/evaluate")
async def evaluate_model(request: Request, body: EvalRequest = EvalRequest()):
    """
    Run full evaluation on test set.
    Returns per-class metrics, confusion matrix, top confusions.
    """
    s = _services(request)
    ts = s.training_service
    ds = s.dataset_service

    cfg = ts.get_config()
    batch_size = cfg.get("training", {}).get("batch_size", 256)

    test_loader = ds.get_dataloader("test", batch_size=batch_size,
                                     num_workers=2, pin_memory=True, shuffle=False)
    if test_loader is None:
        return {"error": "Dataset not prepared"}

    result = ts.evaluate(test_loader, model_path=body.model_path)
    return result