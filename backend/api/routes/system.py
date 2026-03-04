"""
System API routes — GPU, system stats, dashboard overview, shutdown.
"""

import os
import signal
from fastapi import APIRouter, Request

router = APIRouter()


def _services(request: Request):
    return request.app.state.services


@router.get("/gpu")
async def gpu_status(request: Request):
    """Get current GPU utilization and memory."""
    return _services(request).system_service.get_gpu_status()


@router.get("/stats")
async def system_stats(request: Request):
    """Get CPU, RAM, disk stats."""
    return _services(request).system_service.get_system_stats()


@router.get("/dashboard")
async def dashboard(request: Request):
    """Aggregated dashboard overview."""
    s = _services(request)
    return s.system_service.get_dashboard_overview(
        dataset_service=s.dataset_service,
        training_service=s.training_service,
        model_service=s.model_service,
    )


@router.get("/torch")
async def torch_info(request: Request):
    """PyTorch and CUDA version info."""
    return _services(request).system_service.get_torch_info()


@router.get("/class-registry")
async def class_registry(request: Request):
    """Return the full class registry."""
    return _services(request).dataset_service.registry.to_dict()


@router.post("/shutdown")
async def shutdown(request: Request):
    """Gracefully shut down the server."""
    # Protect against accidental shutdowns in production.
    if os.getenv("APP_ENV", "dev").lower() == "prod":
        return {"error": "Shutdown is disabled in production."}
    ts = _services(request).training_service
    if ts.is_training:
        ts.stop()
    os.kill(os.getpid(), signal.SIGTERM)
    return {"status": "shutting down"}


@router.post("/clear-all")
async def clear_all(request: Request):
    """Delete all cached data, models, and exports."""
    import shutil
    from ...utils.config import PROJECT_ROOT

    s = _services(request)

    # In production, this endpoint is too destructive to expose.
    if os.getenv("APP_ENV", "dev").lower() == "prod":
        return {"error": "Clear-all is disabled in production."}

    # Stop any in-flight training job and reset training state.
    if s.training_service.is_training:
        s.training_service.stop()
    s.training_service.reset()

    # Unload model for both inference and model service.
    s.interface_service.clear_model()
    s.model_service.unload_model()

    deleted = []

    # Clear dataset cache
    cache_dir = PROJECT_ROOT / "data" / "cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        deleted.append("dataset cache")

    # Clear models (except index.json)
    models_dir = PROJECT_ROOT / "backend" / "models"
    if models_dir.exists():
        for f in models_dir.iterdir():
            if f.name == "index.json":
                f.write_text("[]")
            elif f.name == "exports":
                if f.is_dir():
                    shutil.rmtree(f)
                    f.mkdir(parents=True, exist_ok=True)
            else:
                f.unlink()
        deleted.append("models")

    return {"message": f"Cleared: {', '.join(deleted)}"}