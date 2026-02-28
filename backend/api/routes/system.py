"""
System API routes — GPU, system stats, dashboard overview.
"""

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