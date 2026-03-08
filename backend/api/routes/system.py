"""
System API routes — GPU, stats, dashboard, language management, project config.
"""

import os
import signal
from fastapi import APIRouter, Request, Body
from typing import Dict, Any

router = APIRouter()


def _services(request: Request):
    return request.app.state.services


@router.get("/gpu")
async def gpu_status(request: Request):
    return _services(request).system_service.get_gpu_status()


@router.get("/stats")
async def system_stats(request: Request):
    return _services(request).system_service.get_system_stats()


@router.get("/dashboard")
async def dashboard(request: Request):
    s = _services(request)
    return s.system_service.get_dashboard_overview(
        dataset_service=s.dataset_service,
        training_service=s.training_service,
        model_service=s.model_service,
    )


@router.get("/torch")
async def torch_info(request: Request):
    return _services(request).system_service.get_torch_info()


@router.get("/class-registry")
async def class_registry(request: Request):
    return _services(request).dataset_service.registry.to_dict()


# ── Language management ──

@router.get("/languages")
async def list_languages(request: Request):
    """Return all available languages with their status."""
    from ..utils.config import ProjectConfig
    pc = ProjectConfig()
    return {
        "languages": pc.get_available_languages(),
        "selected": pc.selected_language,
    }


@router.post("/language")
async def set_language(request: Request, body: Dict[str, str] = Body(...)):
    """Switch all services to a new language."""
    language = body.get("language")
    if not language:
        return {"error": "language is required"}

    s = _services(request)
    results = {}
    results["dataset"] = s.dataset_service.set_language(language)
    results["training"] = s.training_service.set_language(language)
    results["model"] = s.model_service.set_language(language)
    results["interface"] = s.interface_service.set_language(language)

    # Update project config
    from ..utils.config import ProjectConfig
    pc = ProjectConfig()
    pc.selected_language = language

    # Try to load best model for new language
    load_result = s.model_service.load_model("best_model")
    if "error" not in load_result:
        model = s.model_service.get_loaded_model()
        if model:
            s.interface_service.set_model(model)
            results["model_loaded"] = True
    else:
        results["model_loaded"] = False

    return {"status": "switched", "language": language, "results": results}


# ── Project config ──

@router.get("/project-config")
async def get_project_config(request: Request):
    """Return the full project config."""
    from ..utils.config import ProjectConfig
    return ProjectConfig().to_dict()


@router.post("/project-config")
async def update_project_config(request: Request, body: Dict[str, Any] = Body(...)):
    """Update project config fields."""
    from ..utils.config import ProjectConfig
    pc = ProjectConfig()
    for key, value in body.items():
        if key == "ui_state" and isinstance(value, dict):
            for k, v in value.items():
                pc.set_ui_state(k, v)
        elif key == "last_selected_language":
            pc.selected_language = value
        elif key == "last_selected_model":
            pc.selected_model = value
    return {"status": "updated"}


# ── Shutdown / Clear ──

@router.post("/shutdown")
async def shutdown(request: Request):
    if os.getenv("APP_ENV", "dev").lower() == "prod":
        return {"error": "Shutdown is disabled in production."}
    # Save project config before shutdown
    from ..utils.config import ProjectConfig
    ProjectConfig().save()
    ts = _services(request).training_service
    if ts.is_training:
        ts.stop()
        import asyncio
        for _ in range(20):
            await asyncio.sleep(0.5)
            if not ts.is_training:
                break
    os.kill(os.getpid(), signal.SIGTERM)
    return {"status": "shutting down"}


@router.post("/clear-all")
async def clear_all(request: Request):
    """Delete all cached data and models for the current language."""
    import shutil
    from ..utils.config import PROJECT_ROOT

    s = _services(request)

    if os.getenv("APP_ENV", "dev").lower() == "prod":
        return {"error": "Clear-all is disabled in production."}

    if s.training_service.is_training:
        s.training_service.stop()
    s.training_service.reset()
    s.interface_service.clear_model()
    s.model_service.unload_model()

    language = s.dataset_service.language
    deleted = []

    # Clear dataset cache for current language
    from ..utils.config import get_language_paths
    paths = get_language_paths(language)
    cache_dir = paths.dataset_dir / "cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        deleted.append(f"{language} dataset cache")

    # Clear models for current language
    if paths.models_dir.exists():
        for f in paths.models_dir.iterdir():
            if f.name == "index.json":
                f.write_text("[]")
            elif f.name == "exports":
                if f.is_dir():
                    shutil.rmtree(f)
                    f.mkdir(parents=True, exist_ok=True)
            else:
                f.unlink()
        deleted.append(f"{language} models")

    # Update project config
    from ..utils.config import ProjectConfig
    pc = ProjectConfig()
    pc.set_language_config(language, "dataset_prepared", False)
    pc.set_language_config(language, "last_model", None)

    return {"message": f"Cleared: {', '.join(deleted)}", "language": language}
