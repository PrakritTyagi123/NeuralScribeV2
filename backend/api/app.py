"""NeuralScribe v2 — FastAPI application (English only)."""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from contextlib import asynccontextmanager

from ..utils.config import PROJECT_ROOT, ProjectConfig, ensure_all_language_dirs
from ..utils.logging import get_logger, setup_logging
from ..services.dataset_service import DatasetService
from ..services.training_service import TrainingService
from ..services.model_service import ModelService
from ..services.system_service import SystemService
from ..services.interface_service import InterfaceService
from .routes import dataset, training, models, inference, explainability, system
from .ws import ws_router

log = get_logger(__name__)

class AppState:
    def __init__(self):
        self.dataset_service = DatasetService()
        self.training_service = TrainingService()
        self.model_service = ModelService()
        self.system_service = SystemService()
        self.interface_service = InterfaceService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    log.info("NeuralScribe v2 starting up")
    ensure_all_language_dirs()
    state = AppState()
    app.state.services = state
    # Try to load best model for inference
    result = state.model_service.load_model("best_model")
    if "error" not in result:
        model = state.model_service.get_loaded_model()
        if model:
            state.interface_service.set_model(model)
            log.info("Loaded best model for inference")
    else:
        log.info("No model found — inference unavailable until trained")
    yield
    ProjectConfig().save()
    log.info("NeuralScribe v2 shutting down")

def create_app() -> FastAPI:
    app = FastAPI(title="NeuralScribe v2", version="2.0.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                       allow_methods=["*"], allow_headers=["*"])

    app.include_router(system.router, prefix="/api/system", tags=["System"])
    app.include_router(dataset.router, prefix="/api/dataset", tags=["Dataset"])
    app.include_router(training.router, prefix="/api/training", tags=["Training"])
    app.include_router(models.router, prefix="/api/models", tags=["Models"])
    app.include_router(inference.router, prefix="/api/inference", tags=["Inference"])
    app.include_router(explainability.router, prefix="/api/explain", tags=["Explainability"])
    app.include_router(ws_router)

    fd = PROJECT_ROOT / "frontend"
    if fd.exists():
        for name in ["components", "src", "views"]:
            d = fd / name
            if d.exists(): app.mount(f"/{name}", StaticFiles(directory=str(d)), name=name)

        @app.get("/style.css")
        async def css(): return FileResponse(str(fd / "style.css"), media_type="text/css")
        @app.get("/lnn.css")
        async def lnn(): return FileResponse(str(fd / "lnn.css"), media_type="text/css")
        @app.get("/dropdown.css")
        async def dropdown(): return FileResponse(str(fd / "dropdown.css"), media_type="text/css")
        @app.get("/{full_path:path}")
        async def spa(full_path: str):
            from fastapi.responses import JSONResponse
            if full_path.startswith("api/") or full_path.startswith("ws"):
                return JSONResponse({"error": "Not found"}, status_code=404)
            return FileResponse(str(fd / "index.html"), media_type="text/html")

    return app
