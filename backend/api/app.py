"""
NeuralScribe v2 — FastAPI application.
Creates the app, initializes services as shared state, registers all routes,
and serves the frontend static files.
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from pathlib import Path
from contextlib import asynccontextmanager

from ..utils.config import PROJECT_ROOT
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
    """Shared application state — service singletons."""

    def __init__(self):
        self.dataset_service = DatasetService()
        self.training_service = TrainingService()
        self.model_service = ModelService()
        self.system_service = SystemService()
        self.interface_service = InterfaceService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    setup_logging()
    log.info("NeuralScribe v2 starting up")

    # Initialize shared state
    state = AppState()
    app.state.services = state

    # Try to load best model for inference on startup
    result = state.model_service.load_model("best_model")
    if "error" not in result:
        model = state.model_service.get_loaded_model()
        if model:
            state.interface_service.set_model(model)
            log.info("Loaded best model for inference on startup")
    else:
        log.info("No best model found — inference unavailable until a model is trained/loaded")

    yield

    log.info("NeuralScribe v2 shutting down")


def create_app() -> FastAPI:
    """Factory function to create the FastAPI application."""
    app = FastAPI(
        title="NeuralScribe v2",
        description="Handwriting recognition for digits, alphabets, Greek letters, and scientific symbols",
        version="2.0.0",
        lifespan=lifespan,
    )

    # CORS — environment-aware
    app_env = os.getenv("APP_ENV", "dev").lower()
    if app_env == "prod":
        # In production, lock CORS down to an explicit origin to avoid
        # exposing authenticated APIs to arbitrary websites.
        frontend_origin = os.getenv("FRONTEND_ORIGIN", "")
        allow_origins = [frontend_origin] if frontend_origin else []
        allow_credentials = bool(frontend_origin)
    else:
        # During local development we keep CORS permissive for convenience.
        allow_origins = ["*"]
        allow_credentials = True

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Register API routes ──
    app.include_router(system.router, prefix="/api/system", tags=["System"])
    app.include_router(dataset.router, prefix="/api/dataset", tags=["Dataset"])
    app.include_router(training.router, prefix="/api/training", tags=["Training"])
    app.include_router(models.router, prefix="/api/models", tags=["Models"])
    app.include_router(inference.router, prefix="/api/inference", tags=["Inference"])
    app.include_router(explainability.router, prefix="/api/explain", tags=["Explainability"])

    # ── WebSocket ──
    app.include_router(ws_router)

    # ── Serve frontend ──
    frontend_dir = PROJECT_ROOT / "frontend"
    if frontend_dir.exists():
        # Static assets
        assets_dir = frontend_dir / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

        # CSS and JS files
        app.mount("/components", StaticFiles(directory=str(frontend_dir / "components")), name="components")
        app.mount("/src", StaticFiles(directory=str(frontend_dir / "src")), name="src")
        app.mount("/views", StaticFiles(directory=str(frontend_dir / "views")), name="views")

        # Style
        @app.get("/style.css")
        async def serve_css():
            return FileResponse(str(frontend_dir / "style.css"), media_type="text/css")

        @app.get("/lnn.css")
        async def serve_lnn_css():
            return FileResponse(str(frontend_dir / "lnn.css"), media_type="text/css")

        # SPA fallback — serve index.html for all unmatched routes
        @app.get("/{full_path:path}")
        async def serve_spa(full_path: str):
            from fastapi.responses import JSONResponse
            # Don't intercept API or WebSocket routes
            if full_path.startswith("api/") or full_path.startswith("ws"):
                return JSONResponse({"error": "Not found"}, status_code=404)
            index = frontend_dir / "index.html"
            if index.exists():
                return FileResponse(str(index), media_type="text/html")
            return JSONResponse({"error": "Frontend not found"}, status_code=404)

    return app