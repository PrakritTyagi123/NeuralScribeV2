"""
NeuralScribe v2 — Backend entry point.
Creates and runs the FastAPI application with uvicorn.

Usage:
    python run_backend.py
    python run_backend.py --host 0.0.0.0 --port 8000 --reload
"""

import argparse
import uvicorn

from backend.api.app import create_app
from backend.utils.logging import setup_logging, get_logger

setup_logging()
log = get_logger("run_backend")

app = create_app()


def main():
    parser = argparse.ArgumentParser(description="NeuralScribe v2 — Run Backend")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers (use 1 for GPU)")
    parser.add_argument("--auto-shutdown", type=int, default=10, help="Auto-shutdown after N seconds with no clients (0=disabled)")
    args = parser.parse_args()

    if args.auto_shutdown > 0:
        from backend.api.ws import manager
        manager.enable_auto_shutdown(delay=args.auto_shutdown)

    log.info(f"Starting NeuralScribe v2 on http://{args.host}:{args.port}")
    log.info(f"  API docs: http://{args.host}:{args.port}/docs")
    log.info(f"  WebSocket: ws://{args.host}:{args.port}/ws")

    if args.reload:
        # String import needed for reload mode
        uvicorn.run(
            "run_backend:app",
            host=args.host,
            port=args.port,
            reload=True,
            workers=args.workers,
            log_level="warning",
            access_log=False,
            ws_ping_interval=30,
            ws_ping_timeout=10,
        )
    else:
        # Direct app object — os._exit works properly
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            workers=args.workers,
            log_level="warning",
            access_log=False,
            ws_ping_interval=30,
            ws_ping_timeout=10,
        )


if __name__ == "__main__":
    main()