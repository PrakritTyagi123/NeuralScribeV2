"""
NeuralScribe v2 — WebSocket manager.
Handles client connections and broadcasts events (training progress,
data prep progress, GPU stats) to all connected clients.
"""

import json
import asyncio
from typing import Set, Dict, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..utils.logging import get_logger
from ..utils.helpers import safe_json_dump

log = get_logger(__name__)

ws_router = APIRouter()


class ConnectionManager:
    """Manages active WebSocket connections."""

    def __init__(self):
        self._connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)
        log.info(f"WS client connected ({len(self._connections)} total)")

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            self._connections.discard(websocket)
        log.info(f"WS client disconnected ({len(self._connections)} total)")

    async def broadcast(self, data: Dict[str, Any]):
        """Send a message to all connected clients."""
        if not self._connections:
            return

        message = safe_json_dump(data)
        dead = set()

        async with self._lock:
            for ws in self._connections:
                try:
                    await ws.send_text(message)
                except Exception:
                    dead.add(ws)

            self._connections -= dead

    async def send_to(self, websocket: WebSocket, data: Dict[str, Any]):
        """Send a message to a specific client."""
        try:
            await websocket.send_text(safe_json_dump(data))
        except Exception:
            pass

    @property
    def client_count(self) -> int:
        return len(self._connections)


# Global manager instance
manager = ConnectionManager()


async def broadcast_event(data: Dict[str, Any]):
    """Public helper for services to broadcast events."""
    await manager.broadcast(data)


@ws_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Main WebSocket endpoint.
    Clients connect here to receive live updates.
    Also accepts commands from clients (e.g., request GPU stats).
    """
    await manager.connect(websocket)

    try:
        # Send initial connection confirmation
        await manager.send_to(websocket, {
            "type": "connected",
            "message": "Connected to NeuralScribe v2",
        })

        while True:
            # Listen for client messages
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                data = json.loads(raw)
                await _handle_client_message(websocket, data)
            except asyncio.TimeoutError:
                # Send keepalive ping
                try:
                    await manager.send_to(websocket, {"type": "ping"})
                except Exception:
                    break
            except json.JSONDecodeError:
                await manager.send_to(websocket, {
                    "type": "error",
                    "message": "Invalid JSON",
                })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.debug(f"WS error: {e}")
    finally:
        await manager.disconnect(websocket)


async def _handle_client_message(websocket: WebSocket, data: Dict[str, Any]):
    """Handle incoming client WebSocket messages."""
    msg_type = data.get("type", "")

    if msg_type == "pong":
        # Keepalive response, ignore
        return

    elif msg_type == "request_gpu_stats":
        # Client wants GPU stats
        from ..services.system_service import SystemService
        sys_service = SystemService()
        gpu_info = sys_service.get_gpu_status()
        await manager.send_to(websocket, {
            "type": "gpu_stats",
            **gpu_info,
        })

    elif msg_type == "request_training_status":
        # Client wants current training state
        # The training service state is stored in app.state, but we can
        # send a status request event that the frontend can use
        await manager.send_to(websocket, {
            "type": "status_request_received",
            "message": "Use REST API for full status",
        })

    else:
        await manager.send_to(websocket, {
            "type": "unknown_command",
            "received": msg_type,
        })