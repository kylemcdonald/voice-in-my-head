"""
Async web server using aiohttp.

Replaces Flask server with async-first design.
Handles:
- Static file serving
- WebSocket signaling for WebRTC
- Session management
"""

import asyncio
import json
import logging
import os
import secrets
import signal
import sys
from pathlib import Path
from typing import Dict, Optional

import aiofiles
from aiohttp import web, WSMsgType
import aiohttp_cors
from dotenv import load_dotenv

from webrtc_handler import WebRTCSession

load_dotenv()

logger = logging.getLogger(__name__)

# Active sessions
sessions: Dict[str, "VoiceSession"] = {}


async def index(request: web.Request) -> web.Response:
    """Serve the index page."""
    ip_addr = request.headers.get("X-Forwarded-For", request.remote)
    user_agent = request.headers.get("User-Agent", "")
    logger.info(f"Page loaded from {ip_addr} {user_agent}")

    return web.FileResponse(Path("static/index.html"))


async def run_page(request: web.Request) -> web.Response:
    """Serve the run page."""
    return web.FileResponse(Path("static/run.html"))


async def serve_static(request: web.Request) -> web.Response:
    """Serve static files."""
    filename = request.match_info.get("filename", "")
    filepath = Path("static") / filename

    if filepath.exists() and filepath.is_file():
        return web.FileResponse(filepath)

    raise web.HTTPNotFound()


async def serve_sound(request: web.Request) -> web.Response:
    """Serve sound files."""
    filename = request.match_info.get("filename", "")
    filepath = Path("sound") / filename

    if filepath.exists() and filepath.is_file():
        return web.FileResponse(filepath)

    raise web.HTTPNotFound()


async def info(request: web.Request) -> web.Response:
    """Return server info."""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        commit = result.stdout.strip()
    except Exception:
        commit = "unknown"

    location = os.getenv("LOCATION", "unknown")
    return web.Response(text=f"{commit} {location}")


async def spin_up_session(request: web.Request) -> web.Response:
    """
    Create a new voice session.

    Returns a session ID that the client uses to connect via WebSocket.
    """
    # Parse query parameters
    query = request.query

    # Generate unique session ID
    session_id = secrets.token_urlsafe(16)

    # Get configuration from query params
    config = {
        "language": query.get("l", "en"),
        "voice": query.get("v", "Mimi"),
        "script": query.get("s", "script.csv"),
    }

    logger.info(f"Creating session {session_id} with config: {config}")

    return web.json_response({
        "session_id": session_id,
        "config": config,
    })


async def websocket_handler(request: web.Request) -> web.WebSocketResponse:
    """
    Handle WebSocket connection for a session.

    This is the main signaling and communication channel.
    """
    session_id = request.match_info.get("session_id", "")

    if not session_id:
        raise web.HTTPBadRequest(text="Missing session_id")

    ws = web.WebSocketResponse()
    await ws.prepare(request)

    logger.info(f"WebSocket connected for session {session_id}")

    # Import here to avoid circular imports
    from voice_session import VoiceSession

    # Create session
    session = VoiceSession(session_id)
    sessions[session_id] = session

    # Set up message sending
    async def send_message(msg: str) -> None:
        if not ws.closed:
            await ws.send_str(msg)

    session.set_send_callback(send_message)

    try:
        # Start the WebRTC session
        await session.start_webrtc()

        # Handle incoming messages
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                await session.handle_signaling_message(msg.data)

            elif msg.type == WSMsgType.ERROR:
                logger.error(f"WebSocket error: {ws.exception()}")
                break

    except asyncio.CancelledError:
        logger.info(f"Session {session_id} cancelled")
    except Exception as e:
        logger.error(f"Session {session_id} error: {e}")
    finally:
        # Clean up
        logger.info(f"Closing session {session_id}")
        await session.close()
        sessions.pop(session_id, None)

    return ws


async def health(request: web.Request) -> web.Response:
    """Health check endpoint."""
    return web.json_response({
        "status": "ok",
        "sessions": len(sessions),
    })


def create_app() -> web.Application:
    """Create and configure the aiohttp application."""
    app = web.Application()

    # Add routes
    app.router.add_get("/", index)
    app.router.add_get("/run", run_page)
    app.router.add_get("/static/{filename:.*}", serve_static)
    app.router.add_get("/sound/{filename:.*}", serve_sound)
    app.router.add_get("/info", info)
    app.router.add_get("/health", health)
    app.router.add_post("/spin-up-session", spin_up_session)
    app.router.add_get("/ws/{session_id}", websocket_handler)

    # Style.css shortcut (for backward compatibility)
    app.router.add_get("/style.css", lambda r: web.FileResponse(Path("static/style.css")))

    # Set up CORS
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
        )
    })

    # Apply CORS to all routes
    for route in list(app.router.routes()):
        cors.add(route)

    return app


async def cleanup_sessions(app: web.Application) -> None:
    """Cleanup handler called on shutdown."""
    logger.info(f"Cleaning up {len(sessions)} sessions")
    for session_id, session in list(sessions.items()):
        try:
            await session.close()
        except Exception as e:
            logger.error(f"Error closing session {session_id}: {e}")
    sessions.clear()


def main():
    """Run the server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Reduce noise from some libraries
    logging.getLogger("aiortc").setLevel(logging.WARNING)
    logging.getLogger("aioice").setLevel(logging.WARNING)

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    app = create_app()
    app.on_cleanup.append(cleanup_sessions)

    logger.info(f"Starting server on {host}:{port}")
    web.run_app(app, host=host, port=port, print=None)


if __name__ == "__main__":
    main()
