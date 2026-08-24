#!/usr/bin/env python3
"""Saathi web server — WebRTC signalling + the talk page.

    python server.py            # http://localhost:7860
    python server.py --verbose  # frame-level Pipecat tracing

Config is validated at boot, before anyone can connect. A missing key should
stop the server, not surface as silence in the middle of a demo.
"""

from __future__ import annotations

import argparse
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# The project isn't pip-installed (no site-packages footprint by design), so
# put src/ on the path. Keeps `python server.py` working from a clean checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import uvicorn  # noqa: E402
from fastapi import BackgroundTasks, FastAPI  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from loguru import logger  # noqa: E402
from pipecat.transports.smallwebrtc.request_handler import (  # noqa: E402
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)

from saathi import helplines as helplines_module  # noqa: E402
from saathi.bot import run_bot  # noqa: E402
from saathi.config import ConfigError, Settings  # noqa: E402

CLIENT_DIR = Path(__file__).resolve().parent / "client"


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await app.state.webrtc.close()


app = FastAPI(lifespan=lifespan)
app.state.webrtc = SmallWebRTCRequestHandler()


@app.post("/api/offer")
async def offer(request: SmallWebRTCRequest, background_tasks: BackgroundTasks):
    """Browser sends an SDP offer; we answer and start a bot for it."""

    async def on_connection(connection):
        background_tasks.add_task(run_bot, connection, app.state.settings)

    return await app.state.webrtc.handle_web_request(
        request=request,
        webrtc_connection_callback=on_connection,
    )


@app.patch("/api/offer")
async def ice_candidate(request: SmallWebRTCPatchRequest):
    """Trickle ICE candidates for the connection identified by pc_id."""
    await app.state.webrtc.handle_patch_request(request)
    return {"status": "success"}


@app.get("/api/helplines")
async def get_helplines():
    """The talk page renders these, so the numbers on screen and the numbers
    Saathi speaks come from the same data/helplines.json."""
    settings: Settings = app.state.settings
    directory = helplines_module.load(settings.helplines_path, settings.helpline_region)
    return {
        "region": settings.helpline_region,
        "helplines": [h.as_dict() for h in directory],
    }


@app.get("/")
async def index():
    return FileResponse(CLIENT_DIR / "index.html")


def main() -> int:
    parser = argparse.ArgumentParser(description="Saathi voice companion")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--verbose", "-v", action="count", default=0)
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="TRACE" if args.verbose else "INFO")

    try:
        settings = Settings.load()
    except ConfigError as exc:
        logger.error(f"\n\nConfiguration problem:\n\n{exc}\n")
        return 1

    app.state.settings = settings
    host = args.host or settings.host
    port = args.port or settings.port

    logger.info(f"Saathi ready — open http://{host}:{port}")
    logger.info(f"config: {settings.redacted()}")
    uvicorn.run(app, host=host, port=port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
