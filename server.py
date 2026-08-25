#!/usr/bin/env python3
"""Saathi web server — WebRTC signalling + the talk page.

    python server.py            # http://localhost:7860
    python server.py --verbose  # frame-level Pipecat tracing

Config is validated at boot, before anyone can connect. A missing key should
stop the server, not surface as silence in the middle of a demo.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# The project isn't pip-installed (no site-packages footprint by design), so
# put src/ on the path. Keeps `python server.py` working from a clean checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import hmac  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

import uvicorn  # noqa: E402
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from loguru import logger  # noqa: E402
from pipecat.transports.smallwebrtc.connection import IceServer  # noqa: E402
from pipecat.transports.smallwebrtc.request_handler import (  # noqa: E402
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)

from saathi import helplines as helplines_module  # noqa: E402
from saathi import languages as languages_module  # noqa: E402
from saathi.bot import run_bot  # noqa: E402
from saathi.config import ConfigError, Settings  # noqa: E402

CLIENT_DIR = Path(__file__).resolve().parent / "client"


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await app.state.webrtc.close()


app = FastAPI(lifespan=lifespan)
# On a cloud VM the process sits behind NAT: its host candidates are private
# addresses the browser can't reach, so the server side needs STUN to offer a
# public one. Localhost needs nothing. Comma-separated URLs, e.g.
#   SAATHI_ICE_SERVERS=stun:stun.l.google.com:19302
import os as _os  # noqa: E402

_ice = [IceServer(urls=u) for u in
        (_os.getenv("SAATHI_ICE_SERVERS") or "").replace(" ", "").split(",") if u]
app.state.webrtc = SmallWebRTCRequestHandler(ice_servers=_ice or None)


def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", *args], cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:  # noqa: BLE001 — version display must never block boot
        return ""


# Which release is actually serving. Deploys check out tags, so a healthy
# production box reports e.g. "v0.2.1"; a working tree reports "dev".
app.state.version = _git("describe", "--tags", "--always", "--dirty") or "dev"
_remote_match = re.search(r"github\.com[:/]+([^/]+/[^/.]+)",
                          _git("config", "--get", "remote.origin.url"))
app.state.repo = _remote_match.group(1) if _remote_match else None


_DASH_FAILS: dict[str, list] = {}  # ip -> [wrong_attempts, locked_until_ts]
_DASH_MAX_TRIES = 5
_DASH_LOCK_SECS = 300  # the UI sentences you to 12,314,281 years; we serve five minutes


def _dashboard_guard(request: Request) -> None:
    """Admin gate for the dashboard's data endpoints.

    SAATHI_DASHBOARD_TOKEN unset = open (local development). Set it in the
    deployment's .env and the dashboard demands it as a password. Five wrong
    passwords lock that IP out for five minutes — real brute-force pacing,
    however the front-end chooses to dramatise it. The talk page's own
    endpoints stay open — callers are anonymous.
    """
    expected = (_os.getenv("SAATHI_DASHBOARD_TOKEN") or "").strip()
    if not expected:
        return
    # Behind Caddy every connection is 127.0.0.1; Caddy appends the real
    # client to X-Forwarded-For, so the LAST entry is the trustworthy one.
    forwarded = request.headers.get("x-forwarded-for", "")
    ip = (forwarded.split(",")[-1].strip() if forwarded
          else (request.client.host if request.client else "?"))
    fails, locked_until = _DASH_FAILS.get(ip, [0, 0.0])
    now = time.time()
    if now < locked_until:
        raise HTTPException(status_code=429,
                            detail={"locked_for": int(locked_until - now)})
    supplied = request.headers.get("authorization", "")
    if supplied.lower().startswith("bearer "):
        supplied = supplied[7:]
    if supplied and hmac.compare_digest(supplied.encode(), expected.encode()):
        _DASH_FAILS.pop(ip, None)
        return
    if supplied:  # an actual wrong password burns a try; a bare page load doesn't
        fails += 1
        if fails >= _DASH_MAX_TRIES:
            _DASH_FAILS[ip] = [0, now + _DASH_LOCK_SECS]
            raise HTTPException(status_code=429,
                                detail={"locked_for": _DASH_LOCK_SECS})
        _DASH_FAILS[ip] = [fails, 0.0]
    raise HTTPException(status_code=401,
                        detail={"attempts_left": _DASH_MAX_TRIES - fails})


@app.post("/api/offer")
async def offer(request: SmallWebRTCRequest, background_tasks: BackgroundTasks):
    """Browser sends an SDP offer; we answer and start a bot for it.

    The talk page tucks the caller's language pick into request_data; an
    unknown or missing id falls back to the default mode — a bad picker value
    must never cost someone the call.
    """
    data = request.request_data if isinstance(request.request_data, dict) else {}
    language = languages_module.get(data.get("language"))
    device_id = str(data.get("device_id") or "")[:64] or None
    remember = bool(data.get("remember"))

    async def on_connection(connection):
        background_tasks.add_task(
            run_bot,
            connection,
            app.state.settings,
            language,
            app.state.retriever,
            app.state.understander,
            app.state.quotes,
            app.state.store,
            device_id,
            remember,
            app.state.profile_updater,
        )

    return await app.state.webrtc.handle_web_request(
        request=request,
        webrtc_connection_callback=on_connection,
    )


@app.get("/api/languages")
async def get_languages():
    """Language modes for the talk page picker."""
    return {"languages": languages_module.as_api_list()}


def _session_payload(store, row):
    """Turns + transcript for one session — shared by the talk page's
    latest-session poll and the dashboard's per-session drill-down."""
    sid, started, ended, lang = row
    turns = [
        {
            "turn": t, "user_text": u, "mode": m, "emotion": e, "intensity": i,
            "confidence": c, "query": q, "raw": u,
            "passages": json.loads(s) if s else [],
            "plan_notes": json.loads(pl) if pl else [],
            "understand_ms": um, "retrieve_ms": rm, "speculative_hit": bool(sp),
        }
        for (t, u, m, e, i, c, q, s, pl, um, rm, sp) in store.query(
            "SELECT turn, user_text, mode, emotion, intensity, confidence, query,"
            " sources, plan, understand_ms, retrieve_ms, speculative_hit"
            " FROM turns WHERE session_id = ? ORDER BY turn", (sid,))
    ]
    utterances = [
        {"ts": ts, "role": role, "text": text}
        for (ts, role, text) in store.query(
            "SELECT ts, role, text FROM utterances WHERE session_id = ? ORDER BY ts", (sid,))
    ]
    return {
        "session": {"id": sid, "started_at": started, "ended_at": ended, "language": lang},
        "turns": turns,
        "utterances": utterances,
    }


@app.get("/api/sessions/latest")
async def latest_session():
    """The most recent session's turns and transcript — feeds the transcript
    ribbon and the insight drawer, polled by the client while connected.
    Single-box demo semantics: 'latest' is good enough without session auth."""
    store = app.state.store
    rows = store.query("SELECT id, started_at, ended_at, language FROM sessions ORDER BY started_at DESC LIMIT 1")
    if not rows:
        return {"session": None, "turns": [], "utterances": []}
    return _session_payload(store, rows[0])


@app.get("/api/sessions/{session_id}")
async def session_detail(session_id: str, _: None = Depends(_dashboard_guard)):
    """One session's full record — the dashboard's drill-down. Admin-gated."""
    store = app.state.store
    rows = store.query(
        "SELECT id, started_at, ended_at, language FROM sessions WHERE id = ?",
        (session_id,))
    if not rows:
        return {"session": None, "turns": [], "utterances": []}
    return _session_payload(store, rows[0])


@app.get("/api/metrics")
async def metrics(range: str = "all", _: None = Depends(_dashboard_guard)):
    """Aggregates for the dashboard (admin-gated). Small data — aggregation in
    Python keeps the SQL portable between SQLite and Postgres. `range`
    (today | week | all) windows everything except the eval block, which
    always reflects the latest harness run — a deploy gate, not a time
    series."""
    import statistics
    import time as _t

    store = app.state.store

    def pct(values, q):
        if not values:
            return None
        values = sorted(values)
        idx = (len(values) - 1) * q
        lo, hi = int(idx), min(int(idx) + 1, len(values) - 1)
        return round(values[lo] + (values[hi] - values[lo]) * (idx - lo), 3)

    cutoff = {"today": _t.time() - 86400, "week": _t.time() - 7 * 86400}.get(range, 0)
    sessions = [s for s in store.query(
        "SELECT id, started_at, ended_at, language FROM sessions ORDER BY started_at DESC")
        if (s[1] or 0) >= cutoff]
    in_range = {s[0] for s in sessions}

    turn_rows = [r for r in store.query(
        "SELECT session_id, mode, emotion, sources, speculative_hit, degraded,"
        " understand_ms, retrieve_ms, query FROM turns") if r[0] in in_range]
    lat_rows = [r for r in store.query(
        "SELECT session_id, voice_to_voice_secs, stages FROM latency") if r[0] in in_range]
    v2v = [r[1] for r in lat_rows if r[1] is not None]

    modes, source_counts, emotion_counts = {}, {}, {}
    spec_hits = spec_total = guide_total = guide_grounded = degraded = 0
    unanswered = {}
    per_session_turns: dict[str, int] = {}
    session_modes: dict[str, set] = {}
    lane_hits = {"semantic": 0, "lexical": 0, "emotion": 0}
    grounded_turns = 0
    for sid, mode, emotion, sources, spec, deg, um, rm, query in turn_rows:
        per_session_turns[sid] = per_session_turns.get(sid, 0) + 1
        session_modes.setdefault(sid, set()).add(mode)
        modes[mode] = modes.get(mode, 0) + 1
        if emotion:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        spec_total += 1
        spec_hits += 1 if spec else 0
        degraded += 1 if deg else 0
        passages = json.loads(sources) if sources else []
        if mode == "guide":
            guide_total += 1
            if passages:
                guide_grounded += 1
            elif query:
                unanswered[query] = unanswered.get(query, 0) + 1
        if passages:
            grounded_turns += 1
            turn_lanes = set()
            for passage in passages:
                turn_lanes.update((passage.get("lanes") or {}).keys())
            for lane in turn_lanes:
                if lane in lane_hits:
                    lane_hits[lane] += 1
        for passage in passages:
            src = (passage.get("source") or "?").split(" (")[0]
            source_counts[src] = source_counts.get(src, 0) + 1

    stages: dict[str, list[float]] = {}
    per_session_v2v: dict[str, list[float]] = {}
    for sid, secs, raw in lat_rows:
        if secs is not None:
            per_session_v2v.setdefault(sid, []).append(secs)
        for s in json.loads(raw) if raw else []:
            name = str(s.get("processor", "?")).split("#")[0].replace("Service", "")
            if s.get("duration_secs"):
                stages.setdefault(name, []).append(float(s["duration_secs"]))

    safety_rows = [r for r in store.query(
        "SELECT session_id, ts, kind, matched, user_text FROM safety_events ORDER BY ts DESC")
        if r[0] in in_range]
    session_safety: dict[str, set] = {}
    for sid, ts, kind, matched, text in safety_rows:
        session_safety.setdefault(sid, set()).add(kind)
    safety = [
        {"ts": ts, "kind": kind, "matched": matched, "text": (text or "")[:80]}
        for (sid, ts, kind, matched, text) in safety_rows[:20]
    ]

    # First user utterance per session — the table's "opening turn" column.
    openings: dict[str, str] = {}
    for sid, text in store.query(
            "SELECT session_id, text FROM utterances WHERE role = ? ORDER BY ts", ("user",)):
        openings.setdefault(sid, text)

    def outcome(sid: str) -> str:
        kinds = session_safety.get(sid, set())
        if "crisis" in kinds:
            return "handoff"
        if "injection" in kinds:
            return "deflected"
        if "scope" in kinds:
            return "declined"
        had = session_modes.get(sid, set())
        if "guide" in had:
            return "guided"
        if "hold" in had:
            return "held"
        return "company" if had else "quiet"

    # The latest eval-harness run: the safety panel's deploy-gating numbers.
    evals = None
    eval_path = app.state.settings.log_dir / "safety-eval-latest.json"
    if eval_path.exists():
        try:
            evals = json.loads(eval_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 — a corrupt report shows as "not run"
            evals = None

    day_ago = _t.time() - 86400
    return {
        "version": {"running": app.state.version, "repo": app.state.repo},
        "window": {"range": range if range in ("today", "week") else "all",
                   "sessions": len(sessions), "turns": spec_total},
        "tiles": {
            "sessions": len(sessions),
            "sessions_24h": sum(1 for s in sessions if s[1] and s[1] >= day_ago),
            "turns": spec_total,
            "v2v_p50": pct(v2v, .5),
            "v2v_p95": pct(v2v, .95),
            "spec_rate": round(spec_hits / spec_total, 3) if spec_total else None,
            "grounding_rate": round(guide_grounded / guide_total, 3) if guide_total else None,
            "degraded": degraded,
        },
        "modes": modes,
        "emotions": emotion_counts,
        "stages": {
            k: {"p50": round(statistics.median(vs), 3), "p95": pct(vs, .95), "n": len(vs)}
            for k, vs in sorted(stages.items())
        },
        "lanes": {k: round(v / grounded_turns, 3) if grounded_turns else None
                  for k, v in lane_hits.items()},
        "kb": getattr(app.state, "kb_stats", None),
        "evals": evals,
        "sources": dict(sorted(source_counts.items(), key=lambda kv: -kv[1])[:12]),
        "unanswered": dict(sorted(unanswered.items(), key=lambda kv: -kv[1])[:10]),
        "safety_events": safety,
        "sessions": [
            {"id": s[0], "started_at": s[1], "ended_at": s[2], "language": s[3],
             "turns": per_session_turns.get(s[0], 0),
             "opening": (openings.get(s[0]) or "")[:110],
             "p50": pct(per_session_v2v.get(s[0], []), .5),
             "outcome": outcome(s[0])}
            for s in sessions[:20]
        ],
    }


@app.get("/dashboard")
async def dashboard():
    return FileResponse(CLIENT_DIR / "dashboard.html")


@app.get("/api/profile")
async def profile_status(device_id: str):
    """Whether Saathi remembers this device — never the content itself."""
    return app.state.store.profile_status(device_id[:64])


@app.delete("/api/profile")
async def delete_profile(device_id: str):
    """The one-tap delete promised by the consent copy."""
    app.state.store.delete_profile(device_id[:64])
    return {"deleted": True}


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


@app.get("/fonts/{name}")
async def font(name: str):
    path = (CLIENT_DIR / "fonts" / name).resolve()
    if path.parent != (CLIENT_DIR / "fonts").resolve() or not path.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path, media_type="font/woff2")


_BRAND_TYPES = {".svg": "image/svg+xml", ".png": "image/png",
                ".webmanifest": "application/manifest+json"}


@app.get("/brand/{name}")
async def brand(name: str):
    """Favicon, app icons, webmanifest — the mark is the presence, reduced."""
    path = (CLIENT_DIR / "brand" / name).resolve()
    if path.parent != (CLIENT_DIR / "brand").resolve() or not path.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path, media_type=_BRAND_TYPES.get(path.suffix, "application/octet-stream"))


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

    # Load the knowledge base and warm the embedding model NOW, not inside a
    # caller's first turn. A missing index stops the boot: shipping a
    # wellbeing companion that silently can't ground its guidance is worse
    # than not starting.
    import time as _time

    from saathi.kb.embed import FastEmbedEmbedder
    from saathi.kb.index import KnowledgeBase
    from saathi.kb.retrieve import Retriever
    from saathi.rag import QueryUnderstander

    try:
        kb = KnowledgeBase.load(settings.index_dir)
    except FileNotFoundError as exc:
        logger.error(f"\n\n{exc}\n")
        return 1
    embedder = FastEmbedEmbedder()
    warm_started = _time.perf_counter()
    embedder.warm()
    logger.info(
        f"knowledge base: {len(kb)} chunks from {len({c.doc_id for c in kb.chunks})} "
        f"documents · embedder warm in {_time.perf_counter() - warm_started:.1f}s"
    )
    app.state.retriever = Retriever(kb, embedder)
    app.state.kb_stats = {
        "chunks": len(kb),
        "documents": len({c.doc_id for c in kb.chunks}),
    }

    import os as _os

    from saathi import quotes as quotes_module
    from saathi.profile import ProfileUpdater
    from saathi.store import create_store

    app.state.quotes = quotes_module.load(
        settings.helplines_path.parent / "quotes" / "quotes.json"
    )
    logger.info(f"quote index: {len(app.state.quotes)} emotion-tagged quotes")
    app.state.store = create_store(
        _os.getenv("SAATHI_DATABASE_URL"), settings.log_dir / "saathi.db"
    )
    logger.info(f"session store: {app.state.store.path}")
    if settings.llm_provider == "openai":
        app.state.profile_updater = ProfileUpdater(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            base_url=settings.openai_base_url,
        )
    else:
        # Profile updates need an OpenAI-compatible endpoint; on other
        # providers memory injection still works, only the end-of-session
        # learning is off.
        app.state.profile_updater = None
        logger.info("profile updater: off (needs SAATHI_LLM_PROVIDER=openai)")
    if settings.llm_provider == "azure":
        from saathi.rag import AzureQueryUnderstander

        app.state.understander = AzureQueryUnderstander(
            api_key=settings.azure_openai_key,
            endpoint=settings.azure_openai_endpoint,
            deployment=settings.azure_openai_deployment,
        )
    elif settings.llm_provider == "openai":
        from saathi.rag import OpenAICompatUnderstander

        app.state.understander = OpenAICompatUnderstander(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            base_url=settings.openai_base_url,
        )
    else:
        app.state.understander = QueryUnderstander(
            api_key=settings.google_api_key, model=settings.classifier_model
        )

    host = args.host or settings.host
    port = args.port or settings.port

    logger.info(f"Saathi ready — open http://{host}:{port}")
    logger.info(f"config: {settings.redacted()}")
    uvicorn.run(app, host=host, port=port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
