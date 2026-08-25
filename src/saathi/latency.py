"""Per-stage latency instrumentation.

Wired in before the first conversation, not after the first complaint: a
voice-to-voice latency claim is only worth quoting if every session was
measured, and as a distribution rather than a lucky single run.

What gets measured: Pipecat's own UserBotLatencyObserver already times the gap
from "user stopped speaking" to "bot started speaking" and, with metrics
enabled, hands over a per-service TTFB breakdown. This module does the two
things it doesn't: persist every turn as JSONL for the dashboard to read, and
compute p50/p95.

JSONL schema, one object per turn (the dashboard contract):
    session_id            str
    turn                  int    1-based
    voice_to_voice_secs   float  user stopped speaking -> bot started speaking
    stages                [{processor, model, start_time, duration_secs}, ...]
    user_turn_secs        float | None   how long the caller spoke
    events                [str] | None   chronological, human-readable
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from loguru import logger


def percentile(values: list[float], pct: float) -> float | None:
    """Linear-interpolation percentile. Returns None for an empty sample."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100.0)
    low, high = math.floor(rank), math.ceil(rank)
    if low == high:
        return ordered[int(rank)]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


class LatencyRecorder:
    """Collects turn latencies, appends JSONL, prints a session summary.

    Pipecat fires the measurement and the breakdown as two separate events with
    no guaranteed order, so a turn is buffered until both halves land (or the
    session ends, whichever comes first).
    """

    def __init__(self, session_id: str, log_dir: Path, store=None):
        self.session_id = session_id
        self._store = store
        self.path = log_dir / f"latency-{session_id}.jsonl"
        self._turn_index = 0
        self._buffer: dict[str, Any] = {}
        self._voice_to_voice: list[float] = []
        self._per_stage: dict[str, list[float]] = {}
        self.first_response_secs: float | None = None

    # ---- wiring -----------------------------------------------------------

    def attach(self, observer) -> None:
        """Register handlers on a UserBotLatencyObserver.

        Kept in a try/except because these three event names are the piece of
        this project most exposed to a Pipecat version bump. If they ever
        change, the call must still work — losing telemetry is an annoyance,
        dropping a conversation is a failure.
        """
        try:

            @observer.event_handler("on_latency_measured")
            async def _on_measured(_obs, latency_seconds: float):
                self._buffer["voice_to_voice_secs"] = round(float(latency_seconds), 4)
                self._maybe_flush()

            @observer.event_handler("on_latency_breakdown")
            async def _on_breakdown(_obs, breakdown):
                self._buffer["stages"] = self._read_stages(breakdown)
                self._buffer["user_turn_secs"] = _maybe_round(
                    getattr(breakdown, "user_turn_secs", None)
                )
                events = getattr(breakdown, "chronological_events", None)
                if callable(events):
                    self._buffer["events"] = list(events())
                self._maybe_flush()

            @observer.event_handler("on_first_bot_speech_latency")
            async def _on_first(_obs, latency_seconds: float):
                self.first_response_secs = round(float(latency_seconds), 4)
                logger.info(f"⏱  first bot speech after {self.first_response_secs:.2f}s")

        except Exception as exc:  # pragma: no cover - version-drift guard
            logger.warning(
                f"Latency observer events did not bind ({exc}). The call will still "
                f"work; per-turn latency logging is off. Check UserBotLatencyObserver's "
                f"event names against your installed Pipecat version."
            )

    @staticmethod
    def _read_stages(breakdown) -> list[dict[str, Any]]:
        stages = []
        for item in getattr(breakdown, "ttfb", []) or []:
            stages.append(
                {
                    "processor": getattr(item, "processor", "?"),
                    "model": getattr(item, "model", None),
                    "start_time": _maybe_round(getattr(item, "start_time", None)),
                    "duration_secs": _maybe_round(getattr(item, "duration_secs", None)),
                }
            )
        return stages

    # ---- recording --------------------------------------------------------

    def _maybe_flush(self) -> None:
        if "voice_to_voice_secs" in self._buffer and "stages" in self._buffer:
            self._write(self._buffer)
            self._buffer = {}

    def flush_partial(self) -> None:
        """Write a half-complete turn at session end rather than losing it."""
        if self._buffer:
            self._write(self._buffer)
            self._buffer = {}

    def _write(self, turn: dict[str, Any]) -> None:
        self._turn_index += 1
        record = {"session_id": self.session_id, "turn": self._turn_index, **turn}

        v2v = record.get("voice_to_voice_secs")
        if isinstance(v2v, (int, float)):
            self._voice_to_voice.append(float(v2v))
        for stage in record.get("stages") or []:
            duration = stage.get("duration_secs")
            if isinstance(duration, (int, float)):
                self._per_stage.setdefault(stage["processor"], []).append(float(duration))

        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        except OSError as exc:
            logger.warning(f"Could not append latency record: {exc}")
        if self._store is not None:
            self._store.log_latency(
                self.session_id, self._turn_index, v2v, record.get("stages") or []
            )

        if isinstance(v2v, (int, float)):
            slowest = max(
                (s for s in record.get("stages") or [] if s.get("duration_secs")),
                key=lambda s: s["duration_secs"],
                default=None,
            )
            tail = (
                f"  slowest stage: {slowest['processor']} {slowest['duration_secs']:.2f}s"
                if slowest
                else ""
            )
            logger.info(f"⏱  turn {self._turn_index}: {v2v:.2f}s voice-to-voice{tail}")

    # ---- reporting --------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turns": len(self._voice_to_voice),
            "first_response_secs": self.first_response_secs,
            "voice_to_voice": {
                "p50": _maybe_round(percentile(self._voice_to_voice, 50)),
                "p95": _maybe_round(percentile(self._voice_to_voice, 95)),
                "min": _maybe_round(min(self._voice_to_voice, default=None)),
                "max": _maybe_round(max(self._voice_to_voice, default=None)),
            },
            "stages": {
                name: {
                    "p50": _maybe_round(percentile(samples, 50)),
                    "p95": _maybe_round(percentile(samples, 95)),
                    "n": len(samples),
                }
                for name, samples in sorted(self._per_stage.items())
            },
        }

    def log_summary(self) -> None:
        self.flush_partial()
        if not self._voice_to_voice:
            logger.info("⏱  no complete turns measured this session")
            return

        report = self.summary()
        v2v = report["voice_to_voice"]
        lines = [
            "",
            "─" * 58,
            f"  LATENCY — session {self.session_id} · {report['turns']} turn(s)",
            "─" * 58,
            f"  voice-to-voice   p50 {v2v['p50']:.2f}s   p95 {v2v['p95']:.2f}s"
            f"   (min {v2v['min']:.2f} / max {v2v['max']:.2f})",
        ]
        if report["stages"]:
            lines.append("  per stage (time to first byte):")
            width = max(len(n) for n in report["stages"])
            for name, stat in report["stages"].items():
                lines.append(
                    f"    {name:<{width}}  p50 {stat['p50']:.2f}s  "
                    f"p95 {stat['p95']:.2f}s  n={stat['n']}"
                )
        lines += [f"  raw: {self.path}", "─" * 58, ""]
        logger.info("\n".join(lines))


def _maybe_round(value: Any, places: int = 4) -> Any:
    return round(float(value), places) if isinstance(value, (int, float)) else value
