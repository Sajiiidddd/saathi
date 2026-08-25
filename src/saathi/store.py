"""Session store — every conversation becomes data you can learn from.

Two interchangeable backends behind one interface:

- SQLite (default): zero setup, one file in logs/, perfect for local work.
- Postgres: set SAATHI_DATABASE_URL (e.g. a Supabase connection string) and
  the same schema and calls run there — sessions survive redeploys and the
  dashboard can read from anywhere.

Tables:
    sessions     one row per call: language mode, voice, timings
    turns        one row per user turn: text, mode, emotion, query rewrite,
                 sources used, stage timings — the RAG-improvement goldmine
    utterances   raw transcript stream, both directions, timestamped
    latency      per-turn voice-to-voice + stage breakdown
    profiles     opt-in caller memory (see saathi.profile) — consent
                 timestamped, deletable in one call

Privacy posture unchanged: audio is never recorded; this stores text and
numbers, deletable by dropping one file (SQLite) or one command (Postgres).
Every write is fail-silent — a storage hiccup must never take down a call.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from loguru import logger

_TABLES = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    started_at DOUBLE PRECISION NOT NULL,
    ended_at DOUBLE PRECISION,
    language TEXT,
    tts_provider TEXT,
    tts_voice TEXT
);
CREATE TABLE IF NOT EXISTS turns (
    session_id TEXT NOT NULL,
    turn INTEGER NOT NULL,
    ts DOUBLE PRECISION NOT NULL,
    user_text TEXT,
    mode TEXT,
    emotion TEXT,
    intensity DOUBLE PRECISION,
    confidence DOUBLE PRECISION,
    wants_guidance INTEGER,
    query TEXT,
    sources TEXT,
    understand_ms DOUBLE PRECISION,
    retrieve_ms DOUBLE PRECISION,
    speculative_hit INTEGER,
    degraded INTEGER,
    PRIMARY KEY (session_id, turn)
);
CREATE TABLE IF NOT EXISTS utterances (
    session_id TEXT NOT NULL,
    ts DOUBLE PRECISION NOT NULL,
    role TEXT NOT NULL,
    text TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS latency (
    session_id TEXT NOT NULL,
    turn INTEGER NOT NULL,
    voice_to_voice_secs DOUBLE PRECISION,
    stages TEXT
);
CREATE TABLE IF NOT EXISTS profiles (
    device_id TEXT PRIMARY KEY,
    consented_at DOUBLE PRECISION,
    updated_at DOUBLE PRECISION,
    data TEXT
);
CREATE TABLE IF NOT EXISTS safety_events (
    session_id TEXT NOT NULL,
    ts DOUBLE PRECISION NOT NULL,
    kind TEXT NOT NULL,        -- 'crisis' | 'scope'
    matched TEXT,              -- the pattern that fired
    user_text TEXT
);
"""


class _BaseStore:
    """Shared logic; subclasses provide _execute/_fetch and a placeholder."""

    placeholder = "?"

    def _migrate(self) -> None:
        """Additive migrations for schema drift during development.

        CREATE TABLE IF NOT EXISTS never adds columns to existing tables, so
        each new column needs one guarded ALTER here.
        """
        for probe, alter in (
            ("SELECT updated_at FROM profiles LIMIT 1",
             "ALTER TABLE profiles ADD COLUMN updated_at DOUBLE PRECISION"),
            ("SELECT plan FROM turns LIMIT 1",
             "ALTER TABLE turns ADD COLUMN plan TEXT"),
        ):
            try:
                self._fetch(probe, ())
            except Exception:  # noqa: BLE001
                try:
                    self._execute(alter, ())
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"migration failed ({alter}): {exc}")

    def _sql(self, sql: str) -> str:
        return sql if self.placeholder == "?" else sql.replace("?", self.placeholder)

    def _write(self, sql: str, params: tuple) -> None:
        try:
            self._execute(self._sql(sql), params)
        except Exception as exc:  # noqa: BLE001 — never let storage kill a call
            logger.warning(f"session store write failed: {exc}")

    def query(self, sql: str, params: tuple = ()) -> list[tuple]:
        return self._fetch(self._sql(sql), params)

    # ---- session lifecycle ------------------------------------------------

    def start_session(self, session_id: str, language: str, provider: str, voice: str) -> None:
        self._write(
            "INSERT INTO sessions (id, started_at, language, tts_provider, tts_voice)"
            " VALUES (?, ?, ?, ?, ?)"
            " ON CONFLICT (id) DO UPDATE SET started_at = excluded.started_at,"
            " language = excluded.language, tts_provider = excluded.tts_provider,"
            " tts_voice = excluded.tts_voice",
            (session_id, time.time(), language, provider, voice),
        )

    def end_session(self, session_id: str) -> None:
        self._write("UPDATE sessions SET ended_at = ? WHERE id = ?", (time.time(), session_id))

    # ---- per-turn records ---------------------------------------------------

    def log_turn(self, session_id: str, turn: int, record: dict) -> None:
        emotion = record.get("emotion", {})
        # Full provenance survives into the store — the insight drawer renders
        # lane ranks and previews, not just source names.
        sources = record.get("passages", [])
        self._write(
            "INSERT INTO turns (session_id, turn, ts, user_text, mode, emotion,"
            " intensity, confidence, wants_guidance, query, sources, understand_ms,"
            " retrieve_ms, speculative_hit, degraded, plan)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT (session_id, turn) DO NOTHING",
            (
                session_id,
                turn,
                time.time(),
                record.get("raw"),
                record.get("mode"),
                emotion.get("state"),
                emotion.get("intensity"),
                emotion.get("confidence"),
                int(bool(record.get("wants_guidance"))),
                record.get("query"),
                json.dumps(sources, ensure_ascii=False),
                record.get("understand_ms"),
                record.get("retrieve_ms"),
                int(bool(record.get("speculative_hit"))),
                int(bool(record.get("degraded"))),
                json.dumps(record.get("plan_notes") or [], ensure_ascii=False),
            ),
        )

    def log_utterance(self, session_id: str, role: str, text: str) -> None:
        if text.strip():
            self._write(
                "INSERT INTO utterances (session_id, ts, role, text) VALUES (?, ?, ?, ?)",
                (session_id, time.time(), role, text.strip()),
            )

    def log_latency(self, session_id: str, turn: int, v2v, stages: list) -> None:
        self._write(
            "INSERT INTO latency (session_id, turn, voice_to_voice_secs, stages)"
            " VALUES (?, ?, ?, ?)",
            (session_id, turn, v2v, json.dumps(stages)),
        )

    def log_safety_event(self, session_id: str, kind: str, matched: str, user_text: str) -> None:
        self._write(
            "INSERT INTO safety_events (session_id, ts, kind, matched, user_text)"
            " VALUES (?, ?, ?, ?, ?)",
            (session_id, time.time(), kind, matched, user_text),
        )

    # ---- opt-in caller profiles ----------------------------------------------

    def get_profile(self, device_id: str) -> dict | None:
        rows = self.query("SELECT data FROM profiles WHERE device_id = ?", (device_id,))
        if not rows or not rows[0][0]:
            return None
        try:
            return json.loads(rows[0][0])
        except (json.JSONDecodeError, TypeError):
            return None

    def save_profile(self, device_id: str, data: dict) -> None:
        now = time.time()
        self._write(
            "INSERT INTO profiles (device_id, consented_at, updated_at, data)"
            " VALUES (?, ?, ?, ?)"
            " ON CONFLICT (device_id) DO UPDATE SET data = excluded.data,"
            " updated_at = excluded.updated_at",
            (device_id, now, now, json.dumps(data, ensure_ascii=False)),
        )

    def profile_status(self, device_id: str) -> dict:
        rows = self.query(
            "SELECT consented_at, updated_at FROM profiles WHERE device_id = ?", (device_id,)
        )
        if not rows:
            return {"exists": False}
        return {"exists": True, "consented_at": rows[0][0], "updated_at": rows[0][1]}

    def delete_profile(self, device_id: str) -> None:
        self._write("DELETE FROM profiles WHERE device_id = ?", (device_id,))


class SQLiteSessionStore(_BaseStore):
    placeholder = "?"

    def __init__(self, path: Path):
        import sqlite3

        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.executescript(_TABLES)
        self._conn.commit()
        self._lock = threading.Lock()
        self.path = path
        self._migrate()

    def _execute(self, sql: str, params: tuple) -> None:
        with self._lock:
            self._conn.execute(sql, params)
            self._conn.commit()

    def _fetch(self, sql: str, params: tuple) -> list[tuple]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()


class PostgresSessionStore(_BaseStore):
    placeholder = "%s"

    def __init__(self, database_url: str):
        import psycopg

        self._conn = psycopg.connect(database_url, autocommit=True)
        with self._conn.cursor() as cur:
            cur.execute(_TABLES)
        self._lock = threading.Lock()
        self.path = database_url.split("@")[-1]  # host part only — no credentials in logs
        self._migrate()

    def _execute(self, sql: str, params: tuple) -> None:
        with self._lock, self._conn.cursor() as cur:
            cur.execute(sql, params)

    def _fetch(self, sql: str, params: tuple) -> list[tuple]:
        with self._lock, self._conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()


# Existing call sites and tests use this name for the local default.
SessionStore = SQLiteSessionStore


def create_store(database_url: str | None, sqlite_path: Path):
    """Postgres when a URL is configured, SQLite otherwise."""
    if database_url:
        store = PostgresSessionStore(database_url)
        logger.info(f"session store: postgres @ {store.path}")
        return store
    return SQLiteSessionStore(sqlite_path)
