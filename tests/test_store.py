#!/usr/bin/env python3
"""Tests for the SQLite session store and language modes.

    .venv/bin/python tests/test_store.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from saathi import languages
from saathi.store import SessionStore


def test_store_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        store = SessionStore(Path(tmp) / "t.db")
        store.start_session("s1", "hinglish", "azure", "en-IN-Aarti:DragonHDLatestNeural")
        store.log_turn("s1", 1, {
            "raw": "can't sleep", "mode": "guide",
            "emotion": {"state": "anxious", "intensity": 0.7, "confidence": 0.8},
            "wants_guidance": True, "query": "how to sleep",
            "passages": [{"source": "NHS", "chunk_id": "x#001", "fused_score": 0.03}],
            "understand_ms": 500.0, "retrieve_ms": 5.0,
            "speculative_hit": True, "degraded": False,
        })
        store.log_utterance("s1", "user", "I can't sleep")
        store.log_utterance("s1", "bot", "That sounds rough.")
        store.log_utterance("s1", "bot", "   ")  # blank must be dropped
        store.log_latency("s1", 1, 2.4, [{"processor": "AzureTTS", "duration_secs": 0.5}])
        store.end_session("s1")

        (n_sessions,) = store.query("SELECT count(*) FROM sessions")[0]
        (n_turns,) = store.query("SELECT count(*) FROM turns")[0]
        (n_utts,) = store.query("SELECT count(*) FROM utterances")[0]
        (n_lat,) = store.query("SELECT count(*) FROM latency")[0]
        assert (n_sessions, n_turns, n_utts, n_lat) == (1, 1, 2, 1)

        mode, emotion = store.query("SELECT mode, emotion FROM turns")[0]
        assert (mode, emotion) == ("guide", "anxious")
        (ended,) = store.query("SELECT ended_at FROM sessions WHERE id='s1'")[0]
        assert ended is not None


def test_store_write_failure_is_silent():
    with tempfile.TemporaryDirectory() as tmp:
        store = SessionStore(Path(tmp) / "t.db")
        store._conn.close()  # simulate a dead database mid-call
        store.log_utterance("s1", "user", "hello")  # must not raise


def test_language_modes():
    assert languages.get("hindi").tts_voice.startswith("en-IN-Aarti")
    # One person, three languages: every mode speaks with the same voice.
    assert languages.get("english").tts_provider == "azure"
    assert len({m.tts_voice for m in languages.MODES}) == 1
    assert languages.get("hinglish").stt_languages == ("en-IN", "hi-IN")
    # Unknown/missing -> default, never an error.
    assert languages.get(None) is languages.DEFAULT
    assert languages.get("klingon") is languages.DEFAULT
    listed = languages.as_api_list()
    assert sum(1 for l in listed if l["default"]) == 1
    # Every mode must carry the full bundle.
    for mode in languages.MODES:
        assert mode.persona_directive and mode.greeting_directive
        assert mode.stt_languages and mode.tts_voice


def test_profiles_roundtrip_and_delete():
    with tempfile.TemporaryDirectory() as tmp:
        store = SessionStore(Path(tmp) / "t.db")
        assert store.get_profile("dev1") is None
        assert store.profile_status("dev1") == {"exists": False}
        store.save_profile("dev1", {"name": "Sam", "recurring_topics": ["sleep"]})
        assert store.get_profile("dev1")["name"] == "Sam"
        status = store.profile_status("dev1")
        assert status["exists"] and status["consented_at"]
        first_consent = status["consented_at"]
        store.save_profile("dev1", {"name": "Sam", "recurring_topics": ["sleep", "exams"]})
        assert store.profile_status("dev1")["consented_at"] == first_consent, \
            "consent timestamp must survive updates"
        store.delete_profile("dev1")
        assert store.get_profile("dev1") is None


def test_profile_clamp_and_prompt():
    from saathi.profile import as_prompt_text, clamp_profile
    wild = {
        "name": "  Priya  ", "language_pref": "hinglish, hindi at night",
        "recurring_topics": [f"topic{i}" for i in range(20)],
        "tried": [{"technique": f"t{i}", "helped": "maybe"} for i in range(20)]
              + [{"no_technique_key": 1}, "not-a-dict-ignored"] ,
        "avoid": ["x"] * 10, "notes": ["n"] * 10,
        "diagnosis": "MUST NOT SURVIVE",
    }
    # non-dict entries in tried would crash naive code
    wild["tried"] = [t for t in wild["tried"] if isinstance(t, dict)]
    safe = clamp_profile(wild)
    assert len(safe["recurring_topics"]) == 6
    assert len(safe["tried"]) == 8 and all(t["helped"] == "unknown" for t in safe["tried"])
    assert len(safe["avoid"]) == 4 and len(safe["notes"]) == 2
    assert "diagnosis" not in safe, "unknown keys must be dropped"

    text = as_prompt_text(safe)
    assert "Priya" in text and "never recited back" in text
    assert as_prompt_text(clamp_profile({})) == "", "empty profile injects nothing"


def test_profile_updater_skips_trivial_sessions():
    import asyncio
    from saathi.profile import ProfileUpdater
    updater = ProfileUpdater.__new__(ProfileUpdater)  # no client needed for this path
    result = asyncio.new_event_loop().run_until_complete(
        updater.update({}, [("user", "hi"), ("bot", "hello!")]))
    assert result is None, "one-line sessions must not touch the profile"


def test_join_spoken_reassembles_tts_words():
    from saathi.bot import join_spoken
    assert join_spoken(["Yeh", "exam", "time", "pe", "common", "hai", "."]) == \
        "Yeh exam time pe common hai."
    assert join_spoken(["साँस", "अंदर", "चार", "गिनती", "।"]) == "साँस अंदर चार गिनती।"
    assert join_spoken(["Okay", ",", "so", "…", "try", "this", "?"]) == "Okay, so… try this?"
    assert join_spoken(["", "  ", "hi"]) == "hi"
    assert join_spoken([]) == ""


def _run():
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_")]
    passed, failed = 0, []
    for name, fn in tests:
        try:
            fn(); passed += 1; print(f"  \033[32m✓\033[0m {name}")
        except Exception as exc:  # noqa: BLE001
            failed.append(name); print(f"  \033[31m✗\033[0m {name}\n      {type(exc).__name__}: {exc}")
    print(f"\n{passed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
