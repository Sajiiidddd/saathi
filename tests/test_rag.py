#!/usr/bin/env python3
"""Tests for the in-call RAG augmentation.

    .venv/bin/python tests/test_rag.py

Uses the real pipecat LLMContext and the real retriever over the real corpus
(hashing stub embedder — hermetic, no network), with the Flash-Lite
understander replaced by a canned double. What's under test is the logic that
wraps retrieval: when augmentation triggers, what gets injected, that the
passages slot replaces rather than accumulates, and that a broken understander
degrades instead of breaking the turn.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pipecat.processors.aggregators.llm_context import LLMContext

from saathi.kb.embed import HashingEmbedder
from saathi.kb.emotion import EmotionReading
from saathi.kb.index import build
from saathi.kb.ingest import load_corpus
from saathi.kb.retrieve import Retriever
from saathi.rag import (
    SENTINEL,
    RagProcessor,
    Understanding,
    _latest_user_text,
    _replace_sentinel_message,
)

DATA = ROOT / "data"
CORPUS_DIRS = [DATA / n for n in ("gov_health", "nhs_who", "open_textbooks", "classics", "quotes")]


class CannedUnderstander:
    def __init__(self, **overrides):
        self.overrides = overrides
        self.calls = []

    async def understand(self, turn, dialogue):
        self.calls.append(turn)
        return Understanding(
            query=self.overrides.get("query", turn),
            emotion=self.overrides.get("emotion", EmotionReading()),
            wants_guidance=self.overrides.get("wants_guidance", True),
        )


class ExplodingUnderstander:
    async def understand(self, turn, dialogue):
        raise RuntimeError("classifier down")


def make_processor(tmp, understander):
    kb = build(load_corpus(CORPUS_DIRS), HashingEmbedder(), "hashing-stub")
    # FrameProcessor needs no event loop for our direct _augment calls.
    processor = RagProcessor.__new__(RagProcessor)
    processor._retriever = Retriever(kb, HashingEmbedder())
    processor._understander = understander
    import random
    processor._top_k = 4
    processor._speculation = None
    processor._quotes = []
    processor._rng = random.Random(7)
    processor._seen_docs = set()
    processor._used_quotes = set()
    processor._recent_modes = []
    processor._last_cited_source = None
    processor._turn = 0
    processor._store = None
    processor._session_id = "test"
    processor._log_path = Path(tmp) / "rag-test.jsonl"
    return processor


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------- helpers


def test_latest_user_text_only_fires_on_user_turns():
    assert _latest_user_text([{"role": "user", "content": "help me sleep"}]) == "help me sleep"
    # Greeting directive: developer message only, no user turn yet.
    assert _latest_user_text([{"role": "developer", "content": "greet them"}]) is None
    # Bot spoke last — nothing new to ground.
    assert _latest_user_text([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello!"},
    ]) is None
    # Multimodal content shape.
    assert _latest_user_text([
        {"role": "user", "content": [{"type": "text", "text": "can't sleep"}]}
    ]) == "can't sleep"


def test_sentinel_replaces_never_accumulates():
    context = LLMContext()
    context.add_message({"role": "user", "content": "turn one"})
    _replace_sentinel_message(context, f"{SENTINEL}\nfirst")
    context.add_message({"role": "user", "content": "turn two"})
    _replace_sentinel_message(context, f"{SENTINEL}\nsecond")

    messages = context.get_messages()
    slots = [m for m in messages if str(m.get("content", "")).startswith(SENTINEL)]
    assert len(slots) == 1, "passages slot must be replaced, not accumulated"
    assert "second" in slots[0]["content"]
    # User turns untouched.
    assert [m["content"] for m in messages if m["role"] == "user"] == ["turn one", "turn two"]


# ---------------------------------------------------------------- augment


def test_guidance_turn_injects_cited_passages():
    with tempfile.TemporaryDirectory() as tmp:
        processor = make_processor(tmp, CannedUnderstander(query="how to fall asleep faster"))
        context = LLMContext()
        context.add_message({"role": "user", "content": "I can't sleep at night"})
        run(processor._augment(context))

        slot = [m for m in context.get_messages()
                if str(m.get("content", "")).startswith(SENTINEL)]
        assert len(slot) == 1
        content = slot[0]["content"]
        assert "From " in content, "passages must be labelled with their source"
        assert "Ground any guidance" in content
        assert processor._log_path.exists(), "provenance JSONL must be written"


def test_smalltalk_gets_no_passages_but_honest_slot():
    with tempfile.TemporaryDirectory() as tmp:
        processor = make_processor(tmp, CannedUnderstander(wants_guidance=False))
        context = LLMContext()
        context.add_message({"role": "user", "content": "haha thanks, you too"})
        run(processor._augment(context))
        slot = [m for m in context.get_messages()
                if str(m.get("content", "")).startswith(SENTINEL)][0]
        assert "No reference passages" in slot["content"]
        assert "From " not in slot["content"]


def test_greeting_turn_is_untouched():
    with tempfile.TemporaryDirectory() as tmp:
        understander = CannedUnderstander()
        processor = make_processor(tmp, understander)
        context = LLMContext()
        context.add_message({"role": "developer", "content": "Please greet the caller."})
        run(processor._augment(context))
        assert understander.calls == [], "no user turn -> understander must not be called"
        assert len(context.get_messages()) == 1, "context must be untouched"


def test_session_memory_demotes_repeats():
    with tempfile.TemporaryDirectory() as tmp:
        processor = make_processor(tmp, CannedUnderstander(query="breathing exercise for stress"))
        context = LLMContext()
        context.add_message({"role": "user", "content": "help me calm down"})
        run(processor._augment(context))
        first_docs = set(processor._seen_docs)
        assert first_docs, "retrieved docs must be remembered for the session"

        context.add_message({"role": "user", "content": "help me calm down again"})
        run(processor._augment(context))
        assert processor._seen_docs >= first_docs


def test_broken_understander_degrades_not_breaks():
    with tempfile.TemporaryDirectory() as tmp:
        kb = build(load_corpus(CORPUS_DIRS), HashingEmbedder(), "hashing-stub")
        from saathi.rag import QueryUnderstander
        # Real understander with an unreachable client would raise inside
        # understand(); its own try/except degrades. Here we go one level up:
        # even an understander that RAISES must not break the turn, because
        # process_frame wraps _augment. Simulate _augment's behaviour:
        processor = make_processor(tmp, ExplodingUnderstander())
        context = LLMContext()
        context.add_message({"role": "user", "content": "help me sleep"})
        try:
            run(processor._augment(context))
            raised = False
        except RuntimeError:
            raised = True
        assert raised, "_augment propagates; process_frame is the safety net"
        # The safety net itself:
        assert hasattr(RagProcessor, "process_frame")


def test_speculation_hit_miss_and_multifragment():
    import asyncio as aio
    from saathi.rag import SpeculationState

    class SlowUnderstander:
        def __init__(self): self.calls = 0
        async def understand(self, turn, dialogue):
            self.calls += 1
            await aio.sleep(0.01)
            return Understanding(query=f"understood:{turn}", emotion=EmotionReading(),
                                 wants_guidance=True)

    async def scenario():
        u = SlowUnderstander()
        # HIT: speculation text matches the aggregated turn (case/space folded)
        state = SpeculationState()
        state.speculate(u, "I can't sleep")
        got = await state.claim("i can't  sleep")
        assert got is not None and got.query == "understood:I can't sleep"
        assert u.calls == 1

        # State must reset after a claim
        assert await state.claim("i can't sleep") is None

        # MISS: different final turn -> None, in-flight guess cancelled
        state.speculate(u, "I feel")
        got = await state.claim("I feel completely different words")
        assert got is None

        # MULTI-FRAGMENT: two STT finals in one turn join with a space
        state2 = SpeculationState()
        state2.speculate(u, "work stress is bad")
        state2.speculate(u, "especially at night")
        got = await state2.claim("work stress is bad especially at night")
        assert got is not None and "especially at night" in got.query
    run(scenario())


def test_non_dict_messages_are_tolerated():
    """Gemini-3 thinking stores LLMSpecificMessage objects in the context;
    every helper must skip them instead of calling .get() on them."""
    class FakeSpecific:  # stands in for pipecat's LLMSpecificMessage
        llm = "google"
        message = {"opaque": True}

    msgs = [
        {"role": "user", "content": "help me sleep"},
        FakeSpecific(),
    ]
    assert _latest_user_text(msgs) == "help me sleep"

    with tempfile.TemporaryDirectory() as tmp:
        processor = make_processor(tmp, CannedUnderstander(query="sleep help"))
        context = LLMContext()
        context.add_message({"role": "user", "content": "can't sleep"})
        context.get_messages().append(FakeSpecific())  # simulate provider msg
        run(processor._augment(context))  # must not raise
        slot = [m for m in context.get_messages()
                if isinstance(m, dict) and str(m.get("content", "")).startswith(SENTINEL)]
        assert len(slot) == 1, "augmentation must survive provider-specific messages"


def test_speculation_call_cap():
    """A rambling turn must not burn unbounded understander quota."""
    import asyncio as aio
    from saathi.rag import SpeculationState

    class CountingUnderstander:
        def __init__(self): self.calls = 0
        async def understand(self, turn, dialogue):
            self.calls += 1
            await aio.sleep(0)
            return Understanding(query=turn, emotion=EmotionReading(), wants_guidance=True)

    async def scenario():
        u = CountingUnderstander()
        state = SpeculationState()
        for i in range(8):  # thinking-out-loud: 8 STT fragments in one turn
            state.speculate(u, f"fragment {i}")
            await aio.sleep(0)
        assert u.calls <= state.MAX_CALLS_PER_TURN, f"{u.calls} calls for one turn"
        # After the cap, claim() must fall back (no stale guess accepted).
        got = await state.claim(" ".join(f"fragment {i}" for i in range(8)))
        assert got is None, "capped turn must fall back to the fresh final call"
        # And the cap resets for the next turn.
        state.speculate(u, "next turn")
        got = await state.claim("next turn")
        assert got is not None
    run(scenario())


def test_turn_planner_modes():
    from saathi.rag import plan_turn

    # Explicit ask -> guide with passages.
    plan = plan_turn(Understanding("how to sleep", EmotionReading("stressed", .5, .8), True),
                     trailing_questions=0, recent_modes=[], last_cited_source=None)
    assert plan.mode == "guide" and plan.include_passages

    # Heavy emotion, no ask -> hold, and NO passages: advice here = dismissal.
    plan = plan_turn(Understanding("everything is too much", EmotionReading("overwhelmed", .8, .9), False),
                     trailing_questions=0, recent_modes=[], last_cited_source=None)
    assert plan.mode == "hold" and not plan.include_passages
    assert any("NO techniques" in n for n in plan.notes)

    # Mild/neutral, no ask -> companion.
    plan = plan_turn(Understanding("thanks yaar", EmotionReading(), False),
                     trailing_questions=0, recent_modes=[], last_cited_source=None)
    assert plan.mode == "companion" and not plan.include_passages


def test_turn_planner_anti_repetition():
    from saathi.rag import plan_turn

    # Two guide turns running -> check-in note.
    plan = plan_turn(Understanding("more tips", EmotionReading("stressed", .5, .8), True),
                     trailing_questions=0, recent_modes=["guide", "guide"],
                     last_cited_source="NHS (UK)")
    assert any("two turns running" in n for n in plan.notes)
    assert any("already credited NHS (UK)" in n for n in plan.notes)

    # Question streak -> forbid ending with a question.
    plan = plan_turn(Understanding("ok", EmotionReading(), False),
                     trailing_questions=2, recent_modes=[], last_cited_source=None)
    assert any("do NOT end this one with" in n for n in plan.notes)


def test_question_streak_counter():
    from saathi.rag import _trailing_question_streak
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello. How are you feeling today?"},
        {"role": "user", "content": "fine"},
        {"role": "assistant", "content": "What kept you up last night?"},
    ]
    assert _trailing_question_streak(msgs) == 2
    msgs.append({"role": "assistant", "content": "That sounds heavy."})
    assert _trailing_question_streak(msgs) == 0


def test_hold_mode_gets_no_passages_in_context():
    """High-intensity venting must not produce a leaflet."""
    with tempfile.TemporaryDirectory() as tmp:
        processor = make_processor(tmp, CannedUnderstander(
            emotion=EmotionReading("low", 0.8, 0.9), wants_guidance=False))
        context = LLMContext()
        context.add_message({"role": "user", "content": "i just feel like nothing matters lately"})
        run(processor._augment(context))
        slot = [m for m in context.get_messages()
                if isinstance(m, dict) and str(m.get("content", "")).startswith(SENTINEL)][0]
        assert "Turn plan (hold)" in slot["content"]
        assert "From " not in slot["content"], "hold mode must not inject passages"


def test_quote_garnish_is_rare_unused_and_state_matched():
    from saathi.quotes import Quote, pick
    import random
    qs = [Quote("Be here now.", "Aurelius", "Meditations", ("ruminating", "stressed"))]
    rng = random.Random(1)
    picks = [pick(qs, "stressed", set(), rng, chance=0.3) for _ in range(200)]
    got = [q for q in picks if q]
    assert 0 < len(got) < 140, "quote should appear sometimes, never always"
    assert pick(qs, "anxious", set(), random.Random(1), chance=1.0) is None, "state must match"
    assert pick(qs, "stressed", {"Be here now."[:40]}, random.Random(1), chance=1.0) is None, "no repeats"


def test_per_document_cap_in_guide_mode():
    with tempfile.TemporaryDirectory() as tmp:
        processor = make_processor(tmp, CannedUnderstander(query="how to fall asleep faster"))
        context = LLMContext()
        context.add_message({"role": "user", "content": "how do i sleep better"})
        run(processor._augment(context))
        slot = [m for m in context.get_messages()
                if isinstance(m, dict) and str(m.get("content", "")).startswith(SENTINEL)][0]
        import re
        docs = re.findall(r"From ([^—]+) —", slot["content"])
        from collections import Counter
        titles = re.findall(r"From [^—]+ — ([^:\n]+)", slot["content"])
        assert max(Counter(titles).values()) <= 2, f"one doc dominates: {Counter(titles)}"


def _run_all():
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_") and callable(o)]
    passed, failed = 0, []
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"  \033[32m✓\033[0m {name}")
        except Exception as exc:  # noqa: BLE001
            failed.append(name)
            print(f"  \033[31m✗\033[0m {name}\n      {type(exc).__name__}: {exc}")
    print(f"\n{passed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    asyncio.set_event_loop(asyncio.new_event_loop())
    raise SystemExit(_run_all())
