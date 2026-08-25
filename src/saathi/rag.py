"""Grounded retrieval inside the voice loop.

Sits between the user aggregator and the LLM. For each user turn:

1. UNDERSTAND — one Flash-Lite call returns, as JSON: a self-contained
   English retrieval query (voice turns are fragments — "and at night?" only
   means something given the dialogue; Hindi/Hinglish turns are translated
   here too, because the corpus and both lexical lanes are English), an
   emotional state reading for the crosswalk lane, and whether the caller is
   asking for guidance at all. One call, not three — every serial network
   round-trip lands inside the caller's silence.

2. RETRIEVE — the three-lane fused search (semantic + BM25 + emotion),
   with session memory: documents already used this call are demoted so the
   same exercise isn't offered twice.

3. INJECT — retrieved passages, labelled with their sources, are placed into
   the LLM context as a single developer message that REPLACES last turn's
   passages rather than accumulating. The message carries the grounding
   contract: guidance only from these passages, cite the source naturally in
   speech, and if nothing fits — say so honestly.

Failure posture: retrieval is an enhancement, never a gate. If the
understander times out or the index misbehaves, the turn proceeds ungrounded
(the persona prompt then forbids invented citations) — a retrieval hiccup
must never cost someone the conversation. The safety gate, when it lands in
front of this processor, has exactly the opposite posture: deterministic, and
allowed to stop the world.

Every turn writes provenance to logs/rag-<session>.jsonl: raw vs rewritten
query, emotion reading, per-stage milliseconds, and each passage's per-lane
ranks. That file is the "why did she say that?" record.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from pipecat.frames.frames import Frame, LLMContextFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from .kb.emotion import STATES, EmotionReading
from .kb.retrieve import Retriever

SENTINEL = "[reference passages]"

UNDERSTAND_TIMEOUT_SECS = 3.0

_UNDERSTAND_PROMPT = """\
You prepare a voice-companion caller's turn for knowledge-base retrieval.
The knowledge base is in ENGLISH and covers everyday wellbeing: stress, sleep,
anxiety, low mood, breathing and grounding techniques, mindfulness, gratitude.

Given the recent dialogue and the caller's latest turn, return ONLY a JSON
object with exactly these keys:
- "query": the latest turn rewritten as one self-contained ENGLISH search
  query. Resolve pronouns and fragments from the dialogue ("and at night?"
  after a stress chat -> "how to manage stress at night"). Translate Hindi or
  mixed Hindi-English into English. Keep it under 20 words.
- "emotion": the caller's current emotional state, exactly one of:
  {states}. Use "neutral" when nothing clearly applies.
- "intensity": 0.0-1.0, how strong that emotion reads.
- "confidence": 0.0-1.0, how sure you are of the emotion label.
- "wants_guidance": true if the caller is asking for help, information, or a
  technique; false for pure smalltalk, greetings, or acknowledgements.

Recent dialogue:
{dialogue}

Caller's latest turn: {turn}
"""


@dataclass
class Understanding:
    query: str
    emotion: EmotionReading
    wants_guidance: bool
    degraded: bool = False  # True when the classifier failed and we fell back


class QueryUnderstander:
    """The single Flash-Lite call: rewrite + emotion + intent."""

    def __init__(self, api_key: str, model: str):
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._model = model

    async def understand(self, turn: str, dialogue: list[str]) -> Understanding:
        from google.genai import types

        prompt = _UNDERSTAND_PROMPT.format(
            states=", ".join(STATES),
            dialogue="\n".join(dialogue[-6:]) or "(start of call)",
            turn=turn,
        )
        try:
            # The latency budget is enforced client-side with wait_for: the
            # API rejects http_options deadlines under 10s, and 10s of dead
            # air in a voice call is an eternity. On timeout we degrade to
            # the raw query rather than keep the caller waiting.
            response = await asyncio.wait_for(
                self._client.aio.models.generate_content(
                    model=self._model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        max_output_tokens=200,
                        response_mime_type="application/json",
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    ),
                ),
                timeout=UNDERSTAND_TIMEOUT_SECS,
            )
            return _parse_understanding(response.text, turn)
        except Exception as exc:  # noqa: BLE001 — degrade, never break the turn
            logger.warning(f"understander degraded ({type(exc).__name__}: {exc}) — raw query")
            return Understanding(
                query=turn, emotion=EmotionReading(), wants_guidance=True, degraded=True
            )


class OpenAICompatUnderstander:
    """Understander over any OpenAI-compatible endpoint (GitHub Models, or
    api.openai.com itself). Same contract, same degrade-on-failure posture."""

    def __init__(self, api_key: str, model: str, base_url: str | None = None):
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url or None)
        self._model = model

    async def understand(self, turn: str, dialogue: list[str]) -> Understanding:
        prompt = _UNDERSTAND_PROMPT.format(
            states=", ".join(STATES),
            dialogue="\n".join(dialogue[-6:]) or "(start of call)",
            turn=turn,
        )
        kwargs = {}
        if "gpt-oss" in self._model:
            # Without this, reasoning burns the whole latency budget and JSON
            # mode fails validation. See services.build_llm for the same knob.
            kwargs["reasoning_effort"] = "low"
        try:
            response = await asyncio.wait_for(
                self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=250,
                    response_format={"type": "json_object"},
                    **kwargs,
                ),
                timeout=UNDERSTAND_TIMEOUT_SECS,
            )
            return _parse_understanding(response.choices[0].message.content, turn)
        except Exception as exc:  # noqa: BLE001 — degrade, never break the turn
            logger.warning(f"understander degraded ({type(exc).__name__}: {exc}) — raw query")
            return Understanding(
                query=turn, emotion=EmotionReading(), wants_guidance=True, degraded=True
            )


class AzureQueryUnderstander:
    """Same contract as QueryUnderstander, served by an Azure OpenAI deployment.

    Used when SAATHI_LLM_PROVIDER=azure so the whole reasoning stack lives on
    one vendor and one quota. The deployment is shared with the main
    conversation model — a rewrite+classify call is ~200 tokens, noise next
    to the conversation itself.
    """

    def __init__(self, api_key: str, endpoint: str, deployment: str):
        from openai import AsyncAzureOpenAI

        self._client = AsyncAzureOpenAI(
            api_key=api_key, azure_endpoint=endpoint, api_version="2024-10-21"
        )
        self._deployment = deployment

    async def understand(self, turn: str, dialogue: list[str]) -> Understanding:
        prompt = _UNDERSTAND_PROMPT.format(
            states=", ".join(STATES),
            dialogue="\n".join(dialogue[-6:]) or "(start of call)",
            turn=turn,
        )
        try:
            response = await asyncio.wait_for(
                self._client.chat.completions.create(
                    model=self._deployment,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=200,
                    response_format={"type": "json_object"},
                ),
                timeout=UNDERSTAND_TIMEOUT_SECS,
            )
            return _parse_understanding(response.choices[0].message.content, turn)
        except Exception as exc:  # noqa: BLE001 — degrade, never break the turn
            logger.warning(f"understander degraded ({type(exc).__name__}: {exc}) — raw query")
            return Understanding(
                query=turn, emotion=EmotionReading(), wants_guidance=True, degraded=True
            )


def _parse_understanding(raw: str, turn: str) -> Understanding:
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        # Some models wrap JSON in prose or code fences despite instructions —
        # salvage the first {...} block before giving up.
        import re

        match = re.search(r"\{.*\}", raw or "", re.S)
        if not match:
            raise
        data = json.loads(match.group(0))
    state = data.get("emotion", "neutral")
    reading = EmotionReading(
        state=state if state in STATES else "neutral",
        intensity=_clamp(data.get("intensity", 0.0)),
        confidence=_clamp(data.get("confidence", 0.0)),
    )
    return Understanding(
        query=(data.get("query") or "").strip() or turn,
        emotion=reading,
        wants_guidance=bool(data.get("wants_guidance", True)),
    )


def _normalise(text: str) -> str:
    return " ".join(text.split()).casefold()


# ---------------------------------------------------------------------------
# Turn planning — the difference between a companion and a leaflet dispenser.
#
# Retrieval answers "what could she say?"; the plan answers "what does this
# moment need?". Without it every turn takes the same shape — empathise,
# technique, citation, question — and after three exchanges she is a mode,
# not a person. The emotion classifier already runs on every turn; here its
# output chooses the response POSTURE, not just the retrieval boost.
# ---------------------------------------------------------------------------

# States where someone mostly needs to feel heard before being handed a tool.
_HEAVY_STATES = ("low", "overwhelmed", "anxious", "ruminating")


@dataclass
class TurnPlan:
    mode: str  # "guide" | "hold" | "companion"
    include_passages: bool
    notes: list[str]


def plan_turn(
    understanding: Understanding,
    trailing_questions: int,
    recent_modes: list[str],
    last_cited_source: str | None,
) -> TurnPlan:
    """Choose this turn's posture. Pure function — the tests own it.

    guide     — they asked for help: passages in, one technique, cited.
    hold      — heavy emotion, no ask: no passages, no advice. Reflect,
                validate, sit with it. Advice given here lands as dismissal.
    companion — ordinary conversation: no citations, no coaching, just talk.
    """
    notes: list[str] = []
    emotion = understanding.emotion

    if understanding.wants_guidance:
        mode = "guide"
        if recent_modes[-2:] == ["guide", "guide"]:
            notes.append(
                "You have given guidance two turns running. Before anything new, "
                "briefly check how the last suggestion sat with them — and keep "
                "any new idea to one short thought."
            )
        if last_cited_source:
            notes.append(
                f"You already credited {last_cited_source} recently — if you use "
                f"it again, don't re-announce the name, just speak the guidance."
            )
    elif emotion.state in _HEAVY_STATES and emotion.intensity >= 0.6:
        mode = "hold"
        notes.append(
            "This is a listening moment, not a fixing one. Reflect back the one "
            "thing that seems to matter most, in their own words. NO techniques, "
            "no tips, no sources this turn. If it feels right, gently ask whether "
            "they want to just talk or would like to try something — and accept "
            "either answer."
        )
    else:
        mode = "companion"
        notes.append(
            "Plain conversation. No guidance, no citations, no coaching energy — "
            "respond the way a close friend would, brief and natural."
        )

    if emotion.intensity >= 0.7:
        notes.append("Keep it short — one or two unhurried sentences. Let silence do some work.")
    if trailing_questions >= 2:
        notes.append(
            "Your last two turns ended with questions — do NOT end this one with "
            "a question. A statement that shows you heard them is enough."
        )
    return TurnPlan(mode=mode, include_passages=(mode == "guide"), notes=notes)


def _trailing_question_streak(messages) -> int:
    """How many consecutive recent assistant turns ended in a question."""
    streak = 0
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        text = _text_of(message) or ""
        if text.rstrip().endswith(("?", "?\"", "?'")):
            streak += 1
        else:
            break
    return streak


class SpeculationState:
    """Shared slot between the Speculator and the RagProcessor.

    The understander round-trip costs ~900ms and can't get faster (it's
    network floor, not model time). But Azure STT delivers final transcript
    fragments while turn-taking is still deciding the caller has finished —
    a runway of roughly 0.5-1s. Firing the understander on those fragments
    means its answer is usually ready the moment the turn closes: the ~900ms
    disappears from the caller's wait entirely on a hit. On a miss (the
    aggregated turn text differs from what we speculated on), the fallback
    is exactly today's behaviour, so speculation can only ever help.
    """

    # A long thinking-out-loud turn can produce a dozen STT fragments, and
    # each speculative call spends real quota whether or not it gets used —
    # cancelling an in-flight request does not refund it. Cap the spend per
    # turn: fire on the first fragment, refresh once on the second, then go
    # quiet and let the final fallback call handle whatever the turn became.
    MAX_CALLS_PER_TURN = 2

    def __init__(self):
        self.fragments: list[str] = []
        self.speculated_text: str | None = None
        self.task: asyncio.Task | None = None
        self.calls_this_turn = 0
        self.dialogue: list[str] = []  # tail from the PREVIOUS turn — correct
        # for speculation, since the current turn isn't dialogue history yet.

    def speculate(self, understander: "QueryUnderstander", fragment: str) -> None:
        self.fragments.append(fragment)
        if self.calls_this_turn >= self.MAX_CALLS_PER_TURN:
            # Still track fragments so claim() correctly detects the mismatch
            # and falls back — but spend no more quota on guesses.
            self.speculated_text = None
            return
        text = " ".join(self.fragments)
        # A newer fragment supersedes any in-flight guess for this turn.
        if self.task and not self.task.done():
            self.task.cancel()
        self.speculated_text = text
        self.calls_this_turn += 1
        self.task = asyncio.create_task(understander.understand(text, self.dialogue))

    async def claim(self, turn_text: str) -> Understanding | None:
        """Return the speculative result iff it was computed for this turn."""
        task, speculated = self.task, self.speculated_text
        self.fragments = []
        self.speculated_text = None
        self.task = None
        self.calls_this_turn = 0
        if task is None or speculated is None:
            return None
        if _normalise(speculated) != _normalise(turn_text):
            task.cancel()
            return None
        try:
            return await task
        except asyncio.CancelledError:
            return None


class Speculator(FrameProcessor):
    """Watches final STT fragments upstream of the aggregator and pre-runs
    the understander. Passes every frame through untouched — pure observer."""

    def __init__(self, state: SpeculationState, understander: "QueryUnderstander", **kwargs):
        super().__init__(**kwargs)
        self._state = state
        self._understander = understander

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, TranscriptionFrame):
            text = (frame.text or "").strip()
            if text:
                self._state.speculate(self._understander, text)
        await self.push_frame(frame, direction)


class RagProcessor(FrameProcessor):
    """Pipecat processor: augments the LLM context with cited passages."""

    def __init__(
        self,
        retriever: Retriever,
        understander: QueryUnderstander,
        session_id: str,
        log_dir: Path,
        top_k: int = 4,
        speculation: SpeculationState | None = None,
        quotes: list | None = None,
        rng=None,
        store=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        import random

        self._retriever = retriever
        self._understander = understander
        self._top_k = top_k
        self._speculation = speculation
        self._quotes = quotes or []
        self._rng = rng or random.Random()
        self._seen_docs: set[str] = set()
        self._used_quotes: set[str] = set()
        self._recent_modes: list[str] = []
        self._last_cited_source: str | None = None
        self._turn = 0
        self._store = store
        self._session_id = session_id
        self._log_path = log_dir / f"rag-{session_id}.jsonl"

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, LLMContextFrame):
            try:
                await self._augment(frame.context)
            except Exception as exc:  # noqa: BLE001 — see module docstring
                logger.warning(f"rag augmentation skipped ({type(exc).__name__}: {exc})")
        await self.push_frame(frame, direction)

    # ---- the actual work -------------------------------------------------

    async def _augment(self, context) -> None:
        messages = context.get_messages()
        turn_text = _latest_user_text(messages)
        if not turn_text:
            return  # greeting directive or tool traffic — nothing to ground

        started = time.perf_counter()
        understanding = None
        speculative_hit = False
        if self._speculation is not None:
            understanding = await self._speculation.claim(turn_text)
            speculative_hit = understanding is not None
        if understanding is None:
            understanding = await self._understander.understand(
                turn_text, _dialogue_tail(messages)
            )
        t_understand = time.perf_counter() - started
        if self._speculation is not None:
            # Current turn becomes dialogue history for the next speculation.
            self._speculation.dialogue = _dialogue_tail(messages)

        plan = plan_turn(
            understanding,
            trailing_questions=_trailing_question_streak(messages),
            recent_modes=self._recent_modes,
            last_cited_source=self._last_cited_source,
        )

        results = []
        t_retrieve = 0.0
        if plan.include_passages:
            r0 = time.perf_counter()
            # Fetch extra, then cap per document: four chunks of the same NHS
            # page is one voice repeated, not four options.
            candidates = self._retriever.retrieve(
                understanding.query,
                emotion=understanding.emotion,
                top_k=self._top_k + 2,
                seen_doc_ids=frozenset(self._seen_docs),
            )
            per_doc: dict[str, int] = {}
            for result in candidates:
                doc = result.chunk.doc_id
                if per_doc.get(doc, 0) >= 2:
                    continue
                per_doc[doc] = per_doc.get(doc, 0) + 1
                results.append(result)
                if len(results) >= self._top_k:
                    break
            t_retrieve = time.perf_counter() - r0
            self._seen_docs.update(r.chunk.doc_id for r in results)
            self._last_cited_source = results[0].chunk.source if results else None

        quote = None
        if plan.mode in ("hold", "companion") and understanding.emotion.intensity < 0.8:
            from . import quotes as quotes_module

            quote = quotes_module.pick(
                self._quotes, understanding.emotion.state, self._used_quotes, self._rng
            )
            if quote is not None:
                self._used_quotes.add(quote.key())

        _replace_sentinel_message(context, self._render(plan, results, quote))
        self._recent_modes = (self._recent_modes + [plan.mode])[-4:]
        self._last_plan_notes = plan.notes

        self._turn += 1
        self._log_turn(
            turn_text, understanding, results, t_understand, t_retrieve, speculative_hit,
            mode=plan.mode,
        )

    def _render(self, plan: TurnPlan, results, quote=None) -> str:
        lines = [SENTINEL, f"Turn plan ({plan.mode}):"]
        lines.extend(f"- {note}" for note in plan.notes)

        if plan.mode == "guide":
            if results:
                blocks = []
                for i, result in enumerate(results, start=1):
                    chunk = result.chunk
                    blocks.append(f"({i}) From {chunk.source} — {chunk.title}:\n{chunk.text}")
                lines.append(
                    "- Ground any guidance in the passages below; pick the ONE that "
                    "actually fits, mention its source naturally in speech (once, "
                    "briefly, no URLs). If none genuinely fit, say you don't have "
                    "good guidance on that instead of stretching. Never present "
                    "knowledge beyond these passages as fact."
                )
                lines.append("")
                lines.append("\n\n".join(blocks))
            else:
                lines.append(
                    "- No reference passages matched. Say honestly that you don't "
                    "have good guidance on that — do not improvise wellbeing advice "
                    "or invent a source. Warm conversation is still fine."
                )
        else:
            lines.append(
                "- No reference passages this turn (none are needed — see the plan)."
            )

        if quote is not None:
            lines.append(
                f"- Optional, only if the moment truly invites it: you may close "
                f"with this thought, crediting {quote.author} in passing — "
                f"\"{quote.text}\" Skip it if it would feel pasted on."
            )
        return "\n".join(lines)

    def _log_turn(
        self, raw, understanding, results, t_understand, t_retrieve,
        speculative_hit=False, mode="guide",
    ) -> None:
        record = {
            "turn": self._turn,
            "raw": raw,
            "mode": mode,
            "plan_notes": list(getattr(self, "_last_plan_notes", [])),
            "speculative_hit": speculative_hit,
            "query": understanding.query,
            "emotion": {
                "state": understanding.emotion.state,
                "intensity": understanding.emotion.intensity,
                "confidence": understanding.emotion.confidence,
            },
            "wants_guidance": understanding.wants_guidance,
            "degraded": understanding.degraded,
            "understand_ms": round(t_understand * 1000, 1),
            "retrieve_ms": round(t_retrieve * 1000, 1),
            "passages": [r.explain() for r in results],
        }
        try:
            with self._log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning(f"could not write rag provenance: {exc}")
        if self._store is not None:
            self._store.log_turn(self._session_id, self._turn, record)
        cited = ", ".join(r.chunk.source for r in results) or "none"
        hit = " ⚡speculative" if speculative_hit else ""
        logger.info(
            f"🔎 turn {self._turn}: [{mode}] {understanding.emotion.state} · "
            f"understand {t_understand*1000:.0f}ms{hit} · retrieve {t_retrieve*1000:.0f}ms · "
            f"sources: {cited}"
        )


# ---- context helpers -------------------------------------------------------


def _latest_user_text(messages) -> str | None:
    """Text of the newest user message, only if it's the newest real turn.

    Non-dict entries are skipped everywhere in these helpers: the context can
    also hold LLMSpecificMessage objects (e.g. Gemini-3 thought signatures),
    which are provider plumbing, not conversation.
    """
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "user":
            return _text_of(message)
        if role == "assistant":
            return None  # newest turn is the bot's — nothing to ground
    return None


def _dialogue_tail(messages, limit: int = 6) -> list[str]:
    tail = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role not in ("user", "assistant"):
            continue
        text = _text_of(message)
        if text:
            tail.append(f"{role}: {text[:160]}")
    return tail[-limit:]


def _text_of(message) -> str | None:
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if isinstance(content, str):
        return content.strip() or None
    if isinstance(content, list):  # multimodal shape: list of parts
        parts = [p.get("text", "") for p in content if isinstance(p, dict)]
        joined = " ".join(p for p in parts if p).strip()
        return joined or None
    return None


def _replace_sentinel_message(context, rendered: str) -> None:
    """One passages slot, replaced every turn — never accumulated."""
    kept = [
        message
        for message in context.get_messages()
        if not (
            isinstance(message, dict)
            and message.get("role") == "developer"
            and isinstance(message.get("content"), str)
            and message["content"].startswith(SENTINEL)
        )
    ]
    kept.append({"role": "developer", "content": rendered})
    context.set_messages(kept)


def _clamp(value) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0
