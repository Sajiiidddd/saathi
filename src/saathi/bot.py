"""Voice pipeline: browser mic -> Azure STT -> Gemini Flash -> Azure TTS.

Latency instrumentation is wired in from the first version rather than
retrofitted: a quoted voice-to-voice number is only credible if every session
ever run was measured the same way.

Pipeline order is not arbitrary. In Pipecat 1.x the assistant aggregator sits
*after* transport.output(), so context is written from what was actually spoken
rather than what the LLM planned to say — if the caller barges in halfway
through a sentence, the transcript reflects the interruption instead of
pretending the full reply happened.
"""

from __future__ import annotations

from datetime import datetime

import time

from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    LLMRunFrame,
    TTSSpeakFrame,
    TTSTextFrame,
    TranscriptionFrame,
)
from pipecat.observers.base_observer import BaseObserver
from pipecat.observers.loggers.transcription_log_observer import TranscriptionLogObserver
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.workers.runner import WorkerRunner

from . import helplines as helplines_module
from . import languages as languages_module
from .captions import CaptionPublisher
from .config import Settings, language_from_code
from .safety import SafetyGate
from .latency import LatencyRecorder
from .persona import GREETING_DIRECTIVE, system_prompt
from .rag import QueryUnderstander, RagProcessor, SpeculationState, Speculator


async def run_bot(
    webrtc_connection,
    settings: Settings | None = None,
    language=None,  # languages.LanguageMode; None -> configured defaults
    retriever=None,
    understander=None,  # any *Understander (duck-typed)
    quotes=None,
    store=None,  # store.SessionStore
    device_id: str | None = None,
    remember: bool = False,
    profile_updater=None,  # profile.ProfileUpdater
) -> None:
    """Drive one conversation over an established WebRTC connection.

    language is the caller's pick from the talk page — it sets the voice,
    the STT candidate languages, and how the persona is told to respond, for
    this call only. retriever/understander/store are shared across sessions;
    each session gets its own RagProcessor so document memory — "already
    suggested box breathing" — stays per call.
    """
    settings = settings or Settings.load()
    greeting = GREETING_DIRECTIVE
    if language is not None:
        from dataclasses import replace

        settings = replace(
            settings,
            tts_provider=language.tts_provider,
            tts_voice=language.tts_voice,
            tts_style=None,
            tts_language=language_from_code(language.tts_language),
            stt_language=language_from_code(language.stt_languages[0]),
            stt_languages=tuple(language.stt_languages),
        )
        greeting = language.greeting_directive
    session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    logger.info(f"session {session_id} starting · {settings.redacted()}")
    if store is not None:
        store.start_session(
            session_id,
            language.id if language else "default",
            settings.tts_provider,
            settings.tts_voice,
        )

    directory = helplines_module.load(settings.helplines_path, settings.helpline_region)
    prompt = system_prompt(helplines_module.as_prompt_text(directory))
    if language is not None:
        prompt = f"{prompt}\n\n{language.persona_directive}"

    # Opt-in memory: injected only when the caller turned "remember me" on.
    remembering = bool(remember and device_id and store is not None)
    if remembering:
        from . import profile as profile_module

        existing = store.get_profile(device_id)
        if existing:
            memory_text = profile_module.as_prompt_text(existing)
            if memory_text:
                prompt = f"{prompt}\n\n{memory_text}"
                logger.info(f"profile loaded for device …{device_id[-6:]}")

    # Lazy import so a bad provider config fails here with our error message
    # rather than at module import time.
    from .services import build_llm, build_stt, build_tts

    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(audio_in_enabled=True, audio_out_enabled=True),
    )

    stt = build_stt(settings)
    tts = build_tts(settings)
    llm = build_llm(settings, prompt)

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            # VAD lives on the aggregator now, not the transport — moved in
            # Pipecat 1.0. Barge-in is on by default, and the default stop
            # strategy is LocalSmartTurnAnalyzerV3, so we get semantic
            # end-of-turn detection (it waits through "I'm feeling... um...")
            # for free rather than cutting off on raw silence.
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(stop_secs=settings.vad_stop_secs)
            ),
        ),
    )

    # The gate exists on EVERY pipeline, grounded or not — crisis handling
    # must never depend on which optional components loaded.
    gate = SafetyGate(
        session_id=session_id,
        language_id=language.id if language else "hinglish",
        helplines=directory,
        store=store,
    )

    if retriever is not None and understander is not None:
        # The speculator runs the understander on STT finals while turn-taking
        # is still deciding the caller has finished, so the ~900ms model call
        # overlaps the natural end-of-turn pause instead of following it.
        speculation = SpeculationState()
        pre_aggregator = [Speculator(speculation, understander)]
        rag = RagProcessor(
            retriever=retriever,
            understander=understander,
            session_id=session_id,
            log_dir=settings.log_dir,
            top_k=settings.rag_top_k,
            speculation=speculation,
            quotes=quotes,
            store=store,
        )
        agent_core = [gate, rag]
    else:
        # Standalone/dev runs without an index still get a conversation; the
        # persona prompt forbids invented citations either way.
        logger.warning("no retriever provided — running UNGROUNDED")
        pre_aggregator = []
        agent_core = [gate]

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            CaptionPublisher(),  # live user captions: interims as they form
            *pre_aggregator,
            user_aggregator,
            # ── agent core (order matters) ───────────────────────────────
            #   SafetyGate    crisis -> deterministic handoff, LLM bypassed;
            #                 scope  -> binding decline directive
            #   RagProcessor  turn planner + retrieve + cited passages
            # The gate is BEFORE retrieval and the LLM on purpose: crisis
            # handling must not depend on retrieval quality or on the model
            # behaving today.
            # ─────────────────────────────────────────────────────────────
            *agent_core,
            llm,
            tts,
            CaptionPublisher(),  # live bot captions: her words as she says them
            transport.output(),
            assistant_aggregator,
        ]
    )

    latency_observer = UserBotLatencyObserver()
    recorder = LatencyRecorder(session_id, settings.log_dir, store=store)
    recorder.attach(latency_observer)

    observers = [latency_observer, TranscriptionLogObserver()]
    if store is not None:
        observers.append(_TranscriptRecorder(store, session_id))

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            # Required for the per-service TTFB breakdown. Without this the
            # latency observer reports a total and nothing else.
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=observers,
    )

    @transport.event_handler("on_client_connected")
    async def _on_connected(_transport, _client):
        logger.info("caller connected — Saathi speaking first")
        # "developer" role seeds an instruction without putting words in the
        # caller's mouth. She opens the conversation so the caller isn't left
        # talking into silence wondering if the mic works.
        context.add_message({"role": "developer", "content": greeting})
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def _on_disconnected(_transport, _client):
        logger.info("caller disconnected")
        await worker.cancel()

    # A voice agent must never fail into dead air. If the LLM errors (a 429
    # under free-tier quota being the proven case), TTS is usually still
    # alive — so say so. Debounced: repeated errors within the window get one
    # apology, not a stutter of them.
    last_apology = 0.0

    @worker.event_handler("on_pipeline_error")
    async def _on_error(worker_ref, error_frame):
        nonlocal last_apology
        message = str(getattr(error_frame, "error", error_frame))[:160]
        logger.error(f"pipeline error surfaced to caller-handler: {message}")
        now = time.monotonic()
        if now - last_apology < 15.0:
            return
        last_apology = now
        try:
            await worker_ref.queue_frames(
                [
                    TTSSpeakFrame(
                        "Sorry — I'm having a little trouble on my side right now. "
                        "Give me a moment, or try me again in a bit."
                    )
                ]
            )
        except Exception as exc:  # noqa: BLE001 — apology must never crash the call
            logger.warning(f"could not speak the error apology: {exc}")

    runner = WorkerRunner(handle_sigint=False)
    try:
        await runner.add_workers(worker)
        await runner.run()
    finally:
        # Runs on hangup, error, and Ctrl-C alike, so no session's
        # measurements are lost.
        recorder.log_summary()
        if store is not None:
            store.end_session(session_id)
        if remembering and profile_updater is not None:
            try:
                transcript = store.query(
                    "SELECT role, text FROM utterances WHERE session_id = ? ORDER BY ts",
                    (session_id,),
                )
                updated = await profile_updater.update(store.get_profile(device_id), transcript)
                if updated is not None:
                    store.save_profile(device_id, updated)
                    logger.info(f"profile updated for device …{device_id[-6:]}")
            except Exception as exc:  # noqa: BLE001 — memory is never worth an error
                logger.warning(f"profile update failed: {exc}")


def join_spoken(words: list[str]) -> str:
    """Assemble TTS word-boundary fragments into a sentence.

    Azure's streaming TTS reports text word by word; naive joining produces
    'kinda " कोड "' spacing artifacts, so punctuation is re-attached.
    """
    import re

    text = " ".join(w.strip() for w in words if w and w.strip())
    text = re.sub(r"\s+([,.!?;:।…])", r"\1", text)
    text = re.sub(r"([‘“(])\s+", r"\1", text)
    return re.sub(r"\s{2,}", " ", text).strip()


class _TranscriptRecorder(BaseObserver):
    """Observer that persists both sides of the conversation as text.

    Bot text arrives as ONE WORD PER FRAME (Azure word boundaries), so it is
    buffered and flushed as a full utterance when the bot stops speaking —
    otherwise the transcript reads one word per line. Frames pass every
    pipeline edge, so dedupe by frame id. Audio is never touched.
    """

    def __init__(self, store, session_id: str, **kwargs):
        super().__init__(**kwargs)
        self._store = store
        self._session_id = session_id
        self._seen: set[int] = set()
        self._bot_words: list[str] = []

    def _flush_bot(self) -> None:
        if self._bot_words:
            self._store.log_utterance(self._session_id, "bot", join_spoken(self._bot_words))
            self._bot_words = []

    async def on_push_frame(self, data):
        frame = data.frame
        if frame.id in self._seen:
            return
        if isinstance(frame, TranscriptionFrame):
            self._seen.add(frame.id)
            self._flush_bot()  # a user final mid-stream means she was interrupted
            self._store.log_utterance(self._session_id, "user", frame.text or "")
        elif isinstance(frame, TTSTextFrame):
            self._seen.add(frame.id)
            if frame.text:
                self._bot_words.append(frame.text)
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._seen.add(frame.id)
            self._flush_bot()
        if len(self._seen) > 4000:
            self._seen.clear()
