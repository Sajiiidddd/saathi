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

from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import LLMRunFrame
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
from .config import Settings
from .latency import LatencyRecorder
from .persona import GREETING_DIRECTIVE, system_prompt


async def run_bot(webrtc_connection, settings: Settings | None = None) -> None:
    """Drive one conversation over an established WebRTC connection."""
    settings = settings or Settings.load()
    session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    logger.info(f"session {session_id} starting · {settings.redacted()}")

    directory = helplines_module.load(settings.helplines_path, settings.helpline_region)
    prompt = system_prompt(helplines_module.as_prompt_text(directory))

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

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            # ── agent core inserts here, in this order ───────────────────
            #   SafetyGate()   crisis -> deterministic handoff, bypass LLM
            #   ScopeGate()    diagnosis/meds -> canned decline
            #   RagProcessor() retrieve + attach cited passages
            # Gates go BEFORE the LLM on purpose: crisis handling must not
            # depend on retrieval quality or on the model behaving today.
            # ─────────────────────────────────────────────────────────────
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    latency_observer = UserBotLatencyObserver()
    recorder = LatencyRecorder(session_id, settings.log_dir)
    recorder.attach(latency_observer)

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            # Required for the per-service TTFB breakdown. Without this the
            # latency observer reports a total and nothing else.
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[latency_observer, TranscriptionLogObserver()],
    )

    @transport.event_handler("on_client_connected")
    async def _on_connected(_transport, _client):
        logger.info("caller connected — Saathi speaking first")
        # "developer" role seeds an instruction without putting words in the
        # caller's mouth. She opens the conversation so the caller isn't left
        # talking into silence wondering if the mic works.
        context.add_message({"role": "developer", "content": GREETING_DIRECTIVE})
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def _on_disconnected(_transport, _client):
        logger.info("caller disconnected")
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=False)
    try:
        await runner.add_workers(worker)
        await runner.run()
    finally:
        # Runs on hangup, error, and Ctrl-C alike, so no session's
        # measurements are lost.
        recorder.log_summary()
