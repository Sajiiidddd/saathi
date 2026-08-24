"""Speech + LLM service factory.

The point of this file is that the provider is a config value, not a rewrite.
The stack decision (Azure Speech, chosen for its Hindi and Indian-English
neural voices) is a decision, not a constraint — if Azure's latency
disappoints under load, swapping STT is one env var and one branch here, and
nothing in bot.py changes.

Note on constructor style: Pipecat 1.x moved per-service options out of
top-level kwargs into `Service.Settings(...)`. Passing voice=/language=/model=
directly still works but is deprecated and slated for removal in 2.0, so
everything below uses `settings=`.
"""

from __future__ import annotations

import os

from pipecat.services.azure.stt import AzureSTTService
from pipecat.services.azure.tts import AzureTTSService
from pipecat.services.google.llm import GoogleLLMService

from .config import ConfigError, Settings

SPEECH_PROVIDER = (os.getenv("SAATHI_SPEECH_PROVIDER") or "azure").lower()


def build_stt(settings: Settings):
    """Streaming speech-to-text.

    AzureSTTService wraps the Speech SDK's continuous recognition, so it is
    already streaming and emits interim results — there is no separate "live"
    class to reach for.
    """
    if SPEECH_PROVIDER == "azure":
        return AzureSTTService(
            api_key=settings.azure_speech_key,
            region=settings.azure_speech_region,
            settings=AzureSTTService.Settings(language=settings.stt_language),
        )

    if SPEECH_PROVIDER == "deepgram":
        # Left as an explicit branch rather than dead code: install the extra
        # (`pipecat-ai[deepgram]`), import DeepgramSTTService here, and set
        # SAATHI_SPEECH_PROVIDER=deepgram. Verify the constructor against your
        # installed version first — this signature is NOT verified, unlike the
        # Azure one above.
        raise ConfigError(
            "SAATHI_SPEECH_PROVIDER=deepgram is a documented extension point, not "
            "wired up. Add the deepgram extra and implement this branch, or unset "
            "the variable to use Azure. Note Deepgram's Aura TTS is English-only, "
            "which rules out Hindi voice support."
        )

    raise ConfigError(
        f"Unknown SAATHI_SPEECH_PROVIDER '{SPEECH_PROVIDER}'. Expected 'azure' or 'deepgram'."
    )


def build_tts(settings: Settings):
    """Streaming text-to-speech.

    AzureTTSService is the WebSocket streaming variant — audio chunks come back
    as they are synthesised, which is what keeps first-audio latency low.
    AzureHttpTTSService exists but buffers the whole utterance; do not use it
    in a live call.
    """
    if SPEECH_PROVIDER in ("azure", "deepgram"):
        # Even a Deepgram-STT setup keeps Azure TTS — Aura has no Hindi voice.
        return AzureTTSService(
            api_key=settings.azure_speech_key,
            region=settings.azure_speech_region,
            settings=AzureTTSService.Settings(
                voice=settings.tts_voice,
                language=settings.tts_language,
                # Slightly under normal pace — she is talking to someone who
                # may be stressed. The crisis handoff additionally gets softer
                # prosody via SSML.
                rate="0.95",
            ),
        )

    raise ConfigError(f"Unknown SAATHI_SPEECH_PROVIDER '{SPEECH_PROVIDER}'.")


def build_llm(settings: Settings, system_prompt: str):
    """Gemini Flash, cascaded (STT -> LLM -> TTS).

    Deliberately NOT GeminiLiveLLMService (the native speech-to-speech model).
    Speech-to-speech is lower latency and would sound better, but it swallows
    the pipeline: no text turn to run a safety gate on, no place to inject
    retrieved passages, no per-stage latency to measure. The gate and the
    grounding are the point of this system, so the cascade wins the trade.

    Pipecat disables Gemini 2.5's thinking budget by default to cut latency,
    which is what we want for a voice turn.
    """
    return GoogleLLMService(
        api_key=settings.google_api_key,
        settings=GoogleLLMService.Settings(
            model=settings.llm_model,
            system_instruction=system_prompt,
            temperature=settings.llm_temperature,
            # A spoken turn is 2-4 sentences. This is a backstop against a
            # runaway monologue, not the primary control (the prompt is).
            max_tokens=300,
        ),
    )
