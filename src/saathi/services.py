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


def _tune_segmentation(speech_config) -> None:
    """Shorten Azure's end-of-utterance silence window.

    Measured on real sessions: the final transcript lands ~1.0-1.2s after the
    caller stops speaking, and the turn-taking stop strategy waits for that
    transcript — so Azure's default segmentation silence (~650ms+) sits
    squarely on the voice-to-voice critical path. 350ms finalises noticeably
    faster; mid-sentence pauses splitting into two finals is fine here because
    the aggregator joins finals within a turn and the smart-turn model, not
    segmentation, decides when the caller is done.

    Reaches into _speech_config on the plain service (private attr, pinned
    pipecat 1.7.0) — the multilingual subclass calls this from its own init.
    """
    import os

    from azure.cognitiveservices.speech import PropertyId

    ms = (os.getenv("SAATHI_STT_SEGMENTATION_MS") or "350").strip()
    speech_config.set_property(PropertyId.Speech_SegmentationSilenceTimeoutMs, ms)


class MultilingualAzureSTTService(AzureSTTService):
    """Azure STT with continuous language identification across candidates.

    Pipecat's AzureSTTService fixes one recognition language when it builds
    the recognizer. The Azure SDK underneath supports auto-detection over up
    to 10 candidate languages, re-evaluated continuously — which is what lets
    a caller switch between English and Hindi mid-conversation (or mix them
    in one sentence, as Indian speech usually does).

    _connect() below restates the parent's stream setup because the parent
    gives no injection point for the auto-detect argument. Version-coupled to
    pipecat 1.7.0: if the pin moves, re-diff this against the parent method.
    """

    def __init__(self, *, candidate_languages: tuple[str, ...], **kwargs):
        super().__init__(**kwargs)
        self._candidate_languages = list(candidate_languages)

        from azure.cognitiveservices.speech import PropertyId

        # "Continuous" re-detects language throughout the stream. The default
        # ("AtStart") locks onto whatever language the first utterance used.
        self._speech_config.set_property(
            PropertyId.SpeechServiceConnection_LanguageIdMode, "Continuous"
        )
        _tune_segmentation(self._speech_config)

    async def _connect(self):
        if self._audio_stream:
            return
        try:
            from azure.cognitiveservices.speech import (
                AutoDetectSourceLanguageConfig,
                SpeechRecognizer,
            )
            from azure.cognitiveservices.speech.audio import (
                AudioStreamFormat,
                PushAudioInputStream,
            )
            from azure.cognitiveservices.speech.dialog import AudioConfig

            stream_format = AudioStreamFormat(samples_per_second=self.sample_rate, channels=1)
            self._audio_stream = PushAudioInputStream(stream_format)
            audio_config = AudioConfig(stream=self._audio_stream)

            self._speech_recognizer = SpeechRecognizer(
                speech_config=self._speech_config,
                audio_config=audio_config,
                auto_detect_source_language_config=AutoDetectSourceLanguageConfig(
                    languages=self._candidate_languages
                ),
            )
            self._speech_recognizer.recognizing.connect(self._on_handle_recognizing)
            self._speech_recognizer.recognized.connect(self._on_handle_recognized)
            self._speech_recognizer.canceled.connect(self._on_handle_canceled)
            self._speech_recognizer.start_continuous_recognition_async()
        except Exception as exc:  # mirror parent behaviour: report, don't raise
            await self.push_error(
                error_msg=f"Uncaught exception during initialization: {exc}", exception=exc
            )


def build_stt(settings: Settings):
    """Streaming speech-to-text.

    AzureSTTService wraps the Speech SDK's continuous recognition, so it is
    already streaming and emits interim results — there is no separate "live"
    class to reach for.
    """
    if SPEECH_PROVIDER == "azure":
        if len(settings.stt_languages) > 1:
            return MultilingualAzureSTTService(
                candidate_languages=settings.stt_languages,
                api_key=settings.azure_speech_key,
                region=settings.azure_speech_region,
                settings=AzureSTTService.Settings(language=settings.stt_language),
            )
        stt = AzureSTTService(
            api_key=settings.azure_speech_key,
            region=settings.azure_speech_region,
            settings=AzureSTTService.Settings(language=settings.stt_language),
        )
        _tune_segmentation(stt._speech_config)  # noqa: SLF001 — see helper docstring
        return stt

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

    ElevenLabs: set SAATHI_TTS_PROVIDER=elevenlabs plus ELEVENLABS_API_KEY and
    ELEVENLABS_VOICE_ID. Uses eleven_flash_v2_5 by default — their low-latency
    model, and one of the ones with Hindi support. STT stays Azure either way.
    No extra install needed: the service runs on aiohttp/websockets, already
    in the tree.
    """
    tts_provider = settings.tts_provider

    if tts_provider == "elevenlabs":
        api_key = (os.getenv("ELEVENLABS_API_KEY") or "").strip()
        # Language modes carry the voice id in tts_voice; the env var is the
        # fallback for a global elevenlabs default.
        voice_id = settings.tts_voice if settings.tts_voice.startswith(("en-", "hi-")) is False else ""
        voice_id = voice_id or (os.getenv("ELEVENLABS_VOICE_ID") or "").strip()
        if not api_key or not voice_id:
            raise ConfigError(
                "SAATHI_TTS_PROVIDER=elevenlabs needs ELEVENLABS_API_KEY and "
                "ELEVENLABS_VOICE_ID in .env. Key: elevenlabs.io -> profile -> "
                "API keys (free tier works). Voice id: pick a voice in their "
                "Voice Library and copy its ID."
            )
        from pipecat.services.elevenlabs.tts import ElevenLabsTTSService

        return ElevenLabsTTSService(
            api_key=api_key,
            voice_id=voice_id,
            settings=ElevenLabsTTSService.Settings(
                model=os.getenv("ELEVENLABS_MODEL") or "eleven_flash_v2_5",
                language=settings.tts_language,
            ),
        )

    if SPEECH_PROVIDER in ("azure", "deepgram"):
        # Even a Deepgram-STT setup keeps Azure TTS — Aura has no Hindi voice.
        tts_settings = dict(
            voice=settings.tts_voice,
            language=settings.tts_language,
            rate=settings.tts_rate,
        )
        # Style only when configured AND the voice supports it — sending a
        # style tag to a voice without that style makes Azure fall back to
        # neutral, or on some voices error out. DragonHD voices generate
        # their own prosody and take no style at all.
        if settings.tts_style:
            tts_settings["style"] = settings.tts_style
            tts_settings["style_degree"] = "1.3"
        return AzureTTSService(
            api_key=settings.azure_speech_key,
            region=settings.azure_speech_region,
            settings=AzureTTSService.Settings(**tts_settings),
        )

    raise ConfigError(f"Unknown SAATHI_SPEECH_PROVIDER '{SPEECH_PROVIDER}'.")


def build_llm(settings: Settings, system_prompt: str):
    """The conversation model, cascaded (STT -> LLM -> TTS).

    Two providers:
    - azure  -> Azure OpenAI deployment. Production-grade rate limits, billed
      against Azure credit, single-vendor story with the speech stack. The
      understander moves with it (see saathi.rag), removing the Gemini
      dependency entirely.
    - gemini -> Google AI Studio. Fine on paid tier; the free tier's
      20-requests/day/model wall makes it unfit for real testing.

    Deliberately NOT GeminiLiveLLMService (the native speech-to-speech model).
    Speech-to-speech is lower latency and would sound better, but it swallows
    the pipeline: no text turn to run a safety gate on, no place to inject
    retrieved passages, no per-stage latency to measure. The gate and the
    grounding are the point of this system, so the cascade wins the trade.

    Thinking is disabled/minimised explicitly: pipecat auto-disables it for
    the 2.5 series, but Gemini 3 models take thinking_level instead — left
    unset they burn the whole token budget on thoughts and return EMPTY text,
    which in a voice call is pure silence.
    """
    if settings.llm_provider == "openai":
        from pipecat.services.openai.llm import OpenAILLMService

        extra = {}
        if "gpt-oss" in settings.openai_model:
            # gpt-oss models reason before answering. Left at the default
            # effort they spend seconds thinking; "low" cuts a persona turn
            # to ~300ms on Groq and their reasoning stays out of `content`,
            # so nothing unspeakable reaches TTS.
            extra["reasoning_effort"] = "low"
        return OpenAILLMService(
            api_key=settings.openai_api_key,
            # base_url=None means api.openai.com; Groq is
            # https://api.groq.com/openai/v1 with a gsk_ key.
            base_url=settings.openai_base_url or None,
            settings=OpenAILLMService.Settings(
                model=settings.openai_model,
                system_instruction=system_prompt,
                temperature=settings.llm_temperature,
                max_tokens=300,
                extra=extra,
            ),
        )

    if settings.llm_provider == "azure":
        from pipecat.services.azure.llm import AzureLLMService

        return AzureLLMService(
            api_key=settings.azure_openai_key,
            endpoint=settings.azure_openai_endpoint,
            settings=AzureLLMService.Settings(
                model=settings.azure_openai_deployment,
                system_instruction=system_prompt,
                temperature=settings.llm_temperature,
                # A spoken turn is 2-4 sentences; backstop, not the control.
                max_tokens=300,
            ),
        )

    llm_settings = dict(
        model=settings.llm_model,
        system_instruction=system_prompt,
        temperature=settings.llm_temperature,
        # A spoken turn is 2-4 sentences. This is a backstop against a
        # runaway monologue, not the primary control (the prompt is).
        # Generous enough that a Gemini-3 model's minimal thinking plus the
        # actual reply both fit.
        max_tokens=500,
    )
    if settings.llm_model.startswith("gemini-3"):
        llm_settings["thinking"] = GoogleLLMService.ThinkingConfig(thinking_level="minimal")
    return GoogleLLMService(
        api_key=settings.google_api_key,
        settings=GoogleLLMService.Settings(**llm_settings),
    )
