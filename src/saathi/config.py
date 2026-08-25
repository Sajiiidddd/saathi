"""Runtime configuration, loaded once from the environment (.env).

Fails fast and loudly. A voice demo that starts up and *then* dies mid-call
because a key was missing is worse than one that refuses to boot.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from pipecat.transcriptions.language import Language

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env", override=True)

# SSL_CERT_FILE is only needed behind TLS-intercepting corporate proxies, and
# it must point at a real file: httpx hard-fails on a missing bundle. A .env
# carried from one machine to another shouldn't take every HTTPS call with it.
_cert = os.environ.get("SSL_CERT_FILE")
if _cert and not Path(_cert).exists():
    os.environ.pop("SSL_CERT_FILE", None)


class ConfigError(RuntimeError):
    """Raised at startup when required configuration is missing or wrong."""


def _required(name: str, hint: str) -> str:
    value = (os.getenv(name) or "").strip()
    if not value:
        raise ConfigError(
            f"{name} is not set.\n"
            f"  -> {hint}\n"
            f"  -> Add it to {PROJECT_ROOT / '.env'} (copy .env.example if you haven't)."
        )
    return value


def language_from_code(code: str) -> Language:
    """Public alias — language modes resolve their BCP-47 codes through this."""
    return _language(code)


def _language(code: str) -> Language:
    """Resolve a BCP-47 code (e.g. 'en-IN') to Pipecat's Language enum.

    Tried by value first, then by the underscored member name, because the
    enum spells members EN_IN but carries values like 'en-IN'.
    """
    code = code.strip()
    try:
        return Language(code)
    except ValueError:
        pass
    member = code.replace("-", "_").upper()
    if hasattr(Language, member):
        return getattr(Language, member)
    raise ConfigError(
        f"'{code}' is not a language Pipecat recognises. Expected something like "
        f"en-IN, en-US or hi-IN. (Looked for value '{code}' and member '{member}'.)"
    )


def _float(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ConfigError(f"{name} must be a number, got '{raw}'.")


@dataclass(frozen=True)
class Settings:
    """Everything Saathi needs to boot, resolved and validated."""

    azure_speech_key: str
    azure_speech_region: str
    google_api_key: str

    tts_provider: str  # "azure" | "elevenlabs" — per-session overridable
    tts_voice: str
    tts_style: str | None
    tts_rate: str
    tts_language: Language
    stt_language: Language
    # BCP-47 codes for STT language auto-detection (e.g. ["en-IN", "hi-IN"]).
    # One entry = fixed-language recognition; several = continuous language
    # identification, which is what enables mid-conversation code-switching.
    stt_languages: tuple[str, ...]

    llm_model: str
    llm_temperature: float

    # The cheap fast model behind query rewriting + emotion classification.
    classifier_model: str
    index_dir: Path
    rag_top_k: int

    # LLM provider:
    #   "gemini" — Google AI Studio (free tier: 20 req/day/model)
    #   "azure"  — Azure OpenAI (blocked on Azure-for-Students subscriptions)
    #   "openai" — any OpenAI-compatible endpoint. With base_url
    #              https://models.github.ai/inference and a GitHub token this
    #              is GitHub Models: free, no card, ~150 req/day on minis.
    # Whatever is chosen runs the main model AND the understander.
    llm_provider: str
    azure_openai_endpoint: str
    azure_openai_key: str
    azure_openai_deployment: str
    openai_api_key: str
    openai_base_url: str
    openai_model: str

    vad_stop_secs: float

    helpline_region: str
    log_dir: Path
    host: str
    port: int

    # Read by the crisis handoff, the system prompt, and the landing page.
    helplines_path: Path = field(default=PROJECT_ROOT / "data" / "helplines.json")

    @classmethod
    def load(cls) -> "Settings":
        log_dir = PROJECT_ROOT / (os.getenv("SAATHI_LOG_DIR") or "logs")
        log_dir.mkdir(parents=True, exist_ok=True)

        # SAATHI_STT_LANGUAGES (comma-separated) is canonical; the older
        # singular SAATHI_STT_LANGUAGE still works for one language.
        raw_langs = os.getenv("SAATHI_STT_LANGUAGES") or os.getenv("SAATHI_STT_LANGUAGE") or "en-IN"
        stt_langs = [code.strip() for code in raw_langs.split(",") if code.strip()]
        for code in stt_langs:
            _language(code)  # validate each; raises ConfigError with the bad code
        if not 1 <= len(stt_langs) <= 10:
            raise ConfigError(
                f"SAATHI_STT_LANGUAGES has {len(stt_langs)} entries; Azure language "
                f"identification supports 1-10 candidate languages."
            )

        llm_provider = (os.getenv("SAATHI_LLM_PROVIDER") or "gemini").lower()
        if llm_provider not in ("gemini", "azure", "openai"):
            raise ConfigError(
                f"SAATHI_LLM_PROVIDER must be 'gemini', 'azure' or 'openai', got '{llm_provider}'."
            )

        openai_api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        openai_base_url = (os.getenv("OPENAI_BASE_URL") or "").strip()
        openai_model = (os.getenv("OPENAI_MODEL") or "openai/gpt-4o-mini").strip()
        if llm_provider == "openai" and not openai_api_key:
            raise ConfigError(
                "SAATHI_LLM_PROVIDER=openai needs OPENAI_API_KEY.\n"
                "  -> For GitHub Models (free): github.com > Settings > Developer settings >\n"
                "     Personal access tokens > Fine-grained > enable 'Models' read permission,\n"
                "     then set OPENAI_BASE_URL=https://models.github.ai/inference"
            )

        if llm_provider == "azure":
            azure_openai_endpoint = _required(
                "AZURE_OPENAI_ENDPOINT",
                "Azure AI Foundry > your project > the deployment's 'Target URI' base, "
                "e.g. https://YOUR-RESOURCE.openai.azure.com/",
            )
            azure_openai_key = _required(
                "AZURE_OPENAI_API_KEY", "Same page as the endpoint — Keys"
            )
            azure_openai_deployment = _required(
                "AZURE_OPENAI_DEPLOYMENT",
                "The DEPLOYMENT NAME you chose when deploying the model (not the model id)",
            )
            google_api_key = (os.getenv("GOOGLE_API_KEY") or "").strip()
        else:
            azure_openai_endpoint = (os.getenv("AZURE_OPENAI_ENDPOINT") or "").strip()
            azure_openai_key = (os.getenv("AZURE_OPENAI_API_KEY") or "").strip()
            azure_openai_deployment = (os.getenv("AZURE_OPENAI_DEPLOYMENT") or "").strip()
            if llm_provider == "gemini":
                google_api_key = _required(
                    "GOOGLE_API_KEY",
                    "https://aistudio.google.com/apikey (free tier, no card) — or set "
                    "SAATHI_LLM_PROVIDER=openai for GitHub Models (free, 150 req/day)",
                )
            else:
                google_api_key = (os.getenv("GOOGLE_API_KEY") or "").strip()

        return cls(
            azure_speech_key=_required(
                "AZURE_SPEECH_API_KEY",
                "Azure portal > your Speech resource > Keys and Endpoint > KEY 1",
            ),
            azure_speech_region=_required(
                "AZURE_SPEECH_REGION",
                "Same page, 'Location/Region'. Use the slug (centralindia), not 'Central India'",
            ),
            google_api_key=google_api_key,
            tts_provider=(os.getenv("SAATHI_TTS_PROVIDER") or "azure").lower(),
            tts_voice=os.getenv("SAATHI_TTS_VOICE") or "en-IN-NeerjaNeural",
            tts_style=(os.getenv("SAATHI_TTS_STYLE") or "").strip() or None,
            tts_rate=(os.getenv("SAATHI_TTS_RATE") or "1.0").strip(),
            tts_language=_language(os.getenv("SAATHI_TTS_LANGUAGE") or "en-IN"),
            stt_language=_language(stt_langs[0]),
            stt_languages=tuple(stt_langs),
            llm_model=os.getenv("SAATHI_LLM_MODEL") or "gemini-2.5-flash",
            llm_temperature=_float("SAATHI_LLM_TEMPERATURE", 0.4),
            classifier_model=os.getenv("SAATHI_CLASSIFIER_MODEL") or "gemini-2.5-flash-lite",
            index_dir=PROJECT_ROOT / (os.getenv("SAATHI_INDEX_DIR") or "index"),
            rag_top_k=int(os.getenv("SAATHI_RAG_TOP_K") or 4),
            llm_provider=llm_provider,
            azure_openai_endpoint=azure_openai_endpoint,
            azure_openai_key=azure_openai_key,
            azure_openai_deployment=azure_openai_deployment,
            openai_api_key=openai_api_key,
            openai_base_url=openai_base_url,
            openai_model=openai_model,
            vad_stop_secs=_float("SAATHI_VAD_STOP_SECS", 0.4),
            helpline_region=(os.getenv("SAATHI_HELPLINE_REGION") or "IN").upper(),
            log_dir=log_dir,
            host=os.getenv("SAATHI_HOST") or "localhost",
            port=int(os.getenv("SAATHI_PORT") or 7860),
        )

    def redacted(self) -> dict[str, object]:
        """Safe-to-log view — used in the startup banner."""

        def tail(secret: str) -> str:
            return f"…{secret[-4:]} ({len(secret)} chars)" if secret else "MISSING"

        return {
            "azure_region": self.azure_speech_region,
            "azure_key": tail(self.azure_speech_key),
            "google_key": tail(self.google_api_key),
            "tts_voice": self.tts_voice,
            "tts_style": self.tts_style,
            "tts_rate": self.tts_rate,
            "stt_languages": list(self.stt_languages),
            "llm_provider": self.llm_provider,
            "llm_model": {
                "azure": self.azure_openai_deployment,
                "openai": self.openai_model,
            }.get(self.llm_provider, self.llm_model),
            "classifier_model": {
                "azure": self.azure_openai_deployment,
                "openai": self.openai_model,
            }.get(self.llm_provider, self.classifier_model),
            "vad_stop_secs": self.vad_stop_secs,
        }
