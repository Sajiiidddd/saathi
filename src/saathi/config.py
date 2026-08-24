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

    tts_voice: str
    tts_language: Language
    stt_language: Language

    llm_model: str
    llm_temperature: float

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

        return cls(
            azure_speech_key=_required(
                "AZURE_SPEECH_API_KEY",
                "Azure portal > your Speech resource > Keys and Endpoint > KEY 1",
            ),
            azure_speech_region=_required(
                "AZURE_SPEECH_REGION",
                "Same page, 'Location/Region'. Use the slug (centralindia), not 'Central India'",
            ),
            google_api_key=_required(
                "GOOGLE_API_KEY",
                "https://aistudio.google.com/apikey (free tier, no card)",
            ),
            tts_voice=os.getenv("SAATHI_TTS_VOICE") or "en-IN-NeerjaNeural",
            tts_language=_language(os.getenv("SAATHI_TTS_LANGUAGE") or "en-IN"),
            stt_language=_language(os.getenv("SAATHI_STT_LANGUAGE") or "en-IN"),
            llm_model=os.getenv("SAATHI_LLM_MODEL") or "gemini-2.5-flash",
            llm_temperature=_float("SAATHI_LLM_TEMPERATURE", 0.4),
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
            "stt_language": str(self.stt_language),
            "llm_model": self.llm_model,
            "vad_stop_secs": self.vad_stop_secs,
        }
