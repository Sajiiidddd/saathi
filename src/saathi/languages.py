"""Language modes — the caller chooses a language, not a voice.

A voice list asks the caller to care about our vendor catalogue. A language
choice is a real preference. Each mode bundles everything language touches:
which voice speaks, which languages STT listens for, how the persona is told
to respond, and how the greeting opens.

Voice notes:
- hindi/hinglish ride Azure's Aarti DragonHD — Hindi-native, effectively free
  (500k chars/month on the Speech free tier).
- english uses ElevenLabs Sarah per product choice. Mind the free tier:
  ~10k chars/month is roughly ONE long session. If English goes silent
  mid-month, that's the quota — swap the spec below to the Azure line kept
  in the comment (en-GB Ada HD, same family as Aarti) and restart.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LanguageMode:
    id: str
    label: str
    tagline: str
    stt_languages: tuple[str, ...]
    tts_provider: str  # "azure" | "elevenlabs"
    tts_voice: str  # azure voice shortname, or elevenlabs voice id
    tts_language: str  # BCP-47 hint for the synthesiser
    persona_directive: str
    greeting_directive: str


MODES: tuple[LanguageMode, ...] = (
    LanguageMode(
        id="hinglish",
        label="Hinglish",
        # Romanised on purpose: in Hinglish mode the UI shows no pure
        # Devanagari (finalised language rules — the design handoff, §3).
        tagline="mix mein baat karo — the way you actually talk",
        stt_languages=("en-IN", "hi-IN"),
        tts_provider="azure",
        tts_voice="en-IN-Aarti:DragonHDLatestNeural",
        tts_language="en-IN",
        persona_directive=(
            "Language: mirror the caller's mix of Hindi and English exactly — "
            "their balance, not yours. Hindi words in Devanagari, English words "
            "in Latin script. Never switch languages on them unprompted."
        ),
        greeting_directive=(
            "Greet the caller in one short English sentence, then one short "
            "natural Hindi sentence (Devanagari) saying they can talk in "
            "whatever mix feels right. Ask how they're doing."
        ),
    ),
    LanguageMode(
        id="hindi",
        label="हिंदी",
        tagline="पूरी बातचीत हिंदी में · the whole conversation in Hindi",
        stt_languages=("hi-IN", "en-IN"),  # hi first: code-mixed asides still land
        tts_provider="azure",
        tts_voice="en-IN-Aarti:DragonHDLatestNeural",
        tts_language="hi-IN",
        persona_directive=(
            "Language: respond in Hindi, written in Devanagari. Common English "
            "loanwords the caller uses (exam, stress, office) may stay in Latin "
            "script inside the Hindi sentence. Do not switch to English unless "
            "the caller clearly does."
        ),
        greeting_directive=(
            "Greet the caller warmly in Hindi (Devanagari), say your name is "
            "Saathi, and ask how they are feeling right now. One or two short "
            "sentences."
        ),
    ),
    LanguageMode(
        id="english",
        label="English",
        tagline="calm, clear English",
        stt_languages=("en-IN",),
        # Same voice as the other modes on purpose: Saathi is ONE person who
        # speaks three ways, and Aarti HD's Indian English fits who she is.
        # Alternatives if the product ever wants a different English flavour:
        #   UK:  tts_provider="azure", tts_voice="en-GB-Ada:DragonHDLatestNeural"
        #   EL:  tts_provider="elevenlabs", tts_voice="EXAVITQu4vr4xnSDxMaL"
        #        (Sarah — mind the free tier's ~10k chars/month)
        tts_provider="azure",
        tts_voice="en-IN-Aarti:DragonHDLatestNeural",
        tts_language="en-IN",
        persona_directive=(
            "Language: respond in English only, plain and warm. If the caller "
            "drops in a Hindi word, understand it but keep your reply English."
        ),
        greeting_directive=(
            "Greet the caller in one or two sentences of English. Say your name "
            "is Saathi and ask how they're doing right now. Do not list your "
            "capabilities. Do not mention being an AI."
        ),
    ),
)

DEFAULT = MODES[0]
_BY_ID = {mode.id: mode for mode in MODES}


def get(mode_id: str | None) -> LanguageMode:
    """Unknown/missing ids fall back to the default — a stale picker value
    must never cost someone the call."""
    return _BY_ID.get((mode_id or "").strip().lower(), DEFAULT)


def as_api_list() -> list[dict[str, str | bool]]:
    return [
        {
            "id": mode.id,
            "label": mode.label,
            "tagline": mode.tagline,
            "default": mode is DEFAULT,
        }
        for mode in MODES
    ]
