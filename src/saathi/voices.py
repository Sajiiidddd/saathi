"""Curated voice options the caller can pick from on the talk page.

Every entry must satisfy two constraints:
1. Available in the configured Azure region (all of these are verified in
   centralindia — re-verify with scripts/check_keys.py after a region change).
2. Able to speak BOTH English and Hindi, because language mixing is a core
   feature and the voice must follow the caller wherever they go. This is why
   the classic en-IN voices (NeerjaNeural etc.) are absent: they are
   single-locale and would garble Devanagari.

The default entry mirrors .env's SAATHI_TTS_VOICE; the picker simply overrides
voice/style for one session. Style is only set for voices that support it —
DragonHD voices generate their own prosody and take no style tag.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VoiceOption:
    id: str
    voice: str
    label: str
    tagline: str
    style: str | None = None


CURATED: tuple[VoiceOption, ...] = (
    VoiceOption(
        id="aarti-hd",
        voice="en-IN-Aarti:DragonHDLatestNeural",
        label="Aarti",
        tagline="warm and natural — the default",
    ),
    VoiceOption(
        id="meera-hd",
        voice="en-IN-Meera:DragonHDLatestNeural",
        label="Meera",
        tagline="calm and unhurried",
    ),
    VoiceOption(
        id="diya-hd",
        voice="en-IN-Diya:DragonHDLatestNeural",
        label="Diya",
        tagline="bright and friendly",
    ),
    VoiceOption(
        id="neerja-hd",
        voice="en-IN-Neerja:DragonHDLatestNeural",
        label="Neerja",
        tagline="clear and steady",
    ),
    VoiceOption(
        id="arjun-hd",
        voice="en-IN-Arjun:DragonHDLatestNeural",
        label="Arjun",
        tagline="steady, male voice",
    ),
    VoiceOption(
        id="priya-soft",
        voice="hi-IN-Priya:MAI-Voice-2",
        label="Priya",
        tagline="soft-spoken and gentle",
        style="softvoice",
    ),
    VoiceOption(
        id="aarti-fast",
        voice="en-IN-AartiIndicNeural",
        label="Aarti Swift",
        tagline="quickest replies, a little less expressive",
    ),
)

_BY_ID = {option.id: option for option in CURATED}


def get(voice_id: str | None) -> VoiceOption | None:
    """Resolve a picker id to an option; None for unknown/missing ids so the
    caller falls back to the configured default rather than erroring a call."""
    if not voice_id:
        return None
    return _BY_ID.get(voice_id)


def as_api_list(default_voice: str) -> list[dict[str, str | bool]]:
    return [
        {
            "id": option.id,
            "label": option.label,
            "tagline": option.tagline,
            "default": option.voice == default_voice,
        }
        for option in CURATED
    ]
