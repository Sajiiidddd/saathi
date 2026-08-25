"""Opt-in caller memory — "Remember me between conversations."

Not model training: a small structured profile injected into the system
prompt, which gets the same "she knows me" effect while staying inspectable,
correctable, and deletable in one tap. It exists ONLY for callers who turned
the toggle on; everyone else remains fully anonymous, per the product's
privacy posture.

Profile shape (JSON, capped small — this is a sketch of a person, not a file
on them):
    name              what they asked to be called, if they offered it
    language_pref     how they like to mix languages
    recurring_topics  ["exam stress", "sleep"] — max 6
    tried             [{"technique": "...", "helped": "yes|no|unknown"}] — max 8
    avoid             things that landed badly ("dislikes breathing exercises")
    notes             one or two short free-text observations — max 2

The update runs once, at session end, from the session transcript — one LLM
call. A failed update silently keeps the old profile: memory is an
enhancement, never a gate.
"""

from __future__ import annotations

import asyncio
import json

from loguru import logger

UPDATE_TIMEOUT_SECS = 15.0

_UPDATE_PROMPT = """\
You maintain a small memory profile for a voice wellbeing companion's caller,
WITH the caller's explicit consent. Update it from this session's transcript.

Rules:
- Keep it small and useful: max 6 recurring_topics, max 8 tried entries,
  max 2 short notes. Drop the least useful old items to stay within caps.
- "tried" records techniques the companion suggested and how the caller
  responded ("helped": "yes", "no", or "unknown").
- "avoid" records what clearly landed badly.
- Record a name ONLY if the caller volunteered one.
- Never record: health conditions, diagnoses, medications, crisis content,
  third parties' personal details. This is a wellbeing sketch, not a record.

Return ONLY the updated JSON object with keys:
name (string or null), language_pref (string or null),
recurring_topics (list of strings), tried (list of objects),
avoid (list of strings), notes (list of strings).

Current profile:
{profile}

This session's transcript:
{transcript}
"""

EMPTY_PROFILE: dict = {
    "name": None,
    "language_pref": None,
    "recurring_topics": [],
    "tried": [],
    "avoid": [],
    "notes": [],
}


def clamp_profile(data: dict) -> dict:
    """Enforce the caps no matter what the model returned — and scrub.

    The profile is model-written text that gets re-injected into the system
    prompt next session, which makes it an injection channel: a caller could
    talk the extractor into storing "ignore your rules" as a memory. Every
    string runs through the same deterministic net as the live gate; anything
    that reads as an instruction to the model is silently dropped.
    """
    from .safety import check_injection

    def clean(value) -> bool:
        return not check_injection(str(value))

    safe = dict(EMPTY_PROFILE)
    if isinstance(data, dict):
        name = data.get("name")
        safe["name"] = (str(name)[:60]
                        if isinstance(name, str) and name.strip() and clean(name) else None)
        pref = data.get("language_pref")
        safe["language_pref"] = (str(pref)[:80]
                                 if isinstance(pref, str) and pref.strip() and clean(pref) else None)
        safe["recurring_topics"] = [str(t)[:60] for t in (data.get("recurring_topics") or [])
                                    if clean(t)][:6]
        safe["tried"] = [
            {
                "technique": str(t.get("technique"))[:80],
                "helped": t.get("helped") if t.get("helped") in ("yes", "no", "unknown") else "unknown",
            }
            for t in (data.get("tried") or [])
            if isinstance(t, dict) and t.get("technique") and clean(t.get("technique"))
        ][:8]
        safe["avoid"] = [str(a)[:80] for a in (data.get("avoid") or []) if clean(a)][:4]
        safe["notes"] = [str(n)[:160] for n in (data.get("notes") or []) if clean(n)][:2]
    return safe


def as_prompt_text(profile: dict) -> str:
    """Render the profile for the system prompt. Skipped entirely when empty."""
    parts: list[str] = []
    if profile.get("name"):
        parts.append(f"They asked to be called {profile['name']}.")
    if profile.get("language_pref"):
        parts.append(f"Language habit: {profile['language_pref']}.")
    if profile.get("recurring_topics"):
        parts.append("Comes up often: " + ", ".join(profile["recurring_topics"]) + ".")
    for item in profile.get("tried", []):
        verdict = {"yes": "it helped", "no": "it did not help"}.get(item["helped"], "unclear if it helped")
        parts.append(f"Previously tried {item['technique']} — {verdict}.")
    if profile.get("avoid"):
        parts.append("Landed badly before: " + "; ".join(profile["avoid"]) + ".")
    parts.extend(profile.get("notes", []))
    if not parts:
        return ""
    body = "\n".join(f"- {p}" for p in parts)
    return (
        "What you remember about this caller from earlier conversations (they "
        "chose to be remembered):\n"
        f"{body}\n"
        "Use this the way a friend uses memory — naturally, never recited back "
        "as a list, never referred to as notes or data. Don't re-suggest what "
        "didn't help. If they contradict a memory, trust them, not the memory."
    )


class ProfileUpdater:
    """One end-of-session LLM call over any OpenAI-compatible endpoint."""

    def __init__(self, api_key: str, model: str, base_url: str | None = None):
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url or None)
        self._model = model

    async def update(self, old_profile: dict | None, transcript: list[tuple[str, str]]) -> dict | None:
        """transcript: [(role, text), ...]. Returns the new profile, or None
        to keep the old one (on failure or an empty session)."""
        user_turns = [t for role, t in transcript if role == "user"]
        if len(user_turns) < 2:
            return None  # nothing meaningful to learn from

        lines = "\n".join(f"{role}: {text}" for role, text in transcript[-60:])
        prompt = _UPDATE_PROMPT.format(
            profile=json.dumps(old_profile or EMPTY_PROFILE, ensure_ascii=False),
            transcript=lines[:8000],
        )
        kwargs = {"reasoning_effort": "low"} if "gpt-oss" in self._model else {}
        try:
            response = await asyncio.wait_for(
                self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=600,
                    response_format={"type": "json_object"},
                    **kwargs,
                ),
                timeout=UPDATE_TIMEOUT_SECS,
            )
            return clamp_profile(json.loads(response.choices[0].message.content))
        except Exception as exc:  # noqa: BLE001 — keep the old profile
            logger.warning(f"profile update skipped ({type(exc).__name__}: {exc})")
            return None
