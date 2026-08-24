"""Loader for data/helplines.json — the crisis-handoff directory.

This is not retrieval content and must never go through the RAG lane. It is a
fixed list read by the safety gate, injected into the system prompt, and
printed on the landing page. One source of truth, because these are numbers a
distressed person might actually dial.
"""

from __future__ import annotations

import json
from pathlib import Path


class Helpline:
    __slots__ = ("name", "phone", "alt", "web")

    def __init__(self, name: str, phone: str, alt: str | None = None, web: str | None = None):
        self.name = name
        self.phone = phone
        self.alt = alt
        self.web = web

    def spoken(self) -> str:
        """How Saathi should say it out loud — digits grouped, no URLs."""
        return f"{self.name} on {self.phone}"

    def as_dict(self) -> dict[str, str | None]:
        return {"name": self.name, "phone": self.phone, "alt": self.alt, "web": self.web}


def load(path: Path, region: str) -> list[Helpline]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries = raw.get(region.upper())
    if not entries:
        available = sorted(k for k in raw if not k.startswith("_"))
        raise ValueError(
            f"No helplines configured for region '{region}'. Available: {available}. "
            f"Set SAATHI_HELPLINE_REGION to one of those."
        )
    return [
        Helpline(e["name"], e["phone"], e.get("alt"), e.get("web"))
        for e in entries
    ]


def as_prompt_text(helplines: list[Helpline]) -> str:
    return "\n".join(f"- {h.spoken()}" for h in helplines)
