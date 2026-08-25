"""The quote index — emotion-tagged closing thoughts from the classics.

Not retrieval content: these are garnish, offered occasionally when the
moment is reflective, never when someone asked a how-to question and never
twice in a session. A quote every turn is a fortune cookie machine; one
well-placed Marcus Aurelius line a session is character.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Quote:
    text: str
    author: str
    work: str
    states: tuple[str, ...]

    def key(self) -> str:
        return self.text[:40]


def load(path: Path) -> list[Quote]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        Quote(
            text=q["text"],
            author=q["author"],
            work=q["work"],
            states=tuple(q.get("states", [])),
        )
        for q in raw.get("quotes", [])
    ]


def pick(
    quotes: list[Quote],
    state: str,
    used: set[str],
    rng: random.Random,
    chance: float = 0.3,
) -> Quote | None:
    """Sometimes return a fitting, unused quote. Deliberately unreliable —
    the rarity is the charm."""
    if not quotes or rng.random() > chance:
        return None
    fitting = [q for q in quotes if state in q.states and q.key() not in used]
    if not fitting:
        return None
    return rng.choice(fitting)
