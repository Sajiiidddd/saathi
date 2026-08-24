"""The emotion crosswalk — the third retrieval lane.

Semantic and lexical search both answer "what did they ask?". Neither answers
"what do they need?". Someone who says "I keep replaying that conversation from
this morning" has asked about a conversation; what helps is rumination material.
Cosine similarity will hand back documents about conversations.

So: a classifier tags each turn with an emotional state, and a hand-built map
turns that state into topic-tag boosts. It is a curated lookup table, not a
learned component — inspectable, testable, and it cannot drift.

The tags below are checked against the live corpus by tests/test_kb.py — a
crosswalk pointing at topics no document carries is a lane that silently does
nothing, which is worse than not having it.
"""

from __future__ import annotations

from dataclasses import dataclass

# Recognised states. `neutral` is real and important: it means "boost nothing",
# which is the correct behaviour for "what does the NHS say about caffeine?".
STATES = ("anxious", "low", "stressed", "overwhelmed", "ruminating", "neutral")

CROSSWALK: dict[str, tuple[str, ...]] = {
    # Acute anxiety wants something to do with the body, right now.
    "anxious": ("grounding", "breathing", "relaxation", "panic", "anxiety", "worry-management"),
    # Low mood wants behavioural activation — movement, people, small structure.
    # Explicitly NOT "positivity": telling a flat person to be grateful lands badly.
    "low": ("exercise", "social-connection", "routine", "giving", "purpose", "self-efficacy"),
    "stressed": ("stress", "coping", "relaxation", "breathing", "limits-on-news"),
    # Overwhelm is a load problem, not a feelings problem: reduce input, triage.
    "overwhelmed": ("stress", "coping", "limits-on-news", "worry-management", "self-care"),
    # Rumination wants distance from the thought, not engagement with it.
    "ruminating": ("mindfulness", "meditation", "journaling", "worry-management", "gratitude"),
    "neutral": (),
}

# How hard the lane pushes. Applied to the emotion lane's rank contribution in
# RRF, scaled by classifier confidence and intensity — a hesitant "possibly
# anxious" should nudge, not dominate.
BASE_WEIGHT = 1.0


@dataclass(frozen=True)
class EmotionReading:
    """Output of the per-turn emotion classifier. Constructed by hand in tests."""

    state: str = "neutral"
    intensity: float = 0.0  # 0..1
    confidence: float = 0.0  # 0..1

    def __post_init__(self):
        if self.state not in STATES:
            raise ValueError(f"unknown emotional state {self.state!r}; expected one of {STATES}")
        for name in ("intensity", "confidence"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within 0..1, got {value}")

    @property
    def boosted_topics(self) -> tuple[str, ...]:
        return CROSSWALK.get(self.state, ())

    @property
    def weight(self) -> float:
        """Zero for neutral or an unsure classifier — the lane switches itself off."""
        if self.state == "neutral":
            return 0.0
        return BASE_WEIGHT * self.confidence * max(self.intensity, 0.25)


def topic_overlap(chunk_topics: list[str], boosted: tuple[str, ...]) -> float:
    """Fraction of the boosted set a chunk covers, in 0..1.

    Fraction rather than raw count so a chunk tagged with fifteen topics can't
    win on breadth alone.
    """
    if not boosted or not chunk_topics:
        return 0.0
    hits = len(set(chunk_topics) & set(boosted))
    return hits / len(boosted)


def all_crosswalk_topics() -> set[str]:
    return {topic for topics in CROSSWALK.values() for topic in topics}
