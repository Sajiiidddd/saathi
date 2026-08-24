"""Three-lane retrieval fused with Reciprocal Rank Fusion.

    RRF(d) = Σ_lanes  weight_lane / (k + rank_lane(d))        k = 60

Why RRF and not score blending: the three lanes produce incomparable numbers.
BM25 is unbounded and corpus-dependent, cosine sits in -1..1, topic overlap is
a fraction. Normalising them against each other means inventing a conversion
rate and re-tuning it every time the corpus changes. RRF throws the magnitudes
away and keeps only the ordering each lane is actually competent to express.
k=60 is the value from the original Cormack et al. paper — large enough that
rank 1 and rank 2 aren't wildly far apart, small enough that rank 40 stops
mattering.

Every result carries the full reason it surfaced: which lanes ranked it, where,
and with what raw score. That object is what the "Why did she say that?" drawer
renders and what the dashboard logs. Retrieval you can't explain is retrieval
you can't debug — and in this domain, can't audit either.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .emotion import EmotionReading, topic_overlap
from .embed import Embedder
from .index import KnowledgeBase
from .ingest import Chunk

RRF_K = 60
LANE_POOL = 20  # candidates each lane contributes before fusion

SEMANTIC_WEIGHT = 1.0
LEXICAL_WEIGHT = 1.0

# A chunk from a document already used this call is demoted, not banned: if it's
# genuinely the only good answer she should still be able to reach it.
REPEAT_PENALTY = 0.55

# Mild nudge toward actionable material when the caller asked a how-to question.
TECHNIQUE_BOOST = 1.15

_HOW_TO = re.compile(
    r"\b(how (do|can|should) i|how to|what (can|should) i do|help me|any (tips|advice)|"
    r"give me|teach me|technique|exercise|steps?)\b",
    re.I,
)


@dataclass
class LaneHit:
    rank: int  # 1-based
    score: float
    contribution: float


@dataclass
class RetrievalResult:
    chunk: Chunk
    fused_score: float
    lanes: dict[str, LaneHit] = field(default_factory=dict)
    penalties: dict[str, float] = field(default_factory=dict)

    def explain(self) -> dict:
        """Serialisable provenance for the drawer / dashboard / eval harness."""
        return {
            "chunk_id": self.chunk.chunk_id,
            "source": self.chunk.source,
            "title": self.chunk.title,
            "doc_type": self.chunk.doc_type,
            "topics": self.chunk.topics,
            "fused_score": round(self.fused_score, 5),
            "lanes": {
                name: {
                    "rank": hit.rank,
                    "score": round(hit.score, 4),
                    "contribution": round(hit.contribution, 5),
                }
                for name, hit in self.lanes.items()
            },
            "penalties": {k: round(v, 3) for k, v in self.penalties.items()},
            "preview": self.chunk.text[:160].replace("\n", " "),
        }


class Retriever:
    def __init__(self, kb: KnowledgeBase, embedder: Embedder, rrf_k: int = RRF_K):
        self.kb = kb
        self.embedder = embedder
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        emotion: EmotionReading | None = None,
        top_k: int = 5,
        seen_doc_ids: frozenset[str] | set[str] | tuple[str, ...] = (),
        pool: int = LANE_POOL,
    ) -> list[RetrievalResult]:
        if not query.strip() or not len(self.kb):
            return []

        seen = set(seen_doc_ids)
        accumulated: dict[int, RetrievalResult] = {}

        def add(index: int, lane: str, rank: int, score: float, weight: float):
            contribution = weight / (self.rrf_k + rank)
            result = accumulated.get(index)
            if result is None:
                result = RetrievalResult(chunk=self.kb.chunks[index], fused_score=0.0)
                accumulated[index] = result
            result.lanes[lane] = LaneHit(rank=rank, score=score, contribution=contribution)
            result.fused_score += contribution

        # ---- lane 1: semantic
        query_vector = self.embedder.encode_query(query)
        for rank, (index, score) in enumerate(self.kb.semantic_search(query_vector, pool), start=1):
            add(index, "semantic", rank, score, SEMANTIC_WEIGHT)

        # ---- lane 2: lexical
        for rank, (index, score) in enumerate(self.kb.lexical_search(query, pool), start=1):
            add(index, "lexical", rank, score, LEXICAL_WEIGHT)

        # ---- lane 3: emotion crosswalk
        # Switches itself off for neutral turns or an unconfident classifier,
        # rather than injecting noise into every retrieval.
        if emotion is not None and emotion.weight > 0:
            boosted = emotion.boosted_topics
            overlaps = [
                (i, topic_overlap(chunk.topics, boosted))
                for i, chunk in enumerate(self.kb.chunks)
            ]
            ranked = sorted(
                (pair for pair in overlaps if pair[1] > 0),
                key=lambda pair: (-pair[1], pair[0]),
            )[:pool]
            for rank, (index, overlap) in enumerate(ranked, start=1):
                add(index, "emotion", rank, overlap, emotion.weight)

        # ---- post-fusion adjustments
        wants_how_to = bool(_HOW_TO.search(query))
        for result in accumulated.values():
            if wants_how_to and result.chunk.doc_type == "technique":
                result.fused_score *= TECHNIQUE_BOOST
                result.penalties["technique_boost"] = TECHNIQUE_BOOST
            if result.chunk.doc_id in seen:
                result.fused_score *= REPEAT_PENALTY
                result.penalties["already_used_this_call"] = REPEAT_PENALTY

        ordered = sorted(
            accumulated.values(),
            key=lambda r: (-r.fused_score, r.chunk.chunk_id),
        )
        return ordered[:top_k]


def format_for_prompt(results: list[RetrievalResult], max_chars: int = 2400) -> str:
    """Render retrieved passages for the grounded generation prompt.

    Each passage is labelled with its source so the model can attribute in
    speech, and so a citation it emits can be checked against what it was
    actually given — which is how the eval harness measures grounding rate.
    """
    if not results:
        return ""
    parts: list[str] = []
    budget = max_chars
    for i, result in enumerate(results, start=1):
        chunk = result.chunk
        block = f"[{i}] Source: {chunk.source} — {chunk.title}\n{chunk.text}"
        if len(block) > budget:
            break
        parts.append(block)
        budget -= len(block)
    return "\n\n".join(parts)
