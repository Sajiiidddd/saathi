"""Okapi BM25, hand-rolled.

Deliberately not a dependency: at this corpus size it's ~50 lines, every
constant is visible and tunable, and there is no library behaviour to reverse-
engineer when a ranking looks wrong.

    score(D,Q) = Σ  IDF(q) · f(q,D)·(k1+1) / (f(q,D) + k1·(1 − b + b·|D|/avgdl))
    IDF(q)     = ln(1 + (N − n(q) + 0.5) / (n(q) + 0.5))

k1=1.5 controls term-frequency saturation (a chunk saying "sleep" six times
isn't six times more about sleep). b=0.75 is length normalisation — without it,
long education chunks outrank the short technique chunks that actually answer
"how do I get to sleep".
"""

from __future__ import annotations

import math
import re
from collections import Counter

K1 = 1.5
B = 0.75

_WORD = re.compile(r"[a-z0-9']+")

# Small and deliberate. An aggressive stopword list would eat "how", "what" and
# "can" — exactly the question words that distinguish a how-to ask from a
# reflective one. These are only the terms that carry no retrieval signal at all.
STOPWORDS = frozenset("""
a an and are as at be been being but by for from had has have he her him his i
if in into is it its me my of on or our ours she that the their them then there
these they this those to was were will with you your
""".split())


def tokenize(text: str) -> list[str]:
    """Lowercase, split, drop stopwords, light suffix strip.

    The stemmer is intentionally crude — plurals and -ing/-ed only. It exists so
    "sleeping" matches "sleep" and "breathing exercises" matches "breathing
    exercise". A full Porter stemmer would also fold "meditation"/"meditate",
    which is fine, but it's another thing to explain for little measured gain
    at this corpus size.
    """
    out = []
    for word in _WORD.findall(text.lower()):
        if word in STOPWORDS or len(word) < 2:
            continue
        out.append(_stem(word))
    return out


def _stem(word: str) -> str:
    """Crude suffix stripping. Order and the sibilant rules both matter.

    A naive "strip -es" folds `exercises -> exercis` while leaving `exercise`
    alone, so the singular and plural stop matching and a search for "breathing
    exercise" misses a chunk that says "exercises". Plain -s comes first, and
    -es is only stripped after a sibilant where the e is epenthetic
    (boxes -> box, classes -> class).
    """
    if len(word) <= 3:
        return word
    if len(word) > 4 and word.endswith(("ies", "ied")):
        return word[:-3] + "y"
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith(("ches", "shes", "xes", "zes")):
        return word[:-2]
    # `ss`/`us`/`is` are not plural markers: stress, focus, this.
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    if len(word) > 5 and word.endswith("ing"):
        return _undouble(word[:-3])
    if len(word) > 4 and word.endswith("ed"):
        return _undouble(word[:-2])
    return word


def _undouble(stem: str) -> str:
    """English doubles the final consonant before -ing/-ed: running, stopping.

    Without this, `running -> runn` never matches `run`. l/s/z are excluded
    because they double legitimately (still, class, buzz).

    Known limitation: -ck words are unfixable by rule. `panicking -> panick`
    still misses `panic`, because the obvious fix (strip a trailing k) would
    break `checking -> check`. The emotion crosswalk covers panic/anxiety
    material by topic tag, and the semantic lane matches it regardless — which
    is precisely why retrieval is fused across three lanes rather than trusting
    any single one.
    """
    if len(stem) > 3 and stem[-1] == stem[-2] and stem[-1] not in "lsz":
        return stem[:-1]
    return stem


class BM25:
    """A lexical index over pre-tokenized documents."""

    def __init__(self, documents: list[list[str]], k1: float = K1, b: float = B):
        self.k1 = k1
        self.b = b
        self.doc_count = len(documents)
        self.doc_lens = [len(d) for d in documents]
        self.avg_len = (sum(self.doc_lens) / self.doc_count) if self.doc_count else 0.0
        self.term_freqs: list[Counter] = [Counter(d) for d in documents]

        doc_freq: Counter = Counter()
        for freqs in self.term_freqs:
            doc_freq.update(freqs.keys())

        # Precompute IDF — the query path should do as little arithmetic as
        # possible when it's sitting inside a voice turn's latency budget.
        self.idf: dict[str, float] = {
            term: math.log(1 + (self.doc_count - n + 0.5) / (n + 0.5))
            for term, n in doc_freq.items()
        }

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Return (doc_index, score) for the best matches, score-descending."""
        terms = tokenize(query)
        if not terms or not self.doc_count:
            return []

        scores: list[tuple[int, float]] = []
        for index, freqs in enumerate(self.term_freqs):
            total = 0.0
            for term in terms:
                freq = freqs.get(term)
                if not freq:
                    continue
                norm = 1 - self.b + self.b * (self.doc_lens[index] / self.avg_len)
                total += self.idf[term] * (freq * (self.k1 + 1)) / (freq + self.k1 * norm)
            if total > 0:
                scores.append((index, total))

        scores.sort(key=lambda pair: (-pair[1], pair[0]))
        return scores[:top_k]
