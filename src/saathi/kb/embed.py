"""Embeddings, computed locally.

WHY LOCAL AND NOT AN EMBEDDING API. This runs inside a voice turn. A hosted
embedding call costs 100-200ms of network round-trip; bge-small on CPU costs
~10-20ms for one short query. In a ~1.2s voice-to-voice budget that difference
is a tenth of the whole thing, spent on the least interesting part of the
pipeline. It also removes a network failure mode from the request path and a
per-call cost, and it ships inside the container on deploy.

The trade-off to state honestly if asked: a hosted large embedding model would
retrieve somewhat better than a 384-dim small one. At a 40-80 document corpus
that gap is small, and the emotion lane and BM25 are doing work that a bigger
embedder wouldn't do anyway.

bge models are asymmetric — they expect queries and passages encoded
differently (queries get an instruction prefix). fastembed's query_embed /
passage_embed handle that; using plain embed() for both is a common and quietly
costly mistake.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_DIM = 384

# For Hindi/English support, swap to
# "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" (384-dim too, so
# the index shape is unchanged). English retrieval quality drops noticeably —
# only make that trade when the multilingual demo is actually happening.


class Embedder(Protocol):
    """Anything that can turn text into normalised vectors."""

    dim: int

    def encode_passages(self, texts: list[str]) -> np.ndarray: ...
    def encode_query(self, text: str) -> np.ndarray: ...


class FastEmbedEmbedder:
    """ONNX embeddings via fastembed. No torch, reuses pipecat's onnxruntime."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        cache_dir: Path | None = None,
        dim: int = DEFAULT_DIM,
    ):
        self.model_name = model_name
        self.dim = dim
        self._cache_dir = str(cache_dir) if cache_dir else None
        self._model = None  # lazy: don't pull ~130MB at import time

    def _ensure(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(
                model_name=self.model_name,
                cache_dir=self._cache_dir,
            )
        return self._model

    def warm(self) -> float:
        """Load the model and run one throwaway embed. Call at boot.

        Lazy loading is right for tests and CLI use, but in a voice call it
        means the FIRST caller of a session pays ~900ms of ONNX session setup
        inside their first turn — dwarfing the entire latency budget. Two
        seconds at startup is free; 900ms mid-conversation is the difference
        between "she's thinking" and "it's broken". Returns seconds taken so
        the boot log can show it.
        """
        import time

        started = time.perf_counter()
        self.encode_query("warm up")
        return time.perf_counter() - started

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        model = self._ensure()
        vectors = np.array(list(model.passage_embed(texts)), dtype=np.float32)
        return _normalise(vectors)

    def encode_query(self, text: str) -> np.ndarray:
        model = self._ensure()
        vector = np.array(list(model.query_embed([text]))[0], dtype=np.float32)
        return _normalise(vector.reshape(1, -1))[0]


class HashingEmbedder:
    """Deterministic stand-in for tests and offline work.

    Not a real semantic model — a hashed bag of character trigrams. It exists so
    the index, fusion and retrieval logic can be tested fast and hermetically,
    without a 130MB download or any network. Never use it to serve: it has no
    notion of meaning, only of overlapping spelling.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _encode_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        cleaned = " ".join(text.lower().split())
        for i in range(len(cleaned) - 2):
            trigram = cleaned[i : i + 3]
            vector[hash(trigram) % self.dim] += 1.0
        return vector

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalise(np.vstack([self._encode_one(t) for t in texts]))

    def encode_query(self, text: str) -> np.ndarray:
        return _normalise(self._encode_one(text).reshape(1, -1))[0]


def _normalise(vectors: np.ndarray) -> np.ndarray:
    """L2-normalise rows so a dot product IS cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms
