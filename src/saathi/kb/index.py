"""The index: chunks + vectors + lexical index, built once, loaded at boot.

WHY NOT AN ANN LIBRARY (FAISS et al.): approximate indexes earn their place
when exact search stops being affordable, which is somewhere north of ~100k
vectors. This corpus is 40-80 documents: call it 60-600 chunks, or under a
megabyte of float32. Exact search over that is a single matrix multiply —
roughly 0.2ms, dependency-free, and it returns the true nearest neighbours
rather than an approximation. An ANN index here would buy a slower query, an
extra dependency, and recall < 100%. The semantic lane sits behind an
interface, so swapping in FAISS if the corpus ever grows 100x is a contained
change.

Everything is stored as plain .jsonl + .npy. Inspectable with a text editor,
diffable, and no pickle — an index format you can't read is one you can't debug
at 2am on demo eve.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .bm25 import BM25, tokenize
from .embed import Embedder
from .ingest import Chunk

FORMAT_VERSION = 1


@dataclass
class KnowledgeBase:
    """Everything retrieval needs, in memory."""

    chunks: list[Chunk]
    vectors: np.ndarray  # (n_chunks, dim), L2-normalised
    bm25: BM25
    model_name: str

    def __post_init__(self):
        if len(self.chunks) != self.vectors.shape[0]:
            raise ValueError(
                f"index is inconsistent: {len(self.chunks)} chunks but "
                f"{self.vectors.shape[0]} vectors"
            )

    def __len__(self) -> int:
        return len(self.chunks)

    @property
    def topics(self) -> set[str]:
        return {topic for chunk in self.chunks for topic in chunk.topics}

    def semantic_search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        """Exact cosine search. Vectors are normalised, so this is a dot product."""
        if not len(self.chunks):
            return []
        scores = self.vectors @ query_vector
        # argpartition beats a full sort when top_k << n; the sort is only over
        # the k candidates we keep.
        k = min(top_k, len(scores))
        candidates = np.argpartition(-scores, k - 1)[:k]
        ranked = sorted(candidates, key=lambda i: (-scores[i], i))
        return [(int(i), float(scores[i])) for i in ranked]

    def lexical_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        return self.bm25.search(query, top_k=top_k)

    # ---- persistence ----------------------------------------------------

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "chunks.jsonl").open("w", encoding="utf-8") as handle:
            for chunk in self.chunks:
                handle.write(json.dumps(chunk.as_dict(), ensure_ascii=False) + "\n")
        np.save(directory / "vectors.npy", self.vectors)
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "format_version": FORMAT_VERSION,
                    "model_name": self.model_name,
                    "chunk_count": len(self.chunks),
                    "dim": int(self.vectors.shape[1]) if len(self.chunks) else 0,
                    "doc_count": len({c.doc_id for c in self.chunks}),
                    "topics": sorted(self.topics),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> "KnowledgeBase":
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"No index at {directory}. Build it first:\n"
                f"  .venv/bin/python scripts/build_kb.py"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format_version") != FORMAT_VERSION:
            raise ValueError(
                f"Index format v{manifest.get('format_version')} but this code expects "
                f"v{FORMAT_VERSION}. Rebuild with scripts/build_kb.py."
            )

        chunks = [
            Chunk.from_dict(json.loads(line))
            for line in (directory / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        vectors = np.load(directory / "vectors.npy")
        return cls(
            chunks=chunks,
            vectors=vectors,
            bm25=_build_bm25(chunks),
            model_name=manifest["model_name"],
        )


def _build_bm25(chunks: list[Chunk]) -> BM25:
    """Rebuilt on load rather than serialised — it's milliseconds, and a derived
    artefact you can't accidentally leave stale."""
    documents = []
    for chunk in chunks:
        # Title and topics are indexed alongside the body: a chunk from
        # "Breathing Exercise for Stress" should match "breathing" even if the
        # body never repeats the title.
        enriched = f"{chunk.title} {' '.join(chunk.topics)} {chunk.text}"
        documents.append(tokenize(enriched))
    return BM25(documents)


def build(chunks: list[Chunk], embedder: Embedder, model_name: str) -> KnowledgeBase:
    vectors = embedder.encode_passages([c.text for c in chunks])
    return KnowledgeBase(
        chunks=chunks,
        vectors=vectors,
        bm25=_build_bm25(chunks),
        model_name=model_name,
    )
