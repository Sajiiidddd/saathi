"""Saathi's knowledge base: ingestion, indexing, and three-lane retrieval.

    ingest    markdown + front-matter -> attributable Chunks
    bm25      hand-rolled Okapi BM25 (lexical lane)
    embed     local ONNX embeddings   (semantic lane)
    emotion   state -> topic crosswalk (emotion lane)
    index     chunks + vectors + lexical index, saved as jsonl/npy
    retrieve  RRF fusion, with per-lane provenance on every result
"""

from .emotion import EmotionReading
from .index import KnowledgeBase
from .ingest import Chunk
from .retrieve import Retriever, format_for_prompt

__all__ = ["Chunk", "EmotionReading", "KnowledgeBase", "Retriever", "format_for_prompt"]
