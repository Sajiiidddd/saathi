#!/usr/bin/env python3
"""Build the retrieval index from data/.

    .venv/bin/python scripts/build_kb.py
    .venv/bin/python scripts/build_kb.py --probe "i can't sleep before exams"

Reports the corpus composition as it goes, because the failure mode here is
silent: an index that builds fine but is 80% philosophy will answer "help me
sleep" with Marcus Aurelius. The technique/education ratio is printed so that
drift is visible at build time rather than discovered in a demo.
"""

from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from saathi.kb import emotion as emotion_module  # noqa: E402
from saathi.kb.embed import DEFAULT_MODEL, FastEmbedEmbedder, HashingEmbedder  # noqa: E402
from saathi.kb.index import build  # noqa: E402
from saathi.kb.ingest import load_corpus  # noqa: E402
from saathi.kb.retrieve import Retriever  # noqa: E402

DATA_DIRS = ["gov_health", "nhs_who", "open_textbooks", "classics", "quotes"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data"))
    parser.add_argument("--out", default=str(ROOT / "index"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--hashing",
        action="store_true",
        help="Use the offline stand-in embedder (no model download). Testing only.",
    )
    parser.add_argument("--probe", action="append", default=[], help="Query the index after building")
    args = parser.parse_args()

    data_root = Path(args.data)
    roots = [data_root / name for name in DATA_DIRS]

    print("Ingesting…")
    started = time.perf_counter()
    chunks = load_corpus(roots)
    if not chunks:
        print(f"  No .md files found under {data_root}. Nothing to build.")
        return 1

    by_doc = collections.Counter(c.doc_id for c in chunks)
    by_type = collections.Counter(c.doc_type for c in chunks)
    lengths = sorted(c.char_len for c in chunks)

    print(f"  {len(by_doc)} documents -> {len(chunks)} chunks in {time.perf_counter()-started:.2f}s")
    print(f"  chunk chars: min {lengths[0]} · median {lengths[len(lengths)//2]} · max {lengths[-1]}")
    print(f"  by type: {dict(by_type)}")

    technique_share = by_type.get("technique", 0) / len(chunks)
    if technique_share < 0.35:
        print(
            f"  ⚠ technique chunks are {technique_share:.0%} of the corpus. Guideline: "
            f"~60% techniques/education vs 40% wisdom — a how-to question ('help me sleep') "
            f"needs actionable chunks to retrieve. Consider adding NHS/NIMH technique docs "
            f"before adding more classics."
        )

    # A crosswalk entry pointing at a topic no document carries is a lane that
    # silently does nothing. Surface it here, not in a demo.
    corpus_topics = {t for c in chunks for t in c.topics}
    orphans = sorted(emotion_module.all_crosswalk_topics() - corpus_topics)
    if orphans:
        print(f"  ⚠ crosswalk topics with no matching chunk: {', '.join(orphans)}")

    if args.hashing:
        embedder, model_name = HashingEmbedder(), "hashing-stub"
        print("Embedding with the offline stub (NOT for serving)…")
    else:
        embedder, model_name = FastEmbedEmbedder(args.model), args.model
        print(f"Embedding with {model_name} (first run downloads the model)…")

    started = time.perf_counter()
    kb = build(chunks, embedder, model_name)
    print(f"  {kb.vectors.shape[0]} vectors × {kb.vectors.shape[1]} dims in {time.perf_counter()-started:.2f}s")

    out = Path(args.out)
    kb.save(out)
    print(f"Saved -> {out}  ({sum(f.stat().st_size for f in out.iterdir())/1024:.0f} KB)")

    for query in args.probe:
        print(f"\n  probe: {query!r}")
        started = time.perf_counter()
        results = Retriever(kb, embedder).retrieve(query, top_k=3)
        elapsed = (time.perf_counter() - started) * 1000
        print(f"  {elapsed:.1f}ms")
        for rank, result in enumerate(results, start=1):
            lanes = ",".join(f"{n}#{h.rank}" for n, h in sorted(result.lanes.items()))
            print(f"   {rank}. [{result.fused_score:.4f}] {result.chunk.source} — {result.chunk.title}")
            print(f"      lanes: {lanes} · {result.chunk.text[:88]!r}…")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
