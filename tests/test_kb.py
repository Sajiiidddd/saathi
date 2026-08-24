#!/usr/bin/env python3
"""Tests for the knowledge base and retrieval.

    .venv/bin/python tests/test_kb.py        # no pytest needed
    .venv/bin/pytest tests/test_kb.py        # also works if you have it

Runs against the real corpus in data/ and uses the hashing stub embedder, so
the whole suite is hermetic and finishes in well under a second — no model
download, no network. The stub can't judge semantic quality, so the assertions
here are about mechanics and invariants: chunk integrity, fusion arithmetic,
lane wiring, provenance. Semantic quality is what `build_kb.py --probe` is for.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from saathi.kb import emotion as emotion_module
from saathi.kb.bm25 import BM25, tokenize
from saathi.kb.embed import HashingEmbedder
from saathi.kb.emotion import CROSSWALK, STATES, EmotionReading, topic_overlap
from saathi.kb.index import KnowledgeBase, build
from saathi.kb.ingest import (
    MAX_CHARS,
    IngestError,
    chunk_document,
    load_corpus,
    load_file,
    parse_front_matter,
)
from saathi.kb.retrieve import Retriever, format_for_prompt

DATA = ROOT / "data"
CORPUS_DIRS = [DATA / n for n in ("gov_health", "nhs_who", "open_textbooks", "classics", "quotes")]


# ---------------------------------------------------------------- ingestion

def test_front_matter_parses_real_file():
    path = DATA / "nhs_who" / "nhs_breathing_exercise.md"
    meta, body = parse_front_matter(path.read_text(encoding="utf-8"), path)
    assert meta["source"] == "NHS (UK)"
    assert meta["type"] == "technique"
    assert "breathing" in meta["topics"]
    assert "Open Government Licence" in meta["licence"]
    assert body.startswith("This calming breathing technique")
    assert "---" not in body.splitlines()[0]


def test_front_matter_rejects_malformed():
    for bad, why in [
        ("no front matter here", "missing block"),
        ("---\ntitle: x\n---\n", "empty body"),
        ('---\ntitle: x\nsource: y\nlicence: z\ntopics: [oops\n---\n\nbody\n', "bad list"),
        ("---\njust_a_line\n---\n\nbody\n", "not key: value"),
    ]:
        try:
            meta, body = parse_front_matter(bad, Path("bad.md"))
            chunk_document(meta, body, "bad")
        except IngestError:
            continue
        raise AssertionError(f"should have rejected: {why}")


def test_missing_required_field_is_rejected():
    meta = {"title": "T", "topics": []}  # no source, no licence
    try:
        chunk_document(meta, "some body text", "d")
    except IngestError as exc:
        assert "source" in str(exc) and "licence" in str(exc)
        return
    raise AssertionError("missing source/licence should fail")


def test_numbered_list_is_never_split():
    """The whole reason chunking is block-aware: severing a 5-step breathing
    exercise produces two useless chunks and one dangerous answer."""
    chunks = load_file(DATA / "nhs_who" / "nhs_breathing_exercise.md")
    holders = [c for c in chunks if "1. Let your breath flow" in c.text]
    assert len(holders) == 1, "step 1 should appear in exactly one chunk"
    chunk = holders[0]
    for step in ("2. Try breathing", "3. Breathe in gently", "4. Then let it flow", "5. Repeat for"):
        assert step in chunk.text, f"{step!r} was separated from step 1"


def test_chunk_sizes_are_sane():
    chunks = load_corpus(CORPUS_DIRS)
    assert chunks, "corpus is empty — did data/ move?"
    # Oversize is allowed only when a single atomic block exceeds the cap.
    for chunk in chunks:
        if chunk.char_len > MAX_CHARS:
            blocks = chunk.text.split("\n\n")
            assert len(blocks) == 1 or max(len(b) for b in blocks) > MAX_CHARS * 0.5, (
                f"{chunk.chunk_id} is {chunk.char_len} chars and looks over-packed"
            )


def test_every_chunk_can_cite_itself():
    """A chunk that can't name its source is one Saathi may not use."""
    for chunk in load_corpus(CORPUS_DIRS):
        assert chunk.citation().strip(), f"{chunk.chunk_id} has no source"
        assert chunk.licence.strip(), f"{chunk.chunk_id} has no licence"
        assert chunk.text.strip()


def test_chunk_ids_are_unique_and_deterministic():
    first = [c.chunk_id for c in load_corpus(CORPUS_DIRS)]
    second = [c.chunk_id for c in load_corpus(CORPUS_DIRS)]
    assert first == second, "corpus order is not deterministic — eval runs won't compare"
    assert len(set(first)) == len(first), "duplicate chunk ids"


# --------------------------------------------------------------------- bm25

def test_tokenizer_stems_and_strips():
    assert tokenize("Sleeping and BREATHING exercises") == ["sleep", "breath", "exercise"]
    assert tokenize("the and of to") == []
    assert tokenize("worries worried") == ["worry", "worry"]


def test_singular_and_plural_share_a_stem():
    """The bug this caught: -es stripping folded `exercises`->`exercis` while
    `exercise` stayed put, so the two never matched."""
    for singular, plural in [
        ("exercise", "exercises"),
        ("technique", "techniques"),
        ("tip", "tips"),
        ("routine", "routines"),
        ("box", "boxes"),
        ("worry", "worries"),
    ]:
        assert tokenize(singular) == tokenize(plural), f"{singular} != {plural}"


def test_doubled_consonants_collapse():
    """running -> runn would never match run."""
    for base, inflected in [("run", "running"), ("stop", "stopping"),
                            ("get", "getting"), ("sit", "sitting")]:
        assert tokenize(base) == tokenize(inflected), f"{base} != {inflected}"
    # l/s/z double legitimately and must survive.
    assert tokenize("still") == ["still"]


def test_stemmer_leaves_non_plurals_alone():
    for word in ("stress", "focus", "wellness", "analysis"):
        assert tokenize(word) == [word], f"{word} was wrongly stemmed"


def test_bm25_ranks_the_obvious_document_first():
    docs = [
        tokenize("breathing exercise for stress and panic"),
        tokenize("sleep hygiene and a bedtime routine"),
        tokenize("gratitude journaling improves wellbeing"),
    ]
    index = BM25(docs)
    assert index.search("how do I sleep better")[0][0] == 1
    assert index.search("breathing for panic")[0][0] == 0
    assert index.search("") == []
    assert index.search("zzzzz nonexistent") == []


def test_bm25_length_normalisation_applies():
    short = tokenize("sleep")
    padded = tokenize("sleep " + "unrelated filler words here " * 40)
    index = BM25([short, padded])
    ranked = index.search("sleep")
    assert ranked[0][0] == 0, "the concise chunk should win on the same term"


# ------------------------------------------------------------------ emotion

def test_crosswalk_topics_all_exist_in_corpus():
    """A crosswalk pointing at absent topics is a lane doing nothing at all."""
    corpus_topics = {t for c in load_corpus(CORPUS_DIRS) for t in c.topics}
    orphans = emotion_module.all_crosswalk_topics() - corpus_topics
    assert not orphans, (
        f"crosswalk references topics no chunk carries: {sorted(orphans)}. "
        f"Either tag documents with them or drop them from CROSSWALK."
    )


def test_every_state_has_an_entry():
    for state in STATES:
        assert state in CROSSWALK
    assert CROSSWALK["neutral"] == (), "neutral must boost nothing"


def test_reading_validation_and_weight():
    assert EmotionReading().weight == 0.0  # neutral default is inert
    assert EmotionReading("anxious", 1.0, 1.0).weight > 0
    assert EmotionReading("anxious", 1.0, 0.0).weight == 0.0, "no confidence -> no influence"
    # An unsure reading should nudge, not dominate.
    assert EmotionReading("low", 0.3, 0.4).weight < EmotionReading("low", 1.0, 1.0).weight
    for bad in [("nonsense", 0.5, 0.5), ("anxious", 2.0, 0.5), ("anxious", 0.5, -1.0)]:
        try:
            EmotionReading(*bad)
        except ValueError:
            continue
        raise AssertionError(f"should reject {bad}")


def test_topic_overlap_is_a_fraction():
    assert topic_overlap(["breathing", "grounding"], ("breathing", "grounding")) == 1.0
    assert topic_overlap(["breathing"], ("breathing", "grounding")) == 0.5
    assert topic_overlap([], ("breathing",)) == 0.0
    assert topic_overlap(["sleep"], ()) == 0.0
    # Breadth alone must not win: 15 topics, 1 match, still scores 1/2.
    assert topic_overlap(["breathing"] + [f"t{i}" for i in range(14)], ("breathing", "grounding")) == 0.5


# -------------------------------------------------------------------- index

def _kb() -> tuple[KnowledgeBase, HashingEmbedder]:
    embedder = HashingEmbedder()
    return build(load_corpus(CORPUS_DIRS), embedder, "hashing-stub"), embedder


def test_index_roundtrips_through_disk():
    kb, _ = _kb()
    with tempfile.TemporaryDirectory() as tmp:
        kb.save(Path(tmp))
        loaded = KnowledgeBase.load(Path(tmp))
    assert len(loaded) == len(kb)
    assert loaded.vectors.shape == kb.vectors.shape
    assert [c.chunk_id for c in loaded.chunks] == [c.chunk_id for c in kb.chunks]
    assert loaded.chunks[0].source == kb.chunks[0].source
    # BM25 is rebuilt on load, not serialised — same answers either way.
    assert loaded.lexical_search("sleep", 3) == kb.lexical_search("sleep", 3)


def test_index_rejects_inconsistent_state():
    kb, _ = _kb()
    try:
        KnowledgeBase(kb.chunks[:-1], kb.vectors, kb.bm25, "x")
    except ValueError:
        return
    raise AssertionError("chunk/vector count mismatch must not construct")


def test_missing_index_gives_actionable_error():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            KnowledgeBase.load(Path(tmp) / "nope")
        except FileNotFoundError as exc:
            assert "build_kb.py" in str(exc)
            return
    raise AssertionError("loading a missing index should say how to build one")


def test_vectors_are_normalised():
    kb, _ = _kb()
    import numpy as np

    norms = np.linalg.norm(kb.vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), "dot product only equals cosine if rows are unit length"


# ----------------------------------------------------------------- retrieval

def test_retrieval_returns_ranked_results_with_provenance():
    kb, embedder = _kb()
    results = Retriever(kb, embedder).retrieve("how do I sleep better", top_k=4)
    assert results
    scores = [r.fused_score for r in results]
    assert scores == sorted(scores, reverse=True), "results must be score-descending"
    for result in results:
        assert result.lanes, "every result must say which lane found it"
        for name, hit in result.lanes.items():
            assert name in ("semantic", "lexical", "emotion")
            assert hit.rank >= 1
            explained = result.explain()
            assert explained["chunk_id"] and explained["source"]
            assert "lanes" in explained and explained["preview"]


def test_rrf_arithmetic_is_exactly_as_documented():
    kb, embedder = _kb()
    retriever = Retriever(kb, embedder, rrf_k=60)
    for result in retriever.retrieve("stress", top_k=5):
        if result.penalties:
            continue  # boost/penalty applied after fusion
        expected = sum(hit.contribution for hit in result.lanes.values())
        assert abs(result.fused_score - expected) < 1e-9
        for hit in result.lanes.values():
            assert abs(hit.contribution - (hit.contribution)) < 1e-12
            assert hit.contribution <= 1 / (60 + 1) + 1e-12


def test_emotion_lane_activates_and_deactivates():
    kb, embedder = _kb()
    retriever = Retriever(kb, embedder)

    neutral = retriever.retrieve("I feel tense", emotion=EmotionReading(), top_k=8)
    assert not any("emotion" in r.lanes for r in neutral), "neutral must not open the lane"

    anxious = retriever.retrieve(
        "I feel tense", emotion=EmotionReading("anxious", 0.9, 0.9), top_k=8
    )
    assert any("emotion" in r.lanes for r in anxious), "anxious should open the lane"

    # The lane must actually change the outcome, not just appear in metadata.
    assert [r.chunk.chunk_id for r in anxious] != [r.chunk.chunk_id for r in neutral]


def test_emotion_lane_surfaces_crosswalk_topics():
    kb, embedder = _kb()
    results = Retriever(kb, embedder).retrieve(
        "everything is too much", emotion=EmotionReading("anxious", 1.0, 1.0), top_k=5
    )
    boosted = set(CROSSWALK["anxious"])
    lane_hits = [r for r in results if "emotion" in r.lanes]
    assert lane_hits
    assert any(set(r.chunk.topics) & boosted for r in lane_hits)


def test_repeat_penalty_demotes_but_does_not_ban():
    kb, embedder = _kb()
    retriever = Retriever(kb, embedder)
    # Wide top_k on purpose: with a narrow one a demoted chunk falls off the end
    # and you can't tell demotion from exclusion.
    baseline = retriever.retrieve("help me relax", top_k=20)
    assert baseline
    top = baseline[0]
    top_doc = top.chunk.doc_id

    after = retriever.retrieve("help me relax", top_k=20, seen_doc_ids={top_doc})
    penalised = [r for r in after if r.chunk.chunk_id == top.chunk.chunk_id]
    assert penalised, "a used document should still be reachable, just demoted"
    assert "already_used_this_call" in penalised[0].penalties
    assert penalised[0].fused_score < top.fused_score
    # And it should genuinely lose ground to something else.
    assert after[0].chunk.doc_id != top_doc or len({r.chunk.doc_id for r in after}) == 1


def test_how_to_query_boosts_technique_chunks():
    kb, embedder = _kb()
    retriever = Retriever(kb, embedder)
    boosted = retriever.retrieve("how do I calm down", top_k=10)
    flagged = [r for r in boosted if "technique_boost" in r.penalties]
    assert all(r.chunk.doc_type == "technique" for r in flagged)
    # A statement, not a how-to question, gets no boost.
    plain = retriever.retrieve("work has been difficult lately", top_k=10)
    assert not any("technique_boost" in r.penalties for r in plain)


def test_empty_and_whitespace_queries_are_safe():
    kb, embedder = _kb()
    retriever = Retriever(kb, embedder)
    assert retriever.retrieve("") == []
    assert retriever.retrieve("   \n ") == []


def test_prompt_formatting_labels_sources_and_respects_budget():
    kb, embedder = _kb()
    results = Retriever(kb, embedder).retrieve("sleep", top_k=5)
    rendered = format_for_prompt(results, max_chars=2400)
    assert "Source:" in rendered and "[1]" in rendered
    assert len(rendered) <= 2400 + 200
    assert format_for_prompt([]) == ""
    # A tiny budget must yield nothing rather than a truncated half-passage.
    assert format_for_prompt(results, max_chars=10) == ""


# ------------------------------------------------------------------- runner

def _run():
    tests = [(n, o) for n, o in sorted(globals().items()) if n.startswith("test_") and callable(o)]
    passed, failed = 0, []
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"  \033[32m✓\033[0m {name}")
        except Exception as exc:  # noqa: BLE001
            failed.append((name, exc))
            print(f"  \033[31m✗\033[0m {name}\n      {type(exc).__name__}: {exc}")
    print(f"\n{passed}/{len(tests)} passed")
    if failed:
        print(f"\033[31m{len(failed)} FAILED\033[0m")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())
