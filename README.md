# Saathi

A real-time voice companion for everyday wellbeing — stress, sleep, exam
anxiety, low motivation. You talk to it; it listens, reflects, and teaches one
evidence-based coping technique at a time, citing where the guidance comes from
("this comes from the NHS self-help guide on sleep").

The design centre is not what it does but what it refuses to do. It does not
diagnose. It does not discuss medication. And when a caller signals crisis, it
stops being a coach entirely: a deterministic safety gate hands off to human
helplines before retrieval or generation ever run.

> **This is a portfolio demonstration. It is not therapy, not medical advice,
> and not for use in a crisis.** If you need to talk to someone now: in India,
> Tele-MANAS 14416 or AASRA +91-9820466726; in the UK, Samaritans 116 123; in
> the US, call or text 988.

## Architecture

```
browser mic (WebRTC, peer-to-peer)
        │
        ▼
┌─────────────────────────┐
│  Pipecat pipeline        │  turn-taking, barge-in, VAD + smart-turn
└──────────┬──────────────┘
           ▼
   Azure STT (streaming)
           │
           ▼
┌─────────────────────────┐
│ 1. SAFETY GATE           │  crisis? → deterministic handoff, LLM bypassed
│ 2. SCOPE GATE            │  diagnosis/meds? → decline
│ 3. RETRIEVE              │  3-lane hybrid: semantic + BM25 + emotion, RRF
│ 4. GENERATE              │  Gemini Flash, grounded, 2–4 spoken sentences
└──────────┬──────────────┘
           ▼
   Azure TTS (streaming) ──► caller
           │
           ▼
   per-stage latency log (JSONL, p50/p95)
```

The gates run before retrieval and generation on purpose: crisis handling must
not depend on retrieval quality or on the model behaving well today. It is a
fixed code path with a fixed script and verified phone numbers.

## Retrieval

Three lanes, fused with Reciprocal Rank Fusion (k=60):

1. **Semantic** — local ONNX embeddings (bge-small-en-v1.5, 384-dim), exact
   cosine search. Local because a hosted embedding call costs 100–200ms of
   network inside a voice turn's latency budget; on-CPU it costs ~5ms. Exact
   rather than an ANN index because the corpus is a few hundred chunks — a
   single matrix multiply returns true nearest neighbours in ~0.2ms, so an
   approximate index would only add a dependency and lower recall.
2. **Lexical** — Okapi BM25, implemented in ~50 lines rather than imported, so
   every constant is visible and tunable.
3. **Emotion crosswalk** — a classifier tags each turn with an emotional state
   (anxious / low / stressed / overwhelmed / ruminating / neutral); a
   hand-built table maps states to boosted topic tags. Someone saying "I keep
   replaying that conversation from this morning" has *asked* about a
   conversation, but what helps is rumination material — similarity search
   alone can't make that jump. The table is data, not a model: inspectable,
   testable, can't drift. It switches itself off on neutral turns or an
   unconfident classifier.

On top of fusion: technique chunks get a mild boost when the question is a
how-to, and documents already used in the session are demoted so the same
exercise isn't suggested twice. Every result carries full provenance — which
lanes surfaced it, at what rank, with what score — so any answer can be traced
back to why it was given.

Voice turns are conversational fragments ("and at night?"), so queries are
rewritten with dialogue context before retrieval.

If nothing relevant is retrieved, the answer is an honest "I don't have good
guidance on that" — the system prompt forbids uncited claims, and the
generation step only sees passages the retriever actually returned.

## Knowledge base

Curated open-licence sources only, each chunk tagged with source, licence, and
topics (the front-matter schema is in `docs/README_DATA.md`):

| Source | Licence |
|---|---|
| NIMH, CDC, MedlinePlus | US government works — public domain |
| NHS / Every Mind Matters | Open Government Licence v3.0 |
| WHO guidance | CC BY-NC-SA 3.0 IGO |
| NOBA psychology chapters | CC BY-NC-SA 4.0 |
| Stoic & contemplative classics (Marcus Aurelius, Epictetus, Dhammapada, Bhagavad-Gita, William James, James Allen) | Public domain |

Modern self-help books, CBT workbooks, and the DSM are deliberately excluded —
copyrighted, and the DSM would pull the system toward the diagnosis it must
never offer. Full licence obligations: `docs/LICENSES.md`.

Chunking is heading-aware and never splits a numbered list: a five-step
breathing exercise severed at step three is two useless chunks and one bad
answer.

## Engineering notes

- **Latency is measured from the first session ever run**, per stage (STT,
  LLM, TTS time-to-first-byte) and end-to-end, persisted as JSONL with
  p50/p95 printed at session close. Targeting ~1.2s voice-to-voice p50.
- **Barge-in works**: interrupt mid-sentence and it stops and listens
  (Pipecat's interruption handling + Silero VAD + a local smart-turn model,
  so it also sits through "I feel… um…" without jumping in).
- **Voice-shaped prompting**: answers capped at 2–4 spoken sentences,
  citations woven into speech, no lists — a voice turn is a different medium
  from a chat bubble.
- **Privacy**: audio is never recorded. Transcripts and stage timings are
  logged locally for engineering analysis.
- **Two-layer safety**: the deterministic gate is the control; the system
  prompt repeats the same refusals as defence in depth. A prompt is a request;
  a gate is a guarantee.

## Setup

Requires Python 3.11+, an Azure Speech key (free tier works), and a Google AI
Studio key (free). On Intel macOS or Windows see `docs/PLATFORMS.md` first.

```bash
python3 -m venv .venv            # or: uv venv --python 3.11 .venv
.venv/bin/pip install -r requirements.txt

cp .env.example .env             # then fill in the three keys

python3 scripts/check_keys.py    # validates both keys + voice + model id
.venv/bin/python scripts/fetch_classics.py   # optional: classics raw texts
.venv/bin/python scripts/build_kb.py         # build the retrieval index
.venv/bin/python server.py       # open http://localhost:7860
```

On Windows, use the platform lock file and PowerShell equivalents instead of
the POSIX venv commands above:

```powershell
python -m venv .venv
.venv\Scripts\pip.exe install -r requirements-windows.lock
.venv\Scripts\python.exe scripts\check_keys.py
.venv\Scripts\python.exe scripts\fetch_classics.py
.venv\Scripts\python.exe scripts\build_kb.py
.venv\Scripts\python.exe server.py
```

See `docs/PLATFORMS.md` for the Intel macOS override details and browser
requirements, `docs/README_DATA.md` for the corpus layout, and
`docs/LICENSES.md` for source licence obligations.

`scripts/check_keys.py` is stdlib-only and checks the things that otherwise
fail silently mid-call: key validity, region slug, whether the configured TTS
voice exists in your region, and whether the model id resolves.

## Tests

```bash
.venv/bin/python tests/test_kb.py
```

The retrieval suite runs hermetically (a deterministic stub embedder — no
network, no model download) and covers chunk integrity, citation completeness,
fusion arithmetic, lane activation, and the repeat-suggestion penalty.
Retrieval quality against the real embedding model is spot-checked with
`scripts/build_kb.py --probe "..."`.

## Status

Working: the full voice loop (WebRTC ↔ STT ↔ LLM ↔ TTS with barge-in), the
retrieval module, the knowledge base, latency instrumentation.

In progress: the deterministic safety and scope gates (currently enforced at
prompt level only — the gate processors are the next milestone), the emotion
classifier, a scripted evaluation harness for the safety funnel (target: 100%
crisis recall on oblique phrasings before any public deployment), a live
dashboard, and deployment.

## Data attribution

Contains public sector information licensed under the Open Government Licence
v3.0 (NHS). WHO and NOBA content used under CC BY-NC-SA (non-commercial).
US federal publications are public domain. Classic texts are public domain,
via Project Gutenberg.
