# Saathi v0.2.0

Public release of the current Saathi codebase.

## What shipped

- FastAPI + WebRTC voice companion server
- Browser talk page with captions, settings, and crisis takeover flow
- Grounded retrieval over the curated public-licence corpus
- Deterministic safety gate for crisis, injection, and scope handling
- Session store, opt-in memory, latency instrumentation, and dashboard
- Docker and deployment guidance for local and cloud runs

## Release boundary

- Private notes, design archives, and local agent config stay out of the public tree.
- `.env` remains gitignored; `.env.example` documents the required settings.

## Verification

- Repo was pushed cleanly to `origin/main`.
- No tracked `.zip` or `.pdf` files are present in the public tree.
