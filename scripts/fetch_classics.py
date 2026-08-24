#!/usr/bin/env python3
"""Fetch the public-domain classics used by the knowledge base.

    python3 scripts/fetch_classics.py

Downloads plain-text editions from a Project Gutenberg mirror
(gutenberg.pglaf.org) into data/classics/ and strips the licence
header/footer boilerplate. The main gutenberg.org site rate-limits and
sometimes blocks automated access; the pglaf mirror serves the same files.

The raw .txt files are inputs, not corpus: the retrievable corpus is the
curated data/classics/*.md files, which hold selected passages with
front-matter. Re-run this only to re-curate or add books.

Stdlib only. If HTTPS fails with a certificate error, either run
"Install Certificates.command" from your python.org install, or set
SSL_CERT_FILE to a CA bundle that includes your network's proxy root.
"""

from __future__ import annotations

import re
import sys
import urllib.request
from pathlib import Path

# (gutenberg ebook id, save slug). Verify new additions on gutenberg.org
# and record the translation in the curated .md front-matter.
BOOKS = [
    (2680, "meditations_marcus_aurelius"),      # Casaubon translation
    (45109, "enchiridion_epictetus"),           # Higginson translation
    (4507, "as_a_man_thinketh_james_allen"),
    (2017, "dhammapada_muller"),                # Müller translation
    (2388, "bhagavad_gita_song_celestial"),     # Edwin Arnold
    (16287, "talks_to_teachers_william_james"), # incl. The Gospel of Relaxation
]

MIRROR = "https://gutenberg.pglaf.org/cache/epub/{id}/pg{id}.txt"
OUT = Path(__file__).resolve().parent.parent / "data" / "classics"

_START = re.compile(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.I | re.S)
_END = re.compile(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.I | re.S)


def strip_boilerplate(text: str) -> str:
    start, end = _START.search(text), _END.search(text)
    return text[start.end():end.start()].strip() if start and end else text.strip()


def fetch(book_id: int) -> str:
    url = MIRROR.format(id=book_id)
    request = urllib.request.Request(url, headers={"User-Agent": "SaathiKB/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read().decode("utf-8", errors="ignore")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    failures = 0
    for book_id, slug in BOOKS:
        dest = OUT / f"{slug}.txt"
        if dest.exists():
            print(f"  skip (exists): {slug}")
            continue
        try:
            text = strip_boilerplate(fetch(book_id))
            dest.write_text(text, encoding="utf-8")
            print(f"  OK: #{book_id} -> {dest.name} ({len(text) // 1000} KB)")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  FAILED: #{book_id} {slug}: {exc}")
    if failures:
        print(f"\n{failures} download(s) failed — retry, or fetch manually from gutenberg.org")
    print("\nSpot-check a couple of files: the title inside must match the slug.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
