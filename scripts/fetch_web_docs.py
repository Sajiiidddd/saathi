#!/usr/bin/env python3
"""Fetch open-licence web pages and extract their main content as text.

    python3 scripts/fetch_web_docs.py

Writes raw extractions to data/_staging/<slug>.txt for review. These are NOT
corpus files: the corpus is curated .md with front-matter, produced by a human
pass over the staging text (trim residual navigation, keep the substance,
verify the licence). The staging step exists so what enters the KB is the
source's actual wording, reviewed — not a paraphrase from memory.

Only add URLs whose licence is verified: NHS (OGL v3.0), US federal health
agencies (public domain), WHO (CC BY-NC-SA, per document). Stdlib only.
"""

from __future__ import annotations

import ssl
import sys
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAGING = ROOT / "data" / "_staging"

# (url, slug) — licence and framing are decided at curation time.
PAGES = [
    ("https://www.nhs.uk/every-mind-matters/mental-health-issues/anxiety/",
     "nhs_emm_anxiety_selfhelp"),
    ("https://www.nhs.uk/every-mind-matters/mental-health-issues/low-mood/",
     "nhs_emm_low_mood_selfhelp"),
    ("https://www.nhs.uk/every-mind-matters/mental-health-issues/stress/",
     "nhs_emm_stress_selfhelp"),
    ("https://www.nhs.uk/mental-health/self-help/tips-and-support/mindfulness/",
     "nhs_mindfulness"),
    ("https://www.nhs.uk/mental-health/conditions/panic-disorder/",
     "nhs_panic_disorder"),
    ("https://www.nccih.nih.gov/health/relaxation-techniques-what-you-need-to-know",
     "nccih_relaxation_techniques"),
]

SKIP_TAGS = {"script", "style", "nav", "header", "footer", "aside", "form", "button", "svg"}


class MainTextExtractor(HTMLParser):
    """Readability-lite: text inside <main> (fallback: <article>, then <body>),
    with h2/h3 rendered as markdown headings and list items as bullets."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.in_main = False
        self.depth = 0
        self.skip_depth = 0
        self.heading: str | None = None
        self.in_li = False
        self.parts: list[str] = []
        self.buffer: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "main":
            self.in_main = True
            self.depth = 0
        if not self.in_main:
            return
        self.depth += 1
        if tag in SKIP_TAGS:
            self.skip_depth += 1
        if self.skip_depth:
            return
        if tag in ("h2", "h3"):
            self._flush()
            self.heading = tag
        elif tag == "li":
            self._flush()
            self.in_li = True
        elif tag == "p":
            self._flush()

    def handle_endtag(self, tag):
        if not self.in_main:
            return
        if tag in SKIP_TAGS and self.skip_depth:
            self.skip_depth -= 1
        if tag in ("h2", "h3", "p", "li"):
            self._flush()
            self.heading = None
            self.in_li = False
        if tag == "main":
            self._flush()
            self.in_main = False

    def handle_data(self, data):
        if self.in_main and not self.skip_depth:
            self.buffer.append(data)

    def _flush(self):
        text = " ".join(" ".join(self.buffer).split())
        self.buffer = []
        if not text or len(text) < 3:
            return
        if self.heading == "h2":
            self.parts.append(f"\n## {text}\n")
        elif self.heading == "h3":
            self.parts.append(f"\n### {text}\n")
        elif self.in_li:
            self.parts.append(f"- {text}")
        else:
            self.parts.append(text)

    def text(self) -> str:
        return "\n".join(self.parts).strip()


def fetch(url: str) -> str:
    import os

    if os.getenv("SSL_CERT_FILE"):
        # An explicit bundle (e.g. one including a corporate proxy root)
        # outranks certifi — create_default_context honours the env var.
        context = ssl.create_default_context()
    else:
        try:
            import certifi

            context = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            context = None
    request = urllib.request.Request(url, headers={"User-Agent": "SaathiKB/1.0 (research)"})
    with urllib.request.urlopen(request, timeout=60, context=context) as response:
        return response.read().decode("utf-8", errors="ignore")


def main() -> int:
    STAGING.mkdir(parents=True, exist_ok=True)
    failures = 0
    for url, slug in PAGES:
        try:
            parser = MainTextExtractor()
            parser.feed(fetch(url))
            text = parser.text()
            if len(text) < 400:
                print(f"  THIN  {slug}: only {len(text)} chars — check extraction")
                failures += 1
            (STAGING / f"{slug}.txt").write_text(f"SOURCE: {url}\n\n{text}", encoding="utf-8")
            print(f"  OK    {slug}  ({len(text)//1000}KB extracted)")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  FAIL  {slug}: {exc}")
    print(f"\nStaged under {STAGING} — review, then curate into data/ with front-matter.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
