"""Strip Project Gutenberg header/footer boilerplate from .txt files.
Usage: python3 clean_gutenberg.py <folder>
Writes <name>_clean.txt next to each file. The *** START/END *** markers
delimit the licensed boilerplate — everything outside them must not be ingested.
"""
import re, sys
from pathlib import Path

def clean(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    start = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", text, re.I)
    end = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", text, re.I)
    if start and end:
        text = text[start.end():end.start()]
    out = path.with_name(path.stem + "_clean.txt")
    out.write_text(text.strip(), encoding="utf-8")
    print(f"cleaned {path.name} -> {out.name} ({len(text)} chars)")

if __name__ == "__main__":
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    for f in folder.glob("*.txt"):
        if not f.stem.endswith("_clean"):
            clean(f)
