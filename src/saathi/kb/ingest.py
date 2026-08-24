"""Turn the KB markdown files into retrievable, attributable chunks.

Two jobs, and the second one is the interesting one.

FRONT-MATTER. Every doc carries `source`, `licence`, `type` and `topics[]`.
`source` is not bookkeeping — it is what Saathi says out loud ("this comes from
the NHS self-help guide on sleep"), so it has to survive chunking intact. A
chunk that can't name its origin is a chunk she isn't allowed to use.

CHUNKING. Fixed-width splitting would be wrong here. These documents are
mostly technique instructions, and the NHS breathing exercise is five numbered
steps — cutting between step 3 and step 4 produces two chunks that are each
useless and one answer that's actively bad advice. So blocks are atomic: a
paragraph or a whole list stays whole, and chunks are assembled from blocks
until they reach target size. A slightly oversized chunk beats a severed
instruction.

Target 300-800 chars per the plan. Short enough that a voice turn can be built
from one or two, long enough to carry a complete technique.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

MIN_CHARS = 300
MAX_CHARS = 800

_FRONT_MATTER = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.S)
_LIST_LINE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Chunk:
    """One retrievable passage, carrying everything needed to cite it."""

    chunk_id: str
    doc_id: str
    text: str
    title: str
    source: str
    url: str
    licence: str
    doc_type: str  # technique | education | wisdom
    topics: list[str] = field(default_factory=list)
    heading: str | None = None

    @property
    def char_len(self) -> int:
        return len(self.text)

    def citation(self) -> str:
        """What Saathi says when she attributes this. Spoken, so no URLs."""
        return self.source

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(**data)


class IngestError(ValueError):
    pass


def parse_front_matter(raw: str, path: Path) -> tuple[dict[str, object], str]:
    """Split a doc into (metadata, body).

    Deliberately a narrow parser rather than a YAML dependency: the schema is
    ours, documented in docs/README_DATA.md, and only two shapes occur — scalar
    strings and JSON-style lists. A real YAML parser here would accept far more
    than we want and hide typos in files we control.
    """
    match = _FRONT_MATTER.match(raw)
    if not match:
        raise IngestError(f"{path.name}: no YAML front-matter block at the top of the file")

    meta: dict[str, object] = {}
    for line in match.group(1).splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise IngestError(f"{path.name}: front-matter line is not 'key: value' -> {line!r}")
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if value.startswith("["):
            try:
                meta[key] = json.loads(value)
            except json.JSONDecodeError as exc:
                raise IngestError(f"{path.name}: '{key}' is not a valid JSON list -> {value!r}") from exc
        else:
            meta[key] = value.strip("'\"")

    body = raw[match.end():].strip()
    if not body:
        raise IngestError(f"{path.name}: front-matter present but the body is empty")
    return meta, body


def _blocks(body: str) -> list[tuple[str | None, str]]:
    """Split a body into (heading, block) pairs.

    A block is a paragraph, or a run of consecutive list items treated as one
    unit. Keeping lists intact is the whole point — see the module docstring.
    """
    out: list[tuple[str | None, str]] = []
    heading: str | None = None
    pending: list[str] = []
    in_list = False

    def flush():
        nonlocal pending, in_list
        if pending:
            text = "\n".join(pending).strip()
            if text:
                out.append((heading, text))
        pending = []
        in_list = False

    for line in body.splitlines():
        stripped = line.strip()

        head = _HEADING.match(stripped)
        if head:
            flush()
            heading = head.group(2).strip()
            continue

        if not stripped:
            # Blank line ends a paragraph, but not a list that continues after it.
            if not in_list:
                flush()
            continue

        is_item = bool(_LIST_LINE.match(line))
        if is_item and not in_list:
            # A list starting mid-paragraph: keep the lead-in ("The exercise:")
            # attached, since it's what makes the steps make sense.
            in_list = True
        elif in_list and not is_item and not line.startswith((" ", "\t")):
            flush()

        pending.append(stripped)

    flush()
    return out


def chunk_document(
    meta: dict[str, object],
    body: str,
    doc_id: str,
    min_chars: int = MIN_CHARS,
    max_chars: int = MAX_CHARS,
) -> list[Chunk]:
    """Assemble blocks into chunks of roughly min..max chars."""
    required = ("title", "source", "licence")
    missing = [k for k in required if not meta.get(k)]
    if missing:
        raise IngestError(f"{doc_id}: front-matter missing required field(s): {', '.join(missing)}")

    topics = meta.get("topics") or []
    if not isinstance(topics, list):
        raise IngestError(f"{doc_id}: 'topics' must be a list, got {type(topics).__name__}")

    chunks: list[Chunk] = []
    current: list[str] = []
    current_heading: str | None = None
    size = 0

    def flush():
        nonlocal current, size, current_heading
        if not current:
            return
        text = "\n\n".join(current).strip()
        chunks.append(
            Chunk(
                chunk_id=f"{doc_id}#{len(chunks):03d}",
                doc_id=doc_id,
                text=text,
                title=str(meta["title"]),
                source=str(meta["source"]),
                url=str(meta.get("url", "")),
                licence=str(meta["licence"]),
                doc_type=str(meta.get("type", "education")),
                topics=[str(t) for t in topics],
                heading=current_heading,
            )
        )
        current = []
        size = 0

    for heading, block in _blocks(body):
        # A heading change starts a new chunk: mixing sections dilutes both.
        if current and heading != current_heading and size >= min_chars:
            flush()
        if not current:
            current_heading = heading

        # Adding this block would overflow, and we already have enough — flush.
        if current and size + len(block) > max_chars and size >= min_chars:
            flush()
            current_heading = heading

        current.append(block)
        size += len(block) + 2

    flush()

    # A trailing scrap is better merged backwards than left as a chunk too small
    # to carry meaning on its own.
    if len(chunks) > 1 and chunks[-1].char_len < min_chars // 2:
        tail = chunks.pop()
        merged = chunks[-1]
        merged.text = f"{merged.text}\n\n{tail.text}"

    return chunks


def load_file(path: Path) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8")
    meta, body = parse_front_matter(raw, path)
    return chunk_document(meta, body, doc_id=path.stem)


def load_corpus(roots: list[Path]) -> list[Chunk]:
    """Ingest every .md under the given directories, sorted for determinism.

    Determinism matters: chunk ids feed the index, the citations, and the eval
    harness. A corpus that reorders between builds makes eval results
    incomparable run to run.
    """
    chunks: list[Chunk] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.md")):
            chunks.extend(load_file(path))
    return chunks
