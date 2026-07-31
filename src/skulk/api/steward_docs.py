"""Docs grounding for the steward: a dependency-free section index.

Splits the repository's own documentation into heading-delimited sections
and scores them against a query with a hand-rolled tf-idf, so the steward
can answer Explain/Guide questions from the running checkout's docs rather
than from base-model priors. Version-correct by construction: the index
reads whatever documentation ships with the running code, so it can never
describe a different release. No embedding model or external dependency is
involved, deliberately: grounding must work on a cluster with nothing else
mounted.

The index degrades honestly: installations without the documentation files
(a wheel install outside a repository checkout) report that absence as the
tool result instead of pretending an empty corpus is knowledge.
"""

from __future__ import annotations

import math
import re
import threading
from dataclasses import dataclass
from pathlib import Path

MAX_SECTION_CHARS = 2400
"""Sections are truncated to stay digestible in a small model's context."""

MAX_RESULTS = 4
"""Result budget per query."""

_WORD_RE = re.compile(r"[a-z0-9_./-]+")

_QUERY_STOPWORDS = frozenset(
    """a an and are can d do does doing for from how i in is it its of on or
    s t that the this to was we what when where which who why will with you
    your""".split()
)
"""Filler terms dropped from queries so 'what does zenoh do' ranks on
'zenoh' rather than on sections dense in common words. Applied to queries
only; section indexing keeps every term."""

_DOC_SOURCES: tuple[str, ...] = (
    "website/docs/architecture-reference.md",
    "website/docs/api-guide.md",
    "website/docs/architecture.md",
    "website/docs/node-doctor.md",
    "README.md",
)
"""Repo-relative documentation files indexed, most fact-dense first.

architecture-reference.md exists specifically as the LLM-readable fact
sheet, which makes it the corpus anchor.
"""


@dataclass(frozen=True)
class DocSection:
    """One heading-delimited slice of a documentation file."""

    source: str
    heading: str
    text: str


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def split_sections(source: str, text: str) -> list[DocSection]:
    """Split a markdown document on headings into bounded sections."""
    sections: list[DocSection] = []
    current_heading = source
    current_lines: list[str] = []

    def flush() -> None:
        body = "\n".join(current_lines).strip()
        if not body:
            return
        # Long sections are chunked, not truncated: truncating before the
        # index is built would make every fact after the cutoff
        # unsearchable rather than merely bounding returned context.
        for offset in range(0, len(body), MAX_SECTION_CHARS):
            chunk = body[offset : offset + MAX_SECTION_CHARS]
            heading = (
                current_heading
                if offset == 0
                else f"{current_heading} (cont.)"
            )
            sections.append(
                DocSection(source=source, heading=heading, text=chunk)
            )

    for line in text.splitlines():
        if line.startswith("#"):
            flush()
            current_heading = line.lstrip("# ").strip() or source
            current_lines = []
        else:
            current_lines.append(line)
    flush()
    return sections


def _repo_root() -> Path | None:
    """The repository root containing the docs, if this install has one.

    Walks up from this module: a checkout places it at
    ``<root>/src/skulk/api/steward_docs.py``, so the root is four parents
    up. A wheel install has no ``website/`` and returns None.
    """
    candidate = Path(__file__).resolve().parent.parent.parent.parent
    if (candidate / "website" / "docs").is_dir():
        return candidate
    return None


class DocsIndex:
    """Lazily built, process-cached tf-idf index over the doc sections."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._built = False
        self._sections: list[DocSection] = []
        self._term_frequencies: list[dict[str, float]] = []
        self._document_frequency: dict[str, int] = {}

    def _build(self) -> None:
        root = _repo_root()
        if root is None:
            return
        for relative in _DOC_SOURCES:
            path = root / relative
            if not path.is_file():
                continue
            try:
                text = path.read_text(errors="replace")
            except OSError:
                continue
            self._sections.extend(split_sections(relative, text))
        for section in self._sections:
            tokens = _tokenize(section.heading + " " + section.text)
            frequencies: dict[str, float] = {}
            for token in tokens:
                frequencies[token] = frequencies.get(token, 0.0) + 1.0
            length = max(1.0, float(len(tokens)))
            self._term_frequencies.append(
                {term: count / length for term, count in frequencies.items()}
            )
            for term in frequencies:
                self._document_frequency[term] = (
                    self._document_frequency.get(term, 0) + 1
                )

    def search(self, query: str) -> list[DocSection] | None:
        """Top sections for the query, or None when no corpus exists."""
        with self._lock:
            if not self._built:
                self._build()
                self._built = True
        if not self._sections:
            return None
        query_terms = [
            term for term in _tokenize(query) if term not in _QUERY_STOPWORDS
        ]
        if not query_terms:
            return []
        total = len(self._sections)
        scored: list[tuple[float, int]] = []
        for index, frequencies in enumerate(self._term_frequencies):
            score = 0.0
            for term in query_terms:
                term_frequency = frequencies.get(term)
                if term_frequency is None:
                    continue
                document_frequency = self._document_frequency.get(term, total)
                score += term_frequency * math.log(
                    1.0 + total / document_frequency
                )
            if score > 0.0:
                scored.append((score, index))
        scored.sort(reverse=True)
        return [self._sections[index] for _score, index in scored[:MAX_RESULTS]]


_INDEX = DocsIndex()


def search_docs(query: str) -> list[DocSection] | None:
    """Module-level search over the process-cached index."""
    return _INDEX.search(query)
