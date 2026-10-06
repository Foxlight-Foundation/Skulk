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

MAX_RESULT_TEXT_CHARS = 1200
"""Per-result text budget in tool output: four results plus JSON overhead
must fit the harness's 6000-char tool-result cap without mid-JSON
truncation. Indexing keeps the full section; only returned context is
sliced."""

_WORD_RE = re.compile(r"[a-z0-9_./-]+")

_QUERY_STOPWORDS = frozenset(
    ["a", "an", "and", "are", "can", "d", "do", "does", "doing", "for", "from", "how", "i", "in", "is", "it", "its", "of", "on", "or", "s", "t", "that", "the", "this", "to", "was", "we", "what", "when", "where", "which", "who", "why", "will", "with", "you", "your"]
)
"""Filler terms dropped from queries so 'what does zenoh do' ranks on
'zenoh' rather than on sections dense in common words. Applied to queries
only; section indexing keeps every term."""

_ANCHOR_SOURCES: tuple[str, ...] = (
    "website/docs/architecture-reference.md",
    "website/docs/api-guide.md",
)
"""Indexed first: architecture-reference.md exists specifically as the
LLM-readable fact sheet, making it the corpus anchor."""


def _doc_sources(root: Path) -> list[str]:
    """Every top-level docs page plus the README, anchors first.

    Globbing instead of an allowlist keeps dedicated how-to guides (service
    setup, logging, and future pages) searchable without maintenance;
    generated content lives in gitignored subdirectories and is untouched
    by the top-level glob.
    """
    sources = [s for s in _ANCHOR_SOURCES if (root / s).is_file()]
    docs_dir = root / "website" / "docs"
    for path in sorted(docs_dir.glob("*.md")):
        relative = str(path.relative_to(root))
        if relative not in sources:
            sources.append(relative)
    if (root / "README.md").is_file():
        sources.append("README.md")
    return sources


@dataclass(frozen=True)
class DocSection:
    """One heading-delimited slice of a documentation file.

    Attributes:
        source: Repo-relative path of the document the slice came from.
        heading: The section's heading text; continuation chunks of an
            oversized section carry a ``"(cont.)"`` suffix.
        text: The section body, bounded to ``MAX_SECTION_CHARS``.
    """

    source: str
    heading: str
    text: str


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def split_sections(source: str, text: str) -> list[DocSection]:
    """Split a markdown document on headings into bounded sections.

    Args:
        source: Repo-relative path of the document, recorded on every
            resulting section and used as the fallback heading.
        text: The document's full markdown text.

    Returns:
        Sections in document order. A heading whose body exceeds
        ``MAX_SECTION_CHARS`` yields multiple sections, the continuations
        titled ``"<heading> (cont.)"``, so nothing becomes unsearchable.
    """
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
        offset = 0
        first = True
        while offset < len(body):
            end = min(offset + MAX_SECTION_CHARS, len(body))
            if end < len(body):
                # Cut at the last whitespace inside the window so a term
                # spanning the boundary never becomes unsearchable.
                boundary = body.rfind(" ", offset, end)
                newline_boundary = body.rfind("\n", offset, end)
                boundary = max(boundary, newline_boundary)
                if boundary > offset:
                    end = boundary
            chunk = body[offset:end].strip()
            if chunk:
                heading = current_heading if first else f"{current_heading} (cont.)"
                sections.append(
                    DocSection(source=source, heading=heading, text=chunk)
                )
            first = False
            offset = end + (1 if end < len(body) else 0)

    in_fence = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            current_lines.append(line)
        elif line.startswith("#") and not in_fence:
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
        for relative in _doc_sources(root):
            path = root / relative
            if not path.is_file():
                continue
            try:
                text = path.read_text(errors="replace")
            except OSError:
                continue
            self._sections.extend(split_sections(relative, text))
        for section in self._sections:
            # Continuation chunks skip their (repeated) heading terms so a
            # heading-matching query cannot make tiny tail chunks outrank
            # the substantive first chunk via length normalization.
            if section.heading.endswith("(cont.)"):
                tokens = _tokenize(section.text)
            else:
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
        """Rank documentation sections against a query.

        Args:
            query: Free-text search terms; common filler words are ignored.

        Returns:
            Up to ``MAX_RESULTS`` sections by descending relevance. An
            empty list means the corpus exists but nothing matched;
            ``None`` means this installation ships no documentation
            directory at all.

        Side effects:
            The first call builds the process-cached index.
        """
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
    """Search the documentation corpus for sections relevant to a query.

    Args:
        query: Free-text search terms; common filler words are ignored.

    Returns:
        Up to ``MAX_RESULTS`` sections ranked by relevance. An empty list
        means the corpus exists but nothing matched; ``None`` means this
        installation has no documentation directory at all, which callers
        must report honestly rather than treating as a silent no-match.

    Side effects:
        The first call builds a process-cached index from the checkout's
        documentation files; later calls reuse it.
    """
    return _INDEX.search(query)
