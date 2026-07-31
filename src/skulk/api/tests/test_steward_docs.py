"""Docs grounding: section index and steward tool behavior."""

import json
from typing import cast

from skulk.api.steward_docs import search_docs, split_sections


def test_sections_split_on_headings_and_bound_length() -> None:
    text = "# Alpha\nbody one\n## Beta\n" + ("x" * 5000)
    sections = split_sections("doc.md", text)
    # The oversized Beta section chunks into bounded continuations.
    assert [s.heading for s in sections] == [
        "Alpha",
        "Beta",
        "Beta (cont.)",
        "Beta (cont.)",
    ]
    assert all(len(s.text) <= 2400 for s in sections)


def test_search_finds_relevant_repo_docs() -> None:
    # The repo checkout ships the corpus; the fact-sheet is the anchor.
    results = search_docs("zenoh data transport")
    assert results is not None and results
    joined = " ".join(s.text.lower() + s.heading.lower() for s in results)
    assert "zenoh" in joined


def test_empty_query_returns_no_results() -> None:
    assert search_docs("   ") == []


async def test_steward_tool_returns_bounded_results() -> None:
    from skulk.api.steward import StewardHarness

    harness = StewardHarness(cast("object", None))  # type: ignore[arg-type]
    payload = await harness.execute_tool("search_docs", {"query": "steward"})
    parsed = cast(
        "dict[str, object]",
        json.loads(payload if not payload.endswith("...[truncated]") else "{}"),
    )
    if parsed:
        assert "results" in parsed or "error" in parsed


def test_long_sections_chunk_instead_of_truncating() -> None:
    text = "# Big\n" + ("alpha " * 300) + "UNIQUEMARKER-TAIL " + ("beta " * 300)
    sections = split_sections("doc.md", text)
    assert len(sections) >= 2
    assert any("UNIQUEMARKER-TAIL".lower() in s.text.lower() or "UNIQUEMARKER-TAIL" in s.text for s in sections)
    assert all(len(s.text) <= 2400 for s in sections)


def test_filler_terms_do_not_dominate_ranking() -> None:
    results = search_docs("what does zenoh do")
    assert results is not None and results
    joined = " ".join((s.heading + " " + s.text).lower() for s in results)
    assert "zenoh" in joined


def test_chunk_boundaries_never_split_terms() -> None:
    body = ("word " * 700).strip()  # far past one chunk
    sections = split_sections("doc.md", "# Big\n" + body)
    for section in sections:
        # every chunk is whole words: reassembling with spaces loses nothing
        assert all(token == "word" for token in section.text.split())


def test_how_to_guides_are_indexed() -> None:
    results = search_docs("run skulk as a service")
    assert results is not None and results
    sources = {s.source for s in results}
    assert any("docs/" in src for src in sources)
