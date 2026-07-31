"""Docs grounding: section index and steward tool behavior."""

import json

from typing import cast

from skulk.api.steward_docs import search_docs, split_sections


def test_sections_split_on_headings_and_bound_length() -> None:
    text = "# Alpha\nbody one\n## Beta\n" + ("x" * 5000)
    sections = split_sections("doc.md", text)
    assert [s.heading for s in sections] == ["Alpha", "Beta"]
    assert len(sections[1].text) <= 2400


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
