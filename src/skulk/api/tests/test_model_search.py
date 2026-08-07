# pyright: reportPrivateUsage=false
"""Hugging Face repository and GGUF filename search behavior."""

from collections.abc import Iterable
from types import SimpleNamespace
from typing import cast

import pytest
from huggingface_hub import ModelInfo

from skulk.api import model_search


def _model(
    model_id: str,
    *,
    files: tuple[str, ...] = (),
    downloads: int = 0,
) -> ModelInfo:
    return cast(
        ModelInfo,
        cast(
            object,
            SimpleNamespace(
                id=model_id,
                author=model_id.split("/", 1)[0],
                downloads=downloads,
                likes=0,
                last_modified=None,
                tags=["gguf"],
                siblings=[SimpleNamespace(rfilename=path) for path in files],
                safetensors=None,
                card_data=None,
                pipeline_tag=None,
                library_name=None,
                gated=False,
            ),
        ),
    )


def test_repository_search_uses_the_standard_hub_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_list_models(**kwargs: object) -> Iterable[ModelInfo]:
        calls.append(kwargs)
        return [_model("org/Qwen-GGUF", downloads=12)]

    monkeypatch.setattr(model_search, "list_models", fake_list_models)

    results = model_search.search_hugging_face_models("qwen", 5, mlx_only=False)

    assert [result.id for result in results] == ["org/Qwen-GGUF"]
    assert results[0].matched_file is None
    assert calls == [
        {
            "search": "qwen",
            "author": None,
            "pipeline_tag": None,
            "sort": "downloads",
            "limit": 5,
            "expand": list(model_search._EXPAND_FIELDS),
        }
    ]


def test_exact_gguf_filename_finds_owning_repository(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested = "hy3-1M-MTP-IQ3_XXS.gguf"
    calls: list[dict[str, object]] = []

    def fake_list_models(**kwargs: object) -> Iterable[ModelInfo]:
        calls.append(kwargs)
        search = kwargs.get("search")
        if search == "hy3-1M":
            return [
                _model(
                    "satgeze/Hy3-1M-GGUF",
                    files=("README.md", requested),
                    downloads=50,
                )
            ]
        return []

    monkeypatch.setattr(model_search, "list_models", fake_list_models)

    results = model_search.search_hugging_face_models(requested, 20, mlx_only=False)

    assert [(result.id, result.matched_file) for result in results] == [
        ("satgeze/Hy3-1M-GGUF", requested)
    ]
    assert any(
        call.get("search") == "hy3-1M" and call.get("full") is True
        for call in calls
    )
    assert not any(call.get("search") == "hy3" for call in calls)


def test_filename_match_preserves_nested_repo_path_and_author_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested = "Qwen-Q4_K_M-00001-of-00002.gguf"
    repo_path = f"Q4_K_M/{requested}"
    authors: list[object] = []

    def fake_list_models(**kwargs: object) -> Iterable[ModelInfo]:
        authors.append(kwargs.get("author"))
        if kwargs.get("full") is True:
            return [_model("mlx-community/Qwen-GGUF", files=(repo_path,))]
        return []

    monkeypatch.setattr(model_search, "list_models", fake_list_models)

    results = model_search.search_hugging_face_models(requested, 5, mlx_only=True)

    assert results[0].matched_file == repo_path
    assert set(authors) == {"mlx-community"}


def test_filename_fallback_rejects_repositories_without_exact_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_list_models(**kwargs: object) -> Iterable[ModelInfo]:
        if kwargs.get("full") is True:
            return [
                _model(
                    "org/similar",
                    files=("hy3-1M-MTP-IQ2_M.gguf",),
                )
            ]
        return []

    monkeypatch.setattr(model_search, "list_models", fake_list_models)

    assert (
        model_search.search_hugging_face_models(
            "hy3-1M-MTP-IQ3_XXS.gguf", 20, mlx_only=False
        )
        == []
    )


def test_filename_fallback_caps_unique_manifest_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspected: set[str] = set()

    def fake_list_models(**kwargs: object) -> Iterable[ModelInfo]:
        if kwargs.get("full") is not True:
            return []
        search = str(kwargs["search"])
        limit = int(cast(int, kwargs["limit"]))
        models = [
            _model(f"org/{search}-{index}", files=("other.gguf",))
            for index in range(limit)
        ]
        inspected.update(model.id for model in models)
        return models

    monkeypatch.setattr(model_search, "list_models", fake_list_models)

    results = model_search.search_hugging_face_models(
        "long-model-name-with-many-parts-IQ3_XXS.gguf",
        20,
        mlx_only=False,
    )

    assert results == []
    assert len(inspected) == model_search._MAX_FILENAME_SEARCH_CANDIDATES


def test_trending_sort_offset_and_metadata_enrichment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty query sorts by trending, offset slices, and GGUF/card metadata maps."""
    calls: list[dict[str, object]] = []
    enriched = cast(
        ModelInfo,
        cast(
            object,
            SimpleNamespace(
                id="org/Big-GGUF",
                author="",
                downloads=7,
                likes=1,
                last_modified=None,
                tags=["gguf"],
                siblings=None,
                safetensors=None,
                card_data=SimpleNamespace(license="apache-2.0"),
                pipeline_tag="text-generation",
                library_name=None,
                gated="manual",
                gguf={
                    "total": 8_000_000_000,
                    "totalFileSize": 5_000_000_000,
                    "context_length": 131072,
                },
            ),
        ),
    )

    def fake_list_models(**kwargs: object) -> Iterable[ModelInfo]:
        calls.append(kwargs)
        return [_model("org/skipped", downloads=1), enriched]

    monkeypatch.setattr(model_search, "list_models", fake_list_models)

    results = model_search.search_hugging_face_models("", 1, mlx_only=False, offset=1)

    assert calls[0]["sort"] == "trending_score"
    assert calls[0]["limit"] == 2  # requested page plus the skipped offset
    assert [result.id for result in results] == ["org/Big-GGUF"]
    result = results[0]
    assert result.author == "org"  # backfilled from the repo id
    assert result.gated is True
    assert result.license == "apache-2.0"
    assert result.pipeline_tag == "text-generation"
    assert result.param_count == 8_000_000_000
    assert result.total_file_size == 5_000_000_000
    assert result.context_length == 131072


def test_gguf_quant_options_group_and_exclude_companions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Quant options group shard families and never offer companion artifacts."""

    def fake_siblings(model_id: object, source_revision: object = None) -> list[tuple[str, int]]:
        return [
            ("dspark-Big-Q8_0.gguf", 10),
            ("UD-Q2_K_XL/Big-UD-Q2_K_XL-00001-of-00002.gguf", 50),
            ("UD-Q2_K_XL/Big-UD-Q2_K_XL-00002-of-00002.gguf", 40),
            ("Big-Q4_K_M.gguf", 30),
        ]

    monkeypatch.setattr(model_search, "gguf_weight_siblings", fake_siblings)

    options = model_search.list_gguf_quant_options("org/Big-GGUF")

    assert [(o.label, o.total_bytes, o.shard_count) for o in options] == [
        ("Q4_K_M", 30, 1),
        ("UD-Q2_K_XL", 90, 2),
    ]
    assert options[1].gguf_file == "UD-Q2_K_XL/Big-UD-Q2_K_XL-00001-of-00002.gguf"
    assert all("dspark" not in o.gguf_file for o in options)
