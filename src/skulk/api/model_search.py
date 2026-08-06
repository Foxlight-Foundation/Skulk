"""Hugging Face repository and GGUF filename search helpers."""

import re
from pathlib import PurePosixPath
from typing import cast

from huggingface_hub import ModelInfo, list_models

from skulk.api.types import HuggingFaceSearchResult

_GGUF_EXTENSION = ".gguf"
_GGUF_SHARD_SUFFIX = re.compile(r"-\d{5}-of-\d{5}$", re.IGNORECASE)
_MIN_FILENAME_SEARCH_TERM_LENGTH = 4
_MAX_FILENAME_SEARCH_CANDIDATES = 100


# Enrichment requested from the list endpoint. Deliberately narrow: the
# ``gguf`` expansion also ships each repo's full chat template, which is
# stripped before the result leaves this module.
_EXPAND_FIELDS = (
    "downloads",
    "likes",
    "lastModified",
    "tags",
    "pipeline_tag",
    "library_name",
    "gated",
    "cardData",
    "safetensors",
    "gguf",
)


def _to_search_result(
    model: ModelInfo, *, matched_file: str | None = None
) -> HuggingFaceSearchResult:
    """Map one Hugging Face listing onto the dashboard's search-result shape."""
    param_count: int | None = None
    if model.safetensors is not None and model.safetensors.total:
        param_count = int(model.safetensors.total)

    gguf_raw = cast(object, getattr(model, "gguf", None))
    gguf: dict[str, object] = (
        cast("dict[str, object]", gguf_raw) if isinstance(gguf_raw, dict) else {}
    )
    gguf_total = gguf.get("total")
    if param_count is None and isinstance(gguf_total, int):
        param_count = gguf_total
    total_file_size = gguf.get("totalFileSize")
    context_length = gguf.get("context_length")

    license_id: str | None = None
    if model.card_data is not None:
        license_value = cast(object, getattr(model.card_data, "license", None))
        if isinstance(license_value, str):
            license_id = license_value

    author = model.author or ""
    if not author and "/" in model.id:
        author = model.id.split("/", 1)[0]

    return HuggingFaceSearchResult(
        id=model.id,
        author=author,
        downloads=model.downloads or 0,
        likes=model.likes or 0,
        last_modified=str(model.last_modified or ""),
        tags=list(model.tags or []),
        matched_file=matched_file,
        pipeline_tag=model.pipeline_tag,
        library_name=model.library_name,
        # ``gated`` is False, "auto", or "manual"; any truthy value means the
        # user must accept the license and present a token before downloading.
        gated=bool(model.gated),
        license=license_id,
        param_count=param_count,
        total_file_size=total_file_size if isinstance(total_file_size, int) else None,
        context_length=context_length if isinstance(context_length, int) else None,
    )


def _gguf_filename(query: str) -> str | None:
    normalized = query.strip().replace("\\", "/")
    filename = PurePosixPath(normalized).name
    if not filename.casefold().endswith(_GGUF_EXTENSION):
        return None
    return filename


def _gguf_repository_search_terms(filename: str) -> tuple[str, ...]:
    stem = filename[: -len(_GGUF_EXTENSION)]
    stem = _GGUF_SHARD_SUFFIX.sub("", stem)
    parts = tuple(part for part in stem.split("-") if part)

    terms: list[str] = []
    for end in range(len(parts), 0, -1):
        term = "-".join(parts[:end])
        if len(term) < _MIN_FILENAME_SEARCH_TERM_LENGTH or term in terms:
            continue
        terms.append(term)
    return tuple(terms)


def _matching_file(model: ModelInfo, requested_filename: str) -> str | None:
    requested = requested_filename.casefold()
    for sibling in model.siblings or ():
        repo_path = sibling.rfilename
        if PurePosixPath(repo_path).name.casefold() == requested:
            return repo_path
    return None


def search_hugging_face_models(
    query: str, limit: int, mlx_only: bool, offset: int = 0
) -> list[HuggingFaceSearchResult]:
    """Search repositories and resolve exact GGUF filenames when requested.

    Hugging Face's model search indexes repository metadata, not repository file
    manifests. A pasted GGUF filename can therefore return no results even when
    the file exists. Filename-shaped queries get a bounded fallback: progressively
    broaden the model-name prefix, fetch full metadata only for those candidate
    repositories, and retain repositories whose manifest contains the exact
    basename. Ordinary text searches keep the single upstream repository query.

    An empty query sorts by Hugging Face's trending score (what is hot right
    now); a text query sorts by downloads. Results carry the additive metadata
    the dashboard's discovery rows surface: task and library tags, license and
    gating, parameter counts, and exact GGUF artifact sizes when reported.

    Args:
        query: User-entered repository text or GGUF filename.
        limit: Maximum number of repositories to return.
        mlx_only: Restrict both search paths to the ``mlx-community`` author.
        offset: Number of leading results to skip, for "show more" paging.

    Returns:
        Search results, with ``matched_file`` set for exact GGUF filename matches.
    """
    if limit <= 0:
        return []

    author = "mlx-community" if mlx_only else None
    # The list endpoint pages by cursor with no offset parameter; emulate
    # offset by over-fetching one bounded window and slicing.
    primary_models = list(
        list_models(
            search=query or None,
            author=author,
            sort="downloads" if query else "trending_score",
            limit=limit + max(offset, 0),
            expand=list(_EXPAND_FIELDS),
        )
    )[max(offset, 0):]

    filename = _gguf_filename(query)
    if filename is None:
        return [_to_search_result(model) for model in primary_models]

    candidate_limit = min(
        max(limit * 2, 20),
        _MAX_FILENAME_SEARCH_CANDIDATES,
    )
    inspected_repositories: set[str] = set()
    exact_matches: dict[str, tuple[ModelInfo, str]] = {}
    for term in _gguf_repository_search_terms(filename):
        remaining_candidates = (
            _MAX_FILENAME_SEARCH_CANDIDATES - len(inspected_repositories)
        )
        if remaining_candidates <= 0:
            break
        candidates = list_models(
            search=term,
            author=author,
            sort="downloads",
            limit=min(candidate_limit, remaining_candidates),
            full=True,
        )
        for model in candidates:
            if model.id in inspected_repositories:
                continue
            inspected_repositories.add(model.id)
            matched_file = _matching_file(model, filename)
            if matched_file is not None:
                exact_matches[model.id] = (model, matched_file)
        # The first tier that produces an exact manifest match is the most
        # specific useful repository query. Stop there instead of broadening to
        # short, high-cardinality terms after the requested file is already found.
        if exact_matches:
            break

    if exact_matches:
        ordered_matches = sorted(
            exact_matches.values(),
            key=lambda match: match[0].downloads or 0,
            reverse=True,
        )
        return [
            _to_search_result(model, matched_file=matched_file)
            for model, matched_file in ordered_matches[:limit]
        ]

    return [_to_search_result(model) for model in primary_models]
