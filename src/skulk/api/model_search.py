"""Hugging Face repository and GGUF filename search helpers."""

import re
from pathlib import PurePosixPath

from huggingface_hub import ModelInfo, list_models

from skulk.api.types import HuggingFaceSearchResult

_GGUF_EXTENSION = ".gguf"
_GGUF_SHARD_SUFFIX = re.compile(r"-\d{5}-of-\d{5}$", re.IGNORECASE)
_MIN_FILENAME_SEARCH_TERM_LENGTH = 4
_MAX_FILENAME_SEARCH_CANDIDATES = 100


def _to_search_result(
    model: ModelInfo, *, matched_file: str | None = None
) -> HuggingFaceSearchResult:
    return HuggingFaceSearchResult(
        id=model.id,
        author=model.author or "",
        downloads=model.downloads or 0,
        likes=model.likes or 0,
        last_modified=str(model.last_modified or ""),
        tags=list(model.tags or []),
        matched_file=matched_file,
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
    query: str, limit: int, mlx_only: bool
) -> list[HuggingFaceSearchResult]:
    """Search repositories and resolve exact GGUF filenames when requested.

    Hugging Face's model search indexes repository metadata, not repository file
    manifests. A pasted GGUF filename can therefore return no results even when
    the file exists. Filename-shaped queries get a bounded fallback: progressively
    broaden the model-name prefix, fetch full metadata only for those candidate
    repositories, and retain repositories whose manifest contains the exact
    basename. Ordinary text searches keep the single upstream repository query.

    Args:
        query: User-entered repository text or GGUF filename.
        limit: Maximum number of repositories to return.
        mlx_only: Restrict both search paths to the ``mlx-community`` author.

    Returns:
        Search results, with ``matched_file`` set for exact GGUF filename matches.
    """
    if limit <= 0:
        return []

    author = "mlx-community" if mlx_only else None
    primary_models = list(
        list_models(
            search=query or None,
            author=author,
            sort="downloads",
            limit=limit,
        )
    )

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
