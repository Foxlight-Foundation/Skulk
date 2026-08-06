"""Hugging Face repository and GGUF filename search helpers."""

import re
from functools import lru_cache
from pathlib import PurePosixPath
from typing import cast

from huggingface_hub import (
    ModelInfo,
    hf_hub_download,  # pyright: ignore[reportUnknownVariableType]
    list_models,
)

from skulk.api.types import GgufQuantOption, HuggingFaceSearchResult
from skulk.shared.models.model_cards import (
    ModelId,
    gguf_weight_siblings,
    is_companion_gguf,
)

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
    "config",
)


_BASE_MODEL_TAG = re.compile(r"^base_model:(finetune|quantized|merge|adapter):(.+)$")
_ARXIV_TAG = re.compile(r"^arxiv:(\d{4}\.\d{4,5})$")
_LANGUAGE_TAG = re.compile(r"^[a-z]{2}$")
_MAX_LANGUAGES = 6

_FRONTMATTER = re.compile(r"^---\n.*?\n---\n", re.S)
_HTML_TAG = re.compile(r"<[^>]+>")
_MARKDOWN_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_CARD_SUMMARY_MAX_CHARS = 700


def extract_card_summary(markdown: str) -> str:
    """First prose paragraphs of a model card, stripped of markup.

    The model card README is the only place Hugging Face describes what a
    model actually is, so this powers the dashboard's "what is this thing"
    popover text. Headings, tables, images, HTML, and code fences are
    dropped; short fragments are skipped so badges and one-liners don't win.
    """
    body = _FRONTMATTER.sub("", markdown)
    body = _MARKDOWN_IMAGE.sub("", body)
    body = _HTML_TAG.sub("", body)
    summary_parts: list[str] = []
    for paragraph in body.split("\n\n"):
        text = " ".join(paragraph.split())
        if len(text) < 80 or text.startswith(("#", "|", "```", "[", "-", "*")):
            continue
        summary_parts.append(text)
        if sum(len(part) for part in summary_parts) >= 300:
            break
    summary = "\n\n".join(summary_parts)
    if len(summary) > _CARD_SUMMARY_MAX_CHARS:
        summary = summary[:_CARD_SUMMARY_MAX_CHARS].rsplit(" ", 1)[0] + "…"
    return summary


_QUANT_LABEL = re.compile(r"(I?Q\d[A-Z0-9_]*|BF16|F16|F32|FP8|FP16)", re.IGNORECASE)


def list_gguf_quant_options(model_id: str) -> list[GgufQuantOption]:
    """Enumerate a GGUF repository's downloadable quantizations.

    Groups the repo's LM-weight GGUF files into shard groups, excluding
    companion artifacts (drafters, imatrix calibration files, projectors),
    and returns one option per quant with its loadable first shard, a human
    label, exact total bytes, and shard count, smallest first. Empty for a
    non-GGUF repo or one that ships only companions.
    """
    files = gguf_weight_siblings(ModelId(model_id))
    groups: dict[str, list[tuple[str, int]]] = {}
    for name, size in files:
        if is_companion_gguf(name):
            continue
        stem = name[: -len(_GGUF_EXTENSION)]
        group_key = _GGUF_SHARD_SUFFIX.sub("", stem)
        groups.setdefault(group_key, []).append((name, size))

    options: list[GgufQuantOption] = []
    for group_key, members in groups.items():
        first = min(name for name, _ in members)
        basename = group_key.rsplit("/", 1)[-1]
        directory = group_key.rsplit("/", 1)[0] if "/" in group_key else ""
        label_match = _QUANT_LABEL.search(directory or basename) or _QUANT_LABEL.search(basename)
        label = (directory or (label_match.group(1) if label_match else basename))
        options.append(
            GgufQuantOption(
                gguf_file=first,
                label=label,
                total_bytes=sum(size for _, size in members),
                shard_count=len(members),
            )
        )
    options.sort(key=lambda option: option.total_bytes)
    return options


@lru_cache(maxsize=256)
def fetch_card_summary(model_id: str) -> str:
    """Download one repository's README and extract its prose summary.

    Cached per process; exceptions propagate uncached so a transient network
    failure does not pin an empty summary.
    """
    readme_path = hf_hub_download(model_id, "README.md")
    with open(readme_path, encoding="utf-8", errors="replace") as handle:
        return extract_card_summary(handle.read())


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

    base_model_repo: str | None = None
    base_model_relation: str | None = None
    arxiv_ids: list[str] = []
    languages: list[str] = []
    for tag in model.tags or ():
        base_match = _BASE_MODEL_TAG.match(tag)
        if base_match and base_model_repo is None:
            base_model_relation = base_match.group(1)
            base_model_repo = base_match.group(2)
            continue
        arxiv_match = _ARXIV_TAG.match(tag)
        if arxiv_match:
            arxiv_ids.append(arxiv_match.group(1))
            continue
        if _LANGUAGE_TAG.match(tag) and len(languages) < _MAX_LANGUAGES:
            languages.append(tag)

    architecture: str | None = None
    config_raw = cast(object, getattr(model, "config", None))
    if isinstance(config_raw, dict):
        architectures = cast("dict[str, object]", config_raw).get("architectures")
        if isinstance(architectures, list) and architectures:
            first = cast("list[object]", architectures)[0]
            if isinstance(first, str):
                architecture = first
    if architecture is None:
        gguf_architecture = gguf.get("architecture")
        if isinstance(gguf_architecture, str):
            architecture = gguf_architecture

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
        base_model_repo=base_model_repo,
        base_model_relation=base_model_relation,
        arxiv_ids=arxiv_ids,
        languages=languages,
        architecture=architecture,
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
    query: str,
    limit: int,
    mlx_only: bool,
    offset: int = 0,
    pipeline_tag: str | None = None,
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
        pipeline_tag: Restrict results to one Hugging Face task, for example
            ``text-generation`` or ``automatic-speech-recognition``.

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
            pipeline_tag=pipeline_tag,
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
