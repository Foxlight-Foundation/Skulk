"""How an embedding model turns encoder output into one vector per text.

sentence-transformers models ship this beside their weights: ``modules.json``
lists the pipeline (Transformer, Pooling, optionally Dense and Normalize), the
pooling module's ``config.json`` picks the pooling mode, and
``sentence_bert_config.json`` caps the input length. Reading them is what makes
a model's vectors match its reference implementation. A single hard-coded
recipe (mean pooling, L2 normalization, 512 tokens) got two of the registry's
two cards wrong: bge-small pools the [CLS] token, and MiniLM truncates at 256.

Pure and torch-free, so it is unit-testable on any platform.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, cast

PoolingMode = Literal["mean", "cls", "max", "last"]

_LEGACY_MAX_TOKENS: Final = 512
"""Input cap for an artifact with no sentence-transformers metadata (the
runner's historical behavior)."""

_TRANSFORMER_MODULE: Final = "sentence_transformers.models.Transformer"
_POOLING_MODULE: Final = "sentence_transformers.models.Pooling"
_NORMALIZE_MODULE: Final = "sentence_transformers.models.Normalize"

# sentence-transformers pooling flags, mapped to the modes the runner
# implements. Other flags (mean_sqrt_len, weightedmean) and combinations of
# several, which sentence-transformers concatenates, are refused rather than
# approximated.
_POOLING_FLAGS: Final[dict[str, PoolingMode]] = {
    "pooling_mode_mean_tokens": "mean",
    "pooling_mode_cls_token": "cls",
    "pooling_mode_max_tokens": "max",
    "pooling_mode_lasttoken": "last",
}


class UnsupportedEmbeddingModelError(ValueError):
    """The artifact's embedding pipeline needs a step this runner does not implement.

    Raised at load so the placement fails loudly instead of serving vectors that
    silently disagree with the model's reference implementation.
    """


@dataclass(frozen=True)
class EmbeddingRecipe:
    """The post-encoder steps that define one model's vectors.

    Attributes:
        pooling: How token states collapse to one vector per text.
        normalize: Whether the pooled vector is L2-normalized.
        max_tokens: The input length past which text is truncated.
    """

    pooling: PoolingMode
    normalize: bool
    max_tokens: int


def resolve_embedding_recipe(
    model_dir: Path, *, model_max_positions: int | None
) -> EmbeddingRecipe:
    """Read an embedding model's recipe from its sentence-transformers metadata.

    Args:
        model_dir: The staged model directory.
        model_max_positions: The encoder's position limit
            (``max_position_embeddings``), when the config declares one. The
            truncation length never exceeds it.

    Returns:
        The model's recipe. An artifact without ``modules.json`` keeps the
        historical recipe: mean pooling, L2 normalization, 512 tokens (capped
        by ``model_max_positions``).

    Raises:
        UnsupportedEmbeddingModelError: The pipeline needs a module or pooling
            mode this runner does not implement, or the metadata is malformed.
    """
    modules_path = model_dir / "modules.json"
    if not modules_path.is_file():
        return EmbeddingRecipe(
            pooling="mean",
            normalize=True,
            max_tokens=_cap(_LEGACY_MAX_TOKENS, model_max_positions),
        )
    modules = _read_json(modules_path)
    if not isinstance(modules, list):
        raise UnsupportedEmbeddingModelError(f"{modules_path} is not a module list")
    pooling: PoolingMode | None = None
    normalize = False
    for raw_module in cast(list[object], modules):
        if not isinstance(raw_module, dict):
            raise UnsupportedEmbeddingModelError(f"{modules_path} has a malformed entry")
        module = cast(dict[str, object], raw_module)
        module_type = module.get("type")
        if module_type == _TRANSFORMER_MODULE:
            continue
        if module_type == _POOLING_MODULE:
            module_path = module.get("path")
            if not isinstance(module_path, str):
                raise UnsupportedEmbeddingModelError(
                    f"{modules_path} names a pooling module without a path"
                )
            pooling = _pooling_mode(model_dir / module_path / "config.json")
            continue
        if module_type == _NORMALIZE_MODULE:
            normalize = True
            continue
        raise UnsupportedEmbeddingModelError(
            f"embedding module {module_type!r} is not implemented (for example a "
            "Dense projection); serving without it would return vectors that "
            "disagree with the model's reference"
        )
    if pooling is None:
        raise UnsupportedEmbeddingModelError(
            f"{modules_path} has no pooling module, so the model yields token "
            "states rather than one vector per text"
        )
    return EmbeddingRecipe(
        pooling=pooling,
        normalize=normalize,
        max_tokens=_cap(_declared_max_tokens(model_dir), model_max_positions),
    )


def _pooling_mode(config_path: Path) -> PoolingMode:
    """The single pooling mode a sentence-transformers pooling config enables."""
    if not config_path.is_file():
        raise UnsupportedEmbeddingModelError(f"missing pooling config {config_path}")
    config = _read_json(config_path)
    if not isinstance(config, dict):
        raise UnsupportedEmbeddingModelError(f"{config_path} is not an object")
    enabled = [
        flag for flag, value in cast(dict[str, object], config).items()
        if flag.startswith("pooling_mode_") and value is True
    ]
    if len(enabled) != 1 or enabled[0] not in _POOLING_FLAGS:
        raise UnsupportedEmbeddingModelError(
            f"pooling {enabled or 'none'} in {config_path} is not implemented; "
            f"supported: one of {sorted(_POOLING_FLAGS)}"
        )
    return _POOLING_FLAGS[enabled[0]]


def _declared_max_tokens(model_dir: Path) -> int:
    """``sentence_bert_config.json``'s ``max_seq_length``, else the legacy cap."""
    config_path = model_dir / "sentence_bert_config.json"
    if not config_path.is_file():
        return _LEGACY_MAX_TOKENS
    config = _read_json(config_path)
    if isinstance(config, dict):
        value = cast(dict[str, object], config).get("max_seq_length")
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return _LEGACY_MAX_TOKENS


def _cap(max_tokens: int, model_max_positions: int | None) -> int:
    if model_max_positions is not None and model_max_positions > 0:
        return min(max_tokens, model_max_positions)
    return max_tokens


def _read_json(path: Path) -> object:
    try:
        return cast(object, json.loads(path.read_text()))
    except (OSError, ValueError) as error:
        raise UnsupportedEmbeddingModelError(f"cannot read {path}: {error}") from error
