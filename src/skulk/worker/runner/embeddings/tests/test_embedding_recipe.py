"""An embedding model's recipe comes from its own sentence-transformers metadata.

Invariant: the runner reproduces each model's reference vectors. bge-small
pools the [CLS] token and MiniLM truncates at 256 tokens, which one hard-coded
recipe (mean, normalize, 512) got wrong; a pipeline step the runner cannot
reproduce is refused at load instead of silently approximated.
"""

import json
from pathlib import Path

import pytest

from skulk.worker.runner.embeddings.recipe import (
    EmbeddingRecipe,
    UnsupportedEmbeddingModelError,
    resolve_embedding_recipe,
)

_TRANSFORMER = {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"}
_POOLING = {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"}
_NORMALIZE = {"idx": 2, "name": "2", "path": "2_Normalize", "type": "sentence_transformers.models.Normalize"}


def _write(model_dir: Path, modules: object, pooling: dict[str, bool] | None, max_seq: int | None) -> None:
    (model_dir / "modules.json").write_text(json.dumps(modules))
    if pooling is not None:
        (model_dir / "1_Pooling").mkdir()
        (model_dir / "1_Pooling" / "config.json").write_text(
            json.dumps({"word_embedding_dimension": 384, **pooling})
        )
    if max_seq is not None:
        (model_dir / "sentence_bert_config.json").write_text(
            json.dumps({"max_seq_length": max_seq, "do_lower_case": False})
        )


def _flags(enabled: str) -> dict[str, bool]:
    names = (
        "pooling_mode_cls_token",
        "pooling_mode_mean_tokens",
        "pooling_mode_max_tokens",
        "pooling_mode_mean_sqrt_len_tokens",
    )
    return {name: name == enabled for name in names}


def test_minilm_metadata_gives_mean_normalize_256(tmp_path: Path) -> None:
    _write(tmp_path, [_TRANSFORMER, _POOLING, _NORMALIZE], _flags("pooling_mode_mean_tokens"), 256)
    assert resolve_embedding_recipe(tmp_path, model_max_positions=512) == EmbeddingRecipe(
        pooling="mean", normalize=True, max_tokens=256
    )


def test_bge_metadata_gives_cls_pooling(tmp_path: Path) -> None:
    _write(tmp_path, [_TRANSFORMER, _POOLING, _NORMALIZE], _flags("pooling_mode_cls_token"), 512)
    assert resolve_embedding_recipe(tmp_path, model_max_positions=512) == EmbeddingRecipe(
        pooling="cls", normalize=True, max_tokens=512
    )


def test_last_token_pooling_and_no_normalize(tmp_path: Path) -> None:
    _write(tmp_path, [_TRANSFORMER, _POOLING], {"pooling_mode_lasttoken": True}, 8192)
    recipe = resolve_embedding_recipe(tmp_path, model_max_positions=None)
    assert recipe == EmbeddingRecipe(pooling="last", normalize=False, max_tokens=8192)


def test_truncation_never_exceeds_the_encoder_positions(tmp_path: Path) -> None:
    _write(tmp_path, [_TRANSFORMER, _POOLING, _NORMALIZE], _flags("pooling_mode_max_tokens"), 1024)
    assert resolve_embedding_recipe(tmp_path, model_max_positions=512).max_tokens == 512


def test_artifact_without_metadata_keeps_the_legacy_recipe(tmp_path: Path) -> None:
    assert resolve_embedding_recipe(tmp_path, model_max_positions=None) == EmbeddingRecipe(
        pooling="mean", normalize=True, max_tokens=512
    )
    assert resolve_embedding_recipe(tmp_path, model_max_positions=128).max_tokens == 128


def test_missing_max_seq_length_falls_back_to_512(tmp_path: Path) -> None:
    _write(tmp_path, [_TRANSFORMER, _POOLING], _flags("pooling_mode_mean_tokens"), None)
    assert resolve_embedding_recipe(tmp_path, model_max_positions=None).max_tokens == 512


@pytest.mark.parametrize(
    ("modules", "pooling", "message"),
    [
        (
            [_TRANSFORMER, _POOLING, {"idx": 2, "path": "2_Dense", "type": "sentence_transformers.models.Dense"}],
            _flags("pooling_mode_mean_tokens"),
            "not implemented",
        ),
        ([_TRANSFORMER, _POOLING], {"pooling_mode_cls_token": True, "pooling_mode_mean_tokens": True}, "not implemented"),
        ([_TRANSFORMER, _POOLING], _flags("pooling_mode_mean_sqrt_len_tokens"), "not implemented"),
        ([_TRANSFORMER, _POOLING], {}, "not implemented"),
        ([_TRANSFORMER, _NORMALIZE], None, "no pooling module"),
        ({"not": "a list"}, None, "not a module list"),
    ],
)
def test_unreproducible_pipelines_are_refused(
    tmp_path: Path, modules: object, pooling: dict[str, bool] | None, message: str
) -> None:
    _write(tmp_path, modules, pooling, 256)
    with pytest.raises(UnsupportedEmbeddingModelError, match=message):
        resolve_embedding_recipe(tmp_path, model_max_positions=512)


def test_missing_pooling_config_is_refused(tmp_path: Path) -> None:
    _write(tmp_path, [_TRANSFORMER, _POOLING], None, 256)
    with pytest.raises(UnsupportedEmbeddingModelError, match="missing pooling config"):
        resolve_embedding_recipe(tmp_path, model_max_positions=512)
