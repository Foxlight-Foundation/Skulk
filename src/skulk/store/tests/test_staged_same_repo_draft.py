# pyright: reportPrivateUsage=false
"""Staged-directory completeness for a same-repo served draft GGUF (#422).

A served-engine card may bundle its speculative draft GGUF in the SAME repo as
the base (``served_spec_draft_repo`` == the base repo, e.g. Gemma 4: base +
``mtp-*`` draft). That draft shares the base's store entry and staging dir, so
the generic completeness probe (which only checks the base shard group) treats a
draft-less staged dir as complete. These guards force a re-stage so the draft is
co-fetched, and scope strictly to the same-repo case.
"""

from pathlib import Path

from skulk.shared.models.model_cards import (
    ModelCard,
    ModelId,
    ModelTask,
    RuntimeCapabilityCardConfig,
)
from skulk.shared.types.memory import Memory
from skulk.shared.types.worker.shards import PipelineShardMetadata
from skulk.store.model_store_client import (
    _same_repo_draft_files,
    _staged_same_repo_draft_missing,
)


def _card(
    *,
    model_id: str,
    draft_repo: str | None,
    draft_file: str | None,
) -> ModelCard:
    runtime = None
    if draft_repo is not None or draft_file is not None:
        runtime = RuntimeCapabilityCardConfig(
            served_spec_type="draft_mtp",
            served_spec_draft_repo=draft_repo,
            served_spec_draft_file=draft_file,
        )
    return ModelCard(
        model_id=ModelId(model_id),
        storage_size=Memory.from_gb(1.0),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        gguf_file="base-IQ4_XS.gguf",
        runtime=runtime,
    )


def _shard(card: ModelCard) -> PipelineShardMetadata:
    return PipelineShardMetadata(
        model_card=card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=1,
        n_layers=1,
    )


def _write(directory: Path, *names: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_text("x")
    return directory


def test_same_repo_draft_files_returns_draft_when_repo_matches() -> None:
    card = _card(
        model_id="org/bundle",
        draft_repo="org/bundle",
        draft_file="draft.gguf",
    )
    assert _same_repo_draft_files(card) == ["draft.gguf"]


def test_same_repo_draft_files_empty_for_separate_repo() -> None:
    # A separate-repo draft has its own model_id / staging dir; not co-fetched.
    card = _card(
        model_id="org/base",
        draft_repo="org/separate-draft",
        draft_file="draft.gguf",
    )
    assert _same_repo_draft_files(card) == []


def test_same_repo_draft_files_empty_without_runtime() -> None:
    card = _card(model_id="org/base", draft_repo=None, draft_file=None)
    assert _same_repo_draft_files(card) == []


def test_staged_dir_missing_same_repo_draft_is_flagged(tmp_path: Path) -> None:
    staged = _write(tmp_path, "base-IQ4_XS.gguf", "config.json")
    shard = _shard(
        _card(
            model_id="org/bundle",
            draft_repo="org/bundle",
            draft_file="draft.gguf",
        )
    )
    assert _staged_same_repo_draft_missing(shard, staged) is True


def test_staged_dir_with_same_repo_draft_is_complete(tmp_path: Path) -> None:
    staged = _write(tmp_path, "base-IQ4_XS.gguf", "config.json", "draft.gguf")
    shard = _shard(
        _card(
            model_id="org/bundle",
            draft_repo="org/bundle",
            draft_file="draft.gguf",
        )
    )
    assert _staged_same_repo_draft_missing(shard, staged) is False


def test_separate_repo_draft_never_flags_base_dir(tmp_path: Path) -> None:
    # The base dir of a separate-repo draft card is complete without the draft
    # (the draft lives in its own staging dir), so the fast path must not break.
    staged = _write(tmp_path, "base-IQ4_XS.gguf", "config.json")
    shard = _shard(
        _card(
            model_id="org/base",
            draft_repo="org/separate-draft",
            draft_file="draft.gguf",
        )
    )
    assert _staged_same_repo_draft_missing(shard, staged) is False


def test_non_served_card_never_flags(tmp_path: Path) -> None:
    staged = _write(tmp_path, "base-IQ4_XS.gguf", "config.json")
    shard = _shard(_card(model_id="org/base", draft_repo=None, draft_file=None))
    assert _staged_same_repo_draft_missing(shard, staged) is False
