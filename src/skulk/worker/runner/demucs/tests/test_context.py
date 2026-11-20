"""Normal create/download/load ordering with actual bindings and injected native effects."""

from pathlib import Path

import pytest

from skulk.shared.models.model_cards import ModelCard, ModelId
from skulk.shared.types.worker.instances import BoundInstance
from skulk.utils.runtime_files import RuntimeLock
from skulk.worker.runner.demucs import context as module
from skulk.worker.runner.demucs.context import SeparationRunnerContext
from skulk.worker.runner.demucs.model import SeparationModel
from skulk.worker.runner.demucs.source_fence import SourceReadFence
from skulk.worker.runner.demucs.tests.test_model import fixture


def test_idle_context_does_not_resolve_weights_before_download_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A first mount may create its idle runner before any weight directory exists."""
    actual, factory, _, _ = fixture(tmp_path)
    reader = RuntimeLock(tmp_path / "reader", name="attachment.lock")
    fence = SourceReadFence(reader.descriptor)
    reader.close()
    context = SeparationRunnerContext(
        actual.runtime, actual.root, actual.source_root, fence
    )
    calls: list[ModelId] = []
    staged = [False]

    def resolve(
        model_id: ModelId,
        source_revision: str | None = None,
        artifact_root: str | None = None,
        *,
        expected_card: ModelCard | None = None,
    ) -> Path:
        assert expected_card == actual.bound.bound_shard.model_card
        calls.append(model_id)
        if not staged[0]:
            raise FileNotFoundError("not downloaded yet")
        return tmp_path / "model"

    def bind(
        owner: SeparationRunnerContext, bound: BoundInstance, directory: Path
    ) -> SeparationModel:
        assert (
            owner is context
            and bound == actual.bound
            and directory == tmp_path / "model"
        )
        return actual

    monkeypatch.setattr(module, "build_model_path", resolve)
    monkeypatch.setattr(SeparationRunnerContext, "bind", bind)
    model = context.model(actual.bound)
    try:
        assert not calls and not factory.calls and not model.is_ready()
        with pytest.raises(InterruptedError):
            model.start(is_cancelled=lambda: True)
        assert not calls and not factory.calls
        with pytest.raises(FileNotFoundError):
            model.start(is_cancelled=lambda: False)
        assert not factory.calls and not model.is_ready()
        staged[0] = True
        model.start(is_cancelled=lambda: False)
        assert model.is_ready() and len(factory.calls) == 1
        factory.resident.fail_close = True
        with pytest.raises(RuntimeError, match="drain failure"):
            model.close()
        model.close()
        assert not model.is_ready()
        model.start(is_cancelled=lambda: False)
        assert model.is_ready() and len(factory.calls) == 2
    finally:
        model.close()
        fence.close()


def test_closed_reader_context_cannot_construct_an_idle_model(tmp_path: Path) -> None:
    """A stale local context cannot reactivate a source lifetime its owner released."""
    actual, _, _, _ = fixture(tmp_path)
    reader = RuntimeLock(tmp_path / "reader", name="attachment.lock")
    fence = SourceReadFence(reader.descriptor)
    reader.close()
    fence.close()
    context = SeparationRunnerContext(
        actual.runtime, actual.root, actual.source_root, fence
    )
    with pytest.raises(RuntimeError, match="fence is closed"):
        context.model(actual.bound)
