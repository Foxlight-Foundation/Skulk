"""Real native bindings behind first-mount ordering and private source custody."""

import threading
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path

import pytest

from skulk.shared.models.model_cards import ModelCard
from skulk.shared.types.worker.instances import BoundInstance
from skulk.utils.runtime_files import RuntimeLock
from skulk.worker.runner.ace_step import context as module
from skulk.worker.runner.ace_step.context import NativeMusicRunnerContext
from skulk.worker.runner.ace_step.model import NativeMusicModel
from skulk.worker.runner.ace_step.tests.test_model import fixture
from skulk.worker.runner.ace_step.tests.test_resident import create_resident_interpreter
from skulk.worker.runner.demucs.source_fence import SourceReadFence


def resource_guard(_process_id: int) -> str | None:
    """Supply a controlled admission effect; this fixture performs no native load."""
    return None


def test_first_mount_resolves_complete_sources_only_at_load_and_retires_replacements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Actual file verification starts after download; failed close remains owned."""
    actual, factory, task = fixture(tmp_path)
    reader = RuntimeLock(tmp_path / "reader", name="attachment.lock")
    fence = SourceReadFence(reader.descriptor)
    reader.close()
    context = NativeMusicRunnerContext(
        actual.runtime, actual.root, actual.source_root, fence, resource_guard
    )
    calls: list[tuple[str, str, bool]] = []
    staged = [False]
    bindings: list[NativeMusicModel] = []

    def resolve(
        card: ModelCard,
        repository: str,
        revision: str,
        *,
        require_card_identity: bool = False,
    ) -> Path | None:
        assert card == actual.bound.bound_shard.model_card
        calls.append((repository, revision, require_card_identity))
        return actual.sources[(repository, revision)] if staged[0] else None

    def bind(
        owner: NativeMusicRunnerContext,
        bound: BoundInstance,
        sources: Mapping[tuple[str, str], Path],
    ) -> NativeMusicModel:
        assert owner is context and bound == actual.bound and sources == actual.sources
        model = NativeMusicModel(
            bound,
            owner.runtime,
            sources,
            owner.root,
            owner.source_root,
            create_resident=factory,
            resource_guard=owner.resource_guard,
        )
        bindings.append(model)
        return model

    monkeypatch.setattr(module, "resolve_native_music_source_directory", resolve)
    monkeypatch.setattr(NativeMusicRunnerContext, "bind", bind)
    model = context.model(actual.bound)
    try:
        assert not calls and not factory.calls and not model.is_ready()
        with pytest.raises(InterruptedError):
            model.start(is_cancelled=lambda: True)
        assert not calls and not bindings
        with pytest.raises(FileNotFoundError, match="not installed"):
            model.start(is_cancelled=lambda: False)
        assert not bindings and not factory.calls and not model.is_ready()
        calls.clear()
        staged[0] = True
        model.start(is_cancelled=lambda: False)
        card = actual.bound.bound_shard.model_card
        assert calls == [
            (
                source.repository,
                source.revision,
                (source.repository, source.revision)
                == (str(card.artifact_repository), card.source_revision),
            )
            for source in card.native_music_artifact_sources()
        ]
        assert model.is_ready() and len(bindings) == factory.calls == 1
        resident = factory.resident
        assert resident is not None
        resident.fail_close = True
        with pytest.raises(RuntimeError, match="drain failure"):
            model.close()
        assert not model.is_ready()
        with pytest.raises(RuntimeError, match="not loaded"):
            model.run(task, progress=lambda _update: None, is_cancelled=lambda: False)
        model.close()
        with pytest.raises(RuntimeError, match="new unloaded lifetime"):
            model.start(is_cancelled=lambda: False)
        replacement = context.model(actual.bound)
        try:
            replacement.start(is_cancelled=lambda: False)
            assert replacement.is_ready() and len(bindings) == factory.calls == 2
            assert bindings[0] is not bindings[1]
        finally:
            replacement.close()
    finally:
        model.close()
        fence.close()


def test_closed_context_cannot_create_or_start_a_source_reader(tmp_path: Path) -> None:
    """Released reader custody cannot become an idle or loaded native lifetime."""
    actual, factory, _task = fixture(tmp_path)
    owner = RuntimeLock(tmp_path / "reader", name="attachment.lock")
    fence = SourceReadFence(owner.descriptor)
    owner.close()
    context = NativeMusicRunnerContext(
        actual.runtime, actual.root, actual.source_root, fence, resource_guard
    )
    model = context.model(actual.bound)
    fence.close()
    with pytest.raises(RuntimeError, match="fence is closed"):
        context.model(actual.bound)
    with pytest.raises(RuntimeError, match="fence is closed"):
        model.start(is_cancelled=lambda: False)
    assert not factory.calls and not model.is_ready()
    model.close()


def test_real_context_spawn_retains_reader_and_calls_the_live_resource_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Actual synthetic native processes keep source custody until guardian drain."""
    interpreter = create_resident_interpreter(tmp_path / "interpreter")
    actual, _factory, _task = fixture(tmp_path / "model", interpreter_path=interpreter)
    root = tmp_path / "reader"
    owner = RuntimeLock(root, name="attachment.lock")
    fence = SourceReadFence(owner.descriptor)
    owner.close()
    observed: list[int] = []

    def guard(process_id: int) -> str | None:
        observed.append(process_id)
        return None

    context = NativeMusicRunnerContext(
        actual.runtime, actual.root, actual.source_root, fence, guard
    )

    def resolve(
        card: ModelCard,
        repository: str,
        revision: str,
        *,
        require_card_identity: bool = False,
    ) -> Path | None:
        assert card == actual.bound.bound_shard.model_card
        return actual.sources[(repository, revision)]

    monkeypatch.setattr(module, "resolve_native_music_source_directory", resolve)
    model = context.model(actual.bound)
    try:
        assert not model.is_ready() and not observed
        model.start(is_cancelled=lambda: False)
        assert model.is_ready() and observed and all(pid > 0 for pid in observed)
        # Close only Core's copy. Native descendants must still protect their
        # command-original source files, even after an abrupt Core exit.
        fence.close()
        with pytest.raises(BlockingIOError):
            RuntimeLock(root, name="attachment.lock")
        assert not model.is_ready()
        model.close()
        reclaimed = RuntimeLock(root, name="attachment.lock")
        reclaimed.close()
    finally:
        model.close()
        fence.close()


@pytest.mark.parametrize("close_during_resolution", [False, True])
def test_source_resolution_cannot_lose_a_concurrent_load_or_outlive_close(
    tmp_path: Path,
    close_during_resolution: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Serialize binding creation and join a cancelled lookup before returning close."""
    actual, factory, _task = fixture(tmp_path)
    owner = RuntimeLock(tmp_path / "reader", name="attachment.lock")
    fence = SourceReadFence(owner.descriptor)
    owner.close()
    context = NativeMusicRunnerContext(
        actual.runtime, actual.root, actual.source_root, fence, resource_guard
    )
    entered, release = threading.Event(), threading.Event()
    bindings: list[NativeMusicModel] = []

    def resolve(
        card: ModelCard,
        repository: str,
        revision: str,
        *,
        require_card_identity: bool = False,
    ) -> Path | None:
        assert card == actual.bound.bound_shard.model_card
        entered.set()
        assert release.wait(10)
        return actual.sources[(repository, revision)]

    def bind(
        owner: NativeMusicRunnerContext,
        bound: BoundInstance,
        sources: Mapping[tuple[str, str], Path],
    ) -> NativeMusicModel:
        assert owner is context and bound == actual.bound and sources == actual.sources
        bindings.append(actual)
        return actual

    monkeypatch.setattr(module, "resolve_native_music_source_directory", resolve)
    monkeypatch.setattr(NativeMusicRunnerContext, "bind", bind)
    model = context.model(actual.bound)
    with ThreadPoolExecutor(max_workers=2) as executor:
        loading = executor.submit(model.start, is_cancelled=lambda: False)
        try:
            assert entered.wait(10)
            with pytest.raises(RuntimeError, match="active model load"):
                model.start(is_cancelled=lambda: False)
            assert not bindings and not factory.calls
            if close_during_resolution:
                closing = executor.submit(model.close)
                with pytest.raises(FutureTimeout):
                    closing.result(timeout=0.1)
                release.set()
                with pytest.raises(RuntimeError, match="new unloaded lifetime"):
                    loading.result(timeout=10)
                closing.result(timeout=10)
                assert not bindings and not factory.calls and not model.is_ready()
            else:
                release.set()
                loading.result(timeout=10)
                assert bindings == [actual] and factory.calls == 1 and model.is_ready()
        finally:
            release.set()
            model.close()
            fence.close()


@pytest.mark.parametrize("timeout", [0.0, -1.0, float("inf"), float("nan"), 7201.0])
def test_invalid_deadline_does_not_resolve_selected_files(
    tmp_path: Path, timeout: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Malformed load budgets are rejected before any installation lookup."""
    actual, factory, _task = fixture(tmp_path)
    owner = RuntimeLock(tmp_path / "reader", name="attachment.lock")
    fence = SourceReadFence(owner.descriptor)
    owner.close()
    context = NativeMusicRunnerContext(
        actual.runtime, actual.root, actual.source_root, fence, resource_guard
    )

    def resolve(
        card: ModelCard,
        repository: str,
        revision: str,
        *,
        require_card_identity: bool = False,
    ) -> Path | None:
        pytest.fail("an invalid load budget cannot resolve staged weights")

    monkeypatch.setattr(module, "resolve_native_music_source_directory", resolve)
    model = context.model(actual.bound)
    try:
        with pytest.raises(ValueError, match="bounded deadline"):
            model.start(is_cancelled=lambda: False, timeout_seconds=timeout)
        assert not factory.calls
    finally:
        model.close()
        fence.close()
