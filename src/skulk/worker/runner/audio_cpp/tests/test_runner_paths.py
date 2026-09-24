# pyright: reportPrivateUsage=false
"""The server must receive the card's loader root for nested bundles."""

import threading
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from skulk.download import download_utils
from skulk.shared.constants import RESOURCES_DIR
from skulk.shared.models.model_cards import ModelCard, ModelId
from skulk.shared.types.common import CommandId, NodeId
from skulk.shared.types.music import MusicGenerationTaskParams
from skulk.shared.types.tasks import MusicGeneration
from skulk.shared.types.worker.instances import InstanceId
from skulk.shared.types.worker.runners import RunnerRunning
from skulk.worker.runner.audio_cpp.runner import Runner, model_directory
from skulk.worker.runner.audio_cpp.server import AudioCppServer


def test_ace_step_server_uses_nested_loader_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A repository root would make ACE-Step select the wrong model variant."""
    card_path = (
        Path(RESOURCES_DIR)
        / "music_model_cards"
        / "audio-cpp--ACE-Step1.5-Turbo-BF16.toml"
    )
    card = ModelCard.model_validate(tomllib.loads(card_path.read_text()))
    loader_root = tmp_path / "ACE-Step1.5-GGUF" / "turbo"
    loader_root.mkdir(parents=True)
    artifact = loader_root / "ace-step-1.5-turbo-bf16.gguf"
    artifact.touch()

    def local_model_path(model_id: object, revision: object, root: object) -> Path:
        assert model_id == card.model_id
        assert revision == card.source_revision
        assert root == "ACE-Step1.5-GGUF/turbo"
        return loader_root

    monkeypatch.setattr(download_utils, "build_model_path", local_model_path)
    assert model_directory(card) == loader_root
    assert card.artifact_bundle is not None
    assert download_utils.resolve_artifact_file(
        loader_root,
        card.artifact_bundle.root,
        card.artifact_bundle.files[0].path,
    ) == artifact


def test_cancelled_generation_restores_server_before_next_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A queued job can run after cancellation kills the active sidecar."""
    first_alive = True
    first = cast(
        AudioCppServer,
        cast(object, SimpleNamespace(alive=lambda: first_alive, teardown=lambda: None)),
    )
    second = cast(
        AudioCppServer,
        cast(object, SimpleNamespace(alive=lambda: True, teardown=lambda: None)),
    )
    runner = object.__new__(Runner)
    runner.server = first
    runner._status_lock = threading.Lock()
    runner._inflight = 1
    runner.current_status = RunnerRunning()
    runner.model_id = ModelId("audio-cpp/test-music")
    task = MusicGeneration(
        instance_id=InstanceId(), command_id=CommandId(), owner_node=NodeId("api"),
        task_params=MusicGenerationTaskParams(
            model=str(runner.model_id), prompt="piano", seconds=20,
        ),
    )
    cancellation_checks = 0

    def cancelled(_runner: Runner, _id: object) -> bool:
        nonlocal cancellation_checks
        cancellation_checks += 1
        return cancellation_checks > 1

    monkeypatch.setattr(Runner, "_is_cancelled", cancelled)

    def render(_runner: Runner, *_args: object) -> None:
        nonlocal first_alive
        first_alive = False
        # The dispatch loop can poll here while the active request unwinds.
        runner._ensure_server_alive()

    def load_model(_runner: Runner) -> None:
        runner.server = second

    monkeypatch.setattr(Runner, "_render", render)
    monkeypatch.setattr(Runner, "_load_model", load_model)
    runner._generate(task)
    runner._inflight = 0
    runner._ensure_server_alive()
    assert runner.server is second
