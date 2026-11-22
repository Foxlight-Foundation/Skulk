"""Placement/source policy bindings over injected resident effects, not inference."""

import io
import wave
from collections.abc import Callable
from pathlib import Path

import pytest

from skulk.master.tests.test_separation_admission import command, state
from skulk.provisioning.demucs_inventory import DEMUCS_PACKAGES
from skulk.provisioning.demucs_runtime import PreparedDemucsRuntime
from skulk.shared.models.separation_runtime import (
    HTDEMUCS_WEIGHTS_SHA256,
    resolve_demucs_policy,
)
from skulk.shared.types.audio_separation import (
    NativeSeparationJob,
    NativeSeparationProgress,
    NativeSeparationReady,
    NativeSeparationResult,
    NativeSeparationServeConfig,
    source_from_wav,
)
from skulk.shared.types.common import CommandId, NodeId
from skulk.shared.types.tasks import AudioSeparation
from skulk.shared.types.worker.instances import BoundInstance, InstanceId
from skulk.shared.types.worker.runners import RunnerId, RunnerReady
from skulk.worker.runner.demucs.model import (
    SeparationModel,
    separation_result_directory,
)


class Resident:
    """Model effects record policy and liveness without claiming a real native load."""

    def __init__(self) -> None:
        self.ready = False
        self.requests: list[NativeSeparationJob] = []
        self.fail_close = False
        self.closes = 0

    def start(self, *, is_cancelled: Callable[[], bool], timeout_seconds: float = 600.0) -> NativeSeparationReady:
        if is_cancelled():
            raise InterruptedError
        self.ready = True
        return NativeSeparationReady(
            payload_sha256=DEMUCS_PACKAGES[0].payload_sha256,
            weights_sha256=HTDEMUCS_WEIGHTS_SHA256,
            device="cpu", threads=2, load_seconds=0.0,
            sample_rate=44100, audio_channels=2, stems=("drums", "bass", "other", "vocals"),
        )

    def is_ready(self) -> bool:
        return self.ready

    def run(self, job: NativeSeparationJob, *, progress: Callable[[NativeSeparationProgress], None], is_cancelled: Callable[[], bool], timeout_seconds: float = 6 * 60 * 60) -> NativeSeparationResult | None:
        self.requests.append(job)
        return None  # The lower-level resident tests own actual output verification.

    def close(self) -> None:
        self.closes += 1
        if self.fail_close:
            self.fail_close = False
            raise RuntimeError("injected drain failure")
        self.ready = False


class Factory:
    """Observe the precise arguments passed to the real supervisor boundary."""

    def __init__(self) -> None:
        self.resident = Resident()
        self.calls: list[tuple[Path, Path, NativeSeparationServeConfig, str, str]] = []

    def __call__(self, interpreter: Path, root: Path, configuration: NativeSeparationServeConfig, *, payload_sha256: str, weights_sha256: str, resource_guard: Callable[[int], str | None] | None = None) -> Resident:
        self.calls.append((interpreter, root, configuration, payload_sha256, weights_sha256))
        return self.resident


def fixture(tmp_path: Path, *, nested: bool = False) -> tuple[SeparationModel, Factory, AudioSeparation, Path]:
    """Use the actual card/policy and real private source file, with resident effects."""
    snapshot = state(RunnerReady())
    instance = snapshot.instances[InstanceId("instance-0")]
    assignments = instance.shard_assignments
    shard = assignments.runner_to_shard[RunnerId("runner-0")]
    if nested:
        card = shard.model_card
        bundle = card.artifact_bundle
        assert bundle is not None
        root = "nested/loader"
        files = tuple(item.model_copy(update={"path": root + "/" + item.path}) for item in bundle.files)
        card = card.model_copy(update={"artifact_bundle": bundle.model_copy(update={"root": root, "files": files})})
        shard = shard.model_copy(update={"model_card": card, "separation_policy": resolve_demucs_policy(card, "demucs-cpu")})
    runtime = PreparedDemucsRuntime(
        wheel=DEMUCS_PACKAGES[0], generation=tmp_path / "runtime",
        interpreter=tmp_path / "runtime/python/bin/python3", device="cpu",
    )
    shard = shard.model_copy(update={"resolved_engine_build": runtime.wheel.build_identity})
    instance = instance.model_copy(update={"shard_assignments": assignments.model_copy(update={"runner_to_shard": {RunnerId("runner-0"): shard}})})
    bound = BoundInstance(instance=instance, bound_runner_id=RunnerId("runner-0"), bound_node_id=NodeId("node-0"))
    model = tmp_path / "model"
    model.mkdir()
    assert shard.separation_policy is not None
    weights = model / shard.separation_policy.weights_path
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.write_bytes(b"fixture only; no actual model")
    factory = Factory()
    bundle = shard.model_card.artifact_bundle
    assert bundle is not None
    loader = model / bundle.root if bundle.root is not None else model
    loaded = SeparationModel(bound, runtime, loader, tmp_path / "resident", tmp_path / "inputs", create_resident=factory)
    data = io.BytesIO()
    with wave.open(data, "wb") as original:
        original.setparams((1, 2, 8000, 8000, "NONE", "not compressed"))
        original.writeframes(bytes(16000))
    request = command()
    task = AudioSeparation(
        command_id=CommandId("job-1"), instance_id=instance.instance_id,
        owner_node=request.owner_node,
        task_params=request.task_params.model_copy(update={"source": source_from_wav(data.getvalue())}),
    )
    source = tmp_path / "inputs/job-1/original.wav"
    source.parent.mkdir(parents=True, mode=0o700)
    source.write_bytes(data.getvalue())
    source.chmod(0o600)
    return loaded, factory, task, source


def test_only_actual_start_publishes_liveness_and_bound_native_policy(tmp_path: Path) -> None:
    model, factory, task, source = fixture(tmp_path)
    assert not model.is_ready() and not factory.calls
    model.start(is_cancelled=lambda: False)
    assert model.is_ready() and len(factory.calls) == 1
    _, root, config, payload, weights = factory.calls[0]
    assert root == model.root and config.threads == 2 and config.device == "cpu"
    assert payload == model.runtime.wheel.payload_sha256 and weights == HTDEMUCS_WEIGHTS_SHA256
    assert model.separate(task, progress=lambda _: None, is_cancelled=lambda: False) is None
    request = factory.resident.requests[0]
    assert request.source_path == str(source) and request.source == task.task_params.source
    assert request.segment_seconds == 7.8 and request.output_dir == str(separation_result_directory(root, task.command_id))
    factory.resident.ready = False
    assert not model.is_ready()


@pytest.mark.parametrize("change", ["source", "instance", "model", "permissions"])
def test_source_or_model_binding_failure_precedes_native_inference(tmp_path: Path, change: str) -> None:
    model, factory, task, source = fixture(tmp_path)
    model.start(is_cancelled=lambda: False)
    if change == "source":
        data = bytearray(source.read_bytes())
        data[-1] = 1
        source.write_bytes(data)
    elif change == "instance":
        task = task.model_copy(update={"instance_id": InstanceId("another-instance")})
    elif change == "model":
        task = task.model_copy(update={"task_params": task.task_params.model_copy(update={"model": "other/model"})})
    else:
        source.chmod(0o644)
    with pytest.raises(ValueError):
        model.separate(task, progress=lambda _: None, is_cancelled=lambda: False)
    assert not factory.resident.requests
    assert model.is_ready()


def test_preflight_cancel_does_not_submit_or_destroy_a_healthy_idle_model(tmp_path: Path) -> None:
    model, factory, task, _ = fixture(tmp_path)
    model.start(is_cancelled=lambda: False)
    assert model.separate(task, progress=lambda _: None, is_cancelled=lambda: True) is None
    assert not factory.resident.requests and model.is_ready()


def test_failed_cleanup_keeps_original_handle_for_retry(tmp_path: Path) -> None:
    model, factory, _, _ = fixture(tmp_path)
    model.start(is_cancelled=lambda: False)
    factory.resident.fail_close = True
    with pytest.raises(RuntimeError, match="drain failure"):
        model.close()
    assert factory.resident.closes == 1
    model.close()
    assert factory.resident.closes == 2 and not model.is_ready()
    model.close()
    assert factory.resident.closes == 2


@pytest.mark.parametrize("identifier", ["../outside", "/absolute", "a/b", "x" * 129])
def test_local_command_mapping_cannot_select_arbitrary_paths(tmp_path: Path, identifier: str) -> None:
    with pytest.raises(ValueError):
        separation_result_directory(tmp_path, CommandId(identifier))



def test_nested_artifact_loader_root_is_not_duplicated(tmp_path: Path) -> None:
    """A mirrored rooted component resolves inside the actual staged generation."""
    model, factory, _, _ = fixture(tmp_path, nested=True)
    model.start(is_cancelled=lambda: False)
    expected = tmp_path / "model" / model.policy.weights_path
    assert factory.calls[0][2].weights_path == str(expected)
    assert expected.is_file()
