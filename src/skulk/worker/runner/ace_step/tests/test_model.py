"""Exact model/execution binding and closed task translation, not music qualification."""

import io
import threading
import wave
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path

import pytest

from skulk.provisioning.ace_step_runtime import AceStepWheel, PreparedAceStepRuntime
from skulk.shared.models.model_cards import ModelCard
from skulk.shared.models.native_music_runtime import resolve_native_music_policy
from skulk.shared.models.tests.test_native_music_runtime import probe
from skulk.shared.types.audio_separation import source_from_wav
from skulk.shared.types.common import CommandId, NodeId
from skulk.shared.types.music import MusicGenerationTaskParams
from skulk.shared.types.music_edits import (
    MusicAddPartRequest,
    MusicContinueRequest,
    MusicRepaintRequest,
    resolve_music_edit,
)
from skulk.shared.types.native_music import (
    NativeMusicProgress,
    NativeMusicReady,
    NativeMusicRequest,
    NativeMusicResult,
    NativeMusicServeConfig,
    NativeMusicWav,
)
from skulk.shared.types.tasks import MusicEditing, MusicGeneration
from skulk.shared.types.worker.instances import (
    BoundInstance,
    InstanceId,
    MlxRingInstance,
)
from skulk.shared.types.worker.runners import RunnerId, ShardAssignments
from skulk.shared.types.worker.shards import PipelineShardMetadata
from skulk.utils.runtime_files import write_private
from skulk.worker.runner.ace_step import artifacts
from skulk.worker.runner.ace_step import model as model_module
from skulk.worker.runner.ace_step import result_custody as result_module
from skulk.worker.runner.ace_step.artifacts import (
    ObservedMusicArtifacts,
    observe_native_music_artifacts,
)
from skulk.worker.runner.ace_step.model import NativeMusicModel
from skulk.worker.runner.ace_step.resident import ResidentMusic
from skulk.worker.runner.ace_step.result_custody import (
    PendingMusicResult,
    music_result_directory,
    verify_prepared_result_directory,
)
from skulk.worker.runner.ace_step.tests.test_artifacts import selected_fixture
from skulk.worker.runner.ace_step.tests.test_resident import create_resident_interpreter
from skulk.worker.runner.ace_step.workspace import NativeMusicWorkspace


class Resident:
    """Controlled native effects; real resident tests own full output verification."""

    def __init__(
        self,
        workspace: NativeMusicWorkspace,
        configuration: NativeMusicServeConfig,
        payload: str,
        card: ModelCard,
    ) -> None:
        self.workspace, self.configuration, self.payload = (
            workspace,
            configuration,
            payload,
        )
        self.card = card
        self.closed = False
        self.live = False
        self.requests: list[NativeMusicRequest] = []
        self.fail_close = False
        self.closes = 0
        self.foreign_ready = False
        self.cancel_run = False
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()

    def start(
        self, *, is_cancelled: Callable[[], bool], timeout_seconds: float = 600.0
    ) -> NativeMusicReady:
        """Return a controlled loaded receipt after the adapter has verified real files."""
        assert timeout_seconds > 0
        if is_cancelled():
            raise InterruptedError
        self.live = True
        policy = resolve_native_music_policy(
            self.card,
            "ace_step-cpu",
            probe().model_copy(update={"payload_sha256": self.payload}),
        )
        return NativeMusicReady(
            model_identity=self.configuration.model_identity,
            payload_sha256="f" * 64 if self.foreign_ready else self.payload,
            execution=policy.execution,
            load_seconds=0.01,
        )

    def is_ready(self) -> bool:
        """Expose current controlled liveness, never remembered load success."""
        return self.live

    def run(
        self,
        request: NativeMusicRequest,
        *,
        progress: Callable[[NativeMusicProgress], None],
        is_cancelled: Callable[[], bool],
        timeout_seconds: float = 7200.0,
    ) -> NativeMusicResult | None:
        """Record closed requests with actual pre-write custody, then emit fixture output."""
        assert timeout_seconds > 0
        directory = Path(request.output_directory)
        verify_prepared_result_directory(
            directory,
            request.output_directory_identity,
            self.workspace.requests_identity,
        )
        self.requests.append(request)
        self.entered.set()
        assert self.release.wait(3)
        if self.cancel_run or is_cancelled():
            self.live = False
            return None
        output = io.BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setparams(
                (2, 3, 48000, request.target_frames, "NONE", "not compressed")
            )
            wav.writeframes(bytes(request.target_frames * 6))
        data = output.getvalue()
        measured = source_from_wav(data)
        write_private(directory / "result.wav", data)
        result = NativeMusicResult(
            request_id=request.request_id,
            model_identity=self.configuration.model_identity,
            payload_sha256=self.payload,
            operation=request.operation,
            seed=request.seed,
            source_sha256=request.source.sha256 if request.source is not None else None,
            conditioning_sha256="a" * 64 if request.source is not None else None,
            conditioning_gain=1.0 if request.source is not None else None,
            conditioning_padding_frames=(-request.target_frames % 1920)
            if request.source is not None
            else 0,
            execution=self.start(is_cancelled=lambda: False).execution,
            output=NativeMusicWav(
                sha256=measured.sha256,
                size_bytes=len(data),
                frames=request.target_frames,
                peak=0.0,
                gain=1.0,
            ),
            generation_seconds=0.01,
        )
        write_private(directory / "result.json", result.model_dump_json().encode())
        return result

    def close(self) -> None:
        """Keep actual workspace custody on injected drain failure; retry later."""
        self.closes += 1
        if self.closed:
            return
        if self.fail_close:
            self.fail_close = False
            raise RuntimeError("controlled drain failure")
        self.live = False
        assert self.workspace.retire_loading()
        self.workspace.close()
        self.closed = True


class Factory:
    """Retain a controlled resident without native imports, model weights or spawn."""

    def __init__(self, bound: BoundInstance) -> None:
        self.bound = bound
        self.resident: Resident | None = None
        self.calls = 0
        self.foreign_ready = False

    def __call__(
        self,
        interpreter: Path,
        workspace: NativeMusicWorkspace,
        configuration: NativeMusicServeConfig,
        *,
        payload_sha256: str,
        resource_guard: Callable[[int], str | None] | None = None,
    ) -> Resident:
        """Take the already-open real workspace and exact observed file inventory."""
        self.calls += 1
        resident = Resident(
            workspace, configuration, payload_sha256, self.bound.bound_shard.model_card
        )
        resident.foreign_ready = self.foreign_ready
        self.resident = resident
        return resident


def fixture(
    tmp_path: Path, *, interpreter_path: Path | None = None
) -> tuple[NativeMusicModel, Factory, MusicGeneration]:
    """Use real pinned tiny files and actual policy, injecting only resident effects."""
    card, sources = selected_fixture(tmp_path / "data")
    observed = observe_native_music_artifacts(card, sources)
    library = probe().model_copy(update={"payload_sha256": "c" * 64})
    runtime = PreparedAceStepRuntime(
        wheel=AceStepWheel(
            variant="cpu",
            platform_tag="macosx_14_0_arm64",
            filename="skulk_ace_step_cpu-0.1.0-cp312-cp312-macosx_14_0_arm64.whl",
            sha256="d" * 64,
            payload_sha256="c" * 64,
            size_bytes=1,
        ),
        generation=tmp_path / "runtime"
        if interpreter_path is None
        else interpreter_path.parent.parent,
        interpreter=tmp_path / "runtime/bin/python"
        if interpreter_path is None
        else interpreter_path,
        probe=library,
    )
    policy = resolve_native_music_policy(
        card, "ace_step-cpu", library, observed_layouts=observed.tensor_layouts
    )
    node, runner = NodeId("model-test"), RunnerId("model-test")
    instance = MlxRingInstance(
        instance_id=InstanceId("model-test"),
        shard_assignments=ShardAssignments(
            model_id=card.model_id,
            node_to_runner={node: runner},
            runner_to_shard={
                runner: PipelineShardMetadata(
                    model_card=card,
                    device_rank=0,
                    world_size=1,
                    start_layer=0,
                    end_layer=24,
                    n_layers=24,
                    resolved_backend="ace_step-cpu",
                    resolved_engine_build=runtime.wheel.build_identity,
                    native_music_policy=policy,
                )
            },
        ),
        hosts_by_node={},
        ephemeral_port=50000,
    )
    bound = BoundInstance(instance=instance, bound_runner_id=runner, bound_node_id=node)
    factory = Factory(bound)
    model = NativeMusicModel(
        bound,
        runtime,
        sources,
        tmp_path / "owner",
        tmp_path / "originals",
        create_resident=factory,
    )
    task = MusicGeneration(
        instance_id=instance.instance_id,
        command_id=CommandId("generation"),
        owner_node=node,
        task_params=MusicGenerationTaskParams(
            model=str(card.model_id),
            prompt="  original caption\nkeep this  ",
            seconds=5,
            seed=12025,
        ),
    )
    return model, factory, task


def edit_task(model: NativeMusicModel, mode: str) -> MusicEditing:
    """Create a real private original, using Core's actual public request resolver."""
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setparams((1, 2, 8000, 8000, "NONE", "not compressed"))
        wav.writeframes(bytes(16000))
    source = source_from_wav(output.getvalue())
    command = CommandId(mode)
    directory = model.source_root / str(command)
    directory.mkdir(parents=True, mode=0o700)
    write_private(directory / "original.wav", output.getvalue())
    card = model.bound.bound_shard.model_card
    if mode == "add_part":
        request = MusicAddPartRequest(
            task="add_part",
            part_class="strings",
            region_start_frame=1,
            region_end_frame=7999,
            model=str(card.model_id),
            prompt="original musical caption",
            source_sample_rate=8000,
            seed=12025,
        )
    elif mode == "repaint":
        request = MusicRepaintRequest(
            task="repaint",
            region_start_frame=1,
            region_end_frame=7999,
            model=str(card.model_id),
            prompt="original musical caption",
            source_sample_rate=8000,
            seed=12025,
        )
    else:
        request = MusicContinueRequest(
            task="continue",
            target_frames=16001,
            model=str(card.model_id),
            prompt="original musical caption",
            source_sample_rate=8000,
            seed=12025,
        )
    return MusicEditing(
        instance_id=model.bound.instance.instance_id,
        command_id=command,
        owner_node=model.bound.bound_node_id,
        task_params=resolve_music_edit(
            request, source, card, resolved_seed=12025, total_input_chunks=1
        ),
    )


def test_load_verifies_actual_data_and_binds_complete_policy_before_readiness(
    tmp_path: Path,
) -> None:
    """Construction and package preparation cannot stand in for actual model load."""
    model, factory, _ = fixture(tmp_path)
    assert factory.calls == 0 and not model.root.exists() and not model.is_ready()
    try:
        ready = model.start(is_cancelled=lambda: False)
        assert factory.calls == 1 and model.is_ready()
        assert ready.model_identity == model.policy.model_identity
        assert ready.execution == model.policy.execution
        assert factory.resident is not None
        assert len(factory.resident.configuration.weights.text_encoder.files) == 9
        model.start(is_cancelled=lambda: False)
        assert factory.calls == 1
        factory.resident.live = False
        assert not model.is_ready()
    finally:
        model.close()


@pytest.mark.parametrize(
    "fault",
    ["rank", "world", "range", "layers", "backend", "build", "policy", "payload"],
)
def test_incomplete_or_changed_placement_fails_before_file_or_native_effects(
    tmp_path: Path, fault: str
) -> None:
    """A valid component package cannot override a different admitted shard/policy."""
    model, factory, _ = fixture(tmp_path)
    shard = model.bound.bound_shard
    updates = {
        "rank": {"device_rank": 1},
        "world": {"world_size": 2},
        "range": {"end_layer": 23},
        "layers": {"n_layers": 25},
        "backend": {"resolved_backend": "ace_step-metal"},
        "build": {"resolved_engine_build": None},
        "policy": {"native_music_policy": None},
        "payload": {
            "native_music_policy": model.policy.model_copy(
                update={"payload_sha256": "b" * 64}
            )
        },
    }
    assignments = model.bound.instance.shard_assignments.model_copy(
        update={
            "runner_to_shard": {
                model.bound.bound_runner_id: shard.model_copy(update=updates[fault])
            }
        }
    )
    bound = model.bound.model_copy(
        update={
            "instance": model.bound.instance.model_copy(
                update={"shard_assignments": assignments}
            )
        }
    )
    with pytest.raises(ValueError):
        NativeMusicModel(
            bound,
            model.runtime,
            model.sources,
            model.root,
            model.source_root,
            create_resident=factory,
        )
    assert factory.calls == 0 and not model.root.exists()


def test_changed_complete_file_blocks_start_before_native_or_workspace(
    tmp_path: Path,
) -> None:
    """The observed policy is not trusted after the installed bytes have changed."""
    model, factory, _ = fixture(tmp_path)
    original = next(iter(model.sources.values())) / "model.safetensors"
    data = bytearray(original.read_bytes())
    data[-1] ^= 1
    original.write_bytes(data)
    with pytest.raises(ValueError, match="object identity"):
        model.start(is_cancelled=lambda: False)
    assert factory.calls == 0 and not model.root.exists() and not model.is_ready()


def test_foreign_loaded_receipt_is_retired_and_cannot_be_ready(tmp_path: Path) -> None:
    """Actual model identity and execution must still match after initialization."""
    model, factory, _ = fixture(tmp_path)
    factory.foreign_ready = True
    with pytest.raises(ValueError, match="loaded readiness"):
        model.start(is_cancelled=lambda: False)
    assert not model.is_ready() and factory.resident is not None
    assert factory.resident.closes == 1
    model.close()


def test_admitted_header_facts_are_verified_against_actual_pinned_bytes(
    tmp_path: Path,
) -> None:
    """A plausible restored projection is not an authority over the selected file."""
    model, factory, _ = fixture(tmp_path)
    policy = model.policy
    assert policy.tensor_layouts is not None
    base = policy.tensor_layouts.base.model_copy(update={"header_sha256": "b" * 64})
    policy = policy.model_copy(
        update={
            "tensor_layouts": policy.tensor_layouts.model_copy(update={"base": base})
        }
    )
    shard = model.bound.bound_shard.model_copy(update={"native_music_policy": policy})
    assignments = model.bound.instance.shard_assignments.model_copy(
        update={"runner_to_shard": {model.bound.bound_runner_id: shard}}
    )
    bound = model.bound.model_copy(
        update={
            "instance": model.bound.instance.model_copy(
                update={"shard_assignments": assignments}
            )
        }
    )
    changed = NativeMusicModel(
        bound,
        model.runtime,
        model.sources,
        model.root,
        model.source_root,
        create_resident=factory,
    )
    with pytest.raises(ValueError, match="observed native music files"):
        changed.start(is_cancelled=lambda: False)
    assert factory.calls == 0 and not model.root.exists()


def test_generate_preserves_intent_seed_and_retained_output_across_close(
    tmp_path: Path,
) -> None:
    """Only typed intent is translated; model loader cleanup cannot remove a take."""
    model, factory, task = fixture(tmp_path)
    model.start(is_cancelled=lambda: False)
    result = model.run(task, progress=lambda _: None, is_cancelled=lambda: False)
    assert result is not None and factory.resident is not None
    request = factory.resident.requests[0]
    assert request.prompt == task.task_params.prompt
    assert request.seed == 12025 and request.target_frames == 240000
    assert request.lyrics == "[Instrumental]" and request.source is None
    data = (result.directory / "result.wav").read_bytes()
    model.close()
    assert (result.directory / "result.wav").read_bytes() == data


def test_duplicate_command_cannot_replace_a_retained_completed_take(
    tmp_path: Path,
) -> None:
    """Output receipt conflicts preserve the original and the healthy native model."""
    model, factory, task = fixture(tmp_path)
    try:
        model.start(is_cancelled=lambda: False)
        first = model.run(task, progress=lambda _: None, is_cancelled=lambda: False)
        assert first is not None and factory.resident is not None
        before = (first.directory / "result.wav").read_bytes()
        with pytest.raises(ValueError, match="unresolved ownership receipt"):
            model.run(task, progress=lambda _: None, is_cancelled=lambda: False)
        assert len(factory.resident.requests) == 1 and model.is_ready()
        assert (first.directory / "result.wav").read_bytes() == before
    finally:
        model.close()


def test_unsubmitted_cancel_cleanup_failure_retains_ownership_for_close_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deletion refusal before native send keeps the real result receipt for retry."""
    model, factory, task = fixture(tmp_path)
    model.start(is_cancelled=lambda: False)
    calls = 0
    cleanups = 0
    reclaim = model_module.reclaim_unsubmitted_result

    def cancelled() -> bool:
        nonlocal calls
        calls += 1
        return calls >= 3

    def fail_once(pending: PendingMusicResult) -> bool:
        nonlocal cleanups
        cleanups += 1
        return False if cleanups == 1 else reclaim(pending)

    monkeypatch.setattr(model_module, "reclaim_unsubmitted_result", fail_once)
    with pytest.raises(
        RuntimeError, match="unsubmitted result cleanup remains pending"
    ):
        model.run(task, progress=lambda _: None, is_cancelled=cancelled)
    directory = music_result_directory(model.root, task.command_id)
    assert directory.exists() and not model.is_ready()
    assert factory.resident is not None and not factory.resident.requests
    model.close()
    assert not directory.exists() and cleanups == 2


@pytest.mark.parametrize("mode", ["add_part", "repaint", "continue"])
def test_edit_preserves_measured_source_and_exact_resolved_native_frames(
    tmp_path: Path, mode: str
) -> None:
    """Original-rate floor/ceil mappings and Continue tail survive the model boundary."""
    model, factory, _ = fixture(tmp_path)
    task = edit_task(model, mode)
    original = model.source_root / str(task.command_id) / "original.wav"
    data = original.read_bytes()
    try:
        model.start(is_cancelled=lambda: False)
        assert (
            model.run(task, progress=lambda _: None, is_cancelled=lambda: False)
            is not None
        )
        assert factory.resident is not None
        request = factory.resident.requests[0]
        assert (
            request.prompt,
            request.lyrics,
            request.seed,
            request.source,
            request.source_path,
        ) == (
            task.task_params.prompt,
            task.task_params.lyrics,
            12025,
            task.task_params.source,
            str(original),
        )
        assert (
            request.region_start_frame,
            request.region_end_frame,
            request.target_frames,
        ) == ((48000, 96006, 96006) if mode == "continue" else (6, 47994, 48000))
        assert request.part_class == ("strings" if mode == "add_part" else None)
    finally:
        model.close()
    assert original.read_bytes() == data


@pytest.mark.parametrize("fault", ["model", "instance", "seed", "duration", "blank"])
def test_invalid_generation_creates_no_request_output_or_native_call(
    tmp_path: Path, fault: str
) -> None:
    """A bad job cannot consume a different mounted model or silently draw a new seed."""
    model, factory, task = fixture(tmp_path)
    if fault == "instance":
        task = task.model_copy(update={"instance_id": InstanceId("other")})
    else:
        changes = {
            "model": {"model": "other/model"},
            "seed": {"seed": None},
            "duration": {"seconds": 1},
            "blank": {"prompt": " \n "},
        }[fault]
        task = task.model_copy(
            update={"task_params": task.task_params.model_copy(update=changes)}
        )
    try:
        model.start(is_cancelled=lambda: False)
        with pytest.raises(ValueError):
            model.run(task, progress=lambda _: None, is_cancelled=lambda: False)
        assert factory.resident is not None and not factory.resident.requests
        assert not music_result_directory(model.root, task.command_id).exists()
        assert model.is_ready()
    finally:
        model.close()


def test_changed_original_is_rejected_before_edit_output_or_native_call(
    tmp_path: Path,
) -> None:
    """Admitted source metadata is rechecked against actual private source bytes."""
    model, factory, _ = fixture(tmp_path)
    task = edit_task(model, "repaint")
    path = model.source_root / str(task.command_id) / "original.wav"
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    try:
        model.start(is_cancelled=lambda: False)
        with pytest.raises(ValueError, match="admitted source"):
            model.run(task, progress=lambda _: None, is_cancelled=lambda: False)
        assert factory.resident is not None and not factory.resident.requests
        assert not music_result_directory(model.root, task.command_id).exists()
    finally:
        model.close()


def test_cancelled_resident_requires_normal_recreation_and_failed_close_retries(
    tmp_path: Path,
) -> None:
    """Cancellation cannot leave ready capacity; failed drain retains the same owner."""
    model, factory, task = fixture(tmp_path)
    model.start(is_cancelled=lambda: False)
    assert factory.resident is not None
    factory.resident.cancel_run = True
    assert model.run(task, progress=lambda _: None, is_cancelled=lambda: False) is None
    assert not model.is_ready()
    factory.resident.fail_close = True
    with pytest.raises(RuntimeError, match="drain failure"):
        model.close()
    model.close()
    assert factory.resident.closes == 2
    with pytest.raises(RuntimeError, match="recreation"):
        model.start(is_cancelled=lambda: False)


def test_busy_admission_does_not_create_another_result_or_cancel_active_work(
    tmp_path: Path,
) -> None:
    """The model actor also serializes pre-write ownership, before resident submission."""
    model, factory, task = fixture(tmp_path)
    model.start(is_cancelled=lambda: False)
    assert factory.resident is not None
    factory.resident.release.clear()
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            active = executor.submit(
                model.run, task, progress=lambda _: None, is_cancelled=lambda: False
            )
            assert factory.resident.entered.wait(3)
            other = task.model_copy(update={"command_id": CommandId("other")})
            with pytest.raises(RuntimeError, match="active operation"):
                model.run(other, progress=lambda _: None, is_cancelled=lambda: False)
            assert not music_result_directory(model.root, other.command_id).exists()
            factory.resident.release.set()
            assert active.result(timeout=3) is not None
            assert model.is_ready()
    finally:
        factory.resident.release.set()
        model.close()


def test_close_during_inspection_prevents_late_native_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Close waits for the actual model actor; preparation cannot escape its lifetime."""
    model, factory, _ = fixture(tmp_path)
    entered, release = threading.Event(), threading.Event()
    original = artifacts.observe_native_music_artifacts

    def inspect(
        card: ModelCard,
        sources: Mapping[tuple[str, str], Path],
        *,
        checkpoint: Callable[[], None] = lambda: None,
    ) -> ObservedMusicArtifacts:
        entered.set()
        assert release.wait(3)
        return original(card, sources, checkpoint=checkpoint)

    monkeypatch.setattr(model_module, "observe_native_music_artifacts", inspect)
    with ThreadPoolExecutor(max_workers=2) as executor:
        loading = executor.submit(model.start, is_cancelled=lambda: False)
        assert entered.wait(3)
        closing = executor.submit(model.close)
        assert model._stop_requested.wait(3)  # pyright: ignore[reportPrivateUsage] -- Pin the concurrent close transition before releasing inspection.
        assert not closing.done()
        release.set()
        with pytest.raises(InterruptedError):
            loading.result(timeout=3)
        closing.result(timeout=3)
    assert factory.calls == 0 and not model.root.exists() and not model.is_ready()


def test_model_adapter_runs_actual_supervisor_and_preserves_verified_wav(
    tmp_path: Path,
) -> None:
    """Real isolated synthetic CLI, process group, protocol, custody and WAV verification."""
    interpreter = create_resident_interpreter(tmp_path / "interpreter")
    fixture_model, _, task = fixture(tmp_path, interpreter_path=interpreter)
    model = NativeMusicModel(
        fixture_model.bound,
        fixture_model.runtime,
        fixture_model.sources,
        fixture_model.root,
        fixture_model.source_root,
    )
    output = None
    try:
        model.start(is_cancelled=lambda: False, timeout_seconds=10)
        output = model.run(
            task,
            progress=lambda _: None,
            is_cancelled=lambda: False,
            timeout_seconds=10,
        )
        assert output is not None
        measured = source_from_wav((output.directory / "result.wav").read_bytes())
        assert measured.sha256 == output.result.output.sha256
        assert measured.frames == 240000 and measured.sample_rate == 48000
        assert output.result.model_identity == model.policy.model_identity
    finally:
        model.close()
    assert output is not None and output.directory.exists()


def test_observation_can_refine_an_admitted_unknown_layout(tmp_path: Path) -> None:
    """Missing optional geometry is not a loading permission or a policy change."""
    original, factory, _ = fixture(tmp_path)
    unknown = original.policy.model_copy(update={"tensor_layouts": None})
    shard = original.bound.bound_shard.model_copy(
        update={"native_music_policy": unknown}
    )
    assignments = original.bound.instance.shard_assignments.model_copy(
        update={"runner_to_shard": {original.bound.bound_runner_id: shard}}
    )
    bound = original.bound.model_copy(
        update={
            "instance": original.bound.instance.model_copy(
                update={"shard_assignments": assignments}
            )
        }
    )
    model = NativeMusicModel(
        bound,
        original.runtime,
        original.sources,
        original.root,
        original.source_root,
        create_resident=factory,
    )
    try:
        ready = model.start(is_cancelled=lambda: False)
        assert ready.model_identity == unknown.model_identity
        assert ready.execution == unknown.execution and model.is_ready()
        assert model.policy == unknown and model.policy.tensor_layouts is None
    finally:
        model.close()


def test_idle_readiness_poll_cannot_race_native_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Actual resident actor contention cannot reject or retire a valid generation."""
    interpreter = create_resident_interpreter(tmp_path / "interpreter")
    original, _, task = fixture(tmp_path, interpreter_path=interpreter)
    captured: list[ResidentMusic] = []

    def create(
        interpreter: Path,
        workspace: NativeMusicWorkspace,
        configuration: NativeMusicServeConfig,
        *,
        payload_sha256: str,
        resource_guard: Callable[[int], str | None] | None = None,
    ) -> ResidentMusic:
        resident = ResidentMusic(
            interpreter,
            workspace,
            configuration,
            payload_sha256=payload_sha256,
            resource_guard=resource_guard,
        )
        captured.append(resident)
        return resident

    model = NativeMusicModel(
        original.bound,
        original.runtime,
        original.sources,
        original.root,
        original.source_root,
        create_resident=create,
    )
    entered, release = threading.Event(), threading.Event()
    try:
        model.start(is_cancelled=lambda: False, timeout_seconds=10)
        workspace = captured[0].workspace
        verify = workspace.verify

        def block_once() -> None:
            verify()
            if not entered.is_set():
                entered.set()
                assert release.wait(3)

        monkeypatch.setattr(workspace, "verify", block_once)
        with ThreadPoolExecutor(max_workers=2) as executor:
            polling = executor.submit(model.is_ready)
            assert entered.wait(3)
            generation = executor.submit(
                model.run,
                task,
                progress=lambda _: None,
                is_cancelled=lambda: False,
                timeout_seconds=10,
            )
            try:
                with pytest.raises(FutureTimeout):
                    generation.result(timeout=0.1)
            finally:
                release.set()
            assert polling.result(timeout=3)
            assert generation.result(timeout=3) is not None
        assert model.is_ready()
    finally:
        release.set()
        model.close()


@pytest.mark.parametrize("fault", ["parent_sync", "receipt_write"])
def test_failed_result_preparation_cannot_leave_an_untracked_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    """A pre-submit failure cleans only its new result; completed takes survive."""
    model, factory, original = fixture(tmp_path)
    model.start(is_cancelled=lambda: False)
    completed = model.run(original, progress=lambda _: None, is_cancelled=lambda: False)
    assert completed is not None and factory.resident is not None
    parent_inode = factory.resident.workspace.requests_identity.inode
    retained = (completed.directory / "result.wav").read_bytes()
    task = original.model_copy(update={"command_id": CommandId("failed-preparation")})
    directory = music_result_directory(model.root, task.command_id)
    failed = False
    sync = result_module.os.fsync
    write = result_module.write_private

    def sync_once(descriptor: int) -> None:
        nonlocal failed
        if not failed and result_module.os.fstat(descriptor).st_ino == parent_inode:
            failed = True
            raise OSError("controlled parent sync failure")
        sync(descriptor)

    def write_once(path: Path, data: bytes) -> None:
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("controlled receipt publication failure")
        write(path, data)

    monkeypatch.setattr(
        result_module.os, "fsync", sync_once
    ) if fault == "parent_sync" else monkeypatch.setattr(
        result_module, "write_private", write_once
    )
    try:
        with pytest.raises(OSError, match="controlled"):
            model.run(task, progress=lambda _: None, is_cancelled=lambda: False)
        assert failed and model.is_ready()
        assert len(factory.resident.requests) == 1
    finally:
        model.close()
    assert not directory.exists()
    assert (completed.directory / "result.wav").read_bytes() == retained


@pytest.mark.parametrize("interference", ["none", "replacement", "content"])
def test_failed_preparation_rollback_fences_and_retains_exact_retry_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interference: str,
) -> None:
    """Failed rollback cannot forget cleanup or remove even an empty replacement."""
    model, factory, task = fixture(tmp_path)
    model.start(is_cancelled=lambda: False)
    task = task.model_copy(update={"command_id": CommandId("failed-preparation")})
    directory = music_result_directory(model.root, task.command_id)
    write = result_module.write_private
    remove = result_module.os.rmdir
    refuse = True

    def no_receipt(path: Path, data: bytes) -> None:
        if refuse:
            raise OSError("controlled receipt failure")
        write(path, data)

    def no_remove(path: str | bytes | Path, *, dir_fd: int | None = None) -> None:
        if refuse and path == directory.name:
            raise OSError("controlled directory removal failure")
        remove(path, dir_fd=dir_fd)

    monkeypatch.setattr(result_module, "write_private", no_receipt)
    monkeypatch.setattr(result_module.os, "rmdir", no_remove)
    try:
        with pytest.raises(RuntimeError, match="preparation cleanup remains pending"):
            model.run(task, progress=lambda _: None, is_cancelled=lambda: False)
        assert directory.exists() and not model.is_ready()
        assert factory.resident is not None and not factory.resident.requests
        refuse = False
        if interference == "replacement":
            moved = directory.with_name("retained-original-inode")
            directory.rename(moved)
            directory.mkdir(mode=0o700)
            replacement = directory.stat().st_ino
            with pytest.raises(RuntimeError, match="cleanup remains pending"):
                model.close()
            assert directory.exists() and directory.stat().st_ino == replacement
            assert moved.exists()
            directory.rmdir()
            moved.rename(directory)
        elif interference == "content":
            foreign = directory / "result.wav"
            foreign.write_bytes(b"No native writer was admitted to create this file")
            with pytest.raises(RuntimeError, match="cleanup remains pending"):
                model.close()
            assert foreign.read_bytes() == (
                b"No native writer was admitted to create this file"
            )
            foreign.unlink()
        model.close()
        assert not directory.exists()
    finally:
        refuse = False
        model.close()
