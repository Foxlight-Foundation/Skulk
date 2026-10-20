# pyright: reportPrivateUsage=false
"""Native controls, process cleanup and internal dispatch failure-state regression."""

from __future__ import annotations

import asyncio
import sys
import time
from concurrent.futures import Future
from pathlib import Path

import pytest

from skulk.api.types.api import VideoCapabilitySection
from skulk.shared.backends import ENGINE_UNAVAILABLE_FAILURE_MARKER
from skulk.shared.models.model_cards import ModelCard
from skulk.shared.tests.model_card_fixtures import load_fixture_card
from skulk.shared.types.chunks import ErrorChunk
from skulk.shared.types.common import CommandId, ModelId, NodeId
from skulk.shared.types.events import ChunkGenerated, Event, TaskStatusUpdated
from skulk.shared.types.tasks import Task, TaskId, TaskStatus, VideoGeneration
from skulk.shared.types.video import VideoGenerationTaskParams
from skulk.shared.types.worker.instances import (
    BoundInstance,
    InstanceId,
    MlxRingInstance,
)
from skulk.shared.types.worker.runners import RunnerId, ShardAssignments
from skulk.shared.types.worker.shards import PipelineShardMetadata
from skulk.utils.channels import mp_channel
from skulk.worker.runner.comfy import runner as comfy_runner_module
from skulk.worker.runner.mlx_video import runner as runner_module
from skulk.worker.runner.mlx_video.plan import (
    native_engine_settings,
    plan_native_job,
    selected_model_view,
)
from skulk.worker.runner.mlx_video.process import NativeProgress, run_native_job


def _card() -> ModelCard:
    card = asyncio.run(load_fixture_card(ModelId("Lightricks/LTX-2.5-distilled-comfy-int8")))
    assert card is not None
    return card


def _params(**updates: object) -> VideoGenerationTaskParams:
    return VideoGenerationTaskParams.model_validate({"model": str(_card().model_id), "prompt": "A fox in a meadow", "seconds": 2, "size": "768x512", "seed": 4, **updates})


def test_native_plan_preserves_exact_requested_workload(tmp_path: Path) -> None:
    job, plan = plan_native_job(_params(), _card(), tmp_path / "model", tmp_path / "output", tmp_path / "input")
    assert (job.width, job.height, job.frame_count, job.seed) == (768, 512, 49, 4)
    assert job.pipeline == "distilled_two_stage" and job.base_steps == 8
    assert job.audio and plan.sample_rate == 48000


def test_native_distilled_honors_its_fixed_ancestral_sampler(tmp_path: Path) -> None:
    job, _ = plan_native_job(_params(sampler="euler_ancestral"), _card(), tmp_path, tmp_path, tmp_path)
    assert job.pipeline == "distilled_two_stage"


def test_native_refuses_ambiguous_comfy_artifact_layout(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="pack-root"):
        selected_model_view(_card(), tmp_path)


def test_missing_native_engine_fails_placement_without_restart_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(_build: str) -> None:
        return None
    monkeypatch.setattr(runner_module, "prepared_runtime", missing)
    with pytest.raises(RuntimeError, match=ENGINE_UNAVAILABLE_FAILURE_MARKER):
        runner_module.resolve_placed_runtime("mlx_video-metal", "expected-build")


def test_unexpected_native_failure_publishes_failed_task(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    card = _card()
    identifier, node = RunnerId("native-test-runner"), NodeId("node")
    shard = PipelineShardMetadata(model_card=card, device_rank=0, world_size=1, start_layer=0, end_layer=card.n_layers, n_layers=card.n_layers, resolved_backend="mlx_video-metal", resolved_engine_build="tested-build")
    instance = MlxRingInstance(instance_id=InstanceId("native-test-instance"), shard_assignments=ShardAssignments(model_id=card.model_id, node_to_runner={node: identifier}, runner_to_shard={identifier: shard}), hosts_by_node={}, ephemeral_port=50000)
    events, received = mp_channel[Event]()
    tasks, task_receiver = mp_channel[Task]()
    cancels, cancel_receiver = mp_channel[TaskId]()
    runner = runner_module.Runner(BoundInstance(instance=instance, bound_runner_id=identifier, bound_node_id=node), events, task_receiver, cancel_receiver)
    runner.interpreter = tmp_path / "interpreter"
    def missing(_build: str) -> None:
        return None
    monkeypatch.setattr(runner_module, "prepared_runtime", missing)
    task = VideoGeneration(instance_id=instance.instance_id, command_id=CommandId("native-render"), task_params=_params())
    try:
        runner._dispatch_permits.acquire()
        runner._remember_task_id(task.task_id)
        runner._note_generation_started()
        runner._run_one_generation(task)
        finished: Future[None] = Future()
        finished.set_result(None)
        runner._finish_generation(task, finished)
        output = received.receive_at_least(3)
        assert any(isinstance(event, ChunkGenerated) and isinstance(event.chunk, ErrorChunk) for event in output)
        terminal = [event.task_status for event in output if isinstance(event, TaskStatusUpdated)]
        assert terminal == [TaskStatus.Failed]
    finally:
        events.close()
        tasks.close()
        cancels.close()
        received.close()
        task_receiver.close()
        cancel_receiver.close()


def test_rejected_comfy_source_publishes_failed_task(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    card = _card()
    identifier, node = RunnerId("native-test-runner"), NodeId("node")
    shard = PipelineShardMetadata(model_card=card, device_rank=0, world_size=1, start_layer=0, end_layer=card.n_layers, n_layers=card.n_layers, resolved_backend="mlx_video-metal", resolved_engine_build="tested-build")
    instance = MlxRingInstance(instance_id=InstanceId("native-test-instance"), shard_assignments=ShardAssignments(model_id=card.model_id, node_to_runner={node: identifier}, runner_to_shard={identifier: shard}), hosts_by_node={}, ephemeral_port=50000)
    events, received = mp_channel[Event]()
    tasks, task_receiver = mp_channel[Task]()
    cancels, cancel_receiver = mp_channel[TaskId]()
    runner = comfy_runner_module.Runner(BoundInstance(instance=instance, bound_runner_id=identifier, bound_node_id=node), events, task_receiver, cancel_receiver)
    def rejected(_task: TaskId, _command: CommandId, _params: VideoGenerationTaskParams) -> None:
        raise ValueError("source preparation refused malformed audio")
    monkeypatch.setattr(runner, "_render", rejected)
    task = VideoGeneration(instance_id=instance.instance_id, command_id=CommandId("native-render"), task_params=_params())
    try:
        runner._dispatch_permits.acquire()
        runner._remember_task_id(task.task_id)
        runner._note_generation_started()
        runner._run_one_generation(task)
        finished: Future[None] = Future()
        finished.set_result(None)
        runner._finish_generation(task, finished)
        output = received.receive_at_least(3)
        assert any(isinstance(event, ChunkGenerated) and isinstance(event.chunk, ErrorChunk) for event in output)
        terminal = [event.task_status for event in output if isinstance(event, TaskStatusUpdated)]
        assert terminal == [TaskStatus.Failed]
    finally:
        events.close()
        tasks.close()
        cancels.close()
        received.close()
        task_receiver.close()
        cancel_receiver.close()


@pytest.mark.parametrize("prerequisite", ["audio_output", "frame_grid_multiple"])
def test_metadata_does_not_offer_malformed_ltx_pipeline(prerequisite: str) -> None:
    card = _card()
    assert card.video is not None
    broken = card.model_copy(update={"video": card.video.model_copy(update={prerequisite: False if prerequisite == "audio_output" else 1})})
    metadata = VideoCapabilitySection.from_model_card(broken)
    assert metadata is not None
    assert metadata.pipelines == [] and metadata.decoders == []


def test_cancel_terminates_native_process_group(tmp_path: Path) -> None:
    interpreter = tmp_path / "interpreter"
    # A real isolated group with a child lets this test catch the orphaned
    # FFmpeg failure class, independent of Metal or model artifacts.
    interpreter.write_text(f"#!{sys.executable}\nimport os,time,json,subprocess\nfrom pathlib import Path\nchild=subprocess.Popen([{sys.executable!r},'-c','import time;time.sleep(60)'])\nPath({str(tmp_path / 'child.pid')!r}).write_text(str(child.pid))\nprint(json.dumps({{'stage':'sampling','step':1,'total_steps':8,'sampling_stage':1}}),flush=True)\ntime.sleep(60)\n")
    interpreter.chmod(0o755)
    job, _ = plan_native_job(_params(), _card(), tmp_path / "model", tmp_path / "output", tmp_path / "input")
    records: list[NativeProgress] = []
    started = time.monotonic()
    assert run_native_job(interpreter, job, progress=records.append, is_cancelled=lambda: bool(records)) is None
    assert time.monotonic() - started < 6
    child = int((tmp_path / "child.pid").read_text())
    # A terminated child may briefly remain a zombie until the OS reaps it.
    import psutil

    try:
        process = psutil.Process(child)
        assert process.status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        pass
    assert not (tmp_path / "output/job.json").exists()


def test_native_hq_plan_uses_its_own_step_and_guidance_defaults(tmp_path: Path) -> None:
    """HQ metadata and the native job agree without loading model weights."""
    card = asyncio.run(load_fixture_card(ModelId("Lightricks/LTX-2.5-dev-comfy-int8")))
    assert card is not None
    job, render = plan_native_job(_params(pipeline="hq_two_stage"), card, tmp_path, tmp_path, tmp_path)
    assert job.base_steps == render.steps == 15
    assert job.guidance is not None and job.guidance.modality_scale == 3.0


@pytest.mark.parametrize("high_quality", [True, False])
def test_native_job_record_preserves_its_actual_backend_profile(tmp_path: Path, high_quality: bool) -> None:
    """Public audit values follow the native job, including a nondefault adapter."""
    card = asyncio.run(load_fixture_card(ModelId("Lightricks/LTX-2.5-dev-comfy-int8")))
    assert card is not None
    params = _params(pipeline="hq_two_stage" if high_quality else "dev_two_stage")
    job, _ = plan_native_job(params, card, tmp_path, tmp_path, tmp_path)
    job = job.model_copy(update={"refinement_strength": 0.375})
    settings = native_engine_settings(job, steps=18 if high_quality else 33, frame_count=49)
    assert settings.steps == (18 if high_quality else 33)
    assert settings.adapter_strength == 0.375
    assert settings.stage_lora_strengths == ({"base": 0.0, "refine": 0.375} if high_quality else {})
    assert settings.guidance_video_rescale == (0.45 if high_quality else None)
    assert settings.guidance_audio_rescale == (1.0 if high_quality else None)
    assert settings.pipeline == job.pipeline and settings.guidance == job.guidance
    assert settings.frame_count == 49 and settings.width == job.width
