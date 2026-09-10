# pyright: reportPrivateUsage=false, reportAny=false
"""The deterministic test video engine: planning, rendering, muxing, provisioning."""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import struct
from pathlib import Path
from typing import Any, cast

import pytest
from anyio import Path as AsyncPath

from skulk.download.download_utils import (
    is_model_directory_complete,
    resolve_model_in_path,
)
from skulk.shared import constants
from skulk.shared.backends import engine_of, resolve_node_backend
from skulk.shared.models.model_cards import ModelCard, ModelId, get_bundled_card
from skulk.shared.types.video import (
    VIDEO_OUTPUT_FILENAME,
    VIDEO_THUMBNAIL_FILENAME,
    VideoGenerationTaskParams,
    VideoStage,
)
from skulk.worker.runner.test_video.provision import (
    TEST_VIDEO_MODEL_ID,
    bundled_card_path,
    install_test_video_card,
    provision_test_video_model,
    register_test_video_card,
)
from skulk.worker.runner.test_video.render import (
    RenderPlan,
    mux_mp4,
    plan_render,
    render_clip,
)


def _card() -> ModelCard:
    card = asyncio.run(get_bundled_card(TEST_VIDEO_MODEL_ID))
    assert card is not None and card.video is not None
    return card


def _ignore_progress(
    stage: VideoStage, step: int | None, total_steps: int | None, fraction: float
) -> None:
    del stage, step, total_steps, fraction


def _boxes(data: bytes, start: int = 0, end: int | None = None) -> list[tuple[bytes, int, int]]:
    """Top-level (kind, payload_start, payload_end) triples of an ISO BMFF span."""

    end = len(data) if end is None else end
    boxes: list[tuple[bytes, int, int]] = []
    offset = start
    while offset < end:
        size, kind = struct.unpack(">I4s", data[offset : offset + 8])
        assert size >= 8
        boxes.append((kind, offset + 8, offset + size))
        offset += size
    assert offset == end
    return boxes


def _find(data: bytes, path: list[bytes], start: int = 0, end: int | None = None) -> tuple[int, int]:
    span = (start, len(data) if end is None else end)
    for kind in path:
        matches = [(s, e) for k, s, e in _boxes(data, *span) if k == kind]
        assert len(matches) == 1, f"{kind!r} not unique under {path}"
        span = matches[0]
    return span


def test_plan_resolves_canvas_grid_and_defaults_from_the_card() -> None:
    video = _card().video
    assert video is not None
    params = VideoGenerationTaskParams(
        prompt="a fox", model=str(TEST_VIDEO_MODEL_ID), seconds=1, aspect_ratio="16:9"
    )
    plan = plan_render(params, video)
    # 64 short edge, 16:9 gives 113.8 wide, snapped to the 8 pixel grid.
    assert (plan.width, plan.height) == (112, 64)
    # 8 fps for one second is 8 frames, aligned up onto the 4k+1 grid.
    assert plan.frame_count == 9 and plan.fps == 8
    assert plan.steps == video.default_steps and plan.audio is True
    assert plan.seed == plan_render(params, video).seed
    explicit = plan_render(
        params.model_copy(update={"size": "96x48", "seed": 7, "steps": 3, "audio": False}), video
    )
    assert (explicit.width, explicit.height, explicit.seed, explicit.steps) == (96, 48, 7, 3)
    assert explicit.audio is False


def test_render_is_deterministic_and_the_manifest_describes_the_file(tmp_path: Path) -> None:
    plan = RenderPlan(
        width=32, height=24, fps=8, frame_count=5, steps=2, seed=11, audio=True,
        sample_rate=32000, channels=2,
    )
    stages: list[tuple[VideoStage, int | None, int | None, float]] = []
    result = render_clip(
        plan,
        tmp_path / "one",
        progress=lambda stage, step, total, fraction: stages.append((stage, step, total, fraction)),
        is_cancelled=lambda: False,
        step_seconds=0,
    )
    assert result is not None
    manifest, stats = result
    container = (tmp_path / "one" / VIDEO_OUTPUT_FILENAME).read_bytes()
    thumbnail = (tmp_path / "one" / VIDEO_THUMBNAIL_FILENAME).read_bytes()
    assert manifest.sha256 == hashlib.sha256(container).hexdigest()
    assert manifest.size_bytes == len(container)
    assert manifest.thumbnail_sha256 == hashlib.sha256(thumbnail).hexdigest()
    assert manifest.thumbnail_size_bytes == len(thumbnail)
    assert thumbnail.startswith(b"\xff\xd8")
    assert (manifest.width, manifest.height, manifest.frame_count, manifest.fps) == (32, 24, 5, 8)
    assert abs(manifest.seconds - 5 / 8) < 1e-9
    assert (manifest.audio_sample_rate, manifest.audio_channels) == (32000, 2)
    assert stats.steps == 2 and stats.total_generation_time >= 0
    assert [stage for stage, *_ in stages] == ["encoding", "sampling", "sampling", "decoding", "muxing"]
    assert stages[2][1:] == (2, 2, 0.8)
    again = render_clip(
        plan, tmp_path / "two", progress=_ignore_progress, is_cancelled=lambda: False, step_seconds=0
    )
    assert again is not None and again[0].sha256 == manifest.sha256
    other = render_clip(
        dataclasses.replace(plan, seed=12),
        tmp_path / "three",
        progress=_ignore_progress,
        is_cancelled=lambda: False,
        step_seconds=0,
    )
    assert other is not None and other[0].sha256 != manifest.sha256


def test_container_is_a_well_formed_mp4_with_both_tracks(tmp_path: Path) -> None:
    frames = [b"\xff\xd8" + bytes([index]) * 10 + b"\xff\xd9" for index in range(3)]
    audio = b"\x00\x01" * 2 * 16
    container = mux_mp4(
        frames, width=16, height=8, fps=8, audio=audio, sample_rate=16, channels=2
    )
    top = [kind for kind, _s, _e in _boxes(container)]
    assert top == [b"ftyp", b"mdat", b"moov"]
    mdat_start, mdat_end = _find(container, [b"mdat"])
    assert container[mdat_start:mdat_end] == b"".join(frames) + audio
    moov = _find(container, [b"moov"])
    traks = [(s, e) for k, s, e in _boxes(container, *moov) if k == b"trak"]
    assert len(traks) == 2
    video_stbl = _find(container, [b"mdia", b"minf", b"stbl"], *traks[0])
    stsz = _find(container, [b"stsz"], *video_stbl)
    _version_flags, sample_size, count = struct.unpack(">IIII", container[stsz[0] : stsz[0] + 16])[:3]
    assert (sample_size, count) == (0, 3)
    sizes = struct.unpack(">3I", container[stsz[0] + 12 : stsz[0] + 24])
    assert list(sizes) == [len(frame) for frame in frames]
    stco = _find(container, [b"stco"], *video_stbl)
    assert struct.unpack(">II", container[stco[0] + 4 : stco[0] + 12]) == (1, mdat_start)
    audio_stbl = _find(container, [b"mdia", b"minf", b"stbl"], *traks[1])
    stsz_audio = _find(container, [b"stsz"], *audio_stbl)
    assert struct.unpack(">II", container[stsz_audio[0] + 4 : stsz_audio[0] + 12]) == (4, 16)
    stco_audio = _find(container, [b"stco"], *audio_stbl)
    assert struct.unpack(">II", container[stco_audio[0] + 4 : stco_audio[0] + 12]) == (
        1,
        mdat_start + sum(len(frame) for frame in frames),
    )
    silent = mux_mp4(frames, width=16, height=8, fps=8, audio=None, sample_rate=16, channels=2)
    moov = _find(silent, [b"moov"])
    assert sum(1 for k, _s, _e in _boxes(silent, *moov) if k == b"trak") == 1


def test_render_without_audio_declares_no_track(tmp_path: Path) -> None:
    plan = RenderPlan(
        width=16, height=16, fps=8, frame_count=1, steps=1, seed=1, audio=False,
        sample_rate=32000, channels=2,
    )
    result = render_clip(
        plan, tmp_path, progress=_ignore_progress, is_cancelled=lambda: False, step_seconds=0
    )
    assert result is not None
    assert result[0].audio_sample_rate is None and result[0].audio_channels is None


def test_render_stops_between_steps_when_cancelled(tmp_path: Path) -> None:
    plan = RenderPlan(
        width=16, height=16, fps=8, frame_count=1, steps=4, seed=1, audio=False,
        sample_rate=32000, channels=2,
    )
    calls = {"count": 0}

    def cancelled() -> bool:
        calls["count"] += 1
        return calls["count"] > 2

    result = render_clip(
        plan, tmp_path / "out", progress=_ignore_progress, is_cancelled=cancelled, step_seconds=0
    )
    assert result is None
    assert not (tmp_path / "out").exists()


def test_provision_writes_a_directory_the_resolver_accepts(tmp_path: Path) -> None:
    directory = provision_test_video_model(tmp_path)
    assert directory == tmp_path / TEST_VIDEO_MODEL_ID.normalize()
    assert is_model_directory_complete(directory)
    assert provision_test_video_model(tmp_path) == directory
    assert constants.SKULK_MODELS_PATH is not None and tmp_path in constants.SKULK_MODELS_PATH
    assert resolve_model_in_path(TEST_VIDEO_MODEL_ID, None, expected_card=_card()) == directory


def test_registering_the_card_copies_the_bundled_toml_once(tmp_path: Path) -> None:
    target = register_test_video_card(tmp_path / "custom")
    assert target == tmp_path / "custom" / (TEST_VIDEO_MODEL_ID.normalize() + ".toml")
    assert target.read_bytes() == bundled_card_path().read_bytes()
    target.write_text(target.read_text() + "\n# operator edit\n")
    assert register_test_video_card(tmp_path / "custom") == target
    assert target.read_text().endswith("# operator edit\n")
    loaded = asyncio.run(ModelCard.load_from_path(AsyncPath(target)))
    assert loaded.model_id == TEST_VIDEO_MODEL_ID and loaded.video is not None


def test_runner_refuses_any_card_but_its_own() -> None:
    from skulk.shared.types.common import NodeId
    from skulk.shared.types.tasks import LoadModel
    from skulk.shared.types.worker.instances import (
        BoundInstance,
        InstanceId,
        MlxRingInstance,
    )
    from skulk.shared.types.worker.runners import RunnerId, ShardAssignments
    from skulk.shared.types.worker.shards import PipelineShardMetadata
    from skulk.worker.runner.test_video.runner import Runner

    impostor = _card().model_copy(update={"model_id": ModelId("org/other-video")})
    runner_id = RunnerId("r")
    shard = PipelineShardMetadata(
        model_card=impostor, device_rank=0, world_size=1, start_layer=0, end_layer=1, n_layers=1
    )
    instance = MlxRingInstance(
        instance_id=InstanceId("i"),
        shard_assignments=ShardAssignments(
            model_id=impostor.model_id,
            node_to_runner={NodeId("n"): runner_id},
            runner_to_shard={runner_id: shard},
        ),
        hosts_by_node={},
        ephemeral_port=50000,
    )
    sent: list[object] = []

    class _Sender:
        def send(self, event: object) -> None:
            sent.append(event)

    runner = Runner(
        BoundInstance(instance=instance, bound_runner_id=runner_id, bound_node_id=NodeId("n")),
        cast("Any", _Sender()),
        cast("Any", None),
        cast("Any", None),
    )
    with pytest.raises(RuntimeError, match="serves only foxlight/test-video"):
        runner.handle_task(LoadModel(instance_id=instance.instance_id))


def test_installing_the_card_makes_it_visible_immediately(tmp_path: Path) -> None:
    from skulk.shared.models import model_cards

    model_cards._card_cache.pop(TEST_VIDEO_MODEL_ID, None)
    try:
        card = asyncio.run(install_test_video_card(tmp_path / "custom"))
        assert card.is_custom and card.model_id == TEST_VIDEO_MODEL_ID
        assert model_cards._card_cache[TEST_VIDEO_MODEL_ID] is card
    finally:
        model_cards._card_cache.pop(TEST_VIDEO_MODEL_ID, None)


def test_bundled_card_places_on_a_test_video_node() -> None:
    card = _card()
    node_tags = frozenset({"mlx", "mlx-metal", "test_video", "test_video-cpu"})
    resolved = resolve_node_backend(
        card.placement.compatible_backends, card.placement.backend_preference, node_tags
    )
    assert resolved == "test_video-cpu" and engine_of(resolved) == "test_video"
    assert resolve_node_backend(
        card.placement.compatible_backends, card.placement.backend_preference, frozenset({"mlx"})
    ) is None
    assert card.video is not None and card.video.audio_output
    assert card.video.frame_count_for_seconds(1) == 9


async def _drive_render(
    supervisor: object,
    instance_id: object,
    runner_id: object,
    command_id: object,
    params: VideoGenerationTaskParams,
    node: object,
) -> None:
    """Run the supervisor and walk the runner through one render."""

    import anyio

    from skulk.shared.types.common import CommandId, NodeId
    from skulk.shared.types.tasks import LoadModel, Shutdown, StartWarmup
    from skulk.shared.types.tasks import VideoGeneration as VideoGenerationTask
    from skulk.shared.types.worker.instances import InstanceId
    from skulk.shared.types.worker.runners import RunnerId
    from skulk.worker.runner.runner_supervisor import RunnerSupervisor

    assert isinstance(supervisor, RunnerSupervisor)
    assert isinstance(instance_id, str) and isinstance(runner_id, str)
    assert isinstance(command_id, str) and isinstance(node, str)
    instance = InstanceId(instance_id)
    async with anyio.create_task_group() as group:
        group.start_soon(supervisor.run)
        with anyio.fail_after(180):
            await supervisor.start_task(LoadModel(instance_id=instance), wait_for_terminal=True)
            await supervisor.start_task(StartWarmup(instance_id=instance), wait_for_terminal=True)
            await supervisor.start_task(
                VideoGenerationTask(
                    command_id=CommandId(command_id),
                    instance_id=instance,
                    task_params=params,
                    owner_node=NodeId(node),
                ),
                wait_for_terminal=True,
            )
            await supervisor.start_task(
                Shutdown(instance_id=instance, runner_id=RunnerId(runner_id)),
                wait_for_terminal=True,
            )
        supervisor.shutdown()


@pytest.mark.asyncio
async def test_runner_subprocess_renders_and_hands_the_manifest_to_the_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real runner in a spawned process: load, warm up, render, shut down.

    This is the closest thing to an end-to-end check without a node: the
    supervisor's data frames, the terminal manifest hook the worker streams
    from, and the file on disk must all agree.
    """

    import multiprocessing
    import sys

    import anyio
    from loguru import logger

    from skulk.shared.types.chunks import DataChunk, VideoChunk
    from skulk.shared.types.common import CommandId, NodeId
    from skulk.shared.types.events import Event
    from skulk.shared.types.worker.instances import (
        BoundInstance,
        InstanceId,
        MlxRingInstance,
    )
    from skulk.shared.types.worker.runners import RunnerId, ShardAssignments
    from skulk.shared.types.worker.shards import PipelineShardMetadata
    from skulk.utils.channels import channel
    from skulk.worker.runner.runner_supervisor import RunnerSupervisor

    # The supervisor spawns the runner the way the node does; restore the
    # process-wide start method afterwards so later tests see the default.
    previous_method = multiprocessing.get_start_method(allow_none=True)
    multiprocessing.set_start_method("spawn", force=True)
    # The supervisor hands the logger to the spawned process, which pickles
    # its sinks; pytest's captured stderr is not picklable, so log to an
    # enqueued file sink (the shape production uses) for the duration of the
    # spawn and restore the stream sink afterwards.
    logger.remove()
    sink_id = logger.add(str(tmp_path / "runner.log"), level="INFO", enqueue=True)
    # SKULK_HOME roots every Skulk directory on every platform (an absolute
    # value replaces the home prefix), so the spawned runner writes under
    # the test's temporary tree.
    cache_home = tmp_path / "home"
    monkeypatch.setenv("SKULK_HOME", str(cache_home))
    monkeypatch.setenv("SKULK_TEST_VIDEO_STEP_SECONDS", "0")
    card = await get_bundled_card(TEST_VIDEO_MODEL_ID)
    assert card is not None
    runner_id = RunnerId("test-video-runner")
    node = NodeId("video-node")
    shard = PipelineShardMetadata(
        model_card=card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=1,
        n_layers=1,
        resolved_backend="test_video-cpu",
    )
    instance = MlxRingInstance(
        instance_id=InstanceId("test-video-instance"),
        shard_assignments=ShardAssignments(
            model_id=card.model_id,
            node_to_runner={node: runner_id},
            runner_to_shard={runner_id: shard},
        ),
        hosts_by_node={},
        ephemeral_port=50000,
    )
    bound = BoundInstance(instance=instance, bound_runner_id=runner_id, bound_node_id=node)
    event_sender, _event_receiver = channel[Event](512)
    data_sender, data_receiver = channel[DataChunk](512)
    outputs: list[tuple[CommandId, NodeId | None, VideoChunk]] = []
    supervisor = RunnerSupervisor.create(
        bound_instance=bound,
        event_sender=event_sender,
        data_sender=data_sender,
        on_video_output=lambda command_id, owner, chunk: outputs.append((command_id, owner, chunk)),
    )
    command_id = CommandId("test-video-command")
    params = VideoGenerationTaskParams(
        prompt="a fox in the snow", model=str(card.model_id), seconds=1, seed=3
    )
    try:
        await _drive_render(supervisor, instance.instance_id, runner_id, command_id, params, node)
    finally:
        logger.remove(sink_id)
        logger.add(sys.stderr)
        multiprocessing.set_start_method(previous_method, force=True)
    frames: list[DataChunk] = []
    while True:
        try:
            frames.append(data_receiver.receive_nowait())
        except anyio.WouldBlock:
            break
    kinds = [frame.kind for frame in frames]
    assert kinds[0] == "started" and kinds[-1] == "completed"
    assert all(frame.command_id == command_id for frame in frames)
    terminal = frames[-1].chunk
    assert isinstance(terminal, VideoChunk) and terminal.output is not None
    assert terminal.finish_reason == "stop" and terminal.stats is not None
    progress = [frame.chunk for frame in frames[1:-1]]
    assert all(isinstance(chunk, VideoChunk) and chunk.output is None for chunk in progress)
    assert any(isinstance(chunk, VideoChunk) and chunk.stage == "sampling" for chunk in progress)
    assert outputs == [(command_id, node, terminal)]
    manifest = terminal.output
    assert (manifest.width, manifest.height, manifest.frame_count, manifest.fps) == (64, 64, 9, 8)
    rendered = list(cache_home.rglob(VIDEO_OUTPUT_FILENAME))
    assert len(rendered) == 1 and rendered[0].parent.name == str(command_id)
    assert hashlib.sha256(rendered[0].read_bytes()).hexdigest() == manifest.sha256
    assert manifest.size_bytes == rendered[0].stat().st_size
    thumbnail = rendered[0].parent / VIDEO_THUMBNAIL_FILENAME
    assert hashlib.sha256(thumbnail.read_bytes()).hexdigest() == manifest.thumbnail_sha256
