# pyright: reportPrivateUsage=false, reportAny=false
"""The ComfyUI video engine: graph binding, server control, and the runner."""

from __future__ import annotations

import asyncio
import hashlib
import shutil
import sys
from pathlib import Path
from typing import Any, cast

import pytest

from skulk.shared.backends import COMFY_BIN_ENV, COMFY_ROOT_ENV
from skulk.shared.models.model_cards import (
    ModelCard,
    ModelId,
    VideoMode,
    get_bundled_card,
)
from skulk.shared.types.chunks import ErrorChunk, VideoChunk
from skulk.shared.types.common import CommandId, NodeId
from skulk.shared.types.events import ChunkGenerated
from skulk.shared.types.tasks import (
    LoadModel,
    Shutdown,
    StartWarmup,
    TaskId,
    VideoGeneration,
)
from skulk.shared.types.video import (
    VIDEO_OUTPUT_FILENAME,
    VIDEO_THUMBNAIL_FILENAME,
    VideoGenerationTaskParams,
    VideoReferenceSpec,
)
from skulk.shared.types.worker.instances import (
    BoundInstance,
    InstanceId,
    MlxRingInstance,
)
from skulk.shared.types.worker.runners import RunnerId, ShardAssignments
from skulk.shared.types.worker.shards import PipelineShardMetadata
from skulk.worker.runner.comfy import orphan_sweep
from skulk.worker.runner.comfy import runner as runner_module
from skulk.worker.runner.comfy.graph import (
    NODE_AUDIO_VAE,
    NODE_CONDITION,
    NODE_CREATE_VIDEO,
    NODE_DECODE_AUDIO,
    NODE_LORA,
    NODE_SAVE_THUMBNAIL,
    NODE_SAVE_VIDEO,
    NODE_SCHEDULER,
    NODE_SHIFT,
    bind_references,
    build_prompt,
    extra_model_paths_yaml,
    plan_comfy_render,
    resolve_model_files,
    stage_for_node,
)
from skulk.worker.runner.comfy.runner import Runner
from skulk.worker.runner.comfy.server import server_args

FL2VA_ID = ModelId("Comfy-Org/MiniMax-H3-FL2VA-comfy-int8")
REF2VA_ID = ModelId("Comfy-Org/MiniMax-H3-Ref2VA-comfy-int8")
_DIGEST = "0" * 64


def _card(model_id: ModelId) -> ModelCard:
    card = asyncio.run(get_bundled_card(model_id))
    assert card is not None and card.video is not None
    return card


def _params(model_id: ModelId, **overrides: Any) -> VideoGenerationTaskParams:
    base: dict[str, Any] = {"prompt": "a fox at dusk", "model": str(model_id), "seconds": 5, "seed": 7}
    base.update(overrides)
    return VideoGenerationTaskParams(**base)


def _reference(slot: int, kind: str, role: str, media_type: str, local_path: Path) -> VideoReferenceSpec:
    return VideoReferenceSpec(
        slot=slot, kind=cast("Any", kind), role=cast("Any", role), media_type=media_type,
        size_bytes=1, sha256=_DIGEST, local_path=str(local_path),
    )


def test_h3_cards_resolve_their_loader_files() -> None:
    fl = resolve_model_files(_card(FL2VA_ID))
    assert fl.diffusion_model == "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
    assert fl.text_encoder == "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
    assert fl.video_vae == "minimax_h3_video_vae_fp16.safetensors"
    assert fl.audio_vae == "minimax_h3_audio_vae_fp32.safetensors"
    ref = resolve_model_files(_card(REF2VA_ID))
    assert ref.diffusion_model == "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
    assert (ref.video_vae, ref.audio_vae) == (fl.video_vae, fl.audio_vae)


def test_plan_takes_steps_and_shifts_from_the_named_adapter() -> None:
    card = _card(FL2VA_ID)
    full = plan_comfy_render(_params(FL2VA_ID), card)
    assert full.adapter is None and full.steps == 20 and (full.video_shift, full.audio_shift) == (12.0, 3.0)
    assert (full.plan.width, full.plan.height, full.plan.frame_count, full.plan.fps) == (768, 768, 124, 24)
    # Wide ratios keep their shape inside the trained pixel budget.
    wide = plan_comfy_render(_params(FL2VA_ID, aspect_ratio="21:9"), card).plan
    assert (wide.width, wide.height) == (1536, 672) and wide.width * wide.height <= 1032192
    tall = plan_comfy_render(_params(FL2VA_ID, aspect_ratio="9:16"), card).plan
    assert (tall.width, tall.height) == (768, 1344)
    turbo = plan_comfy_render(_params(FL2VA_ID, lora="turbo_fl2v_4step_768p"), card)
    assert turbo.adapter is not None and turbo.adapter.file == "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"
    assert turbo.steps == 4 and turbo.video_shift == 6.0 and turbo.adapter.strength == 1.0
    explicit = plan_comfy_render(_params(FL2VA_ID, lora="turbo_fl2v_8step", steps=6, lora_strength=0.5), card)
    assert explicit.steps == 6 and explicit.adapter is not None and explicit.adapter.strength == 0.5
    with pytest.raises(ValueError, match="no lora companion"):
        plan_comfy_render(_params(FL2VA_ID, lora="nope"), card)
    with pytest.raises(ValueError, match="is a model_patch"):
        plan_comfy_render(_params(FL2VA_ID, lora="fun_controlnet_union"), card)
    with pytest.raises(ValueError, match="does not serve mode"):
        plan_comfy_render(_params(FL2VA_ID, mode=VideoMode.ReferenceToAudioVideo), card)


def test_t2va_graph_mirrors_the_official_template() -> None:
    card = _card(FL2VA_ID)
    render = plan_comfy_render(_params(FL2VA_ID, aspect_ratio="16:9"), card)
    prompt = build_prompt(render, _params(FL2VA_ID, aspect_ratio="16:9"), (), "cmd-1")
    assert prompt["unet"]["inputs"] == {"unet_name": render.files.diffusion_model, "weight_dtype": "default"}
    assert prompt["clip"]["inputs"]["type"] == "minimax"
    assert NODE_LORA not in prompt
    assert prompt[NODE_SHIFT]["inputs"] == {"model": ["unet", 0], "shift_video": 12.0, "shift_audio": 3.0}
    condition = prompt[NODE_CONDITION]
    assert condition["class_type"] == "MiniMaxH3ImageToVideo"
    assert condition["inputs"]["width"] == 1344 and condition["inputs"]["height"] == 768
    assert condition["inputs"]["length"] == 124 and "first_frame" not in condition["inputs"]
    assert prompt[NODE_SCHEDULER]["inputs"]["steps"] == 20 and prompt[NODE_SCHEDULER]["inputs"]["model"] == [NODE_SHIFT, 0]
    assert prompt["sampler_select"]["inputs"]["sampler_name"] == "res_multistep"
    assert prompt["noise"]["inputs"]["noise_seed"] == 7
    assert prompt[NODE_CREATE_VIDEO]["inputs"]["audio"] == [NODE_DECODE_AUDIO, 0]
    assert prompt[NODE_SAVE_VIDEO]["inputs"]["filename_prefix"] == "cmd-1/output"
    assert prompt[NODE_SAVE_VIDEO]["inputs"]["format"] == "mp4" and prompt[NODE_SAVE_VIDEO]["inputs"]["format.codec"] == "h264"
    assert prompt[NODE_SAVE_THUMBNAIL]["inputs"]["filename_prefix"] == "cmd-1/thumbnail"
    silent = build_prompt(plan_comfy_render(_params(FL2VA_ID, audio=False), card), _params(FL2VA_ID, audio=False), (), "c")
    assert NODE_DECODE_AUDIO not in silent and NODE_AUDIO_VAE not in silent
    assert "audio" not in silent[NODE_CREATE_VIDEO]["inputs"]
    turbo_params = _params(FL2VA_ID, lora="turbo_fl2v_8step")
    turbo = build_prompt(plan_comfy_render(turbo_params, card), turbo_params, (), "c")
    assert turbo[NODE_LORA]["inputs"]["lora_name"].startswith("minimax_h3_fl2v_turbo_8step")
    assert turbo[NODE_SHIFT]["inputs"]["model"] == [NODE_LORA, 0]


def test_keyframe_graph_loads_first_and_last_frames(tmp_path: Path) -> None:
    card = _card(FL2VA_ID)
    input_dir = tmp_path / "video_input"
    (input_dir / "cmd").mkdir(parents=True)
    first = input_dir / "cmd" / "0.png"
    last = input_dir / "cmd" / "1.jpg"
    first.write_bytes(b"x")
    last.write_bytes(b"x")
    refs = (
        _reference(0, "image", "first_frame", "image/png", first),
        _reference(1, "image", "last_frame", "image/jpeg", last),
    )
    params = _params(FL2VA_ID, references=refs, reference_bytes=2, total_input_chunks=2)
    bindings = bind_references(params.references, input_dir)
    assert [b.input_name for b in bindings] == ["cmd/0.png", "cmd/1.jpg"]
    render = plan_comfy_render(params, card)
    assert render.mode is VideoMode.FramesToAudioVideo
    prompt = build_prompt(render, params, bindings, "cmd")
    assert prompt["load_first_frame"] == {"class_type": "LoadImage", "inputs": {"image": "cmd/0.png"}, "_meta": {"title": "Load first frame"}}
    assert prompt[NODE_CONDITION]["inputs"]["first_frame"] == ["load_first_frame", 0]
    assert prompt[NODE_CONDITION]["inputs"]["last_frame"] == ["load_last_frame", 0]
    outside = (_reference(0, "image", "first_frame", "image/png", tmp_path / "elsewhere.png"),)
    with pytest.raises(ValueError, match="outside the video input directory"):
        bind_references(outside, input_dir)
    with pytest.raises(ValueError, match="no local file"):
        bind_references((refs[0].model_copy(update={"local_path": None}),), input_dir)


def test_reference_graph_numbers_attachments_per_kind(tmp_path: Path) -> None:
    card = _card(REF2VA_ID)
    input_dir = tmp_path / "video_input"
    (input_dir / "cmd").mkdir(parents=True)
    names = ["0.png", "1.mp4", "2.jpg", "3.wav"]
    for name in names:
        (input_dir / "cmd" / name).write_bytes(b"x")
    refs = (
        _reference(0, "image", "reference", "image/png", input_dir / "cmd" / names[0]),
        _reference(1, "video", "reference", "video/mp4", input_dir / "cmd" / names[1]),
        _reference(2, "image", "reference", "image/jpeg", input_dir / "cmd" / names[2]),
        _reference(3, "audio", "reference", "audio/wav", input_dir / "cmd" / names[3]),
    )
    params = _params(REF2VA_ID, references=refs, reference_bytes=4, total_input_chunks=4)
    render = plan_comfy_render(params, card)
    assert render.mode is VideoMode.ReferenceToAudioVideo
    prompt = build_prompt(render, params, bind_references(params.references, input_dir), "cmd")
    condition = prompt[NODE_CONDITION]
    assert condition["class_type"] == "MiniMaxH3ReferenceToVideo"
    inputs = condition["inputs"]
    assert inputs["ref_images.ref_image_0"] == ["ref_image_0", 0]
    assert inputs["ref_images.ref_image_1"] == ["ref_image_1", 0]
    assert inputs["ref_videos.ref_video_0"] == ["ref_video_components_0", 0]
    assert inputs["ref_video_audios.ref_video_audio_0"] == ["ref_video_components_0", 1]
    assert inputs["ref_audios.ref_audio_0"] == ["ref_audio_0", 0]
    assert inputs["audio_vae"] == [NODE_AUDIO_VAE, 0] and inputs["ref_image_size"] == "match"
    assert prompt["ref_video_0"] == {"class_type": "LoadVideo", "inputs": {"file": "cmd/1.mp4"}, "_meta": {"title": "Reference video 1"}}
    assert prompt["ref_video_components_0"]["class_type"] == "GetVideoComponents"
    assert prompt["ref_image_1"]["inputs"]["image"] == "cmd/2.jpg"
    assert prompt["ref_audio_0"]["inputs"]["audio"] == "cmd/3.wav"
    keyframe = (_reference(0, "image", "first_frame", "image/png", input_dir / "cmd" / names[0]),)
    bad = _params(REF2VA_ID, mode=VideoMode.ReferenceToAudioVideo, references=keyframe, reference_bytes=1, total_input_chunks=1)
    with pytest.raises(ValueError, match="numbered references only"):
        build_prompt(plan_comfy_render(bad, card), bad, bind_references(bad.references, input_dir), "cmd")


def test_extra_model_paths_lists_the_folders_the_artifact_has(tmp_path: Path) -> None:
    for folder in ("diffusion_models", "vae", "loras"):
        (tmp_path / folder).mkdir()
    text = extra_model_paths_yaml(tmp_path)
    assert text.splitlines()[0] == "skulk:"
    assert f"  base_path: {tmp_path.resolve().as_posix()}" in text
    assert "  diffusion_models: diffusion_models" in text and "  loras: loras" in text
    assert "text_encoders" not in text and "embeddings" not in text


def test_server_args_are_headless_and_skulk_owned(tmp_path: Path) -> None:
    args = server_args(
        Path("/venv/bin/python"), Path("/opt/ComfyUI"), port=5555, extra_model_paths=tmp_path / "x.yaml",
        input_dir=tmp_path / "in", output_dir=tmp_path / "out", temp_dir=tmp_path / "tmp", user_dir=tmp_path / "user",
        extra=("--bf16-vae",),
    )
    assert args[:2] == ["/venv/bin/python", "/opt/ComfyUI/main.py"]
    assert args[args.index("--port") + 1] == "5555" and args[args.index("--listen") + 1] == "127.0.0.1"
    for flag in ("--disable-auto-launch", "--disable-all-custom-nodes", "--disable-api-nodes", "--disable-metadata"):
        assert flag in args
    assert args[args.index("--user-directory") + 1] == str(tmp_path / "user") and args[-1] == "--bf16-vae"


def test_stage_mapping_covers_every_graph_node() -> None:
    assert stage_for_node("sampler") == "sampling" and stage_for_node("decode_audio") == "decoding"
    assert stage_for_node("ref_video_components_1") == "encoding" and stage_for_node("load_last_frame") == "encoding"
    assert stage_for_node("save_thumbnail") == "muxing" and stage_for_node("unknown") is None


def _proc_entry(root: Path, pid: int, ppid: int, argv: list[str]) -> None:
    entry = root / str(pid)
    entry.mkdir()
    (entry / "stat").write_text(f"{pid} (python) S {ppid} 1 1 0 -1")
    (entry / "cmdline").write_bytes(b"\x00".join(arg.encode() for arg in argv) + b"\x00")


def test_orphan_sweep_matches_only_skulk_launched_servers(tmp_path: Path) -> None:
    proc = tmp_path / "proc"
    proc.mkdir()
    marker = tmp_path / "cache" / "comfy"
    skulk_args = ["/venv/bin/python", "/opt/ComfyUI/main.py", "--listen", "127.0.0.1", "--user-directory", str(marker / "r1" / "user")]
    _proc_entry(proc, 100, 1, skulk_args)
    _proc_entry(proc, 101, 4242, skulk_args)
    _proc_entry(proc, 102, 1, ["/venv/bin/python", "/opt/ComfyUI/main.py", "--user-directory", "/home/op/comfy/user"])
    _proc_entry(proc, 103, 1, ["/venv/bin/python", "/opt/ComfyUI/main.py"])
    _proc_entry(proc, 104, 1, ["pgrep", "-f", "main.py", "--user-directory", str(marker / "x")])
    (proc / "self").mkdir()
    assert orphan_sweep.find_orphaned_comfy_pids(proc, marker) == [100]


def _bound_instance(card: ModelCard, runner_id: RunnerId, node: NodeId) -> BoundInstance:
    shard = PipelineShardMetadata(
        model_card=card, device_rank=0, world_size=1, start_layer=0, end_layer=1, n_layers=1,
        resolved_backend="comfy-cuda",
    )
    instance = MlxRingInstance(
        instance_id=InstanceId("comfy-instance"),
        shard_assignments=ShardAssignments(
            model_id=card.model_id, node_to_runner={node: runner_id}, runner_to_shard={runner_id: shard}
        ),
        hosts_by_node={},
        ephemeral_port=50000,
    )
    return BoundInstance(instance=instance, bound_runner_id=runner_id, bound_node_id=node)


class _Sender:
    def __init__(self) -> None:
        self.events: list[Any] = []

    def send(self, event: Any) -> None:
        self.events.append(event)


class _Cancels:
    def __init__(self) -> None:
        self.pending: list[TaskId] = []

    def receive_nowait(self) -> TaskId:
        from anyio import WouldBlock

        if not self.pending:
            raise WouldBlock
        return self.pending.pop(0)


@pytest.fixture
def fake_comfy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake ComfyUI checkout plus a staged H3 artifact, wired through the env."""
    root = tmp_path / "ComfyUI"
    root.mkdir()
    shutil.copyfile(Path(__file__).with_name("fake_comfy_main.py"), root / "main.py")
    monkeypatch.setenv(COMFY_BIN_ENV, sys.executable)
    monkeypatch.setenv(COMFY_ROOT_ENV, str(root))
    monkeypatch.setenv("FAKE_COMFY_STEP_SECONDS", "0.01")
    monkeypatch.delenv("FAKE_COMFY_FAIL_NODE", raising=False)
    card = _card(FL2VA_ID)
    files = resolve_model_files(card)
    model_dir = tmp_path / "model"
    for folder, name in (("diffusion_models", files.diffusion_model), ("text_encoders", files.text_encoder), ("vae", files.video_vae), ("vae", files.audio_vae)):
        assert name is not None
        (model_dir / folder).mkdir(parents=True, exist_ok=True)
        (model_dir / folder / name).write_bytes(b"")
    monkeypatch.setattr(runner_module, "_model_directory", lambda card: model_dir)  # pyright: ignore[reportUnknownLambdaType, reportUnknownArgumentType]
    monkeypatch.setattr(runner_module, "SKULK_CACHE_HOME", tmp_path / "cache")
    monkeypatch.setattr(runner_module, "SKULK_VIDEO_INPUT_DIR", tmp_path / "video_input")
    monkeypatch.setattr(runner_module, "SKULK_VIDEO_OUTPUT_DIR", tmp_path / "video_output")
    return tmp_path


def _runner(sender: _Sender, cancels: _Cancels) -> Runner:
    card = _card(FL2VA_ID)
    bound = _bound_instance(card, RunnerId("comfy-runner"), NodeId("n"))
    return Runner(bound, cast("Any", sender), cast("Any", None), cast("Any", cancels))


def _video_chunks(sender: _Sender) -> list[VideoChunk]:
    return [e.chunk for e in sender.events if isinstance(e, ChunkGenerated) and isinstance(e.chunk, VideoChunk)]


def test_runner_renders_through_a_comfy_server(fake_comfy: Path) -> None:
    sender = _Sender()
    runner = _runner(sender, _Cancels())
    instance = runner.bound_instance.instance.instance_id
    try:
        runner.handle_task(LoadModel(instance_id=instance))
        assert runner.server is not None and runner.server.alive()
        assert (fake_comfy / "cache" / "comfy" / "comfy-runner" / "extra_model_paths.yaml").read_text().startswith("skulk:")
        runner.handle_task(StartWarmup(instance_id=instance))
        params = _params(FL2VA_ID, seconds=4, steps=3, aspect_ratio="16:9")
        command = CommandId("cmd-render")
        runner.handle_task(VideoGeneration(command_id=command, instance_id=instance, task_params=params, owner_node=NodeId("n")))
        chunks = _video_chunks(sender)
        stages = [chunk.stage for chunk in chunks]
        assert stages[0] == "queued" and "encoding" in stages and "decoding" in stages
        sampling = [chunk for chunk in chunks if chunk.stage == "sampling"]
        assert [chunk.step for chunk in sampling] == [1, 2, 3] and sampling[-1].total_steps == 3
        terminal = chunks[-1]
        assert terminal.finish_reason == "stop" and terminal.output is not None and terminal.stats is not None
        out_dir = fake_comfy / "video_output" / str(command)
        container = (out_dir / VIDEO_OUTPUT_FILENAME).read_bytes()
        thumbnail = (out_dir / VIDEO_THUMBNAIL_FILENAME).read_bytes()
        assert terminal.output.sha256 == hashlib.sha256(container).hexdigest()
        assert terminal.output.size_bytes == len(container) and thumbnail.startswith(b"\xff\xd8")
        assert terminal.output.thumbnail_sha256 == hashlib.sha256(thumbnail).hexdigest()
        assert (terminal.output.width, terminal.output.height, terminal.output.frame_count, terminal.output.fps) == (1344, 768, 107, 24)
        assert (terminal.output.audio_sample_rate, terminal.output.audio_channels) == (32000, 2)
        assert terminal.stats.steps == 3 and terminal.stats.total_generation_time > 0
        # Three fake steps of 10 ms each: the mean covers every step, not two.
        assert terminal.stats.seconds_per_step >= 0.01
        assert not list(out_dir.glob("*_00001_.*"))
        # The graph ComfyUI received is the one the builder produced.
        submitted = next(fake_comfy.glob("video_output/*.prompt.json"))
        assert '"MiniMaxH3ImageToVideo"' in submitted.read_text()
    finally:
        runner.handle_task(Shutdown(instance_id=instance, runner_id=RunnerId("comfy-runner")))
    assert runner.server is None


def test_runner_cancel_interrupts_the_server_and_keeps_serving(fake_comfy: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_COMFY_STEP_SECONDS", "0.3")
    sender = _Sender()
    cancels = _Cancels()
    runner = _runner(sender, cancels)
    instance = runner.bound_instance.instance.instance_id
    try:
        runner.handle_task(LoadModel(instance_id=instance))
        runner.handle_task(StartWarmup(instance_id=instance))
        task = VideoGeneration(command_id=CommandId("cmd-cancel"), instance_id=instance, task_params=_params(FL2VA_ID, steps=50), owner_node=NodeId("n"))
        cancels.pending.append(task.task_id)
        runner.handle_task(task)
        chunks = _video_chunks(sender)
        assert all(chunk.finish_reason is None for chunk in chunks)
        assert not (fake_comfy / "video_output" / "cmd-cancel").exists()
        assert runner._is_cancelled(task.task_id)
        # The server survived and the next render succeeds.
        sender.events.clear()
        runner.handle_task(VideoGeneration(command_id=CommandId("cmd-after"), instance_id=instance, task_params=_params(FL2VA_ID, steps=1), owner_node=NodeId("n")))
        assert _video_chunks(sender)[-1].finish_reason == "stop"
    finally:
        runner.handle_task(Shutdown(instance_id=instance, runner_id=RunnerId("comfy-runner")))


def test_runner_execution_error_fails_the_task_only(fake_comfy: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAKE_COMFY_FAIL_NODE", "decode_video")
    sender = _Sender()
    runner = _runner(sender, _Cancels())
    instance = runner.bound_instance.instance.instance_id
    try:
        runner.handle_task(LoadModel(instance_id=instance))
        runner.handle_task(StartWarmup(instance_id=instance))
        runner.handle_task(VideoGeneration(command_id=CommandId("cmd-fail"), instance_id=instance, task_params=_params(FL2VA_ID, steps=1), owner_node=NodeId("n")))
        errors = [e.chunk for e in sender.events if isinstance(e, ChunkGenerated) and isinstance(e.chunk, ErrorChunk)]
        assert len(errors) == 1 and "fake failure at decode_video" in (errors[0].error_message or "")
        assert runner.server is not None and runner.server.alive()
        # A graph ComfyUI rejects outright (a missing reference file) also fails only the task.
        missing = _reference(0, "image", "first_frame", "image/png", fake_comfy / "video_input" / "cmd-bad" / "0.png")
        bad = _params(FL2VA_ID, steps=1, references=(missing,), reference_bytes=1, total_input_chunks=1)
        runner.handle_task(VideoGeneration(command_id=CommandId("cmd-bad"), instance_id=instance, task_params=bad, owner_node=NodeId("n")))
        errors = [e.chunk for e in sender.events if isinstance(e, ChunkGenerated) and isinstance(e.chunk, ErrorChunk)]
        assert len(errors) == 2 and "rejected the graph" in (errors[1].error_message or "")
        assert runner.server.alive()
    finally:
        runner.handle_task(Shutdown(instance_id=instance, runner_id=RunnerId("comfy-runner")))


def test_runner_refuses_to_start_without_a_configured_install(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(COMFY_BIN_ENV, raising=False)
    monkeypatch.delenv(COMFY_ROOT_ENV, raising=False)
    runner = _runner(_Sender(), _Cancels())
    with pytest.raises(RuntimeError, match="not both set"):
        runner.handle_task(LoadModel(instance_id=runner.bound_instance.instance.instance_id))


def test_runner_dies_when_the_server_dies(fake_comfy: Path) -> None:
    sender = _Sender()
    runner = _runner(sender, _Cancels())
    instance = runner.bound_instance.instance.instance_id
    runner.handle_task(LoadModel(instance_id=instance))
    runner.handle_task(StartWarmup(instance_id=instance))
    assert runner.server is not None and runner.server.process is not None
    runner.server.process.kill()
    runner.server.process.wait(timeout=10)
    with pytest.raises(RuntimeError, match="not running"):
        runner.handle_task(VideoGeneration(command_id=CommandId("cmd-dead"), instance_id=instance, task_params=_params(FL2VA_ID, steps=1), owner_node=NodeId("n")))
    assert runner.server is None


def test_bootstrap_routes_the_comfy_backend_to_the_runner() -> None:
    from skulk.shared.backends import engine_of

    assert engine_of("comfy-cuda") == "comfy" and engine_of("comfy") == "comfy"
    assert Runner.__module__ == "skulk.worker.runner.comfy.runner"
