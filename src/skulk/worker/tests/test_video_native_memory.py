"""Native workspace admission, bounded metadata reads and real pressure cleanup."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest
from pydantic import TypeAdapter

from skulk.shared.models.model_cards import ModelCard, RuntimeCapabilityCardConfig
from skulk.shared.tests.model_card_fixtures import load_fixture_card
from skulk.shared.types.common import ModelId
from skulk.shared.types.profiling import MemoryUsage
from skulk.shared.types.video import VideoGenerationTaskParams
from skulk.worker.runner.mlx_video import process as native_process
from skulk.worker.runner.mlx_video.memory import (
    NativeMemoryPlan,
    NativeWeightProfile,
    plan_native_memory,
    read_native_image_pixels,
    read_native_weight_profile,
)
from skulk.worker.runner.mlx_video.plan import (
    NativeImageAnchor,
    NativeVideoJob,
    plan_native_job,
)
from skulk.worker.runner.mlx_video.process import run_native_job, serialize_native_job


def _card() -> ModelCard:
    card = asyncio.run(load_fixture_card(ModelId("dgrauet/LTX-2.5-distilled-mlx-q4")))
    assert card is not None
    return card


def _job(root: Path, *, size: str = "512x512", seconds: int = 2) -> NativeVideoJob:
    card = _card()
    job, _ = plan_native_job(VideoGenerationTaskParams(model=str(card.model_id), prompt="A fox walks through grass", seconds=seconds, size=size), card, root / "model", root / "output", root / "input")
    return job


def _memory(total_gib: int = 24, available_gib: int = 22) -> MemoryUsage:
    return MemoryUsage.from_bytes(ram_total=total_gib * 1024**3, ram_available=available_gib * 1024**3, swap_total=0, swap_available=0)


def _weights() -> NativeWeightProfile:
    # Structural q4 header sizes, with additional conservative sampling weight
    # margin; no timing or hardware-class benchmark is needed for this plan.
    return NativeWeightProfile(encoder_bytes=14_595_241_056, sampling_bytes=4_200_000_000, decoder_bytes=1_179_005_524)


def test_native_request_fits_staged_reservation_and_keeps_default_decode_budget(tmp_path: Path) -> None:
    plan = plan_native_memory(_job(tmp_path), _card(), _weights(), _memory())
    assert plan.estimated_peak_bytes <= plan.reservation_bytes < 18 * 1024**3
    assert plan.decode_budget_bytes == 12 * 1024**3
    with pytest.raises(ValueError, match="currently usable"):
        plan_native_memory(_job(tmp_path), _card(), _weights(), _memory(16, 14))


def test_larger_request_can_reserve_more_memory_without_hardware_allowlist(tmp_path: Path) -> None:
    job = _job(tmp_path, size="1024x1024", seconds=3)
    with pytest.raises(ValueError, match="increase the video memory reservation"):
        plan_native_memory(job, _card(), _weights(), _memory(160, 150))
    configured = _card().model_copy(update={"runtime": RuntimeCapabilityCardConfig(video_memory_budget_bytes=96 * 1024**3)})
    plan = plan_native_memory(job, configured, _weights(), _memory(160, 150))
    assert plan.estimated_peak_bytes <= plan.reservation_bytes == 96 * 1024**3


def test_no_fitting_conv_tile_is_refused_before_unsafe_upstream_fallback(tmp_path: Path) -> None:
    weights = _weights().model_copy(update={"decoder_bytes": 14_500_000_000})
    with pytest.raises(ValueError, match="smallest native video decode tile"):
        plan_native_memory(_job(tmp_path, size="128x128"), _card(), weights, _memory())


def _write_weights(path: Path, sizes: dict[str, int], *, metadata: object = None) -> None:
    offset = 0
    header: dict[str, object] = {"__metadata__": metadata}
    for name, size in sizes.items():
        header[name] = {"dtype": "U8", "shape": [size], "data_offsets": [offset, offset + size]}
        offset += size
    encoded = json.dumps(header).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + bytes(offset))


def _pack(root: Path) -> None:
    for name, size in {"text_encoder": 16, "connector": 8, "vae_encoder_conv": 7, "vae_decoder_conv": 5, "audio_vae": 4, "vocoder": 6, "spatial_upscaler_x2_v1_0": 9}.items():
        _write_weights(root / f"{name}.safetensors", {"weight": size})
    _write_weights(root / "transformer-distilled.safetensors", {"input.weight": 4, "transformer_blocks.0.weight": 8, "transformer_blocks.1.weight": 12})


def test_selected_headers_account_non_block_weights_and_two_largest_blocks(tmp_path: Path) -> None:
    job = _job(tmp_path)
    _pack(Path(job.model_dir))
    profile = read_native_weight_profile(job)
    assert (profile.encoder_bytes, profile.sampling_bytes, profile.decoder_bytes) == (24, 59, 15)


@pytest.mark.parametrize("metadata", [None, {"format": "test"}, "converted-pack-note"])
def test_unused_header_metadata_does_not_change_weight_accounting(tmp_path: Path, metadata: object) -> None:
    job = _job(tmp_path)
    _pack(Path(job.model_dir))
    _write_weights(Path(job.model_dir) / "text_encoder.safetensors", {"weight": 16}, metadata=metadata)
    assert read_native_weight_profile(job).encoder_bytes == 24


def test_silent_output_still_reserves_joint_audio_sampling(tmp_path: Path) -> None:
    job = _job(tmp_path, size="1024x1024", seconds=3)
    card = _card().model_copy(update={"runtime": RuntimeCapabilityCardConfig(video_memory_budget_bytes=96 * 1024**3)})
    audible = plan_native_memory(job, card, _weights(), _memory(160, 150))
    silent = plan_native_memory(job.model_copy(update={"audio": False}), card, _weights(), _memory(160, 150))
    assert silent.estimated_peak_bytes == audible.estimated_peak_bytes


def test_generated_keyframe_planes_participate_in_request_admission(tmp_path: Path) -> None:
    """Internal keyframes add transformer tokens even when output geometry stays fixed."""
    job = _job(tmp_path, seconds=5)
    ordinary = plan_native_memory(job, _card(), _weights(), _memory())
    assert ordinary.estimated_peak_bytes <= ordinary.reservation_bytes
    with pytest.raises(ValueError, match="increase the video memory reservation"):
        plan_native_memory(job.model_copy(update={"generated_keyframe_interval": 8}), _card(), _weights(), _memory())


def test_native_guided_strength_one_requires_the_actual_stage_two_checkpoint(tmp_path: Path) -> None:
    """A LoRA cannot stand in for the pre-fused checkpoint used by the pinned streamer."""
    job = _job(tmp_path).model_copy(update={"pipeline": "dev_two_stage"})
    _pack(Path(job.model_dir))
    (Path(job.model_dir) / "transformer-distilled.safetensors").rename(Path(job.model_dir) / "transformer-dev.safetensors")
    with pytest.raises(ValueError, match="missing transformer-distilled"):
        read_native_weight_profile(job)


def test_full_source_image_dimensions_are_charged_before_resizing(tmp_path: Path) -> None:
    """A small output canvas must not hide much larger decoded reference buffers."""
    path = tmp_path / "large-image.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + (13).to_bytes(4, "big") + b"IHDR" + (12000).to_bytes(4, "big") * 2)
    anchors = tuple(NativeImageAnchor(path=str(path), frame=index) for index in range(8))
    job = _job(tmp_path).model_copy(update={"images": anchors})
    image_pixels = read_native_image_pixels(job)
    assert image_pixels == (12000**2,) * 8
    with pytest.raises(ValueError, match="increase the video memory reservation"):
        plan_native_memory(job, _card(), _weights(), _memory(), image_pixels=image_pixels)


def test_unknown_source_shape_cannot_use_an_assumed_small_canvas(tmp_path: Path) -> None:
    """Missing image facts fail closed before allocating native model tensors."""
    job = _job(tmp_path).model_copy(update={"images": (NativeImageAnchor(path=str(tmp_path / "missing.png"), frame=0),)})
    with pytest.raises(ValueError, match="source image dimensions"):
        read_native_image_pixels(job)
    with pytest.raises(ValueError, match="verified source image dimensions"):
        plan_native_memory(job, _card(), _weights(), _memory())


@pytest.mark.parametrize("corruption", ["oversized", "truncated", "out_of_file"])
def test_invalid_headers_fail_before_loading_tensor_payloads(tmp_path: Path, corruption: str) -> None:
    job = _job(tmp_path)
    _pack(Path(job.model_dir))
    path = Path(job.model_dir) / "transformer-distilled.safetensors"
    if corruption == "oversized":
        path.write_bytes((16 * 1024**2 + 1).to_bytes(8, "little"))
    elif corruption == "truncated":
        path.write_bytes((64).to_bytes(8, "little") + b"{}")
    else:
        path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="header|exceeds its file"):
        read_native_weight_profile(job)


def _fake_interpreter(root: Path, *, peak: int) -> Path:
    path = root / "interpreter"
    path.write_text(f"#!{sys.executable}\nimport json,os,sys\nfrom pathlib import Path\njob=json.loads(Path(sys.argv[sys.argv.index('--job')+1]).read_text())\noutput=Path(job['output_dir'])\n(output/'environment.json').write_text(json.dumps(dict(os.environ)))\n(output/'received-job.json').write_text(json.dumps(job))\n(output/'result.json').write_text(json.dumps({{'steps':11,'sampling_seconds':1.0,'total_generation_time':2.0,'peak_memory_bytes':{peak}}}))\n")
    path.chmod(0o755)
    return path


def test_owned_decode_budget_overrides_inherited_tuning_without_changing_job_protocol(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LTX2_VAE_DECODE_BUDGET_GB", "9999")
    monkeypatch.setenv("LTX2_OTHER_UNTRUSTED_TUNING", "1")
    job = _job(tmp_path)
    plan = NativeMemoryPlan(reservation_bytes=1024, estimated_peak_bytes=512, decode_budget_bytes=256)
    result = run_native_job(_fake_interpreter(tmp_path, peak=100), job, progress=lambda _record: None, is_cancelled=lambda: False, memory_plan=plan)
    assert result is not None and result.peak_memory_bytes == 100
    environment = TypeAdapter(dict[str, str]).validate_json((Path(job.output_dir) / "environment.json").read_bytes())
    assert environment["LTX2_VAE_DECODE_BUDGET_GB"] == str(256 / 1024**3)
    assert "LTX2_OTHER_UNTRUSTED_TUNING" not in environment
    assert json.loads((Path(job.output_dir) / "received-job.json").read_text()) == json.loads(serialize_native_job(job))


def test_measured_peak_over_reservation_is_a_failed_job(tmp_path: Path) -> None:
    plan = NativeMemoryPlan(reservation_bytes=1024, estimated_peak_bytes=512, decode_budget_bytes=256)
    with pytest.raises(ValueError, match="reserved Metal working set"):
        run_native_job(_fake_interpreter(tmp_path, peak=4096), _job(tmp_path), progress=lambda _record: None, is_cancelled=lambda: False, memory_plan=plan)


def test_live_pressure_refusal_reaps_native_child_group(tmp_path: Path) -> None:
    """Pressure failure keeps its resource reason after stopping every encoder."""
    interpreter = tmp_path / "interpreter"
    child_file = tmp_path / "child.pid"
    interpreter.write_text(f"#!{sys.executable}\nimport subprocess,time\nfrom pathlib import Path\nchild=subprocess.Popen([{sys.executable!r},'-c','import time;time.sleep(60)'])\nPath({str(child_file)!r}).write_text(str(child.pid))\ntime.sleep(60)\n")
    interpreter.chmod(0o755)
    job = _job(tmp_path)
    with pytest.raises(ValueError, match="controlled pressure refusal"):
        run_native_job(interpreter, job, progress=lambda _record: None, is_cancelled=lambda: False, resource_guard=lambda _pid: "controlled pressure refusal" if child_file.exists() else None)
    _assert_child_stopped(int(child_file.read_text()))
    assert not (Path(job.output_dir) / "job.json").exists()


def _assert_child_stopped(identifier: int) -> None:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        try:
            if psutil.Process(identifier).status() == psutil.STATUS_ZOMBIE:
                return
        except psutil.NoSuchProcess:
            return
        time.sleep(0.025)
    pytest.fail("native render left a live encoder child")


def test_successful_wrapper_retains_its_pid_until_encoder_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An exited wrapper cannot leave a TERM-resistant encoder or release its PID early."""
    interpreter = _fake_interpreter(tmp_path, peak=100)
    body = interpreter.read_text().split("\n", 1)[1]
    child_file = tmp_path / "encoder.pid"
    child = f"import os,signal,time;from pathlib import Path;signal.signal(signal.SIGTERM,signal.SIG_IGN);Path({str(child_file)!r}).write_text(str(os.getpid()));time.sleep(60)"
    interpreter.write_text(f"#!{sys.executable}\nimport subprocess,time\nfrom pathlib import Path\nsubprocess.Popen([{sys.executable!r},'-c',{child!r}])\nwhile not Path({str(child_file)!r}).exists():time.sleep(.01)\n" + body)
    original = native_process.terminate_process_group
    observed: list[int] = []

    def cleanup(process: subprocess.Popen[str], *, grace_seconds: float = 1.0) -> None:
        assert process.returncode is None
        assert native_process.observe_process_exit(process) is not None
        observed.append(process.pid)
        original(process, grace_seconds=grace_seconds)

    monkeypatch.setattr(native_process, "terminate_process_group", cleanup)
    job = _job(tmp_path)
    result = run_native_job(interpreter, job, progress=lambda _record: None, is_cancelled=lambda: False)
    assert result is not None and result.steps == 11 and len(observed) == 1
    _assert_child_stopped(int(child_file.read_text()))
    assert not (Path(job.output_dir) / "job.json").exists()


def test_native_request_is_removed_when_group_cleanup_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A propagated cleanup error must not retain prompts or reference paths."""
    original = native_process.terminate_process_group

    def cleanup(process: subprocess.Popen[str], *, grace_seconds: float = 1.0) -> None:
        original(process, grace_seconds=grace_seconds)
        raise RuntimeError("controlled cleanup failure")

    monkeypatch.setattr(native_process, "terminate_process_group", cleanup)
    job = _job(tmp_path)
    with pytest.raises(RuntimeError, match="controlled cleanup failure"):
        run_native_job(_fake_interpreter(tmp_path, peak=100), job, progress=lambda _record: None, is_cancelled=lambda: False)
    assert not (Path(job.output_dir) / "job.json").exists()
