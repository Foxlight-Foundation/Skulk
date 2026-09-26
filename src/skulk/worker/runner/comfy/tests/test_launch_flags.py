"""The ComfyUI launch line varies only by the stamped compute backend."""

from __future__ import annotations

from pathlib import Path

import pytest

import skulk.shared.backends as backends
from skulk.worker.runner.comfy.runner import (
    CUDA_LAUNCH_FLAGS,
    ROCM_LAUNCH_FLAGS,
    ROCM_MMAP_CEILING_BYTES,
    launch_flags,
)

_GIB = 1024**3
_STRIX_RAM = 64 * _GIB
_STRIX_FLAGS = (*ROCM_LAUNCH_FLAGS, "--cache-ram", "25")
from skulk.worker.runner.comfy.server import server_environment


def test_rocm_backends_get_the_validated_strix_flags() -> None:
    assert ROCM_LAUNCH_FLAGS == ("--bf16-vae",)
    assert launch_flags("comfy-rocm", total_ram_bytes=_STRIX_RAM) == _STRIX_FLAGS


def test_rocm_keeps_models_resident_between_renders() -> None:
    """Rebuilding every model per prompt doubled a warm render on gfx1151."""
    assert "--cache-none" not in launch_flags("comfy-rocm", total_ram_bytes=_STRIX_RAM)


def test_rocm_cache_headroom_scales_with_host_ram() -> None:
    """40% of host RAM stays free: 24 GiB on the 61.4 GiB host it was measured on."""
    def headroom(total: int) -> str:
        flags = launch_flags("comfy-rocm", total_ram_bytes=total)
        return flags[flags.index("--cache-ram") + 1]

    assert headroom(int(61.4 * _GIB)) == "24"
    assert headroom(128 * _GIB) == "51"
    assert headroom(1 * _GIB) == "1"


def test_rocm_maps_weights_up_to_the_ceiling(tmp_path: Path) -> None:
    """The pruned H3 files map; mapping stays off only past the 64 GB ceiling."""
    small = tmp_path / "transformer.safetensors"
    small.write_bytes(b"x")
    assert launch_flags("comfy-rocm", [small], total_ram_bytes=_STRIX_RAM) == _STRIX_FLAGS


def test_rocm_disables_mmap_for_a_file_past_the_ceiling(tmp_path: Path) -> None:
    """A sparse file stands in for an unpruned checkpoint larger than 64 GB."""
    small = tmp_path / "vae.safetensors"
    small.write_bytes(b"x")
    large = tmp_path / "transformer.safetensors"
    with large.open("wb") as handle:
        handle.truncate(ROCM_MMAP_CEILING_BYTES + 1)
    assert launch_flags("comfy-rocm", [small, large], total_ram_bytes=_STRIX_RAM) == (
        *_STRIX_FLAGS,
        "--disable-mmap",
    )


def test_the_mmap_ceiling_never_touches_the_cuda_lane(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backends, "probe_node_backends", lambda: {"comfy", "comfy-cuda"})
    large = tmp_path / "transformer.safetensors"
    with large.open("wb") as handle:
        handle.truncate(ROCM_MMAP_CEILING_BYTES + 1)
    assert launch_flags("comfy-cuda", [large]) == CUDA_LAUNCH_FLAGS


def test_cuda_backends_launch_on_the_native_allocator(monkeypatch: pytest.MonkeyPatch) -> None:
    """The async allocator aborts ControlNet renders on the GB10; CUDA uses PyTorch's own."""
    monkeypatch.setattr(backends, "probe_node_backends", lambda: {"comfy", "comfy-cuda"})
    assert CUDA_LAUNCH_FLAGS == ("--disable-cuda-malloc",)
    assert launch_flags("comfy-cuda") == CUDA_LAUNCH_FLAGS
    assert launch_flags("comfy") == CUDA_LAUNCH_FLAGS
    assert launch_flags(None) == CUDA_LAUNCH_FLAGS


def test_unstamped_shards_use_the_tag_this_node_advertises(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unstamped shard on a ROCm node still launches with the qualified flags."""
    monkeypatch.setattr(backends, "probe_node_backends", lambda: {"comfy", "comfy-rocm", "llama_server-vulkan"})
    assert launch_flags(None, total_ram_bytes=_STRIX_RAM) == _STRIX_FLAGS
    monkeypatch.setattr(backends, "probe_node_backends", lambda: {"llama_server-vulkan"})
    assert launch_flags(None) == CUDA_LAUNCH_FLAGS


def test_a_bare_engine_stamp_also_uses_the_tag_this_node_advertises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A card declaring only ``comfy`` still gets the ROCm flags on a ROCm node."""
    monkeypatch.setattr(backends, "probe_node_backends", lambda: {"comfy", "comfy-rocm"})
    assert launch_flags("comfy", total_ram_bytes=_STRIX_RAM) == _STRIX_FLAGS
    monkeypatch.setattr(backends, "probe_node_backends", lambda: {"comfy", "comfy-cuda"})
    assert launch_flags("comfy") == CUDA_LAUNCH_FLAGS


def test_server_environment_leads_with_the_interpreter_and_drops_pythonpath(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The interpreter's bin dir leads PATH and the Skulk PYTHONPATH never leaks."""
    monkeypatch.setenv("PYTHONPATH", "/skulk/src")
    monkeypatch.setenv("PATH", "/usr/bin")
    interpreter = tmp_path / "venv" / "bin" / "python"
    env = server_environment(interpreter)
    assert "PYTHONPATH" not in env
    assert env["PATH"].startswith(str(interpreter.parent))
