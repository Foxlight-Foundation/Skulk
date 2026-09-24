"""The ComfyUI launch line varies only by the stamped compute backend."""

from __future__ import annotations

from pathlib import Path

import pytest

import skulk.shared.backends as backends
from skulk.worker.runner.comfy.runner import (
    CUDA_LAUNCH_FLAGS,
    ROCM_LAUNCH_FLAGS,
    launch_flags,
)
from skulk.worker.runner.comfy.server import server_environment


def test_rocm_backends_get_the_validated_strix_flags() -> None:
    assert launch_flags("comfy-rocm") == ROCM_LAUNCH_FLAGS
    assert ROCM_LAUNCH_FLAGS == ("--bf16-vae", "--disable-mmap", "--cache-none")


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
    assert launch_flags(None) == ROCM_LAUNCH_FLAGS
    monkeypatch.setattr(backends, "probe_node_backends", lambda: {"llama_server-vulkan"})
    assert launch_flags(None) == CUDA_LAUNCH_FLAGS


def test_a_bare_engine_stamp_also_uses_the_tag_this_node_advertises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A card declaring only ``comfy`` still gets the ROCm flags on a ROCm node."""
    monkeypatch.setattr(backends, "probe_node_backends", lambda: {"comfy", "comfy-rocm"})
    assert launch_flags("comfy") == ROCM_LAUNCH_FLAGS
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
