"""The ComfyUI launch line varies only by the stamped compute backend."""

from __future__ import annotations

from pathlib import Path

import pytest

import skulk.shared.backends as backends
from skulk.worker.runner.comfy.runner import (
    ROCM_LAUNCH_ENVIRONMENT,
    ROCM_LAUNCH_FLAGS,
    launch_environment,
    launch_flags,
)
from skulk.worker.runner.comfy.server import server_environment


def test_rocm_backends_get_the_validated_strix_flags() -> None:
    assert launch_flags("comfy-rocm") == ROCM_LAUNCH_FLAGS
    assert ROCM_LAUNCH_FLAGS == ("--bf16-vae", "--disable-mmap", "--cache-none")


def test_other_backends_launch_with_the_headless_defaults_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backends, "probe_node_backends", lambda: {"comfy", "comfy-cuda"})
    assert launch_flags("comfy-cuda") == ()
    assert launch_flags("comfy") == ()
    assert launch_flags(None) == ()


def test_unstamped_shards_use_the_tag_this_node_advertises(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unstamped shard on a ROCm node still launches with the qualified flags."""
    monkeypatch.setattr(backends, "probe_node_backends", lambda: {"comfy", "comfy-rocm", "llama_server-vulkan"})
    assert launch_flags(None) == ROCM_LAUNCH_FLAGS
    monkeypatch.setattr(backends, "probe_node_backends", lambda: {"llama_server-vulkan"})
    assert launch_flags(None) == ()


def test_a_bare_engine_stamp_also_uses_the_tag_this_node_advertises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A card declaring only ``comfy`` still gets the ROCm flags on a ROCm node."""
    monkeypatch.setattr(backends, "probe_node_backends", lambda: {"comfy", "comfy-rocm"})
    assert launch_flags("comfy") == ROCM_LAUNCH_FLAGS
    monkeypatch.setattr(backends, "probe_node_backends", lambda: {"comfy", "comfy-cuda"})
    assert launch_flags("comfy") == ()


def test_rocm_lane_server_environment_prefers_hipblaslt(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ROCm lane routes torch GEMMs through hipBLASLt; other lanes add nothing."""
    assert launch_environment("comfy-rocm") == {"TORCH_BLAS_PREFER_HIPBLASLT": "1"}
    assert launch_environment("comfy-rocm") == ROCM_LAUNCH_ENVIRONMENT
    assert launch_environment("comfy-cuda") == {}
    monkeypatch.setattr(backends, "probe_node_backends", lambda: {"comfy", "comfy-rocm"})
    assert launch_environment(None) == ROCM_LAUNCH_ENVIRONMENT
    assert launch_environment("comfy") == ROCM_LAUNCH_ENVIRONMENT


def test_server_environment_layers_lane_variables_over_the_process(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Lane variables win, the interpreter's bin dir leads PATH, PYTHONPATH never leaks."""
    monkeypatch.setenv("PYTHONPATH", "/skulk/src")
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("TORCH_BLAS_PREFER_HIPBLASLT", "0")
    interpreter = tmp_path / "venv" / "bin" / "python"
    env = server_environment(interpreter, {"TORCH_BLAS_PREFER_HIPBLASLT": "1"})
    assert "PYTHONPATH" not in env
    assert env["PATH"].startswith(str(interpreter.parent))
    assert env["TORCH_BLAS_PREFER_HIPBLASLT"] == "1"
