# pyright: reportPrivateUsage=false
"""Tests for the backend capability tag vocabulary and node probing."""

import sys

import pytest

from skulk.shared import backends
from skulk.shared.backends import (
    LLAMA_CPP_BACKENDS_ENV,
    engine_of,
    make_backend_tag,
    probe_node_backends,
)


def test_make_backend_tag_is_compound() -> None:
    assert make_backend_tag("mlx", "metal") == "mlx-metal"
    assert make_backend_tag("llama_cpp", "vulkan") == "llama_cpp-vulkan"


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        ("mlx", "mlx"),
        ("mlx-metal", "mlx"),
        ("llama_cpp", "llama_cpp"),
        ("llama_cpp-vulkan", "llama_cpp"),
        ("llama_cpp-rocm", "llama_cpp"),
        ("cuda", None),  # bare compute, no engine
        ("vllm-cuda", None),  # unknown engine
        ("", None),
    ],
)
def test_engine_of(tag: str, expected: str | None) -> None:
    assert engine_of(tag) == expected


def test_probe_includes_mlx_on_darwin() -> None:
    tags = probe_node_backends()
    if sys.platform == "darwin":
        # Bare engine tag kept for back-compat with original {"mlx"} cards.
        assert "mlx" in tags
        assert "mlx-metal" in tags
    else:
        assert "mlx" not in tags


def test_llama_cpp_probe_empty_without_binding(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force the probe import to fail regardless of what is installed locally.
    monkeypatch.setitem(sys.modules, "llama_cpp", None)
    assert backends._probe_llama_cpp_backends() == frozenset()


def test_llama_cpp_probe_reads_declared_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Stand in a dummy module so the import succeeds without the real binding.
    monkeypatch.setitem(sys.modules, "llama_cpp", object())
    monkeypatch.setenv(LLAMA_CPP_BACKENDS_ENV, "vulkan, rocm , bogus, metal")
    tags = backends._probe_llama_cpp_backends()
    assert "llama_cpp" in tags  # bare engine tag
    assert "llama_cpp-vulkan" in tags
    assert "llama_cpp-rocm" in tags
    assert "llama_cpp-bogus" not in tags  # not a known compute backend
    assert "llama_cpp-metal" not in tags  # metal is MLX-only, ignored for llama.cpp


def test_llama_cpp_probe_defaults_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "llama_cpp", object())
    monkeypatch.delenv(LLAMA_CPP_BACKENDS_ENV, raising=False)
    tags = backends._probe_llama_cpp_backends()
    # Without an operator declaration we claim only CPU, never over-claim GPU.
    assert tags == frozenset({"llama_cpp", "llama_cpp-cpu"})
