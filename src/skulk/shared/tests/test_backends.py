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
    resolve_node_engine,
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


def _fake_llama_cpp(gpu_offload: bool | None) -> object:
    """A stand-in llama_cpp module whose llama_supports_gpu_offload is controllable.

    ``gpu_offload=None`` omits the symbol so the introspection call raises and the
    probe treats the build as unverifiable.
    """
    from types import SimpleNamespace

    if gpu_offload is None:
        return SimpleNamespace()

    def _supports() -> bool:
        return gpu_offload

    return SimpleNamespace(llama_supports_gpu_offload=_supports)


def test_llama_cpp_probe_keeps_gpu_when_build_supports_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "llama_cpp", _fake_llama_cpp(gpu_offload=True))
    monkeypatch.setenv(LLAMA_CPP_BACKENDS_ENV, "vulkan")
    tags = backends._probe_llama_cpp_backends()
    assert "llama_cpp-vulkan" in tags  # GPU build verified -> advertised


def test_llama_cpp_probe_drops_gpu_when_build_is_cpu_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The clobber case: env declares vulkan but the wheel has no GPU offload
    # compiled in (e.g. uv sync restored the CPU PyPI wheel). The node must NOT
    # advertise vulkan, only cpu, so GPU GGUF work is not routed here.
    monkeypatch.setitem(sys.modules, "llama_cpp", _fake_llama_cpp(gpu_offload=False))
    monkeypatch.setenv(LLAMA_CPP_BACKENDS_ENV, "vulkan, rocm")
    tags = backends._probe_llama_cpp_backends()
    assert "llama_cpp-vulkan" not in tags
    assert "llama_cpp-rocm" not in tags
    assert tags == frozenset({"llama_cpp", "llama_cpp-cpu"})


def test_llama_cpp_probe_trusts_declaration_when_unverifiable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Introspection unavailable -> trust the operator's declaration rather than
    # punish a possibly-working GPU node for a binding quirk.
    monkeypatch.setitem(sys.modules, "llama_cpp", _fake_llama_cpp(gpu_offload=None))
    monkeypatch.setenv(LLAMA_CPP_BACKENDS_ENV, "vulkan")
    tags = backends._probe_llama_cpp_backends()
    assert "llama_cpp-vulkan" in tags


def test_resolve_node_engine_existing_mlx_cards_unchanged() -> None:
    # An original {"mlx"} card on a Mac node ({"mlx","mlx-metal"}) -> mlx.
    engine = resolve_node_engine(
        frozenset({"mlx"}), (), frozenset({"mlx", "mlx-metal"})
    )
    assert engine == "mlx"


def test_resolve_node_engine_picks_llama_cpp() -> None:
    engine = resolve_node_engine(
        frozenset({"llama_cpp-vulkan", "llama_cpp-rocm", "llama_cpp-cpu"}),
        ("llama_cpp-vulkan", "llama_cpp-rocm"),
        frozenset({"llama_cpp", "llama_cpp-vulkan"}),
    )
    assert engine == "llama_cpp"


def test_resolve_node_engine_none_when_no_intersection() -> None:
    # Node advertises only mlx but the card requires llama_cpp -> no match
    # (placement would have excluded this node; caller falls back to default).
    engine = resolve_node_engine(
        frozenset({"llama_cpp-vulkan"}), (), frozenset({"mlx", "mlx-metal"})
    )
    assert engine is None
