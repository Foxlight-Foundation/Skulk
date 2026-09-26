"""Audio.cpp advertises only a pinned, probed, ready executable."""

import hashlib
import json
from pathlib import Path

import pytest

from skulk.facts.derive import derive_node_backends
from skulk.facts.inventory import engine_build_inventory
from skulk.facts.probe import gather_node_facts, probe_audio_cpp
from skulk.provisioning.audio_cpp import AUDIO_CPP_SOURCE_REVISION


def _fake_server(
    path: Path,
    *,
    revision: str = AUDIO_CPP_SOURCE_REVISION,
    backends: str = "cpu,metal",
    devices: str = "MTL:0 Apple GPU\nCPU:0 Apple CPU",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    specs = path.parent.parent / "model_specs"
    specs.mkdir(parents=True, exist_ok=True)
    source = (
        Path(__file__).resolve().parents[4]
        / "packaging/skulk-audio-cpp-cpu/src/skulk_audio_cpp_cpu/model_specs"
    )
    for item in source.glob("*.json"):
        (specs / item.name).write_bytes(item.read_bytes())
    path.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "--version" ]; then\n'
        "  echo 'audio.cpp dev'\n"
        f"  echo 'git: {revision} 2026-09-23'\n"
        f"  echo 'backends: {backends}'\n"
        'elif [ "$1" = "--list-devices" ]; then\n'
        + "".join(f"  echo '{line}'\n" for line in devices.splitlines())
        + "fi\n"
    )
    path.chmod(0o755)
    return path


def test_installed_package_is_not_ready_until_selected(tmp_path: Path) -> None:
    """An installable wheel alone cannot expand node placement."""
    facts = gather_node_facts(env={}, platform="darwin", drm_root=tmp_path)
    assert not any(
        tag.startswith("audio_cpp") for tag in derive_node_backends(facts).backends
    )


def test_missing_cuda_libraries_have_actionable_probe_diagnostic(tmp_path: Path) -> None:
    """A missing CUDA loader library keeps the engine unavailable with a fix hint."""
    binary = tmp_path / "audiocpp_server"
    binary.write_text(
        "#!/bin/sh\n"
        "echo 'error while loading shared libraries: libcublas.so.12: "
        "cannot open shared object file' >&2\n"
        "exit 127\n"
    )
    binary.chmod(0o755)
    probe = probe_audio_cpp(str(binary))
    assert probe.outcome == "failed"
    assert "install the host CUDA 12 runtime, cuBLAS, and NCCL" in (probe.detail or "")


def test_pinned_binary_probe_and_build_inventory(tmp_path: Path) -> None:
    """Ready tags and exact build identity come from the same executable."""
    binary = _fake_server(tmp_path / "bin" / "audiocpp_server")
    facts = gather_node_facts(
        env={"SKULK_AUDIO_CPP_BIN": str(binary)},
        platform="darwin",
        drm_root=tmp_path,
    )
    assert facts.audio_cpp_probe.outcome == "ready"
    assert facts.audio_cpp_probe.computes == ("cpu", "metal")
    backends = derive_node_backends(facts).backends
    assert {"audio_cpp", "audio_cpp-cpu", "audio_cpp-metal"} <= backends
    expected = "audio.cpp@sha256:" + hashlib.sha256(binary.read_bytes()).hexdigest()
    inventory = engine_build_inventory(backends, facts, environ={})
    assert inventory["audio_cpp-metal"] == expected


def test_cpu_and_vulkan_binaries_advertise_distinct_exact_builds(
    tmp_path: Path,
) -> None:
    """Preparing a Vulkan variant does not relabel an existing CPU mount's build."""
    cpu = _fake_server(
        tmp_path / "cpu/bin/audiocpp_server",
        backends="cpu",
        devices="CPU:0 AMD CPU",
    )
    vulkan = _fake_server(
        tmp_path / "vulkan/bin/audiocpp_server",
        backends="cpu,vulkan",
        devices="VK:0 Radeon 8060S\nCPU:0 AMD CPU",
    )
    facts = gather_node_facts(
        env={
            "SKULK_AUDIO_CPP_BIN": str(cpu),
            "SKULK_AUDIO_CPP_VULKAN_BIN": str(vulkan),
        },
        platform="linux",
        drm_root=tmp_path,
    )
    backends = derive_node_backends(facts).backends
    assert {"audio_cpp-cpu", "audio_cpp-vulkan"} <= backends
    inventory = engine_build_inventory(backends, facts, environ={})
    assert inventory["audio_cpp-cpu"] == (
        "audio.cpp@sha256:" + hashlib.sha256(cpu.read_bytes()).hexdigest()
    )
    assert inventory["audio_cpp-vulkan"] == (
        "audio.cpp@sha256:" + hashlib.sha256(vulkan.read_bytes()).hexdigest()
    )
    assert inventory["audio_cpp-cpu"] != inventory["audio_cpp-vulkan"]


def test_cuda_binary_needs_a_real_cuda_device_and_has_its_own_build(
    tmp_path: Path,
) -> None:
    """A CUDA package cannot borrow CPU readiness or another executable hash."""
    cpu = _fake_server(
        tmp_path / "cpu/bin/audiocpp_server",
        backends="cpu", devices="CPU:0 Arm CPU",
    )
    cuda = _fake_server(
        tmp_path / "cuda/bin/audiocpp_server",
        backends="cpu,cuda", devices="CUDA:0 NVIDIA GB10\nCPU:0 Arm CPU",
    )
    facts = gather_node_facts(
        env={
            "SKULK_AUDIO_CPP_BIN": str(cpu),
            "SKULK_AUDIO_CPP_CUDA_BIN": str(cuda),
        },
        platform="linux", drm_root=tmp_path,
    )
    backends = derive_node_backends(facts).backends
    assert {"audio_cpp-cpu", "audio_cpp-cuda"} <= backends
    inventory = engine_build_inventory(backends, facts, environ={})
    assert inventory["audio_cpp-cuda"] == (
        "audio.cpp@sha256:" + hashlib.sha256(cuda.read_bytes()).hexdigest()
    )
    assert inventory["audio_cpp-cuda"] != inventory["audio_cpp-cpu"]

    no_device = _fake_server(
        cuda, backends="cpu,cuda", devices="CPU:0 Arm CPU",
    )
    missing = gather_node_facts(
        env={"SKULK_AUDIO_CPP_CUDA_BIN": str(no_device)},
        platform="linux", drm_root=tmp_path,
    )
    assert "audio_cpp-cuda" not in derive_node_backends(missing).backends
    assert any(
        "CUDA" in conflict.message
        for conflict in derive_node_backends(missing).conflicts
    )


def test_missing_model_specs_prevent_ready_advertisement(tmp_path: Path) -> None:
    """A CLI-ready override must also have its pinned runtime specs."""
    binary = _fake_server(tmp_path / "bin" / "audiocpp_server")
    (tmp_path / "model_specs" / "ace_step.json").unlink()
    facts = gather_node_facts(
        env={"SKULK_AUDIO_CPP_BIN": str(binary)},
        platform="darwin",
        drm_root=tmp_path,
    )
    assert facts.audio_cpp_probe.outcome == "failed"
    assert "SKULK_AUDIO_CPP_SPECS_DIR" in (facts.audio_cpp_probe.detail or "")
    assert "audio_cpp" not in derive_node_backends(facts).backends


def test_audio_cpp_inventory_rehashes_and_ignores_declared_build(tmp_path: Path) -> None:
    """A model support claim cannot match a stale or operator-invented digest."""
    binary = _fake_server(tmp_path / "bin" / "audiocpp_server")
    facts = gather_node_facts(
        env={"SKULK_AUDIO_CPP_BIN": str(binary)},
        platform="darwin",
        drm_root=tmp_path,
    )
    backends = derive_node_backends(facts).backends
    declared = {"SKULK_ENGINE_BUILDS": json.dumps({"audio_cpp": "invented"})}
    first = engine_build_inventory(backends, facts, environ=declared)
    assert first["audio_cpp-cpu"] == (
        "audio.cpp@sha256:" + hashlib.sha256(binary.read_bytes()).hexdigest()
    )
    binary.write_bytes(binary.read_bytes() + b"\n# changed\n")
    second = engine_build_inventory(backends, facts, environ=declared)
    assert second["audio_cpp-cpu"] != first["audio_cpp-cpu"]


def test_wrong_source_or_invalid_override_never_advertises_false_lane(
    tmp_path: Path,
) -> None:
    """Neither an unrelated build nor a declaration can fabricate readiness."""
    binary = _fake_server(tmp_path / "bin" / "audiocpp_server", revision="deadbeef")
    bad = gather_node_facts(
        env={"SKULK_AUDIO_CPP_BIN": str(binary)},
        platform="darwin",
        drm_root=tmp_path,
    )
    assert bad.audio_cpp_probe.outcome == "failed"
    assert "audio_cpp" not in derive_node_backends(bad).backends

    _fake_server(binary)
    restricted = gather_node_facts(
        env={
            "SKULK_AUDIO_CPP_BIN": str(binary),
            "SKULK_AUDIO_CPP_BACKENDS": "cuda",
        },
        platform="darwin",
        drm_root=tmp_path,
    )
    derived = derive_node_backends(restricted)
    assert "audio_cpp" not in derived.backends
    assert [conflict.code for conflict in derived.conflicts] == [
        "backend_override_conflict"
    ]


def test_unverified_short_revision_cannot_advertise_engine(tmp_path: Path) -> None:
    """A matching seven-character prefix is insufficient for an override."""
    binary = _fake_server(tmp_path / "bin" / "audiocpp_server", revision="4d88768")
    facts = gather_node_facts(
        env={"SKULK_AUDIO_CPP_BIN": str(binary)},
        platform="darwin",
        drm_root=tmp_path,
    )
    assert facts.audio_cpp_probe.outcome == "failed"


def test_verified_wheel_may_report_upstream_short_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The wheel digest establishes exact provenance when upstream abbreviates it."""
    import skulk.provisioning.audio_cpp as audio_cpp

    binary = _fake_server(tmp_path / "bin" / "audiocpp_server", revision="4d88768")
    def verified_wheel(_path: Path) -> bool:
        return True

    monkeypatch.setattr(audio_cpp, "verified_cached_audio_cpp_binary", verified_wheel)
    facts = gather_node_facts(
        env={"SKULK_AUDIO_CPP_BIN": str(binary)},
        platform="darwin",
        drm_root=tmp_path,
    )
    assert facts.audio_cpp_probe.outcome == "ready"
