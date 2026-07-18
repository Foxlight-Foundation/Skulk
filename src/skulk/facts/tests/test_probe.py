# pyright: reportPrivateUsage=false
"""Gathering-side tests: env classification, device-list parsing, injection."""

from pathlib import Path

import pytest

from skulk.facts import probe
from skulk.facts.probe import gather_node_facts, parse_list_devices_output
from skulk.shared.types.node_facts import LlamaServerDeviceProbe


def _make_executable(path: Path) -> Path:
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755)
    return path


# --- --list-devices parsing ----------------------------------------------


def test_parse_cuda_device_list() -> None:
    output = (
        "Available devices:\n"
        "  CUDA0: NVIDIA A40 (46068 MiB, 45403 MiB free)\n"
    )
    assert parse_list_devices_output(output) == ("cuda",)


def test_parse_vulkan_device_list() -> None:
    output = (
        "Available devices:\n"
        "  Vulkan0: AMD Radeon Graphics (RADV GFX1151) (65536 MiB, 64000 MiB free)\n"
    )
    assert parse_list_devices_output(output) == ("vulkan",)


def test_parse_dedupes_and_preserves_order() -> None:
    output = (
        "Available devices:\n"
        "  ROCm0: AMD Instinct MI300X (196608 MiB, 190000 MiB free)\n"
        "  ROCm1: AMD Instinct MI300X (196608 MiB, 190000 MiB free)\n"
        "  Vulkan0: AMD Radeon Graphics (8192 MiB, 8000 MiB free)\n"
    )
    assert parse_list_devices_output(output) == ("rocm", "vulkan")


def test_parse_cpu_only_build_reports_nothing() -> None:
    assert parse_list_devices_output("Available devices:\n") == ()


def test_parse_ignores_metal_and_unknown_prefixes() -> None:
    output = "  Metal0: Apple M4 (unified)\n  Widget0: nonsense\n"
    assert parse_list_devices_output(output) == ()


# --- binary classification -----------------------------------------------


def test_binary_fact_states(tmp_path: Path) -> None:
    binary = _make_executable(tmp_path / "llama-server")
    non_exec = tmp_path / "flat-file"
    non_exec.write_text("data")

    facts = gather_node_facts(
        env={"SKULK_LLAMA_SERVER_BIN": str(binary), "SKULK_VLLM_BIN": str(non_exec)},
        platform="linux",
        nvidia_presence=False,
        drm_root=tmp_path / "no-drm",
        # A synthetic device probe is irrelevant here; the real probe would
        # run the tmp shell script, so keep the declaration chain empty and
        # accept whatever it reports -- assertions below only cover states.
    )
    assert facts.llama_server_binary.state == "ok"
    assert facts.vllm_binary.state == "not_executable"
    assert facts.rpc_server_binary.state == "not_configured"


def test_binary_fact_missing_path(tmp_path: Path) -> None:
    facts = gather_node_facts(
        env={"SKULK_LLAMA_SERVER_BIN": str(tmp_path / "nope")},
        platform="linux",
        nvidia_presence=False,
        drm_root=tmp_path / "no-drm",
    )
    assert facts.llama_server_binary.state == "missing"
    # No usable binary => the device probe never ran.
    assert facts.llama_server_device_probe.outcome == "not_run"


# --- gathering ------------------------------------------------------------


def test_darwin_gathers_apple_gpu(tmp_path: Path) -> None:
    facts = gather_node_facts(env={}, platform="darwin", drm_root=tmp_path / "no-drm")
    assert [gpu.vendor for gpu in facts.gpus] == ["apple"]
    assert facts.gpus[0].detection_source == "apple_platform"


def test_nvidia_presence_only_yields_device_node_fact(tmp_path: Path) -> None:
    # NVML unavailable but the device is visibly present: the #612 shape.
    facts = gather_node_facts(
        env={},
        platform="linux",
        nvidia_presence=True,
        drm_root=tmp_path / "no-drm",
    )
    nvidia = facts.gpus_of("nvidia")
    assert len(nvidia) == 1
    assert nvidia[0].detection_source == "nvidia_device_node"
    assert nvidia[0].vram_total_bytes is None


def test_amd_sysfs_devices_gathered(tmp_path: Path) -> None:
    device = tmp_path / "card0" / "device"
    device.mkdir(parents=True)
    (device / "gpu_busy_percent").write_text("3\n")
    (device / "mem_info_vram_total").write_text(str(8 * 2**30))
    (device / "mem_info_gtt_total").write_text(str(120 * 2**30))
    facts = gather_node_facts(
        env={}, platform="linux", nvidia_presence=False, drm_root=tmp_path
    )
    amd = facts.gpus_of("amd")
    assert len(amd) == 1
    assert amd[0].vram_total_bytes == 8 * 2**30
    assert amd[0].gtt_total_bytes == 120 * 2**30


def test_declarations_recorded_verbatim(tmp_path: Path) -> None:
    facts = gather_node_facts(
        env={
            "SKULK_LLAMA_CPP_BACKENDS": "vulkan, rocm",
            "SKULK_LLAMA_SERVER_BACKENDS": "cuda",
        },
        platform="linux",
        nvidia_presence=False,
        drm_root=tmp_path / "no-drm",
    )
    assert facts.declared_llama_cpp_backends == "vulkan, rocm"
    assert facts.declared_llama_server_backends == "cuda"
    assert facts.declared_vllm_backends is None


def test_device_probe_skipped_when_declaration_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A declaration overrides derivation, so the subprocess probe must not run.
    binary = _make_executable(tmp_path / "llama-server")

    def _must_not_run(path: str) -> LlamaServerDeviceProbe:
        raise AssertionError("device probe ran despite a declaration")

    monkeypatch.setattr(probe, "probe_llama_server_devices", _must_not_run)
    facts = gather_node_facts(
        env={
            "SKULK_LLAMA_SERVER_BIN": str(binary),
            "SKULK_LLAMA_SERVER_BACKENDS": "cuda",
        },
        platform="linux",
        nvidia_presence=False,
        drm_root=tmp_path / "no-drm",
    )
    assert facts.llama_server_device_probe.outcome == "not_run"


def test_device_probe_runs_real_binary(tmp_path: Path) -> None:
    # A stub llama-server that answers --list-devices like a CUDA build.
    binary = tmp_path / "llama-server"
    binary.write_text(
        "#!/bin/sh\n"
        'echo "Available devices:"\n'
        'echo "  CUDA0: NVIDIA A40 (46068 MiB, 45403 MiB free)"\n'
    )
    binary.chmod(0o755)
    facts = gather_node_facts(
        env={"SKULK_LLAMA_SERVER_BIN": str(binary)},
        platform="linux",
        nvidia_presence=False,
        drm_root=tmp_path / "no-drm",
    )
    assert facts.llama_server_device_probe.outcome == "devices"
    assert facts.llama_server_device_probe.computes == ("cuda",)


def test_device_probe_unsupported_flag(tmp_path: Path) -> None:
    binary = tmp_path / "llama-server"
    binary.write_text(
        "#!/bin/sh\n"
        'echo "error: invalid argument: --list-devices" >&2\n'
        "exit 1\n"
    )
    binary.chmod(0o755)
    facts = gather_node_facts(
        env={"SKULK_LLAMA_SERVER_BIN": str(binary)},
        platform="linux",
        nvidia_presence=False,
        drm_root=tmp_path / "no-drm",
    )
    assert facts.llama_server_device_probe.outcome == "unsupported"


def test_device_probe_crash_is_failed(tmp_path: Path) -> None:
    binary = tmp_path / "llama-server"
    binary.write_text("#!/bin/sh\necho boom >&2\nexit 7\n")
    binary.chmod(0o755)
    facts = gather_node_facts(
        env={"SKULK_LLAMA_SERVER_BIN": str(binary)},
        platform="linux",
        nvidia_presence=False,
        drm_root=tmp_path / "no-drm",
    )
    assert facts.llama_server_device_probe.outcome == "failed"
    assert "boom" in facts.llama_server_device_probe.detail
