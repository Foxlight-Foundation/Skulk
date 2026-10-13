"""Native image preparation must never confuse available packages with usable GPUs."""

import json
import subprocess
import threading
from pathlib import Path

import pytest

from skulk.facts.derive import derive_node_backends
from skulk.facts.inventory import engine_build_inventory
from skulk.provisioning import stable_diffusion_cpp as image_engine
from skulk.shared.types.node_facts import NodeFacts, StableDiffusionCppProbe


def test_native_probe_rejects_cpu_only_devices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = tmp_path / "sd-server"
    binary.write_bytes(b"native")

    def run_probe(
        *_args: object, **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            [],
            0,
            json.dumps(
                {
                    "protocol": 1,
                    "source_revision": image_engine.SOURCE_REVISION,
                    "device_report": "CPU\tprocessor\nBLAS\tCPU accelerator\n",
                }
            ),
            "",
        )

    monkeypatch.setattr(image_engine.subprocess, "run", run_probe)
    with pytest.raises(ValueError, match="no GPU"):
        image_engine.probe_native_image(binary)


def test_native_probe_binds_build_to_executable_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = tmp_path / "sd-server"
    binary.write_bytes(b"native")

    def run_probe(
        *_args: object, **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            [],
            0,
            json.dumps(
                {
                    "protocol": 1,
                    "source_revision": image_engine.SOURCE_REVISION,
                    "device_report": "MTL0\tApple GPU\nCPU\tprocessor\n",
                }
            ),
            "",
        )

    monkeypatch.setattr(image_engine.subprocess, "run", run_probe)
    first = image_engine.probe_native_image(binary)
    binary.write_bytes(b"changed")
    second = image_engine.probe_native_image(binary)
    assert first.devices == ("MTL0",)
    assert first.computes == ("metal",)
    assert first.build_identity != second.build_identity


def test_unpublished_package_is_not_installable() -> None:
    assert (
        image_engine.wheel_for_host("stable_diffusion_cpp-metal", frozenset({"apple"}))
        is None
    )
    with pytest.raises(
        image_engine.ImageEngineInstallError, match="no exact published"
    ):
        image_engine.install_native_image(
            "stable_diffusion_cpp-metal",
            "unknown",
            frozenset({"apple"}),
            offline=False,
            stop=threading.Event(),
            environ={},
        )


def test_installed_native_build_cannot_be_overridden_by_a_declaration() -> None:
    from skulk.shared.types.node_facts import EngineBinaryFact

    missing = EngineBinaryFact(env_var="unset")
    facts = NodeFacts(
        platform="darwin",
        llama_server_binary=missing,
        vllm_binary=missing,
        rpc_server_binary=missing,
        stable_diffusion_cpp_probe=StableDiffusionCppProbe(
            outcome="ready",
            build_identity="measured",
            devices=("MTL0",),
            computes=("metal",),
        ),
    )
    inventory = engine_build_inventory(
        frozenset({"stable_diffusion_cpp-metal"}),
        facts,
        environ={"SKULK_ENGINE_BUILDS": '{"stable_diffusion_cpp-metal":"claimed"}'},
    )
    assert inventory["stable_diffusion_cpp-metal"] == "measured"


def test_installable_native_build_is_not_installed_inventory() -> None:
    from skulk.shared.types.node_facts import EngineBinaryFact

    missing = EngineBinaryFact(env_var="unset")
    facts = NodeFacts(
        platform="darwin",
        llama_server_binary=missing,
        vllm_binary=missing,
        rpc_server_binary=missing,
        stable_diffusion_cpp_installable_builds={
            "stable_diffusion_cpp-metal": "expected"
        },
    )
    derived = derive_node_backends(facts)
    assert "stable_diffusion_cpp-metal" in derived.on_demand_backends
    assert "stable_diffusion_cpp-metal" not in derived.installed_backends
    assert not engine_build_inventory(
        frozenset({"stable_diffusion_cpp-metal"}), facts, environ={}
    )


@pytest.mark.parametrize(
    "description",
    ["llvmpipe (LLVM)", "lavapipe", "SwiftShader Device", "software rasterizer"],
)
def test_native_probe_refuses_vulkan_software_devices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, description: str
) -> None:
    binary = tmp_path / "sd-server"
    binary.write_bytes(b"native")

    def run_probe(
        *_args: object, **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            [],
            0,
            json.dumps(
                {
                    "protocol": 1,
                    "source_revision": image_engine.SOURCE_REVISION,
                    "device_report": f"Vulkan0\t{description}\nCPU\tprocessor\n",
                }
            ),
            "",
        )

    monkeypatch.setattr(image_engine.subprocess, "run", run_probe)
    with pytest.raises(ValueError, match="no GPU"):
        image_engine.probe_native_image(binary)


def test_native_image_profile_cannot_be_placed_on_a_legacy_loader() -> None:
    from skulk.shared.backends import platform_compatible_backends

    candidates = frozenset({"mlx", "mlx-metal", "stable_diffusion_cpp-metal"})
    assert platform_compatible_backends(candidates, card_serves_vision=False,
        card_has_native_image_profile=True) == frozenset({"stable_diffusion_cpp-metal"})
    assert platform_compatible_backends(candidates, card_serves_vision=False) == frozenset({"mlx", "mlx-metal"})
