"""Native image preparation must never confuse available packages with usable GPUs."""

import hashlib
import json
import subprocess
import threading
import zipfile
from pathlib import Path

import pytest

from skulk.facts.derive import derive_node_backends
from skulk.facts.inventory import engine_build_inventory
from skulk.provisioning import stable_diffusion_cpp as image_engine
from skulk.shared.types.node_facts import NodeFacts, StableDiffusionCppProbe


def test_spawned_runner_resolves_exact_prepared_cache_without_inherited_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "prepared"
    prefix = "skulk_stable_diffusion_cpp_vulkan"
    filename = "skulk_stable_diffusion_cpp_vulkan-0.1.0-py3-none-manylinux_2_35_x86_64.whl"
    payload = {
        f"{prefix}/bin/sd-server": b"verified executable",
        f"{prefix}/lib/runtime.txt": b"verified runtime",
    }
    root.mkdir()
    with zipfile.ZipFile(root / filename, "w") as archive:
        for name, data in payload.items():
            archive.writestr(name, data)
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
    binary = root / prefix / "bin/sd-server"
    binary.chmod(0o755)
    wheel = image_engine.StableDiffusionCppWheel(
        filename=filename,
        sha256=hashlib.sha256((root / filename).read_bytes()).hexdigest(),
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        compute="vulkan", system="linux", architecture="x86_64",
        hardware_classes=frozenset({"qualified-gpu"}),
    )
    monkeypatch.setattr(image_engine, "PUBLISHED_WHEELS", (wheel,))
    monkeypatch.setattr(image_engine.platform, "system", lambda: "Linux")
    monkeypatch.setattr(image_engine.platform, "machine", lambda: "x86_64")
    def prepared_root(_wheel: image_engine.StableDiffusionCppWheel) -> Path:
        return root

    monkeypatch.setattr(image_engine, "_root", prepared_root)
    monkeypatch.delenv(image_engine.BINARY_ENV, raising=False)
    classes = frozenset({"qualified-gpu"})
    assert image_engine.prepared_native_image("stable_diffusion_cpp-vulkan", wheel.build_identity, classes) == binary
    assert image_engine.prepared_native_image("stable_diffusion_cpp-vulkan", "another-build", classes) is None
    assert image_engine.prepared_native_image("stable_diffusion_cpp-vulkan", wheel.build_identity, frozenset({"another-gpu"})) is None
    (root / prefix / "lib/runtime.txt").write_bytes(b"corrupt runtime")
    assert image_engine.prepared_native_image("stable_diffusion_cpp-vulkan", wheel.build_identity, classes) is None


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


def test_package_selection_requires_compiled_cpu_instructions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wheel = image_engine.StableDiffusionCppWheel(
        filename="skulk_stable_diffusion_cpp_vulkan-0.1.0-py3-none-manylinux_2_35_x86_64.whl",
        sha256="a" * 64,
        binary_sha256="b" * 64,
        compute="vulkan",
        system="linux",
        architecture="x86_64",
        hardware_classes=frozenset({"qualified-gpu"}),
        required_cpu_features=frozenset({"avx2", "f16c"}),
    )
    monkeypatch.setattr(image_engine, "PUBLISHED_WHEELS", (wheel,))
    monkeypatch.setattr(image_engine.platform, "system", lambda: "Linux")
    monkeypatch.setattr(image_engine.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(image_engine, "_host_cpu_features", lambda: frozenset({"avx2"}))
    assert image_engine.wheel_for_host("stable_diffusion_cpp-vulkan", frozenset({"qualified-gpu"})) is None
    monkeypatch.setattr(image_engine, "_host_cpu_features", lambda: frozenset({"avx2", "f16c"}))
    assert image_engine.wheel_for_host("stable_diffusion_cpp-vulkan", frozenset({"qualified-gpu"})) == wheel


def test_native_probe_refuses_missing_cpu_instructions_before_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = tmp_path / "native" / "bin" / "sd-server"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"native")
    (binary.parent.parent / "build.json").write_text(
        json.dumps({"required_cpu_features": ["avx2", "f16c"], "source_revision": "a" * 40})
    )
    monkeypatch.setattr(image_engine, "_host_cpu_features", lambda: frozenset({"avx2"}))

    def refuse_execution(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("a CPU-incompatible binary must not execute")

    monkeypatch.setattr(image_engine.subprocess, "run", refuse_execution)
    with pytest.raises(ValueError, match="unavailable CPU instructions"):
        image_engine.probe_native_image(binary)


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
