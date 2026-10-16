"""Native image preparation must never confuse available packages with usable GPUs."""

import hashlib
import json
import subprocess
import threading
import zipfile
from pathlib import Path
from typing import cast

import pytest

from skulk.facts.derive import derive_node_backends
from skulk.facts.inventory import engine_build_inventory
from skulk.provisioning import stable_diffusion_cpp as image_engine
from skulk.shared.types.node_facts import NodeFacts, StableDiffusionCppProbe


@pytest.fixture(autouse=True)
def compatible_wheel_platforms(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep simulated hardware tests independent from the test runner's own ABI."""
    tags = tuple(wheel.filename.rsplit("-", 1)[1].removesuffix(".whl") for wheel in image_engine.PUBLISHED_WHEELS)
    monkeypatch.setattr(image_engine, "platform_tags", lambda: iter(tags))


@pytest.mark.parametrize(
    "wheel",
    [wheel for wheel in image_engine.PUBLISHED_WHEELS if wheel.system == "linux"],
)
@pytest.mark.parametrize(
    "host_abi", ["glibc-old", "musl", "unknown", "glibc-supported"]
)
def test_linux_abi_gates_install_offers_and_prevents_package_acquisition(
    wheel: image_engine.StableDiffusionCppWheel,
    host_abi: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Qualified GPUs on old/non-glibc hosts never advertise or acquire a wheel."""
    from skulk.shared.types.node_facts import EngineBinaryFact, GpuDeviceFact

    monkeypatch.setattr(image_engine.platform, "system", lambda: "Linux")
    monkeypatch.setattr(image_engine.platform, "machine", lambda: wheel.architecture)
    monkeypatch.setattr(
        image_engine, "_host_cpu_features", lambda: wheel.required_cpu_features
    )
    monkeypatch.setattr(image_engine, "PUBLISHED_WHEELS", (wheel,))
    monkeypatch.delenv(image_engine.AUTOPROVISION_OPT_OUT_ENV, raising=False)
    monkeypatch.delenv("SKULK_NODE_PARTICIPATION", raising=False)
    tags = {
        "glibc-old": (
            f"manylinux_2_34_{wheel.architecture}",
            f"linux_{wheel.architecture}",
        ),
        "musl": (f"musllinux_1_2_{wheel.architecture}", f"linux_{wheel.architecture}"),
        "unknown": (),
        "glibc-supported": (
            f"manylinux_2_36_{wheel.architecture}",
            f"manylinux_2_35_{wheel.architecture}",
        ),
    }[host_abi]
    monkeypatch.setattr(image_engine, "platform_tags", lambda: iter(tags))
    classes = wheel.qualified_hardware_classes
    backend = f"stable_diffusion_cpp-{wheel.compute}"
    missing = EngineBinaryFact(env_var="unset")
    facts = NodeFacts(
        platform="linux",
        llama_server_binary=missing,
        vllm_binary=missing,
        rpc_server_binary=missing,
        gpus=(
            GpuDeviceFact(
                vendor="nvidia" if wheel.compute == "cuda" else "amd",
                name="qualified GPU",
                index=0,
                detection_source="nvml" if wheel.compute == "cuda" else "amdgpu_sysfs",
            ),
        ),
    )
    eligible = host_abi == "glibc-supported"
    assert image_engine.wheel_for_host(backend, classes) == (
        wheel if eligible else None
    )
    assert image_engine.installable_image_builds(facts, classes, offline=False) == (
        {backend: wheel.build_identity} if eligible else {}
    )
    if not eligible:

        def refuse_cache(_wheel: image_engine.StableDiffusionCppWheel) -> Path:
            raise AssertionError(
                "ABI-incompatible hosts must fail before package/cache acquisition"
            )

        monkeypatch.setattr(image_engine, "_root", refuse_cache)
        with pytest.raises(
            image_engine.ImageEngineInstallError, match="eligibility check"
        ):
            image_engine.install_native_image(
                backend,
                wheel.build_identity,
                classes,
                offline=False,
                stop=threading.Event(),
                environ={},
            )


def test_managed_build_binds_runtime_and_isolates_inherited_libraries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "skulk_stable_diffusion_cpp_rocm"
    binary = root / "bin/sd-server"
    library = root / "lib/runtime.so"
    binary.parent.mkdir(parents=True)
    library.parent.mkdir(parents=True)
    binary.write_bytes(b"native executable")
    library.write_bytes(b"qualified runtime")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/unrelated/sdk")
    monkeypatch.setenv("LD_PRELOAD", "/unrelated/injection.so")
    monkeypatch.setattr(image_engine.platform, "system", lambda: "Linux")

    def record_manifest() -> str:
        members = {str(path.relative_to(tmp_path)): image_engine.file_sha256(path) for path in (binary, library)}
        digest = hashlib.sha256(json.dumps(members, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        (root / "build.json").write_text(json.dumps({"payload_sha256": digest, "members": members}))
        return digest

    def run_probe(*_args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        environment = cast("dict[str, str]", kwargs["env"])
        assert environment["LD_LIBRARY_PATH"] == str(library.parent)
        assert "LD_PRELOAD" not in environment
        return subprocess.CompletedProcess([], 0, json.dumps({"protocol": 1, "source_revision": image_engine.SOURCE_REVISION, "device_report": "HIP0\tAMD GPU\n"}), "")

    monkeypatch.setattr(image_engine.subprocess, "run", run_probe)
    first_digest = record_manifest()
    first = image_engine.probe_native_image(binary)
    assert first.build_identity == f"stable_diffusion_cpp@sha256:{first_digest}"
    library.write_bytes(b"different runtime")
    with pytest.raises(ValueError, match="payload member differs"):
        image_engine.probe_native_image(binary)
    record_manifest()
    second = image_engine.probe_native_image(binary)
    assert second.binary_sha256 == first.binary_sha256
    assert second.build_identity != first.build_identity


def test_package_sizes_are_bounded_and_bind_descriptor_identity() -> None:
    wheel = image_engine.StableDiffusionCppWheel(
        filename="skulk_stable_diffusion_cpp_rocm-0.1.0-py3-none-manylinux_2_35_x86_64.whl",
        sha256="a" * 64,
        binary_sha256="b" * 64,
        payload_sha256="c" * 64,
        compute="rocm",
        system="linux",
        architecture="x86_64",
        qualified_hardware_classes=frozenset({"qualified-gpu"}),
        size_bytes=2 * 1024**3,
        expanded_size_bytes=6 * 1024**3,
    )
    assert wheel.build_identity == "stable_diffusion_cpp@sha256:" + "c" * 64
    with pytest.raises(ValueError):
        image_engine.StableDiffusionCppWheel.model_validate(
            {**wheel.model_dump(), "size_bytes": 3 * 1024**3 + 1}
        )


def test_spawned_runner_resolves_exact_prepared_cache_without_inherited_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "prepared"
    prefix = "skulk_stable_diffusion_cpp_vulkan"
    filename = (
        "skulk_stable_diffusion_cpp_vulkan-0.1.0-py3-none-manylinux_2_35_x86_64.whl"
    )
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
        compute="vulkan",
        system="linux",
        architecture="x86_64",
        qualified_hardware_classes=frozenset({"qualified-gpu"}),
    )
    monkeypatch.setattr(image_engine, "PUBLISHED_WHEELS", (wheel,))
    monkeypatch.setattr(image_engine.platform, "system", lambda: "Linux")
    monkeypatch.setattr(image_engine.platform, "machine", lambda: "x86_64")

    def prepared_root(_wheel: image_engine.StableDiffusionCppWheel) -> Path:
        return root

    monkeypatch.setattr(image_engine, "_root", prepared_root)
    monkeypatch.delenv(image_engine.BINARY_ENV, raising=False)
    classes = frozenset({"qualified-gpu"})
    assert (
        image_engine.prepared_native_image(
            "stable_diffusion_cpp-vulkan", wheel.build_identity, classes
        )
        == binary
    )
    assert (
        image_engine.prepared_native_image(
            "stable_diffusion_cpp-vulkan", "another-build", classes
        )
        is None
    )
    assert (
        image_engine.prepared_native_image(
            "stable_diffusion_cpp-vulkan",
            wheel.build_identity,
            frozenset({"another-gpu"}),
        )
        == binary
    )
    (root / prefix / "lib/runtime.txt").write_bytes(b"corrupt runtime")
    assert (
        image_engine.prepared_native_image(
            "stable_diffusion_cpp-vulkan", wheel.build_identity, classes
        )
        is None
    )


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


def test_unpublished_package_is_not_installable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(image_engine, "PUBLISHED_WHEELS", ())
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


@pytest.mark.parametrize(
    ("system", "machine", "backend", "qualified", "untested", "compatible"),
    [
        ("Darwin", "arm64", "metal", "apple:apple-m4", "apple:apple-m5", True),
        ("Linux", "aarch64", "cuda", "nvidia:nvidia-gb10", "nvidia:sm-12.0", False),
        (
            "Linux",
            "x86_64",
            "cuda",
            "nvidia:nvidia-a100-sxm4-80gb",
            "nvidia:sm-9.0",
            True,
        ),
        (
            "Linux",
            "x86_64",
            "cuda",
            "nvidia:nvidia-geforce-rtx-4090",
            "nvidia:sm-8.6",
            True,
        ),
        ("Linux", "x86_64", "vulkan", "amd:pci-1002-1586", "intel", True),
        ("Linux", "x86_64", "rocm", "amd:pci-1002-74b9", "amd:pci-1002-74a1", True),
    ],
)
def test_published_packages_accept_compatible_untested_hardware(
    monkeypatch: pytest.MonkeyPatch,
    system: str,
    machine: str,
    backend: str,
    qualified: str,
    untested: str,
    compatible: bool,
) -> None:
    """Shipping packages admit untested devices unless compiled requirements disagree."""
    monkeypatch.setattr(image_engine.platform, "system", lambda: system)
    monkeypatch.setattr(image_engine.platform, "machine", lambda: machine)
    monkeypatch.setattr(
        image_engine,
        "_host_cpu_features",
        lambda: frozenset({"avx", "avx2", "bmi2", "f16c", "fma", "sse4_2"}),
    )
    tag = f"stable_diffusion_cpp-{backend}"
    wheel = image_engine.wheel_for_host(tag, frozenset({qualified}))
    assert wheel is not None
    assert wheel.payload_sha256 is not None
    assert wheel.size_bytes is not None
    assert wheel.expanded_size_bytes is not None
    assert (
        image_engine.wheel_for_host(tag, frozenset({untested})) is not None
    ) == compatible


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
        qualified_hardware_classes=frozenset({"qualified-gpu"}),
        required_cpu_features=frozenset({"avx2", "f16c"}),
    )
    monkeypatch.setattr(image_engine, "PUBLISHED_WHEELS", (wheel,))
    monkeypatch.setattr(image_engine.platform, "system", lambda: "Linux")
    monkeypatch.setattr(image_engine.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(image_engine, "_host_cpu_features", lambda: frozenset({"avx2"}))
    assert (
        image_engine.wheel_for_host(
            "stable_diffusion_cpp-vulkan", frozenset({"qualified-gpu"})
        )
        is None
    )
    monkeypatch.setattr(
        image_engine, "_host_cpu_features", lambda: frozenset({"avx2", "f16c"})
    )
    assert (
        image_engine.wheel_for_host(
            "stable_diffusion_cpp-vulkan", frozenset({"qualified-gpu"})
        )
        == wheel
    )


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


@pytest.mark.parametrize("device_count", [0, 1, 2])
def test_install_offer_requires_one_observed_device(device_count: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject a multi-GPU host before a download the native loader will reject."""
    from skulk.shared.types.node_facts import EngineBinaryFact, GpuDeviceFact

    monkeypatch.setattr(image_engine.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(image_engine.platform, "machine", lambda: "arm64")
    monkeypatch.delenv(image_engine.AUTOPROVISION_OPT_OUT_ENV, raising=False)
    monkeypatch.delenv("SKULK_NODE_PARTICIPATION", raising=False)
    missing = EngineBinaryFact(env_var="unset")
    facts = NodeFacts(platform="darwin", llama_server_binary=missing, vllm_binary=missing, rpc_server_binary=missing, gpus=tuple(GpuDeviceFact(vendor="apple", name="Apple M4", index=index, detection_source="apple_platform") for index in range(device_count)))
    builds = image_engine.installable_image_builds(facts, frozenset({"apple:apple-m4"}), offline=False)
    assert bool(builds) == (device_count == 1)


def test_compiled_targets_remain_real_requirements_not_qualification() -> None:
    """Known code-object mismatches reject the package while compatible targets pass."""
    cuda = next(
        wheel
        for wheel in image_engine.PUBLISHED_WHEELS
        if wheel.compute == "cuda" and wheel.architecture == "x86_64"
    )
    hip = next(
        wheel for wheel in image_engine.PUBLISHED_WHEELS if wheel.compute == "rocm"
    )
    assert image_engine.native_image_build_hardware_matches(
        cuda.build_identity, frozenset({"nvidia", "nvidia:sm-12.0"})
    )
    assert not image_engine.native_image_build_hardware_matches(
        cuda.build_identity, frozenset({"nvidia", "nvidia:sm-7.5"})
    )
    assert image_engine.native_image_build_hardware_matches(
        hip.build_identity, frozenset({"amd", "amd:gfx942"})
    )
    assert not image_engine.native_image_build_hardware_matches(
        hip.build_identity, frozenset({"amd", "amd:gfx1151"})
    )
