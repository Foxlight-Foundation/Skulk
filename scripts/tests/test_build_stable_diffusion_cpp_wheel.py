"""CUDA image packages must not inherit build-host collective dependencies."""

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from scripts import build_stable_diffusion_cpp_wheel as builder


@pytest.mark.parametrize("nccl", ["ON", None])
def test_cuda_package_requires_explicit_single_gpu_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nccl: str | None
) -> None:
    source = tmp_path / "source"
    build = tmp_path / "build"
    (build / "bin").mkdir(parents=True)
    (build / "bin/sd-server").write_bytes(b"native")
    settings = "SD_SERVER_BUILD_FRONTEND:BOOL=OFF\nGGML_NATIVE:BOOL=OFF\n"
    if nccl is not None:
        settings += f"GGML_CUDA_NCCL:BOOL={nccl}\n"
    (build / "CMakeCache.txt").write_text(settings)
    patch = (
        Path(__file__).parents[2]
        / "packaging/skulk-stable-diffusion-cpp/server-lifecycle.patch"
    ).read_bytes()

    def git(root: Path, *arguments: str) -> bytes:
        if arguments == ("rev-parse", "HEAD"):
            revision = (
                builder.GGML_REVISION if root.name == "ggml" else builder.SOURCE_REVISION
            )
            return revision.encode()
        if arguments == ("diff", "--", "examples/server"):
            return patch
        return b""

    monkeypatch.setattr(builder, "_git", git)
    with pytest.raises(ValueError, match="NCCL explicitly disabled"):
        builder.assemble_wheel(
            source, build, tmp_path / "output", "cuda", "manylinux_2_35_x86_64"
        )
    assert not (tmp_path / "output").exists()


def test_native_package_records_compiled_cpu_requirements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    build = tmp_path / "build"
    (build / "bin").mkdir(parents=True)
    (build / "bin/sd-server").write_bytes(b"native")
    (build / "CMakeCache.txt").write_text(
        "SD_SERVER_BUILD_FRONTEND:BOOL=OFF\nGGML_NATIVE:BOOL=OFF\n"
        "GGML_AVX2:BOOL=ON\nGGML_F16C:BOOL=ON\nGGML_FMA:BOOL=OFF\n"
    )
    for relative in (
        "LICENSE", "ggml/LICENSE", "thirdparty/libwebp/COPYING",
        "thirdparty/oniguruma/COPYING", "thirdparty/utf8proc/LICENSE.md",
        "thirdparty/LICENSE.darts_clone.txt",
    ):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture license")
    patch = (
        Path(__file__).parents[2]
        / "packaging/skulk-stable-diffusion-cpp/server-lifecycle.patch"
    ).read_bytes()

    def git(root: Path, *arguments: str) -> bytes:
        if arguments == ("rev-parse", "HEAD"):
            return (builder.GGML_REVISION if root.name == "ggml" else builder.SOURCE_REVISION).encode()
        return patch if arguments == ("diff", "--", "examples/server") else b""

    monkeypatch.setattr(builder, "_git", git)
    runtime = tmp_path / "runtime"
    kernel = runtime / "rocblas/library/TensileLibrary_gfx942.dat"
    kernel.parent.mkdir(parents=True)
    kernel.write_bytes(b"immutable runtime kernel index")
    wheel = builder.assemble_wheel(
        source, build, tmp_path / "output", "vulkan", "manylinux_2_35_x86_64", runtime
    )
    descriptor = json.loads(wheel.with_suffix(".json").read_text())
    assert descriptor["required_cpu_features"] == ["avx2", "f16c"]
    assert descriptor["size_bytes"] == wheel.stat().st_size
    assert descriptor["expanded_size_bytes"] > 0
    with zipfile.ZipFile(wheel) as archive:
        manifest = json.loads(archive.read("skulk_stable_diffusion_cpp_vulkan/build.json"))
        name = "skulk_stable_diffusion_cpp_vulkan/lib/rocblas/library/TensileLibrary_gfx942.dat"
        assert archive.read(name) == kernel.read_bytes()
        assert name in manifest["members"]
    assert manifest["required_cpu_features"] == descriptor["required_cpu_features"]
    assert manifest["payload_sha256"] == descriptor["payload_sha256"]


def test_native_probe_revision_matches_assembled_source() -> None:
    """The C++ probe must bind the same immutable source as package provenance."""
    from skulk.provisioning import stable_diffusion_cpp as runtime

    patch = (Path(__file__).parents[2] / "packaging/skulk-stable-diffusion-cpp/server-lifecycle.patch").read_bytes()
    assert f'{{"source_revision", "{builder.SOURCE_REVISION}"}}'.encode() in patch
    assert hashlib.sha256(patch).hexdigest() == builder.PATCH_SHA256
    assert runtime.SOURCE_REVISION == builder.SOURCE_REVISION
    assert runtime.SERVER_PATCH_SHA256 == builder.PATCH_SHA256
