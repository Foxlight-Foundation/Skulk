"""CUDA image packages must not inherit build-host collective dependencies."""

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
