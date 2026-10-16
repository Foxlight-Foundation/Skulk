"""Assemble a native image engine wheel from pinned, already-built source.

Compilation happens in the release workflow, never at model placement. The
assembler records exact provenance and deterministic payload hashes; it does
not grant model or hardware support and does not publish artifacts.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path
from typing import Literal

SOURCE_REVISION = "d637cfb935feb2254898a8f1524046c958f3df80"
GGML_REVISION = "d25b121ce1d6ae0c7e62ad080460d9fc1e46dd93"
PATCH_SHA256 = "5c97917794372067d71ef6a49083f0c1afcfa3bbf46a368d08a7d0591691769b"
VERSION = "0.1.0.post1"
Variant = Literal["metal", "cuda", "vulkan", "rocm"]
PLATFORMS = {
    "macosx_15_0_arm64": ("darwin", "arm64"),
    "manylinux_2_35_x86_64": ("linux", "x86_64"),
    "manylinux_2_35_aarch64": ("linux", "aarch64"),
}


class Arguments(argparse.Namespace):
    """Typed options populated and validated by the command-line parser."""

    source: Path
    build: Path
    output: Path
    variant: Variant
    platform_tag: str
    runtime: Path | None


def _git(root: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(root), *arguments], capture_output=True, check=True
    ).stdout


def assemble_wheel(
    source: Path,
    build: Path,
    output: Path,
    variant: Variant,
    platform_tag: str,
    runtime: Path | None = None,
) -> Path:
    """Validate pinned source/patch and create an immutable native wheel plus its digest descriptor."""
    if platform_tag not in PLATFORMS or (variant == "metal") != platform_tag.startswith(
        "macosx_"
    ):
        raise ValueError("unsupported native image wheel target")
    if _git(source, "rev-parse", "HEAD").decode().strip() != SOURCE_REVISION:
        raise ValueError("engine source differs from the immutable pin")
    if _git(source / "ggml", "rev-parse", "HEAD").decode().strip() != GGML_REVISION:
        raise ValueError("ggml source differs from its immutable pin")
    if (
        hashlib.sha256(_git(source, "diff", "--", "examples/server")).hexdigest()
        != PATCH_SHA256
    ):
        raise ValueError("server lifecycle patch differs from its digest")
    if _git(source, "diff", "--", ":!examples/server") or _git(source / "ggml", "diff"):
        raise ValueError("unrecorded native source changes are not publishable")
    binary = build / "bin/sd-server"
    if not binary.is_file():
        raise ValueError("native server was not built")
    cache = (build / "CMakeCache.txt").read_text()
    if (
        "SD_SERVER_BUILD_FRONTEND:BOOL=OFF" not in cache
        or "GGML_NATIVE:BOOL=OFF" not in cache
    ):
        raise ValueError(
            "portable native wheel requires frontend off and host-specific instructions off"
        )
    cache_values = {
        line.split(":", 1)[0]: line.split("=", 1)[1]
        for line in cache.splitlines()
        if ":" in line and "=" in line and not line.startswith(("#", "//"))
    }
    cpu_flags = {
        "GGML_AVX": "avx",
        "GGML_AVX2": "avx2",
        "GGML_BMI2": "bmi2",
        "GGML_F16C": "f16c",
        "GGML_FMA": "fma",
        "GGML_SSE42": "sse4_2",
    }
    required_cpu_features = (
        sorted(feature for key, feature in cpu_flags.items() if cache_values.get(key) == "ON")
        if platform_tag.endswith("x86_64")
        else []
    )
    if variant == "metal" and cache_values.get("CMAKE_OSX_DEPLOYMENT_TARGET") != "15.0":
        raise ValueError("Metal wheel deployment target differs from its wheel tag")
    # A developer image can silently add NCCL when its headers are present.
    # Single-GPU image serving must not inherit that extra loader dependency.
    if variant == "cuda" and cache_values.get("GGML_CUDA_NCCL") != "OFF":
        raise ValueError("single-GPU CUDA wheels require NCCL explicitly disabled")
    module = f"skulk_stable_diffusion_cpp_{variant}"
    distribution = f"{module}-{VERSION}.dist-info"
    payload: dict[str, bytes] = {f"{module}/bin/sd-server": binary.read_bytes()}
    payload[f"{module}/__init__.py"] = (
        '"""Locate the pinned native image engine payload."""\n'
        "from pathlib import Path\n\n"
        "def binary_path() -> Path:\n"
        '    """Return the native executable shipped by this platform wheel."""\n'
        '    return Path(__file__).parent / "bin" / "sd-server"\n'
    ).encode()
    licenses = {
        "STABLE_DIFFUSION_CPP_LICENSE.txt": source / "LICENSE",
        "GGML_LICENSE.txt": source / "ggml/LICENSE",
        "WEBP_LICENSE.txt": source / "thirdparty/libwebp/COPYING",
        "ONIGURUMA_LICENSE.txt": source / "thirdparty/oniguruma/COPYING",
        "UTF8PROC_LICENSE.txt": source / "thirdparty/utf8proc/LICENSE.md",
        "DARTS_CLONE_LICENSE.txt": source / "thirdparty/LICENSE.darts_clone.txt",
    }
    for name, path in licenses.items():
        payload[f"{module}/licenses/{name}"] = path.read_bytes()
    if runtime is not None:
        for path in sorted(runtime.rglob("*")):
            if path.is_file():
                relative = path.relative_to(runtime).as_posix()
                payload[f"{module}/lib/{relative}"] = path.read_bytes()
    provenance = {
        "source_revision": SOURCE_REVISION,
        "ggml_revision": GGML_REVISION,
        "server_patch_sha256": PATCH_SHA256,
        "backend": variant,
        "platform_tag": platform_tag,
        "binary_sha256": hashlib.sha256(payload[f"{module}/bin/sd-server"]).hexdigest(),
        "required_cpu_features": required_cpu_features,
        "members": {
            name: hashlib.sha256(data).hexdigest()
            for name, data in sorted(payload.items())
        },
    }
    provenance["payload_sha256"] = hashlib.sha256(
        json.dumps(provenance["members"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    payload[f"{module}/build.json"] = json.dumps(
        provenance, sort_keys=True, indent=2
    ).encode()
    payload[f"{distribution}/METADATA"] = (
        f"Metadata-Version: 2.4\nName: {module.replace('_', '-')}\nVersion: {VERSION}\n"
        "Requires-Python: >=3.13\nLicense: See bundled component license notices\n"
        f"Summary: Pinned {variant} stable-diffusion.cpp image engine for Skulk\n\n"
    ).encode()
    payload[f"{distribution}/WHEEL"] = (
        f"Wheel-Version: 1.0\nGenerator: skulk-native-image-wheel\nRoot-Is-Purelib: false\nTag: py3-none-{platform_tag}\n"
    ).encode()
    records: list[str] = []
    for name, data in sorted(payload.items()):
        digest = (
            base64.urlsafe_b64encode(hashlib.sha256(data).digest())
            .rstrip(b"=")
            .decode()
        )
        records.append(f"{name},sha256={digest},{len(data)}")
    records.append(f"{distribution}/RECORD,,")
    payload[f"{distribution}/RECORD"] = ("\n".join(records) + "\n").encode()
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"{module}-{VERSION}-py3-none-{platform_tag}.whl"
    if destination.exists():
        raise FileExistsError("refusing to overwrite an existing wheel")
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(payload.items()):
            item = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            item.compress_type = zipfile.ZIP_DEFLATED
            item.external_attr = (
                0o100755 if name.endswith("/bin/sd-server") else 0o100644
            ) << 16
            archive.writestr(item, data)
    system, architecture = PLATFORMS[platform_tag]
    descriptor = {
        "filename": destination.name,
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "binary_sha256": provenance["binary_sha256"],
        "payload_sha256": provenance["payload_sha256"],
        "size_bytes": destination.stat().st_size,
        "expanded_size_bytes": sum(len(data) for name, data in payload.items() if name.startswith(module + "/")),
        "compute": variant,
        "system": system,
        "architecture": architecture,
        "source_revision": SOURCE_REVISION,
        "server_patch_sha256": PATCH_SHA256,
        "qualification_status": "unqualified",
        "required_cpu_features": required_cpu_features,
    }
    destination.with_suffix(".json").write_text(
        json.dumps(descriptor, sort_keys=True, indent=2) + "\n"
    )
    return destination


def main() -> None:
    """Parse explicit build inputs and emit one candidate package without publishing it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--variant", choices=("metal", "cuda", "vulkan", "rocm"), required=True
    )
    parser.add_argument("--platform-tag", choices=tuple(PLATFORMS), required=True)
    parser.add_argument("--runtime", type=Path)
    arguments = Arguments()
    parser.parse_args(namespace=arguments)
    print(
        assemble_wheel(
            arguments.source,
            arguments.build,
            arguments.output,
            arguments.variant,
            arguments.platform_tag,
            arguments.runtime,
        )
    )


if __name__ == "__main__":
    main()
