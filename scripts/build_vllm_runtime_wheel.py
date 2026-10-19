"""Build deterministic vLLM runtime recipe wheels and verify upstream mirrors."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

VERSION = "0.32.0.post1"
SOURCE = "5df63a7203cdb9328d824a27b682e77c7b9928be"
ROCM_LIBRARIES = (
    "libamdhip64.so.7",
    "libhiprtc.so.7",
    "libhipblas.so.3",
    "libhipblaslt.so.1",
    "libhipfft.so.0",
    "libhiprand.so.1",
    "libhipsparse.so.4",
    "libhipsolver.so.1",
    "librocsolver.so.0",
    "libhipsparselt.so.0",
    "librccl.so.1",
    "librocblas.so.5",
    "libMIOpen.so.1",
    "libroctx64.so.4",
    "librocprofiler-sdk.so.1",
    "libmpi_cxx.so.40",
    "libmpi.so.40",
)
VARIANTS = {
    "cuda-aarch64": (
        "cu129",
        "manylinux_2_28_aarch64",
        "258418994620a9bf904544d20f90e2cf6963788614b88d26d1b429f5d46a5194",
    ),
    "cuda-x86_64": (
        "cu129",
        "manylinux_2_28_x86_64",
        "9918be806fb2808c406547ff4f9f43870ac6ff28dc4b179291fcf532145bb50b",
    ),
    "rocm-x86_64": (
        "rocm723",
        "manylinux_2_39_x86_64",
        "733504dae65ee4cfa1b9ac8cf771116eec0f81264d4e0561b4ea255b6ba2190f",
    ),
}


def build_recipe(
    variant: str, output: Path, *, mirror: bool = True
) -> dict[str, object]:
    """Build one deterministic recipe and optionally download its verified mirror.

    Returns the package manifest entry; downloaded bytes are streamed to disk.
    A hash mismatch removes the partial mirror and fails the build.
    """
    channel, tag, upstream_digest = VARIANTS[variant]
    compute, architecture = variant.split("-", 1)
    abi = "cp312-cp312" if compute == "rocm" else "cp38-abi3"
    upstream_filename = f"vllm-0.32.0+{channel}-{abi}-{tag}.whl"
    upstream_root = "https://wheels.vllm.ai/" + ("rocm/" if compute == "rocm" else "")
    upstream_url = f"{upstream_root}{SOURCE}/{upstream_filename.replace('+', '%2B')}"
    root = Path(__file__).resolve().parents[1]
    lock = (root / "packaging/skulk-vllm-runtime/locks" / f"{variant}.txt").read_text()
    lock = lock.replace(
        f"vllm==0.32.0+{channel}",
        f"vllm @ https://wheels.foxlight.ai/runtime-inputs/vllm/{upstream_filename.replace('+', '%2B')}",
    )
    if f"--hash=sha256:{upstream_digest}" not in lock:
        raise ValueError("upstream wheel digest does not match the frozen lock")
    distribution = f"skulk_vllm_runtime_{compute}"
    info = f"{distribution}-{VERSION}.dist-info"
    payload: dict[str, bytes] = {
        f"{distribution}/requirements.txt": lock.encode(),
        f"{distribution}/runtime.json": (
            json.dumps(
                {
                    "version": VERSION,
                    "upstream": f"0.32.0+{channel}",
                    "source": SOURCE,
                    "variant": variant,
                    "python": "3.12.13",
                    "required_system_libraries": ROCM_LIBRARIES
                    if compute == "rocm"
                    else (),
                },
                sort_keys=True,
            )
            + "\n"
        ).encode(),
        f"{info}/METADATA": f"Metadata-Version: 2.1\nName: {distribution.replace('_', '-')}\nVersion: {VERSION}\nSummary: Hashed isolated vLLM runtime recipe\nRequires-Python: >=3.12\nLicense: Apache-2.0\n".encode(),
        f"{info}/WHEEL": f"Wheel-Version: 1.0\nGenerator: Foxlight\nRoot-Is-Purelib: true\nTag: py3-none-{tag}\n".encode(),
    }
    records = io.StringIO()
    writer = csv.writer(records, lineterminator="\n")
    for name, content in sorted(payload.items()):
        digest = (
            base64.urlsafe_b64encode(hashlib.sha256(content).digest())
            .rstrip(b"=")
            .decode()
        )
        writer.writerow((name, f"sha256={digest}", len(content)))
    writer.writerow((f"{info}/RECORD", "", ""))
    payload[f"{info}/RECORD"] = records.getvalue().encode()
    output.mkdir(parents=True, exist_ok=True)
    wheel = output / f"{distribution}-{VERSION}-py3-none-{tag}.whl"
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(payload.items()):
            entry = zipfile.ZipInfo(name, date_time=(2026, 10, 18, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = 0o644 << 16
            archive.writestr(entry, content)
    if mirror:
        destination = output / upstream_filename
        digest_builder = hashlib.sha256()
        try:
            with (
                urllib.request.urlopen(upstream_url, timeout=120) as source,
                destination.open("wb") as target,
            ):
                while chunk := source.read(1024 * 1024):
                    digest_builder.update(chunk)
                    target.write(chunk)
            if digest_builder.hexdigest() != upstream_digest:
                raise ValueError("upstream mirror checksum mismatch")
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
    result: dict[str, object] = {
        "variant": compute,
        "architecture": architecture,
        "platform_tag": tag,
        "filename": wheel.name,
        "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "size_bytes": wheel.stat().st_size,
        "lock_sha256": hashlib.sha256(lock.encode()).hexdigest(),
        "upstream_filename": upstream_filename,
        "upstream_sha256": upstream_digest,
        "source_revision": SOURCE,
        "version": f"0.32.0+{channel}",
        "required_system_libraries": ROCM_LIBRARIES if compute == "rocm" else (),
    }
    (output / f"{variant}.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    """Build the requested candidate packages without publishing anything."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=VARIANTS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recipe-only", action="store_true")
    args = parser.parse_args()
    result = build_recipe(args.variant, args.output, mirror=not args.recipe_only)
    manifest = (
        Path(__file__).resolve().parents[1]
        / "src/skulk/provisioning/vllm_manifest.json"
    )
    trusted = json.loads(manifest.read_text())
    if json.dumps(result, sort_keys=True) not in [
        json.dumps(entry, sort_keys=True) for entry in trusted
    ]:
        raise ValueError("candidate does not match the pinned runtime manifest")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
