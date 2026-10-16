"""Bundle pinned-container ROCm dependencies and one target's runtime code objects."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


class Arguments(argparse.Namespace):
    """Explicit native executable, destination and qualified GPU compilation target."""

    binary: Path
    output: Path
    gpu_target: str


def bundle_runtime(binary: Path, output: Path, gpu_target: str) -> None:
    """Copy resolved nonbaseline libraries, target code objects and license notices.

    This runs in the immutable build container. The resulting files enter the
    wheel's digest manifest; it grants no hardware support claim. Missing runtime
    dependencies, code objects or notices fail the build rather than using a host
    ROCm installation implicitly.
    """
    if gpu_target != "gfx942":
        raise ValueError("only the selected CDNA3 candidate runtime is defined")
    completed = subprocess.run(
        ["ldd", str(binary)], capture_output=True, text=True, check=True
    )
    baseline = {
        "libc.so.6", "libstdc++.so.6", "libgcc_s.so.1", "libm.so.6",
        "libpthread.so.0", "libdl.so.2", "librt.so.1",
    }
    output.mkdir(parents=True, exist_ok=True)
    libraries: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        if "=>" not in line:
            continue
        name, target = (part.strip() for part in line.split("=>", 1))
        if name in baseline:
            continue
        path = Path(target.split()[0])
        if not path.is_absolute() or not path.is_file():
            raise ValueError(f"unresolved ROCm dependency: {name}")
        shutil.copyfile(path, output / name)
        subprocess.run(
            ["patchelf", "--set-rpath", "$ORIGIN", str(output / name)], check=True
        )
        libraries[name] = str(path)
    if not {"libamdhip64.so.7", "librocblas.so.5", "libhipblas.so.3"} <= libraries.keys():
        raise ValueError("native dependency inventory lacks the pinned ROCm runtime")
    for component in ("rocblas", "hipblaslt"):
        source = Path("/opt/rocm/lib") / component / "library"
        if not source.is_dir():
            raise ValueError(f"runtime data directory missing: {component}")
        destination = output / component / "library"
        destination.mkdir(parents=True)
        selected = 0
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            # The vendor packages carry code for many GPUs. This candidate's
            # engine and support evidence select only CDNA3; retain common indices.
            if "gfx" in path.name and gpu_target not in path.name:
                continue
            relative = path.relative_to(source)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            selected += 1
        if selected == 0:
            raise ValueError(f"no runtime code objects selected: {component}")
    notices = Path("/opt/rocm/share/doc")
    if not notices.is_dir():
        raise ValueError("ROCm compiler image lacks runtime license notices")
    shutil.copytree(notices, output / "licenses/rocm", symlinks=False)
    for component in ("libgomp1", "libnuma1", "libelf1", "libdrm2", "libdrm-amdgpu1", "libzstd1", "zlib1g"):
        notice = Path("/usr/share/doc") / component / "copyright"
        if notice.is_file():
            shutil.copyfile(notice, output / f"{component}-COPYRIGHT.txt")
    (output / "runtime-contract.json").write_text(
        json.dumps({"gpu_target": gpu_target, "libraries": libraries, "baseline_libraries": sorted(baseline)}, sort_keys=True, indent=2) + "\n"
    )


def main() -> None:
    """Parse build-container paths and assemble a recorded runtime directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-target", required=True)
    arguments = Arguments()
    parser.parse_args(namespace=arguments)
    bundle_runtime(arguments.binary, arguments.output, arguments.gpu_target)


if __name__ == "__main__":
    main()
