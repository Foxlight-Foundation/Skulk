"""Exercise real offline cache preparation, reuse and lost-selection recovery.

The caller supplies a reviewed package and independent whole/payload identities.
Only a new task-owned directory is accepted. This never downloads weights,
publishes a package or advertises loaded-model readiness.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from skulk.provisioning.demucs_runtime import (
    DemucsDevice,
    DemucsPlatform,
    DemucsRuntimeCache,
    DemucsWheel,
    PreparationPhase,
)
from skulk.utils.runtime_files import private_directory, write_private


class Arguments(argparse.Namespace):
    """Original package, reviewed identities and new private qualification root."""

    wheel: Path
    sha256: str
    payload_sha256: str
    target: DemucsPlatform
    output: Path
    mps: bool


def main() -> None:
    """Run actual native CPU preparation twice, recover it, then optionally MPS.

    Seed only the original package object; every prepare remains offline and
    executes the real verifier, bootstrap, supervisor and finite device probe.
    Reuse and simulated lost-selection recovery must retain one generation.
    Store bounded measurements in the new private root for qualification review.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--payload-sha256", required=True)
    parser.add_argument(
        "--target",
        choices=(
            "macosx_14_0_arm64",
            "manylinux_2_28_x86_64",
            "manylinux_2_28_aarch64",
        ),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mps", action="store_true")
    arguments = Arguments()
    parser.parse_args(namespace=arguments)
    output = arguments.output.absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("cache qualification requires a new owned directory")
    source = arguments.wheel.resolve(strict=True)
    wheel = DemucsWheel(
        platform_tag=arguments.target,
        filename=source.name,
        sha256=arguments.sha256,
        payload_sha256=arguments.payload_sha256,
        size_bytes=source.stat().st_size,
    )
    if arguments.mps and wheel.platform_tag != "macosx_14_0_arm64":
        raise ValueError("MPS qualification requires the Apple Silicon package")
    private_directory(output)
    cache_root = output / "cache"
    objects = cache_root / wheel.sha256 / "objects"
    private_directory(cache_root)
    private_directory(objects.parent)
    private_directory(objects)
    shutil.copyfile(source, objects / wheel.filename)
    controller = DemucsRuntimeCache(cache_root)
    observations: list[dict[str, str | float | list[str]]] = []

    def prepare(name: str, device: DemucsDevice) -> Path:
        phases: list[str] = []

        def progress(phase: PreparationPhase) -> None:
            phases.append(phase)
            print(json.dumps({"case": name, "phase": phase}), flush=True)

        started = time.monotonic()
        prepared = controller.prepare(
            wheel,
            device=device,
            offline=True,
            is_cancelled=lambda: False,
            progress=progress,
        )
        if "downloading" in phases:
            raise AssertionError("offline cache qualification attempted a download")
        observations.append(
            {
                "case": name,
                "device": prepared.device,
                "seconds": time.monotonic() - started,
                "generation": prepared.generation.name,
                "phases": phases,
            }
        )
        print(json.dumps(observations[-1]), flush=True)
        return prepared.generation

    try:
        generation = prepare("fresh-offline", "cpu")
        if prepare("verified-offline-reuse", "cpu") != generation:
            raise AssertionError("cache reuse unexpectedly installed a generation")
        (cache_root / wheel.sha256 / "selected.json").unlink()
        if prepare("lost-selection-offline-recovery", "cpu") != generation:
            raise AssertionError("lost-selection recovery replaced its intact runtime")
        if arguments.mps and prepare("verified-mps-no-fallback", "mps") != generation:
            raise AssertionError("MPS probing replaced its intact CPU/MPS runtime")
        if len(tuple(generation.parent.iterdir())) != 1:
            raise AssertionError("cache qualification retained unexpected generations")
    finally:
        controller.close()
    write_private(
        output / "qualification.json",
        json.dumps(
            {
                "status": "pass",
                "package_sha256": wheel.sha256,
                "payload_sha256": wheel.payload_sha256,
                "target": wheel.platform_tag,
                "observations": observations,
                "ready_model_claim": False,
            },
            indent=2,
        ).encode(),
    )


if __name__ == "__main__":
    main()
