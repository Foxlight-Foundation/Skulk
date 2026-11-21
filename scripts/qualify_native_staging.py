"""Exercise Core's normal native installer against an independently signed fixture.

Inputs must be a disposable qualification release and its explicit public trust.
This stages/selects a disabled runtime inside a new task directory. It never
starts the owner or proves first-party account/state migration.
"""

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import final

from skulk.extensions.runtime_artifacts import (
    RuntimeTrust,
    measure_host,
    verify_runtime,
)
from skulk.extensions.runtime_files import (
    private_directory,
    read_private,
    write_private,
)
from skulk.extensions.runtime_selection import RuntimeSelector


async def qualify(source: Path, trust_file: Path, work: Path, receipt: Path) -> None:
    """Stage exact native bytes, retain disabled selection and prove refusal paths.

    The measured Core runtime and ordinary installer are used without injected
    host observations. All writes stay under a previously absent work directory,
    except the explicitly requested payload-free receipt.
    """
    if work.exists():
        raise ValueError("qualification requires a new task-owned directory")
    metadata = (source / "release.json").read_bytes()
    trust = RuntimeTrust.model_validate_json(trust_file.read_bytes())
    host = measure_host()
    runtime = verify_runtime(metadata, trust, host, now=int(time.time()))
    native = runtime.native_manifest
    if native is None:
        raise ValueError("qualification requires a native runtime")
    private_directory(work)
    selector = RuntimeSelector(work / "installation")
    write_private(
        selector.root / "publisher-trust.json", trust.model_dump_json().encode()
    )
    canary = b"Synthetic protected-state retention witness; no real credentials."
    write_private(selector.root / "protected-state-canary", canary)
    checks = ["measured-core", "authenticated-native-fixture"]
    operation = await selector.installer.stage(metadata, source)
    if operation.state != "staged" or operation.runtime_digest != runtime.digest:
        raise ValueError("normal Core staging did not complete")
    checks.extend(("normal-offline-install", "installed-native-reader"))
    generation = selector.root / "generations" / runtime.digest
    if (
        read_private(generation / "native-manifest.json", 262144)
        != native.canonical_json.encode()
    ):
        raise ValueError("canonical manifest custody differs")
    checks.append("full-manifest-identity")
    if (
        await selector.installer.stage(
            metadata, source, operation_id=operation.operation_id
        )
        != operation
    ):
        raise ValueError("exact restaging differs")
    checks.append("exact-restage")
    selected = await selector.activate(
        runtime.digest, expected_revision=0, enabled=False
    )
    if (
        selected.selection.enabled
        or selected.selection.configuration_schema != runtime.configuration_schema
    ):
        raise ValueError("disabled native selection differs")
    checks.append("normal-disabled-selection")
    if (selector.root / "service-status.json").exists():
        raise ValueError("staging unexpectedly started a service")
    checks.append("no-owner-start")
    if (
        measure_host() != host
        or read_private(selector.root / "protected-state-canary") != canary
    ):
        raise ValueError("host or synthetic retained state changed")
    checks.append("host-and-synthetic-state-retained")
    changed = b"changed native manifest evidence"
    write_private(generation / "native-manifest.json", changed)
    try:
        async with selector.installer.locked_generation(runtime.digest):
            raise AssertionError("changed native bytes were admitted")
    except ValueError:
        pass
    checks.append("changed-native-bytes-refused")
    try:
        await selector.installer.stage(
            metadata, source, operation_id=operation.operation_id
        )
    except ValueError:
        pass
    else:
        raise ValueError("completed native tampering was repaired")
    if read_private(generation / "native-manifest.json") != changed:
        raise ValueError("tamper evidence was replaced")
    checks.append("no-completed-repair")
    disabled = selector.disable(expected_revision=selected.selection.revision)
    if disabled.selection.enabled:
        raise ValueError("retained disable failed")
    checks.append("disable-retains-evidence")
    body = {
        "format": 1,
        "passed": True,
        "scope": "Normal Core native offline staging and disabled selection; no startup or real protected-state migration.",
        "checks": checks,
        "runtime_digest": runtime.digest,
        "manifest_digest": native.digest,
        "metadata_sha256": hashlib.sha256(metadata).hexdigest(),
        "host": {
            "platform": host.platform,
            "python_version": host.python_version,
            "skulk_version": host.skulk_version,
            "skulk_build_sha256": host.skulk_build_sha256,
        },
        "inventory": runtime.inventory,
    }
    receipt.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")


@final
class _Arguments(argparse.Namespace):
    """Typed argparse destinations; every value is required by the parser."""

    source: Path = Path()
    trust: Path = Path()
    work: Path = Path()
    receipt: Path = Path()


def main() -> None:
    """Run the explicit disposable qualification without touching configured stores."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--trust", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    arguments = _Arguments()
    _ = parser.parse_args(namespace=arguments)
    asyncio.run(
        qualify(arguments.source, arguments.trust, arguments.work, arguments.receipt)
    )


if __name__ == "__main__":
    main()
