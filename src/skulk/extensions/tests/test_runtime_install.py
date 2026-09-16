"""Generic signed runtime staging with real offline pip and durable refusal evidence."""

import asyncio
import hashlib
import importlib.metadata
import io
import json
import sqlite3
import time
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import JsonValue, TypeAdapter

from skulk.extensions.runtime_artifacts import (
    QualifiedHost,
    RuntimeTrust,
    canonical_json,
    verified_artifacts,
    verify_runtime,
)
from skulk.extensions.runtime_files import (
    private_directory,
    read_private,
    write_private,
)
from skulk.extensions.runtime_install import RuntimeInstaller


def artifacts(
    directory: Path,
    *,
    dependency: str | None = None,
    sequence: int = 1,
    owner_entrypoint: bool = True,
    owner_source: str = "print('owner fixture')\n",
    setup_source: str | None = None,
    management_source: str | None = None,
    signing_key: Ed25519PrivateKey | None = None,
    permissions: tuple[str, ...] = ("local synthetic operation",),
    launcher: bool = False,
    duplicate_launcher: bool = False,
    platform: str = "macos-arm64",
    runtime_protocol: int = 2,
    release_protocol: int = 1,
) -> tuple[bytes, RuntimeTrust, QualifiedHost]:
    """Create an independently signed generic package with no private SDK metadata."""
    private_directory(directory)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        prefix = "example_dep-1.0.dist-info/"
        files = {
            "example_dep/__init__.py": "VALUE = 7\n",
            prefix
            + "METADATA": "Metadata-Version: 2.1\nName: example-dep\nVersion: 1.0\n"
            + (f"Requires-Dist: {dependency}\n" if dependency else ""),
            prefix
            + "WHEEL": "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        }
        if launcher:
            # A wheel that declares the owner launcher lets the bundle omit the
            # fixed shim; the launcher then comes from signed, pinned bytes.
            files[prefix + "entry_points.txt"] = (
                "[skulk.capability_runtime]\n"
                "owner = example_dep:main\n"
                "manage = example_dep:main\n"
                "setup = example_dep:main\n"
            )
        files[prefix + "RECORD"] = (
            "".join(f"{name},,\n" for name in files) + prefix + "RECORD,,\n"
        )
        for name, content in files.items():
            archive.writestr(name, content)
    wheel = buffer.getvalue()
    extra: bytes | None = None
    if duplicate_launcher:
        # A second signed wheel claiming the same launcher: metadata order,
        # not the signature, would decide which owner starts.
        dup = io.BytesIO()
        dup_prefix = "example_dup-1.0.dist-info/"
        dup_files = {
            "example_dup/__init__.py": "",
            dup_prefix
            + "METADATA": "Metadata-Version: 2.1\nName: example-dup\nVersion: 1.0\n",
            dup_prefix
            + "WHEEL": "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
            dup_prefix
            + "entry_points.txt": "[skulk.capability_runtime]\nowner = example_dup:main\n",
        }
        dup_files[dup_prefix + "RECORD"] = (
            "".join(f"{name},,\n" for name in dup_files) + dup_prefix + "RECORD,,\n"
        )
        with zipfile.ZipFile(dup, "w") as archive:
            for name, content in dup_files.items():
                archive.writestr(name, content)
        extra = dup.getvalue()
    bundle_buffer = io.BytesIO()
    with zipfile.ZipFile(bundle_buffer, "w") as archive:
        archive.writestr(
            "__owner__.py" if owner_entrypoint else "__main__.py",
            owner_source,
        )
        if setup_source is not None:
            archive.writestr("__setup__.py", setup_source)
        if management_source is not None:
            archive.writestr("__manage__.py", management_source)
    bundle = bundle_buffer.getvalue()
    write_private(directory / "bundle.pyz", bundle)
    filename = "example_dep-1.0-py3-none-any.whl"
    write_private(directory / filename, wheel)
    inventory: dict[str, JsonValue] = {"example-dep": "1.0"}
    wheel_claims: list[JsonValue] = [
        {
            "filename": filename,
            "sha256": hashlib.sha256(wheel).hexdigest(),
            "size": len(wheel),
        }
    ]
    if extra is not None:
        dup_name = "example_dup-1.0-py3-none-any.whl"
        write_private(directory / dup_name, extra)
        inventory["example-dup"] = "1.0"
        wheel_claims.append(
            {
                "filename": dup_name,
                "sha256": hashlib.sha256(extra).hexdigest(),
                "size": len(extra),
            }
        )
    now = int(time.time())
    host = QualifiedHost(platform, "3.13.13", "1.5.2", "a" * 64)
    key = signing_key or Ed25519PrivateKey.generate()
    trust = RuntimeTrust(
        revision=1,
        expires_at=now + 3600,
        publishers={"fixture": key.public_key().public_bytes_raw().hex()},
    )
    payload: dict[str, JsonValue] = {
        "protocol": runtime_protocol,
        "implementation": "cpython",
        "platform": host.platform,
        "release": {
            "protocol": release_protocol,
            "publisher": "fixture",
            "sequence": sequence,
            "created_at": now - 1,
            "expires_at": now + 3600,
            "artifact_name": "bundle.pyz",
            "artifact_size": len(bundle),
            "manifest": {
                "bundle_id": "example.plugin",
                "bundle_version": "1.0.0",
                "skulk_requires": "==1.5.2",
                "executable": "bundle.pyz",
                "executable_sha256": hashlib.sha256(bundle).hexdigest(),
                "plugin_specific_policy": {"opaque": True},
            },
            "platforms": ["darwin" if platform.startswith("macos") else "linux"],
            "python_requires": "==3.13.*",
            "skulk_build_sha256": host.skulk_build_sha256,
            "dependency_lock_sha256": hashlib.sha256(
                canonical_json(inventory)
            ).hexdigest(),
            "state_schema": "example.v1",
            "compatible_state_schemas": [],
            "permissions": list(permissions),
        },
        "wheels": wheel_claims,
    }
    signed = canonical_json(
        {"runtime": payload, "signature": key.sign(canonical_json(payload)).hex()}
    )
    return signed, trust, host


@pytest.mark.parametrize(
    "fault", ["signature", "expired", "revoked", "core", "platform", "python"]
)
def test_generic_runtime_refuses_untrusted_or_incompatible(
    tmp_path: Path, fault: str
) -> None:
    """Exact authenticated claims gate every platform and build before execution."""
    metadata, trust, host = artifacts(tmp_path)
    now = int(time.time())
    if fault == "signature":
        metadata = metadata.replace(
            b"local synthetic operation", b"other synthetic operation"
        )
    elif fault == "expired":
        now += 7200
    elif fault == "revoked":
        digest = verify_runtime(metadata, trust, host, now=now).digest
        trust = trust.model_copy(update={"revoked_artifacts": (digest,)})
    elif fault == "core":
        host = replace(host, skulk_build_sha256="b" * 64)
    elif fault == "platform":
        host = replace(host, platform="ubuntu-24.04-x86_64")
    else:
        host = replace(host, python_version="3.14.0")
    with pytest.raises(ValueError):
        verify_runtime(metadata, trust, host, now=now)


@pytest.mark.parametrize(
    "dependency",
    ["missing==1.0", "example-dep>=2", "other @ https://invalid.example/other.whl"],
)
def test_complete_wheel_closure_required(tmp_path: Path, dependency: str) -> None:
    """No missing, incompatible or externally fetched wheel dependency is admitted."""
    metadata, trust, host = artifacts(tmp_path, dependency=dependency)
    runtime = verify_runtime(metadata, trust, host, now=int(time.time()))
    with pytest.raises(ValueError, match="dependency"):
        verified_artifacts(runtime, tmp_path)


def test_owner_launcher_comes_from_a_shim_or_a_signed_wheel(tmp_path: Path) -> None:
    """A bundle supplies the owner shim or a signed wheel declares the launcher.

    Neither is a refusal: the bundle would carry the SDK's host machinery for no
    reason, or nothing would be able to start the owner at all.
    """
    metadata, trust, host = artifacts(tmp_path, owner_entrypoint=False)
    runtime = verify_runtime(metadata, trust, host, now=int(time.time()))
    with pytest.raises(ValueError, match="owner entrypoint"):
        verified_artifacts(runtime, tmp_path)
    declared = tmp_path / "declared"
    metadata, trust, host = artifacts(declared, owner_entrypoint=False, launcher=True)
    runtime = verify_runtime(metadata, trust, host, now=int(time.time()))
    assert "bundle.pyz" in verified_artifacts(runtime, declared)
    # Two signed wheels claiming the same launcher is an ambiguity, not a
    # choice: metadata order would pick the owner, and it is unbound by the
    # signature.
    ambiguous = tmp_path / "ambiguous"
    metadata, trust, host = artifacts(
        ambiguous, owner_entrypoint=False, launcher=True, duplicate_launcher=True
    )
    runtime = verify_runtime(metadata, trust, host, now=int(time.time()))
    with pytest.raises(ValueError, match="twice"):
        verified_artifacts(runtime, ambiguous)


def test_launcher_declarations_follow_importlib_not_a_narrower_grammar() -> None:
    """What ``EntryPoint.load`` resolves is accepted; what it cannot read is refused."""
    from skulk.extensions.runtime_artifacts import (
        runtime_launchers,
        valid_launcher_target,
    )

    assert valid_launcher_target("pkg.cli : Runner.main")
    assert valid_launcher_target("pkg:main")
    assert not valid_launcher_target("pkg.cli")
    assert not valid_launcher_target("pkg:main [extra]")
    assert not valid_launcher_target("pkg:Runner..main")
    # A combining mark is an identifier to str.isidentifier and not a word
    # character to EntryPoint.pattern; the latter decides at load().
    assert not valid_launcher_target("pkg:a\u0301")

    def wheel(declaration: bytes) -> zipfile.ZipFile:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("d-1.0.dist-info/entry_points.txt", declaration)
        return zipfile.ZipFile(io.BytesIO(buffer.getvalue()))

    assert runtime_launchers(
        wheel(
            b"[skulk.capability_runtime]\n# generated-by = tooling\n"
            b"owner = pkg.cli : Runner.main\n"
        )
    ) == ["owner"]
    # importlib.metadata treats a ";" line as a declaration, so a wheel
    # carrying one is refused here for the same reason load() would fail.
    with pytest.raises(ValueError, match="invalid launcher"):
        runtime_launchers(wheel(b"[skulk.capability_runtime]\n; note = kept\n"))
    # importlib.metadata decodes the installed metadata strictly, so a byte
    # accepted here would become an owner that exits on every start.
    with pytest.raises(ValueError, match="UTF-8"):
        runtime_launchers(wheel(b"[skulk.capability_runtime]\nowner = pkg:main\n\xff"))


def test_platform_family_is_derived_and_legacy_names_still_match() -> None:
    """The installer names any host and still installs already-signed releases."""
    from skulk.extensions.runtime_artifacts import canonical_platform, platform_matches

    assert canonical_platform("ubuntu-24.04-x86_64") == "linux-glibc-x86_64"
    assert platform_matches("ubuntu-24.04-x86_64", "linux-unknown-x86_64")
    assert platform_matches("linux-glibc-aarch64", "linux-glibc-aarch64")
    assert not platform_matches("linux-glibc-x86_64", "linux-glibc-aarch64")
    assert not platform_matches("macos-arm64", "macos-x86_64")


async def test_offline_stage_and_reconnect_leave_host_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Install an exact wheel in a fresh environment, then reuse only cached bytes."""
    source = tmp_path / "source"
    metadata, trust, host = artifacts(source)
    monkeypatch.setattr("skulk.extensions.runtime_install.measure_host", lambda: host)
    root = tmp_path / "installation"
    installer = RuntimeInstaller(root)
    write_private(root / "publisher-trust.json", trust.model_dump_json().encode())
    inventory = sorted(
        (d.metadata["Name"], d.version) for d in importlib.metadata.distributions()
    )
    operation = await installer.stage(metadata, source, operation_id="1" * 32)
    assert operation.state == "staged"
    assert installer.operation(operation.operation_id) == operation
    assert (
        await installer.stage(
            metadata, tmp_path / "no-source", operation_id=operation.operation_id
        )
    ).state == "staged"
    assert inventory == sorted(
        (d.metadata["Name"], d.version) for d in importlib.metadata.distributions()
    )
    assert not (root / "active.json").exists()
    generation = root / "generations" / operation.runtime_digest
    assert read_private(generation / "staged.json") == metadata
    write_private(generation / "artifacts" / "bundle.pyz", b"tampered")
    with pytest.raises(ValueError, match="bundle differs"):
        await installer.stage(metadata, source, operation_id=operation.operation_id)
    assert installer.operation(operation.operation_id).state == "recovery_required"


async def test_interrupted_generation_is_retained_without_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A partial generation remains stopped and keeps evidence for explicit recovery."""
    metadata, trust, host = artifacts(tmp_path / "source")
    monkeypatch.setattr("skulk.extensions.runtime_install.measure_host", lambda: host)
    installer = RuntimeInstaller(tmp_path / "installation")
    write_private(
        installer.root / "publisher-trust.json", trust.model_dump_json().encode()
    )
    runtime = verify_runtime(metadata, trust, host, now=int(time.time()))
    partial = installer.root / "generations" / runtime.digest
    private_directory(installer.root / "generations")
    private_directory(partial)
    write_private(partial / "evidence", b"interrupted installation")
    operation = await installer.stage(metadata, tmp_path / "source")
    assert operation.state == "recovery_required"
    assert read_private(partial / "evidence") == b"interrupted installation"
    assert not (partial / "runtime").exists()
    assert (
        await installer.stage(
            metadata, tmp_path / "source", operation_id=operation.operation_id
        )
    ) == operation


async def test_refused_artifact_cannot_roll_back_observed_owner_trust(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed install still retains the owner's newer trust revision."""
    metadata, trust, host = artifacts(tmp_path / "source")
    monkeypatch.setattr("skulk.extensions.runtime_install.measure_host", lambda: host)
    installer = RuntimeInstaller(tmp_path / "installation")
    updated = trust.model_copy(update={"revision": 2})
    write_private(
        installer.root / "publisher-trust.json", updated.model_dump_json().encode()
    )
    with pytest.raises(ValueError, match="signature"):
        await installer.stage(
            metadata.replace(
                b"local synthetic operation", b"other synthetic operation"
            ),
            tmp_path / "source",
        )
    write_private(
        installer.root / "publisher-trust.json", trust.model_dump_json().encode()
    )
    with pytest.raises(ValueError, match="rollback"):
        await installer.stage(metadata, tmp_path / "source")
    changed = updated.model_copy(update={"expires_at": updated.expires_at + 1})
    write_private(
        installer.root / "publisher-trust.json", changed.model_dump_json().encode()
    )
    with pytest.raises(ValueError, match="equivocation"):
        await installer.stage(metadata, tmp_path / "source")


async def test_operation_conflict_does_not_rewrite_prior_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reusing an operation ID cannot change the operation the owner already observed."""
    metadata, trust, host = artifacts(tmp_path / "source")
    monkeypatch.setattr("skulk.extensions.runtime_install.measure_host", lambda: host)
    installer = RuntimeInstaller(tmp_path / "installation")
    write_private(
        installer.root / "publisher-trust.json", trust.model_dump_json().encode()
    )
    with sqlite3.connect(installer.database) as connection:
        record = json.dumps(
            {
                "operation_id": "2" * 32,
                "runtime_digest": "b" * 64,
                "state": "staging",
                "error_code": None,
            }
        )
        connection.execute(
            "INSERT INTO operations VALUES (?,?,?)", ("2" * 32, "b" * 64, record)
        )
    before = installer.operation("2" * 32)
    with pytest.raises(ValueError, match="identity differs"):
        await installer.stage(metadata, tmp_path / "source", operation_id="2" * 32)
    assert installer.operation("2" * 32) == before


async def test_cancelled_request_retains_installer_ownership(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Disconnect cannot release the installation fence while artifact work runs."""
    import threading

    from skulk.extensions import runtime_install

    metadata, trust, host = artifacts(tmp_path / "source")
    monkeypatch.setattr(runtime_install, "measure_host", lambda: host)
    entered, resume = threading.Event(), threading.Event()
    original = runtime_install.verified_artifacts

    def held_verify(
        runtime: runtime_install.VerifiedRuntime, directory: Path
    ) -> dict[str, bytes]:
        entered.set()
        assert resume.wait(10)
        return original(runtime, directory)

    monkeypatch.setattr(runtime_install, "verified_artifacts", held_verify)
    installer = RuntimeInstaller(tmp_path / "installation")
    write_private(
        installer.root / "publisher-trust.json", trust.model_dump_json().encode()
    )
    task = asyncio.create_task(
        installer.stage(metadata, tmp_path / "source", operation_id="3" * 32)
    )
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        with pytest.raises(BlockingIOError):
            await installer.stage(metadata, tmp_path / "source")
    finally:
        resume.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert installer.operation("3" * 32).state == "staged"


def test_the_protocol_window_is_current_and_previous_and_refuses_beyond_by_name(
    tmp_path: Path,
) -> None:
    """Every accepted protocol installs; one beyond is refused naming the window.

    Both windows hold one member today; the moment a second exists this runs
    both rows, so compatibility code is never left untested.
    """
    from skulk.extensions.runtime_artifacts import (
        ACCEPTED_RELEASE_PROTOCOLS,
        ACCEPTED_RUNTIME_PROTOCOLS,
        ProtocolUnsupportedError,
    )

    assert 1 <= len(ACCEPTED_RELEASE_PROTOCOLS) <= 2
    assert 1 <= len(ACCEPTED_RUNTIME_PROTOCOLS) <= 2
    for runtime_protocol in ACCEPTED_RUNTIME_PROTOCOLS:
        for release_protocol in ACCEPTED_RELEASE_PROTOCOLS:
            directory = tmp_path / f"{runtime_protocol}-{release_protocol}"
            metadata, trust, host = artifacts(
                directory,
                runtime_protocol=runtime_protocol,
                release_protocol=release_protocol,
            )
            assert verify_runtime(metadata, trust, host, now=int(time.time()))
    beyond = tmp_path / "beyond"
    metadata, trust, host = artifacts(
        beyond, runtime_protocol=max(ACCEPTED_RUNTIME_PROTOCOLS) + 1
    )
    with pytest.raises(ProtocolUnsupportedError) as refused:
        verify_runtime(metadata, trust, host, now=int(time.time()))
    # The refusal is typed and carries only the numbers, so every surface can
    # name it without disclosing anything else from the release.
    assert (refused.value.kind, refused.value.offered, refused.value.accepted) == (
        "runtime",
        max(ACCEPTED_RUNTIME_PROTOCOLS) + 1,
        ACCEPTED_RUNTIME_PROTOCOLS,
    )
    assert "is not accepted; this host accepts" in str(refused.value)
    metadata, trust, host = artifacts(
        tmp_path / "beyond-release",
        release_protocol=max(ACCEPTED_RELEASE_PROTOCOLS) + 1,
    )
    with pytest.raises(ProtocolUnsupportedError) as refused:
        verify_runtime(metadata, trust, host, now=int(time.time()))
    assert (refused.value.kind, refused.value.offered) == (
        "release",
        max(ACCEPTED_RELEASE_PROTOCOLS) + 1,
    )


def test_a_protocol_refusal_is_named_only_after_the_signature_verifies(
    tmp_path: Path,
) -> None:
    """A feed cannot forge upgrade guidance: an unsigned payload is a signature refusal."""
    from skulk.extensions.runtime_artifacts import ACCEPTED_RUNTIME_PROTOCOLS

    metadata, trust, host = artifacts(
        tmp_path, runtime_protocol=max(ACCEPTED_RUNTIME_PROTOCOLS) + 1
    )
    document = TypeAdapter(dict[str, JsonValue]).validate_json(metadata)
    document["signature"] = "0" * len(str(document["signature"]))
    with pytest.raises(ValueError, match="runtime signature refused"):
        verify_runtime(json.dumps(document).encode(), trust, host, now=int(time.time()))
