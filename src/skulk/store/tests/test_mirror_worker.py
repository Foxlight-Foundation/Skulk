"""Real fixed-worker recovery and Stop/EOF without registry, Core or source files."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from foxlight_model_preservation.mirror_models import Terms
from foxlight_model_preservation.mirror_state import MirrorLedger, policy_id
from foxlight_model_preservation.models import Recipe, TrustProfile
from foxlight_model_preservation.signing import public_record
from foxlight_model_preservation.storage import (
    manifest_path,
    object_path,
    pack,
    verify_store,
)

from skulk.store.mirror_contracts import MirrorConfiguration, MirrorRequest
from skulk.store.mirror_worker import (
    MirrorRetainSource,
    MirrorWorkerEvent,
    MirrorWorkerRequest,
)


def configuration() -> MirrorConfiguration:
    """Select small explicit synthetic budgets and loopback HTTP only."""
    return MirrorConfiguration(
        name="Synthetic Core mirror",
        disk_bytes=2 * 1024**3,
        ingress_daily=1024**3,
        egress_daily=1024**3,
        egress_monthly=1024**3,
        http=True,
        port=0,
    )


async def launch(
    root: Path, request: MirrorWorkerRequest
) -> asyncio.subprocess.Process:
    """Launch through the same source/frozen-dispatched family as a real Core."""
    control = root / (request.decision.operation_id + ".json")
    control.write_text(request.model_dump_json())
    control.chmod(0o600)
    return await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "skulk.store.preservation_worker",
        str(control),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        pass_fds=(request.source.descriptor,) if request.source else (),
    )


async def complete(
    process: asyncio.subprocess.Process,
) -> tuple[int, tuple[MirrorWorkerEvent, ...]]:
    """Read without closing the live parent's input pipe prematurely."""
    assert process.stdout is not None and process.stderr is not None
    try:
        async with asyncio.timeout(25):
            output = await process.stdout.read()
            code = await process.wait()
            errors = await process.stderr.read()
        assert not errors, errors.decode()
        return code, tuple(
            MirrorWorkerEvent.model_validate_json(line) for line in output.splitlines()
        )
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
        if process.stdin is not None:
            process.stdin.close()


def archive(root: Path) -> tuple[Path, str, TrustProfile]:
    """Make a genuine signed two-member public closure with future card data."""
    members: list[dict[str, object]] = []
    sources: dict[str, Path] = {}
    for name, role, dependencies in (
        ("model", "primary", ["tokenizer"]),
        ("tokenizer", "tokenizer", []),
    ):
        path = root / name
        path.mkdir()
        (path / "payload.bin").write_bytes((name + " synthetic bytes\n").encode() * 400)
        (path / "LICENSE").write_text("Synthetic fixture, no actual model weights.\n")
        sources[name] = path
        members.append(
            {
                "member_id": name,
                "role": role,
                "format": "fixture",
                "origin_repository": "fixture/" + name,
                "origin_revision": "a" * 40,
                "artifact_identity": "fixture:" + name,
                "dependencies": dependencies,
                "retained_card": {"future": {"opaque": [1, True, "unchanged"]}},
                "intake": {
                    "audience": "public",
                    "license_id": "fixture-only",
                    "terms_reference": "LICENSE",
                    "notice_paths": ["LICENSE"],
                    "decision": "synthetic fixture",
                },
                "files": [{"path": "payload.bin"}, {"path": "LICENSE"}],
            }
        )
    recipe = Recipe.model_validate_json(
        json.dumps(
            {
                "schema_version": "foxlight.model-preservation.v1",
                "name": "Synthetic complete closure",
                "root_member": "model",
                "members": members,
                "workloads": ["fixture"],
                "runtime_requirements": ["no inference"],
            }
        )
    )
    key = Ed25519PrivateKey.generate()
    trust = TrustProfile(
        schema_version="foxlight.model-preservation.trust.v1",
        keys=(public_record(key),),
        threshold=1,
        as_of="2026-10-24T00:00:00Z",
    )
    destination = root / "input-archive"
    identity = pack(recipe, sources, destination, (key,))
    return destination, identity, trust


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", ["cancel", "eof"])
async def test_real_controller_retains_complete_original_evidence_and_drains_on_parent_loss(
    tmp_path: Path, stop: str
) -> None:
    mirror = tmp_path / "mirror"
    decision = MirrorRequest(
        owner_id="library",
        operation_id="configure",
        kind="configure",
        configuration=configuration(),
    )
    configured, events = await complete(
        await launch(
            tmp_path, MirrorWorkerRequest(mirror_path=mirror, decision=decision)
        )
    )
    assert configured == 0 and events[-1].event == "result"
    ledger = MirrorLedger(mirror)
    assert (
        ledger.read().paused and not ledger.running() and ledger.read().restart is None
    )
    root_identity = ledger.read().identity["."]
    original, identity, trust = archive(tmp_path)
    wrapper = (original / manifest_path(identity)).read_bytes()
    descriptor = os.open(original, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    info = os.fstat(descriptor)
    source = MirrorRetainSource(
        descriptor=descriptor,
        device=info.st_dev,
        inode=info.st_ino,
        trust=trust,
        terms=(Terms(license_id="fixture-only", terms_reference="LICENSE"),),
    )
    # The original path is replaced before the worker starts. Only its pinned
    # descriptor is authorized; never follow a newly mounted/replaced archive.
    moved = original.with_name("moved-input")
    original.rename(moved)
    original.mkdir()
    decision = MirrorRequest(
        owner_id="library",
        operation_id="retain",
        kind="retain",
        policy_id=policy_id(ledger.read()),
        copy_id="accepted-copy",
        manifest_id=identity,
        confirm_redistribution=True,
    )
    try:
        process = await launch(
            tmp_path,
            MirrorWorkerRequest(
                mirror_path=mirror,
                mirror_identity=root_identity,
                decision=decision,
                source=source,
            ),
        )
    finally:
        os.close(descriptor)
    retained, events = await complete(process)
    assert retained == 0 and events[-1].manifest_id == identity
    shutil.rmtree(moved)
    shutil.rmtree(tmp_path / "model")
    shutil.rmtree(tmp_path / "tokenizer")
    manifest = verify_store(mirror / "store", identity, trust)
    assert len(manifest.members) == 2 and not tuple(original.iterdir())
    assert (mirror / "store" / manifest_path(identity)).read_bytes() == wrapper
    assert all(
        member.retained_card == {"future": {"opaque": [1, True, "unchanged"]}}
        for member in manifest.members
    )
    decision = MirrorRequest(
        owner_id="library",
        operation_id="run",
        kind="run",
        policy_id=policy_id(ledger.read()),
        public_participation=True,
        seconds=60,
    )
    process = await launch(
        tmp_path,
        MirrorWorkerRequest(
            mirror_path=mirror, mirror_identity=root_identity, decision=decision
        ),
    )
    assert process.stdout is not None and process.stdin is not None
    try:
        async with asyncio.timeout(20):
            ready = MirrorWorkerEvent.model_validate_json(
                await process.stdout.readline()
            )
            assert (
                ready.event == "ready" and ready.activity and ready.activity.http_port
            )
            port = ready.activity.http_port
            async with httpx.AsyncClient(
                base_url=f"http://127.0.0.1:{port}", trust_env=False
            ) as client:
                while True:
                    response = await client.get("/" + manifest_path(identity))
                    if response.status_code == 200:
                        break
                    await asyncio.sleep(0.025)
                assert response.content == wrapper
                for member in manifest.members:
                    for file in member.files:
                        payload = await client.get("/" + object_path(file.sha256))
                        assert (
                            payload.content
                            == (
                                mirror / "store" / object_path(file.sha256)
                            ).read_bytes()
                        )
        if stop == "cancel":
            process.stdin.write(b"cancel\n")
            await process.stdin.drain()
        else:
            process.stdin.close()
        code, events = await complete(process)
        assert code == 1 and events[-1].error_code == "cancelled"
        assert any(
            event.activity
            and not event.activity.running
            and event.activity.native_workers == 0
            and event.activity.http_port is None
            for event in events
        )
        assert (
            not ledger.running()
            and verify_store(mirror / "store", identity, trust) == manifest
        )
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()


@pytest.mark.asyncio
async def test_stale_policy_refuses_before_listener_and_never_adopts_replaced_root(
    tmp_path: Path,
) -> None:
    mirror = tmp_path / "mirror"
    config = MirrorRequest(
        owner_id="library",
        operation_id="setup",
        kind="configure",
        configuration=configuration(),
    )
    assert (
        await complete(
            await launch(
                tmp_path, MirrorWorkerRequest(mirror_path=mirror, decision=config)
            )
        )
    )[0] == 0
    ledger = MirrorLedger(mirror)
    identity = ledger.read().identity["."]
    decision = MirrorRequest(
        owner_id="library",
        operation_id="stale",
        kind="run",
        policy_id="0" * 64,
        public_participation=True,
    )
    code, events = await complete(
        await launch(
            tmp_path,
            MirrorWorkerRequest(
                mirror_path=mirror, mirror_identity=identity, decision=decision
            ),
        )
    )
    assert (
        code == 1
        and all(event.event != "ready" for event in events)
        and not ledger.running()
    )
    mirror.rename(tmp_path / "original-mirror")
    assert (
        await complete(
            await launch(
                tmp_path, MirrorWorkerRequest(mirror_path=mirror, decision=config)
            )
        )
    )[0] == 0
    replacement = MirrorLedger(mirror)
    decision = decision.model_copy(
        update={"operation_id": "replaced", "policy_id": policy_id(replacement.read())}
    )
    code, events = await complete(
        await launch(
            tmp_path,
            MirrorWorkerRequest(
                mirror_path=mirror, mirror_identity=identity, decision=decision
            ),
        )
    )
    assert (
        code == 1
        and all(event.event != "ready" for event in events)
        and replacement.read().paused
    )
