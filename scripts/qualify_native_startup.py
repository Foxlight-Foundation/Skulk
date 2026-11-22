"""Qualify normal Core installation and startup of a disposable signed native app.

The caller supplies an independently built release and explicit disposable trust.
All state belongs to a new work directory. No Core observation, installer,
signature reader, process launcher or native protocol implementation is replaced.
This qualifies startup on the measured platform, not live native invocation or
application-state migration.
"""

import argparse
import asyncio
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import final
from uuid import uuid4

from pydantic import JsonValue, TypeAdapter

from skulk.extensions.managed import ManagedNode
from skulk.extensions.runtime_artifacts import (
    RuntimeTrust,
    measure_host,
    verify_runtime,
)
from skulk.extensions.runtime_attachment import HostSettings
from skulk.extensions.runtime_controller import LifecycleRequest, RuntimeController
from skulk.extensions.runtime_files import (
    RuntimeLock,
    private_directory,
    read_private,
    write_private,
)
from skulk.extensions.runtime_manager import (
    InstallationRequest,
    RuntimeManager,
    SubmitRequest,
)

_OBJECT = TypeAdapter(dict[str, JsonValue])
_NODES = TypeAdapter(tuple[ManagedNode, ...])
_STRING = TypeAdapter(str)
_MAXIMUM_DESCRIPTION_BYTES = 524288


@final
@dataclass(frozen=True)
class _Observation:
    transport_node_id: str
    node: ManagedNode


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


async def _describe(root: Path) -> _Observation:
    # This is the ordinary owner management socket, whose placement is derived
    # from the absolute owner root rather than the runner's temporary directory.
    digest = hashlib.sha256(str(root.absolute()).encode()).hexdigest()[:24]
    socket = Path("/tmp") / f"skulk-control-{os.getuid()}-{digest}" / "control.sock"
    async with asyncio.timeout(2):
        reader, writer = await asyncio.open_unix_connection(
            socket, limit=_MAXIMUM_DESCRIPTION_BYTES + 1
        )
        try:
            writer.write(b'{"operation":"describe-extended"}\n')
            await writer.drain()
            raw = await reader.readline()
            _require(
                raw.endswith(b"\n") and len(raw) <= _MAXIMUM_DESCRIPTION_BYTES,
                "owner description exceeded its management bound",
            )
            reply = _OBJECT.validate_json(raw)
            _require(set(reply) == {"result"}, "owner refused description")
            description = _OBJECT.validate_python(reply["result"], strict=True)
            nodes = _NODES.validate_json(json.dumps(description["nodes"]))
            _require(len(nodes) == 1, "qualification needs one independent child")
            return _Observation(
                _STRING.validate_python(description["transport_node_id"], strict=True),
                nodes[0],
            )
        finally:
            writer.close()
            await writer.wait_closed()


async def _ready(root: Path) -> _Observation:
    async with asyncio.timeout(60):
        while True:
            try:
                observed = await _describe(root)
            except (OSError, ValueError, TimeoutError):
                # Startup and fenced restart briefly withdraw this read-only
                # socket. No lifecycle request or child call is replayed here.
                await asyncio.sleep(0.1)
                continue
            if observed.node.status == "ready":
                return observed
            await asyncio.sleep(0.1)


async def _completed(controller: RuntimeController, request: LifecycleRequest) -> None:
    work = controller.work
    if work is None:
        raise ValueError("manager did not own the requested lifecycle transition")
    await work
    _require(
        controller.operation(request.operation_id).state == "complete",
        "normal manager lifecycle transition did not complete",
    )


async def qualify(source: Path, trust_file: Path, work: Path, receipt: Path) -> None:
    """Install, activate, interrupt and disable one independently signed fixture.

    ``source`` contains the normal publisher's release.json and exact artifacts;
    ``trust_file`` grants only its disposable publisher. ``work`` must not exist.
    The ordinary Core manager/controller/service own activation and cleanup.
    ``receipt`` receives only identities and fixed check names after all checks
    pass. Retained state is a synthetic canary, never operator account material.
    """
    work = work.resolve()
    receipt = receipt.absolute()
    if work.exists() or receipt.exists():
        raise ValueError("qualification requires a new work directory and receipt")
    metadata = (source / "release.json").read_bytes()
    trust = RuntimeTrust.model_validate_json(trust_file.read_bytes())
    host = measure_host()
    runtime = verify_runtime(metadata, trust, host, now=int(time.time()))
    native = runtime.native_manifest
    if native is None:
        raise ValueError("qualification requires a native runtime")
    private_directory(work)
    binding = HostSettings(
        transport_node_id="native-qualification-transport", profile_id=uuid4().hex
    )
    write_private(work / "host.json", binding.model_dump_json().encode())
    manager = RuntimeManager(work)
    identifier = "managed.native-startup"
    checks = ["measured-core", "authenticated-native-runtime"]
    try:
        await manager.start()
        await manager.dispatch(
            InstallationRequest(action="register", plugin_id=identifier)
        )
        controller = manager.controllers[identifier]
        root = controller.root
        write_private(root / "publisher-trust.json", trust.model_dump_json().encode())
        canary = b"Synthetic retained state; no account data."
        write_private(root / "protected-state-canary", canary)
        staged = await controller.selector.installer.stage(metadata, source)
        _require(staged.state == "staged", "normal offline install did not complete")
        checks.append("normal-offline-install")

        activation = LifecycleRequest(
            operation_id=uuid4().hex,
            action="activate",
            expected_revision=0,
            runtime_digest=runtime.digest,
        )
        await manager.dispatch(SubmitRequest(plugin_id=identifier, request=activation))
        await _completed(controller, activation)
        checks.append("normal-manager-activation")
        first = await _ready(root)
        _require(
            first.transport_node_id == binding.transport_node_id,
            "owner changed the host transport identity",
        )
        _require(
            first.node.descriptors == (),
            "startup-only fixture advertised unqualified legacy capacity",
        )
        checks.extend(
            (
                "signed-owner-startup",
                "compiled-native-child-handshake-and-health",
                "legacy-capacity-not-advertised",
            )
        )
        service = controller.service
        if service is None or service.process is None:
            raise ValueError("normal service has no owned owner process")
        owner_pid = service.process.pid
        _require(owner_pid != os.getpid(), "owner must be an independent process")
        service.process.kill()
        async with asyncio.timeout(60):
            while True:
                service = controller.service
                process = service.process if service is not None else None
                if (
                    process is not None
                    and process.pid != owner_pid
                    and process.returncode is None
                ):
                    break
                await asyncio.sleep(0.1)
        second = await _ready(root)
        _require(second.node.node_id == first.node.node_id, "child identity changed")
        checks.append("abrupt-owner-death-and-fenced-restart")
        child_name = (
            "bundle-" + hashlib.sha256(native.bundle_id.encode()).hexdigest()[:24]
        )
        cleanup_marker = root / child_name / "native-cleanup.done"
        _require(not cleanup_marker.exists(), "cleanup marker predates normal disable")

        withdrawal = LifecycleRequest(
            operation_id=uuid4().hex, action="disable", expected_revision=1
        )
        await manager.dispatch(SubmitRequest(plugin_id=identifier, request=withdrawal))
        await _completed(controller, withdrawal)
        selected = controller.selector.current()
        _require(
            selected is not None and not selected.enabled and selected.revision == 2,
            "normal disable did not retain the expected selection revision",
        )
        service = controller.service
        process = service.process if service is not None else None
        _require(
            process is None or process.returncode is not None,
            "disable left the owner running",
        )
        # Acquiring the same lock proves that no inherited child still owns the
        # original owner's lifetime fence after ordinary manager withdrawal.
        RuntimeLock(root, "supervisor.lock").close()
        checks.append("disable-reaps-owner-and-descendant-fence")
        # The independent fixture's fixed, nonsecret marker uses its own file
        # mode. It is evidence inside a private root, not a credential record.
        with cleanup_marker.open("rb") as marker:
            cleanup = marker.read(9)
        _require(
            cleanup == b"complete",
            "native asynchronous cleanup did not finish before escalation",
        )
        checks.append("native-async-cleanup-finishes-before-escalation")
        _require(
            read_private(root / "protected-state-canary") == canary
            and read_private(
                root / "generations" / runtime.digest / "native-manifest.json", 262144
            )
            == native.canonical_json.encode(),
            "canonical manifest or synthetic retained state changed",
        )
        checks.append("canonical-manifest-and-synthetic-state-retained")

        # Recheck the original authority now, so an expired/incompatible fixture
        # cannot masquerade as a successful revocation refusal.
        verify_runtime(metadata, trust, measure_host(), now=int(time.time()))
        previous_work = controller.work
        revoked = trust.model_copy(
            update={
                "revision": trust.revision + 1,
                "revoked_artifacts": (runtime.digest,),
            }
        )
        write_private(root / "publisher-trust.json", revoked.model_dump_json().encode())
        try:
            await manager.dispatch(
                SubmitRequest(
                    plugin_id=identifier,
                    request=LifecycleRequest(
                        operation_id=uuid4().hex,
                        action="activate",
                        expected_revision=2,
                        runtime_digest=runtime.digest,
                    ),
                )
            )
        except ValueError as error:
            _require(
                str(error) == "runtime compatibility or revocation refused"
                and controller.work is previous_work
                and controller.selector.current() == selected,
                "revocation was not refused before accepting lifecycle work",
            )
        else:
            raise ValueError("revoked runtime was readmitted")
        checks.append("revoked-runtime-refuses-before-relaunch")
    finally:
        await manager.close()
    _require(measure_host() == host, "measured Core host changed")
    checks.append("measured-host-unchanged")
    result = {
        "format": 1,
        "passed": True,
        "checks": checks,
        "runtime_digest": runtime.digest,
        "manifest_digest": native.digest,
        "runtime_protocol": runtime.claims.protocol,
        "host": {
            "platform": host.platform,
            "python_version": host.python_version,
            "skulk_version": host.skulk_version,
            "skulk_build_sha256": host.skulk_build_sha256,
        },
        "scope": (
            "Account-free normal Core manager/installer/service to signed owner and "
            "native child on the measured platform. Synthetic retained state only; "
            "no live native Fabric invocation, real application-state migration, "
            "full node bootstrap or packaged application qualification."
        ),
    }
    # This payload-free observation may be retained in a shared evidence folder;
    # runtime credential-directory modes do not apply to that evidence folder.
    with receipt.open("x", encoding="utf-8") as output:
        output.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps({"passed": True, "platform": host.platform, "checks": len(checks)})
    )


@final
class _Arguments(argparse.Namespace):
    source: Path
    trust: Path
    work: Path
    receipt: Path


def main() -> None:
    """Run the signed native startup rehearsal in a new disposable owner root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--trust", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    arguments = _Arguments()
    _ = parser.parse_args(namespace=arguments)
    asyncio.run(
        qualify(arguments.source, arguments.trust, arguments.work, arguments.receipt)
    )


if __name__ == "__main__":
    main()
