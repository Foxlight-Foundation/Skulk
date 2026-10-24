"""Fixed persistent-mirror worker; parent EOF drains owned sharing and byte work."""

from __future__ import annotations

import os
import sys
import threading
from collections.abc import Generator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Literal, final

from foxlight_model_preservation.mirror import model_action, run_mirror
from foxlight_model_preservation.mirror_intake import retain_local
from foxlight_model_preservation.mirror_models import (
    MirrorObservation,
    SourceGrant,
    Terms,
)
from foxlight_model_preservation.mirror_native import configure
from foxlight_model_preservation.mirror_prune import prune
from foxlight_model_preservation.mirror_state import (
    MirrorLedger,
    grant_restart,
    policy_id,
)
from foxlight_model_preservation.models import TrustProfile
from foxlight_model_preservation.operations import (
    OperationControl,
    Progress,
    failure_code,
)
from pydantic import BaseModel, ConfigDict, Field

from skulk.store.mirror_contracts import MirrorActivity, MirrorRequest


@final
class MirrorRetainSource(BaseModel):
    """Core-authored accepted public source, pinned before spawning this worker."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    descriptor: int = Field(
        ge=0, description="Inherited public archive directory descriptor."
    )
    device: int = Field(ge=0, description="Expected source filesystem device.")
    inode: int = Field(ge=0, description="Expected source directory inode.")
    trust: TrustProfile = Field(
        description="Previously accepted copy verification keys, not registry authority."
    )
    terms: tuple[Terms, ...] = Field(
        min_length=1,
        max_length=100,
        description="Exact every-member terms reviewed by the operator.",
    )


@final
class MirrorNativeRuntime(BaseModel):
    """Selected Core runtime and routing policy; no browser chooses executable code."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    binary: str = Field(description="Core-selected qualified Kubo executable.")
    sha256: str = Field(
        pattern=r"^[a-f0-9]{64}$",
        description="Pinned actual runtime bytes, rechecked before use.",
    )
    bootstrap: tuple[str, ...] = Field(
        max_length=8, description="Separately reviewed numeric bootstrap contacts."
    )
    allow_private: bool = Field(description="Separate private peer address permission.")


@final
class MirrorWorkerRequest(BaseModel):
    """Protected known-worker request; paths/descriptors never enter browser controls."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    worker_family: Literal["preservation-mirror"] = "preservation-mirror"
    mirror_path: Path = Field(description="Fixed Core-owned mirror directory.")
    mirror_identity: tuple[int, int] | None = Field(
        default=None,
        description="Enrolled root identity, absent only during initial offline setup.",
    )
    decision: MirrorRequest = Field(description="Immutable original operator decision.")
    source: MirrorRetainSource | None = Field(
        default=None, description="Accepted pinned source for one retain operation."
    )
    runtime: MirrorNativeRuntime | None = Field(
        default=None,
        description="Explicitly selected native runtime for configuration only.",
    )


@final
class MirrorWorkerEvent(BaseModel):
    """Bounded owned observations; internal root identities are not public responses."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    event: Literal["ready", "activity", "progress", "result", "error"]
    activity: MirrorActivity | None = None
    policy_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    mirror_identity: tuple[int, int] | None = None
    manifest_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    error_code: str | None = Field(default=None, max_length=80)
    bytes_hashed: int = Field(default=0, ge=0)
    bytes_copied: int = Field(default=0, ge=0)
    bytes_received: int = Field(default=0, ge=0)


def _emit(event: MirrorWorkerEvent) -> None:
    encoded = event.model_dump_json(exclude_none=True)
    if len(encoded.encode()) > 16384:
        raise ValueError("mirror observation exceeds bound")
    print(encoded, flush=True)


class _ExpectedMirrorLedger(MirrorLedger):
    """Check the reviewed policy inside the standalone controller's owner lock."""

    def __init__(self, path: Path, expected_policy: str | None):
        super().__init__(path)
        self._expected_policy = expected_policy

    @contextmanager
    def service_lock(self) -> Generator[None]:
        with super().service_lock():
            if (
                self._expected_policy is not None
                and policy_id(self.read()) != self._expected_policy
            ):
                raise ValueError("mirror policy changed")
            yield


def _ledger(request: MirrorWorkerRequest) -> MirrorLedger:
    decision = request.decision
    configuration = decision.configuration
    if request.mirror_identity is None and decision.kind != "configure":
        raise ValueError("mirror enrollment required")
    if request.mirror_identity is not None and decision.policy_id is None:
        raise ValueError("review current mirror policy before changes")
    if not request.mirror_path.exists():
        if configuration is None or request.mirror_identity is not None:
            raise ValueError("mirror directory unavailable")
        MirrorLedger.initialize(
            request.mirror_path,
            name=configuration.name,
            disk_bytes=configuration.disk_bytes,
            ingress_daily=configuration.ingress_daily,
            egress_daily=configuration.egress_daily,
            egress_monthly=configuration.egress_monthly,
            port=configuration.port,
        )
    ledger = _ExpectedMirrorLedger(request.mirror_path, decision.policy_id)
    state = ledger.read()
    if decision.policy_id is not None and policy_id(state) != decision.policy_id:
        raise ValueError("mirror policy changed")
    if request.mirror_identity is not None:
        if state.identity["."] != request.mirror_identity:
            raise ValueError("mirror directory identity changed")
    elif state.models or state.sources or state.restart is not None:
        # A crash before initial enrollment may leave only offline configuration.
        # Never adopt unknown model/source authority or standing grants.
        raise ValueError("unselected mirror contains authority")
    return ledger


def _configure(request: MirrorWorkerRequest, ledger: MirrorLedger) -> None:
    configuration = request.decision.configuration
    if configuration is None:
        raise ValueError("configuration required")
    if configuration.ipfs and request.runtime is None:
        raise ValueError("qualified peer runtime required")
    with ledger.service_lock():
        ledger.update(
            lambda state: state.model_copy(
                update={
                    "schema_version": "foxlight.mirror-state.v4",
                    "name": configuration.name,
                    "disk_bytes": configuration.disk_bytes,
                    "ingress_daily": configuration.ingress_daily,
                    "egress_daily": configuration.egress_daily,
                    "egress_monthly": configuration.egress_monthly,
                    "bind": "0.0.0.0"
                    if configuration.exposure == "network"
                    else "127.0.0.1",  # noqa: S104
                    "port": configuration.port,
                    "http_enabled": configuration.http,
                    "paused": True,
                    "restart": None,
                }
            )
        )
    # Native configuration retains its independently enrolled caches. Use the
    # normal ledger after our intentional policy change; any partial failure
    # leaves the role paused with no standing grant, never serving old choices.
    runtime = request.runtime if configuration.ipfs else None
    configure(
        MirrorLedger(request.mirror_path),
        ipfs_binary=runtime.binary if runtime else None,
        ipfs_sha256=runtime.sha256 if runtime else None,
        bootstrap=runtime.bootstrap if runtime else (),
        allow_private_peers=runtime.allow_private if runtime else False,
        torrent=configuration.torrent,
        max_active=configuration.max_active,
        session_seconds=configuration.session_seconds,
        session_bytes=configuration.session_bytes,
        upload_rate=configuration.upload_rate,
    )


def run_mirror_worker(request: MirrorWorkerRequest) -> int:
    """Execute one Core decision, preserving evidence and draining on Stop/EOF."""

    def progress(value: Progress) -> None:
        _emit(
            MirrorWorkerEvent(
                event="progress",
                bytes_hashed=value.bytes_hashed,
                bytes_copied=value.bytes_copied,
                bytes_received=value.bytes_received,
            )
        )

    control = OperationControl(progress, interval_seconds=0.25)

    def commands() -> None:
        while line := sys.stdin.readline(1025):
            if line == "cancel\n" or len(line) > 1024:
                break
        control.cancel()

    threading.Thread(target=commands, daemon=True).start()
    try:
        control.checkpoint()
        ledger = _ledger(request)
        decision = request.decision
        if decision.kind == "configure":
            _configure(request, ledger)
        elif decision.kind == "retain":
            source = request.source
            if source is None or decision.manifest_id is None:
                raise ValueError("accepted public source required")
            info = os.fstat(source.descriptor)
            if (info.st_dev, info.st_ino) != (source.device, source.inode):
                raise ValueError("public archive identity changed")
            os.fchdir(source.descriptor)
            retain_local(
                ledger,
                Path("."),
                decision.manifest_id,
                SourceGrant(
                    trust=source.trust, terms=source.terms, max_bytes=2**53 - 1
                ),
                control=control,
                confirm_reintake=decision.confirm_reintake,
            )
        elif decision.kind == "run":
            with ledger.service_lock():
                ledger.update(lambda state: state.model_copy(update={"paused": False}))

            def prepared(port: int | None) -> None:
                _emit(
                    MirrorWorkerEvent(
                        event="ready",
                        activity=MirrorActivity(
                            running=True, http_port=port, native_workers=0
                        ),
                    )
                )

            def observe(value: MirrorObservation) -> None:
                _emit(
                    MirrorWorkerEvent(
                        event="activity",
                        activity=MirrorActivity.model_validate_json(
                            value.model_dump_json()
                        ),
                    )
                )

            run_mirror(
                ledger,
                public=decision.mode == "finite",
                resume=decision.mode == "resume",
                seconds=decision.seconds,
                control=control,
                prepared=prepared,
                observe=observe,
            )
        elif decision.kind == "grant_resume":
            if decision.resume_expires is None:
                raise ValueError("standing expiry required")
            with ledger.service_lock():
                grant_restart(ledger, decision.resume_expires)
        elif decision.kind == "revoke_resume":
            with ledger.service_lock():
                ledger.update(lambda state: state.model_copy(update={"restart": None}))
        elif decision.kind == "model":
            if decision.manifest_id is None or decision.action is None:
                raise ValueError("model action required")
            model_action(ledger, decision.manifest_id, decision.action)
        else:
            if decision.manifest_id is None or decision.confirm_delete_manifest is None:
                raise ValueError("exact deletion identity required")
            prune(ledger, decision.manifest_id, decision.confirm_delete_manifest)
        state = ledger.read()
        _emit(
            MirrorWorkerEvent(
                event="result",
                policy_id=policy_id(state),
                mirror_identity=state.identity["."],
                manifest_id=decision.manifest_id,
            )
        )
        return 0
    except (OSError, ValueError) as error:
        with suppress(BrokenPipeError, ConnectionResetError):
            _emit(MirrorWorkerEvent(event="error", error_code=failure_code(error)))
        return 1
