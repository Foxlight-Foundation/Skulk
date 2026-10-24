"""Known metadata-only child, with no model-store, signer or runtime admission."""

from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Literal, final
from urllib.parse import urlsplit

from foxlight_model_preservation.discovery import (
    list_sources,
    refresh_source,
    refresh_source_i2p,
    search_sources,
    select_entry,
)
from foxlight_model_preservation.i2p_http import I2PHTTPProxy
from foxlight_model_preservation.invitations import invitation_from_envelope
from foxlight_model_preservation.models import MAX_METADATA_BYTES, PreservationError
from foxlight_model_preservation.operations import OperationControl, failure_code
from foxlight_model_preservation.peer import refresh_peer_source
from foxlight_model_preservation.peer_invitations import inspect_peer_model
from foxlight_model_preservation.peer_runtime import PeerClient, content_id
from foxlight_model_preservation.storage import manifest_path
from foxlight_model_preservation.transfer import (
    DEFAULT_POLICY,
    I2PTransferPolicy,
    response,
)
from pydantic import BaseModel, ConfigDict, Field, model_validator
from tuf.api.exceptions import RepositoryError

from skulk.store.discovery_contracts import DiscoveryRequest, I2PSourcePolicy
from skulk.store.preservation_contracts import Digest
from skulk.store.preservation_journal import PreservationHostError
from skulk.store.preservation_references import ModelInvitation as Invitation
from skulk.store.preservation_references import i2p_model_mirrors


@final
class DiscoveryWorkerRequest(BaseModel):
    """Owner-private fixed request; no API accepts these filesystem paths."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    worker_family: Literal["discovery"] = Field(
        default="discovery",
        description="Known metadata worker family, distinct from model-copy requests.",
    )
    cache: Path = Field(
        description="Parent-selected protected cache; never supplied through the API."
    )
    selection: DiscoveryRequest = Field(description="Exact durable metadata request.")
    i2p_policy: I2PSourcePolicy | None = Field(
        default=None,
        description="Parent-resolved exact local proxy and I2P-only metadata locators.",
    )
    peer_repository: Path | None = Field(
        default=None, description="Host-owned active receive profile, never API input."
    )
    peer_binary: str | None = Field(
        default=None, description="Parent-qualified native executable, never API input."
    )

    @model_validator(mode="after")
    def check_i2p(self) -> "DiscoveryWorkerRequest":
        """Bind protected worker routing before any metadata IO can begin."""
        policy = self.i2p_policy
        if self.selection.transport == "i2p":
            if (
                policy is None
                or policy.source_id != self.selection.source_id
                or policy.policy_id != self.selection.i2p_policy_id
            ):
                raise ValueError("I2P worker policy differs from the exact selection")
        elif policy is not None:
            raise ValueError("Other metadata transports cannot carry an I2P policy")
        return self


@final
class DiscoveryWorkerResult(BaseModel):
    """One bounded stdout result; full model metadata never enters a control response."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    outcome: Literal["succeeded", "failed"] = Field(
        description="Actual worker outcome, independently checked by its parent."
    )
    invitation: Invitation | None = Field(
        default=None,
        description="Complete bounded original signed public metadata; no weights.",
    )
    catalog_id: Digest | None = Field(
        default=None,
        description="Accepted catalog identity for source-refresh reconciliation.",
    )
    error_code: str | None = Field(
        default=None,
        description="Safe stable failure without supplier text or private paths.",
    )


def public_model_mirrors(
    values: tuple[str, ...], transport: Literal["http", "peer", "i2p"] = "http"
) -> tuple[str, ...]:
    """Keep bounded credential-free HTTP(S) hints; never grant local file access."""
    accepted: list[str] = []
    if transport == "i2p":
        return i2p_model_mirrors(values)
    for value in values:
        if transport == "peer":
            if value.startswith("ipfs://"):
                try:
                    content_id(value[7:])
                except ValueError:
                    continue
                if value not in accepted:
                    accepted.append(value)
            continue
        try:
            parsed = urlsplit(value)
            port = parsed.port
        except ValueError:
            continue
        if (
            len(value) <= 2048
            and parsed.scheme in {"http", "https"}
            and parsed.hostname
            and parsed.username is None
            and parsed.password is None
            and not parsed.query
            and not parsed.fragment
            and (port is None or 1 <= port <= 65535)
            and not any(
                ord(char) <= 32 or ord(char) == 127 or char == "\\" for char in value
            )
        ):
            normalized = value.rstrip("/")
            if normalized not in accepted:
                accepted.append(normalized)
    return tuple(accepted)


def _invitation(
    request: DiscoveryWorkerRequest, control: OperationControl
) -> Invitation:
    selected = request.selection
    if (
        selected.entry_id is None
        or selected.manifest_id is None
        or not selected.accept_source_keys
    ):
        raise PreservationHostError("source_reference_acceptance_required")
    entry = select_entry(
        request.cache, selected.source_id, selected.entry_id, selected.manifest_id
    )
    matches = search_sources(request.cache, selected.manifest_id, limit=1000)
    candidate = next(
        (
            location.description
            for match in matches
            for location in match.locations
            if location.source_id == selected.source_id
            and location.entry_id == selected.entry_id
        ),
        None,
    )
    if candidate is None or candidate.candidate_bundle_trust is None:
        raise PreservationHostError("candidate_bundle_keys_required")
    # Metadata-source grants never authorize model-supplier connections. Local
    # model paths remain standalone-only and cannot become Core filesystem grants.
    mirrors = public_model_mirrors(entry.mirrors, selected.transport)
    if not mirrors:
        raise PreservationHostError(
            "i2p_model_mirror_required"
            if selected.transport == "i2p"
            else "public_model_mirror_required"
        )
    if selected.transport == "peer":
        return inspect_peer_model(
            entry.manifest,
            candidate.candidate_bundle_trust,
            mirrors,
            _peer_client(request, control),
        )
    policy = (
        I2PTransferPolicy(proxy=_i2p_proxy(request), timeout=10)
        if selected.transport == "i2p"
        else replace(
            DEFAULT_POLICY,
            timeout=10,
            allow_http=selected.allow_http,
            allow_private=selected.allow_private,
        )
    )
    deadline = time.monotonic() + 120
    control.limit_deadline(deadline)
    remaining_bytes = 32 * 1024 * 1024
    for mirror in mirrors:
        control.checkpoint()
        try:
            encoded = bytearray()
            with response(mirror, manifest_path(entry.manifest), policy) as received:
                if received.status != 200:
                    raise PreservationError("partial model metadata refused")
                while len(encoded) <= MAX_METADATA_BYTES:
                    control.checkpoint()
                    if remaining_bytes <= 0:
                        raise PreservationHostError("model_metadata_budget_exceeded")
                    chunk = received.read1(
                        min(
                            65536,
                            MAX_METADATA_BYTES + 1 - len(encoded),
                            remaining_bytes,
                        )
                    )
                    if not chunk:
                        break
                    encoded.extend(chunk)
                    remaining_bytes -= len(chunk)
                    control.advance(received=len(chunk))
            if len(encoded) > MAX_METADATA_BYTES:
                raise PreservationError("model metadata exceeds its bound")
            invitation = invitation_from_envelope(
                bytes(encoded),
                entry.manifest,
                candidate.candidate_bundle_trust,
                mirrors,
            )
            control.checkpoint()
            return invitation
        except (OSError, PreservationError):
            # Cancellation/deadline must not be swallowed as supplier failover.
            control.checkpoint()
    raise PreservationHostError("model_metadata_unavailable")


def _i2p_proxy(request: DiscoveryWorkerRequest) -> I2PHTTPProxy:
    policy = request.i2p_policy
    if (
        policy is None
        or not request.selection.allow_i2p
        or policy.policy_id != request.selection.i2p_policy_id
        or policy.source_id != request.selection.source_id
    ):
        raise PreservationHostError("source_i2p_policy_changed")
    return I2PHTTPProxy(host=policy.proxy.host, port=policy.proxy.port)


def _peer_client(
    request: DiscoveryWorkerRequest, control: OperationControl
) -> PeerClient:
    if (
        not request.selection.allow_peer_network
        or request.peer_repository is None
        or request.peer_binary is None
    ):
        raise PreservationHostError("peer_session_required")
    return PeerClient(
        request.peer_repository,
        request.peer_binary,
        allow_network=True,
        control=control,
        seconds=120,
    )


def run_discovery_worker(request: DiscoveryWorkerRequest) -> int:
    """Run one finite metadata effect; parent pipe loss exits without orphan work.

    Source refresh retains verified floors and its accepted cache atomically.
    Model preparation returns only signed metadata, without object downloads,
    registry trust, private signing keys, code execution or a serving listener.
    """
    control = OperationControl()

    def commands() -> None:
        for line in sys.stdin:
            if line == "cancel\n":
                control.cancel()
        os._exit(75)

    threading.Thread(target=commands, daemon=True).start()
    try:
        if request.selection.kind == "refresh":
            status = (
                refresh_peer_source(
                    request.cache,
                    request.selection.source_id,
                    _peer_client(request, control),
                    historical=request.selection.historical,
                )
                if request.selection.transport == "peer"
                else refresh_source_i2p(
                    request.cache,
                    request.selection.source_id,
                    proxy=_i2p_proxy(request),
                    allow_i2p=request.selection.allow_i2p,
                    endpoints=request.i2p_policy.endpoints
                    if request.i2p_policy
                    else None,
                    control=control,
                )
                if request.selection.transport == "i2p"
                else refresh_source(
                    request.cache, request.selection.source_id, control=control
                )
            )
            result = DiscoveryWorkerResult(
                outcome="succeeded", catalog_id=status.catalog_id
            )
        else:
            # Confirm this remains a followed source before any supplier request.
            if not any(
                source.source_id == request.selection.source_id and source.enabled
                for source in list_sources(request.cache)
            ):
                raise PreservationHostError("source_no_longer_followed")
            result = DiscoveryWorkerResult(
                outcome="succeeded", invitation=_invitation(request, control)
            )
        sys.stdout.write(result.model_dump_json() + "\n")
        sys.stdout.flush()
        return 0
    except (ValueError, OSError, RepositoryError) as error:
        result = DiscoveryWorkerResult(
            outcome="failed",
            error_code=error.code
            if isinstance(error, PreservationHostError)
            else failure_code(error),
        )
        sys.stdout.write(result.model_dump_json() + "\n")
        sys.stdout.flush()
        return 1
