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
    search_sources,
    select_entry,
)
from foxlight_model_preservation.invitations import Invitation, invitation_from_envelope
from foxlight_model_preservation.models import MAX_METADATA_BYTES, PreservationError
from foxlight_model_preservation.operations import OperationControl, failure_code
from foxlight_model_preservation.storage import manifest_path
from foxlight_model_preservation.transfer import DEFAULT_POLICY, response
from pydantic import BaseModel, ConfigDict, Field
from tuf.api.exceptions import RepositoryError

from skulk.store.discovery_contracts import DiscoveryRequest
from skulk.store.preservation_contracts import Digest
from skulk.store.preservation_journal import PreservationHostError


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


def public_model_mirrors(values: tuple[str, ...]) -> tuple[str, ...]:
    """Keep bounded credential-free HTTP(S) hints; never grant local file access."""
    accepted: list[str] = []
    for value in values:
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
    mirrors = public_model_mirrors(entry.mirrors)
    if not mirrors:
        raise PreservationHostError("public_model_mirror_required")
    policy = replace(
        DEFAULT_POLICY,
        timeout=10,
        allow_http=selected.allow_http,
        allow_private=selected.allow_private,
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
            status = refresh_source(
                request.cache, request.selection.source_id, control=control
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
