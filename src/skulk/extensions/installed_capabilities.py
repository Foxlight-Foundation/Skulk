"""Read selected installed declarations separately from live capability admission."""

import json
import time
from datetime import datetime, timezone
from typing import Literal, final

from pydantic import Field

from skulk.extensions.capabilities import CapabilityDescriptor, descriptor_revision
from skulk.extensions.managed_services import (
    ManagedInstallation,
    ManagedInventory,
    ManagedServices,
)
from skulk.extensions.runtime_manager import InstalledContractRequest, InventoryRequest
from skulk.utils.pydantic_ext import FrozenModel


@final
class InstalledCapabilityObservation(FrozenModel):
    """Local installation and independently observed child availability."""

    installation: ManagedInstallation = Field(
        description="Selected installed metadata and current manager observation."
    )
    reachable: bool | None = Field(
        description="Recently observed owner transport; unknown before an owner observation. Does not establish ready capacity."
    )
    ready_capabilities: tuple[str, ...] = Field(
        description="Exact installed contracts also admitted by the current live child. A ready provisioner is not allocated inference capacity."
    )


@final
class InstalledCapabilities(FrozenModel):
    """Bounded declarations on this API host, never a cluster-wide absence proof."""

    source: Literal["/v1/plugins/managed/capabilities"] = Field(
        default="/v1/plugins/managed/capabilities",
        description="Authorized full-detail read endpoint.",
    )
    observed_at: str = Field(description="Time this manager inventory was read.")
    scope: Literal["local_api_host"] = Field(
        default="local_api_host",
        description="Only installations managed by the API host's configured local manager.",
    )
    installations: tuple[InstalledCapabilityObservation, ...] = Field(
        max_length=16,
        description="Installed declarations and separate live observations, including disabled and retained uninstalled selections.",
    )


@final
class InstalledContract(FrozenModel):
    """One exact selected contract; declaring an operation does not authorize it."""

    plugin_id: str = Field(description="Exact local installation identity.")
    selected_digest: str = Field(
        pattern=r"^[a-f0-9]{64}$",
        description="Selected staged generation at read time.",
    )
    descriptor: CapabilityDescriptor = Field(
        description="Full declared purpose, version, operation schemas and media contract."
    )
    revision: str = Field(
        pattern=r"^[a-f0-9]{16}$", description="Exact observed descriptor digest."
    )


async def read_installed_capabilities(
    services: ManagedServices,
) -> InstalledCapabilities:
    """Read configured manager metadata without enabling, repairing or invoking children.

    Failures propagate as unavailable. A successful empty local inventory makes
    no statement about other hosts or non-managed extensions.
    """
    raw = await services.read_installed_metadata(InventoryRequest())
    inventory = ManagedInventory.model_validate_json(json.dumps(raw))
    observations: list[InstalledCapabilityObservation] = []
    for installation in inventory.installations:
        owner = services.owners.get(installation.plugin_id)
        reachable = (
            None
            if owner is None or owner.observed == 0
            else (owner.available and 0 <= time.monotonic() - owner.observed < 3)
        )
        release = installation.release
        ready: list[str] = []
        if (
            owner is not None
            and reachable
            and installation.enabled
            and not installation.uninstalled
            and not installation.stale
            and installation.error_code is None
            and installation.service is not None
            and installation.service.state == "running"
            and installation.service.active_digest == installation.selected_digest
            and release is not None
            and release.registration is not None
        ):
            for summary in release.registration.capabilities:
                matching = [
                    descriptor
                    for node in owner.nodes
                    if node.bundle_id == release.bundle_id
                    and node.version == release.bundle_version
                    for descriptor in node.descriptors
                    if descriptor.qualified_id == summary.qualified_id
                    and descriptor_revision(descriptor) == summary.revision
                ]
                if len(matching) == 1 and owner.capability_ready(summary.qualified_id):
                    ready.append(summary.qualified_id)
        observations.append(
            InstalledCapabilityObservation(
                installation=installation,
                reachable=reachable,
                ready_capabilities=tuple(ready),
            )
        )
    return InstalledCapabilities(
        observed_at=datetime.now(timezone.utc).isoformat(),
        installations=tuple(observations),
    )


async def read_installed_contract(
    services: ManagedServices, request: InstalledContractRequest
) -> InstalledContract:
    """Read the exact selected declaration after its current digest is checked."""
    raw = await services.read_installed_metadata(request)
    contract = InstalledContract.model_validate_json(json.dumps(raw))
    if (
        contract.plugin_id != request.plugin_id
        or contract.descriptor.qualified_id != request.qualified_id
        or descriptor_revision(contract.descriptor) != request.revision
        or contract.revision != request.revision
    ):
        raise ValueError("installed contract differs from selected identity")
    return contract


def steward_installed_result(
    observation: InstalledCapabilities | InstalledContract, limit: int
) -> dict[str, object]:
    """Keep bounded Steward JSON structured, with explicit full-detail coverage.

    Large schemas are available only through the authorized HTTP contract read;
    they are never chopped into invalid JSON or mistaken for an empty schema.
    """
    if isinstance(observation, InstalledContract):
        payload: dict[str, object] = observation.model_dump(mode="json")
        if len(json.dumps(payload)) <= limit:
            return payload
        return {
            "plugin_id": observation.plugin_id,
            "qualified_id": observation.descriptor.qualified_id,
            "revision": observation.revision,
            "selected_digest": observation.selected_digest,
            "purpose": observation.descriptor.description[:256],
            "schemaIncluded": False,
            "reason": "Full contract exceeds the Steward result bound.",
            "full_details_endpoint": f"/v1/plugins/managed/{observation.plugin_id}/capabilities/{observation.descriptor.qualified_id}?revision={observation.revision}",
        }
    rows: list[dict[str, object]] = []
    for value in observation.installations:
        installation = value.installation
        release = installation.release
        rows.append(
            {
                "plugin_id": installation.plugin_id,
                "bundle": release.bundle_id if release else None,
                "version": release.bundle_version if release else None,
                "enabled": installation.enabled
                if installation.selected_digest is not None
                else None,
                "uninstalled": installation.uninstalled,
                "reachable": value.reachable,
                "ready_capabilities": value.ready_capabilities,
                "registration": release.registration.model_dump(mode="json")
                if release and release.registration
                else None,
            }
        )
    payload = {
        "source": observation.source,
        "observedAt": observation.observed_at,
        "scope": observation.scope,
        "installedCount": len(rows),
        "installations": rows,
        "omittedInstallations": 0,
        "note": "Declarations do not grant execution authority. A ready provisioner does not prove allocated inference capacity. Other hosts and non-managed extensions are outside this inventory.",
    }
    while len(json.dumps(payload)) > limit and rows:
        rows.pop()
        payload["omittedInstallations"] = len(observation.installations) - len(rows)
    return payload
