"""Journal local transport renewal without changing durable plugin identities."""

from contextlib import ExitStack
from ipaddress import ip_address
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from skulk.extensions.runtime_artifacts import Digest, Identifier
from skulk.extensions.runtime_files import RuntimeLock, read_private, write_private

ProfileIdentifier = Annotated[str, Field(pattern=r"^[a-f0-9]{32}$")]
InstallationIdentifier = Annotated[
    str, Field(pattern=r"^managed\.[a-z0-9][a-z0-9._-]{0,80}$")
]


def serve_address(value: str) -> str:
    """Refuse a serve address that is not one routable unicast IP address."""
    try:
        address = ip_address(value)
    except ValueError:
        raise ValueError("a serve address is one IP address") from None
    if address.is_loopback or address.is_unspecified or address.is_multicast:
        raise ValueError("a serve address is a routable unicast address")
    return value


ServeAddress = Annotated[str, AfterValidator(serve_address)]
"""An address the host lets installed capabilities serve on besides loopback."""


def _omit_absent(value: str | None) -> bool:
    return value is None


class ServeBinding(BaseModel):
    """Where an installation's capabilities may serve besides loopback.

    Written as ``serve.json`` beside each installation's ``owner.json``, and at
    the manager root to record the address the installations last received.
    A separate file on purpose: owners parse ``owner.json`` strictly, so an
    owner built before this setting would refuse a new field there, while it
    never reads this file.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    serve_host: ServeAddress | None = Field(
        default=None,
        description="The node's Tailscale address; absent without Tailscale.",
    )


def read_serve_host(path: Path) -> str | None:
    """The serve address recorded at ``path``; none when absent or unreadable."""
    try:
        return ServeBinding.model_validate_json(read_private(path)).serve_host
    except (OSError, ValueError):
        return None


def publish_serve_host(
    root: Path, installations: tuple[str, ...], serve_host: str
) -> None:
    """Give every stopped installation the serve address, then record it at the root.

    The caller owns the host manager lock and has stopped every owner; the
    per-installation fences refuse the write while any of them survives.
    Owners read the address once at start. The root record is written last,
    so an interrupted write leaves it naming the previous address and the
    next attachment writes every installation again.
    """
    binding = ServeBinding(serve_host=serve_host).model_dump_json().encode()
    with ExitStack() as locks:
        for identifier in installations:
            installation = root / "installations" / identifier
            for name in ("manager.lock", "service.lock", "supervisor.lock"):
                locks.callback(RuntimeLock(installation, name).close)
        for identifier in installations:
            write_private(root / "installations" / identifier / "serve.json", binding)
        write_private(root / "serve.json", binding)


class OwnerBinding(BaseModel):
    """Private owner's current transport attachment, separate from its own identity."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    transport_node_id: Identifier = Field(
        description="Current local Skulk transport ID."
    )


class HostSettings(OwnerBinding):
    """Local service profile and its most recently attached Skulk transport ID."""

    profile_id: ProfileIdentifier | None = Field(
        default=None,
        description="Locally generated profile ID; legacy unset profiles cannot renew automatically.",
    )

    def owner_binding(self) -> OwnerBinding:
        """Project only transport metadata into the private owner's fixed contract."""
        return OwnerBinding(transport_node_id=self.transport_node_id)


class AttachmentRequest(BaseModel):
    """Internal owner-local attachment renewal; never a remote administration verb."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    action: Literal["attach"] = "attach"
    profile_id: ProfileIdentifier = Field(
        description="Exact locally provisioned profile."
    )
    transport_node_id: Identifier = Field(
        description="Calling Skulk process's actual node ID."
    )
    skulk_build_sha256: Digest = Field(
        description="Build measured in the live Skulk process."
    )
    serve_host: ServeAddress | None = Field(
        default=None,
        exclude_if=_omit_absent,
        description=(
            "The node's Tailscale address as the live Skulk process last saw "
            "it. Absent when it has seen none yet: the manager keeps the "
            "address its installations already have."
        ),
    )


class AttachmentJournal(BaseModel):
    """Exact stopped-owner metadata transition, safe to finish after process death."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    previous: HostSettings = Field(description="Previously accepted local attachment.")
    current: HostSettings = Field(description="New attachment within the same profile.")
    installations: tuple[InstallationIdentifier, ...] = Field(
        max_length=16, description="Exact installations included before publication."
    )
    state: Literal["pending", "complete"] = Field(
        description="Durable local write progress."
    )

    @model_validator(mode="after")
    def same_profile(self) -> Self:
        """Reject foreign profiles and ambiguous installation membership."""
        if (
            self.previous.profile_id is None
            or self.previous.profile_id != self.current.profile_id
            or len(set(self.installations)) != len(self.installations)
        ):
            raise ValueError("attachment profile or membership differs")
        return self


def finish_attachment(root: Path, journal: AttachmentJournal) -> HostSettings:
    """Finish only recorded local metadata while every affected owner is stopped.

    The caller owns the host manager lock. Per-installation controller, service
    and child fences prevent a surviving process from reading a half transition.
    Existing foreign bindings are refused before any file is replaced. No plugin
    identity, selection, configuration, receipt or approval record is rewritten.
    """
    with ExitStack() as locks:
        for identifier in journal.installations:
            installation = root / "installations" / identifier
            for name in ("manager.lock", "service.lock", "supervisor.lock"):
                locks.callback(RuntimeLock(installation, name).close)
        host = HostSettings.model_validate_json(read_private(root / "host.json"))
        if host not in (journal.previous, journal.current):
            raise ValueError("attachment host changed")
        for identifier in journal.installations:
            owner = OwnerBinding.model_validate_json(
                read_private(root / "installations" / identifier / "owner.json")
            )
            if owner not in (
                journal.previous.owner_binding(),
                journal.current.owner_binding(),
            ):
                raise ValueError("attachment installation changed")
        write_private(root / "attachment.json", journal.model_dump_json().encode())
        for identifier in journal.installations:
            write_private(
                root / "installations" / identifier / "owner.json",
                journal.current.owner_binding().model_dump_json().encode(),
            )
        write_private(root / "host.json", journal.current.model_dump_json().encode())
        complete = journal.model_copy(update={"state": "complete"})
        write_private(root / "attachment.json", complete.model_dump_json().encode())
    return journal.current


def recover_attachment(root: Path) -> HostSettings:
    """Recover an interrupted local attachment before starting any plugin owner."""
    try:
        journal = AttachmentJournal.model_validate_json(
            read_private(root / "attachment.json")
        )
    except FileNotFoundError:
        journal = None
    if journal is not None and journal.state == "pending":
        return finish_attachment(root, journal)
    return HostSettings.model_validate_json(read_private(root / "host.json"))


class ServiceConnection(BaseModel):
    """Automatically provisioned local API connection to a fixed service profile."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    manager_root: str = Field(
        min_length=1, max_length=4096, description="Absolute local service-state root."
    )
    profile_id: ProfileIdentifier = Field(
        description="Locally generated service profile ID."
    )

    @model_validator(mode="after")
    def absolute_root(self) -> Self:
        """Refuse relative storage that could depend on an API working directory."""
        if not Path(self.manager_root).is_absolute():
            raise ValueError("service connection root must be absolute")
        return self
