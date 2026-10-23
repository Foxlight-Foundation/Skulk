"""Explicit local peer settings; imports cannot provide executable or file paths."""

from typing import Annotated, Literal, final

from foxlight_model_preservation.peer_runtime import bootstrap_address
from pydantic import Field, model_validator

from skulk.store.preservation_contracts import Digest, PreservationContract


@final
class PeerSetup(PreservationContract):
    """Review the Core's public network exposure and bounded routing contacts."""

    bootstrap: tuple[Annotated[str, Field(max_length=512)], ...] = Field(
        strict=False,
        min_length=1,
        max_length=8,
        description="Numeric libp2p routing contacts, never model or source authority.",
    )
    allow_private: bool = Field(
        default=False,
        description="Permit LAN/private peer connections in a separate receiving profile.",
    )
    confirmed: bool = Field(
        description="Operator understands peers can observe Core network address and requested content identifiers."
    )

    @model_validator(mode="after")
    def check_contacts(self) -> "PeerSetup":
        """Validate only contacts explicitly entered by the local operator."""
        if not self.confirmed or len(set(self.bootstrap)) != len(self.bootstrap):
            raise ValueError("peer setup needs explicit consent and unique contacts")
        for contact in self.bootstrap:
            bootstrap_address(contact, self.allow_private)
        return self


@final
class PeerState(PreservationContract):
    """Offline setup/runtime observation; reading never starts native networking."""

    runtime: Literal["available", "missing", "incompatible"] = Field(
        description="Qualified installed Kubo observation, without a private executable path."
    )
    setup_id: Digest | None = Field(
        default=None, description="Exact locally reviewed connection policy."
    )
    bootstrap: tuple[str, ...] = Field(
        default=(), max_length=8, description="Operator-selected routing hints."
    )
    allow_private: bool = Field(
        default=False,
        description="Whether selected peer scope permits private destinations.",
    )
    active: bool = Field(
        default=False, description="An actually owned finite peer session is running."
    )
