"""Telemetry that lets a cold-started cluster choose the freshest restart record."""

from datetime import datetime
from ipaddress import ip_address
from typing import cast

from pydantic import ConfigDict, Field, field_validator

from skulk.utils.pydantic_ext import TaggedModel

ADVERT_ADDRESS_LIMIT = 16
"""Most addresses one advert carries; a node with more lists the first ones."""


class RestartRecoveryAdvert(TaggedModel):
    """How fresh this node's restart recovery record is, and where to fetch it.

    Every node keeps a record of what the cluster was running. After a
    whole-cluster restart the master compares these adverts and restores from
    the record most recently confirmed current, fetching it from the node that
    holds it. Only the freshness rides telemetry; the record itself, which can
    be large, is fetched once over the node's API.
    """

    model_config = ConfigDict(frozen=True)

    written_at: datetime | None = Field(
        default=None,
        description=(
            "UTC time the record's content was last confirmed current, or None "
            "when the node has no record."
        ),
    )
    placements: int = Field(
        default=0, ge=0, description="Placements the record would bring back."
    )
    api_port: int | None = Field(
        default=None,
        ge=1,
        le=65535,
        description=(
            "Port of this node's API, which serves the record; None when the "
            "node runs without an API."
        ),
    )

    addresses: tuple[str, ...] = Field(
        default=(),
        max_length=ADVERT_ADDRESS_LIMIT,
        description=(
            "This node's own reachable IP addresses, so its record can be "
            "fetched even when it publishes no interface list (a node without "
            "a worker). Loopback and link-local IPv6 are left out."
        ),
    )

    @field_validator("addresses", mode="before")
    @classmethod
    def _coerce_addresses(cls, value: object) -> object:
        """Restore tuples from JSON arrays and require IP literals."""
        if not isinstance(value, (list, tuple)):
            return value
        items = tuple(cast("list[object] | tuple[object, ...]", value))
        for item in items:
            if not isinstance(item, str):
                raise ValueError("addresses must be strings")
            ip_address(item)
        return items

    @field_validator("written_at", mode="before")
    @classmethod
    def _coerce_datetime(cls, value: object) -> object:
        """Restore strict datetimes from JSON wire values, requiring UTC offsets.

        The master orders adverts against offset-aware record stamps, and a
        naive time cannot be compared with them; it is rejected here rather
        than failing later in the planning loop.
        """
        parsed = datetime.fromisoformat(value) if isinstance(value, str) else value
        if isinstance(parsed, datetime) and (
            parsed.tzinfo is None or parsed.utcoffset() is None
        ):
            raise ValueError("written_at must carry a UTC offset")
        return parsed
