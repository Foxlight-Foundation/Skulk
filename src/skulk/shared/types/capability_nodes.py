"""Operator-visible summaries of the capability nodes a host runs.

A capability node is a managed extension child (a plugin bundle installed on
one Skulk host) that may expose its own user interface. The dashboard shows
each one as a satellite of its host in the topology graph and opens its
surfaces and actions from a flyout. This module holds the bounded, credential
free summary a host gossips for that purpose; the host's extension surface
publishes it, the telemetry plane carries it, and ``GET /state`` projects it.

Nothing here is a control-plane record. A summary is a last-write-wins
telemetry reading and disappears with its host.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Literal, cast
from urllib.parse import parse_qsl, urlsplit

from pydantic import Field, JsonValue, field_validator, model_validator

from skulk.utils.pydantic_ext import FrozenModel

MAX_CAPABILITY_NODES_PER_HOST = 16
"""Upper bound on summaries one host publishes in a single reading."""

MAX_CAPABILITY_NODE_SURFACES = 4
"""Upper bound on surfaces one capability node advertises."""

MAX_CAPABILITY_NODE_ACTIONS = 8
"""Upper bound on actions one capability node advertises."""

MAX_ACTION_PAYLOAD_BYTES = 4096
"""Upper bound on a descriptor action's serialized fixed payload."""

MAX_SURFACE_URL_LENGTH = 2048
"""Upper bound on a surface or link URL; longer values are refused."""

_MAX_TEXT_LENGTH = 200

_CREDENTIAL_QUERY_NAMES = frozenset(
    {
        "token",
        "access_token",
        "id_token",
        "refresh_token",
        "auth",
        "authorization",
        "bearer",
        "key",
        "apikey",
        "api_key",
        "secret",
        "password",
        "passwd",
        "pwd",
        "credential",
        "credentials",
        "signature",
        "sig",
        "session",
        "sessionid",
        "session_id",
    }
)

CapabilityNodeStatus = Literal[
    "installed",
    "starting",
    "ready",
    "degraded",
    "disabled",
    "configuration_invalid",
    "failed",
]
"""Lifecycle status of a capability node as its owner reports it."""

CapabilityNodeActionKind = Literal["surface", "descriptor", "link"]
"""What a flyout action does: open a surface, call a capability, or open a link."""


def _validate_public_url(url: str) -> str:
    """Accept only bounded, absolute http(s) URLs without embedded credentials.

    The URL is opened by an operator's browser from the dashboard, so it must
    be something a browser can follow on its own. Userinfo is refused because
    a summary is gossiped to every node and must never carry a credential; the
    same goes for query parameters whose names announce a credential (token,
    key, signature, and the like). That screen is a guard against the obvious
    mistake, not proof of absence: a surface that needs an authenticated URL
    must authenticate in its own page instead. Length is capped so a signed or
    padded URL cannot inflate the reading.
    """
    if len(url) > MAX_SURFACE_URL_LENGTH:
        raise ValueError(
            f"surface and link URLs are limited to {MAX_SURFACE_URL_LENGTH} characters"
        )
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ValueError("surface and link URLs must be absolute http or https URLs")
    if "@" in parts.netloc:
        raise ValueError("surface and link URLs must not embed credentials")
    for name, _value in parse_qsl(parts.query, keep_blank_values=True):
        if name.lower() in _CREDENTIAL_QUERY_NAMES:
            raise ValueError(
                f"surface and link URLs must not carry credential-like query "
                f"parameters ({name!r})"
            )
    return url


def _validate_key_segment(value: str) -> str:
    """Identifiers that form the host-local ``plugin_id/node_id`` key.

    A slash is refused so the joined key stays injective: otherwise
    ``("a/b", "c")`` and ``("a", "b/c")`` would publish and withdraw each
    other's summary.
    """
    _validate_identifier(value)
    if "/" in value:
        raise ValueError("plugin_id and node_id must not contain '/'")
    return value


def _validate_identifier(value: str) -> str:
    stripped = value.strip()
    if not stripped or len(stripped) > _MAX_TEXT_LENGTH or stripped != value:
        raise ValueError(
            "identifiers and titles must be non-empty, carry no leading or "
            f"trailing whitespace, and be at most {_MAX_TEXT_LENGTH} characters"
        )
    return value


class CapabilityNodeSurface(FrozenModel):
    """One user-facing surface a capability node exposes.

    Only ``link`` surfaces are carried today: the dashboard opens ``url`` in a
    new tab. Proxied surfaces (rendered inside the dashboard) are a later
    contract and will add a kind without changing this shape.
    """

    surface_id: str
    """Stable identifier of the surface within its node."""
    title: str
    """Human-readable label shown in the flyout."""
    kind: Literal["link"] = "link"
    """How the dashboard opens the surface."""
    url: str
    """Absolute http(s) URL the browser opens; never carries credentials."""
    ready: bool = True
    """Whether the surface currently answers; unready surfaces render muted."""

    @field_validator("surface_id", "title")
    @classmethod
    def _check_text(cls, value: str) -> str:
        return _validate_identifier(value)

    @field_validator("url")
    @classmethod
    def _check_url(cls, value: str) -> str:
        return _validate_public_url(value)


class CapabilityNodeAction(FrozenModel):
    """One top-level action the flyout offers for a capability node.

    A ``surface`` action opens one of the node's surfaces, a ``link`` action
    opens an external URL, and a ``descriptor`` action performs a fixed unary
    capability call on the host through ``POST /v1/capabilities/call``.
    """

    action_id: str
    """Stable identifier of the action within its node."""
    title: str
    """Human-readable label shown in the flyout."""
    kind: CapabilityNodeActionKind
    """Which of the three action shapes this is."""
    surface_id: str | None = None
    """Target surface for ``surface`` actions."""
    capability_id: str | None = None
    """Capability invoked by ``descriptor`` actions."""
    payload: dict[str, JsonValue] | None = None
    """Fixed input for ``descriptor`` actions, at most 4 KiB serialized."""
    url: str | None = None
    """Target of ``link`` actions."""

    @field_validator("action_id", "title")
    @classmethod
    def _check_text(cls, value: str) -> str:
        return _validate_identifier(value)

    @field_validator("url")
    @classmethod
    def _check_url(cls, value: str | None) -> str | None:
        return None if value is None else _validate_public_url(value)

    @model_validator(mode="after")
    def _check_shape(self) -> CapabilityNodeAction:
        if self.kind == "surface" and self.surface_id is None:
            raise ValueError("surface actions need a surface_id")
        if self.kind == "link" and self.url is None:
            raise ValueError("link actions need a url")
        if self.kind == "descriptor" and self.capability_id is None:
            raise ValueError("descriptor actions need a capability_id")
        # Only descriptor actions have a use for a payload; refusing it on the
        # other kinds keeps every accepted action within the byte bound rather
        # than leaving a side door into the gossiped reading.
        if self.payload is not None and self.kind != "descriptor":
            raise ValueError("only descriptor actions may carry a payload")
        if self.payload is not None:
            encoded = json.dumps(self.payload, separators=(",", ":"))
            if len(encoded.encode("utf-8")) > MAX_ACTION_PAYLOAD_BYTES:
                raise ValueError(
                    f"descriptor action payloads are limited to {MAX_ACTION_PAYLOAD_BYTES} bytes"
                )
        return self


class CapabilityNodeSummary(FrozenModel):
    """What a host says about one capability node it runs.

    Bounded and credential free by construction: it is gossiped to every node
    and rendered by every dashboard. Titles and identifiers are capped, URLs
    are validated, and the counts are what the flyout needs to draw itself
    without a second request.
    """

    plugin_id: str
    """Installed plugin the node belongs to."""
    node_id: str
    """Identifier of the node within its plugin; unique per host."""
    bundle_id: str
    """Bundle the node was installed from (for example ``foxlight.video-studio``)."""
    version: str
    """Installed bundle version."""
    title: str | None = None
    """Display name; the dashboard falls back to ``node_id``."""
    status: CapabilityNodeStatus
    """Owner-reported lifecycle status."""
    owner_available: bool
    """Whether the owning extension currently answers for the node."""
    surfaces: tuple[CapabilityNodeSurface, ...] = Field(
        default=(), max_length=MAX_CAPABILITY_NODE_SURFACES
    )
    """Surfaces the flyout can open."""
    actions: tuple[CapabilityNodeAction, ...] = Field(
        default=(), max_length=MAX_CAPABILITY_NODE_ACTIONS
    )
    """Additional top-level actions the flyout offers."""
    operations_active: int = Field(default=0, ge=0)
    """Durable operations currently running on the node."""

    @field_validator("plugin_id", "node_id")
    @classmethod
    def _check_key_segments(cls, value: str) -> str:
        return _validate_key_segment(value)

    @field_validator("bundle_id", "version")
    @classmethod
    def _check_identifiers(cls, value: str) -> str:
        return _validate_identifier(value)

    @field_validator("title")
    @classmethod
    def _check_title(cls, value: str | None) -> str | None:
        return None if value is None else _validate_identifier(value)

    @field_validator("surfaces", "actions", mode="before")
    @classmethod
    def _coerce_sequences(cls, value: object) -> object:
        # The wire path decodes JSON arrays as lists, which strict mode rejects
        # for a tuple field; coerce before strict validation so readings
        # populate over gossip the same way NodeResources.capability_conflicts
        # does.
        if isinstance(value, list):
            return tuple(cast("Iterable[object]", value))
        return value

    @model_validator(mode="after")
    def _check_unique_ids(self) -> CapabilityNodeSummary:
        surface_ids = [surface.surface_id for surface in self.surfaces]
        if len(set(surface_ids)) != len(surface_ids):
            raise ValueError("surface_id values must be unique within a node")
        action_ids = [action.action_id for action in self.actions]
        if len(set(action_ids)) != len(action_ids):
            raise ValueError("action_id values must be unique within a node")
        for action in self.actions:
            if action.kind == "surface" and action.surface_id not in surface_ids:
                raise ValueError(
                    f"action {action.action_id!r} targets unknown surface "
                    f"{action.surface_id!r}"
                )
        return self

    @property
    def key(self) -> str:
        """Host-local identity of the node: plugin plus node identifier."""
        return f"{self.plugin_id}/{self.node_id}"
