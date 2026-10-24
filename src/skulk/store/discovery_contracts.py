"""Typed metadata controls, separate from model-copy and runtime authority."""

import hashlib
from typing import Literal, final

from foxlight_model_preservation.discovery_models import SourceStatus
from foxlight_model_preservation.i2p_http import i2p_http_endpoint
from pydantic import Field, model_validator

from skulk.store.preservation_contracts import (
    Digest,
    Handle,
    I2PProxySettings,
    JobState,
    OperationId,
    PreservationContract,
    omit_peer_default,
)


@final
class I2PSourcePolicy(PreservationContract):
    """Private receiver connection policy; original signed source bytes stay intact."""

    schema_version: Literal["skulk.i2p-source-policy.v1"] = Field(
        default="skulk.i2p-source-policy.v1",
        description="Local policy version, not a source signing domain.",
    )
    source_id: Digest = Field(description="Exact initial-root source identity.")
    proxy: I2PProxySettings = Field(description="Explicit numeric loopback proxy.")
    endpoints: tuple[str, ...] = Field(
        min_length=1,
        max_length=8,
        description="I2P-only locators; no direct fallback or source authority is granted.",
    )

    @model_validator(mode="after")
    def check_endpoints(self) -> "I2PSourcePolicy":
        """Reject unsupported or mixed routes offline, before any connection."""
        for endpoint in self.endpoints:
            if len(endpoint) > 2048:
                raise ValueError("I2P metadata endpoint exceeds its bound")
            i2p_http_endpoint(endpoint)
        return self

    @property
    def policy_id(self) -> str:
        """Bind the exact local proxy/source/locator policy without network effects."""
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()


@final
class SourceReviewRequest(PreservationContract):
    """Inspect one completed source-purpose upload without any connection."""

    upload_id: Handle = Field(description="Opaque completed bootstrap upload.")


@final
class SourceReview(PreservationContract):
    """An expiring first-contact decision; root bytes remain in private custody."""

    review_id: Digest = Field(description="Exact private review to confirm.")
    source_id: Digest = Field(description="Stable initial-root identity.")
    bootstrap_id: Digest = Field(
        description="Original source/peer identity, unchanged inside an unsigned routing handoff."
    )
    name: str = Field(max_length=200, description="Source-supplied display name.")
    endpoints: tuple[str, ...] = Field(
        max_length=8,
        description="Reviewed HTTP(S) metadata endpoints; none have been contacted.",
    )
    contacts: tuple[str, ...] = Field(
        default=(),
        max_length=8,
        exclude_if=omit_peer_default,
        description="Untrusted handoff starting hints; no automatic Core configuration or network permission.",
    )
    transport: Literal["http", "peer", "i2p"] = Field(
        default="http",
        exclude_if=omit_peer_default,
        description="Reviewed source transport, without connection permission.",
    )
    ipns_name: str | None = Field(
        default=None,
        exclude_if=omit_peer_default,
        max_length=200,
        description="Pinned public source name for a peer file.",
    )
    snapshot_count: int = Field(
        default=0,
        exclude_if=omit_peer_default,
        ge=0,
        le=8,
        description="Trusted immutable historical fallback count.",
    )
    expires_at: int = Field(description="Review expiry in Unix seconds.")


@final
class SourceConfirmation(PreservationContract):
    """Trust only the reviewed discovery file, with local connection grants."""

    review_id: Digest = Field(description="Exact live source review.")
    confirmed: bool = Field(
        description="Operator trusts the supplier of this file for discovery metadata only."
    )
    allow_http: bool = Field(
        default=False, description="Permit plaintext metadata connections only."
    )
    allow_private: bool = Field(
        default=False,
        description="Permit metadata connections to non-global addresses only.",
    )

    allow_i2p: bool = Field(
        default=False,
        exclude_if=omit_peer_default,
        description="Follow using only I2P metadata connections; no model or sharing grant.",
    )
    i2p_proxy: I2PProxySettings | None = Field(
        default=None,
        exclude_if=omit_peer_default,
        description="Explicit numeric proxy on the Core host, not the browser device.",
    )
    i2p_endpoints: tuple[str, ...] = Field(
        default=(),
        max_length=8,
        strict=False,
        exclude_if=omit_peer_default,
        description="Optional I2P-only locators; absent uses only supported original source hints.",
    )

    @model_validator(mode="after")
    def check_i2p(self) -> "SourceConfirmation":
        """Keep explicit I2P authority separate from every direct/peer grant."""
        if self.allow_i2p:
            if self.i2p_proxy is None or self.allow_http or self.allow_private:
                raise ValueError(
                    "I2P following requires its proxy and separate consent"
                )
        elif self.i2p_proxy is not None or self.i2p_endpoints:
            raise ValueError("I2P proxy choices require explicit I2P permission")
        return self


@final
class DiscoveryRequest(PreservationContract):
    """One explicitly requested finite metadata operation with permanent replay identity."""

    owner_id: Handle = Field(
        description="Private client namespace; not a source authority."
    )
    operation_id: OperationId = Field(
        description="Retained exact request replay identity."
    )
    kind: Literal["refresh", "model_metadata"] = Field(
        description="Source update or signed model-metadata handoff; neither fetches weights."
    )
    source_id: Digest = Field(description="Exact followed source.")
    entry_id: str | None = Field(
        default=None,
        max_length=512,
        description="Exact selected catalog entry for model metadata.",
    )
    manifest_id: Digest | None = Field(
        default=None,
        description="Exact selected model manifest, never a mutable alias.",
    )
    accept_source_keys: bool = Field(
        default=False,
        description="Use this source's authenticated model reference and candidate keys for this copy only; no registry or global trust.",
    )
    allow_http: bool = Field(
        default=False,
        description="Separate plaintext permission for model-metadata suppliers.",
    )
    allow_private: bool = Field(
        default=False,
        description="Separate private-address permission for model-metadata suppliers.",
    )

    transport: Literal["http", "peer", "i2p"] = Field(
        default="http",
        exclude_if=omit_peer_default,
        description="Explicit transport; no automatic direct-network fallback.",
    )
    allow_peer_network: bool = Field(
        default=False,
        exclude_if=omit_peer_default,
        description="Permit public peer participation for this exact finite operation.",
    )
    peer_setup_id: Digest | None = Field(
        default=None,
        exclude_if=omit_peer_default,
        description="Exact separately reviewed Core peer settings.",
    )
    historical: bool = Field(
        default=False,
        exclude_if=omit_peer_default,
        description="Explicitly recover only a trusted pinned peer snapshot; never fresh-update authority.",
    )

    allow_i2p: bool = Field(
        default=False,
        exclude_if=omit_peer_default,
        description="Permit only I2P for this exact finite metadata operation.",
    )
    i2p_policy_id: Digest | None = Field(
        default=None,
        exclude_if=omit_peer_default,
        description="Exact separately confirmed source proxy/locator policy; drift refuses.",
    )

    @model_validator(mode="after")
    def check_selection(self) -> "DiscoveryRequest":
        """Keep refresh grants separate from exact model-reference acceptance."""
        if self.transport == "i2p":
            if (
                not self.allow_i2p
                or self.i2p_policy_id is None
                or self.allow_http
                or self.allow_private
                or self.allow_peer_network
                or self.peer_setup_id is not None
                or self.historical
            ):
                raise ValueError(
                    "I2P operation requires its exact policy and separate consent"
                )
        elif self.allow_i2p or self.i2p_policy_id is not None:
            raise ValueError("Other transports cannot carry I2P grants")
        if self.transport == "peer":
            if (
                not self.allow_peer_network
                or self.peer_setup_id is None
                or self.allow_http
                or self.allow_private
            ):
                raise ValueError(
                    "peer operation needs its own exact consent and settings"
                )
        elif (
            self.allow_peer_network or self.peer_setup_id is not None or self.historical
        ):
            raise ValueError("HTTP requests cannot carry peer grants")
        if self.historical and self.kind != "refresh":
            raise ValueError("historical peer policy is a source recovery choice")
        if self.kind == "model_metadata":
            if (
                not self.entry_id
                or self.manifest_id is None
                or not self.accept_source_keys
            ):
                raise ValueError("exact source model-reference acceptance required")
        elif (
            self.entry_id is not None
            or self.manifest_id is not None
            or self.accept_source_keys
            or self.allow_http
            or self.allow_private
        ):
            raise ValueError("refresh uses enrolled metadata policy only")
        return self


@final
class DiscoveryJob(PreservationContract):
    """Durable metadata activity without fabricated model/archive/key fields."""

    schema_version: Literal["skulk.discovery-job.v1"] = Field(
        default="skulk.discovery-job.v1",
        description="Typed durable metadata receipt version.",
    )
    job_id: Digest = Field(description="Opaque durable metadata job.")
    owner_id: Handle = Field(
        description="Client namespace retained across Library restarts."
    )
    operation_id: OperationId = Field(description="Original exact operation identity.")
    kind: Literal["refresh", "model_metadata"] = Field(
        description="Metadata operation family."
    )
    source_id: Digest = Field(description="Selected followed source.")
    manifest_id: Digest | None = Field(
        default=None, description="Selected exact model, only for a model-metadata job."
    )
    state: JobState = Field(
        description="Observed owned-worker state; restart never replays it."
    )
    created_at: int = Field(description="Creation time in Unix seconds.")
    updated_at: int = Field(description="Last state change in Unix seconds.")
    error_code: str | None = Field(
        default=None,
        max_length=80,
        description="Safe bounded refusal, without supplier text or private paths.",
    )
    upload_id: Handle | None = Field(
        default=None,
        description="Validated invitation upload from successful model metadata; not downloaded weights.",
    )


@final
class DiscoverySource(SourceStatus):
    """Bounded status projection; original bootstrap and diagnostics remain cached."""

    transport: Literal["http", "peer", "i2p"] = Field(
        default="http",
        exclude_if=omit_peer_default,
        description="Locally followed transport, never an implicit network action.",
    )

    snapshot_count: int = Field(
        default=0,
        exclude_if=omit_peer_default,
        ge=0,
        le=8,
        description="Independently pinned historical peer snapshots available for an explicit recovery choice.",
    )
    name: str = Field(
        max_length=200,
        description="Source display projection; long escaped text may end in an ellipsis.",
    )
    last_error: str | None = Field(
        description="Bounded safe display diagnostic; original status remains retained locally."
    )

    i2p_policy_id: Digest | None = Field(
        default=None,
        exclude_if=omit_peer_default,
        description="Current exact local I2P policy, if selected.",
    )
    i2p_proxy: I2PProxySettings | None = Field(
        default=None,
        exclude_if=omit_peer_default,
        description="Configured Core-local proxy; no liveness or anonymity assertion.",
    )


@final
class DiscoveryPage(PreservationContract):
    """Local cached source states and one bounded page of metadata jobs."""

    i2p_available: Literal[True] = Field(
        default=True,
        description="This owner implements explicit I2P HTTP jobs; not runtime proxy health or privacy qualification.",
    )
    sources: tuple[DiscoverySource, ...] = Field(
        max_length=16,
        description="Retained sources, including disabled and invalid caches; no online probes.",
    )
    jobs: tuple[DiscoveryJob, ...] = Field(
        max_length=4,
        description="Newest metadata jobs in the requested client namespace.",
    )
    next_offset: int | None = Field(description="Next metadata-job page, or exhausted.")


@final
class DiscoveryQuery(PreservationContract):
    """Local-only query kept out of URLs, referrers and supplier requests."""

    query: str = Field(
        default="",
        max_length=500,
        description="Cached model search terms or an exact manifest identity.",
    )
    offset: int = Field(
        default=0, ge=0, le=1000, description="Exact-model result offset."
    )
    supplier_offset: int = Field(
        default=0, ge=0, le=16000, description="Per-model attributed supplier offset."
    )


@final
class DiscoverySupplier(PreservationContract):
    """One attributed statement; descriptions never amend signed model cards."""

    source_id: Digest = Field(description="Exact followed source.")
    source_name: str = Field(max_length=200, description="Source display name.")
    entry_id: str = Field(max_length=512, description="Exact source entry.")
    name: str = Field(
        max_length=200,
        description="Bounded display projection; original metadata remains retained.",
    )
    summary: str = Field(
        max_length=800, description="Bounded source description projection."
    )
    freshness: str = Field(
        max_length=32,
        description="Source age/authority state, not model or registry trust.",
    )
    transport: Literal["http", "peer", "i2p"] = Field(
        default="http",
        exclude_if=omit_peer_default,
        description="Selected reference transport; source statements stay separate.",
    )
    model_keys_available: bool = Field(
        description="Source includes candidate bundle keys for explicit selected-copy review."
    )


@final
class DiscoveryResult(PreservationContract):
    """One exact manifest with separate, paginated supplier statements."""

    manifest_id: Digest = Field(description="Exact immutable bundle payload identity.")
    suppliers: tuple[DiscoverySupplier, ...] = Field(
        max_length=4,
        description="Attributed supplier choices; no descriptions are merged.",
    )
    supplier_count: int = Field(
        ge=1, description="All known aliases/suppliers for this exact identity."
    )
    next_supplier_offset: int | None = Field(
        description="Next supplier page for an exact-manifest query."
    )


@final
class DiscoveryResults(PreservationContract):
    """Bounded cached matches; reading never updates a source or contacts mirrors."""

    results: tuple[DiscoveryResult, ...] = Field(
        max_length=4, description="Exact model identities on this page."
    )
    next_offset: int | None = Field(description="Next exact-model page, or exhausted.")
