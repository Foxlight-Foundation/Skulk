"""Steward-owned observations and deterministic answers over existing state reads.

Read time is not telemetry time. These projections never manufacture liveness,
provider inventory, or backend support from missing evidence.
"""

import re
from datetime import datetime, timezone
from typing import Literal, cast, final

from pydantic import BaseModel, ConfigDict, Field


@final
class StewardObservations(BaseModel):
    """Immutable counts from one complete API snapshot, before prompt compaction."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    source: Literal["/state"] = "/state"
    read_at: datetime = Field(description="UTC API read time, not telemetry freshness.")
    telemetry_observed_at: None = Field(
        default=None, description="The source does not expose download sample times."
    )
    node_scope: Literal["topology transport peers"] = "topology transport peers"
    node_count: int | None = Field(
        default=None, ge=0, description="Null if topology is incomplete."
    )
    download_scope: Literal["node staging records"] = "node staging records"
    queued: int | None = Field(
        default=None, ge=0, description="Pending records; not transfers."
    )
    transferring: int | None = Field(
        default=None, ge=0, description="Ongoing records in this snapshot."
    )
    completed: int | None = Field(
        default=None, ge=0, description="Retained completed records, not live work."
    )
    failed: int | None = Field(
        default=None, ge=0, description="Retained failed records, not live work."
    )


def observe_state(
    payload: dict[str, object], *, read_at: datetime
) -> StewardObservations:
    """Validate source coverage and count records without using compacted detail.

    Missing or malformed sections produce unknown counts. No external effects.
    """
    topology = payload.get("topology")
    nodes = (
        cast("dict[str, object]", topology).get("nodes")
        if isinstance(topology, dict)
        else None
    )
    node_count: int | None = None
    if isinstance(nodes, list):
        entries = cast("list[object]", nodes)
        identities = {node for node in entries if isinstance(node, str) and node}
        if all(isinstance(node, str) and bool(node) for node in entries):
            node_count = len(identities)
    counts = {
        "DownloadPending": 0,
        "DownloadOngoing": 0,
        "DownloadCompleted": 0,
        "DownloadFailed": 0,
    }
    downloads = payload.get("downloads")
    complete = isinstance(downloads, dict)
    if isinstance(downloads, dict):
        for records in cast("dict[str, object]", downloads).values():
            if not isinstance(records, list):
                complete = False
                continue
            for record in cast("list[object]", records):
                if not isinstance(record, dict):
                    complete = False
                    continue
                envelope = cast("dict[str, object]", record)
                if len(envelope) != 1:
                    complete = False
                    continue
                kind = next(iter(envelope))
                if kind not in counts or not isinstance(envelope[kind], dict):
                    complete = False
                    continue
                counts[kind] += 1
    return StewardObservations(
        read_at=read_at,
        node_count=node_count,
        queued=counts["DownloadPending"] if complete else None,
        transferring=counts["DownloadOngoing"] if complete else None,
        completed=counts["DownloadCompleted"] if complete else None,
        failed=counts["DownloadFailed"] if complete else None,
    )


# Intentionally full-match a small read-only vocabulary. Compound requests,
# per-model questions, action requests and ambiguous follow-ups stay with the
# investigation harness; a substring must never swallow an operator's intent.
_PREFIX = r"(?:(?:hi|hiya|hello|hey|good morning|good afternoon|good evening)[,! ]+)?(?:please )?"
_SUFFIX = r"[?.! ]*(?:please[?.! ]*)?"
_NODE_QUERY = re.compile(
    _PREFIX + r"(?:how many (?:cluster )?nodes(?: (?:do (?:you|we) (?:currently )?have|"
    r"are (?:there|connected|in (?:the|this|your|my|our) cluster)(?: (?:right now|currently|in (?:the|your|my) cluster))?))?|"
    r"(?:what is|what's) (?:the |your |my )?(?:current )?node count|"
    r"(?:current )?node count|count (?:the |your |my )?(?:cluster )?nodes)" + _SUFFIX
)
_DOWNLOAD_QUERY = re.compile(
    _PREFIX
    + r"(?:(?:are there|do you have) (?:any )?(?:active |ongoing )?downloads(?: (?:in flight|running|right now|currently))?|"
    r"(?:what is|what's) downloading(?: (?:right now|currently))?|"
    r"how many downloads are (?:active|in flight|running)|(?:current |active )?download status)"
    + _SUFFIX
)


def factual_question(text: str) -> Literal["nodes", "downloads"] | None:
    """Select only supported standalone inventory questions; never infer actions."""
    normalized = " ".join(text.lower().replace("’", "'").split())
    if _NODE_QUERY.fullmatch(normalized):
        return "nodes"
    if _DOWNLOAD_QUERY.fullmatch(normalized):
        return "downloads"
    return None


def render_observations(
    observations: StewardObservations, topic: Literal["nodes", "downloads"]
) -> str:
    """Render supported counts from typed observations, including unknown coverage."""
    timestamp = observations.read_at.astimezone(timezone.utc).strftime("%H:%M:%S UTC")
    if topic == "nodes":
        if observations.node_count is None:
            return "I couldn't determine the node count: the state snapshot has incomplete topology."
        return (
            f"The cluster state reports {observations.node_count} nodes "
            f"(topology transport peers; read at {timestamp}). "
            "This is not a count of physical hosts, capability nodes or cloud Pods."
        )
    if observations.transferring is None:
        return "I couldn't determine download status: the state snapshot has incomplete download records."
    return (
        f"The state snapshot reports {observations.transferring} transferring and "
        f"{observations.queued} queued node-staging download records "
        f"(read at {timestamp}). Retained history: {observations.completed} completed, "
        f"{observations.failed} failed; those are not active downloads. "
        "Per-transfer telemetry timestamps are unavailable, so this does not verify "
        "that bytes are moving now. Model-store fetches are a separate inventory."
    )
