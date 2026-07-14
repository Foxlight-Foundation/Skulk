"""Wire coverage for auditable node-timeout evidence."""

from pydantic import TypeAdapter

from skulk.shared.types.common import NodeId
from skulk.shared.types.events import (
    Event,
    NodeTimedOut,
    NodeTimeoutEvidence,
)


def test_node_timeout_evidence_survives_event_log_wire_round_trip() -> None:
    """The deciding signal ages remain available after event-log replay."""
    event = NodeTimedOut(
        node_id=NodeId("node-a"),
        evidence=NodeTimeoutEvidence(
            last_logged_event_age_seconds=42,
            heartbeat_age_seconds=35,
            fallback_telemetry_age_seconds=38,
            effective_age_seconds=35,
            timeout_seconds=30,
        ),
    )

    event_adapter: TypeAdapter[Event] = TypeAdapter(Event)
    restored: Event = event_adapter.validate_json(event.model_dump_json())

    assert isinstance(restored, NodeTimedOut)
    assert restored.evidence == event.evidence
