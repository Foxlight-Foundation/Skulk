"""Tests for applying master-ordered model trust decisions."""

from pydantic import TypeAdapter

from skulk.shared.apply import apply
from skulk.shared.types.events import (
    Event,
    IndexedEvent,
    ModelTrustApprovalChanged,
    StateSnapshotHydrated,
)
from skulk.shared.types.state import State


def test_apply_model_trust_approval_and_revocation() -> None:
    """Trust events should maintain one sorted immutable identity set."""

    identity_a = f"card_{'a' * 52}"
    identity_b = f"card_{'b' * 52}"
    state = State(model_trust_approved_remote_code_identities=(identity_b,))

    approved = apply(
        state,
        IndexedEvent(
            idx=0,
            event=ModelTrustApprovalChanged(
                trust_identity=identity_a,
                approved=True,
            ),
        ),
    )
    assert approved.model_trust_approved_remote_code_identities == (
        identity_a,
        identity_b,
    )

    revoked = apply(
        approved,
        IndexedEvent(
            idx=1,
            event=ModelTrustApprovalChanged(
                trust_identity=identity_b,
                approved=False,
            ),
        ),
    )
    assert revoked.model_trust_approved_remote_code_identities == (identity_a,)
    assert revoked.last_event_applied_idx == 1


def test_model_trust_round_trips_through_snapshot_wire_shape() -> None:
    """JSON snapshots should restore the immutable model-trust identity tuple."""

    identity = f"card_{'a' * 52}"
    event = StateSnapshotHydrated(
        state=State(model_trust_approved_remote_code_identities=(identity,))
    )
    adapter: TypeAdapter[Event] = TypeAdapter(Event)

    restored = adapter.validate_json(adapter.dump_json(event))

    assert isinstance(restored, StateSnapshotHydrated)
    assert restored.state.model_trust_approved_remote_code_identities == (identity,)
