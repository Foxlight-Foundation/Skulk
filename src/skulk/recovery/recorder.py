"""Keep this node's restart recovery record current."""

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import final
from uuid import UUID

import anyio
from anyio import to_thread
from loguru import logger

from skulk.recovery.record import (
    ClusterIntentRecord,
    derive_cluster_intent,
    read_record,
    write_record,
)
from skulk.shared.constants import SKULK_DATA_HOME
from skulk.shared.types.common import NodeId, SessionId
from skulk.shared.types.state import State
from skulk.shared.types.telemetry import TelemetryView

RECOVERY_RECORD_PATH = SKULK_DATA_HOME / "recovery" / "cluster-intent.json"
"""Where each node keeps its record (under ``SKULK_HOME`` when that is set)."""

RECORD_CHECK_SECONDS = 5.0
"""How often the node compares its view with the record on disk."""

RECORD_REFRESH_AFTER = timedelta(minutes=5)
"""Re-stamp unchanged content this often, so the freshest record across nodes
is the one most recently confirmed current, not the one last changed."""


def node_install_ids(telemetry_view: TelemetryView) -> dict[NodeId, UUID]:
    """Map live node IDs to their stable install IDs, where known."""
    return {
        node_id: identity.node_install_id
        for node_id, identity in telemetry_view.node_identities.items()
        if identity.node_install_id is not None
    }


def record_is_writable(state: State | None) -> bool:
    """Whether a view may replace the record on disk.

    A node that has not applied any event yet holds an empty view, and a
    cluster that started cold holds a nearly empty one until its master has
    restored from the record. Writing either would replace the record that
    restore needs with an empty one, so both wait.
    """
    return (
        state is not None
        and state.last_event_applied_idx >= 0
        and not state.restart_recovery_pending
    )


@final
class ClusterIntentRecorder:
    """Write the node's restart recovery record whenever it would change.

    Args:
        state_provider: Returns the node's current replicated view, or ``None``.
        session_provider: Returns the node's current session, or ``None``.
        telemetry_view: Node telemetry, for stable install IDs.
        path: Record location.
        clock: Injected UTC clock.
    """

    def __init__(
        self,
        *,
        state_provider: Callable[[], State | None],
        session_provider: Callable[[], SessionId | None],
        telemetry_view: TelemetryView,
        path: Path = RECOVERY_RECORD_PATH,
        clock: Callable[[], datetime] = lambda: datetime.now(tz=timezone.utc),
    ) -> None:
        self._state_provider = state_provider
        self._session_provider = session_provider
        self._telemetry_view = telemetry_view
        self._path = path
        self._clock = clock
        self._last_written: ClusterIntentRecord | None = read_record(path)

    async def record_once(self) -> bool:
        """Write the record if the view is writable and it changed or aged.

        The view is read on the event loop, where it is consistent; the
        durable write (two flushes and a rename) runs on a worker thread so a
        slow or congested disk cannot stall routing, elections, or the API.

        Returns:
            True when a new record was written.
        """
        state = self._state_provider()
        session_id = self._session_provider()
        if not record_is_writable(state) or session_id is None:
            return False
        assert state is not None
        now = self._clock()
        candidate = derive_cluster_intent(
            state,
            install_ids=node_install_ids(self._telemetry_view),
            session_id=session_id,
            written_at=now,
        )
        previous = self._last_written
        if (
            candidate.same_content(previous)
            and previous is not None
            and now - previous.stamp.written_at < RECORD_REFRESH_AFTER
        ):
            return False
        try:
            await to_thread.run_sync(write_record, self._path, candidate)
        except OSError as error:
            # A full or read-only disk must not take the node down; the cost
            # is only that this node cannot supply a record after a restart.
            logger.warning(f"Could not write restart recovery record: {error}")
            return False
        self._last_written = candidate
        return True

    async def run(self) -> None:
        """Check every few seconds for the life of the node."""
        while True:
            await self.record_once()
            await anyio.sleep(RECORD_CHECK_SECONDS)
