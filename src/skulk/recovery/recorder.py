"""Keep this node's restart recovery record current."""

import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import final
from uuid import UUID

import anyio
from anyio import to_thread
from loguru import logger

from skulk.recovery.record import (
    ClusterIntentRecord,
    PlacementIntent,
    carry_departed_placements,
    derive_cluster_intent,
    read_record,
    write_record,
)
from skulk.shared.constants import SKULK_DATA_HOME
from skulk.shared.types.common import NodeId, SessionId
from skulk.shared.types.restart_recovery import RestartRecoveryAdvert
from skulk.shared.types.state import State
from skulk.shared.types.telemetry import TelemetryView

RECOVERY_RECORD_PATH = SKULK_DATA_HOME / "recovery" / "cluster-intent.json"
"""Where each node keeps its record (under ``SKULK_HOME`` when that is set)."""

RECORD_CHECK_SECONDS = 5.0
"""How often the node compares its view with the record on disk."""

ADVERT_REPUBLISH_SECONDS = 60.0
"""How often a node re-advertises its record's freshness on telemetry, so a
master that starts later still sees every node's advert."""

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
        advert_sender: Publishes this node's record freshness on telemetry;
            None for a recorder that only keeps the file (tests, tools).
        api_port: Port of this node's API, which serves the record; None when
            the node runs without one.
    """

    def __init__(
        self,
        *,
        state_provider: Callable[[], State | None],
        session_provider: Callable[[], SessionId | None],
        telemetry_view: TelemetryView,
        path: Path = RECOVERY_RECORD_PATH,
        clock: Callable[[], datetime] = lambda: datetime.now(tz=timezone.utc),
        advert_sender: Callable[[RestartRecoveryAdvert], Awaitable[None]] | None = None,
        api_port: int | None = None,
    ) -> None:
        self._state_provider = state_provider
        self._session_provider = session_provider
        self._telemetry_view = telemetry_view
        self._path = path
        self._clock = clock
        self._advert_sender = advert_sender
        self._api_port = api_port
        self._last_written: ClusterIntentRecord | None = read_record(path)
        # Install IDs this node has ever seen. A machine that drops out of
        # telemetry for a while keeps its install ID here, so an operator's
        # exclusion of it survives in the record until it returns.
        self._known_install_ids: dict[NodeId, UUID] = {}
        # Placements kept only because their machine left, and since when.
        self._departed_since: dict[PlacementIntent, datetime] = {}

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
        live_ids = node_install_ids(self._telemetry_view)
        self._known_install_ids.update(live_ids)
        # Keep only IDs still referenced, since node IDs change on every
        # restart and this map would otherwise grow for the life of the node.
        referenced = {
            node_id
            for instance in state.instances.values()
            for node_id in (
                *instance.shard_assignments.node_to_runner,
                *instance.excluded_nodes,
            )
        }
        self._known_install_ids = {
            node_id: install_id
            for node_id, install_id in self._known_install_ids.items()
            if node_id in live_ids or node_id in referenced
        }
        candidate = derive_cluster_intent(
            state,
            install_ids=self._known_install_ids,
            session_id=session_id,
            written_at=now,
        )
        previous = self._last_written
        if previous is not None:
            connected = set(state.topology.list_nodes())
            placements, self._departed_since = carry_departed_placements(
                previous.placements,
                candidate.placements,
                connected_install_ids={
                    install_id
                    for node_id, install_id in self._known_install_ids.items()
                    if node_id in connected
                },
                departed_since=self._departed_since,
                now=now,
            )
            candidate = candidate.model_copy(update={"placements": placements})
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

    def advert(self) -> RestartRecoveryAdvert:
        """Describe the record on disk, for other nodes choosing the freshest."""
        record = self._last_written
        return RestartRecoveryAdvert(
            written_at=record.stamp.written_at if record is not None else None,
            placements=len(record.placements) if record is not None else 0,
            api_port=self._api_port,
        )

    async def run(self) -> None:
        """Check every few seconds for the life of the node.

        The advert goes out at start (describing the record left by the last
        run, which a cold-started master needs before this node writes
        anything new), after every write, and once a minute.
        """
        last_advert: float | None = None
        while True:
            wrote = await self.record_once()
            now = time.monotonic()
            if self._advert_sender is not None and (
                wrote
                or last_advert is None
                or now - last_advert >= ADVERT_REPUBLISH_SECONDS
            ):
                await self._advert_sender(self.advert())
                last_advert = now
            await anyio.sleep(RECORD_CHECK_SECONDS)
