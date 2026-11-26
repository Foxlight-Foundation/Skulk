"""Each node's saved copy of the person-set engine orders.

Replicated ``State`` holds the orders while a cluster runs, and failover
carries them from a surviving node. A single-node reboot, or every node
restarting at once, leaves no survivor, so each node keeps the orders in a
small file of its own and a master that starts with no replicated view reads
its file back.

The file mirrors ``State``; it is never edited on its own. A node writes it
only from a view that has applied at least one event, because a node that
has not caught up yet holds an empty view and would otherwise replace its
saved orders with none.

One limit is known and accepted: the file a new session starts from is the
elected master's own. A node that was offline while an order changed, and
then becomes master of a cluster where every node restarted, brings back its
older copy.
"""

import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Final, Literal, cast, final

import anyio
from anyio import to_thread
from loguru import logger
from pydantic import Field, ValidationError, field_validator

from skulk.shared.constants import SKULK_DATA_HOME
from skulk.shared.types.common import ModelId
from skulk.shared.types.engine_orders import (
    MAX_MODEL_ENGINE_ORDERS,
    engine_order_problem,
)
from skulk.shared.types.state import State
from skulk.utils.pydantic_ext import FrozenModel

MODEL_ENGINE_ORDERS_PATH: Final = SKULK_DATA_HOME / "engine-orders" / "model-engine-orders.json"
"""Where each node keeps its copy (under ``SKULK_HOME`` when that is set)."""

ORDERS_CHECK_SECONDS: Final = 5.0
"""How often the node compares its view with the file on disk."""


class ModelEngineOrdersFile(FrozenModel):
    """The saved orders, one per model name."""

    format_version: Literal[1] = Field(default=1, description="File schema version.")
    orders: dict[str, tuple[str, ...]] = Field(
        default_factory=dict,
        description="Backend tags, most preferred first, by model name.",
    )

    @field_validator("orders", mode="before")
    @classmethod
    def _coerce_orders(cls, value: object) -> object:
        """Restore each order's immutable tuple from a JSON array."""
        if isinstance(value, Mapping):
            return {
                key: tuple(cast("list[object]", item)) if isinstance(item, list) else item
                for key, item in cast("Mapping[object, object]", value).items()
            }
        return value

    @field_validator("orders")
    @classmethod
    def _require_well_formed_orders(
        cls, value: dict[str, tuple[str, ...]]
    ) -> dict[str, tuple[str, ...]]:
        """Refuse a file no command could have produced."""
        if len(value) > MAX_MODEL_ENGINE_ORDERS:
            raise ValueError("too many models hold an engine order")
        for order in value.values():
            if (problem := engine_order_problem(order)) is not None:
                raise ValueError(problem)
        return value


def _as_file(orders: Mapping[ModelId, Sequence[str]]) -> ModelEngineOrdersFile:
    """Build the file content for a set of orders, in a stable key order."""
    return ModelEngineOrdersFile(
        orders={
            str(model_id): tuple(orders[model_id])
            for model_id in sorted(orders)
            if orders[model_id]
        }
    )


def read_model_engine_orders(
    path: Path = MODEL_ENGINE_ORDERS_PATH,
) -> dict[ModelId, tuple[str, ...]] | None:
    """Return the saved orders, or ``None`` when there is no usable file.

    A missing file means no order was ever saved here. A damaged or
    malformed file is reported and ignored: starting without saved orders is
    what happened before this file existed, and never stops the node.

    Args:
        path: File location.

    Returns:
        The saved orders by model name (possibly empty), or ``None``.
    """
    try:
        raw = path.read_text()
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError) as error:
        logger.warning(f"Saved engine orders unreadable at {path}: {error}")
        return None
    try:
        saved = ModelEngineOrdersFile.model_validate_json(raw)
    except ValidationError as error:
        logger.warning(
            f"Saved engine orders at {path} are not valid; ignoring them: "
            f"{error.error_count()} problem(s)"
        )
        return None
    return {ModelId(model_id): order for model_id, order in saved.orders.items()}


def write_model_engine_orders(
    path: Path, orders: Mapping[ModelId, Sequence[str]]
) -> None:
    """Replace the saved orders in one durable step.

    The new content goes to a sibling file, is flushed to disk, and then
    replaces the old file in one rename, after which the directory entry is
    flushed too. A crash at any point leaves either the old file or the new.

    Args:
        path: File location; its folder is created when missing.
        orders: The orders to save, by model name.

    Raises:
        OSError: When the file cannot be written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.tmp")
    descriptor = os.open(staging, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(_as_file(orders).model_dump_json(by_alias=True).encode())
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(staging, path)
    if os.name != "nt":
        # Windows cannot open a folder to flush it; the rename is durable
        # enough there for a file that the next change rewrites anyway.
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def orders_are_writable(state: State | None) -> bool:
    """Whether a view may replace the saved orders.

    A node that has not applied any event yet holds an empty view. Writing
    it would erase the very orders a session starting from this node needs.
    """
    return state is not None and state.last_event_applied_idx >= 0


@final
class ModelEngineOrderRecorder:
    """Keep this node's saved engine orders equal to its replicated view.

    Args:
        state_provider: Returns the node's current replicated view, or ``None``.
        path: File location.
    """

    def __init__(
        self,
        *,
        state_provider: Callable[[], State | None],
        path: Path = MODEL_ENGINE_ORDERS_PATH,
    ) -> None:
        self._state_provider = state_provider
        self._path = path
        self._saved: dict[ModelId, tuple[str, ...]] | None = read_model_engine_orders(
            path
        )

    async def record_once(self) -> bool:
        """Write the file when the view is writable and differs from it.

        A node that has never held an order writes nothing, so a cluster
        where nobody sets an order gains no file. The write runs on a worker
        thread so a slow disk cannot stall routing, elections or the API.

        Returns:
            True when the file was written.
        """
        state = self._state_provider()
        if not orders_are_writable(state):
            return False
        assert state is not None
        current = {
            model_id: tuple(order)
            for model_id, order in state.model_engine_orders.items()
            if order
        }
        if current == (self._saved or {}):
            return False
        try:
            await to_thread.run_sync(write_model_engine_orders, self._path, current)
        except OSError as error:
            # A full or read-only disk must not take the node down; the cost
            # is only that this node cannot bring the orders back after a
            # restart.
            logger.warning(f"Could not save engine orders: {error}")
            return False
        self._saved = current
        return True

    async def run(self) -> None:
        """Check every few seconds for the life of the node."""
        while True:
            await self.record_once()
            await anyio.sleep(ORDERS_CHECK_SECONDS)
