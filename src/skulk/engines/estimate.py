"""Ask an installed engine package what one exact load needs.

An engine knows its own memory need better than a figure declared once for
every model. The worker asks for it just before it dispatches a load, when the
model's files are on the node, and its local memory check then uses the larger
of the engine's figure and the one the launch was admitted on.

The engine reports need only. It never reports what is free: availability is
the master's and the worker's to decide, or two engines on one node would
overbook it. An engine that does not know, refuses to say or cannot be asked
changes nothing, and the launch proceeds on the admitted figure.
"""

import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, final

from loguru import logger

from skulk.engines.adapter import AdapterFailedError, AdapterRefusedError
from skulk.engines.launch import (
    ENGINE_STATE_DIR,
    adapter_session,
    engine_state_directory,
    plan_adapter_launch,
)
from skulk.engines.protocol import EstimateResponse, LoadSettings, ModelIdentity
from skulk.provisioning.engine_install import InstalledEngine
from skulk.provisioning.engine_packages import EnginePackageRefusedError

ESTIMATE_DEADLINE_SECONDS: Final = 10.0
"""Time allowed to start the adapter and receive its estimate, together. The
worker waits for the answer before it dispatches the load, so the bound is
short; an adapter that misses it is treated as not knowing."""

_MINIMUM_CALL_SECONDS: Final = 1.0

MemoryPool = Literal["gpu", "host"]


@final
@dataclass(frozen=True)
class EngineNeed:
    """The memory an engine says one exact load needs.

    Attributes:
        pool: Where the bytes are taken from: ``gpu`` for the device memory of
            a GPU lane, ``host`` for system RAM.
        bytes: The larger of the load-time and steady-state figures the engine
            knew.
    """

    pool: MemoryPool
    bytes: int


def stated_need(response: EstimateResponse) -> EngineNeed | None:
    """Reduce an adapter's estimate to one figure, or ``None`` when it knew none.

    A load must hold both its load-time peak and its steady state, so the
    larger known figure is the need. An absent figure means unknown, never
    zero.
    """
    known = [
        count
        for count in (response.load_bytes, response.steady_bytes)
        if count is not None
    ]
    if not known:
        return None
    return EngineNeed(pool=response.pool, bytes=max(known))


def ask_engine_need(
    installed: InstalledEngine,
    *,
    instance_id: str,
    model: ModelIdentity,
    settings: LoadSettings,
    state_root: Path = ENGINE_STATE_DIR,
    deadline_seconds: float = ESTIMATE_DEADLINE_SECONDS,
) -> EngineNeed | None:
    """Start the package's adapter, ask what this load needs, and stop it.

    The session is separate from the one that later loads the model: it has
    its own state directory, which is removed afterwards, and it loads
    nothing. Blocks for at most about ``deadline_seconds`` plus the adapter's
    stop, so call it off the event loop.

    Args:
        installed: The verified generation the placement was made for.
        instance_id: The instance the load belongs to.
        model: The exact model and its directory on this node.
        settings: The admitted settings the load will state.
        state_root: Root of what running packages may write.
        deadline_seconds: Time allowed for the start and the answer together.

    Returns:
        The engine's figure, or ``None`` when the engine does not know, when
        it refuses to say, or when its adapter cannot be started or does not
        answer in time. None of those fails the launch; the load itself
        reports a broken adapter.
    """
    state_directory = (
        engine_state_directory(installed.digest, root=state_root)
        / "estimates"
        / instance_id
    )
    started = time.monotonic()
    engine = installed.claims.engine_id
    response: EstimateResponse | None = None
    stopped = True
    try:
        launch = plan_adapter_launch(installed, state_directory=state_directory)
        with adapter_session(
            launch, start_deadline_seconds=deadline_seconds
        ) as session:
            remaining = deadline_seconds - (time.monotonic() - started)
            response = session.estimate(
                instance_id=instance_id,
                model=model,
                settings=settings,
                deadline_seconds=max(remaining, _MINIMUM_CALL_SECONDS),
            )
    except AdapterRefusedError as refused:
        # The engine's own words may name local paths; they stay in this log.
        logger.warning(
            f"Engine package {engine} gave no memory estimate for {model.model_id} "
            f"({refused.code}: {refused.reason}); using the admitted figure"
        )
    except (AdapterFailedError, EnginePackageRefusedError, OSError) as error:
        logger.warning(
            f"Engine package {engine} could not be asked for a memory estimate "
            f"for {model.model_id} ({type(error).__name__}: {error}); using the "
            "admitted figure"
        )
    except subprocess.TimeoutExpired:
        # An unconfirmed process-group drain keeps its leader unreaped. An
        # answer already received still counts; the state stays for whoever
        # looks into the process that would not stop.
        stopped = False
        logger.error(
            f"Engine package {engine}'s adapter did not stop cleanly after a "
            f"memory estimate; its state is kept at {state_directory}"
        )
    finally:
        if stopped:
            shutil.rmtree(state_directory, ignore_errors=True)
    return None if response is None else stated_need(response)
