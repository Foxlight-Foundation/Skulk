"""The master's side of a person's engine orders: ordering changes and seeding a session."""

import anyio

from skulk.master.main import Master
from skulk.shared.session_carryover import seed_state_for_new_session
from skulk.shared.types.commands import (
    ForwarderCommand,
    ForwarderDownloadCommand,
    SetModelEngineOrder,
)
from skulk.shared.types.common import ModelId, NodeId, SessionId
from skulk.shared.types.engine_orders import MAX_MODEL_ENGINE_ORDERS
from skulk.shared.types.events import (
    Event,
    GlobalForwarderEvent,
    LocalForwarderEvent,
)
from skulk.shared.types.state import State
from skulk.shared.types.state_sync import StateSyncMessage
from skulk.utils.channels import Receiver, channel

MODEL = ModelId("test/text")
ORDER = ("tensorfold-rocm", "llama_cpp-vulkan")


def _master(
    initial_state: State | None = None,
    initial_model_engine_orders: dict[ModelId, tuple[str, ...]] | None = None,
) -> tuple[Master, Receiver[GlobalForwarderEvent]]:
    global_send, global_recv = channel[GlobalForwarderEvent]()
    master = Master(
        NodeId(),
        SessionId(master_node_id=NodeId(), election_clock=1),
        command_receiver=channel[ForwarderCommand]()[1],
        event_sender=channel[Event]()[0],
        local_event_receiver=channel[LocalForwarderEvent]()[1],
        global_event_sender=global_send,
        state_sync_receiver=channel[StateSyncMessage]()[1],
        state_sync_sender=channel[StateSyncMessage]()[0],
        download_command_sender=channel[ForwarderDownloadCommand]()[0],
        initial_state=initial_state,
        initial_model_engine_orders=initial_model_engine_orders,
    )
    return master, global_recv


async def _index_seed(master: Master, global_recv: Receiver[GlobalForwarderEvent]) -> None:
    async def consume() -> None:
        with global_recv as events:
            async for _ in events:
                return

    with anyio.fail_after(10):
        async with anyio.create_task_group() as tg:
            tg.start_soon(consume)
            await master._index_seed_event()  # pyright: ignore[reportPrivateUsage]


def _order(master: Master, command: SetModelEngineOrder):
    return master._order_model_engine_order(command)  # pyright: ignore[reportPrivateUsage]


def _held(master: Master) -> dict[ModelId, tuple[str, ...]]:
    return master._model_engine_orders  # pyright: ignore[reportPrivateUsage]


def test_a_set_is_ordered_and_answers_its_command() -> None:
    master, _ = _master()
    command = SetModelEngineOrder(model_id=MODEL, engine_order=list(ORDER))

    event = _order(master, command)

    assert event is not None
    assert event.model_id == MODEL
    assert event.engine_order == list(ORDER)
    assert event.mutation_command_id == command.command_id
    assert _held(master) == {MODEL: ORDER}


def test_a_later_set_replaces_and_a_clear_removes() -> None:
    master, _ = _master()
    _order(master, SetModelEngineOrder(model_id=MODEL, engine_order=list(ORDER)))

    _order(master, SetModelEngineOrder(model_id=MODEL, engine_order=["mlx"]))
    assert _held(master) == {MODEL: ("mlx",)}

    event = _order(master, SetModelEngineOrder(model_id=MODEL))
    assert event is not None
    assert event.engine_order is None
    assert _held(master) == {}


def test_clearing_a_model_with_no_order_still_answers() -> None:
    master, _ = _master()

    event = _order(master, SetModelEngineOrder(model_id=MODEL))

    assert event is not None
    assert event.engine_order is None


def test_a_new_model_is_refused_once_the_cluster_holds_its_limit() -> None:
    full = {ModelId(f"test/model-{index}"): ("mlx",) for index in range(MAX_MODEL_ENGINE_ORDERS)}
    master, _ = _master(initial_model_engine_orders=full)

    refused = _order(master, SetModelEngineOrder(model_id=MODEL, engine_order=["mlx"]))
    changed = _order(
        master,
        SetModelEngineOrder(model_id=ModelId("test/model-0"), engine_order=list(ORDER)),
    )
    cleared = _order(master, SetModelEngineOrder(model_id=ModelId("test/model-1")))

    assert refused is None
    assert MODEL not in _held(master)
    assert changed is not None
    assert cleared is not None
    assert len(_held(master)) == MAX_MODEL_ENGINE_ORDERS - 1


def test_a_master_with_nothing_to_carry_indexes_no_seed() -> None:
    """No saved order, no difference: a bare session starts as it always has."""
    master, _ = _master()

    assert master._seed_state is None  # pyright: ignore[reportPrivateUsage]
    assert master.state.model_engine_orders == {}

    master, _ = _master(initial_model_engine_orders={})
    assert master._seed_state is None  # pyright: ignore[reportPrivateUsage]


async def test_a_session_with_no_view_starts_from_the_nodes_saved_orders() -> None:
    master, global_recv = _master(initial_model_engine_orders={MODEL: ORDER})

    await _index_seed(master, global_recv)

    assert master.state.model_engine_orders == {MODEL: ORDER}
    assert master.state.last_event_applied_idx == 0
    assert _held(master) == {MODEL: ORDER}


async def test_a_promoted_master_keeps_the_orders_it_carried() -> None:
    carried = seed_state_for_new_session(State(model_engine_orders={MODEL: ORDER}))
    master, global_recv = _master(initial_state=carried)

    assert _held(master) == {MODEL: ORDER}
    await _index_seed(master, global_recv)

    assert master.state.model_engine_orders == {MODEL: ORDER}


async def test_saved_orders_stand_in_for_a_carried_view_that_never_caught_up() -> None:
    """An empty carried view must not make every node drop its saved orders."""
    never_caught_up = seed_state_for_new_session(State())
    master, global_recv = _master(
        initial_state=never_caught_up, initial_model_engine_orders={MODEL: ORDER}
    )

    await _index_seed(master, global_recv)

    assert master.state.model_engine_orders == {MODEL: ORDER}


async def test_a_carried_view_outranks_nothing_passed_but_not_saved_orders_passed() -> None:
    """The node decides which applies; the master honours what it is handed."""
    carried = seed_state_for_new_session(State(model_engine_orders={MODEL: ("mlx",)}))
    master, global_recv = _master(
        initial_state=carried, initial_model_engine_orders={MODEL: ORDER}
    )

    await _index_seed(master, global_recv)

    assert master.state.model_engine_orders == {MODEL: ORDER}
