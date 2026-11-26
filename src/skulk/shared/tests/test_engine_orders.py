"""A person's engine order for a model: its shape, its levels, and how State holds it."""

import json

import pytest
from pydantic import TypeAdapter, ValidationError

from skulk.shared.apply import apply
from skulk.shared.session_carryover import seed_state_for_new_session
from skulk.shared.types.commands import Command, SetModelEngineOrder
from skulk.shared.types.common import ModelId
from skulk.shared.types.engine_orders import (
    MAX_ENGINE_ORDER_ENTRIES,
    engine_order_level,
    engine_order_problem,
    layered_engine_order,
    model_engine_order,
)
from skulk.shared.types.events import (
    Event,
    IndexedEvent,
    ModelEngineOrderChanged,
    is_persistable_control_event,
)
from skulk.shared.types.state import State

MODEL = ModelId("test/text")
OTHER = ModelId("test/other")


@pytest.mark.parametrize(
    "order",
    [
        ["llama_server-vulkan"],
        ["mlx"],
        ["tensorfold-rocm", "llama_cpp-vulkan", "llama_cpp"],
        ["an_engine_this_build_has_never_heard_of-cuda"],
    ],
)
def test_a_well_formed_order_has_no_problem(order: list[str]) -> None:
    assert engine_order_problem(order) is None


@pytest.mark.parametrize(
    "order",
    [
        [],
        ["llama_server-vulkan", "llama_server-vulkan"],
        ["Llama_Server"],
        ["llama server"],
        ["llama_server-vulkan-extra"],
        ["-vulkan"],
        ["x" * 65],
        [f"engine_{index}" for index in range(MAX_ENGINE_ORDER_ENTRIES + 1)],
    ],
)
def test_a_malformed_order_is_named(order: list[str]) -> None:
    assert engine_order_problem(order) is not None


def test_the_command_refuses_a_malformed_order_where_it_is_built() -> None:
    with pytest.raises(ValidationError):
        SetModelEngineOrder(model_id=MODEL, engine_order=["Not A Tag"])


def test_the_command_may_clear() -> None:
    assert SetModelEngineOrder(model_id=MODEL).engine_order is None


def test_with_no_order_given_the_card_order_is_unchanged() -> None:
    card_order = ("llama_cpp-vulkan", "llama_cpp-rocm")

    assert layered_engine_order(card_order) == card_order
    assert layered_engine_order(card_order, None, None) == card_order
    assert layered_engine_order(()) == ()
    # Not even a repeated entry on a card is tidied when nobody set an order.
    assert layered_engine_order(("mlx", "mlx")) == ("mlx", "mlx")


def test_levels_are_read_most_specific_first_without_repeats() -> None:
    assert layered_engine_order(
        ("llama_cpp-vulkan", "mlx"),
        ("tensorfold-rocm", "llama_cpp-vulkan"),
        ("vllm-cuda",),
    ) == ("vllm-cuda", "tensorfold-rocm", "llama_cpp-vulkan", "mlx")


def test_a_model_without_an_order_has_none() -> None:
    assert model_engine_order(None, MODEL) is None
    assert model_engine_order({}, MODEL) is None
    assert model_engine_order({OTHER: ("mlx",)}, MODEL) is None
    assert model_engine_order({MODEL: ()}, MODEL) is None
    assert model_engine_order({MODEL: ["mlx"]}, MODEL) == ("mlx",)


@pytest.mark.parametrize(
    ("resolved", "expected"),
    [
        ("vllm-cuda", "launch"),
        ("tensorfold-rocm", "model"),
        ("llama_cpp-vulkan", "card"),
        ("llama_server-vulkan", "default"),
        (None, "default"),
    ],
)
def test_the_level_that_chose_a_backend_is_named(
    resolved: str | None, expected: str
) -> None:
    assert (
        engine_order_level(
            resolved,
            card_order=("llama_cpp",),
            model_order=("tensorfold-rocm",),
            launch_order=("vllm-cuda",),
        )
        == expected
    )


def _changed(order: list[str] | None, model_id: ModelId = MODEL) -> IndexedEvent:
    return IndexedEvent(
        idx=0, event=ModelEngineOrderChanged(model_id=model_id, engine_order=order)
    )


def test_apply_sets_replaces_and_clears_one_models_order() -> None:
    state = apply(State(), _changed(["tensorfold-rocm", "llama_cpp-vulkan"]))
    assert state.model_engine_orders == {MODEL: ("tensorfold-rocm", "llama_cpp-vulkan")}

    state = apply(
        state.model_copy(update={"last_event_applied_idx": -1}),
        _changed(["mlx"], OTHER),
    )
    state = apply(
        state.model_copy(update={"last_event_applied_idx": -1}), _changed(["mlx"])
    )
    assert state.model_engine_orders == {MODEL: ("mlx",), OTHER: ("mlx",)}

    state = apply(
        state.model_copy(update={"last_event_applied_idx": -1}), _changed(None)
    )
    assert state.model_engine_orders == {OTHER: ("mlx",)}


def test_clearing_a_model_with_no_order_changes_nothing() -> None:
    assert apply(State(), _changed(None)).model_engine_orders == {}


def test_state_without_orders_serializes_an_empty_mapping() -> None:
    assert State().model_engine_orders == {}


def test_state_orders_survive_a_json_round_trip_in_both_validation_modes() -> None:
    state = State(model_engine_orders={MODEL: ("tensorfold-rocm", "mlx")})
    wire = state.model_dump_json(by_alias=True)

    assert State.model_validate_json(wire).model_engine_orders == state.model_engine_orders
    assert (
        State.model_validate(json.loads(wire)).model_engine_orders
        == state.model_engine_orders
    )


def test_the_event_and_command_survive_their_tagged_wire_form() -> None:
    event = ModelEngineOrderChanged(model_id=MODEL, engine_order=["tensorfold-rocm"])
    events = TypeAdapter[Event](Event)
    assert events.validate_json(events.dump_json(event)) == event

    command = SetModelEngineOrder(model_id=MODEL, engine_order=["tensorfold-rocm"])
    commands = TypeAdapter[Command](Command)
    assert commands.validate_json(commands.dump_json(command)) == command


def test_the_change_is_a_durable_control_event() -> None:
    assert is_persistable_control_event(
        ModelEngineOrderChanged(model_id=MODEL, engine_order=["mlx"])
    )


def test_a_promoted_master_carries_every_order() -> None:
    prior = State(model_engine_orders={MODEL: ("tensorfold-rocm",)})

    assert seed_state_for_new_session(prior).model_engine_orders == {
        MODEL: ("tensorfold-rocm",)
    }
