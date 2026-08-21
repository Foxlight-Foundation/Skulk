"""Authoritative ordering guards temporary qualification-card ownership."""

import pytest

from skulk.master import main as master_main
from skulk.master.main import Master
from skulk.shared.models.model_cards import ModelCard, ModelTask
from skulk.shared.types.commands import AddCustomModelCard, DeleteCustomModelCard
from skulk.shared.types.common import ModelId
from skulk.shared.types.events import CustomModelCardAdded, IndexedEvent
from skulk.shared.types.memory import Memory
from skulk.shared.types.state import State
from skulk.shared.types.telemetry import TelemetryView


def _card(model_id: str, *, qualification_only: bool = False) -> ModelCard:
    """Return one minimal immutable card for ordered-mutation tests."""
    return ModelCard(
        model_id=ModelId(model_id),
        source_revision="b" * 40,
        storage_size=Memory.from_bytes(1),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        is_custom=True,
        qualification_only=qualification_only,
    )


def _master() -> Master:
    """Return a narrow master instance containing only ordered card truth."""
    master = object.__new__(Master)
    object.__setattr__(master, "_ordered_model_cards", {})
    object.__setattr__(master, "state", State())
    object.__setattr__(master, "_telemetry_view", TelemetryView())
    return master


def _hide_signed_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep narrow ordering tests independent from the process registry cache."""

    def no_registry_card(_model_id: ModelId) -> None:
        return None

    monkeypatch.setattr(master_main, "get_current_registry_card", no_registry_card)


def test_service_cannot_shadow_signed_card_at_authoritative_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A signed alias remains protected even after a stale API-side check."""
    _hide_signed_registry(monkeypatch)
    existing = _card("org/model").model_copy(
        update={"is_custom": False, "registry_card_id": f"card_{'a' * 52}"}
    )
    temporary = _card("org/model", qualification_only=True)

    def existing_card(_model_id: ModelId) -> ModelCard:
        return existing

    monkeypatch.setattr(master_main, "get_card", existing_card)
    master = _master()

    event = master._order_custom_model_card_add(  # pyright: ignore[reportPrivateUsage]
        AddCustomModelCard(
            model_card=temporary,
            requires_qualification_ownership=True,
        )
    )

    assert event is None
    assert master._ordered_model_cards[existing.model_id] == existing  # pyright: ignore[reportPrivateUsage]


def test_operator_add_wins_before_stale_service_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Master ordering prevents a later stale cleanup from deleting operator truth."""
    _hide_signed_registry(monkeypatch)
    temporary = _card("org/model", qualification_only=True)
    operator = _card("org/model")

    def existing_card(_model_id: ModelId) -> ModelCard:
        return temporary

    monkeypatch.setattr(master_main, "get_card", existing_card)
    master = _master()

    operator_event = master._order_custom_model_card_add(  # pyright: ignore[reportPrivateUsage]
        AddCustomModelCard(model_card=operator)
    )
    cleanup_event = master._order_custom_model_card_delete(  # pyright: ignore[reportPrivateUsage]
        DeleteCustomModelCard(
            model_id=operator.model_id,
            requires_qualification_ownership=True,
        )
    )

    assert operator_event is not None
    assert cleanup_event is None
    assert master._ordered_model_cards[operator.model_id] == operator  # pyright: ignore[reportPrivateUsage]


def test_operator_add_wins_before_stale_service_overwrite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Master ordering prevents a stale service add from replacing operator truth."""
    _hide_signed_registry(monkeypatch)
    temporary = _card("org/model", qualification_only=True)
    operator = _card("org/model")

    def existing_card(_model_id: ModelId) -> ModelCard:
        return temporary

    monkeypatch.setattr(master_main, "get_card", existing_card)
    master = _master()

    master._order_custom_model_card_add(  # pyright: ignore[reportPrivateUsage]
        AddCustomModelCard(model_card=operator)
    )
    service_event = master._order_custom_model_card_add(  # pyright: ignore[reportPrivateUsage]
        AddCustomModelCard(
            model_card=temporary,
            requires_qualification_ownership=True,
        )
    )

    assert service_event is None
    assert master._ordered_model_cards[operator.model_id] == operator  # pyright: ignore[reportPrivateUsage]


def test_older_indexed_echo_cannot_roll_back_newer_ordered_truth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Event round trips never replace the command processor's newer decision."""
    _hide_signed_registry(monkeypatch)
    temporary = _card("org/model", qualification_only=True)
    operator = _card("org/model")

    def existing_card(_model_id: ModelId) -> ModelCard:
        return temporary

    monkeypatch.setattr(master_main, "get_card", existing_card)
    master = _master()
    old_command = AddCustomModelCard(
        model_card=temporary,
        requires_qualification_ownership=True,
    )
    old_event = master._order_custom_model_card_add(  # pyright: ignore[reportPrivateUsage]
        old_command
    )
    assert isinstance(old_event, CustomModelCardAdded)
    assert old_event.mutation_command_id == old_command.command_id
    master._order_custom_model_card_add(  # pyright: ignore[reportPrivateUsage]
        AddCustomModelCard(model_card=operator)
    )

    master._apply_indexed_event(IndexedEvent(idx=0, event=old_event))  # pyright: ignore[reportPrivateUsage]

    assert master._ordered_model_cards[operator.model_id] == operator  # pyright: ignore[reportPrivateUsage]


def test_signed_registry_refresh_supersedes_stale_qualification_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A newly published card blocks later lifecycle-owned mutations."""
    temporary = _card("org/model", qualification_only=True)
    signed = temporary.model_copy(
        update={
            "is_custom": False,
            "qualification_only": False,
            "registry_card_id": f"card_{'a' * 52}",
        }
    )
    registry_truth: dict[ModelId, ModelCard] = {}

    def existing_card(_model_id: ModelId) -> ModelCard:
        return temporary

    def registry_card(model_id: ModelId) -> ModelCard | None:
        return registry_truth.get(model_id)

    monkeypatch.setattr(master_main, "get_card", existing_card)
    monkeypatch.setattr(master_main, "get_current_registry_card", registry_card)
    master = _master()
    assert master._ordered_model_card(temporary.model_id) == temporary  # pyright: ignore[reportPrivateUsage]

    registry_truth[signed.model_id] = signed
    service_event = master._order_custom_model_card_add(  # pyright: ignore[reportPrivateUsage]
        AddCustomModelCard(
            model_card=temporary,
            requires_qualification_ownership=True,
        )
    )

    assert service_event is None
    assert master._ordered_model_cards[signed.model_id] == signed  # pyright: ignore[reportPrivateUsage]
