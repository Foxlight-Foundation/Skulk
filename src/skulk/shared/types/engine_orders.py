"""A person's own engine order for a model.

Several engines can often run the same model. Which one runs it is decided
by the first of three levels that has an answer:

1. the person's own order for that model, held here and shared across the
   cluster;
2. the model card's default order (``placement.backend_preference``);
3. the built-in fallback, with installed engine packages after built-in
   engines.

An order only ranks. Entries a node does not have are skipped, and when
nothing in a list applies on a node the next level decides there, so an
order can never make a model impossible to launch.

The order is keyed by the model's name, not by its card, so it survives a
card being re-authored. It is never written onto the card.
"""

import re
from collections.abc import Mapping, Sequence
from typing import Final, Literal

from skulk.shared.types.common import ModelId

MAX_ENGINE_ORDER_ENTRIES: Final = 16
"""Most entries one model's order may hold."""

MAX_MODEL_ENGINE_ORDERS: Final = 512
"""Most models that may hold an order of their own at once."""

MAX_ENGINE_ORDER_ENTRY_LENGTH: Final = 64
"""Longest single entry, in characters."""

_ENGINE_ORDER_ENTRY: Final = re.compile(r"[a-z0-9_]+(?:-[a-z0-9_]+)?")

EngineOrderLevel = Literal["launch", "model", "card", "default"]
"""Which level chose an engine: an order given for one launch, the person's
order for the model, the card's default order, or the built-in fallback."""


def engine_order_problem(order: Sequence[str]) -> str | None:
    """Say why a list cannot be stored as an engine order, or ``None``.

    An entry has the shape of a backend tag: an engine name, optionally
    followed by the hardware it runs on (``llama_server-vulkan``). The engine
    need not be one this build knows, because engine packages add engines
    later; an unknown entry is skipped when the order is used.

    Args:
        order: The proposed order, most preferred first.

    Returns:
        A sentence naming the first problem, or ``None`` when the order is
        well formed.
    """
    if not order:
        return "An engine order needs at least one entry; clear it instead."
    if len(order) > MAX_ENGINE_ORDER_ENTRIES:
        return f"An engine order holds at most {MAX_ENGINE_ORDER_ENTRIES} entries."
    seen: set[str] = set()
    for entry in order:
        if (
            len(entry) > MAX_ENGINE_ORDER_ENTRY_LENGTH
            or _ENGINE_ORDER_ENTRY.fullmatch(entry) is None
        ):
            return (
                "Each entry is an engine name, optionally followed by its "
                f"hardware, such as llama_server-vulkan; got {entry[:80]!r}."
            )
        if entry in seen:
            return f"The entry {entry!r} appears more than once."
        seen.add(entry)
    return None


def layered_engine_order(
    card_order: Sequence[str],
    model_order: Sequence[str] | None = None,
    launch_order: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Lay the levels end to end, the most specific first.

    The result is consulted from the front, so the person's whole list is
    read before the card's, and the card's list is reached only where the
    person's names nothing a node has. An entry already given by an earlier
    level is not repeated. With no order given the result is the card's own
    order, unchanged.

    Args:
        card_order: The card's default order.
        model_order: The person's order for the model, when one is set.
        launch_order: An order given for this launch only, when one is.

    Returns:
        One order to rank engines by.
    """
    if not launch_order and not model_order:
        return tuple(card_order)
    layered: list[str] = []
    for level in (launch_order, model_order, card_order):
        for entry in level or ():
            if entry not in layered:
                layered.append(entry)
    return tuple(layered)


def model_engine_order(
    orders: Mapping[ModelId, Sequence[str]] | None, model_id: ModelId
) -> tuple[str, ...] | None:
    """Return the person's order for a model, or ``None`` when none is set."""
    if not orders:
        return None
    order = orders.get(model_id)
    return tuple(order) if order else None


def engine_order_level(
    resolved_backend: str | None,
    *,
    card_order: Sequence[str],
    model_order: Sequence[str] | None = None,
    launch_order: Sequence[str] | None = None,
) -> EngineOrderLevel:
    """Say which level chose a backend.

    An entry matches a backend when it is that backend's tag, or names its
    engine without saying which hardware.

    Args:
        resolved_backend: The backend a node resolved to, if any.
        card_order: The card's default order.
        model_order: The person's order for the model, when one is set.
        launch_order: An order given for this launch only, when one is.

    Returns:
        The first level holding an entry that matches, else ``"default"``.
    """
    if resolved_backend is None:
        return "default"
    engine = resolved_backend.split("-", 1)[0]
    levels: tuple[tuple[EngineOrderLevel, Sequence[str] | None], ...] = (
        ("launch", launch_order),
        ("model", model_order),
        ("card", card_order),
    )
    for level, order in levels:
        if order and any(entry in (resolved_backend, engine) for entry in order):
            return level
    return "default"
