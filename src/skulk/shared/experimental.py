"""The fabric's generic experimental-feature gate.

Skulk stages in-development features behind a single node-local switch so a
released build can carry work-in-progress UX without exposing it by default.
``SKULK_ENABLE_EXPERIMENTAL_MODE`` (off unless explicitly set) is that switch:
it reveals the dashboard's "Experiments" settings section, and any feature that
opts into the experimental gate stays inert on a node that has not set it.

This module is deliberately **feature-agnostic**: it knows nothing about any
particular experiment. A feature that wants to be gated reads this flag (and,
when it needs a per-feature toggle, adds one to its own config section under the
same section in the dashboard). Keeping the gate here, rather than inside any
one feature, lets features come and go without moving the switch.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

EXPERIMENTAL_MODE_ENV_VAR = "SKULK_ENABLE_EXPERIMENTAL_MODE"

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def experimental_mode_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Return whether this node is opted into experimental features.

    Driven by ``SKULK_ENABLE_EXPERIMENTAL_MODE``; ``False`` unless the variable
    is set to a truthy value (``1``/``true``/``yes``/``on``, case-insensitive).
    This both reveals the Experiments settings section in the dashboard and
    gates every experimental feature's effect, so a node without it behaves
    exactly as it does today.

    Args:
        env: Environment mapping to read from. Defaults to ``os.environ``;
            injectable so tests never touch real process state.

    Returns:
        ``True`` when the node is in experimental mode, else ``False``.
    """
    source = os.environ if env is None else env
    value = source.get(EXPERIMENTAL_MODE_ENV_VAR)
    return value is not None and value.strip().lower() in _TRUTHY
