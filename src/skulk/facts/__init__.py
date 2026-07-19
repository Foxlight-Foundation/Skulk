"""Node Facts: single probe subsystem every capability consumer reads (#614).

``probe`` gathers one typed :class:`~skulk.shared.types.node_facts.NodeFacts`
record (hardware observed, software observed, configuration declared);
``derive`` turns it into advertised backend tags plus loud
:class:`~skulk.shared.types.node_facts.CapabilityConflict` entries. The
process-wide cached snapshot lives here: facts are gathered once per process
(the same contract as any import-time capability -- installing a dependency or
changing an env var requires a restart to notice), and every consumer
(backend advertisement, ``NodeResources`` telemetry, node health, doctor)
reads the same snapshot so they can never disagree.
"""

from __future__ import annotations

import threading

from loguru import logger

from skulk.facts.derive import BackendDerivation, derive_node_backends
from skulk.facts.probe import gather_node_facts
from skulk.shared.types.node_facts import CONFLICT_ERROR_CODES, NodeFacts

__all__ = [
    "BackendDerivation",
    "current_backend_derivation",
    "current_node_facts",
    "derive_node_backends",
    "gather_node_facts",
    "refresh_node_facts",
]

_lock = threading.Lock()
_facts: NodeFacts | None = None
_derivation: BackendDerivation | None = None


def _log_derivation(facts: NodeFacts, derivation: BackendDerivation) -> None:
    """Log the node capability summary and every conflict, once per snapshot.

    This is the guaranteed operator-visible surface: it fires on every process
    entry path (worker startup, runner bootstrap fallback, telemetry gather)
    before any serving decision uses the derived tags, so a degraded or
    conflicted node is loud in the log even on a headless node with no
    dashboard.
    """
    gpu_summary = (
        ", ".join(
            f"{gpu.vendor}:{gpu.name}"
            + (
                f" ({gpu.vram_total_bytes / 2**30:.0f} GiB VRAM)"
                if gpu.vram_total_bytes
                else ""
            )
            for gpu in facts.gpus
        )
        or "none"
    )
    logger.info(
        f"node facts: platform={facts.platform} gpus=[{gpu_summary}] "
        f"backends={sorted(derivation.backends)}"
    )
    for note in derivation.notes:
        logger.warning(f"backend derivation: {note}")
    for conflict in derivation.conflicts:
        log = logger.error if conflict.code in CONFLICT_ERROR_CODES else logger.warning
        log(
            f"capability conflict [{conflict.code}]: {conflict.message} "
            f"Remediation: {conflict.remediation}"
        )


def current_node_facts() -> NodeFacts:
    """Return the process-wide facts snapshot, gathering it on first use."""
    global _facts, _derivation
    with _lock:
        if _facts is None:
            _facts = gather_node_facts()
            _derivation = derive_node_backends(_facts)
            _log_derivation(_facts, _derivation)
        return _facts


def current_backend_derivation() -> BackendDerivation:
    """Return the derivation (backends + conflicts) for the current snapshot."""
    global _derivation
    current_node_facts()
    assert _derivation is not None  # set together with _facts under the lock
    return _derivation


def refresh_node_facts() -> NodeFacts:
    """Re-gather facts, replacing the process-wide snapshot.

    For ``skulk doctor`` (which wants a fresh audit) and tests. Ordinary
    runtime code should read the cached snapshot so all consumers agree.
    """
    global _facts, _derivation
    with _lock:
        _facts = gather_node_facts()
        _derivation = derive_node_backends(_facts)
        _log_derivation(_facts, _derivation)
        return _facts
