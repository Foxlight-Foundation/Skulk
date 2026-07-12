"""Generation statistics for runner engines that stream detokenized text.

The MLX engine counts tokens and phases natively inside its generation loop
and attaches a :class:`~skulk.api.types.api.GenerationStats` to the terminal
chunk; the API forwards it to clients (the bench/harness ``generation_stats``
surface) and the field-telemetry tap reads it. The llama.cpp engines
(in-process ``llama_cpp`` and the served ``llama_server`` proxy) stream
detokenized text without those measurements, which left every request from
them stats-less (#532): dashboards, diagnostics, and the results ledger were
blind to token throughput on GPU/Linux nodes.

This module provides the shared pieces those engines need:

- :class:`StreamStatsClock`: wall-clock phase timing (request start -> first
  streamed piece = prefill; first piece -> last piece = decode), mirroring
  the MLX convention of ``generation_tps = generated tokens / decode wall
  time``.
- :func:`stats_from_llama_server_timings`: exact statistics from
  ``llama-server``'s native ``timings`` object when the server provides it.
- :func:`process_peak_memory`: peak RSS of the runner process, the nearest
  llama.cpp analog of MLX's ``mx.get_peak_memory()`` (weights are mapped
  into process memory; GPU-side VRAM is not visible from here).
"""

from __future__ import annotations

import resource
import sys
import time
from pathlib import Path
from typing import Callable, cast, final

from skulk.api.types.api import GenerationStats
from skulk.shared.types.memory import Memory


def process_peak_memory() -> Memory:
    """Peak resident set size of this process as a :class:`Memory`.

    ``ru_maxrss`` is reported in bytes on macOS and kilobytes on Linux; the
    llama.cpp engines run on both (CPU on macOS in tests, GPU on Linux in
    production), so normalize here. Correct for the in-process ``llama_cpp``
    engine only; the served ``llama_server`` proxy must measure its child
    instead (:func:`subprocess_peak_memory`), because the model and KV cache
    live there, not in the proxy.
    """
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        peak *= 1024
    # int() guards the strict Memory model against platforms whose resource
    # stubs type ru_maxrss as a float.
    return Memory.from_bytes(int(peak))


def parse_vm_hwm(status_text: str) -> Memory | None:
    """Extract the ``VmHWM`` (peak RSS) line from a ``/proc/<pid>/status`` body.

    Split out from :func:`subprocess_peak_memory` so the parse is testable
    off-Linux. Returns ``None`` when the field is absent or malformed.
    """
    for line in status_text.splitlines():
        if line.startswith("VmHWM:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return Memory.from_bytes(int(parts[1]) * 1024)
            return None
    return None


def subprocess_peak_memory(pid: int) -> Memory | None:
    """Peak RSS of another process via ``/proc/<pid>/status`` (Linux only).

    The served ``llama_server`` engine holds its weights and KV cache in an
    external subprocess, so the proxy's own RSS would misattribute memory in
    telemetry (PR #536 review). Returns ``None`` off-Linux or when the proc
    entry is unreadable (process exited); callers report unmeasured (zero)
    rather than a misleading proxy figure.
    """
    try:
        status_text = Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return None
    return parse_vm_hwm(status_text)


@final
class StreamStatsClock:
    """Phase timer for a single streamed generation.

    Construct immediately before submitting the request, call
    :meth:`mark_piece` once per streamed token piece, and call :meth:`stats`
    when emitting the terminal chunk. Prefill is measured as request start to
    first piece and decode as first piece to last piece, so the derived rates
    match what the MLX engine reports for the same phases.

    ``now`` is injectable for tests; production uses ``time.perf_counter``.
    """

    def __init__(self, now: Callable[[], float] = time.perf_counter) -> None:
        self._now = now
        self._start = now()
        self._first: float | None = None
        self._last: float | None = None
        self._pieces = 0

    def mark_piece(self) -> None:
        """Record one streamed token piece arriving now."""
        moment = self._now()
        if self._first is None:
            self._first = moment
        self._last = moment
        self._pieces += 1

    @property
    def pieces(self) -> int:
        """Number of streamed token pieces recorded so far."""
        return self._pieces

    def stats(self, prompt_tokens: int, generation_tokens: int) -> GenerationStats:
        """Build final statistics from recorded phases and caller token counts.

        A stream that produced no pieces (or a single piece, making the decode
        span zero) reports 0.0 for the affected rate rather than inventing
        one: a zero reads as "unmeasured this request" while a fabricated
        rate would poison ledger medians.
        """
        first = self._first if self._first is not None else self._start
        last = self._last if self._last is not None else first
        prefill_seconds = max(first - self._start, 0.0)
        decode_seconds = max(last - first, 0.0)
        prompt_tps = prompt_tokens / prefill_seconds if prefill_seconds > 0 else 0.0
        generation_tps = (
            generation_tokens / decode_seconds if decode_seconds > 0 else 0.0
        )
        return GenerationStats(
            prompt_tps=prompt_tps,
            generation_tps=generation_tps,
            prompt_tokens=prompt_tokens,
            generation_tokens=generation_tokens,
            peak_memory_usage=process_peak_memory(),
        )


def stats_from_llama_server_timings(
    timings: dict[str, object],
) -> GenerationStats | None:
    """Convert ``llama-server``'s ``timings`` object into :class:`GenerationStats`.

    ``llama-server`` measures phases inside the engine (``prompt_n`` /
    ``prompt_ms`` for prefill, ``predicted_n`` / ``predicted_ms`` for decode),
    which is strictly more accurate than proxy-side wall clocks, so it wins
    whenever present. Returns ``None`` when the object does not carry the
    expected numeric fields (older servers, or a shape change), letting the
    caller fall back to wall-clock measurement.
    """

    def _number(key: str) -> float | None:
        value = timings.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return None

    prompt_n = _number("prompt_n")
    prompt_ms = _number("prompt_ms")
    predicted_n = _number("predicted_n")
    predicted_ms = _number("predicted_ms")
    if prompt_n is None or predicted_n is None:
        return None
    # On a slot-cache hit llama-server reports only the newly processed prompt
    # suffix as prompt_n, with the cached prefix in cache_n; the request's
    # true prompt size is their sum. The rates stay over processed tokens
    # only (prompt_ms measures exactly that work).
    cache_n = _number("cache_n") or 0.0
    prompt_tps = (
        prompt_n / (prompt_ms / 1000.0) if prompt_ms is not None and prompt_ms > 0 else 0.0
    )
    generation_tps = (
        predicted_n / (predicted_ms / 1000.0)
        if predicted_ms is not None and predicted_ms > 0
        else 0.0
    )
    return GenerationStats(
        prompt_tps=prompt_tps,
        generation_tps=generation_tps,
        prompt_tokens=int(prompt_n + cache_n),
        generation_tokens=int(predicted_n),
        peak_memory_usage=process_peak_memory(),
    )


def blocking_call_stats(
    usage: object, wall_seconds: float
) -> GenerationStats | None:
    """Statistics for a non-streamed (blocking) completion from its ``usage``.

    A blocking call exposes exact token counts but no phase split, so both
    rates are effective end-to-end rates over the whole request wall time
    (necessarily under-reporting each phase's true speed - still honest, and
    far better than the null that hid llama.cpp throughput entirely, #532).
    Returns ``None`` when ``usage`` lacks integer token counts.
    """
    if not isinstance(usage, dict):
        return None
    usage_map = cast("dict[str, object]", usage)
    prompt_tokens = usage_map.get("prompt_tokens")
    completion_tokens = usage_map.get("completion_tokens")
    if not isinstance(prompt_tokens, int) or not isinstance(completion_tokens, int):
        return None
    rate_window = wall_seconds if wall_seconds > 0 else 0.0
    return GenerationStats(
        prompt_tps=prompt_tokens / rate_window if rate_window else 0.0,
        generation_tps=completion_tokens / rate_window if rate_window else 0.0,
        prompt_tokens=prompt_tokens,
        generation_tokens=completion_tokens,
        peak_memory_usage=process_peak_memory(),
    )
