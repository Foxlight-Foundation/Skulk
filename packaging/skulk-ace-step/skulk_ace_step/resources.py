"""Observe actual native allocation/pressure without importing Torch into Core."""

from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, suppress
from typing import Protocol, cast, final

import psutil

from .contracts import (
    NativeMusicFrame,
    NativeMusicMemoryFrame,
    NativeMusicMemoryObservation,
    NativeMusicServeConfig,
    native_music_admission_capacity,
    native_music_live_memory_error,
    native_music_memory_budget_error,
)
from .lifetime import terminate_owned_group
from .rocm_memory import rocm_allocated_bytes


class _ResidentMemory(Protocol):
    """The pinned psutil process-memory tuple's resident-byte field."""

    rss: int


class _ProcessMemory(Protocol):
    """Narrow process sampling API independent of platform-specific tuple stubs."""

    def memory_info(self) -> _ResidentMemory:
        """Return resident memory for this actual process, without allocation credit."""
        ...


class MusicMemoryMonitor(Protocol):
    """Inject the resource lifetime in controlled server tests, never via the CLI."""

    def __call__(
        self,
        configuration: NativeMusicServeConfig,
        payload_sha256: str,
        emit: Callable[[NativeMusicFrame], None],
    ) -> AbstractContextManager[None]:
        """Begin before loading and join after request handling, including idle time."""
        ...


def sample_music_memory(
    configuration: NativeMusicServeConfig, payload_sha256: str
) -> NativeMusicMemoryObservation:
    """Measure this leader and its actual native device; never infer another GPU.

    RSS is not exclusive physical ownership. Torch reserved/device-driver
    allocations are process ceilings, not credits against a separate free query.
    This avoids cross-sample double-credit and unified-memory alias accounting.
    """
    import torch

    from .policy import sample_memory_capacity

    process = cast(_ProcessMemory, cast(object, psutil.Process(os.getpid())))
    allocated = (
        torch.mps.driver_allocated_memory()
        if configuration.device == "mps"
        else rocm_allocated_bytes()
        if configuration.device == "rocm"
        else torch.cuda.memory_reserved(0)
        if configuration.device == "cuda"
        else 0
    )
    return NativeMusicMemoryObservation(
        process_id=os.getpid(),
        payload_sha256=payload_sha256,
        device=configuration.device,
        capacity=sample_memory_capacity(configuration.device),
        host_resident_bytes=process.memory_info().rss,
        device_allocated_bytes=allocated,
    )


@final
class NativeMusicMemoryObserver:
    """Guard load, inference and idle residency in the same owned process group.

    The initial sample is synchronous, so failed sensing cannot start loading.
    Later failures terminate only the already isolated native group, including
    blocked model threads; its independent guardian retains file custody until
    actual drain. Core independently checks bounded heartbeats and ownership.
    """

    def __init__(
        self,
        configuration: NativeMusicServeConfig,
        payload_sha256: str,
        emit: Callable[[NativeMusicFrame], None],
        *,
        sample: Callable[
            [NativeMusicServeConfig, str], NativeMusicMemoryObservation
        ] = sample_music_memory,
        retire: Callable[[], None] = terminate_owned_group,
        clock_ms: Callable[[], int] = lambda: time.time_ns() // 1_000_000,
    ) -> None:
        """Bind a required admitted budget and injected bounded observation effects."""
        if configuration.memory_budget is None:
            raise ValueError(
                "native music serving requires its admitted live memory budget"
            )
        self.configuration, self.payload_sha256, self.emit = (
            configuration,
            payload_sha256,
            emit,
        )
        self.sample, self.retire, self.clock_ms = sample, retire, clock_ms
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._sequence = 0

    def _observe(self, *, initial: bool = False) -> None:
        observation = self.sample(self.configuration, self.payload_sha256)
        if (
            observation.process_id != os.getpid()
            or observation.payload_sha256 != self.payload_sha256
            or observation.device != self.configuration.device
        ):
            raise ValueError(
                "native memory sampler returned a foreign process or device"
            )
        budget = self.configuration.memory_budget
        assert budget is not None
        if initial:
            error = native_music_memory_budget_error(
                budget,
                native_music_admission_capacity(observation.capacity),
                device=self.configuration.device,
                now_unix_ms=self.clock_ms(),
            )
            if error is not None:
                raise MemoryError(error)
        error = native_music_live_memory_error(
            budget, observation, now_unix_ms=self.clock_ms()
        )
        if error is not None:
            raise MemoryError(error)
        self._sequence += 1
        self.emit(
            NativeMusicMemoryFrame(sequence=self._sequence, observation=observation)
        )

    def _watch(self) -> None:
        while not self._stop.wait(0.5):
            try:
                self._observe()
            except Exception as error:
                # This private stderr is bounded by Core. Continuing inference
                # after failed sensing/egress would silently remove its guard.
                try:
                    with suppress(OSError, ValueError):
                        print(
                            "Native music live resource guard stopped: "
                            + str(error)[:512],
                            file=sys.stderr,
                            flush=True,
                        )
                finally:
                    self._stop.set()
                    self.retire()
                return

    def start(self) -> None:
        """Verify one actual sample before creating the background owned monitor."""
        if self._thread is not None or self._stop.is_set():
            raise RuntimeError("native memory observer requires a new lifetime")
        self._observe(initial=True)
        self._thread = threading.Thread(
            target=self._watch, daemon=True, name="native-music-memory"
        )
        self._thread.start()

    def close(self) -> None:
        """Join sensing before output closes; a failed join requires group retirement."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            if self._thread.is_alive():
                raise TimeoutError("native memory observer did not drain")


@contextmanager
def observe_music_memory(
    configuration: NativeMusicServeConfig,
    payload_sha256: str,
    emit: Callable[[NativeMusicFrame], None],
) -> Iterator[None]:
    """Retain actual native sensing across model load, all calls and idle residency."""
    observer = NativeMusicMemoryObserver(configuration, payload_sha256, emit)
    try:
        observer.start()
        yield
    finally:
        observer.close()
