"""Shared lifetime of verified built-in native audio library observations.

Preparation, observation publication, cancellation and retained cleanup have one
implementation. Engine-specific Core facades supply fixed package/lane mappings;
there is no dynamic plugin entry point or user-supplied native option.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping
from collections.abc import Set as AbstractSet
from pathlib import Path
from typing import Protocol

from skulk.provisioning.demucs_runtime import (
    DemucsBuildMismatchError,
    DemucsPreparationCancelledError,
    PreparationPhase,
    RuntimePackage,
)


class RuntimeObservation(Protocol):
    """Read-only original identity, installation and executed native lane."""

    @property
    def wheel(self) -> RuntimePackage:
        """Return the independently pinned verified engine package."""
        ...

    @property
    def generation(self) -> Path:
        """Return the stable verified installation path."""
        ...

    @property
    def device(self) -> str:
        """Return the successfully probed native compute device."""
        ...


class RuntimeCache[Package: RuntimePackage, Device: str, Prepared](Protocol):
    """Cache execution effect; actual production code owns byte/device checks."""

    def prepare(
        self, wheel: Package, /, *, device: Device, offline: bool,
        is_cancelled: Callable[[], bool], progress: Callable[[PreparationPhase], None],
        timeout_seconds: float = 30 * 60,
    ) -> Prepared:
        """Verify or install and probe the exact selected library profile."""
        ...

    def close(self) -> None:
        """Drain owned preparation before releasing lifetime fences."""
        ...


class NativeRuntimeManager[
    Package: RuntimePackage, Device: str, Prepared: RuntimeObservation
]:
    """Serialize preparation and publish only a final verified library observation.

    A mounted model still requires its own actual ready receipt and supervision.
    Keep this object for the worker lifetime, so failed cache cleanup retains its
    original handles. Shutdown cancels queued/current preparation and immediately
    withdraws observations; failed cleanup stays available for a later close retry.
    """

    def __init__(
        self,
        root: Path,
        packages: tuple[Package, ...],
        compatible_platforms: AbstractSet[str],
        *,
        create_cache: Callable[[Path], RuntimeCache[Package, Device, Prepared]],
        backends: Mapping[str, Device],
        eligible: Callable[[Package, Device], bool],
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        """Select immutable pins and observed host ABI without filesystem effects."""
        if (
            not 1 <= len(packages) <= 16
            or len({item.sha256 for item in packages}) != len(packages)
            or len({item.filename for item in packages}) != len(packages)
        ):
            raise ValueError("native runtime inventory requires distinct bounded package pins")
        self.root = root
        self._packages = {item.sha256: item for item in packages}
        self._platforms = frozenset(compatible_platforms)
        self._create_cache = create_cache
        self._backends = dict(backends)
        if not 1 <= len(backends) <= 4 or len(set(backends.values())) != len(backends):
            raise ValueError("native runtime lanes require distinct bounded backend mappings")
        self._backend_for_device = {device: backend for backend, device in backends.items()}
        self._eligible = eligible
        self._monotonic = monotonic
        self._state = threading.RLock()
        self._actor = threading.Lock()
        self._closing = threading.Lock()
        self._stopped = threading.Event()
        self._cache: RuntimeCache[Package, Device, Prepared] | None = None
        self._cache_closed = False
        self._prepared: dict[Device, Prepared] = {}

    def engine_builds(self) -> dict[str, str]:
        """Snapshot verified libraries only; candidates and loaded models differ."""
        with self._state:
            return {
                self._backend_for_device[device]: prepared.wheel.build_identity
                for device, prepared in self._prepared.items()
            }

    def prepared_runtime(
        self, backend: str, expected_build: str
    ) -> Prepared:
        """Snapshot the verified local runtime for an exact admitted compute build.

        The worker passes this private receipt directly to its runner process;
        paths never enter State. This lookup performs no install, device probe
        or model load. A withdrawn, mismatched or unimplemented lane is refused.
        """
        device = self._backends.get(backend)
        if device is None:
            raise DemucsBuildMismatchError("native runtime has no prepared compute lane")
        with self._state:
            prepared = self._prepared.get(device)
            if (
                self._stopped.is_set() or prepared is None
                or prepared.wheel.build_identity != expected_build
            ):
                raise DemucsBuildMismatchError("admitted native runtime is no longer prepared")
            return prepared

    def prepare(
        self,
        wheel: Package,
        *,
        device: Device,
        offline: bool,
        is_cancelled: Callable[[], bool],
        progress: Callable[[PreparationPhase], None],
        timeout_seconds: float = 30 * 60,
    ) -> Prepared:
        """Verify one eligible fixed package, including queue time in its budget.

        Offline requests delegate only verified cache reuse to the cache. No
        runtime lane can arise from an incompatible profile. Starting revalidation
        withdraws observations of this package before any new work; a failure
        leaves other packages/engines untouched. Cancellation or shutdown after
        native probing cannot publish late readiness.
        """
        if (
            self._packages.get(wheel.sha256) != wheel
            or wheel.platform_tag not in self._platforms
            or device not in self._backend_for_device
            or not self._eligible(wheel, device)
        ):
            raise DemucsBuildMismatchError("native package/device does not match this host's fixed inventory")
        if not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 2 * 60 * 60:
            raise ValueError("native preparation requires a finite deadline within two hours")
        deadline = self._monotonic() + timeout_seconds

        def checkpoint() -> None:
            if self._stopped.is_set() or is_cancelled():
                raise DemucsPreparationCancelledError("native runtime preparation cancelled")
            if self._monotonic() >= deadline:
                raise TimeoutError("native runtime preparation exceeded its complete budget")

        while True:
            checkpoint()
            if self._actor.acquire(timeout=0.05):
                break
        try:
            checkpoint()
            with self._state:
                previous: dict[Device, Prepared] = {
                    key: value for key, value in self._prepared.items()
                    if value.wheel.sha256 == wheel.sha256
                }
                self._prepared = {
                    key: value for key, value in self._prepared.items()
                    if value.wheel.sha256 != wheel.sha256
                }
                if self._cache is None:
                    self._cache = self._create_cache(self.root)
                cache = self._cache
            checkpoint()
            prepared = cache.prepare(
                wheel, device=device, offline=offline,
                is_cancelled=lambda: self._stopped.is_set() or is_cancelled(),
                progress=progress,
                timeout_seconds=deadline - self._monotonic(),
            )
            checkpoint()
            if prepared.wheel != wheel or prepared.device != device:
                raise DemucsBuildMismatchError("native cache receipt differs from requested package/device")
            with self._state:
                # Shutdown may race the last native probe. The same lock that
                # clears observations must serialize this final publication.
                checkpoint()
                # A successful full byte revalidation preserves prior device
                # probes only for that same immutable installation. A replaced
                # or repaired generation needs its own explicit device probes.
                for key, value in previous.items():
                    if value.generation == prepared.generation:
                        self._prepared[key] = value
                self._prepared[device] = prepared
            return prepared
        finally:
            self._actor.release()

    def close(self) -> None:
        """Withdraw immediately, cancel all waits and retain failed cleanup for retry."""
        with self._state:
            self._stopped.set()
            self._prepared.clear()
        if not self._closing.acquire(timeout=5):
            raise RuntimeError("native runtime manager cleanup remains in progress")
        try:
            # The cache refuses close during prepare. Withdraw first, then join
            # the serialized manager actor before retrying any retained native
            # cleanup. A slow drain stays owned and can be explicitly retried.
            if not self._actor.acquire(timeout=5):
                raise RuntimeError("native preparation is still draining; retry close")
            try:
                with self._state:
                    cache = None if self._cache_closed else self._cache
                if cache is not None:
                    cache.close()
                    with self._state:
                        self._cache_closed = True
            finally:
                self._actor.release()
        finally:
            self._closing.release()
