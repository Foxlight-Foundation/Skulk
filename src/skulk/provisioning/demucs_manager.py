"""Worker-owned preparation lifetime and verified native library observations.

This controller performs explicit preparation off the event loop. Read-only
inventory starts no download, subprocess or model. The caller supplies the fixed
reviewed package inventory and this machine's actual compatible platform tags.
There is no process-global environment override or import into Core's Python.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from collections.abc import Set as AbstractSet
from pathlib import Path
from typing import Protocol, final

from skulk.provisioning.demucs_runtime import (
    DemucsDevice,
    DemucsRuntimeCache,
    DemucsWheel,
    PreparationPhase,
    PreparedDemucsRuntime,
)
from skulk.provisioning.native_runtime_manager import NativeRuntimeManager


class RuntimeCache(Protocol):
    """Inject only cache execution; production retains its real byte/device checks."""

    def prepare(
        self,
        wheel: DemucsWheel,
        /,
        *,
        device: DemucsDevice,
        offline: bool,
        is_cancelled: Callable[[], bool],
        progress: Callable[[PreparationPhase], None],
        timeout_seconds: float = 30 * 60,
    ) -> PreparedDemucsRuntime:
        """Return actual verified libraries on the explicitly requested device."""
        ...

    def close(self) -> None:
        """Drain owned native installation before releasing its lifetime fences."""
        ...


@final
class DemucsRuntimeManager(
    NativeRuntimeManager[DemucsWheel, DemucsDevice, PreparedDemucsRuntime]
):
    """Retain separation's fixed CPU/Metal policy with shared native ownership."""

    def __init__(
        self,
        root: Path,
        packages: tuple[DemucsWheel, ...],
        compatible_platforms: AbstractSet[str],
        *,
        create_cache: Callable[[Path], RuntimeCache] = DemucsRuntimeCache,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        """Select exact separation inventory without filesystem or native effects."""
        super().__init__(
            root, packages, compatible_platforms,
            create_cache=create_cache,
            backends={"demucs-cpu": "cpu", "demucs-metal": "mps"},
            eligible=lambda package, device: device == "cpu" or (
                device == "mps" and package.platform_tag.startswith("macosx_")
            ),
            monotonic=monotonic,
        )
