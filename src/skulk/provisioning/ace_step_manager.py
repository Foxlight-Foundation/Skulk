"""Verified native ACE library inventory, independent of loaded model readiness."""

from __future__ import annotations

import time
from collections.abc import Callable
from collections.abc import Set as AbstractSet
from pathlib import Path
from typing import final

from skulk.provisioning.ace_step_runtime import (
    AceStepRuntimeCache,
    AceStepWheel,
    PreparedAceStepRuntime,
)
from skulk.provisioning.native_runtime_manager import NativeRuntimeManager, RuntimeCache
from skulk.shared.types.native_music import NativeMusicDevice, NativeMusicLibraryProbe


def _eligible(package: AceStepWheel, device: NativeMusicDevice) -> bool:
    if package.variant == "cpu":
        return device == "cpu" or (
            device == "mps" and package.platform_tag.startswith("macosx_")
        )
    return package.variant == device


@final
class AceStepRuntimeManager(
    NativeRuntimeManager[AceStepWheel, NativeMusicDevice, PreparedAceStepRuntime]
):
    """Publish exact built-in library lanes only after successful preparation.

    Mere package availability never enters resources. Native model loading still
    needs its independently verified components, actual precision/resource policy
    and supervised ready receipt before it can accept a music task.
    """

    def __init__(
        self,
        root: Path,
        packages: tuple[AceStepWheel, ...],
        compatible_platforms: AbstractSet[str],
        *,
        create_cache: Callable[
            [Path],
            RuntimeCache[AceStepWheel, NativeMusicDevice, PreparedAceStepRuntime],
        ] = AceStepRuntimeCache,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        """Bind exact inventory and actual host ABI without preparing any runtime."""
        platforms = frozenset(compatible_platforms)
        super().__init__(
            root,
            packages,
            platforms,
            create_cache=create_cache,
            backends={
                "ace_step-cpu": "cpu",
                "ace_step-metal": "mps",
                "ace_step-cuda": "cuda",
                "ace_step-rocm": "rocm",
            },
            eligible=lambda package, device: (
                _eligible(package, device)
                and package.required_platform_tags.issubset(platforms)
            ),
            monotonic=monotonic,
        )

    def library_probes(self) -> dict[str, NativeMusicLibraryProbe]:
        """Snapshot path-free executed precision facts for the current libraries.

        Hold the manager's observation lock while copying so a concurrent failed
        revalidation or shutdown cannot republish a withdrawn receipt. These
        library observations grant neither a loaded model nor memory admission.
        """
        with self._state:
            return (
                {
                    self._backend_for_device[device]: prepared.probe
                    for device, prepared in self._prepared.items()
                }
                if not self._stopped.is_set()
                else {}
            )
