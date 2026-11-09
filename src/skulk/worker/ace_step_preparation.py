"""Explicit native ACE preparation, separate from mounted model readiness."""

import time
from collections.abc import Callable
from typing import final

from skulk.provisioning.ace_step_manager import AceStepRuntimeManager
from skulk.provisioning.ace_step_runtime import AceStepWheel, PreparedAceStepRuntime
from skulk.shared.types.native_music import NativeMusicDevice, NativeMusicLibraryProbe
from skulk.worker.native_runtime_preparation import NativePreparationController


@final
class AceStepPreparationController(
    NativePreparationController[AceStepWheel, NativeMusicDevice, PreparedAceStepRuntime]
):
    """Own verified library work with the same drain/deadline policy as separation."""

    def __init__(
        self,
        manager: AceStepRuntimeManager,
        *,
        offline: bool,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """Bind the worker's manager and download policy without preparation effects."""
        super().__init__(manager, offline=offline, clock=clock)
        self._ace_manager = manager

    def library_probes(self) -> dict[str, NativeMusicLibraryProbe]:
        """Return current path-free math facts; closed controllers withdraw at once."""
        return {} if self._closed else self._ace_manager.library_probes()
