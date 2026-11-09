"""Typed built-in separator facade over the shared native preparation lifetime."""

from typing import final

from skulk.provisioning.demucs_runtime import (
    DemucsDevice,
    DemucsWheel,
    PreparedDemucsRuntime,
)
from skulk.worker.native_runtime_preparation import NativePreparationController


@final
class DemucsPreparationController(
    NativePreparationController[DemucsWheel, DemucsDevice, PreparedDemucsRuntime]
):
    """Prepare exact native separators with bounded cancellation and retained drain."""
