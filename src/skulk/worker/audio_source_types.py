"""Structural contracts shared by admitted audio source-custody owners.

The actual task remains a strict typed State record. These read-only protocols
let music editing reuse separation's bounded files and crash fences without
recasting an edit as a separator or accepting a caller-supplied source path.
"""

from typing import Protocol

from skulk.shared.types.audio_separation import SeparationSource
from skulk.shared.types.common import CommandId, NodeId
from skulk.shared.types.tasks import TaskId, TaskStatus
from skulk.shared.types.worker.instances import InstanceId


class AudioSourceParams(Protocol):
    """Minimal immutable original facts needed by the shared encoded WAV store."""

    @property
    def model(self) -> str:
        """Return the exact model alias of the admitted source-consuming task."""
        ...

    @property
    def source(self) -> SeparationSource:
        """Return independently verified original identity and complete geometry."""
        ...

    @property
    def total_input_chunks(self) -> int:
        """Return the exact bounded addressed transfer extent."""
        ...


class AudioSourceTask(Protocol):
    """Authoritative accepted task identity; packets cannot create this authority."""

    @property
    def command_id(self) -> CommandId:
        """Return the accepting API's correlated operation identity."""
        ...

    @property
    def task_id(self) -> TaskId:
        """Return the master's exact placed task identity."""
        ...

    @property
    def instance_id(self) -> InstanceId:
        """Return the single-host mounted instance that may consume the original."""
        ...

    @property
    def owner_node(self) -> NodeId:
        """Return the only API permitted to supply this original."""
        ...

    @property
    def task_status(self) -> TaskStatus:
        """Return nonterminal admission truth before accepting source bytes."""
        ...

    @property
    def task_params(self) -> AudioSourceParams:
        """Return immutable source facts, independent of the inference operation."""
        ...
