"""Exact native library preparation decisions stay bounded and separate from models."""

import pytest
from pydantic import TypeAdapter, ValidationError

from skulk.shared.apply import event_apply
from skulk.shared.models.tests.test_native_music_runtime import probe
from skulk.shared.types.commands import Command, PrepareAceStep
from skulk.shared.types.common import CommandId, NodeId
from skulk.shared.types.events import (
    AceStepPreparationCompleted,
    AceStepPreparationRequested,
    Event,
    is_persistable_control_event,
)
from skulk.shared.types.native_music import NativeMusicDevice
from skulk.shared.types.profiling import NodeResources
from skulk.shared.types.state import State


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda", "rocm"])
def test_native_preparation_round_trips_without_model_readiness(
    device: NativeMusicDevice,
) -> None:
    """Use actual command/event unions and resource validators for every lane."""
    command = PrepareAceStep(
        command_id=CommandId("prepare"),
        target_node=NodeId("worker"),
        owner_node=NodeId("api"),
        package_sha256="d" * 64,
        device=device,
        expires_at=1234567890.0,
    )
    assert TypeAdapter(Command).validate_json(command.model_dump_json()) == command
    requested = AceStepPreparationRequested(
        request_id=command.command_id,
        target_node=command.target_node,
        owner_node=command.owner_node,
        package_sha256=command.package_sha256,
        device=device,
        expires_at=1234567890.0,
    )
    observed = probe(device)
    backend = "ace_step-metal" if device == "mps" else f"ace_step-{device}"
    resources = NodeResources(
        backends=frozenset({backend}),
        engine_builds={backend: "ace_step@sha256:" + observed.payload_sha256},
        native_music_probes={backend: observed},
    )
    completed = AceStepPreparationCompleted(
        request_id=command.command_id,
        target_node=command.target_node,
        owner_node=command.owner_node,
        package_sha256=command.package_sha256,
        device=device,
        success=True,
        resources=resources,
    )
    state = State()
    for event in (requested, completed):
        assert TypeAdapter(Event).validate_json(event.model_dump_json()) == event
        assert is_persistable_control_event(event)
        assert event_apply(event, state) is state
        assert "interpreter" not in event.model_dump_json()
    assert not state.instances and not state.tasks


@pytest.mark.parametrize(
    "fault",
    ["no-resources", "error-success", "resources-failure", "no-error", "no-probe"],
)
def test_native_preparation_completion_cannot_publish_ambiguous_success(
    fault: str,
) -> None:
    """Installable-only evidence and mismatched success/failure facts are refused."""
    resources = NodeResources()
    with pytest.raises(ValidationError):
        AceStepPreparationCompleted(
            request_id=CommandId("prepare"),
            target_node=NodeId("worker"),
            owner_node=NodeId("api"),
            package_sha256="d" * 64,
            device="cpu",
            success=fault in {"no-resources", "error-success", "no-probe"},
            resources=None if fault in {"no-resources", "no-error"} else resources,
            error="failure"
            if fault in {"error-success", "resources-failure"}
            else None,
        )


def test_native_preparation_failure_is_durable_without_resources() -> None:
    """Failed verification settles this request without a health or capacity claim."""
    event = AceStepPreparationCompleted(
        request_id=CommandId("prepare"),
        target_node=NodeId("worker"),
        owner_node=NodeId("api"),
        package_sha256="d" * 64,
        device="rocm",
        success=False,
        error="Requested package/device verification failed.",
    )
    assert TypeAdapter(Event).validate_json(event.model_dump_json()) == event
    assert event.resources is None and is_persistable_control_event(event)


@pytest.mark.parametrize("deadline", [float("nan"), float("inf"), -1.0, 0.0])
def test_native_preparation_requires_finite_absolute_deadline(deadline: float) -> None:
    """A malformed ordered request cannot leave a worker with unlimited install work."""
    with pytest.raises(ValidationError):
        PrepareAceStep(
            target_node=NodeId("worker"),
            owner_node=NodeId("api"),
            package_sha256="d" * 64,
            device="cpu",
            expires_at=deadline,
        )
    with pytest.raises(ValidationError):
        AceStepPreparationRequested(
            request_id=CommandId("prepare"),
            target_node=NodeId("worker"),
            owner_node=NodeId("api"),
            package_sha256="d" * 64,
            device="cpu",
            expires_at=deadline,
        )
