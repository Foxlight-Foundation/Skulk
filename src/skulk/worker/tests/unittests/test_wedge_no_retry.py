"""GPU-wedge deaths are never retried (the wedge-exit wired-memory leak).

A runner killed by the deadline watchdog mid-wedge leaks ~a shard of wired
GPU memory (measured 2026-06-09; reboot-only recovery), so the worker must
give the instance up on the FIRST wedge death instead of relaunching —
especially because wedges take ~300s each and would never trip the
3-in-60s crash window.
"""

from skulk.download.download_utils import MODEL_FILES_INCOMPLETE_MARKER
from skulk.shared.models.model_cards import ModelCard, ModelTask
from skulk.shared.models.remote_code_approval import (
    MODEL_TRUST_FAILURE_MARKER,
)
from skulk.shared.types.common import ModelId
from skulk.shared.types.memory import Memory
from skulk.shared.types.state import State
from skulk.shared.types.worker.runners import RunnerFailed, RunnerReady
from skulk.worker.main import (
    _model_files_incomplete_message,  # pyright: ignore[reportPrivateUsage] — unit under test
    _model_load_trust_failure_message,  # pyright: ignore[reportPrivateUsage] — unit under test
    _require_worker_model_execution_identity,  # pyright: ignore[reportPrivateUsage] — unit under test
    _runner_failed_wedged,  # pyright: ignore[reportPrivateUsage] — unit under test
    model_files_incomplete_live_instances,
    model_trust_failed_live_instances,
)
from skulk.worker.runner.bootstrap import WEDGE_EXIT_CODE, WEDGE_FAILURE_MARKER


def test_marker_failure_is_wedged() -> None:
    status = RunnerFailed(
        error_message=(
            f"Terminated ({WEDGE_FAILURE_MARKER}: deadline watchdog declared "
            "a GPU wedge (faulted Metal eval); wired memory may have leaked)"
        )
    )
    assert _runner_failed_wedged(status)


def test_ordinary_failures_are_not_wedged() -> None:
    assert not _runner_failed_wedged(
        RunnerFailed(error_message="Terminated (signal=6 (Abort trap: 6))")
    )
    assert not _runner_failed_wedged(RunnerFailed(error_message=None))
    assert not _runner_failed_wedged(RunnerReady())
    assert not _runner_failed_wedged(None)


def test_wedge_exit_code_is_distinct_from_common_codes() -> None:
    # 0 = clean, 1 = generic python failure, <0 = signals; the watchdog's
    # code must not collide with any of them or the supervisor would
    # misclassify ordinary deaths as wedges (and stop retrying transient
    # failures) or vice versa.
    assert WEDGE_EXIT_CODE not in (0, 1)
    assert WEDGE_EXIT_CODE > 0


def test_supervisor_maps_wedge_exit_code_to_marker() -> None:
    # Mirror the supervisor's cause-classification logic for the wedge code:
    # the marker must round-trip into RunnerFailed.error_message so the
    # worker-side matcher (_runner_failed_wedged) fires on it.
    rc: int = WEDGE_EXIT_CODE
    if rc < 0:
        cause = f"signal={-rc}"
    elif rc == WEDGE_EXIT_CODE:
        cause = (
            f"{WEDGE_FAILURE_MARKER}: deadline watchdog declared a GPU "
            "wedge (faulted Metal eval); wired memory may have leaked"
        )
    else:
        cause = f"exitcode={rc}"
    assert _runner_failed_wedged(RunnerFailed(error_message=f"Terminated ({cause})"))


def test_wedged_live_instances_sweep() -> None:
    """The planning-tick sweep catches LOCAL wedge deaths (single-node case).

    plan._kill_runner never emits Shutdown for a locally failed runner while
    its instance lives, so the sweep is the only path that frees a
    single-node placement from a wedged-dead runner.
    """
    from types import SimpleNamespace
    from typing import cast

    from skulk.worker.main import (
        _wedged_live_instances,  # pyright: ignore[reportPrivateUsage] — unit under test
    )

    def supervisor(instance_id: str, model_id: str, status: object):
        return SimpleNamespace(
            bound_instance=SimpleNamespace(
                instance=SimpleNamespace(instance_id=instance_id)
            ),
            shard_metadata=SimpleNamespace(
                model_card=SimpleNamespace(model_id=model_id)
            ),
            status=status,
        )

    wedged = RunnerFailed(
        error_message=f"Terminated ({WEDGE_FAILURE_MARKER}: ...)"
    )
    ordinary = RunnerFailed(error_message="Terminated (signal=6)")
    runners = cast(
        "dict[object, object]",
        {
            "r-wedged-live": supervisor("inst-a", "model-a", wedged),
            "r-wedged-deleted": supervisor("inst-gone", "model-b", wedged),
            "r-ordinary-failure": supervisor("inst-c", "model-c", ordinary),
            "r-healthy": supervisor("inst-d", "model-d", RunnerReady()),
        },
    )

    from skulk.shared.types.worker.instances import InstanceId
    from skulk.shared.types.worker.runners import RunnerId
    from skulk.worker.runner.runner_supervisor import RunnerSupervisor

    result = _wedged_live_instances(
        cast("dict[RunnerId, RunnerSupervisor]", cast(object, runners)),
        cast("set[InstanceId]", {"inst-a", "inst-c", "inst-d"}),
    )
    # Only the wedge-marked runner whose instance still lives is reported:
    # deleted instances follow the normal Shutdown cleanup, ordinary failures
    # keep the 3-in-60s breaker semantics, healthy runners are untouched.
    assert result == [("inst-a", "model-a")]


def test_model_trust_failure_is_terminal_for_live_instance() -> None:
    """A deterministic trust denial is selected for immediate teardown once."""
    from types import SimpleNamespace
    from typing import cast

    from skulk.shared.types.worker.instances import InstanceId
    from skulk.shared.types.worker.runners import RunnerId
    from skulk.worker.runner.runner_supervisor import RunnerSupervisor

    denied = SimpleNamespace(
        bound_instance=SimpleNamespace(instance=SimpleNamespace(instance_id="inst-a")),
        shard_metadata=SimpleNamespace(
            model_card=SimpleNamespace(model_id="org/model")
        ),
        status=RunnerFailed(
            error_message=f"{MODEL_TRUST_FAILURE_MARKER}: approval required"
        ),
    )
    ordinary = SimpleNamespace(
        bound_instance=SimpleNamespace(instance=SimpleNamespace(instance_id="inst-b")),
        shard_metadata=SimpleNamespace(
            model_card=SimpleNamespace(model_id="org/other")
        ),
        status=RunnerFailed(error_message="ordinary engine crash"),
    )
    runners = cast(
        "dict[RunnerId, RunnerSupervisor]",
        cast(object, {"runner-a": denied, "runner-b": ordinary}),
    )

    assert model_trust_failed_live_instances(
        runners,
        cast("set[InstanceId]", {"inst-a", "inst-b"}),
    ) == [("inst-a", "org/model", f"{MODEL_TRUST_FAILURE_MARKER}: approval required")]


def test_durable_model_trust_failure_excludes_raw_exception_detail() -> None:
    """Only classified trust guidance is suitable for replicated failure truth."""
    from skulk.worker.main import (
        _model_trust_instance_failure_message,  # pyright: ignore[reportPrivateUsage] - durable boundary under test
    )

    message = _model_trust_instance_failure_message(ModelId("org/model"))

    assert message == (
        "runner for org/model was refused by immutable model trust policy; "
        "not retrying until the card approval or installed artifact identity changes."
    )


def test_missing_model_files_are_terminal_but_not_a_trust_refusal() -> None:
    """Incomplete files carry their own non-retry marker, not a trust denial.

    A download whose index lists a weight the folder lacks is a file problem;
    labeling it a trust refusal sent operators to card approval instead of a
    fresh download.
    """
    message = _model_files_incomplete_message(FileNotFoundError("model missing"))

    assert message.startswith(f"{MODEL_FILES_INCOMPLETE_MARKER}:")
    assert MODEL_TRUST_FAILURE_MARKER not in message
    assert "model missing" in message


def test_trust_refusals_keep_the_trust_marker() -> None:
    """An identity refusal is still reported as a trust failure."""
    message = _model_load_trust_failure_message(PermissionError("identity mismatch"))

    assert message.startswith(f"{MODEL_TRUST_FAILURE_MARKER}:")
    assert MODEL_FILES_INCOMPLETE_MARKER not in message


def test_incomplete_model_files_are_terminal_for_live_instance() -> None:
    """Incomplete files are selected for immediate teardown, apart from trust."""
    from types import SimpleNamespace
    from typing import cast

    from skulk.shared.types.worker.instances import InstanceId
    from skulk.shared.types.worker.runners import RunnerId
    from skulk.worker.runner.runner_supervisor import RunnerSupervisor

    def supervisor(instance_id: str, model_id: str, message: str) -> object:
        return SimpleNamespace(
            bound_instance=SimpleNamespace(
                instance=SimpleNamespace(instance_id=instance_id)
            ),
            shard_metadata=SimpleNamespace(
                model_card=SimpleNamespace(model_id=model_id)
            ),
            status=RunnerFailed(error_message=message),
        )

    incomplete = f"{MODEL_FILES_INCOMPLETE_MARKER}: weight missing"
    runners = cast(
        "dict[RunnerId, RunnerSupervisor]",
        cast(
            object,
            {
                "runner-a": supervisor("inst-a", "org/model", incomplete),
                "runner-b": supervisor(
                    "inst-b", "org/trust", f"{MODEL_TRUST_FAILURE_MARKER}: denied"
                ),
                "runner-c": supervisor("inst-gone", "org/gone", incomplete),
            },
        ),
    )
    live = cast("set[InstanceId]", {"inst-a", "inst-b"})

    assert model_files_incomplete_live_instances(runners, live) == [
        ("inst-a", "org/model", incomplete)
    ]
    assert [entry[0] for entry in model_trust_failed_live_instances(runners, live)] == [
        "inst-b"
    ]


def test_durable_incomplete_files_failure_names_the_remedy() -> None:
    """The replicated failure text is classified guidance, not raw detail."""
    from skulk.worker.main import (
        _model_files_incomplete_instance_failure_message,  # pyright: ignore[reportPrivateUsage] - durable boundary under test
    )

    message = _model_files_incomplete_instance_failure_message(ModelId("org/model"))

    assert message.startswith("runner for org/model found the model's files incomplete")
    assert "downloaded again" in message
    assert "trust" not in message


def test_worker_needs_no_secondary_approval_for_published_card() -> None:
    """Runner creation accepts the authorization carried by publication."""

    card = ModelCard(
        model_id=ModelId("org/model"),
        storage_size=Memory.from_mb(100),
        n_layers=4,
        hidden_size=64,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        source_revision="a" * 40,
        registry_card_id="card_" + "a" * 52,
        registry_snapshot_id="snapshot-test",
        registry_provenance="agent",
        trust_remote_code=True,
    )
    _require_worker_model_execution_identity(
        card,
        State(model_trust_approved_remote_code_identities=()),
    )
    _require_worker_model_execution_identity(card, State())
