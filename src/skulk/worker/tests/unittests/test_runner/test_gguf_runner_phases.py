"""Runner-phase flight-recorder emission for the GGUF-engine runners (#479).

The llama_cpp / llama_server / rpc_donor runners historically emitted no
runner-phase diagnostics at all, so the observability Live tab sat at
"created" forever for any GGUF placement. These tests capture the diagnostic
channel directly and assert the key lifecycle emissions exist, using the RPC
donor runner (no model, no GPU) as the cheap representative of the shared
runner skeleton.
"""

import subprocess
import sys
import time
from collections.abc import Iterator

import pytest

from skulk.shared.types.diagnostics import (
    RunnerDiagnosticContext,
    RunnerDiagnosticUpdate,
)
from skulk.shared.types.tasks import Shutdown, TaskId
from skulk.utils.channels import MpReceiver, WouldBlock, mp_channel
from skulk.worker.runner.diagnostics import configure_runner_diagnostics
from skulk.worker.tests.unittests.test_runner.test_rpc_donor_liveness import (
    _donor_runner,  # pyright: ignore[reportPrivateUsage]
)


@pytest.fixture
def phase_updates() -> Iterator[MpReceiver[RunnerDiagnosticUpdate]]:
    """Route runner-phase emissions into a capturable channel, then unwire."""
    sender, receiver = mp_channel[RunnerDiagnosticUpdate]()
    context = RunnerDiagnosticContext(
        node_id="test-node",
        runner_id="test-runner",
        instance_id="test-instance",
        model_id="test/pooled-gguf",
        rank=1,
        world_size=2,
        start_layer=0,
        end_layer=0,
        n_layers=36,
    )
    configure_runner_diagnostics(sender, context)
    try:
        yield receiver
    finally:
        configure_runner_diagnostics(None, context)


def _drain(receiver: MpReceiver[RunnerDiagnosticUpdate]) -> list[RunnerDiagnosticUpdate]:
    """Collect every queued update, tolerating mp.Queue feeder-thread latency.

    A multiprocessing queue transfers items through a background feeder
    thread, so an immediate receive after send can spuriously report empty;
    poll briefly and stop once the queue stays quiet.
    """
    updates: list[RunnerDiagnosticUpdate] = []
    deadline = time.monotonic() + 2.0
    quiet_until = time.monotonic() + 0.2
    while time.monotonic() < deadline:
        try:
            updates.append(receiver.receive_nowait())
            quiet_until = time.monotonic() + 0.2
        except WouldBlock:
            # Return only after the queue stays quiet: consecutive sends can
            # flush through the feeder thread at different times.
            if updates and time.monotonic() >= quiet_until:
                return updates
            time.sleep(0.02)
    return updates


def test_dead_subprocess_records_error_phase(
    phase_updates: MpReceiver[RunnerDiagnosticUpdate],
) -> None:
    runner = _donor_runner()
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    _ = proc.wait(timeout=10)
    runner.server_proc = proc
    with pytest.raises(RuntimeError, match="exited unexpectedly"):
        runner._ensure_server_alive()  # pyright: ignore[reportPrivateUsage]

    updates = _drain(phase_updates)
    assert any(
        update.phase == "error" and update.event == "server_exited"
        for update in updates
    )


def test_shutdown_records_cleanup_phases(
    phase_updates: MpReceiver[RunnerDiagnosticUpdate],
) -> None:
    runner = _donor_runner()
    # No server was ever spawned; teardown is a no-op, but the lifecycle
    # phases must still be recorded for the Live tab.
    runner._handle_task(  # pyright: ignore[reportPrivateUsage]
        Shutdown(
            task_id=TaskId(),
            instance_id=runner.bound_instance.instance.instance_id,
            runner_id=runner.runner_id,
        )
    )

    events = [(update.phase, update.event) for update in _drain(phase_updates)]
    assert ("shutdown_cleanup", "runner_shutdown_requested") in events
    assert ("shutdown_cleanup", "server_teardown_complete") in events
