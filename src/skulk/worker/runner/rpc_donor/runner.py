"""RPC memory-donor runner for multi-node GGUF placements (#328).

A donor node's whole job is to serve ``ggml-rpc-server`` on the endpoint the
placement stamped for it (``LlamaRpcInstance.donor_endpoints``) and lend its
GPU memory to the driver's ``llama-server --rpc``. It never downloads or reads
the model file, never loads anything, and serves no requests itself, so its
lifecycle is deliberately minimal: spawn the server, health-check the port,
report ``RunnerReady``, then wait for ``Shutdown``. The worker plan loop skips
the download/load/warmup gates for donor shards; the driver's ``LoadModel`` is
gated on every donor reporting Ready first (so the endpoints answer before
llama-server dials them).

Failure semantics are inherited, not invented: if the subprocess dies the
runner raises, the supervisor sees the crash, and the peer-failure cascade
tears the whole instance down — correct, because llama-server SIGABRTs the
moment a scheduled-in donor disappears (measured on the Strix pair).
"""

import os
import signal
import socket
import subprocess
import time
from typing import final

from anyio import EndOfStream, WouldBlock

from skulk.shared.backends import RPC_SERVER_BIN_ENV, rpc_server_binary
from skulk.shared.types.events import (
    Event,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
)
from skulk.shared.types.tasks import Shutdown, Task, TaskId, TaskStatus
from skulk.shared.types.worker.instances import BoundInstance, LlamaRpcInstance
from skulk.shared.types.worker.runners import (
    RunnerIdle,
    RunnerReady,
    RunnerShutdown,
    RunnerShuttingDown,
    RunnerStatus,
)
from skulk.shared.types.worker.shards import RpcDonorShardMetadata
from skulk.utils.channels import MpReceiver, MpSender
from skulk.worker.runner.bootstrap import logger

# How long the donor waits for its ggml-rpc-server to accept a TCP connection
# before declaring the spawn failed. The server binds and listens almost
# immediately (no model to load), so this only papers over slow process start.
_HEALTH_DEADLINE_SECONDS: float = 30.0
_HEALTH_POLL_SECONDS: float = 0.25
# Between tasks the donor wakes at this cadence to verify its rpc-server
# subprocess is still alive (a dead server must crash the runner, not leave a
# Ready runner gossiping over a dead port).
_LIVENESS_POLL_SECONDS: float = 2.0


def _set_pdeathsig() -> None:
    """Linux: SIGKILL the subprocess if the runner process dies (reap backstop)."""
    try:
        import ctypes

        pr_set_pdeathsig = 1
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        _result: int = libc.prctl(pr_set_pdeathsig, signal.SIGKILL)  # pyright: ignore[reportAny]
    except Exception:  # noqa: BLE001 - best-effort on non-Linux
        pass


@final
class Runner:
    """Donor-side runner: owns one ``ggml-rpc-server`` subprocess.

    Reports ``RunnerIdle`` at creation (satisfying the worker's first-report
    deadline), spawns and health-checks the RPC server before entering the task
    loop, then reports ``RunnerReady``. The only task it services is
    ``Shutdown``; generation traffic never reaches a donor (the plan loop
    forwards inference tasks to the driver only).
    """

    def __init__(
        self,
        bound_instance: BoundInstance,
        event_sender: MpSender[Event],
        task_receiver: MpReceiver[Task],
        cancel_receiver: MpReceiver[TaskId],
        context_token_limit: int | None = None,
    ):
        self.event_sender = event_sender
        self.task_receiver = task_receiver
        self.cancel_receiver = cancel_receiver
        self.bound_instance = bound_instance
        instance = bound_instance.instance
        if not isinstance(instance, LlamaRpcInstance):
            raise RuntimeError(
                "RPC donor runner requires a LlamaRpcInstance, got "
                f"{instance.__class__.__name__}"
            )
        if not isinstance(bound_instance.bound_shard, RpcDonorShardMetadata):
            raise RuntimeError(
                "RPC donor runner requires an RpcDonorShardMetadata shard, got "
                f"{bound_instance.bound_shard.__class__.__name__}"
            )
        endpoint = instance.donor_endpoints.get(bound_instance.bound_node_id)
        if endpoint is None:
            raise RuntimeError(
                f"No donor endpoint stamped for node {bound_instance.bound_node_id} "
                f"on instance {instance.instance_id}"
            )
        host, _, port = endpoint.rpartition(":")
        self.bind_host = host
        self.bind_port = int(port)
        self.runner_id = bound_instance.bound_runner_id
        self.server_proc: subprocess.Popen[bytes] | None = None
        self.current_status: RunnerStatus = RunnerIdle()
        logger.info(
            f"rpc-donor runner created (endpoint={self.bind_host}:{self.bind_port})"
        )
        self.update_status(RunnerIdle())

    def update_status(self, status: RunnerStatus) -> None:
        """Record and broadcast this runner's status."""
        self.current_status = status
        self.event_sender.send(
            RunnerStatusUpdated(
                runner_id=self.runner_id, runner_status=self.current_status
            )
        )

    def main(self) -> None:
        """Spawn the RPC server, report Ready, then wait for Shutdown."""
        try:
            self._spawn_rpc_server()
            self._await_listening()
            self.update_status(RunnerReady())
            logger.info(
                f"rpc-donor serving on {self.bind_host}:{self.bind_port} "
                f"(pid={self.server_proc.pid if self.server_proc else '?'})"
            )
            with self.task_receiver as tasks:
                while True:
                    try:
                        task = tasks.receive_timeout(_LIVENESS_POLL_SECONDS)
                    except WouldBlock:
                        # No task: verify the rpc-server subprocess is still
                        # alive. A donor whose server died must crash the
                        # runner (supervisor cascade tears down the instance)
                        # rather than gossip Ready over a dead port.
                        self._ensure_server_alive()
                        continue
                    except EndOfStream:
                        break
                    self._handle_task(task)
                    if isinstance(self.current_status, RunnerShutdown):
                        break
        finally:
            # Never leave the RPC server running past the runner loop
            # (PR_SET_PDEATHSIG is the SIGKILL backstop on Linux).
            self._teardown_server()

    def _ensure_server_alive(self) -> None:
        """Raise if the spawned ggml-rpc-server exited behind our back."""
        proc = self.server_proc
        if proc is None or isinstance(
            self.current_status, (RunnerShuttingDown, RunnerShutdown)
        ):
            return
        if proc.poll() is not None:
            raise RuntimeError(
                f"ggml-rpc-server exited unexpectedly (code {proc.returncode})"
            )

    def _handle_task(self, task: Task) -> None:
        match task:
            case Shutdown():
                logger.info("rpc-donor runner shutting down")
                self.update_status(RunnerShuttingDown())
                self.event_sender.send(TaskAcknowledged(task_id=task.task_id))
                self._teardown_server()
                self.event_sender.send(
                    TaskStatusUpdated(
                        task_id=task.task_id, task_status=TaskStatus.Complete
                    )
                )
                self.update_status(RunnerShutdown())
            case _:
                # Donors serve no model: any other task reaching one is a plan
                # bug; fail loud so it surfaces instead of silently stalling a
                # request.
                raise RuntimeError(
                    f"rpc-donor runner received unsupported task "
                    f"{task.__class__.__name__}"
                )

    def _spawn_rpc_server(self) -> None:
        binary = rpc_server_binary()
        if binary is None:
            raise RuntimeError(
                f"No ggml-rpc-server binary: set {RPC_SERVER_BIN_ENV} or place "
                "ggml-rpc-server next to SKULK_LLAMA_SERVER_BIN (build llama.cpp "
                "with -DGGML_RPC=ON; the target is ggml-rpc-server)"
            )
        # -c: local tensor cache. The driver pushes the model's tensors over the
        # network on first load; the cache persists them on the donor's disk so
        # reloads of the same model skip the transfer (measured: ~2min reload).
        cmd = [
            binary,
            "-H",
            self.bind_host,
            "-p",
            str(self.bind_port),
            "-c",
        ]
        env = os.environ.copy()
        bin_dir = os.path.dirname(os.path.realpath(binary))
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{bin_dir}:{existing}" if existing else bin_dir
        logger.info("launching ggml-rpc-server: " + " ".join(cmd))
        self.server_proc = subprocess.Popen(  # noqa: S603 - args built here, not user input
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            preexec_fn=_set_pdeathsig,  # noqa: PLW1509 - Linux reap-on-parent-death
        )

    def _await_listening(self) -> None:
        """Poll the bound endpoint until it accepts a TCP connection."""
        deadline = time.monotonic() + _HEALTH_DEADLINE_SECONDS
        while time.monotonic() < deadline:
            if self.server_proc is not None and self.server_proc.poll() is not None:
                raise RuntimeError(
                    "ggml-rpc-server exited during startup "
                    f"(code {self.server_proc.returncode})"
                )
            try:
                with socket.create_connection(
                    (self.bind_host, self.bind_port), timeout=1.0
                ):
                    return
            except OSError:
                time.sleep(_HEALTH_POLL_SECONDS)
        raise RuntimeError(
            f"ggml-rpc-server did not accept connections on "
            f"{self.bind_host}:{self.bind_port} within "
            f"{_HEALTH_DEADLINE_SECONDS:.0f}s"
        )

    def _teardown_server(self) -> None:
        """Terminate the RPC server: SIGTERM, then SIGKILL after a grace."""
        proc = self.server_proc
        self.server_proc = None
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            _ = proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                _ = proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.error("ggml-rpc-server did not die after SIGKILL")
