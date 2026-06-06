import contextlib
import gc
import os
import resource
import signal
import threading
import time
from collections.abc import Callable, Iterator

import loguru

from exo.shared.constants import preferred_env_value
from exo.shared.models.model_cards import RuntimeCapabilityCardConfig
from exo.shared.types.diagnostics import RunnerDiagnosticContext, RunnerDiagnosticUpdate
from exo.shared.types.events import Event, RunnerStatusUpdated
from exo.shared.types.tasks import Task, TaskId
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runners import RunnerFailed
from exo.utils.channels import ClosedResourceError, MpReceiver, MpSender
from exo.worker.runner.diagnostics import (
    configure_runner_diagnostics,
    record_runner_phase,
)

logger: "loguru.Logger" = loguru.logger

# Set by signal handler so the layer-by-layer loading loop can bail early.
shutdown_requested: bool = False


FAST_SYNCH_CLUSTER_DEFAULT: bool = True
"""Cluster-wide default for ``MLX_METAL_FAST_SYNCH`` when neither the operator
nor the model card has expressed a preference. Currently True for backwards
compatibility with existing deployments. Flipping this default is tracked
under the env-var → typed-config migration; do not flip without measuring
the perf delta on a representative workload first."""


def _card_declares_speculative_decoding(
    card_runtime: RuntimeCapabilityCardConfig | None,
) -> bool:
    """True when the card declares any speculative-decoding mechanism.

    Covers both the Qwen3/DeepSeek embedded-head convention
    (``mtp_heads`` / ``mtp_sidecar_repo``) and the Gemma 4 companion
    assistant convention (``assistant_model_repo``).
    """
    if card_runtime is None:
        return False
    return bool(
        card_runtime.mtp_heads
        or card_runtime.mtp_sidecar_repo
        or card_runtime.assistant_model_repo
    )


def resolve_metal_fast_synch(card_runtime: RuntimeCapabilityCardConfig | None) -> bool:
    """Resolve the effective ``MLX_METAL_FAST_SYNCH`` setting for this runner.

    Priority order, highest to lowest:

    1. **Operator override.** ``SKULK_FAST_SYNCH`` (or legacy ``EXO_FAST_SYNCH``)
       set to ``"on"`` or ``"off"``. Set by the CLI ``--fast-synch`` /
       ``--no-fast-synch`` flags or directly in the runner environment.
       This is the escape hatch when a card preference is wrong in the field.
    2. **Model card.** ``runtime.metal_fast_synch`` on the bound shard's model
       card. Pinned per-model when the model is known to deadlock or
       known to benefit measurably from FAST_SYNCH.
    3. **Speculative-decoding default.** Cards that declare an MTP sidecar,
       embedded MTP heads, or a companion assistant model default to
       ``False``. FAST_SYNCH is catastrophically incompatible with the
       speculative decoding loop: measured 2026-06-06 on Qwen3.5-9B-4bit
       (M4, mlx 0.31.2), the same MTP loop runs 27.7 tok/s with
       ``MLX_METAL_FAST_SYNCH=0`` and 0.6 tok/s with ``=1`` — a 46x
       collapse — while vanilla decode is unaffected (20.8 vs 20.7 tok/s).
       The per-round pattern of small evals across streams that
       speculative decoding requires is exactly the shape FAST_SYNCH's
       completion-signal path pathologizes.
    4. **Cluster default.** ``FAST_SYNCH_CLUSTER_DEFAULT`` (True today).

    Returns ``True`` when ``MLX_METAL_FAST_SYNCH`` should be ``"1"``.
    """
    override = preferred_env_value("SKULK_FAST_SYNCH", "EXO_FAST_SYNCH")
    if override is not None:
        normalized = override.strip().lower()
        if normalized == "on":
            return True
        if normalized == "off":
            return False
        # Anything else (empty string, "auto", garbage) falls through to the
        # next layer rather than silently picking a value. The CLI surface
        # only ever sets "on" or "off", so reaching this branch means
        # someone set the env var by hand to an unknown value.
    if card_runtime is not None and card_runtime.metal_fast_synch is not None:
        return card_runtime.metal_fast_synch
    if _card_declares_speculative_decoding(card_runtime):
        return False
    return FAST_SYNCH_CLUSTER_DEFAULT


WARMUP_DEADLINE_SECONDS_DEFAULT: float = 300.0
"""How long a runner may spend in warmup before it is declared wedged.

Healthy warmups finish in seconds (kernel compile + a 1-token generation);
five minutes is far beyond any legitimate cold start while still bounding
the failure. The canonical wedge (launch smoke, 2026-06-05): a Metal fault
leaves ``mx.eval`` parked in ``IOSurfaceSharedEvent`` forever at 0% CPU —
uninterruptible from Python — and the runner sits in ``RunnerWarmingUp``
silently blocking ALL task dispatch on the node. Override with
``SKULK_WARMUP_DEADLINE_SECONDS``.
"""


def resolve_warmup_deadline_seconds() -> float:
    """Resolve the warmup deadline, honoring the operator env override."""
    raw = preferred_env_value(
        "SKULK_WARMUP_DEADLINE_SECONDS", "EXO_WARMUP_DEADLINE_SECONDS"
    )
    if raw is not None:
        try:
            parsed = float(raw)
            if parsed > 0:
                return parsed
        except ValueError:
            pass
        logger.warning(
            f"Ignoring invalid SKULK_WARMUP_DEADLINE_SECONDS value {raw!r}; "
            f"using default {WARMUP_DEADLINE_SECONDS_DEFAULT:.0f}s"
        )
    return WARMUP_DEADLINE_SECONDS_DEFAULT


@contextlib.contextmanager
def deadline_watchdog(
    seconds: float,
    description: str,
    on_timeout: Callable[[], None] | None = None,
) -> Iterator[None]:
    """Hard deadline for a block that may wedge uninterruptibly.

    A wedged Metal eval blocks inside the driver at 0% CPU and cannot be
    interrupted by signals or async timeouts — the only reliable escape is
    process exit (the supervisor observes the death and reports
    ``RunnerFailed``; Metal memory is reclaimed on exit). A daemon thread
    waits out the deadline and, if the block has not finished, logs a
    CRITICAL diagnosis and terminates the process via ``os._exit``.

    Args:
        seconds: Deadline for the wrapped block.
        description: Human-readable name of the operation (appears in the
            CRITICAL log line).
        on_timeout: Test seam — replaces the default log-and-``os._exit``
            action when provided.
    """
    finished = threading.Event()

    def _default_timeout_action() -> None:
        logger.critical(
            f"{description} exceeded its {seconds:.0f}s deadline — the GPU "
            "may be wedged (a faulted Metal eval blocks forever at 0% CPU). "
            "Terminating this runner so the node can keep dispatching. If "
            "runners keep dying here, test the GPU with a small matmul; if "
            "that hangs too, the machine needs a reboot to reset the Metal "
            "device queue."
        )
        _release_metal_resources()
        os._exit(1)

    action = on_timeout if on_timeout is not None else _default_timeout_action

    def _watch() -> None:
        if not finished.wait(seconds):
            action()

    watcher = threading.Thread(
        target=_watch, name="runner-deadline-watchdog", daemon=True
    )
    watcher.start()
    try:
        yield
    finally:
        finished.set()


def _release_metal_resources() -> None:
    """Best-effort release of Metal/MLX resources before process exit.

    Clears the MLX buffer cache and runs garbage collection so that Metal
    wired memory is returned to the OS instead of leaking when the runner
    subprocess is terminated mid-load.
    """
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:
        pass
    gc.collect()


def _install_parent_death_watchdog(initial_ppid: int, poll_seconds: float = 1.0) -> None:
    """Self-terminate the runner when the agent supervisor dies.

    The runner is a ``mp.Process(daemon=True)`` child of the agent. ``daemon=True``
    only kills the child on a *clean* Python interpreter exit in the parent;
    when the agent receives SIGKILL the child is reparented to launchd (pid 1
    on macOS) and continues holding 4-5 GB of unified GPU memory until killed
    explicitly. Without this watchdog, killing the agent leaves wired Metal
    memory allocated, which is the failure mode that today forces a node
    restart.

    A dedicated daemon thread polls ``os.getppid()`` once per second; if it
    changes from the original supervisor pid we run a best-effort
    ``mx.clear_cache()`` and ``os._exit(1)``. We use ``os._exit`` rather than
    ``sys.exit`` so we bypass any Python-level finally/atexit hooks that could
    block on broken multiprocessing pipes.
    """

    def watchdog() -> None:
        while True:
            time.sleep(poll_seconds)
            try:
                current_ppid = os.getppid()
            except OSError:
                continue
            if current_ppid != initial_ppid:
                logger.warning(
                    f"Runner parent died (ppid {initial_ppid} -> {current_ppid}); "
                    "self-terminating to release Metal/GPU memory."
                )
                _release_metal_resources()
                # Bypass interpreter shutdown to avoid hanging on broken
                # multiprocessing pipes back to the now-dead supervisor.
                os._exit(1)

    thread = threading.Thread(
        target=watchdog,
        name="runner-parent-death-watchdog",
        daemon=True,
    )
    thread.start()


def _metal_cleanup_signal_handler(signum: int, _frame: object) -> None:
    """Handle SIGTERM/SIGINT via cooperative cancellation.

    Sets ``shutdown_requested`` so the per-layer load loop in
    ``utils_mlx.py`` can bail early with ``InterruptedError``.  Then
    raises ``InterruptedError`` directly so that code paths *outside*
    the load loop (inference, idle) also terminate promptly.

    ``InterruptedError`` is a subclass of ``Exception``, so it flows
    through the ``except Exception`` block in ``entrypoint()`` which
    reports ``RunnerFailed`` to the supervisor.  Metal cleanup happens
    in the ``finally`` block of ``entrypoint()``.
    """
    global shutdown_requested
    shutdown_requested = True
    logger.info(f"Runner received signal {signum}, requesting cooperative shutdown")
    # Raise instead of sys.exit() so the normal exception path in
    # entrypoint() can report RunnerFailed to the supervisor.
    raise InterruptedError(f"Runner interrupted by signal {signum}")


def entrypoint(
    bound_instance: BoundInstance,
    event_sender: MpSender[Event],
    diagnostic_sender: MpSender[RunnerDiagnosticUpdate],
    task_receiver: MpReceiver[Task],
    cancel_receiver: MpReceiver[TaskId],
    _logger: "loguru.Logger",
) -> None:
    global logger
    logger = _logger

    # Install signal handlers so that SIGTERM/SIGINT from the supervisor
    # triggers Metal cleanup instead of an abrupt death that leaks wired RAM.
    signal.signal(signal.SIGTERM, _metal_cleanup_signal_handler)
    signal.signal(signal.SIGINT, _metal_cleanup_signal_handler)

    # Backstop for SIGKILL of the supervisor: signal handlers above only fire
    # for graceful agent shutdown. If the agent is SIGKILLed we get reparented
    # to launchd, never receive a signal, and orphan ~5 GB of wired Metal
    # memory. The watchdog detects reparenting and self-exits cleanly.
    _install_parent_death_watchdog(initial_ppid=os.getppid())

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(max(soft, 2048), hard), hard))

    shard = bound_instance.bound_shard
    fast_synch_enabled = resolve_metal_fast_synch(shard.model_card.runtime)
    os.environ["MLX_METAL_FAST_SYNCH"] = "1" if fast_synch_enabled else "0"
    logger.info(
        f"Fast synch flag: {os.environ['MLX_METAL_FAST_SYNCH']} "
        f"(model={shard.model_card.model_id})"
    )
    configure_runner_diagnostics(
        diagnostic_sender,
        RunnerDiagnosticContext(
            node_id=str(bound_instance.bound_node_id),
            runner_id=str(bound_instance.bound_runner_id),
            pid=os.getpid(),
            instance_id=str(bound_instance.instance.instance_id),
            model_id=str(shard.model_card.model_id),
            rank=shard.device_rank,
            world_size=shard.world_size,
            start_layer=shard.start_layer,
            end_layer=shard.end_layer,
            n_layers=shard.n_layers,
        ),
    )
    record_runner_phase("created", event="process_started", include_memory=True)

    # Import main after setting global logger - this lets us just import logger from this module
    try:
        if bound_instance.is_image_model:
            from exo.worker.runner.image_models.runner import Runner as ImageRunner

            runner = ImageRunner(
                bound_instance, event_sender, task_receiver, cancel_receiver
            )
            runner.main()
        elif bound_instance.is_embedding_model:
            from exo.worker.runner.embeddings.runner import Runner as EmbeddingRunner

            runner = EmbeddingRunner(
                bound_instance, event_sender, task_receiver, cancel_receiver
            )
            runner.main()
        else:
            from exo.worker.engines.mlx.patches import apply_mlx_patches
            from exo.worker.runner.llm_inference.runner import Runner

            apply_mlx_patches()

            runner = Runner(
                bound_instance, event_sender, task_receiver, cancel_receiver
            )
            runner.main()

    except ClosedResourceError:
        logger.warning("Runner communication closed unexpectedly")
    except Exception as e:
        record_runner_phase(
            "error",
            event="runner_crashed",
            detail=f"{type(e).__name__}: {e}",
            include_memory=True,
        )
        logger.opt(exception=e).warning(
            f"Runner {bound_instance.bound_runner_id} crashed with critical exception {e}"
        )
        event_sender.send(
            RunnerStatusUpdated(
                runner_id=bound_instance.bound_runner_id,
                runner_status=RunnerFailed(error_message=str(e)),
            )
        )
    finally:
        # Safety net: release Metal resources on any exit path, even if the
        # signal handler didn't fire (e.g. parent was SIGKILL'd and we got
        # a broken pipe, or an unexpected exception during load).
        record_runner_phase(
            "shutdown_cleanup",
            event="metal_cleanup_begin",
            include_memory=True,
        )
        _release_metal_resources()
        record_runner_phase(
            "shutdown_cleanup",
            event="metal_cleanup_complete",
            include_memory=True,
        )
        try:
            event_sender.close()
            task_receiver.close()
            diagnostic_sender.close()
        finally:
            event_sender.join()
            task_receiver.join()
            # Diagnostics are best-effort and must never delay Metal cleanup.
            # Closing is enough; do not wait for a full diagnostic queue to drain.
            logger.info("bye from the runner")
