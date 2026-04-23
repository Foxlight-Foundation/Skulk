import gc
import os
import resource
import signal

import loguru

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

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(max(soft, 2048), hard), hard))

    fast_synch_override = os.environ.get("EXO_FAST_SYNCH")
    if fast_synch_override != "off":
        os.environ["MLX_METAL_FAST_SYNCH"] = "1"
    else:
        os.environ["MLX_METAL_FAST_SYNCH"] = "0"

    logger.info(f"Fast synch flag: {os.environ['MLX_METAL_FAST_SYNCH']}")
    shard = bound_instance.bound_shard
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
