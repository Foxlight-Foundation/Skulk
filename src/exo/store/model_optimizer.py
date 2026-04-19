"""Model optimization pipeline using mlx-optiq mixed-precision quantization.

Provides async wrappers around mlx-optiq's sensitivity analysis and
quantization pipeline for integration with the Skulk API.
"""

import asyncio
import importlib
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, TypedDict, cast

logger = logging.getLogger("exo.store.model_optimizer")


class OptimizationPipelineResult(TypedDict):
    """Normalized OptiQ output fields consumed by the API layer."""

    optiq_path: str
    achieved_bpw: float | None
    estimated_size_mb: float | None


class _RunLlmPipeline(Protocol):
    """Minimal callable surface used from mlx-optiq."""

    def __call__(
        self,
        *,
        model_name: str,
        output_dir: str,
        target_bpw: float = 4.5,
        n_calibration: int = 8,
        candidate_bits: list[int] | None = None,
        group_size: int = 64,
        skip_baselines: bool = False,
        low_bits: int = 4,
        high_bits: int = 8,
    ) -> dict[str, object]: ...


def _float_attr(value: object, attr_name: str) -> float | None:
    """Extract one optional float attribute from an opaque third-party object."""
    attr_value = cast(object, getattr(value, attr_name, None))
    return float(attr_value) if isinstance(attr_value, int | float) else None


@dataclass
class OptimizationProgress:
    """Current state of an optimization job."""

    model_id: str
    status: (
        str  # "pending", "analyzing", "optimizing", "converting", "complete", "failed"
    )
    progress: float = 0.0  # 0.0 - 1.0
    message: str = ""
    result_path: str | None = None
    achieved_bpw: float | None = None
    estimated_size_mb: float | None = None
    error: str | None = None


@dataclass
class ModelOptimizer:
    """Manages model optimization jobs."""

    store_path: Path
    _jobs: dict[str, OptimizationProgress] = field(default_factory=dict)
    _tasks: dict[str, asyncio.Task[None]] = field(default_factory=dict)

    def get_status(self, model_id: str) -> OptimizationProgress | None:
        return self._jobs.get(model_id)

    def list_jobs(self) -> list[OptimizationProgress]:
        return list(self._jobs.values())

    async def optimize(
        self,
        model_id: str,
        target_bpw: float = 4.5,
        candidate_bits: list[int] | None = None,
        group_size: int = 64,
        n_calibration: int = 8,
    ) -> None:
        """Start an optimization job for a model."""
        if model_id in self._tasks and not self._tasks[model_id].done():
            raise RuntimeError(f"Optimization already in progress for {model_id}")

        if candidate_bits is None:
            candidate_bits = [4, 8]

        self._jobs[model_id] = OptimizationProgress(
            model_id=model_id,
            status="pending",
            message="Queued for optimization",
        )

        task = asyncio.create_task(
            self._run_pipeline(
                model_id, target_bpw, candidate_bits, group_size, n_calibration
            )
        )
        self._tasks[model_id] = task

    async def _run_pipeline(
        self,
        model_id: str,
        target_bpw: float,
        candidate_bits: list[int],
        group_size: int,
        n_calibration: int,
    ) -> None:
        """Run the mlx-optiq pipeline in a thread to avoid blocking the event loop."""
        job = self._jobs[model_id]
        output_dir = self.store_path / _optiq_dirname(model_id, target_bpw)

        try:
            job.status = "analyzing"
            job.progress = 0.1
            job.message = "Running sensitivity analysis..."

            result = await asyncio.to_thread(
                _run_optiq_pipeline,
                model_id=model_id,
                output_dir=str(output_dir),
                target_bpw=target_bpw,
                candidate_bits=candidate_bits,
                group_size=group_size,
                n_calibration=n_calibration,
            )

            job.status = "complete"
            job.progress = 1.0
            job.result_path = result["optiq_path"]
            job.achieved_bpw = result.get("achieved_bpw")
            job.estimated_size_mb = result.get("estimated_size_mb")
            job.message = (
                f"Optimization complete — {job.achieved_bpw:.2f} BPW"
                if job.achieved_bpw
                else "Optimization complete"
            )
            logger.info(
                f"ModelOptimizer: {model_id} optimization complete at {job.result_path}"
            )

        except Exception as exc:
            job.status = "failed"
            job.progress = 0.0
            error_str = str(exc)
            # Provide user-friendly messages for known errors
            if "Unrecognized configuration class" in error_str:
                job.error = (
                    "This model architecture is not supported by OptiQ optimization."
                )
                job.message = "Unsupported model architecture"
            elif "trust_remote_code" in error_str.lower():
                job.error = "This model requires trust_remote_code which is disabled for security."
                job.message = "Model requires remote code execution"
            else:
                job.error = error_str
                job.message = f"Optimization failed: {error_str[:100]}"
            logger.error(f"ModelOptimizer: {model_id} optimization failed: {exc}")
            # Clean up partial output
            if output_dir.exists():
                shutil.rmtree(output_dir, ignore_errors=True)

    def cancel(self, model_id: str) -> bool:
        """Cancel an in-progress optimization."""
        task = self._tasks.get(model_id)
        if task and not task.done():
            task.cancel()
            if model_id in self._jobs:
                self._jobs[model_id].status = "failed"
                self._jobs[model_id].message = "Cancelled"
            return True
        return False


def _optiq_dirname(model_id: str, target_bpw: float) -> str:
    """Generate a directory name for the optimized model."""
    normalized = model_id.replace("/", "--")
    return f"{normalized}-OptiQ-{target_bpw:.1f}bpw"


def _run_optiq_pipeline(
    model_id: str,
    output_dir: str,
    target_bpw: float,
    candidate_bits: list[int],
    group_size: int,
    n_calibration: int,
) -> OptimizationPipelineResult:
    """Synchronous wrapper around mlx-optiq pipeline (runs in thread)."""
    try:
        optiq_llm = importlib.import_module("optiq.models.llm")
    except ImportError as exc:
        raise RuntimeError(
            "mlx-optiq[convert] is required for model optimization. "
            "Install with: pip install 'mlx-optiq[convert]'"
        ) from exc
    run_llm_pipeline = cast(
        _RunLlmPipeline,
        cast(dict[str, object], vars(optiq_llm))["run_llm_pipeline"],
    )

    result = run_llm_pipeline(
        model_name=model_id,
        output_dir=output_dir,
        target_bpw=target_bpw,
        n_calibration=n_calibration,
        candidate_bits=candidate_bits,
        group_size=group_size,
        skip_baselines=True,  # We only want the OptiQ result
    )

    # Extract key metrics from result
    optimization = result.get("optimization")
    achieved_bpw = (
        _float_attr(optimization, "achieved_bpw") if optimization is not None else None
    )
    estimated_size_mb = (
        _float_attr(optimization, "estimated_size_mb")
        if optimization is not None
        else None
    )
    optiq_path = result.get("optiq_path")

    return {
        "optiq_path": optiq_path if isinstance(optiq_path, str) else output_dir,
        "achieved_bpw": achieved_bpw,
        "estimated_size_mb": estimated_size_mb,
    }
