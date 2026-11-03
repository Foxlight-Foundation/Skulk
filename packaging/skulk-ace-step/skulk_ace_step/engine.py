"""One offline native ACE model, with verified context and private WAV custody."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import final

import numpy as np
import numpy.typing as npt
import torch
from pydantic import TypeAdapter

from .audio_separation import SeparationSource
from .contracts import (
    NativeMusicProgress,
    NativeMusicReady,
    NativeMusicRequest,
    NativeMusicResult,
    NativeMusicServeConfig,
)
from .generation import prepare_call, result_samples
from .media import (
    export_result,
    prepare_conditioning,
    read_original,
    write_conditioning,
)
from .policy import observe_execution, probe_device
from .upstream_bindings import (
    LoadedHandler,
    NativeAPI,
    SourceProcessor,
    install_offline_precheck,
    native_api,
    tensor_samples,
)
from .weights import ModelCodeFile, prepare_weights

Progress = Callable[[NativeMusicProgress], None]


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_result(path: Path, result: NativeMusicResult) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(result.model_dump_json().encode("utf-8") + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


@dataclass
class _SourceObservation:
    original: SourceProcessor
    path: Path | None
    samples: npt.NDArray[np.float32] | None
    facts: SeparationSource | None
    calls: int = 0

    def __call__(self, audio_file: str | None) -> torch.Tensor | None:
        if (
            self.path is None
            or self.samples is None
            or self.facts is None
            or audio_file != str(self.path)
            or self.calls
        ):
            raise ValueError(
                "native music attempted an unexpected or repeated source input"
            )
        actual = read_original(self.path, self.facts)
        if not np.array_equal(actual, self.samples):
            raise ValueError("native music context changed before source processing")
        tensor = self.original(audio_file)
        if (
            tensor is None
            or tensor.device.type != "cpu"
            or tensor.dtype != torch.float32
            or tuple(tensor.shape) != (2, self.samples.shape[0])
        ):
            raise ValueError(
                "native music source processor did not return its exact canonical geometry"
            )
        if not np.array_equal(tensor_samples(tensor).T, self.samples):
            raise ValueError(
                "native music source tensor differs from canonical conditioning"
            )
        self.calls += 1
        return tensor


@final
class ResidentMusic:
    """Single-admission native model; any failure permanently retires its handle.

    The process supervisor owns final cleanup after exit. In particular, do not
    delete partial files or free model state after a native cooperative timeout:
    upstream may still have a live inference thread using those resources.
    """

    def __init__(
        self,
        configuration: NativeMusicServeConfig,
        handler: LoadedHandler,
        api: NativeAPI,
        ready: NativeMusicReady,
    ) -> None:
        """Retain verified initialization and one exclusive request admission lock."""
        self.configuration = configuration
        self.ready = ready
        self._handler = handler
        self._api = api
        self._admission = threading.Lock()
        self._usable = True

    def run(self, request: NativeMusicRequest, progress: Progress) -> NativeMusicResult:
        """Generate one complete result; preserve all partial work for drained cleanup."""
        if not self._admission.acquire(blocking=False):
            raise RuntimeError("native music model already has an active request")
        original_processor = self._handler.process_src_audio
        try:
            if not self._usable:
                raise RuntimeError(
                    "native music model must restart after its previous failure"
                )
            directory = Path(request.output_directory)
            expected = (
                Path(self.configuration.directory) / "requests" / request.request_id
            )
            if (
                directory != expected
                or expected.parent.is_symlink()
                or Path(self.configuration.directory).is_symlink()
            ):
                raise ValueError(
                    "native music output must use its exclusive lifetime request directory"
                )
            directory.mkdir(mode=0o700)
            context: Path | None = None
            conditioning: SeparationSource | None = None
            source: SeparationSource | None = None
            gain: float | None = None
            samples: npt.NDArray[np.float32] | None = None
            if request.operation != "generate":
                progress(
                    NativeMusicProgress(
                        request_id=request.request_id, stage="conditioning"
                    )
                )
                prepared = prepare_conditioning(request)
                samples, source, gain = prepared.samples, prepared.source, prepared.gain
                context = directory / "conditioning.wav"
                conditioning = write_conditioning(context, samples)
            observation = _SourceObservation(
                original_processor, context, samples, conditioning
            )
            self._handler.process_src_audio = observation
            call = prepare_call(request, self._handler, self._api.constructors, context)
            progress(
                NativeMusicProgress(request_id=request.request_id, stage="generating")
            )
            started = time.monotonic()
            with torch.inference_mode():
                result = self._api.generate(
                    self._handler,
                    None,
                    call.parameters,
                    call.configuration,
                    save_dir=None,
                )
            elapsed = time.monotonic() - started
            rendered = result_samples(result, request)
            if observation.calls != (0 if request.operation == "generate" else 1):
                raise ValueError(
                    "native music did not consume the exact required context"
                )
            execution = observe_execution(self._handler, self.configuration)
            if execution != self.ready.execution:
                raise ValueError(
                    "native music component policy changed during inference"
                )
            progress(
                NativeMusicProgress(request_id=request.request_id, stage="writing")
            )
            output = directory / "result.wav"
            facts = export_result(output, rendered.samples, request)
            with output.open("rb") as stream:
                os.fsync(stream.fileno())
            completed = NativeMusicResult(
                request_id=request.request_id,
                model_identity=self.ready.model_identity,
                payload_sha256=self.ready.payload_sha256,
                operation=request.operation,
                seed=request.seed,
                source_sha256=source.sha256 if source is not None else None,
                conditioning_sha256=conditioning.sha256
                if conditioning is not None
                else None,
                conditioning_gain=gain,
                padding_frames_removed=rendered.padding_frames_removed,
                execution=execution,
                output=facts,
                generation_seconds=elapsed,
            )
            _write_result(directory / "result.json", completed)
            _sync_directory(directory)
            _sync_directory(directory.parent)
            return completed
        except Exception:
            self._usable = False
            raise
        finally:
            self._handler.process_src_audio = original_processor
            self._admission.release()


def _environment(configuration: NativeMusicServeConfig) -> Path:
    root = Path(configuration.directory)
    root.mkdir(mode=0o700)
    (root / "requests").mkdir(mode=0o700)
    # Ambient UI/debug settings can otherwise override this admitted process's
    # VAE, memory policy or progress-file locations. Native source is imported
    # only after this closed policy replaces the entire ACE environment family.
    for name in tuple(os.environ):
        if name.startswith("ACESTEP_"):
            del os.environ[name]
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
            "HF_HOME": str(root / "hf"),
            "HF_MODULES_CACHE": str(root / "modules"),
            "TORCH_HOME": str(root / "torch-cache"),
            "ACESTEP_CHECKPOINTS_DIR": str(root / "checkpoints"),
            "ACESTEP_VAE_CHECKPOINT": "official",
            "ACESTEP_PROJECT_ROOT": str(root),
            "ACESTEP_VAE_ON_CPU": "0",
            "ACESTEP_MPS_DECODE_OFFLOAD": "0",
            "ACESTEP_DISABLE_TQDM": "1",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "ACESTEP_GENERATION_TIMEOUT": str(configuration.generation_timeout_seconds),
            "ACESTEP_ROCM_DTYPE": "float32",
            "MIOPEN_FIND_MODE": "FAST",
        }
    )
    return root


def load_music(
    configuration: NativeMusicServeConfig,
    payload_sha256: str,
    progress: Progress,
    *,
    runtime_root: Path | None = None,
    code: Sequence[ModelCodeFile] | None = None,
    api_loader: Callable[[], NativeAPI] = native_api,
) -> ResidentMusic:
    """Verify all data/code and load the explicit policy before reporting ready.

    Runtime code comes from the sealed library, never model-repository files.
    The one upstream download precheck is replaced with a guard for this complete
    local selection; offline environment settings remain defense in depth.
    Optional internal dependencies support actual file/order tests without
    pretending a small fixture is a qualified music model.
    """
    started = time.monotonic()
    root = _environment(configuration)
    torch.set_num_threads(configuration.threads)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    native = probe_device(configuration)
    library = runtime_root if runtime_root is not None else Path(__file__).parent.parent
    if code is None:
        with (Path(__file__).parent / "model-code.json").open("rb") as stream:
            data = stream.read(16 * 1024 + 1)
        if len(data) > 16 * 1024:
            raise ValueError("native music model-code inventory exceeds its bound")
        code = TypeAdapter(tuple[ModelCodeFile, ...]).validate_json(data)
    progress(NativeMusicProgress(stage="verifying_weights"))
    # The pinned initializer resolves the checkpoint path, including macOS's
    # temporary-directory aliases. Compare the same canonical private view.
    view = prepare_weights(
        root / "checkpoints", configuration.weights, library, code
    ).resolve()
    api = api_loader()
    handler = api.handler()

    def offline_precheck(
        *,
        checkpoint_path: Path,
        config_path: str,
        prefer_source: str | None,
        vae_variant: str | None = None,
    ) -> tuple[str, bool] | None:
        if (
            checkpoint_path != view
            or config_path != "acestep-v15-base"
            or vae_variant not in (None, "official")
        ):
            return (
                "Native music requires its complete verified offline model selection.",
                False,
            )
        return None

    # The broader upstream precheck downloads the default Turbo set even when
    # this base selection is complete. This single audited hook changes only
    # model acquisition; generation and source processing remain upstream calls.
    install_offline_precheck(handler, offline_precheck)
    progress(NativeMusicProgress(stage="loading"))
    status, initialized = handler.initialize_service(
        project_root=str(root),
        config_path="acestep-v15-base",
        device=native,
        use_flash_attention=False,
        compile_model=False,
        offload_to_cpu=False,
        offload_dit_to_cpu=False,
        quantization=None,
        use_mlx_dit=False,
        vae_checkpoint="official",
    )
    if not initialized:
        raise RuntimeError("native music initialization failed: " + status[:2048])
    if handler.use_mlx_dit or handler.use_mlx_vae:
        raise ValueError(
            "native music activated an unadmitted alternate MLX implementation"
        )
    execution = observe_execution(handler, configuration)
    ready = NativeMusicReady(
        model_identity=configuration.model_identity,
        payload_sha256=payload_sha256,
        execution=execution,
        load_seconds=time.monotonic() - started,
    )
    return ResidentMusic(configuration, handler, api, ready)
