"""Actual source decoding, model-policy observation and native wrapper custody.

Diffusion is controlled here; these tests never claim music model inference.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import check_support
import numpy as np
import torch
from pydantic import JsonValue, TypeAdapter
from skulk_ace_step.audio_separation import source_from_wav
from skulk_ace_step.bindings import soundfile
from skulk_ace_step.contracts import (
    NativeMusicReady,
    NativeMusicRequest,
    NativeMusicResult,
)
from skulk_ace_step.engine import ResidentMusic
from skulk_ace_step.policy import observe_execution, probe_device
from skulk_ace_step.upstream_bindings import (
    LoadedHandler,
    NativeAPI,
    NativeResult,
    install_offline_precheck,
    native_api,
)


@dataclass
class TensorResult:
    """A controlled completion carrying real native-format Torch samples."""

    audios: list[dict[str, object]]
    success: bool = True
    error: str | None = None


class NativeEngineTests(unittest.TestCase):
    """Run the actual wrapper and native source processor with tiny CPU modules."""

    def setUp(self) -> None:
        """Retain actual pinned constructors and source handling, without weights."""
        self.temporary = tempfile.TemporaryDirectory(prefix="native-music-engine-")
        self.root = Path(self.temporary.name)
        self.configuration = check_support.configuration(self.root)
        (Path(self.configuration.directory) / "requests").mkdir(parents=True)
        self.api = native_api()
        self.handler = self.api.handler()
        self.handler.model = torch.nn.Linear(1, 1)
        self.handler.text_encoder = torch.nn.Linear(1, 1)
        self.handler.vae = torch.nn.Linear(1, 1)
        self.calls = 0
        self.skip_source = False
        self.simulate_timeout = False
        self.wait: threading.Event | None = None
        self.entered = threading.Event()
        self.ready = NativeMusicReady(
            model_identity=self.configuration.model_identity,
            payload_sha256="b" * 64,
            execution=check_support.execution(),
            load_seconds=0.25,
        )
        self.engine = ResidentMusic(
            self.configuration,
            self.handler,
            NativeAPI(
                handler=self.api.handler,
                constructors=self.api.constructors,
                generate=self.generate,
            ),
            self.ready,
        )

    def tearDown(self) -> None:
        """Remove only the test's private files after all work has finished."""
        self.temporary.cleanup()

    def generate(
        self,
        dit_handler: LoadedHandler,
        llm_handler: None,
        params: object,
        config: object,
        *,
        save_dir: None,
    ) -> NativeResult:
        """Substitute only diffusion; real upstream preprocessing still runs."""
        self.calls += 1
        self.assertIsNone(llm_handler)
        self.assertIsNone(save_dir)
        values = TypeAdapter(dict[str, JsonValue]).validate_json(
            json.dumps(vars(params))
        )
        source = values["src_audio"]
        tensor = None
        if isinstance(source, str) and not self.skip_source:
            tensor = dit_handler.process_src_audio(source)
        self.entered.set()
        if self.wait is not None:
            self.assertTrue(self.wait.wait(5))
        if self.simulate_timeout:
            raise TimeoutError("controlled native timeout")
        if tensor is None:
            tensor = torch.ones((2, 48000), dtype=torch.float32) * 0.25
        return TensorResult(audios=[{"tensor": tensor, "sample_rate": 48000}])

    def edit(self, continuation: bool = False) -> NativeMusicRequest:
        """Create a real over-range FLOAT context to exercise upstream clamping."""
        path = self.root / "original.wav"
        samples = np.zeros((48000, 2), dtype=np.float32)
        samples[100:200, 0], samples[100:200, 1] = 1.5, -3.0
        with path.open("wb") as stream:
            soundfile.write(stream, samples, 48000, format="WAV", subtype="FLOAT")
        values = check_support.request(self.configuration).model_dump(mode="json")
        values.update(
            operation="continue" if continuation else "repaint",
            source=source_from_wav(path.read_bytes()).model_dump(mode="json"),
            source_path=str(path),
            target_frames=96000 if continuation else 48000,
            region_start_frame=48000 if continuation else 0,
            region_end_frame=96000 if continuation else 48000,
        )
        return NativeMusicRequest.model_validate_json(json.dumps(values))

    def test_actual_native_source_processor_receives_identical_canonical_samples(
        self,
    ) -> None:
        """Headroom is recorded, WAV custody is private, and the original is retained."""
        request = self.edit()
        original = Path(request.source_path or "").read_bytes()
        progress = []
        result = self.engine.run(request, progress.append)
        self.assertEqual(self.calls, 1)
        self.assertEqual(
            [p.stage for p in progress], ["conditioning", "generating", "writing"]
        )
        self.assertLess(result.conditioning_gain or 1, 1)
        self.assertNotEqual(result.source_sha256, result.conditioning_sha256)
        directory = Path(request.output_directory)
        self.assertEqual(
            NativeMusicResult.model_validate_json(
                (directory / "result.json").read_bytes()
            ),
            result,
        )
        actual = source_from_wav((directory / "result.wav").read_bytes())
        self.assertEqual(
            (actual.sha256, actual.frames, actual.bit_depth),
            (result.output.sha256, 48000, 24),
        )
        for name in ("result.json", "result.wav", "conditioning.wav"):
            self.assertEqual(os.stat(directory / name).st_mode & 0o777, 0o600)
        self.assertEqual(Path(request.source_path or "").read_bytes(), original)

    def test_continue_consumes_exact_padded_tail_and_keeps_duration(self) -> None:
        """Actual native source processing accepts the explicit extension geometry."""
        result = self.engine.run(self.edit(continuation=True), lambda _: None)
        self.assertEqual(result.output.frames, 96000)

    def test_ignored_source_is_failed_and_model_cannot_be_reused(self) -> None:
        """An apparently successful native result cannot hide unused context."""
        self.skip_source = True
        with self.assertRaisesRegex(ValueError, "consume"):
            self.engine.run(self.edit(), lambda _: None)
        self.assertFalse(
            (
                Path(self.configuration.directory) / "requests/take-1/result.json"
            ).exists()
        )
        with self.assertRaisesRegex(RuntimeError, "restart"):
            self.engine.run(
                check_support.request(self.configuration, "take-2"), lambda _: None
            )
        self.assertEqual(self.calls, 1)

    def test_native_timeout_retains_partial_input_and_poisons_model(self) -> None:
        """No resource is deleted while a timed-out upstream thread could still use it."""
        self.simulate_timeout = True
        with self.assertRaises(TimeoutError):
            self.engine.run(self.edit(), lambda _: None)
        directory = Path(self.configuration.directory) / "requests/take-1"
        self.assertTrue((directory / "conditioning.wav").is_file())
        self.assertFalse((directory / "result.json").exists())
        with self.assertRaisesRegex(RuntimeError, "restart"):
            self.engine.run(
                check_support.request(self.configuration, "take-2"), lambda _: None
            )

    def test_concurrent_admission_rejects_second_request_without_second_inference(
        self,
    ) -> None:
        """A caller cannot overcommit the loaded model during an active generation."""
        self.wait = threading.Event()
        errors: list[BaseException] = []

        def first() -> None:
            try:
                self.engine.run(
                    check_support.request(self.configuration), lambda _: None
                )
            except BaseException as error:
                errors.append(error)

        worker = threading.Thread(target=first)
        worker.start()
        try:
            self.assertTrue(self.entered.wait(5))
            with self.assertRaisesRegex(RuntimeError, "active request"):
                self.engine.run(
                    check_support.request(self.configuration, "take-2"), lambda _: None
                )
        finally:
            self.wait.set()
            worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(self.calls, 1)

    def test_cpu_probe_and_actual_component_observation_reject_wrong_precision(
        self,
    ) -> None:
        """Readiness measures actual tensors rather than echoing a configured label."""
        self.assertEqual(probe_device(self.configuration), "cpu")
        self.assertEqual(
            observe_execution(self.handler, self.configuration),
            check_support.execution(),
        )
        assert self.handler.vae is not None
        self.handler.vae.to(dtype=torch.float16)
        with self.assertRaisesRegex(ValueError, "policy"):
            observe_execution(self.handler, self.configuration)

    def test_output_path_cannot_escape_lifetime(self) -> None:
        """Private request confusion cannot write a file selected outside its owner."""
        request = check_support.request(self.configuration).model_copy(
            update={"output_directory": str(self.root / "elsewhere")}
        )
        with self.assertRaisesRegex(ValueError, "exclusive"):
            self.engine.run(request, lambda _: None)
        self.assertEqual(self.calls, 0)
        self.assertFalse((self.root / "elsewhere").exists())

    def test_actual_native_initializer_calls_the_typed_offline_guard_before_loading(
        self,
    ) -> None:
        """The acquisition hook is exercised by upstream, not merely installed as data."""
        reached: list[tuple[Path, str, str | None]] = []

        def guarded(
            *,
            checkpoint_path: Path,
            config_path: str,
            prefer_source: str | None,
            vae_variant: str | None = None,
        ) -> tuple[str, bool] | None:
            reached.append((checkpoint_path, config_path, vae_variant))
            return "Controlled stop before native tensor loading.", False

        install_offline_precheck(self.handler, guarded)
        checkpoints = self.root / "guarded-checkpoints"
        with patch.dict(
            os.environ,
            {
                "ACESTEP_CHECKPOINTS_DIR": str(checkpoints),
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            },
        ):
            status, initialized = self.handler.initialize_service(
                project_root=str(self.root),
                config_path="acestep-v15-base",
                device="cpu",
                use_flash_attention=False,
                compile_model=False,
                offload_to_cpu=False,
                offload_dit_to_cpu=False,
                quantization=None,
                use_mlx_dit=False,
                vae_checkpoint="official",
            )
        self.assertEqual(
            reached, [(checkpoints.resolve(), "acestep-v15-base", "official")]
        )
        self.assertFalse(initialized)
        self.assertEqual(status, "Controlled stop before native tensor loading.")
        self.assertIsNone(self.handler.model)
