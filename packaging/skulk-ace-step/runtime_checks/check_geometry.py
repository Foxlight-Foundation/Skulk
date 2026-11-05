"""Actual native conditioning and Oobleck geometry with tiny untrained weights.

The codec retains the pinned sampling ratios with reduced channel width. Only
diffusion is replaced by a controlled latent tensor; this is no music-quality
or full-model qualification.
"""

from __future__ import annotations

import importlib
import json
import tempfile
import unittest
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, cast
from unittest.mock import patch

import check_support
import numpy as np
import torch
from skulk_ace_step.audio_separation import source_from_wav
from skulk_ace_step.bindings import soundfile
from skulk_ace_step.contracts import NativeMusicRequest, NativeMusicResult
from skulk_ace_step.generation import result_samples
from skulk_ace_step.media import export_result, prepare_conditioning, write_conditioning
from skulk_ace_step.upstream_bindings import native_api, tensor_samples

# Bind the three inspected methods on the untyped library to the callable
# signatures below. These method identities belong to the pinned source.
_TARGET_PREPARATION_METHOD: Final = "_prepare_target_latents_and_wavs"
_DECODE_PREPARATION_METHOD: Final = "_prepare_generate_music_decode_state"
_WAVEFORM_DECODE_METHOD: Final = "_decode_generate_music_pred_latents"


class CodecFactory(Protocol):
    """Construct the installed codec with the pinned 1,920-sample hop."""

    def __call__(
        self,
        *,
        encoder_hidden_size: int,
        decoder_channels: int,
        decoder_input_channels: int,
        channel_multiples: list[int],
        downsampling_ratios: list[int],
        sampling_rate: int,
        audio_channels: int,
    ) -> torch.nn.Module:
        """Return a tiny actual encoder/decoder, without loading any weights."""
        ...


class PrepareTargets(Protocol):
    """Audited target-conditioning stage, including actual codec computation."""

    def __call__(
        self,
        batch_size: int,
        target_wavs: torch.Tensor,
        audio_code_hints: list[str | None],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
        """Encode source context and retain the native latent padding and mask."""
        ...


class PrepareDecode(Protocol):
    """Audited finite-latent validation before waveform decode."""

    def __call__(
        self,
        outputs: dict[str, object],
        infer_steps_for_progress: int,
        actual_batch_size: int,
        audio_duration: float,
        latent_shift: float,
        latent_rescale: float,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Pass controlled diffusion through the actual native decode preparation."""
        ...


class DecodeWaveforms(Protocol):
    """Audited waveform decode and headroom orchestration."""

    def __call__(
        self,
        pred_latents: torch.Tensor,
        progress: None,
        use_tiled_decode: bool,
        time_costs: dict[str, float],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """Decode with the real tiny Oobleck model and native output policy."""
        ...


class Splice(Protocol):
    """Pinned post-decode waveform splice, including its retained extra tail."""

    def __call__(
        self,
        pred_wavs: torch.Tensor,
        src_wavs: torch.Tensor,
        repainting_starts: list[float],
        repainting_ends: list[float],
    ) -> torch.Tensor:
        """Splice the original around the selected range using native crossfades."""
        ...


@dataclass
class TensorResult:
    """Carry the actual decoded waveform through the adapter's result boundary."""

    audios: list[dict[str, object]]
    success: bool = True
    error: str | None = None


class NativeGeometryTests(unittest.TestCase):
    """Keep real codec, conditioning and splice computation around controlled diffusion."""

    def test_short_edits_remove_only_actual_minimum_latent_padding(self) -> None:
        """Subsecond frame offsets and Continue survive the real padded codec path."""
        self._assert_geometry(
            (
                ("repaint", 48000, 48000, False),
                ("add_part", 48001, 48001, False),
                ("continue", 48000, 96001, False),
                ("repaint", 245759, 245759, False),
                ("repaint", 245760, 245760, False),
            )
        )

    def test_long_non_aligned_edits_preserve_silent_and_audible_geometry(self) -> None:
        """Real codec rounding cannot drop the last frame of an imported context."""
        self._assert_geometry(
            (
                ("repaint", 288001, 288001, False),
                ("repaint", 288001, 288001, True),
                ("add_part", 288001, 288001, False),
                ("continue", 240001, 288001, False),
                ("continue", 240001, 288001, True),
                ("repaint", 288000, 288000, False),
                ("repaint", 1536001, 1536001, False),
                ("continue", 1488001, 1536001, False),
            )
        )

    def _assert_geometry(self, cases: tuple[tuple[str, int, int, bool], ...]) -> None:
        """Run actual preprocessing, codec and splice around controlled diffusion."""
        factory = cast(
            CodecFactory,
            cast(
                object, importlib.import_module("diffusers.models").AutoencoderOobleck
            ),
        )
        splice = cast(
            Splice,
            cast(
                object,
                importlib.import_module(
                    "acestep.core.generation.handler.repaint_waveform_splice"
                ).apply_repaint_waveform_splice,
            ),
        )
        handler = native_api().handler()
        codec = factory(
            encoder_hidden_size=2,
            decoder_channels=2,
            decoder_input_channels=1,
            channel_multiples=[1, 1, 1, 1, 1],
            downsampling_ratios=[2, 4, 4, 6, 10],
            sampling_rate=48000,
            audio_channels=2,
        ).eval()
        with torch.no_grad():
            for parameter in codec.parameters():
                parameter.fill_(0.01)
        handler.vae = codec
        prepare_targets = cast(
            PrepareTargets,
            cast(object, getattr(handler, _TARGET_PREPARATION_METHOD)),
        )
        prepare_decode = cast(
            PrepareDecode,
            cast(object, getattr(handler, _DECODE_PREPARATION_METHOD)),
        )
        decode_waveforms = cast(
            DecodeWaveforms,
            cast(object, getattr(handler, _WAVEFORM_DECODE_METHOD)),
        )
        for operation, source_frames, target_frames, silent in cases:
            with (
                self.subTest(operation=operation, frames=target_frames, silent=silent),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                original = root / "source.wav"
                with original.open("wb") as stream:
                    soundfile.write(
                        stream,
                        np.full(
                            (source_frames, 2),
                            0.0 if silent else 0.125,
                            dtype=np.float32,
                        ),
                        48000,
                        format="WAV",
                        subtype="FLOAT",
                    )
                original_bytes = original.read_bytes()
                source = source_from_wav(original_bytes)
                directory = root / "output"
                directory.mkdir(mode=0o700)
                request = NativeMusicRequest.model_validate_json(
                    json.dumps(
                        dict(
                            request_id="short-edit",
                            operation=operation,
                            prompt="Warm strings",
                            seed=12025,
                            target_frames=target_frames,
                            region_start_frame=source_frames
                            if operation == "continue"
                            else 0,
                            region_end_frame=target_frames,
                            part_class="strings" if operation == "add_part" else None,
                            source=source.model_dump(mode="json"),
                            source_path=str(original),
                            output_directory=str(directory),
                            output_directory_identity=check_support.directory_identity(
                                directory
                            ).model_dump(mode="json"),
                        )
                    )
                )
                prepared = prepare_conditioning(request)
                aligned_frames = (target_frames + 1919) // 1920 * 1920
                self.assertEqual(prepared.samples.shape, (aligned_frames, 2))
                self.assertEqual(
                    prepared.padding_frames, aligned_frames - target_frames
                )
                np.testing.assert_array_equal(
                    prepared.samples[:source_frames],
                    np.full(
                        (source_frames, 2), 0.0 if silent else 0.125, dtype=np.float32
                    ),
                )
                np.testing.assert_array_equal(
                    prepared.samples[source_frames:],
                    np.zeros((aligned_frames - source_frames, 2), dtype=np.float32),
                )
                context = root / "conditioning.wav"
                conditioning = write_conditioning(context, prepared.samples)
                processed = handler.process_src_audio(str(context))
                self.assertIsNotNone(processed)
                assert processed is not None
                np.testing.assert_array_equal(
                    tensor_samples(processed).T, prepared.samples
                )
                with (
                    patch.object(handler, "silence_latent", torch.zeros((1, 16, 1))),
                    patch.object(
                        handler, "_load_model_context", return_value=nullcontext()
                    ),
                ):
                    wavs, latents, masks, maximum, _ = prepare_targets(
                        1, processed.unsqueeze(0), [None]
                    )
                    self.assertLessEqual(int(masks.sum().item()), maximum)
                    pred, costs = prepare_decode(
                        {
                            "target_latents": latents + 0.1,
                            "time_costs": {"total_time_cost": 0.0},
                        },
                        50,
                        1,
                        target_frames / 48000,
                        0.0,
                        1.0,
                    )
                    decoded, _, _ = decode_waveforms(
                        pred,
                        None,
                        target_frames > 15 * 48000 and operation == "continue",
                        costs,
                    )
                spliced = splice(
                    decoded,
                    wavs,
                    [request.region_start_frame / 48000],
                    [target_frames / 48000],
                )
                expected_latents = max(128, (target_frames + 1919) // 1920)
                expected_decoded_frames = expected_latents * 1920
                self.assertEqual(spliced.shape[-1], expected_decoded_frames)
                self.assertEqual(maximum, expected_latents)
                rendered = result_samples(
                    TensorResult([{"tensor": spliced[0], "sample_rate": 48000}]),
                    request,
                )
                self.assertEqual(rendered.samples.shape, (target_frames, 2))
                self.assertEqual(
                    rendered.padding_frames_removed,
                    expected_decoded_frames - target_frames,
                )
                np.testing.assert_array_equal(
                    rendered.samples, tensor_samples(spliced[0]).T[:target_frames]
                )
                output = export_result(root / "result.wav", rendered.samples, request)
                result = NativeMusicResult(
                    request_id=request.request_id,
                    model_identity="a" * 64,
                    payload_sha256="b" * 64,
                    operation=request.operation,
                    seed=request.seed,
                    source_sha256=source.sha256,
                    conditioning_sha256=conditioning.sha256,
                    conditioning_gain=prepared.gain,
                    conditioning_padding_frames=prepared.padding_frames,
                    padding_frames_removed=rendered.padding_frames_removed,
                    execution=check_support.execution(),
                    output=output,
                    generation_seconds=1.0,
                )
                self.assertEqual(result.output.frames, target_frames)
                self.assertEqual(original.read_bytes(), original_bytes)
