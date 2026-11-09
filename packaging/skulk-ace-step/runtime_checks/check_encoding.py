"""Exercise the pinned encoder's actual tiling through the resident binding.

The encoder fixture preserves a visible sample clock and records real input
spans. It does not load model weights or establish music or memory qualification.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from unittest.mock import patch

import torch
from skulk_ace_step.upstream_bindings import (
    install_bounded_source_encoding,
    native_api,
)


@dataclass(frozen=True)
class LatentDistribution:
    """Deterministic clock, optionally with actual seeded Torch sampling."""

    latent: torch.Tensor
    stochastic: bool

    def sample(self) -> torch.Tensor:
        """Return encoded samples, drawing from the current request RNG only."""
        if self.stochastic:
            return self.latent + torch.randn_like(self.latent)
        return self.latent


@dataclass(frozen=True)
class EncodedAudio:
    """Pinned codec result shape consumed by the actual upstream tiler."""

    latent_dist: LatentDistribution


class RecordingEncoder(torch.nn.Module):
    """Record input windows and retain one latent per 1,920 source frames."""

    dtype: torch.dtype = torch.float32

    def __init__(self, *, stochastic: bool = False) -> None:
        """Start one independent codec observation without model parameters."""
        super().__init__()
        self.input_frames: list[int] = []
        self.stochastic = stochastic
        self.fail = False

    def encode(self, audio: torch.Tensor) -> EncodedAudio:
        """Exercise upstream slicing/assembly with a visible source clock."""
        self.input_frames.append(audio.shape[-1])
        if self.fail:
            raise RuntimeError("controlled codec allocation failure")
        return EncodedAudio(
            LatentDistribution(audio[:, :1, ::1920].clone(), self.stochastic)
        )


class NativeEncodingTests(unittest.TestCase):
    """Use actual pinned handlers around an observable, deliberately tiny codec."""

    def test_complete_clock_and_zero_tail_survive_each_actual_tile_boundary(
        self,
    ) -> None:
        """A 120s context is fully consumed through windows bounded to 15s."""
        for seconds in (5, 15, 16, 30, 60, 120):
            for padded in (False, True):
                with self.subTest(seconds=seconds, padded=padded):
                    values = torch.arange(seconds * 48000, dtype=torch.float32)
                    if padded:
                        values[15 * 48000 :] = 0
                    audio = values.reshape(1, 1, -1).expand(1, 2, -1)
                    handler = native_api().handler()
                    encoder = RecordingEncoder()
                    handler.vae = encoder
                    install_bounded_source_encoding(handler)
                    actual = handler.tiled_encode(audio)
                    torch.testing.assert_close(
                        actual, values[::1920].reshape(1, 1, -1), rtol=0, atol=0
                    )
                    self.assertLessEqual(max(encoder.input_frames), 15 * 48000)
                    self.assertEqual(len(encoder.input_frames) == 1, seconds <= 15)

    def test_one_handler_bound_leaves_other_handler_and_upstream_class_unchanged(
        self,
    ) -> None:
        """Capacity-derived defaults remain untouched outside the admitted instance."""
        bounded, other = native_api().handler(), native_api().handler()
        original_class_encoder = type(other).tiled_encode
        first, second = RecordingEncoder(), RecordingEncoder()
        bounded.vae, other.vae = first, second
        audio = torch.zeros((1, 2, 30 * 48000), dtype=torch.float32)
        install_bounded_source_encoding(bounded)
        with patch(
            "acestep.core.generation.handler.vae_encode.get_gpu_memory_gb",
            return_value=60.0,
        ):
            bounded.tiled_encode(audio)
            other.tiled_encode(audio)
        self.assertLessEqual(max(first.input_frames), 15 * 48000)
        self.assertEqual(second.input_frames, [30 * 48000])
        self.assertIs(type(other).tiled_encode, original_class_encoder)

    def test_short_seeded_encode_matches_unmodified_pinned_encoder(self) -> None:
        """The unchanged direct path retains the exact seeded latent sample."""
        handler = native_api().handler()
        handler.vae = RecordingEncoder(stochastic=True)
        original = handler.tiled_encode
        audio = torch.zeros((2, 5 * 48000), dtype=torch.float32)
        saved_state = torch.default_generator.get_state()
        try:
            torch.default_generator.manual_seed(12025)
            expected = original(audio, chunk_size=15 * 48000, overlap=2 * 48000)
            install_bounded_source_encoding(handler)
            torch.default_generator.manual_seed(12025)
            actual = handler.tiled_encode(audio)
        finally:
            torch.default_generator.set_state(saved_state)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_shorter_explicit_chunk_and_offload_choice_keep_actual_geometry(
        self,
    ) -> None:
        """A pre-existing smaller window is never enlarged by the bound."""
        for offload in (False, True):
            with self.subTest(offload=offload):
                handler = native_api().handler()
                encoder = RecordingEncoder()
                handler.vae = encoder
                values = torch.arange(30 * 48000, dtype=torch.float32)
                audio = values.reshape(1, 1, -1).expand(1, 2, -1)
                install_bounded_source_encoding(handler)
                actual = handler.tiled_encode(
                    audio,
                    chunk_size=10 * 48000,
                    overlap=48000,
                    offload_latent_to_cpu=offload,
                )
                self.assertLessEqual(max(encoder.input_frames), 10 * 48000)
                torch.testing.assert_close(
                    actual, values[::1920].reshape(1, 1, -1), rtol=0, atol=0
                )

    def test_codec_allocation_failure_is_not_hidden_by_an_adapter_fallback(
        self,
    ) -> None:
        """Failure still retires through the existing request/lifetime supervision."""
        handler = native_api().handler()
        encoder = RecordingEncoder()
        encoder.fail = True
        handler.vae = encoder
        install_bounded_source_encoding(handler)
        with self.assertRaisesRegex(RuntimeError, "controlled codec allocation"):
            handler.tiled_encode(torch.zeros((1, 2, 30 * 48000)))

    def test_different_source_clock_is_rejected_before_replacing_the_binding(
        self,
    ) -> None:
        """A future upstream clock change cannot silently reuse frame constants."""
        handler = native_api().handler()
        original = handler.tiled_encode
        handler.sample_rate = 44100
        with self.assertRaisesRegex(ValueError, "48kHz clock"):
            install_bounded_source_encoding(handler)
        self.assertEqual(handler.tiled_encode, original)
