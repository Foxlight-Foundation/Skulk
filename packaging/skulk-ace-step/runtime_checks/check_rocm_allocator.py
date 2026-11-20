"""Exercise per-window ROCm cache policy with real modules and controlled effects.

These boundary tests use CPU tensors and mocked allocator calls. They establish
hook ordering, tensor preservation and failure behavior, never GPU qualification.
Only the ROCm dependency closure requires this ROCm-specific adapter behavior;
unchanged published CPU/CUDA packages retain their earlier adapter bytes.
"""

from __future__ import annotations

import gc
import unittest
from typing import cast
from unittest.mock import patch

import torch
from skulk_ace_step import upstream_bindings
from skulk_ace_step.contracts import NativeMusicDevice


class ObservableEncoder(torch.nn.Module):
    """A real parameterized module recording execution and original inputs."""

    def __init__(self, events: list[str]) -> None:
        """Create a visible weight without loading any model artifacts."""
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.5))
        self.events = events
        self.inputs: list[torch.Tensor] = []

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Record the unchanged input object and compute an actual tensor."""
        self.events.append("forward")
        self.inputs.append(audio)
        return audio * self.weight


@unittest.skipUnless(
    torch.version.hip is not None,
    "ROCm-specific installed adapter checks require the ROCm dependency closure",
)
class NativeRocmAllocatorTests(unittest.TestCase):
    """Verify the cache policy around actual nn.Module forward dispatch."""

    def test_each_original_window_reclaims_before_existing_hooks_and_forward(
        self,
    ) -> None:
        """Repeated calls preserve weights, inputs, outputs and hook lifetime."""
        handler = upstream_bindings.native_api().handler()
        events: list[str] = []
        encoder = ObservableEncoder(events)
        vae = torch.nn.Module()
        vae.add_module("encoder", encoder)
        handler.vae = vae
        other = ObservableEncoder([])
        encoder.register_forward_pre_hook(
            lambda _module, _arguments: events.append("existing")
        )
        original_forward = type(encoder).forward
        original_weight = encoder.weight
        saved_weight = encoder.weight.detach().clone()
        source = torch.arange(16, dtype=torch.float32).reshape(1, 2, 8)
        expected = source * saved_weight

        def synchronize_device(_device: int) -> None:
            events.append("synchronize")

        with (
            patch(
                "torch.cuda.synchronize", side_effect=synchronize_device
            ) as synchronize,
            patch(
                "torch.cuda.empty_cache", side_effect=lambda: events.append("release")
            ) as release,
        ):
            upstream_bindings.install_rocm_encoding_cache_release(handler, "rocm")
            gc.collect()
            for _ in range(3):
                # Module dispatch is untyped; exercise it so the real hook runs.
                actual = cast(torch.Tensor, encoder(source))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertIs(encoder.inputs[-1], source)
            other(source)
        self.assertEqual(events, ["synchronize", "release", "existing", "forward"] * 3)
        self.assertEqual([call.args for call in synchronize.call_args_list], [(0,)] * 3)
        self.assertEqual(release.call_count, 3)
        self.assertIs(encoder.weight, original_weight)
        self.assertIs(type(encoder).forward, original_forward)
        torch.testing.assert_close(
            encoder.weight.detach(), saved_weight, rtol=0, atol=0
        )
        torch.testing.assert_close(
            source, torch.arange(16).reshape(1, 2, 8).float(), rtol=0, atol=0
        )

    def test_non_rocm_handlers_need_no_autoencoder_and_touch_no_allocator(self) -> None:
        """Explicit CPU, CUDA and Metal policies remain independent of ROCm."""
        handler = upstream_bindings.native_api().handler()
        self.assertIsNone(handler.vae)
        with (
            patch("torch.cuda.synchronize") as synchronize,
            patch("torch.cuda.empty_cache") as release,
        ):
            devices: tuple[NativeMusicDevice, ...] = ("cpu", "cuda", "mps")
            for device in devices:
                upstream_bindings.install_rocm_encoding_cache_release(handler, device)
        synchronize.assert_not_called()
        release.assert_not_called()

    def test_missing_initialized_autoencoder_fails_before_readiness(self) -> None:
        """ROCm cannot silently omit required per-window resource policy."""
        handler = upstream_bindings.native_api().handler()
        with self.assertRaisesRegex(ValueError, "initialized autoencoder"):
            upstream_bindings.install_rocm_encoding_cache_release(handler, "rocm")

    def test_missing_encoder_contract_fails_instead_of_patching_another_module(
        self,
    ) -> None:
        """A changed pinned codec structure must be reviewed explicitly."""
        handler = upstream_bindings.native_api().handler()
        handler.vae = torch.nn.Linear(1, 1)
        with self.assertRaises(AttributeError):
            upstream_bindings.install_rocm_encoding_cache_release(handler, "rocm")

    def test_allocator_failure_prevents_the_original_encoder_forward(self) -> None:
        """Both synchronization and release errors remain visible to retirement."""
        for failed_effect in ("synchronize", "empty_cache"):
            with self.subTest(failed_effect=failed_effect):
                handler = upstream_bindings.native_api().handler()
                encoder = ObservableEncoder([])
                handler.vae = torch.nn.Module()
                handler.vae.add_module("encoder", encoder)
                upstream_bindings.install_rocm_encoding_cache_release(handler, "rocm")
                with (
                    patch("torch.cuda.synchronize") as synchronize,
                    patch("torch.cuda.empty_cache") as release,
                ):
                    (
                        synchronize if failed_effect == "synchronize" else release
                    ).side_effect = RuntimeError("controlled allocator failure")
                    with self.assertRaisesRegex(
                        RuntimeError, "controlled allocator failure"
                    ):
                        encoder(torch.ones((1, 2, 8)))
                self.assertEqual(encoder.inputs, [])
                if failed_effect == "synchronize":
                    release.assert_not_called()
