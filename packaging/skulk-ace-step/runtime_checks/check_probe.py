"""Weights-free native library probes; CUDA cases test policy, not hardware."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import skulk_ace_step
import torch
from skulk_ace_step.contracts import NativeMusicLibraryProbe
from skulk_ace_step.policy import probe_libraries


class NativeLibraryProbeTests(unittest.TestCase):
    """Exercise real CPU math and exact accelerator-policy refusal/selection."""

    def test_cpu_probe_needs_no_model_configuration_or_upstream_loader(self) -> None:
        """Library readiness has no invented model identity, path or weight read."""
        with patch(
            "skulk_ace_step.upstream_bindings.native_api",
            side_effect=AssertionError("no model loader"),
        ):
            receipt = probe_libraries("cpu", "b" * 64)
        self.assertEqual(receipt.device, "cpu")
        self.assertEqual(receipt.precision, "float32")
        self.assertEqual(receipt.autoencoder_precision, "float32")
        self.assertIsNone(receipt.cuda_compute_capability)
        self.assertEqual(receipt.payload_sha256, "b" * 64)
        self.assertEqual(receipt.torch_version, str(torch.__version__))
        self.assertEqual(
            NativeMusicLibraryProbe.model_validate_json(receipt.model_dump_json()),
            receipt,
        )

    def test_unavailable_mps_cannot_become_cpu_readiness(self) -> None:
        """An explicit accelerator request never quietly selects CPU."""
        with (
            patch.dict(os.environ, {"PYTORCH_ENABLE_MPS_FALLBACK": "0"}),
            patch("torch.backends.mps.is_available", return_value=False),
            self.assertRaisesRegex(ValueError, "Metal"),
        ):
            probe_libraries("mps", "b" * 64)

    def test_cuda_probe_requires_real_cuda_runtime_and_device(self) -> None:
        """CPU and HIP dependency profiles cannot advertise CUDA via a flag."""
        with (
            patch("torch.cuda.is_available", return_value=False),
            self.assertRaisesRegex(ValueError, "unavailable"),
        ):
            probe_libraries("cuda", "b" * 64)

    def test_preampere_policy_ignores_bfloat16_emulation(self) -> None:
        """The pinned initializer requires native SM8 BF16, not emulated support."""
        original = torch.ones

        def cpu_math(
            shape: tuple[int, int], *, dtype: torch.dtype, device: str
        ) -> torch.Tensor:
            self.assertEqual(device, "cuda")
            return original(shape, dtype=dtype, device="cpu")

        with (
            patch("skulk_ace_step.policy.hip", None),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(7, 5)),
            patch("torch.cuda.is_bf16_supported", return_value=True),
            patch("torch.ones", side_effect=cpu_math),
        ):
            receipt = probe_libraries("cuda", "b" * 64)
        self.assertEqual(receipt.precision, "float16")
        self.assertEqual(receipt.autoencoder_precision, "float16")
        self.assertEqual(receipt.cuda_compute_capability, (7, 5))

    def test_ampere_and_later_policy_selects_native_bfloat16(self) -> None:
        """Native capability drives precision without a tested-device allowlist."""
        original = torch.ones

        def cpu_math(
            shape: tuple[int, int], *, dtype: torch.dtype, device: str
        ) -> torch.Tensor:
            self.assertEqual(device, "cuda")
            return original(shape, dtype=dtype, device="cpu")

        for capability in ((8, 0), (8, 9), (12, 1), (13, 0)):
            with (
                self.subTest(capability=capability),
                patch("skulk_ace_step.policy.hip", None),
                patch("torch.cuda.is_available", return_value=True),
                patch("torch.cuda.get_device_capability", return_value=capability),
                patch("torch.ones", side_effect=cpu_math),
            ):
                receipt = probe_libraries("cuda", "b" * 64)
            self.assertEqual(receipt.precision, "bfloat16")
            self.assertEqual(receipt.cuda_compute_capability, capability)

    def test_bad_library_math_fails_instead_of_advertising_readiness(self) -> None:
        """An importable package is insufficient when its tiny execution fails."""
        with (
            patch("torch.equal", return_value=False),
            self.assertRaisesRegex(ValueError, "execution probe"),
        ):
            probe_libraries("cpu", "b" * 64)

    def test_probe_metadata_cannot_describe_a_loaded_model(self) -> None:
        """The strict library record is distinct from NativeMusicReady."""
        receipt = probe_libraries("cpu", "b" * 64)
        with self.assertRaises(ValueError):
            NativeMusicLibraryProbe.model_validate_json(
                receipt.model_dump_json()[:-1] + ',"model_identity":"' + "a" * 64 + '"}'
            )

    def test_actual_private_cli_probes_without_a_model_config(self) -> None:
        """The supervised process emits only the distinct library receipt."""
        package = Path(skulk_ace_step.__file__).resolve().parent.parent
        command = [
            sys.executable,
            "-I",
            "-B",
            "-c",
            'import runpy,sys;sys.path.insert(0,sys.argv.pop(1));runpy.run_module("skulk_ace_step",run_name="__main__")',
            str(package),
            "--probe",
            "cpu",
            "--parent-pid",
            str(os.getpid()),
        ]
        result = subprocess.run(
            command, start_new_session=True, capture_output=True, timeout=30
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        receipt = NativeMusicLibraryProbe.model_validate_json(result.stdout)
        self.assertEqual(receipt.kind, "library_probe")
        self.assertEqual(receipt.device, "cpu")
        self.assertNotIn(b"acestep.gpu_config", result.stderr)

    def test_cli_rejects_mixed_probe_and_serve_intents(self) -> None:
        """Preparation cannot accidentally become a model load through extra flags."""
        package = Path(skulk_ace_step.__file__).resolve().parent.parent
        command = [
            sys.executable,
            "-I",
            "-B",
            "-c",
            'import runpy,sys;sys.path.insert(0,sys.argv.pop(1));runpy.run_module("skulk_ace_step",run_name="__main__")',
            str(package),
            "--probe",
            "cpu",
            "--serve",
            "not-a-model-config.json",
            "--parent-pid",
            str(os.getpid()),
        ]
        result = subprocess.run(
            command, start_new_session=True, capture_output=True, timeout=30
        )
        self.assertEqual(result.returncode, 2)
        self.assertFalse(result.stdout)


if __name__ == "__main__":
    unittest.main()
