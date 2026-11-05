"""Weights-free native library probes; CUDA cases test policy, not hardware."""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import skulk_ace_step
import torch
from skulk_ace_step.contracts import (
    NativeMusicDevice,
    NativeMusicLibraryProbe,
    NativeMusicMemoryCapacity,
)
from skulk_ace_step.policy import probe_libraries, sample_memory_capacity


def controlled_gpu_capacity() -> NativeMusicMemoryCapacity:
    """Separate synthetic capacity from precision tests; never hardware evidence."""
    return NativeMusicMemoryCapacity(
        sampled_at_unix_ms=1,
        host_total_bytes=100,
        host_available_bytes=80,
        device_index=0,
        device_memory_kind="discrete",
        device_total_bytes=100,
        device_available_bytes=80,
    )


class NativeLibraryProbeTests(unittest.TestCase):
    """Exercise real CPU math and exact accelerator-policy refusal/selection."""

    def test_actual_cpu_capacity_needs_no_accelerator_library(self) -> None:
        """Read actual host memory while proving CPU sampling never touches GPU APIs."""
        with patch(
            "torch.cuda.is_available",
            side_effect=AssertionError("CPU does not borrow VRAM"),
        ):
            observed = sample_memory_capacity("cpu")
        self.assertGreater(observed.sampled_at_unix_ms, 0)
        self.assertGreater(observed.host_total_bytes, 0)
        self.assertLessEqual(observed.host_available_bytes, observed.host_total_bytes)
        self.assertIsNone(observed.device_index)
        self.assertIsNone(observed.device_available_bytes)

    def test_selected_gpu_zero_uses_actual_free_and_driver_topology(self) -> None:
        """A larger second GPU cannot fund the implicit selected GPU's allocation."""
        lanes: tuple[NativeMusicDevice, ...] = ("cuda", "rocm")
        for lane in lanes:
            for integrated in (0, 1):
                with (
                    self.subTest(lane=lane, integrated=integrated),
                    patch(
                        "skulk_ace_step.policy.hip",
                        "controlled-HIP" if lane == "rocm" else None,
                    ),
                    patch(
                        "skulk_ace_step.policy.psutil.virtual_memory",
                        return_value=SimpleNamespace(total=1000, available=700),
                    ),
                    patch("torch.cuda.is_available", return_value=True),
                    patch("torch.cuda.current_device", return_value=0),
                    patch("torch.cuda.device_count", return_value=2),
                    patch(
                        "torch.cuda.get_device_properties",
                        return_value=SimpleNamespace(is_integrated=integrated),
                    ) as properties,
                    patch("torch.cuda.mem_get_info", return_value=(17, 25)) as memory,
                ):
                    observed = sample_memory_capacity(lane)
                properties.assert_called_once_with(0)
                memory.assert_called_once_with(0)
                self.assertEqual(observed.device_total_bytes, 25)
                self.assertEqual(observed.device_available_bytes, 17)
                self.assertEqual(
                    observed.device_memory_kind, "unified" if integrated else "discrete"
                )
                self.assertEqual(
                    observed.shared_available_bytes, 700 if integrated else None
                )

    def test_foreign_current_gpu_is_refused_without_sampling_another_device(
        self,
    ) -> None:
        """A hidden current-device switch cannot silently alter the admitted target."""
        with (
            patch("skulk_ace_step.policy.hip", None),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.current_device", return_value=1),
            patch(
                "torch.cuda.mem_get_info", side_effect=AssertionError("no foreign read")
            ),
            self.assertRaisesRegex(ValueError, "device zero"),
        ):
            sample_memory_capacity("cuda")

    def test_metal_capacity_preserves_the_single_physical_host_pool(self) -> None:
        """MPS driver allocation and host availability are separate constraints."""
        with (
            patch(
                "skulk_ace_step.policy.psutil.virtual_memory",
                return_value=SimpleNamespace(total=1000, available=700),
            ),
            patch("torch.backends.mps.is_available", return_value=True),
            patch("torch.mps.recommended_max_memory", return_value=750),
            patch("torch.mps.driver_allocated_memory", return_value=200),
        ):
            observed = sample_memory_capacity("mps")
        self.assertEqual(observed.device_available_bytes, 550)
        self.assertEqual(observed.shared_total_bytes, 1000)
        self.assertEqual(observed.shared_available_bytes, 700)
        self.assertNotEqual(observed.shared_available_bytes, 700 + 550)

    def test_invalid_driver_reading_is_not_invented_capacity(self) -> None:
        """Negative/over-total free values fail the shared strict observation contract."""
        with (
            patch("skulk_ace_step.policy.hip", None),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.current_device", return_value=0),
            patch(
                "torch.cuda.get_device_properties",
                return_value=SimpleNamespace(is_integrated=0),
            ),
            patch("torch.cuda.mem_get_info", return_value=(26, 25)),
            self.assertRaises(ValueError),
        ):
            sample_memory_capacity("cuda")

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
            patch(
                "skulk_ace_step.policy.sample_memory_capacity",
                return_value=controlled_gpu_capacity(),
            ),
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
                patch(
                    "skulk_ace_step.policy.sample_memory_capacity",
                    return_value=controlled_gpu_capacity(),
                ),
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
