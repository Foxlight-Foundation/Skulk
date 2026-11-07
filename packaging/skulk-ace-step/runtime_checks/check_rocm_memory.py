"""Process/device allocation sensing independent of Torch's allocator lock."""

from __future__ import annotations

import os
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import check_support
from skulk_ace_step.contracts import NativeMusicMemoryCapacity
from skulk_ace_step.resources import sample_music_memory
from skulk_ace_step.rocm_memory import (
    _allocation_bytes,
    _pci_address,
    _read_fdinfos,
    rocm_allocated_bytes,
)


def client(
    identity: int = 7,
    pci: str = "0000:02:00.0",
    vram: str = "3 MiB",
    gtt: str = "2 KiB",
    cpu: str = "10",
) -> str:
    """Construct explicit kernel accounting fixtures, never GPU qualification."""
    return (
        f"drm-driver:\tamdgpu\ndrm-client-id:\t{identity}\ndrm-pdev:\t{pci}\n"
        f"drm-total-vram:\t{vram}\ndrm-total-gtt:\t{gtt}\ndrm-total-cpu:\t{cpu}\n"
    )


class RocmMemoryTests(unittest.TestCase):
    """Keep ROCm heartbeat sampling outside the inference allocator mutex."""

    def test_rocm_sampling_does_not_enter_torch_allocator_statistics(self) -> None:
        """A blocked allocator query cannot prevent the independent observation."""
        with tempfile.TemporaryDirectory() as temporary:
            configuration = check_support.configuration(Path(temporary)).model_copy(
                update={"device": "rocm"}
            )
            capacity = NativeMusicMemoryCapacity(
                sampled_at_unix_ms=1000,
                host_total_bytes=1000,
                host_available_bytes=900,
                device_index=0,
                device_memory_kind="unified",
                device_total_bytes=1000,
                device_available_bytes=900,
                shared_total_bytes=1000,
                shared_available_bytes=900,
            )
            with (
                patch(
                    "torch.cuda.memory_reserved",
                    side_effect=AssertionError("inference holds the allocator lock"),
                ),
                patch(
                    "skulk_ace_step.resources.rocm_allocated_bytes",
                    return_value=123,
                ),
                patch(
                    "skulk_ace_step.policy.sample_memory_capacity",
                    return_value=capacity,
                ),
            ):
                observed = sample_music_memory(configuration, "b" * 64)
            self.assertEqual(observed.device_allocated_bytes, 123)
            self.assertEqual(observed.process_id, os.getpid())
            self.assertEqual(observed.device, "rocm")
            self.assertEqual(observed.capacity, capacity)

    def test_rocm_sensor_failure_cannot_fall_back_to_allocator_statistics(self) -> None:
        """Unavailable driver accounting remains a fatal sensing error."""
        with tempfile.TemporaryDirectory() as temporary:
            configuration = check_support.configuration(Path(temporary)).model_copy(
                update={"device": "rocm"}
            )
            with (
                patch(
                    "skulk_ace_step.resources.rocm_allocated_bytes",
                    side_effect=ValueError("missing selected client"),
                ),
                patch(
                    "torch.cuda.memory_reserved",
                    side_effect=AssertionError("must not enter the allocator"),
                ),
                self.assertRaisesRegex(ValueError, "missing selected client"),
            ):
                sample_music_memory(configuration, "b" * 64)

    def test_accounting_includes_requested_not_only_resident_buffers(self) -> None:
        """Evicted, CPU and driver allocations cannot disappear from the ceiling."""
        record = client() + "drm-resident-vram: 1\ndrm-memory-gtt: 1\n"
        self.assertEqual(
            _allocation_bytes([record], "0000:02:00.0"), 3 * 1024**2 + 2 * 1024 + 10
        )

    def test_duplicate_descriptors_do_not_multiply_the_same_client(self) -> None:
        """Alias reads use conservative regional maxima across a sampling race."""
        self.assertEqual(
            _allocation_bytes(
                [
                    client(vram="10", gtt="2", cpu="0"),
                    client(vram="9", gtt="3", cpu="0"),
                ],
                "0000:02:00.0",
            ),
            13,
        )

    def test_distinct_owned_clients_and_new_memory_regions_remain_counted(self) -> None:
        """Shared buffers may be conservatively charged, never silently subtracted."""
        records = [
            client(1, vram="10", gtt="0", cpu="0"),
            client(2, vram="20", gtt="0", cpu="0") + "drm-total-newregion: 5\n",
        ]
        self.assertEqual(_allocation_bytes(records, "0000:02:00.0"), 35)

    def test_engine_clock_is_not_a_memory_region(self) -> None:
        """Standard DRM total-cycle timestamps must not become allocation bytes."""
        self.assertEqual(
            _allocation_bytes(
                [
                    client(vram="10", gtt="0", cpu="0")
                    + "drm-total-cycles-gfx: 999999999999\n"
                ],
                "0000:02:00.0",
            ),
            10,
        )

    def test_other_devices_cannot_fund_or_replace_the_selected_gpu(self) -> None:
        """Runtime-visible zero maps by PCI identity, independently of fd ordering."""
        selected = client(vram="10", gtt="0", cpu="0")
        foreign = client(8, "0000:03:00.0", vram="999 MiB")
        self.assertEqual(_allocation_bytes([foreign, selected], "00000000:02:00.0"), 10)
        with self.assertRaisesRegex(ValueError, "no owned client"):
            _allocation_bytes([foreign], "0000:02:00.0")

    def test_missing_or_ambiguous_accounting_fails_closed(self) -> None:
        """A sensing failure cannot manufacture a zero or a partial allocation."""
        record = client()
        bad = (
            "",
            record.replace("drm-driver:\tamdgpu\n", ""),
            record.replace("drm-pdev:\t0000:02:00.0\n", ""),
            record.replace("drm-client-id:\t7", "drm-client-id:\tunknown"),
            record.replace("drm-total-cpu:\t10\n", ""),
            record + "drm-client-id: 8\n",
            record.replace("3 MiB", "3 MB"),
            record.replace("3 MiB", "-1"),
            record.replace("3 MiB", "1.5 MiB"),
        )
        for value in bad:
            with self.subTest(record=value), self.assertRaises(ValueError):
                _allocation_bytes([value], "0000:02:00.0")

    def test_pci_identity_is_validated_before_accounting(self) -> None:
        """Unknown, malformed and out-of-range addresses cannot match a device."""
        for value in ("", "02:00.0", "0000:02:20.0", "0000:02:00.8", "0000:xx:00.0"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _pci_address(value)
        self.assertEqual(_pci_address("00000000:AB:1F.7"), "0000:ab:1f.7")

    def test_real_descriptor_reader_retains_only_bounded_complete_records(self) -> None:
        """Oversized or unavailable sensing is an error, never fallback statistics."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "7").write_text(client())
            self.assertEqual(_read_fdinfos(root), (client(),))
            (root / "7").write_text("x" * 65537)
            with self.assertRaisesRegex(ValueError, "exceeds its bound"):
                _read_fdinfos(root)
            (root / "7").unlink()
            root.rmdir()
            with self.assertRaises(FileNotFoundError):
                _read_fdinfos(root)

    def test_selected_runtime_initialization_precedes_descriptor_inventory(
        self,
    ) -> None:
        """The first heartbeat must observe clients created by lazy HIP startup."""
        order: list[str] = []

        def selected() -> str:
            order.append("device")
            return "0000:02:00.0"

        def records() -> tuple[str, ...]:
            self.assertEqual(order, ["device"])
            return (client(vram="10", gtt="0", cpu="0"),)

        with (
            patch(
                "skulk_ace_step.rocm_memory._selected_pci_address", side_effect=selected
            ),
            patch("skulk_ace_step.rocm_memory._read_fdinfos", side_effect=records),
        ):
            self.assertEqual(rocm_allocated_bytes(), 10)

    def test_descriptor_limit_is_checked_before_opening_any_record(self) -> None:
        """An excessive descriptor set cannot create an unbounded sample."""
        entries = (SimpleNamespace(path="unused") for _ in range(4097))
        with (
            patch(
                "skulk_ace_step.rocm_memory.os.scandir",
                return_value=nullcontext(entries),
            ),
            self.assertRaisesRegex(ValueError, "inventory exceeds"),
        ):
            _read_fdinfos()


if __name__ == "__main__":
    unittest.main()
