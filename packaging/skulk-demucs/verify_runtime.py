"""Run native media/probe regression inside the isolated inference environment."""

from __future__ import annotations

import importlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
from skulk_demucs.bindings import soundfile
from skulk_demucs.contracts import SeparationSource, source_from_wav
from skulk_demucs.engine import WEIGHT_SIZE, probe_device, verify_weights
from skulk_demucs.media import export_stems, read_original


class NativeMediaTests(unittest.TestCase):
    """Exercise actual native codecs and dependency APIs with analytic samples."""

    def test_model_and_resampler_dependencies_import(self) -> None:
        """Codec success must not hide broken model or resampler dependencies."""
        for name in ("demucs.hf", "demucs.apply", "julius"):
            self.assertIsNotNone(importlib.import_module(name))

    def test_shared_gain_and_encoded_alignment(self) -> None:
        """One attenuation preserves all four source ratios without PCM clipping."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "source.wav"
            samples = np.zeros((8000, 1), dtype=np.float32)
            with path.open("xb") as source:
                soundfile.write(source, samples, 8000, format="WAV", subtype="PCM_24")
            source_facts = source_from_wav(path.read_bytes())
            estimates = [
                np.full((44100, 2), peak, dtype=np.float32)
                for peak in (2.0, 1.0, 0.5, 0.25)
            ]
            output = export_stems(root, source_facts, estimates, 44100)
            self.assertLess(output.common_gain, 0.5)
            self.assertEqual(output.pre_gain_peak, 2.0)
            for record, expected in zip(
                output.stems, (1.0, 0.5, 0.25, 0.125), strict=True
            ):
                self.assertAlmostEqual(record.peak, expected, places=5)
                facts = source_from_wav((root / f"{record.name}.wav").read_bytes())
                self.assertEqual(
                    (facts.frames, facts.sample_rate, facts.channels, facts.bit_depth),
                    (44100, 44100, 2, 24),
                )
            np.testing.assert_array_equal(read_original(path, source_facts), samples)

    def test_decoder_retains_original_and_rejects_changed_bytes(self) -> None:
        """Decode works from the already bounded bytes and checks the full digest."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.wav"
            samples = np.full((8000, 2), 0.25, dtype=np.float32)
            with path.open("xb") as source:
                soundfile.write(source, samples, 8000, format="WAV", subtype="PCM_24")
            original = path.read_bytes()
            facts = source_from_wav(original)
            np.testing.assert_array_equal(read_original(path, facts), samples)
            self.assertEqual(path.read_bytes(), original)
            path.write_bytes(original[:-1] + b"\x01")
            with self.assertRaisesRegex(ValueError, "admitted metadata"):
                read_original(path, facts)

    def test_bad_native_outputs_never_become_a_complete_set(self) -> None:
        """Missing, unaligned or nonfinite tensors fail before any WAV export."""
        source = SeparationSource(
            sha256="a" * 64,
            size_bytes=16044,
            sample_rate=8000,
            channels=1,
            frames=8000,
            encoding="pcm",
            bit_depth=16,
        )
        valid = [np.zeros((44100, 2), dtype=np.float32) for _ in range(4)]
        cases = [
            valid[:3],
            [valid[0][:-1], *valid[1:]],
            [np.full((44100, 2), np.nan, dtype=np.float32), *valid[1:]],
        ]
        for stems in cases:
            with (
                self.subTest(count=len(stems)),
                tempfile.TemporaryDirectory() as temporary,
            ):
                directory = Path(temporary)
                with self.assertRaises(ValueError):
                    export_stems(directory, source, stems, 44100)
                self.assertEqual(list(directory.iterdir()), [])

    def test_export_never_overwrites_previous_files(self) -> None:
        """An old or original artifact is protected even with a reused directory."""
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "drums.wav").write_bytes(b"previous")
            source = SeparationSource(
                sha256="a" * 64,
                size_bytes=16044,
                sample_rate=8000,
                channels=1,
                frames=8000,
                encoding="pcm",
                bit_depth=16,
            )
            with self.assertRaises(FileExistsError):
                export_stems(
                    directory,
                    source,
                    [np.zeros((8000, 2), dtype=np.float32) for _ in range(4)],
                    8000,
                )
            self.assertEqual((directory / "drums.wav").read_bytes(), b"previous")

    def test_incomplete_weights_are_refused_before_loading(self) -> None:
        """Neither upstream Hub model management nor pickle fallback is invoked."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "not-the-model.safetensors"
            path.write_bytes(b"not a model")
            with self.assertRaisesRegex(ValueError, "exact pinned"):
                verify_weights(path)

    def test_cpu_probe_executes_native_float32(self) -> None:
        """Package availability alone is not the result of the native math probe."""
        self.assertEqual(probe_device("cpu"), "cpu")

    def test_wrong_model_digest_is_refused(self) -> None:
        """A correctly sized file still needs the complete immutable object hash."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "changed.safetensors"
            with path.open("wb") as source:
                source.truncate(WEIGHT_SIZE)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                verify_weights(path)


if __name__ == "__main__":
    unittest.main()
