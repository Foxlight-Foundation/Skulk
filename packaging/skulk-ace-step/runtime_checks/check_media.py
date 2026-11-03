"""Real offline native audio checks; no model weights or Core imports."""

import hashlib
import math
import os
import struct
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np
from skulk_ace_step.audio_separation import source_from_wav
from skulk_ace_step.bindings import soundfile
from skulk_ace_step.contracts import (
    NativeMusicExecution,
    NativeMusicRequest,
    NativeMusicResult,
)
from skulk_ace_step.media import (
    export_result,
    prepare_conditioning,
    read_original,
    write_conditioning,
)


class NativeMediaTests(unittest.TestCase):
    """Exercise actual codecs/resampling against original and output manifests."""

    def setUp(self) -> None:
        """Own every generated test file outside product and model stores."""
        self.temporary = tempfile.TemporaryDirectory(prefix="native-music-media-")
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        """Remove only the explicitly owned temporary files."""
        self.temporary.cleanup()

    def request(
        self,
        *,
        rate: int = 48000,
        frames: int = 48000,
        channels: int = 2,
        continuation: bool = False,
    ) -> NativeMusicRequest:
        """Build a measured real PCM16 source and an exact private edit request."""
        path = self.root / "source.wav"
        with wave.open(str(path), "wb") as output:
            output.setparams((channels, 2, rate, frames, "NONE", "not compressed"))
            data = b"".join(
                struct.pack("<h", round(8192 * math.sin(2 * math.pi * 1000 * i / rate)))
                * channels
                for i in range(frames)
            )
            output.writeframes(data)
        source = source_from_wav(path.read_bytes())
        mapped = (frames * 48000 + rate - 1) // rate
        target = mapped + 48000 if continuation else mapped
        return NativeMusicRequest(
            request_id="media-test",
            operation="continue" if continuation else "repaint",
            prompt="A sparse warm string arrangement",
            seed=12025,
            target_frames=target,
            source=source,
            source_path=str(path),
            region_start_frame=mapped if continuation else 0,
            region_end_frame=target,
            output_directory=str(self.root / "result"),
        )

    def test_native_rate_context_preserves_samples_and_original_bytes(self) -> None:
        """Native context avoids unnecessary resampling or original replacement."""
        request = self.request()
        path = Path(request.source_path or "")
        original = path.read_bytes()
        assert request.source is not None
        decoded = read_original(path, request.source)
        prepared = prepare_conditioning(request)
        self.assertTrue(np.array_equal(decoded, prepared.samples))
        self.assertEqual(prepared.source, request.source)
        self.assertEqual(prepared.gain, 1.0)
        self.assertEqual(path.read_bytes(), original)

    def test_mono_coprime_rate_context_has_correct_native_clock_and_tone(self) -> None:
        """The bounded FIR keeps a real tone and frame map at nonstandard input rates."""
        request = self.request(rate=44101, frames=44108, channels=1)
        samples = prepare_conditioning(request).samples
        self.assertEqual(samples.shape, (request.target_frames, 2))
        self.assertTrue(np.isfinite(samples).all())
        self.assertTrue(np.array_equal(samples[:, 0], samples[:, 1]))
        frequency = float(
            np.argmax(np.abs(np.fft.rfft(samples[:, 0]))) * 48000 / len(samples)
        )
        self.assertLess(abs(frequency - 1000), 2)

    def test_continue_pads_only_the_tail_and_records_actual_conditioning(self) -> None:
        """The original remains unchanged and every added context frame is silence."""
        request = self.request(continuation=True)
        assert request.source is not None
        path = Path(request.source_path or "")
        original = path.read_bytes()
        decoded = read_original(path, request.source)
        samples = prepare_conditioning(request).samples
        self.assertTrue(np.array_equal(samples[:48000], decoded))
        self.assertTrue(np.all(samples[48000:] == 0))
        native = self.root / "conditioning.wav"
        facts = write_conditioning(native, samples)
        self.assertEqual(
            (
                facts.sample_rate,
                facts.frames,
                facts.channels,
                facts.encoding,
                facts.bit_depth,
            ),
            (48000, 96000, 2, "float", 32),
        )
        self.assertEqual(facts.sha256, hashlib.sha256(native.read_bytes()).hexdigest())
        with native.open("rb") as stream:
            round_trip, _ = soundfile.read(stream, dtype="float32", always_2d=True)
        self.assertTrue(np.array_equal(round_trip, samples))
        self.assertEqual(path.read_bytes(), original)

    def test_pcm24_export_reports_actual_bytes_and_shared_headroom(self) -> None:
        """Real encoding attenuates excessive peaks and preserves complete stereo duration."""
        request = self.request()
        samples = prepare_conditioning(request).samples
        samples = samples * np.float32(8)
        path = self.root / "result.wav"
        facts = export_result(path, samples, request)
        self.assertLess(facts.gain, 1)
        self.assertLessEqual(facts.peak, 1)
        actual = source_from_wav(path.read_bytes())
        self.assertEqual(
            (
                actual.encoding,
                actual.bit_depth,
                actual.sample_rate,
                actual.channels,
                actual.frames,
            ),
            ("pcm", 24, 48000, 2, 48000),
        )
        self.assertEqual(
            (actual.sha256, actual.size_bytes), (facts.sha256, facts.size_bytes)
        )

    def test_edit_output_with_wrong_alignment_is_never_written(self) -> None:
        """A plausible musical result cannot replace an incorrectly sized timeline edit."""
        request = self.request()
        path = self.root / "wrong.wav"
        with self.assertRaises(ValueError):
            export_result(path, np.zeros((48001, 2), dtype=np.float32), request)
        self.assertFalse(path.exists())

    def test_nonfinite_output_is_never_written(self) -> None:
        """Numerical failures remain failed jobs rather than malformed audio downloads."""
        request = self.request()
        samples = np.zeros((48000, 2), dtype=np.float32)
        samples[123, 1] = np.nan
        path = self.root / "nonfinite.wav"
        with self.assertRaises(ValueError):
            export_result(path, samples, request)
        self.assertFalse(path.exists())

    def test_existing_result_is_never_replaced(self) -> None:
        """Even an accidental repeated private directory cannot overwrite retained work."""
        request = self.request()
        path = self.root / "existing.wav"
        path.write_bytes(b"retained work")
        with self.assertRaises(FileExistsError):
            export_result(path, np.zeros((48000, 2), dtype=np.float32), request)
        self.assertEqual(path.read_bytes(), b"retained work")

    def test_source_identity_change_fails_before_conditioning(self) -> None:
        """Same-sized source substitution cannot silently condition a different edit."""
        request = self.request()
        path = Path(request.source_path or "")
        data = bytearray(path.read_bytes())
        data[-1] ^= 1
        path.write_bytes(data)
        with self.assertRaises(ValueError):
            prepare_conditioning(request)

    def test_symlink_and_fifo_source_are_not_opened_as_audio(self) -> None:
        """Input special files cannot escape the bounded regular-file reader."""
        request = self.request()
        assert request.source is not None
        original = Path(request.source_path or "")
        link = self.root / "link.wav"
        link.symlink_to(original)
        with self.assertRaises(OSError):
            read_original(link, request.source)
        fifo = self.root / "fifo.wav"
        os.mkfifo(fifo)
        with self.assertRaises(ValueError):
            read_original(fifo, request.source)

    def test_result_lineage_retains_original_and_actual_conditioning(self) -> None:
        """The native result keeps transformed model input distinct from the original."""
        request = self.request()
        prepared = prepare_conditioning(request)
        samples, original = prepared.samples, prepared.source
        conditioning = write_conditioning(self.root / "conditioning.wav", samples)
        output = export_result(self.root / "result.wav", samples, request)
        result = NativeMusicResult(
            request_id=request.request_id,
            model_identity="a" * 64,
            payload_sha256="b" * 64,
            operation=request.operation,
            seed=request.seed,
            source_sha256=original.sha256,
            conditioning_sha256=conditioning.sha256,
            conditioning_gain=prepared.gain,
            output=output,
            generation_seconds=1.0,
            execution=NativeMusicExecution(
                device="cpu",
                precision="float32",
                text_encoder_device="cpu",
                text_encoder_precision="float32",
                autoencoder_device="cpu",
                autoencoder_precision="float32",
            ),
        )
        self.assertEqual(
            NativeMusicResult.model_validate_json(result.model_dump_json()), result
        )
        self.assertNotEqual(result.source_sha256, result.conditioning_sha256)

    def test_float_context_headroom_is_explicit_and_does_not_clip_or_modify_original(
        self,
    ) -> None:
        """Pinned native clamping cannot silently flatten an over-range float context."""
        request = self.request()
        original = Path(request.source_path or "")
        samples = np.zeros((48000, 2), dtype=np.float32)
        samples[100:200, 0] = 1.5
        samples[100:200, 1] = -3.0
        with original.open("wb") as stream:
            soundfile.write(stream, samples, 48000, format="WAV", subtype="FLOAT")
        data = original.read_bytes()
        request = NativeMusicRequest.model_validate_json(
            request.model_copy(
                update={"source": source_from_wav(data)}
            ).model_dump_json()
        )
        prepared = prepare_conditioning(request)
        self.assertLess(prepared.gain, 1.0)
        self.assertLess(float(np.abs(prepared.samples).max()), 1.0)
        np.testing.assert_array_equal(prepared.samples, samples * prepared.gain)
        self.assertAlmostEqual(
            float(prepared.samples[150, 1]), -2 * float(prepared.samples[150, 0])
        )
        self.assertEqual(original.read_bytes(), data)


if __name__ == "__main__":
    unittest.main()
