"""Small real WAV edit contracts for custody/admission tests, never model output."""

import hashlib
import io
import time
import wave

from skulk.api.music_edit_jobs import MusicEditJob
from skulk.shared.tests.native_music_fixtures import native_music_fixture
from skulk.shared.types.audio_separation import source_from_wav
from skulk.shared.types.common import CommandId
from skulk.shared.types.music_edits import (
    MusicEditingOutput,
    MusicRepaintRequest,
    resolve_music_edit,
)
from skulk.shared.types.native_music import (
    NativeMusicExecution,
    NativeMusicResult,
    NativeMusicWav,
)


def music_edit_fixture(
    identifier: str = "edit", *, value: int = 1000,
) -> tuple[MusicEditJob, MusicEditingOutput, bytes, bytes]:
    """Return exact one-second source/output bytes and their immutable edit facts.

    The encoded original is mono PCM16/8 kHz; the result is stereo PCM24/48 kHz.
    These synthetic samples exercise transport and retention, not music quality.
    """
    original = io.BytesIO()
    with wave.open(original, "wb") as audio:
        audio.setparams((1, 2, 8000, 8000, "NONE", "not compressed"))
        audio.writeframes(bytes(16000))
    rendered = io.BytesIO()
    samples = bytearray(48000 * 6)
    samples[:3] = value.to_bytes(3, "little", signed=True)
    samples[-3:] = (-value).to_bytes(3, "little", signed=True)
    with wave.open(rendered, "wb") as audio:
        audio.setparams((2, 3, 48000, 48000, "NONE", "not compressed"))
        audio.writeframes(samples)
    source_bytes, result_bytes = original.getvalue(), rendered.getvalue()
    source = source_from_wav(source_bytes)
    card, _ = native_music_fixture()
    params = resolve_music_edit(
        MusicRepaintRequest(
            task="repaint", model=str(card.model_id), prompt="A warm cello response",
            source_sample_rate=8000, region_end_frame=8000, seed=0,
        ), source, card, resolved_seed=0, total_input_chunks=1,
    )
    job = MusicEditJob(
        id=CommandId(identifier), params=params, created_at=int(time.time()),
    )
    output = MusicEditingOutput(
        source=source,
        result=NativeMusicResult(
            request_id=identifier, model_identity="b" * 64, payload_sha256="c" * 64,
            operation="repaint", seed=0, source_sha256=source.sha256,
            conditioning_sha256="d" * 64, conditioning_gain=1.0,
            execution=NativeMusicExecution(
                device="cpu", precision="float32", text_encoder_device="cpu",
                text_encoder_precision="float32", autoencoder_device="cpu",
                autoencoder_precision="float32",
            ),
            output=NativeMusicWav(
                sha256=hashlib.sha256(result_bytes).hexdigest(),
                size_bytes=len(result_bytes), frames=48000,
                peak=abs(value) / 2**23, gain=1.0,
            ),
            generation_seconds=1.0,
        ),
    )
    return job, output, source_bytes, result_bytes
