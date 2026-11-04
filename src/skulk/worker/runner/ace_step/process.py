"""Independently verify native music custody before any addressed result delivery."""

from __future__ import annotations

import os
import stat
from collections.abc import Callable
from pathlib import Path

from skulk.shared.types.audio_separation import (
    MAX_SEPARATION_SOURCE_BYTES,
    pcm24_from_wav,
    source_from_wav,
)
from skulk.shared.types.native_music import (
    NativeMusicDirectoryIdentity,
    NativeMusicExecution,
    NativeMusicReady,
    NativeMusicRequest,
    NativeMusicResult,
    NativeMusicServeConfig,
    native_music_conditioning_frames,
)


def expected_music_execution(
    configuration: NativeMusicServeConfig,
) -> NativeMusicExecution:
    """Resolve the admitted component policy, independently of a child's claims.

    Metal's VAE uses FP16 while its transformer/text encoder remain FP32. Other
    lanes use their explicit selected precision; no backend auto-selection or
    silent CPU fallback is accepted as readiness.
    """
    return NativeMusicExecution(
        device=configuration.device,
        precision=configuration.precision,
        text_encoder_device=configuration.device,
        text_encoder_precision=configuration.precision,
        autoencoder_device=configuration.device,
        autoencoder_precision="float16"
        if configuration.device == "mps"
        else configuration.precision,
    )


def _read_private(path: Path, limit: int, checkpoint: Callable[[], None]) -> bytes:
    checkpoint()
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as source:
        before = os.fstat(source.fileno())
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.getuid()
            or before.st_mode & 0o077
            or before.st_nlink != 1
            or before.st_size > limit
        ):
            raise ValueError("native music result is not bounded private ordinary data")
        blocks = bytearray()
        while block := source.read(min(1024**2, limit + 1 - len(blocks))):
            blocks.extend(block)
            checkpoint()
            if len(blocks) > limit:
                raise ValueError("native music result exceeds its complete-file bound")
        after = os.fstat(source.fileno())
        if (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
            after.st_nlink,
        ) != (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
            before.st_nlink,
        ) or len(blocks) != before.st_size:
            raise ValueError("native music result changed during verification")
    checkpoint()
    return bytes(blocks)


def verify_music_result(
    directory: Path,
    request: NativeMusicRequest,
    ready: NativeMusicReady,
    *,
    checkpoint: Callable[[], None] = lambda: None,
) -> NativeMusicResult:
    """Bind actual WAV bytes, original lineage and manifest to one admitted call.

    The caller owns the serial model actor and keeps native/output-reader fences.
    This performs no resampling or audio repair. It measures WAV geometry, digest
    and PCM peak independently of the child; native gain remains reported
    transformation provenance rather than a fact inferable from the final WAV.
    Invalid or over-limit files are never truncated, delivered or removed here.
    """
    if directory != Path(request.output_directory):
        raise ValueError("native music result does not use its admitted output path")
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)

    def verify_directory() -> None:
        info = os.fstat(descriptor)
        observed = NativeMusicDirectoryIdentity(device=info.st_dev, inode=info.st_ino)
        if (
            observed != request.output_directory_identity
            or info.st_uid != os.getuid()
            or stat.S_IMODE(info.st_mode) != 0o700
            or directory.lstat().st_ino != info.st_ino
            or directory.lstat().st_dev != info.st_dev
        ):
            raise ValueError("native music result directory changed its owned inode")

    try:
        verify_directory()
        expected = {"result.json", "result.wav"}
        if request.source is not None:
            expected.add("conditioning.wav")
        if set(os.listdir(descriptor)) != expected:
            raise ValueError(
                "native music result directory has an incomplete or foreign inventory"
            )
        result = NativeMusicResult.model_validate_json(
            _read_private(directory / "result.json", 16 * 1024, checkpoint)
        )
        if (
            result.request_id,
            result.model_identity,
            result.payload_sha256,
            result.operation,
            result.seed,
            result.source_sha256,
            result.execution,
        ) != (
            request.request_id,
            ready.model_identity,
            ready.payload_sha256,
            request.operation,
            request.seed,
            request.source.sha256 if request.source is not None else None,
            ready.execution,
        ):
            raise ValueError(
                "native music completion differs from its admitted request or model"
            )
        data = _read_private(
            directory / "result.wav", MAX_SEPARATION_SOURCE_BYTES, checkpoint
        )
        facts, peak = pcm24_from_wav(data)
        measured = (
            facts.sha256,
            facts.size_bytes,
            facts.frames,
            facts.sample_rate,
            facts.channels,
            facts.bit_depth,
            facts.encoding,
            peak,
        )
        output = result.output
        if measured != (
            output.sha256,
            output.size_bytes,
            output.frames,
            output.sample_rate,
            output.channels,
            output.bit_depth,
            "pcm",
            output.peak,
        ):
            raise ValueError("native music WAV differs from its completed manifest")
        del data
        checkpoint()
        if request.source is not None:
            if request.source_path is None or output.frames != request.target_frames:
                raise ValueError(
                    "native music edit changed its original timeline geometry"
                )
            source = source_from_wav(
                _read_private(
                    Path(request.source_path), MAX_SEPARATION_SOURCE_BYTES, checkpoint
                )
            )
            if source != request.source:
                raise ValueError(
                    "native music original no longer matches its admitted source"
                )
            conditioning = source_from_wav(
                _read_private(
                    directory / "conditioning.wav",
                    MAX_SEPARATION_SOURCE_BYTES,
                    checkpoint,
                )
            )
            if (
                conditioning.sha256,
                conditioning.frames,
                conditioning.sample_rate,
                conditioning.channels,
                conditioning.encoding,
                conditioning.bit_depth,
            ) != (
                result.conditioning_sha256,
                native_music_conditioning_frames(request.target_frames),
                48000,
                2,
                "float",
                32,
            ):
                raise ValueError(
                    "native music conditioning differs from its complete aligned source"
                )
        checkpoint()
        verify_directory()
        return result
    finally:
        os.close(descriptor)
