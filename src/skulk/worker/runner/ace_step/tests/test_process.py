"""Independent parent-side native WAV, lineage and custody verification."""

import hashlib
import os
import struct
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import JsonValue, TypeAdapter

from skulk.shared.tests.music_edit_fixtures import music_edit_fixture
from skulk.shared.types.audio_separation import MAX_SEPARATION_SOURCE_BYTES
from skulk.shared.types.common import CommandId
from skulk.shared.types.native_music import (
    NativeMusicReady,
    NativeMusicRequest,
    NativeMusicResult,
    NativeMusicServeConfig,
)
from skulk.utils.runtime_files import write_private
from skulk.worker.runner.ace_step.process import (
    expected_music_execution,
    verify_music_result,
)
from skulk.worker.runner.ace_step.result_custody import (
    music_result_directory,
    prepare_result_directory,
)
from skulk.worker.runner.ace_step.tests.test_result_custody import write
from skulk.worker.runner.ace_step.tests.test_workspace import weights
from skulk.worker.runner.ace_step.workspace import NativeMusicWorkspace

_JSON = TypeAdapter[dict[str, JsonValue]](dict[str, JsonValue])


@dataclass(frozen=True)
class ResultFixture:
    """Synthetic complete file facts; this never loads an inference model."""

    directory: Path
    request: NativeMusicRequest
    ready: NativeMusicReady
    result: NativeMusicResult
    configuration: NativeMusicServeConfig


def float_wav() -> bytes:
    """Encode one second of finite zero float conditioning without native libraries."""
    samples = bytes(48000 * 2 * 4)
    payload = (
        b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 3, 2, 48000, 48000 * 8, 8, 32)
        + b"data"
        + struct.pack("<I", len(samples))
        + samples
    )
    return b"RIFF" + struct.pack("<I", len(payload)) + payload


@contextmanager
def fixture(root: Path, *, generate: bool = False) -> Iterator[ResultFixture]:
    """Publish private fixture files only after real pre-write result ownership."""
    owner = NativeMusicWorkspace(root, weights())
    owner.open()
    try:
        configuration = NativeMusicServeConfig(
            model_identity="b" * 64,
            weights=weights(),
            directory=str(owner.directory),
            directory_identity=owner.directory_identity,
            requests_identity=owner.requests_identity,
            loader_identity=owner.loader_identity,
            device="cpu",
            precision="float32",
        )
        _, output, original, data = music_edit_fixture("take")
        directory = music_result_directory(root, CommandId("take"))
        identity = prepare_result_directory(directory, owner.requests_identity)
        write(directory / "result.wav", data)
        source = root / "original.wav"
        write(source, original)
        conditioning = float_wav()
        if not generate:
            write(directory / "conditioning.wav", conditioning)
        request = NativeMusicRequest(
            request_id="take",
            operation="generate" if generate else "repaint",
            prompt="Warm folk",
            seed=0,
            target_frames=48000,
            source=None if generate else output.source,
            source_path=None if generate else str(source),
            region_end_frame=None if generate else 48000,
            output_directory=str(directory),
            output_directory_identity=identity,
        )
        result = output.result.model_copy(
            update={
                "operation": request.operation,
                "source_sha256": None if generate else output.source.sha256,
                "conditioning_sha256": None
                if generate
                else hashlib.sha256(conditioning).hexdigest(),
                "conditioning_gain": None if generate else 1.0,
            }
        )
        result = NativeMusicResult.model_validate_json(result.model_dump_json())
        write_private(directory / "result.json", result.model_dump_json().encode())
        ready = NativeMusicReady(
            model_identity=configuration.model_identity,
            payload_sha256="c" * 64,
            execution=expected_music_execution(configuration),
            load_seconds=0.1,
        )
        yield ResultFixture(directory, request, ready, result, configuration)
    finally:
        owner.close()


@pytest.mark.parametrize("generate", [True, False])
def test_complete_native_result_is_measured_independently(
    tmp_path: Path, generate: bool
) -> None:
    """Both generation and edits require real complete files before delivery."""
    with fixture(tmp_path, generate=generate) as item:
        measured = verify_music_result(item.directory, item.request, item.ready)
        assert measured == item.result
        assert (item.directory / "result.wav").exists()


@pytest.mark.parametrize(
    "field",
    [
        "request_id",
        "model_identity",
        "payload_sha256",
        "seed",
        "operation",
        "source_sha256",
        "conditioning_sha256",
        "execution",
    ],
)
def test_foreign_result_identity_is_not_delivered(tmp_path: Path, field: str) -> None:
    """Valid-looking manifests cannot replace the admitted request or loaded model."""
    with fixture(tmp_path) as item:
        body = _JSON.validate_json(item.result.model_dump_json())
        if field == "seed":
            body[field] = 5
        elif field == "operation":
            body[field] = "continue"
        elif field == "request_id":
            body[field] = "other-take"
        elif field == "execution":
            execution = body[field]
            assert isinstance(execution, dict)
            execution["autoencoder_device"] = "cuda"
        else:
            body[field] = "e" * 64
        write_private(item.directory / "result.json", _JSON.dump_json(body))
        with pytest.raises(ValueError):
            verify_music_result(item.directory, item.request, item.ready)
        assert (item.directory / "result.wav").exists()


def test_a_matching_digest_does_not_hide_a_false_pcm_peak(tmp_path: Path) -> None:
    """Core scans encoded samples rather than accepting only native byte hashes."""
    with fixture(tmp_path) as item:
        path = item.directory / "result.wav"
        data = bytearray(path.read_bytes())
        data[44:47] = (2000).to_bytes(3, "little", signed=True)
        path.write_bytes(data)
        changed = item.result.model_copy(
            update={
                "output": item.result.output.model_copy(
                    update={"sha256": hashlib.sha256(data).hexdigest()}
                )
            }
        )
        write_private(
            item.directory / "result.json", changed.model_dump_json().encode()
        )
        with pytest.raises(ValueError, match="WAV differs"):
            verify_music_result(item.directory, item.request, item.ready)


@pytest.mark.parametrize("name", ["result.wav", "conditioning.wav", "original.wav"])
def test_changed_audio_is_rejected_without_repair_or_deletion(
    tmp_path: Path, name: str
) -> None:
    """Original, conditioning and output bytes each have independent lineage."""
    with fixture(tmp_path) as item:
        path = tmp_path / name if name == "original.wav" else item.directory / name
        data = bytearray(path.read_bytes())
        data[-1] ^= 1
        path.write_bytes(data)
        with pytest.raises(ValueError):
            verify_music_result(item.directory, item.request, item.ready)
        assert path.read_bytes() == data


def test_nonfinite_conditioning_fails_even_with_matching_manifest_hash(
    tmp_path: Path,
) -> None:
    """A corrupted FLOAT file cannot pass by updating its checksum alone."""
    with fixture(tmp_path) as item:
        path = item.directory / "conditioning.wav"
        data = bytearray(path.read_bytes())
        struct.pack_into("<f", data, 44, float("nan"))
        path.write_bytes(data)
        changed = item.result.model_copy(
            update={"conditioning_sha256": hashlib.sha256(data).hexdigest()}
        )
        write_private(
            item.directory / "result.json", changed.model_dump_json().encode()
        )
        with pytest.raises(ValueError, match="finite"):
            verify_music_result(item.directory, item.request, item.ready)


@pytest.mark.parametrize("name", ["result.json", "result.wav", "conditioning.wav"])
@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo", "permissive"])
def test_foreign_file_custody_never_enters_result_transport(
    tmp_path: Path, name: str, kind: str
) -> None:
    """No blocking pipe, alias or permissive file can be accepted as owned output."""
    with fixture(tmp_path) as item:
        path = item.directory / name
        data = path.read_bytes()
        original = tmp_path / "foreign-file"
        write(original, data)
        if kind == "permissive":
            path.chmod(0o644)
        else:
            path.unlink()
            if kind == "symlink":
                path.symlink_to(original)
            elif kind == "hardlink":
                os.link(original, path)
            else:
                os.mkfifo(path, 0o600)
        with pytest.raises((ValueError, OSError)):
            verify_music_result(item.directory, item.request, item.ready)
        assert original.read_bytes() == data


@pytest.mark.parametrize("name", ["result.json", "result.wav", "conditioning.wav"])
def test_oversized_result_is_failed_not_truncated(tmp_path: Path, name: str) -> None:
    """The private manifest has a separate bound from its encoded WAV files."""
    with fixture(tmp_path) as item:
        path = item.directory / name
        limit = 16 * 1024 if name == "result.json" else MAX_SEPARATION_SOURCE_BYTES
        with path.open("r+b") as stream:
            stream.truncate(limit + 1)
        with pytest.raises(ValueError, match="bounded"):
            verify_music_result(item.directory, item.request, item.ready)
        assert path.stat().st_size == limit + 1


def test_changed_output_directory_is_not_retroactively_adopted(tmp_path: Path) -> None:
    """A valid result receipt from an old inode never authorizes a replacement."""
    with fixture(tmp_path) as item:
        retained = item.directory.with_name("retained-take")
        item.directory.rename(retained)
        item.directory.mkdir(mode=0o700)
        foreign = item.directory / "operator-note"
        write(foreign, b"must survive")
        with pytest.raises(ValueError, match="owned inode"):
            verify_music_result(item.directory, item.request, item.ready)
        assert foreign.read_bytes() == b"must survive"
        assert (retained / "result.wav").exists()


def test_cancel_during_independent_verification_retains_all_files(
    tmp_path: Path,
) -> None:
    """Only the separate drained owner can clean a cancelled candidate."""
    with fixture(tmp_path) as item:
        calls = 0

        def checkpoint() -> None:
            nonlocal calls
            calls += 1
            if calls == 6:
                raise InterruptedError("caller cancelled verification")

        with pytest.raises(InterruptedError):
            verify_music_result(
                item.directory, item.request, item.ready, checkpoint=checkpoint
            )
        assert {path.name for path in item.directory.iterdir()} == {
            "result.json",
            "result.wav",
            "conditioning.wav",
        }


def test_metal_autoencoder_precision_is_distinct_from_transformer(
    tmp_path: Path,
) -> None:
    """Metal readiness must match the actual mixed component policy."""
    with fixture(tmp_path) as item:
        configuration = item.configuration.model_copy(update={"device": "mps"})
        execution = expected_music_execution(configuration)
        assert (execution.device, execution.precision) == ("mps", "float32")
        assert (execution.text_encoder_device, execution.text_encoder_precision) == (
            "mps",
            "float32",
        )
        assert (execution.autoencoder_device, execution.autoencoder_precision) == (
            "mps",
            "float16",
        )
