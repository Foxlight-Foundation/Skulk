"""Independently known NVFP4 values, integrity failures and concurrent cache use."""

import hashlib
import json
import struct
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast

import pytest

from skulk.worker.runner.stable_diffusion_cpp.packed_weights import (
    prepare_nvfp4_weights,
)


def _source(path: Path, *, scale: bytes = b"\x38", global_scale: float = 2.0) -> str:
    tensors: list[tuple[str, str, list[int], bytes]] = [
        ("linear.weight", "U8", [1, 8], bytes.fromhex("1032547698badcfe")),
        ("linear.weight_scale", "F8_E4M3", [1, 1], scale),
        ("linear.weight_scale_2", "F32", [], struct.pack("<f", global_scale)),
        ("linear.input_scale", "F32", [], struct.pack("<f", 1.0)),
        ("norm.weight", "BF16", [1], b"\x80\x3f"),
    ]
    header: dict[str, object] = {}
    offset = 0
    data = b""
    for name, dtype, shape, payload in tensors:
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(payload)],
        }
        offset += len(payload)
        data += payload
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _prepare(path: Path, identity: str, cache: Path) -> Path:
    return prepare_nvfp4_weights(
        path, source_identity=identity, parameter_count=17, cache_root=cache
    )


def test_known_nibble_order_block_and_global_scales(tmp_path: Path) -> None:
    source = tmp_path / "source.safetensors"
    identity = _source(source)
    output = _prepare(source, identity, tmp_path / "cache")
    payload = output.read_bytes()
    length = cast("int", struct.unpack("<Q", payload[:8])[0])
    header = cast("dict[str, object]", json.loads(payload[8 : 8 + length]))
    assert header == {
        "linear.weight": {"dtype": "BF16", "shape": [1, 16], "data_offsets": [0, 32]},
        "norm.weight": {"dtype": "BF16", "shape": [1], "data_offsets": [32, 34]},
    }
    words = cast(
        "tuple[int, ...]", struct.unpack("<16H", payload[8 + length : 8 + length + 32])
    )
    assert struct.unpack(
        "<16f", struct.pack("<16I", *(word << 16 for word in words))
    ) == (
        0,
        1,
        2,
        3,
        4,
        6,
        8,
        12,
        0,
        -1,
        -2,
        -3,
        -4,
        -6,
        -8,
        -12,
    )
    assert payload[-2:] == b"\x80\x3f"
    assert source.read_bytes().startswith(
        struct.pack("<Q", struct.unpack("<Q", source.read_bytes()[:8])[0])
    )
    before = output.stat().st_mtime_ns
    assert _prepare(source, identity, tmp_path / "cache") == output
    assert output.stat().st_mtime_ns == before
    output.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="cache differs"):
        _prepare(source, identity, tmp_path / "cache")


@pytest.mark.parametrize(
    "scale,global_scale",
    [(b"\x7f", 1.0), (b"\xb8", 1.0), (b"\x38", float("nan")), (b"\x38", 0.0)],
)
def test_invalid_scales_leave_no_ready_output(
    tmp_path: Path, scale: bytes, global_scale: float
) -> None:
    source = tmp_path / "source.safetensors"
    identity = _source(source, scale=scale, global_scale=global_scale)
    with pytest.raises(ValueError, match="scale"):
        _prepare(source, identity, tmp_path / "cache")
    assert not list((tmp_path / "cache").rglob("*.partial"))
    assert not list((tmp_path / "cache").rglob("receipt.json"))


def test_truncated_data_and_wrong_logical_geometry_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "source.safetensors"
    identity = _source(source)
    with pytest.raises(ValueError, match="logical weight count"):
        prepare_nvfp4_weights(
            source,
            source_identity=identity,
            parameter_count=16,
            cache_root=tmp_path / "cache",
        )
    source.write_bytes(source.read_bytes()[:-1])
    with pytest.raises(ValueError, match="file size"):
        _prepare(source, identity, tmp_path / "cache")


def test_concurrent_preparation_converges_on_one_verified_output(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.safetensors"
    identity = _source(source)

    def prepare(_index: int) -> Path:
        return _prepare(source, identity, tmp_path / "cache")

    with ThreadPoolExecutor(max_workers=2) as pool:
        outputs = list(pool.map(prepare, range(2)))
    assert outputs[0] == outputs[1]
    assert len(list((tmp_path / "cache").rglob("receipt.json"))) == 1


def test_diffusers_index_mapping_preserves_bytes_and_rejects_missing_shard(
    tmp_path: Path,
) -> None:
    """An indexed shard is translated by header names without changing BF16 values."""
    from skulk.worker.runner.stable_diffusion_cpp.packed_weights import (
        prepare_flux3_diffusers,
    )

    shard = tmp_path / "weights.safetensors"
    header = json.dumps(
        {
            "context_embedder.weight": {
                "dtype": "BF16",
                "shape": [1, 1],
                "data_offsets": [0, 2],
            }
        }
    ).encode()
    shard.write_bytes(struct.pack("<Q", len(header)) + header + b"\x80\x3f")
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(
        json.dumps({"weight_map": {"context_embedder.weight": shard.name}})
    )
    output = prepare_flux3_diffusers(
        index,
        source_identity="verified-bundle",
        parameter_count=1,
        cache_root=tmp_path / "cache",
    )
    data = output.read_bytes()
    length = cast("int", struct.unpack("<Q", data[:8])[0])
    assert json.loads(data[8 : 8 + length]) == {
        "txt_in.weight": {"dtype": "BF16", "shape": [1, 1], "data_offsets": [0, 2]}
    }
    assert data[8 + length :] == b"\x80\x3f"
    index.write_text(json.dumps({"weight_map": {"missing.weight": shard.name}}))
    with pytest.raises(ValueError, match="disagrees"):
        prepare_flux3_diffusers(
            index,
            source_identity="different-bundle",
            parameter_count=1,
            cache_root=tmp_path / "cache",
        )


def test_nvfp4_preserves_bf16_range_beyond_fp16(tmp_path: Path) -> None:
    """Restoring original BF16 precision avoids the finite-FP16 blank-render failure."""
    source = tmp_path / "large-range.safetensors"
    identity = _source(source, global_scale=20_000.0)
    output = _prepare(source, identity, tmp_path / "cache")
    payload = output.read_bytes()
    length = cast("int", struct.unpack("<Q", payload[:8])[0])
    words = cast(
        "tuple[int, ...]", struct.unpack("<16H", payload[8 + length : 8 + length + 32])
    )
    values = struct.unpack("<16f", struct.pack("<16I", *(word << 16 for word in words)))
    assert values[7] == 119808.0
    assert values[15] == -119808.0
