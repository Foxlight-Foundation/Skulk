"""Prepare ModelOpt NVFP4 artifacts as bounded, cached BF16 native weights.

The serialized nibble order and block/global scales follow NVIDIA ModelOpt.
This is weight expansion for the existing native engine, not an FP4 GPU kernel.
Only independently pinned and verified component files reach this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import struct
import tempfile
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Literal, cast

import numpy as np
from filelock import FileLock
from numpy.typing import NDArray
from pydantic import Field, field_validator, model_validator

from skulk.shared.models.flux3_tensor_names import native_flux3_tensor_name
from skulk.utils.pydantic_ext import FrozenModel

_HEADER_LIMIT = 16 * 1024**2
_CHUNK_BYTES = 1024**2
_OUTPUT_LIMIT = 32 * 1024**3
# Numeric rules and output precision are part of the derived cache identity.
_RECIPE = "modelopt-nvfp4-bf16-v1"
_UINT16_LE: np.dtype[np.uint16] = np.dtype(np.uint16).newbyteorder("<")
_ELEMENT_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1, "U8": 1}
_FP4: NDArray[np.float32] = np.array(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32
)
_FP8: NDArray[np.float32] = np.array(
    [
        math.nan
        if value & 127 == 127
        else (-1 if value & 128 else 1)
        * (
            math.ldexp(value & 7, -9)
            if (value >> 3) & 15 == 0
            else math.ldexp(1 + (value & 7) / 8, ((value >> 3) & 15) - 7)
        )
        for value in range(256)
    ],
    dtype=np.float32,
)


class _Tensor(FrozenModel):
    """One bounded safetensors entry, before any bytes are allocated."""

    dtype: Literal["BF16", "F16", "F32", "F8_E4M3", "U8"]
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]

    @field_validator("shape", "data_offsets", mode="before")
    @classmethod
    def wire_arrays(cls, value: object) -> object:
        """Restore JSON arrays while retaining strict integer validation."""
        return tuple(cast("list[object]", value)) if isinstance(value, list) else value

    @model_validator(mode="after")
    def bounded(self) -> _Tensor:
        """Reject negative, oversized or inconsistent tensor geometry."""
        if len(self.shape) > 4 or any(size <= 0 for size in self.shape):
            raise ValueError("packed weight tensor has invalid dimensions")
        if (
            self.elements > _OUTPUT_LIMIT
            or not 0 <= self.data_offsets[0] <= self.data_offsets[1]
        ):
            raise ValueError("packed weight tensor exceeds its bounds")
        if (
            self.data_offsets[1] - self.data_offsets[0]
            != self.elements * _ELEMENT_BYTES[self.dtype]
        ):
            raise ValueError("packed weight tensor size differs from its geometry")
        return self

    @property
    def elements(self) -> int:
        """Return scalar count, including one element for a scalar."""
        return math.prod(self.shape)


class _Receipt(FrozenModel):
    """Bind cached output bytes to the immutable source and expansion recipe."""

    recipe: str
    source_identity: str
    parameter_count: int
    output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_bytes: int


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read_header(source: Path) -> tuple[int, dict[str, _Tensor]]:
    with source.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("packed weights have no safetensors length prefix")
        length = cast("int", struct.unpack("<Q", prefix)[0])
        if not 0 < length <= _HEADER_LIMIT:
            raise ValueError("packed weights exceed the header limit")
        document = cast("object", json.loads(stream.read(length)))
    if (
        not isinstance(document, dict)
        or len(cast("dict[str, object]", document)) > 16384
    ):
        raise ValueError("packed weights have no bounded tensor map")
    tensors = {
        name: _Tensor.model_validate(value)
        for name, value in cast("dict[str, object]", document).items()
        if name != "__metadata__"
    }
    previous = 0
    for tensor in sorted(tensors.values(), key=lambda item: item.data_offsets):
        if tensor.data_offsets[0] != previous:
            raise ValueError("packed weight tensor data overlap or have a gap")
        previous = tensor.data_offsets[1]
    if length + 8 + previous != source.stat().st_size:
        raise ValueError("packed weight data differ from the declared file size")
    return length + 8, tensors


def _modules(tensors: dict[str, _Tensor]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, weight in tensors.items():
        if weight.dtype != "U8" or not name.endswith(".weight"):
            continue
        module = name.removesuffix(".weight")
        scale = tensors.get(module + ".weight_scale")
        global_scale = tensors.get(module + ".weight_scale_2")
        input_scale = tensors.get(module + ".input_scale")
        if (
            len(weight.shape) != 2
            or weight.shape[1] % 8
            or scale is None
            or scale.dtype != "F8_E4M3"
            or scale.shape != (weight.shape[0], weight.shape[1] // 8)
            or global_scale is None
            or global_scale.dtype != "F32"
            or global_scale.shape
            or input_scale is None
            or input_scale.dtype != "F32"
            or input_scale.shape
        ):
            raise ValueError(
                "NVFP4 weight lacks its exact block-16 and scalar scale geometry"
            )
        result[name] = module
    if not result:
        raise ValueError("NVFP4 artifact contains no packed weights")
    return result


def _read_exact(stream: BinaryIO, offset: int, size: int) -> bytes:
    stream.seek(offset)
    data = stream.read(size)
    if len(data) != size:
        raise ValueError("packed tensor data are truncated")
    return data


def _expand(source: Path, destination: Path, parameter_count: int) -> None:
    start, tensors = _read_header(source)
    modules = _modules(tensors)
    auxiliary = {
        module + suffix
        for module in modules.values()
        for suffix in (".weight_scale", ".weight_scale_2", ".input_scale")
    }
    selected = {name: value for name, value in tensors.items() if name not in auxiliary}
    header: dict[str, object] = {}
    offset = 0
    logical_count = 0
    for name, tensor in selected.items():
        packed = name in modules
        if not packed and tensor.dtype not in {"BF16", "F16", "F32"}:
            raise ValueError("NVFP4 artifact has unrecognized residual tensor packing")
        shape = (*tensor.shape[:-1], tensor.shape[-1] * 2) if packed else tensor.shape
        count = math.prod(shape)
        logical_count += count
        size = count * (2 if packed else _ELEMENT_BYTES[tensor.dtype])
        header[name] = {
            "dtype": "BF16" if packed else tensor.dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + size],
        }
        offset += size
    if logical_count != parameter_count or not 0 < offset <= _OUTPUT_LIMIT:
        raise ValueError("NVFP4 logical weight count differs from the model contract")
    if shutil.disk_usage(destination.parent).free < offset + 1024**3:
        raise ValueError(
            "insufficient disk space for the declared BF16 weight expansion"
        )
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    if len(encoded) > _HEADER_LIMIT:
        raise ValueError("expanded weight header exceeds its bound")
    with source.open("rb") as incoming, destination.open("wb") as outgoing:
        outgoing.write(struct.pack("<Q", len(encoded)))
        outgoing.write(encoded)
        for name, tensor in selected.items():
            if name not in modules:
                length = tensor.data_offsets[1] - tensor.data_offsets[0]
                for cursor in range(0, length, _CHUNK_BYTES):
                    outgoing.write(
                        _read_exact(
                            incoming,
                            start + tensor.data_offsets[0] + cursor,
                            min(_CHUNK_BYTES, length - cursor),
                        )
                    )
                continue
            module = modules[name]
            scale = tensors[module + ".weight_scale"]
            second = tensors[module + ".weight_scale_2"]
            global_scale = cast(
                "float",
                struct.unpack(
                    "<f", _read_exact(incoming, start + second.data_offsets[0], 4)
                )[0],
            )
            if not math.isfinite(global_scale) or global_scale <= 0:
                raise ValueError(
                    "NVFP4 global weight scale must be finite and positive"
                )
            length = tensor.elements
            for cursor in range(0, length, _CHUNK_BYTES):
                count = min(_CHUNK_BYTES, length - cursor)
                packed = np.frombuffer(
                    _read_exact(
                        incoming, start + tensor.data_offsets[0] + cursor, count
                    ),
                    dtype=np.uint8,
                )
                scales = _FP8[
                    np.frombuffer(
                        _read_exact(
                            incoming,
                            start + scale.data_offsets[0] + cursor // 8,
                            count // 8,
                        ),
                        dtype=np.uint8,
                    )
                ]
                if not bool(np.all(np.isfinite(scales) & (scales >= 0))):
                    raise ValueError(
                        "NVFP4 block scales must be finite and nonnegative"
                    )
                codes = np.empty(count * 2, dtype=np.uint8)
                codes[0::2] = packed & 15
                codes[1::2] = packed >> 4
                # ModelOpt packs the first value in the low nibble; one E4M3
                # block scale covers 16 values and the FP32 scale covers a tensor.
                values = _FP4[codes] * np.repeat(scales, 16) * np.float32(global_scale)
                if not bool(
                    np.all(
                        np.isfinite(values)
                        & (np.abs(values) <= np.float32(3.3895313892515355e38))
                    )
                ):
                    raise ValueError("NVFP4 expansion must contain finite BF16 weights")
                # The reference executes in BF16. FP16 preparation produced
                # blank native FLUX.3 renders despite finite stored weights.
                # Round FP32 to BF16 using round-to-nearest, ties-to-even.
                bits = values.view(np.uint32)
                words = ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16).astype(
                    "<u2"
                )
                outgoing.write(words.tobytes())
        outgoing.flush()
        os.fsync(outgoing.fileno())


def _prepare_cached(
    *,
    source_identity: str,
    parameter_count: int,
    cache_root: Path,
    recipe: str,
    expand: Callable[[Path], None],
) -> Path:
    if parameter_count <= 0:
        raise ValueError("derived native weights require positive logical geometry")
    key = hashlib.sha256(
        f"{recipe}:{source_identity}:{parameter_count}".encode()
    ).hexdigest()
    root = cache_root / "derived-weights" / key
    root.mkdir(parents=True, exist_ok=True)
    output, manifest = root / "model.safetensors", root / "receipt.json"
    with FileLock(str(root / "prepare.lock")):
        if output.is_file() and manifest.is_file():
            receipt = _Receipt.model_validate_json(manifest.read_text())
            if (
                receipt.recipe == recipe
                and receipt.source_identity == source_identity
                and receipt.parameter_count == parameter_count
                and receipt.output_bytes == output.stat().st_size
                and receipt.output_sha256 == _digest(output)
            ):
                return output
            raise ValueError(
                "derived native weight cache differs from its verified identity"
            )
        for partial in root.glob("*.partial"):
            partial.unlink()
        with tempfile.NamedTemporaryFile(
            dir=root, suffix=".partial", delete=False
        ) as temporary:
            path = Path(temporary.name)
        try:
            expand(path)
            receipt = _Receipt(
                recipe=recipe,
                source_identity=source_identity,
                parameter_count=parameter_count,
                output_sha256=_digest(path),
                output_bytes=path.stat().st_size,
            )
            path.replace(output)
            temporary_manifest = root / "receipt.partial"
            temporary_manifest.write_text(receipt.model_dump_json())
            temporary_manifest.replace(manifest)
        finally:
            path.unlink(missing_ok=True)
    return output


def prepare_nvfp4_weights(
    source: Path,
    *,
    source_identity: str,
    parameter_count: int,
    cache_root: Path,
) -> Path:
    """Expand verified ModelOpt weights once; return a hash-checked BF16 cache.

    Inputs must already pass the component's complete immutable size/hash check.
    CPU arrays and reads are chunk-bounded. The source is untouched. Per-artifact
    locking and atomic receipts reconcile interrupted or concurrent preparation.
    """

    def expand(destination: Path) -> None:
        _expand(source, destination, parameter_count)

    return _prepare_cached(
        source_identity=source_identity,
        parameter_count=parameter_count,
        cache_root=cache_root,
        recipe=_RECIPE,
        expand=expand,
    )


def _repack_diffusers(index: Path, destination: Path, parameter_count: int) -> None:
    if index.stat().st_size > _HEADER_LIMIT:
        raise ValueError("FLUX.3 Diffusers index exceeds its bound")
    raw = cast("object", json.loads(index.read_bytes()))
    if not isinstance(raw, dict):
        raise ValueError("FLUX.3 Diffusers has no weight index")
    weights = cast("dict[str, object]", raw).get("weight_map")
    if (
        not isinstance(weights, dict)
        or not 0 < len(cast("dict[str, object]", weights)) <= 16384
    ):
        raise ValueError("FLUX.3 Diffusers has no bounded weight map")
    weight_map = cast("dict[str, object]", weights)
    shards: set[str] = set()
    for shard in weight_map.values():
        if (
            not isinstance(shard, str)
            or not shard.endswith(".safetensors")
            or PurePosixPath(shard).is_absolute()
            or len(PurePosixPath(shard).parts) != 1
        ):
            raise ValueError("FLUX.3 Diffusers index has an unsafe shard path")
        shards.add(shard)
    if len(shards) > 16:
        raise ValueError("FLUX.3 Diffusers has too many shards")
    sources: dict[str, tuple[Path, int, _Tensor]] = {}
    for shard in sorted(shards):
        path = index.parent / shard
        start, tensors = _read_header(path)
        for name, tensor in tensors.items():
            if (
                name in sources
                or weight_map.get(name) != shard
                or tensor.dtype not in {"BF16", "F16", "F32"}
            ):
                raise ValueError(
                    "FLUX.3 Diffusers index disagrees with dense shard tensors"
                )
            sources[name] = path, start, tensor
    if set(sources) != set(weight_map):
        raise ValueError("FLUX.3 Diffusers index is incomplete")
    header: dict[str, object] = {}
    total = 0
    count = 0
    for name, (_, _, tensor) in sources.items():
        native = native_flux3_tensor_name(name)
        if native in header:
            raise ValueError("FLUX.3 Diffusers mapping has duplicate native tensors")
        size = tensor.data_offsets[1] - tensor.data_offsets[0]
        header[native] = {
            "dtype": tensor.dtype,
            "shape": list(tensor.shape),
            "data_offsets": [total, total + size],
        }
        total += size
        count += tensor.elements
    if count != parameter_count or not 0 < total <= _OUTPUT_LIMIT:
        raise ValueError("FLUX.3 Diffusers logical geometry differs from the card")
    if shutil.disk_usage(destination.parent).free < total + 1024**3:
        raise ValueError("insufficient disk space for native Diffusers preparation")
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    if len(encoded) > _HEADER_LIMIT:
        raise ValueError("FLUX.3 native header exceeds its bound")
    with destination.open("wb") as outgoing:
        outgoing.write(struct.pack("<Q", len(encoded)))
        outgoing.write(encoded)
        for path, start, tensor in sources.values():
            length = tensor.data_offsets[1] - tensor.data_offsets[0]
            with path.open("rb") as incoming:
                for cursor in range(0, length, _CHUNK_BYTES):
                    outgoing.write(
                        _read_exact(
                            incoming,
                            start + tensor.data_offsets[0] + cursor,
                            min(_CHUNK_BYTES, length - cursor),
                        )
                    )
        outgoing.flush()
        os.fsync(outgoing.fileno())


def prepare_flux3_diffusers(
    index: Path,
    *,
    source_identity: str,
    parameter_count: int,
    cache_root: Path,
) -> Path:
    """Repack verified FLUX.3 shards with native names, preserving dense bytes.

    The caller verifies every immutable component file before this bounded local
    preparation. The cache identity binds all files, the exact name-mapping
    recipe and geometry; generated output is hash-checked on subsequent loads.
    """

    def expand(destination: Path) -> None:
        _repack_diffusers(index, destination, parameter_count)

    return _prepare_cached(
        source_identity=source_identity,
        parameter_count=parameter_count,
        cache_root=cache_root,
        recipe="flux3-diffusers-native-v1",
        expand=expand,
    )
