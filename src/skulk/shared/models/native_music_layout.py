"""
Bounded intrinsic tensor geometry for the three native music components.

These facts describe immutable serialized objects, not loaded memory or hardware
qualification. Runtime precision, loading transients, buffers and workload remain
the engine's responsibility. Header projection does not checksum tensor bodies;
the caller binds the prefix to the selected object and verifies complete bytes.
"""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal, cast, final

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    field_validator,
    model_validator,
)

NativeTensorDtype = Literal[
    "BOOL",
    "F4",
    "F6_E2M3",
    "F6_E3M2",
    "U8",
    "I8",
    "F8_E5M2",
    "F8_E4M3",
    "F8_E8M0",
    "I16",
    "U16",
    "F16",
    "BF16",
    "I32",
    "U32",
    "F32",
    "C64",
    "F64",
    "I64",
    "U64",
]
"""Serialized scalar codes defined by the pinned Safetensors 0.7 format."""
_DTYPE_BITS: dict[NativeTensorDtype, int] = {
    "BOOL": 8,
    "F4": 4,
    "F6_E2M3": 6,
    "F6_E3M2": 6,
    "U8": 8,
    "I8": 8,
    "F8_E5M2": 8,
    "F8_E4M3": 8,
    "F8_E8M0": 8,
    "I16": 16,
    "U16": 16,
    "F16": 16,
    "BF16": 16,
    "I32": 32,
    "U32": 32,
    "F32": 32,
    "C64": 64,
    "F64": 64,
    "I64": 64,
    "U64": 64,
}
MAX_NATIVE_MUSIC_HEADER_BYTES = 8 * 1024**2
_MIN_PREFIX_BYTES = 10
_MAX_FILE_BYTES = 8 * 1024**3
_MAX_NAME_CHARACTERS = 4096
_MAX_TENSORS = 1_000_000
_MAX_ELEMENTS = 2**63 - 1
_JSON = TypeAdapter[dict[str, JsonValue]](dict[str, JsonValue])


class _LayoutContract(BaseModel):
    """Strict finite metadata, independent of registry services or native libraries."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")


@final
class NativeMusicTensorDtype(_LayoutContract):
    """Complete scalar and tensor counts for one serialized representation."""

    dtype: NativeTensorDtype = Field(
        description="Observed Safetensors scalar code, not requested runtime precision."
    )
    tensor_count: int = Field(
        gt=0,
        le=_MAX_TENSORS,
        description="Number of tensors using this serialized scalar representation.",
    )
    element_count: int = Field(
        ge=0,
        le=_MAX_ELEMENTS,
        description=(
            "Sum of logical scalar elements, including scalar and empty tensors."
        ),
    )

    @model_validator(mode="after")
    def validate_alignment(self) -> NativeMusicTensorDtype:
        """Require byte-aligned complete storage, including sub-byte scalar codes."""
        if self.element_count * _DTYPE_BITS[self.dtype] % 8:
            raise ValueError("native music scalar inventory is not byte-aligned")
        return self

    @property
    def stored_bytes(self) -> int:
        """Return exact serialized tensor bytes, excluding the separate file header."""
        return self.element_count * _DTYPE_BITS[self.dtype] // 8


@final
class NativeMusicTensorLayout(_LayoutContract):
    """Complete dense Safetensors header projection for one role's selected object."""

    header_bytes: int = Field(
        ge=_MIN_PREFIX_BYTES,
        le=MAX_NATIVE_MUSIC_HEADER_BYTES,
        description=(
            "Exact prefix length, including the eight-byte little-endian length word."
        ),
    )
    header_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description=(
            "SHA-256 of the complete prefix, including length word and JSON padding."
        ),
    )
    dtypes: tuple[NativeMusicTensorDtype, ...] = Field(
        min_length=1,
        max_length=20,
        description=(
            "Complete unique scalar-type groups, independent of runtime dtype policy."
        ),
    )

    @field_validator("dtypes", mode="before")
    @classmethod
    def coerce_dtypes(cls, value: object) -> object:
        """Accept JSON/TOML arrays while retaining an immutable runtime tuple."""
        return tuple(cast("list[object]", value)) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_groups(self) -> NativeMusicTensorLayout:
        """Refuse repeated types or oversized inventories that can understate bytes."""
        if len({group.dtype for group in self.dtypes}) != len(self.dtypes):
            raise ValueError("native music tensor layout repeats a scalar type")
        if sum(group.tensor_count for group in self.dtypes) > _MAX_TENSORS:
            raise ValueError("native music tensor layout exceeds its tensor bound")
        return self

    @property
    def stored_tensor_bytes(self) -> int:
        """Return complete serialized tensor bytes, independent of loaded precision."""
        return sum(group.stored_bytes for group in self.dtypes)


@final
class NativeMusicTensorLayouts(_LayoutContract):
    """All-or-none tensor facts for the exact base, encoder and autoencoder objects."""

    base: NativeMusicTensorLayout = Field(
        description="The base bundle's one model.safetensors object."
    )
    text_encoder: NativeMusicTensorLayout = Field(
        description="The encoder bundle's one model.safetensors object."
    )
    autoencoder: NativeMusicTensorLayout = Field(
        description=(
            "The autoencoder bundle's diffusion_pytorch_model.safetensors object."
        )
    )


class _Tensor(_LayoutContract):
    """One bounded tensor declaration read without importing or mapping its body."""

    dtype: NativeTensorDtype
    shape: tuple[Annotated[int, Field(ge=0, le=_MAX_ELEMENTS)], ...] = Field(
        max_length=256
    )
    data_offsets: tuple[
        Annotated[int, Field(ge=0, le=_MAX_FILE_BYTES)],
        Annotated[int, Field(ge=0, le=_MAX_FILE_BYTES)],
    ]

    @field_validator("shape", "data_offsets", mode="before")
    @classmethod
    def coerce_arrays(cls, value: object) -> object:
        """Retain JSON arrays as immutable exact integer geometry."""
        return tuple(cast("list[object]", value)) if isinstance(value, list) else value


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("native music header repeats a JSON object key")
        result[key] = value
    return result


def _header_document(prefix: bytes, file_size: int) -> dict[str, JsonValue]:
    if (
        isinstance(file_size, bool)
        or not _MIN_PREFIX_BYTES <= file_size <= _MAX_FILE_BYTES
    ):
        raise ValueError("native music tensor object exceeds its selected-file bound")
    if (
        not _MIN_PREFIX_BYTES
        <= len(prefix)
        <= min(file_size, MAX_NATIVE_MUSIC_HEADER_BYTES)
    ):
        raise ValueError("native music tensor header exceeds its prefix bound")
    if int.from_bytes(prefix[:8], "little") != len(prefix) - 8 or prefix[8:9] != b"{":
        raise ValueError("native music tensor header has an inconsistent length word")
    try:
        document = _JSON.validate_python(
            json.loads(prefix[8:], object_pairs_hook=_unique_object)
        )
    except RecursionError as error:
        raise ValueError(
            "native music tensor header exceeds JSON nesting bounds"
        ) from error
    if len(document) > _MAX_TENSORS + 1:
        raise ValueError("native music tensor inventory exceeds its tensor bound")
    return document


def _tensor_extent(tensor: _Tensor) -> tuple[int, int, int]:
    elements = 1
    for dimension in tensor.shape:
        elements *= dimension
        if elements > _MAX_ELEMENTS:
            raise ValueError("native music tensor shape exceeds its element bound")
    start, end = tensor.data_offsets
    bits = elements * _DTYPE_BITS[tensor.dtype]
    if bits % 8 or end < start or end - start != bits // 8:
        raise ValueError("native music tensor offsets disagree with its complete shape")
    return elements, start, end


def _tensor_groups(
    document: dict[str, JsonValue], body_size: int
) -> tuple[NativeMusicTensorDtype, ...]:
    counts: dict[NativeTensorDtype, tuple[int, int]] = {}
    intervals: list[tuple[int, int]] = []
    for name, value in document.items():
        if name == "__metadata__":
            if not isinstance(value, dict) or not all(
                isinstance(item, str) for item in value.values()
            ):
                raise ValueError("Safetensors metadata requires string values")
            continue
        if not name or len(name) > _MAX_NAME_CHARACTERS:
            raise ValueError("native music tensor name exceeds its bound")
        tensor = _Tensor.model_validate(value)
        elements, start, end = _tensor_extent(tensor)
        intervals.append((start, end))
        previous_count, previous_elements = counts.get(tensor.dtype, (0, 0))
        counts[tensor.dtype] = previous_count + 1, previous_elements + elements
    position = 0
    for start, end in sorted(intervals):
        if start != position:
            raise ValueError("native music tensor body contains a hole or overlap")
        position = end
    if position != body_size:
        raise ValueError("native music tensor body is not completely indexed")
    return tuple(
        NativeMusicTensorDtype(dtype=dtype, tensor_count=count, element_count=elements)
        for dtype, (count, elements) in sorted(counts.items())
    )


def project_native_music_tensor_header(
    prefix: bytes, *, file_size: int
) -> NativeMusicTensorLayout:
    """
    Validate a complete bounded prefix and derive its exact intrinsic counts.

    Args:
        prefix: Original eight-byte length word plus exactly its JSON bytes.
        file_size: Complete selected immutable object's independently observed size.

    Returns:
        Prefix digest, size and complete scalar groups. No model load, body
        checksum, effective precision, peak-memory or qualification is implied.

    Raises:
        ValueError: Invalid/duplicate metadata, unknown scalar codes, inconsistent
            shape/offsets, overflow, holes/overlap or incomplete body coverage.

    """
    document = _header_document(prefix, file_size)
    return NativeMusicTensorLayout(
        header_bytes=len(prefix),
        header_sha256=hashlib.sha256(prefix).hexdigest(),
        dtypes=_tensor_groups(document, file_size - len(prefix)),
    )
