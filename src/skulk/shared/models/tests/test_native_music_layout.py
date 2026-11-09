"""Exact Registry header evidence and hostile native tensor metadata at the Core edge."""

import copy
import hashlib
import json
import tomllib
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from skulk.shared.models.model_cards import ModelCard
from skulk.shared.models.native_music_layout import (
    MAX_NATIVE_MUSIC_HEADER_BYTES,
    NativeMusicTensorLayout,
    NativeMusicTensorLayouts,
    project_native_music_tensor_header,
)

HEADERS = Path(__file__).with_name("native-music-headers")


def body() -> dict[str, object]:
    """Load the complete unchanged selected-artifact fixture."""
    return cast(
        "dict[str, object]",
        tomllib.loads(Path(__file__).with_name("ace-native-card.toml").read_text()),
    )


def section(document: dict[str, object], *keys: str) -> dict[str, object]:
    """Select a known typed table for a deliberate malformed input."""
    current = document
    for key in keys:
        current = cast("dict[str, object]", current[key])
    return current


def layouts() -> NativeMusicTensorLayouts:
    """Read retained Registry prefix facts, without fetching or loading weights."""
    return NativeMusicTensorLayouts.model_validate_json(
        (HEADERS / "layouts.json").read_bytes()
    )


def prefix(document: dict[str, object]) -> bytes:
    """Encode one complete header with explicit original whitespace padding."""
    data = json.dumps(document, separators=(",", ":")).encode()
    data += b" " * (-len(data) % 8)
    return len(data).to_bytes(8, "little") + data


def test_actual_prefixes_bind_complete_declared_scalar_geometry() -> None:
    """Original object sizes equal complete header plus every indexed tensor byte."""
    observed = NativeMusicTensorLayouts(
        base=project_native_music_tensor_header(
            (HEADERS / "base.header.bin").read_bytes(), file_size=4787825604
        ),
        text_encoder=project_native_music_tensor_header(
            (HEADERS / "text_encoder.header.bin").read_bytes(), file_size=1191586416
        ),
        autoencoder=project_native_music_tensor_header(
            (HEADERS / "autoencoder.header.bin").read_bytes(), file_size=337431388
        ),
    )
    assert observed == layouts()
    assert (
        observed.base.header_sha256
        == "1b5ffdcb5660c9ca5bf91ad3867afc266b2697ba962e6580fd5675f82ca344f1"
    )
    assert (
        observed.text_encoder.header_sha256
        == "1ff013283b0190994f5557f04301841482bcba6c006dd557fcb15563d4f1768b"
    )
    assert (
        observed.autoencoder.header_sha256
        == "ee65a4b1609150f6be18b8b164c77f340a326c3a5912acc686ea5227f60c9528"
    )
    components = (observed.base, observed.text_encoder, observed.autoencoder)
    assert [entry.dtypes[0].dtype for entry in components] == ["BF16"] * 3
    assert [entry.dtypes[0].tensor_count for entry in components] == [677, 310, 365]
    assert [entry.dtypes[0].element_count for entry in components] == [
        2393872518,
        595776512,
        168695426,
    ]
    assert sum(entry.stored_tensor_bytes for entry in components) == 6316688912
    assert (
        NativeMusicTensorLayouts.model_validate_json(observed.model_dump_json())
        == observed
    )


@pytest.mark.parametrize(
    "fault",
    [
        "partial",
        "duplicate",
        "wrong_size",
        "header_size",
        "unknown_type",
        "bool_count",
        "negative_count",
        "invalid_digest",
    ],
)
def test_unbounded_or_incomplete_declared_layout_is_refused(fault: str) -> None:
    """Strict complete groups must still agree with the selected object sizes."""
    document = body()
    metadata = cast("dict[str, object]", layouts().model_dump(mode="json"))
    if fault == "partial":
        del metadata["autoencoder"]
    else:
        base = section(metadata, "base")
        groups = cast("list[dict[str, object]]", base["dtypes"])
        if fault == "duplicate":
            groups.append(copy.deepcopy(groups[0]))
        elif fault == "wrong_size":
            groups[0]["element_count"] = cast("int", groups[0]["element_count"]) - 1
        elif fault == "header_size":
            base["header_bytes"] = 80569
        elif fault == "unknown_type":
            groups[0]["dtype"] = "AUTO"
        elif fault == "bool_count":
            groups[0]["tensor_count"] = True
        elif fault == "negative_count":
            groups[0]["element_count"] = -1
        else:
            base["header_sha256"] = "not-a-digest"
    section(document, "music", "native")["tensor_layouts"] = metadata
    with pytest.raises(ValidationError):
        ModelCard.model_validate(document)


def test_scalar_and_empty_tensors_do_not_disappear_from_projection() -> None:
    """Scalar rank zero is one value; an empty tensor retains its tensor count."""
    header = prefix(
        {
            "weight": {"dtype": "F16", "shape": [4], "data_offsets": [0, 8]},
            "scalar": {"dtype": "I64", "shape": [], "data_offsets": [8, 16]},
            "empty": {"dtype": "BF16", "shape": [0, 32], "data_offsets": [16, 16]},
            "__metadata__": {"format": "pt"},
        }
    )
    observed = project_native_music_tensor_header(header, file_size=len(header) + 16)
    assert observed.header_sha256 == hashlib.sha256(header).hexdigest()
    assert {
        group.dtype: (group.tensor_count, group.element_count)
        for group in observed.dtypes
    } == {"F16": (1, 4), "I64": (1, 1), "BF16": (1, 0)}
    assert observed.stored_tensor_bytes == 16


@pytest.mark.parametrize(
    ("dtype", "elements", "size"),
    [
        ("F4", 2, 1),
        ("F6_E2M3", 4, 3),
        ("F6_E3M2", 4, 3),
        ("F8_E4M3", 4, 4),
        ("F8_E5M2", 4, 4),
        ("F8_E8M0", 4, 4),
        ("C64", 1, 8),
        ("BOOL", 3, 3),
    ],
)
def test_scalar_width_is_observed_without_assuming_byte_aligned_elements(
    dtype: str, elements: int, size: int
) -> None:
    """Packed and complex file facts are retained without claiming runtime execution."""
    header = prefix(
        {"weight": {"dtype": dtype, "shape": [elements], "data_offsets": [0, size]}}
    )
    observed = project_native_music_tensor_header(header, file_size=len(header) + size)
    assert observed.stored_tensor_bytes == size
    assert observed.dtypes[0].element_count == elements


@pytest.mark.parametrize(
    "fault",
    [
        "hole",
        "overlap",
        "trailer",
        "wrong_shape",
        "bool_shape",
        "oversized_shape",
        "metadata",
        "unknown",
        "misaligned",
    ],
)
def test_prefix_validation_refuses_inconsistent_or_unbounded_geometry(
    fault: str,
) -> None:
    """Complete offset coverage and strict shape math precede any trusted projection."""
    tensor: dict[str, object] = {"dtype": "BF16", "shape": [4], "data_offsets": [0, 8]}
    document: dict[str, object] = {"weight": tensor}
    extra = 0
    if fault == "hole":
        tensor["data_offsets"] = [1, 9]
    elif fault == "overlap":
        document["second"] = {"dtype": "BF16", "shape": [2], "data_offsets": [4, 8]}
    elif fault == "trailer":
        extra = 1
    elif fault == "wrong_shape":
        tensor["shape"] = [5]
    elif fault == "bool_shape":
        tensor["shape"] = [True]
    elif fault == "oversized_shape":
        tensor["shape"] = [2**62, 4]
    elif fault == "metadata":
        document["__metadata__"] = {"format": 7}
    elif fault == "unknown":
        tensor["dtype"] = "UNKNOWN"
    elif fault == "misaligned":
        tensor.update(dtype="F4", shape=[3], data_offsets=[0, 2])
    header = prefix(document)
    with pytest.raises(ValueError, match=r"native music|Safetensors|validation error"):
        project_native_music_tensor_header(header, file_size=len(header) + 8 + extra)


def test_prefix_length_word_must_match_exact_retained_bytes() -> None:
    """A partial or overlong header cannot project trustworthy scalar counts."""
    header = prefix({"weight": {"dtype": "BF16", "shape": [4], "data_offsets": [0, 8]}})
    header = (len(header) + 100).to_bytes(8, "little") + header[8:]
    with pytest.raises(ValueError, match="inconsistent length word"):
        project_native_music_tensor_header(header, file_size=len(header) + 8)


def test_duplicate_header_keys_are_rejected_before_json_can_discard_them() -> None:
    """A repeated name cannot hide a tensor from the logical scalar count."""
    data = (
        b'{"weight":{"dtype":"BF16","shape":[4],"data_offsets":[0,8]},'
        b'"weight":{"dtype":"BF16","shape":[2],"data_offsets":[4,8]}}'
    )
    header = len(data).to_bytes(8, "little") + data
    with pytest.raises(ValueError, match="repeats"):
        project_native_music_tensor_header(header, file_size=len(header) + 8)


def test_layout_has_no_runtime_memory_or_hardware_permission_fields() -> None:
    """Intrinsic serialization grants no universal memory floor or GPU allowlist."""
    document = layouts().base.model_dump(mode="json")
    document["required_memory"] = 32 * 1024**3
    with pytest.raises(ValidationError):
        NativeMusicTensorLayout.model_validate(document)


def test_optional_layout_preserves_original_native_serialization_and_selection() -> (
    None
):
    """Absent defaults never rewrite historical signed metadata or selected files."""
    original = ModelCard.model_validate(body())
    # Measured with the unchanged pre-layout reader (2c90b892; identical card
    # source at initiative 8ab8cb0c), not generated from this implementation.
    for aliases, expected in (
        (False, "ec9329da03cbc96f9c10e35fd90ae213677b17a9730831ad1de2fa3f052a1413"),
        (True, "3892f7bab0a5d52fdef3bd078dbae0c8e40f06ff32bbe796e1096848f36dcd6c"),
    ):
        prior = json.dumps(
            original.model_dump(mode="json", by_alias=aliases),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        assert hashlib.sha256(prior).hexdigest() == expected
    encoded = original.model_dump(mode="json", by_alias=True)
    assert "tensor_layouts" not in section(encoded, "music", "native")
    assert original.music is not None and original.music.native is not None
    assert original.music.native.tensor_layouts is None
    with_layout = body()
    section(with_layout, "music", "native")["tensor_layouts"] = layouts().model_dump(
        mode="json"
    )
    updated = ModelCard.model_validate(with_layout)
    assert ModelCard.model_validate_json(updated.model_dump_json()) == updated
    assert (
        updated.native_music_artifact_sources()
        == original.native_music_artifact_sources()
    )
    assert updated.tasks == original.tasks
    assert updated.storage_size == original.storage_size
    assert updated.placement == original.placement


@pytest.mark.parametrize(
    "fault", ["too_large", "zero_length", "bool_size", "deep_json"]
)
def test_prefix_bounds_fail_as_bounded_validation_errors(fault: str) -> None:
    """A finite file cannot request unbounded prefix storage or recursive JSON."""
    header = prefix({"weight": {"dtype": "BF16", "shape": [4], "data_offsets": [0, 8]}})
    size = len(header) + 8
    if fault == "too_large":
        data = b"{" + b" " * MAX_NATIVE_MUSIC_HEADER_BYTES + b"}"
        header = len(data).to_bytes(8, "little") + data
        size = len(header) + 8
    elif fault == "zero_length":
        header = b"\x00" * 8
    elif fault == "bool_size":
        size = True
    else:
        data = b'{"weight":' + b"[" * 2000 + b"0" + b"]" * 2000 + b"}"
        header = len(data).to_bytes(8, "little") + data
        size = len(header) + 8
    with pytest.raises(ValueError):
        project_native_music_tensor_header(header, file_size=size)
