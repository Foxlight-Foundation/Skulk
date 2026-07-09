"""Tests for the capability descriptor meta-contract (fabric-citizenship 2a)."""

import pytest
from pydantic import ValidationError

from skulk.extensions.capabilities import (
    CapabilityDescriptor,
    descriptor_revision,
)

_ECHO_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
}


def _descriptor(**overrides: object) -> CapabilityDescriptor:
    fields: dict[str, object] = {
        "id": "echo",
        "version": "1.0.0",
        "title": "Echo",
        "description": "Returns the input text unchanged.",
        "input_schema": _ECHO_SCHEMA,
        "output_schema": _ECHO_SCHEMA,
    }
    fields.update(overrides)
    return CapabilityDescriptor.model_validate(fields)


def test_descriptor_valid_unary() -> None:
    descriptor = _descriptor()
    assert descriptor.io_mode == "unary"
    assert descriptor.qualified_id == "echo@1.0.0"


@pytest.mark.parametrize("bad_id", ["", "Has Space", "UPPER", "-leading", "tab\tid"])
def test_descriptor_rejects_bad_ids(bad_id: str) -> None:
    with pytest.raises(ValidationError):
        _descriptor(id=bad_id)


def test_descriptor_accepts_namespaced_ids() -> None:
    # Colon/dot/dash namespacing is how tags like "embeddings:bge-m3" work.
    assert _descriptor(id="embeddings:bge-m3").id == "embeddings:bge-m3"


@pytest.mark.parametrize("bad_version", ["1", "1.0", "v1.0.0", "1.0.0-beta", ""])
def test_descriptor_rejects_non_semver(bad_version: str) -> None:
    with pytest.raises(ValidationError):
        _descriptor(version=bad_version)


def test_descriptor_rejects_blank_title_and_description() -> None:
    with pytest.raises(ValidationError):
        _descriptor(title="   ")
    with pytest.raises(ValidationError):
        _descriptor(description="")


def test_streaming_modes_require_matching_chunk_schemas() -> None:
    # server_streaming requires an output chunk schema...
    with pytest.raises(ValidationError):
        _descriptor(io_mode="server_streaming")
    described = _descriptor(
        io_mode="server_streaming", output_chunk_schema=_ECHO_SCHEMA
    )
    assert described.output_chunk_schema is not None
    # ...client_streaming requires an input chunk schema...
    with pytest.raises(ValidationError):
        _descriptor(io_mode="client_streaming")
    # ...bidirectional requires both...
    with pytest.raises(ValidationError):
        _descriptor(io_mode="bidirectional", output_chunk_schema=_ECHO_SCHEMA)
    # ...and unary forbids either (a chunk schema on a unary call would drift
    # silently, since nothing exercises it).
    with pytest.raises(ValidationError):
        _descriptor(output_chunk_schema=_ECHO_SCHEMA)


def test_descriptor_revision_is_stable_and_drift_sensitive() -> None:
    # Same shape -> same revision (deterministic canonical form)...
    assert descriptor_revision(_descriptor()) == descriptor_revision(_descriptor())
    # ...any drift -> different revision, even without a version bump. This is
    # what lets a call detect that discovery and invocation disagree.
    edited = _descriptor(
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}, "loud": {"type": "boolean"}},
        }
    )
    assert descriptor_revision(edited) != descriptor_revision(_descriptor())


def test_descriptor_round_trips_json() -> None:
    descriptor = _descriptor(
        io_mode="server_streaming", output_chunk_schema=_ECHO_SCHEMA
    )
    restored = CapabilityDescriptor.model_validate(descriptor.model_dump(mode="json"))
    assert restored == descriptor
    assert descriptor_revision(restored) == descriptor_revision(descriptor)


def test_descriptor_rejects_non_json_schema_values() -> None:
    # A set/bytes/NaN inside a schema would otherwise surface later as a 500
    # on /v1/capabilities or a crash in descriptor_revision; it must fail at
    # construction instead.
    with pytest.raises(ValidationError):
        _descriptor(input_schema={"enum": {"a", "b"}})  # set
    with pytest.raises(ValidationError):
        _descriptor(output_schema={"default": b"raw"})  # bytes
    with pytest.raises(ValidationError):
        _descriptor(input_schema={"maximum": float("nan")})  # NaN
