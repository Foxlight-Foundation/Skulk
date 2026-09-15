"""Canonical signed connector protocol used by the on-demand operator relay."""

from __future__ import annotations

import hashlib
import os
import struct
from dataclasses import dataclass
from typing import Final, cast, final

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import (
    decode_dss_signature,
    encode_dss_signature,
)

_ENVELOPE_MAGIC: Final = b"SKRL"
_PROOF_MAGIC: Final = b"SKRP"
_ADMISSION_MAGIC: Final = b"SKRA"
_FORMAT_VERSION: Final = 1
_ENVELOPE_HEADER_BYTES: Final = 12
_FIELD_HEADER_BYTES: Final = 8
_MAXIMUM_CONTROL_MESSAGE_BYTES: Final = 65_536
_MAXIMUM_FIELD_BYTES: Final = 16_384
_MAXIMUM_FIELD_COUNT: Final = 64
_ADMISSION_BYTES: Final = 195
_MAXIMUM_PROOF_BYTES: Final = 4_096
_P256_ORDER: Final = int(
    "FFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551",
    16,
)

CONNECTOR_HELLO_KIND: Final = 1
CONNECTOR_ACCEPTED_KIND: Final = 2
CONNECTOR_REJECTED_KIND: Final = 3
LEASE_RENEWAL_KIND: Final = 4
LEASE_REPLACED_KIND: Final = 5
CONNECTOR_REVOKED_KIND: Final = 6
HEARTBEAT_KIND: Final = 7
HEARTBEAT_ACK_KIND: Final = 8
OPEN_CONNECTION_KIND: Final = 9
CONNECTION_ACCEPTED_KIND: Final = 10
DRAIN_KIND: Final = 14
DRAIN_ACK_KIND: Final = 15
GO_AWAY_KIND: Final = 16
PROTOCOL_ERROR_KIND: Final = 17


class RelayProtocolError(RuntimeError):
    """Raised when connector control bytes violate the frozen wire contract."""


@final
@dataclass(frozen=True, slots=True)
class ConnectorLease:
    """One signed connector lease and its canonical portable proof."""

    not_before_unix_millis: int
    expires_at_unix_millis: int
    proof: bytes


@final
@dataclass(frozen=True, slots=True)
class ConnectorAccepted:
    """Validated relay acceptance for the connector's exact authority tuple."""

    lease_expires_at_unix_millis: int
    heartbeat_millis: int
    idle_millis: int


@final
@dataclass(frozen=True, slots=True)
class OpenConnection:
    """One relay request for an independent on-demand data socket."""

    connection_id: bytes
    timeout_millis: int


@final
@dataclass(frozen=True, slots=True)
class HeartbeatAcknowledgement:
    """Relay acknowledgement for one connector heartbeat sequence."""

    sequence: int


@final
@dataclass(frozen=True, slots=True)
class DrainRequest:
    """Relay request to stop accepting new connections by one deadline."""

    deadline_unix_millis: int


type RelayServerMessage = (
    ConnectorAccepted | OpenConnection | HeartbeatAcknowledgement | DrainRequest
)


def load_connector_private_key(
    encoded_pkcs8: bytes,
    expected_key_id: bytes,
) -> ec.EllipticCurvePrivateKey:
    """Load a delegated P-256 key and verify its canonical public-key digest."""

    try:
        private_key = serialization.load_der_private_key(encoded_pkcs8, password=None)
    except (TypeError, ValueError) as exc:
        raise RelayProtocolError("connector authority key is invalid") from exc
    if not isinstance(private_key, ec.EllipticCurvePrivateKey) or not isinstance(
        private_key.curve, ec.SECP256R1
    ):
        raise RelayProtocolError("connector authority key is invalid")
    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    if hashlib.sha256(public_key).digest() != expected_key_id:
        raise RelayProtocolError("connector authority key identifier does not match")
    return private_key


def build_connector_hello(
    *,
    private_key: ec.EllipticCurvePrivateKey,
    authority_key_id: bytes,
    routing_locator: bytes,
    region_id: bytes,
    connector_id: bytes,
    authority_epoch: bytes,
    authority_term: int,
    connector_generation: int,
    now_unix_millis: int,
) -> tuple[bytes, ConnectorLease]:
    """Create one signed protocol-1.0 connector hello and reusable proof."""

    lease = _build_lease(
        private_key=private_key,
        authority_key_id=authority_key_id,
        routing_locator=routing_locator,
        region_id=region_id,
        connector_id=connector_id,
        authority_epoch=authority_epoch,
        authority_term=authority_term,
        connector_generation=connector_generation,
        now_unix_millis=now_unix_millis,
    )
    fields = (
        (1, struct.pack(">HHH", 1, 0, 0)),
        (2, routing_locator),
        (3, region_id),
        (4, connector_id),
        (5, authority_epoch),
        (6, _unsigned_64(authority_term)),
        (7, _unsigned_64(connector_generation)),
        (8, _unsigned_64(lease.not_before_unix_millis)),
        (9, _unsigned_64(lease.expires_at_unix_millis)),
        (10, lease.proof),
        (11, _unsigned_64(0)),
        (12, _unsigned_64(0)),
    )
    return _encode_message(CONNECTOR_HELLO_KIND, fields), lease


def build_lease_renewal(
    *,
    private_key: ec.EllipticCurvePrivateKey,
    authority_key_id: bytes,
    routing_locator: bytes,
    region_id: bytes,
    connector_id: bytes,
    authority_epoch: bytes,
    authority_term: int,
    connector_generation: int,
    now_unix_millis: int,
) -> tuple[bytes, ConnectorLease]:
    """Create one signed renewal for the connector's unchanged authority."""

    lease = _build_lease(
        private_key=private_key,
        authority_key_id=authority_key_id,
        routing_locator=routing_locator,
        region_id=region_id,
        connector_id=connector_id,
        authority_epoch=authority_epoch,
        authority_term=authority_term,
        connector_generation=connector_generation,
        now_unix_millis=now_unix_millis,
    )
    fields = (
        (1, authority_epoch),
        (2, _unsigned_64(authority_term)),
        (3, _unsigned_64(connector_generation)),
        (4, _unsigned_64(lease.not_before_unix_millis)),
        (5, _unsigned_64(lease.expires_at_unix_millis)),
        (6, lease.proof),
    )
    return _encode_message(LEASE_RENEWAL_KIND, fields), lease


def encode_heartbeat(sequence: int, now_unix_millis: int) -> bytes:
    """Encode one canonical connector heartbeat."""

    return _encode_message(
        HEARTBEAT_KIND,
        ((1, _unsigned_64(sequence)), (2, _unsigned_64(now_unix_millis))),
    )


def encode_connection_accepted(connection_id: bytes) -> bytes:
    """Encode the mandatory first frame on one accepted data socket."""

    _require_length(connection_id, 16, "connection identifier")
    return _encode_message(CONNECTION_ACCEPTED_KIND, ((1, connection_id),))


def encode_drain_ack(deadline_unix_millis: int) -> bytes:
    """Acknowledge entry into relay-requested drain mode."""

    return _encode_message(
        DRAIN_ACK_KIND,
        ((1, _unsigned_64(deadline_unix_millis)),),
    )


def decode_server_message(
    payload: bytes,
    *,
    connector_id: bytes,
    authority_epoch: bytes,
    authority_term: int,
    connector_generation: int,
) -> RelayServerMessage:
    """Decode and validate one relay-to-connector control message."""

    kind, fields = _decode_message(payload)
    if kind == CONNECTOR_ACCEPTED_KIND:
        _require_exact_fields(fields, 9)
        if fields[1] != struct.pack(">HH", 1, 0):
            raise RelayProtocolError("relay selected an unsupported protocol")
        if (
            fields[2] != connector_id
            or fields[3] != authority_epoch
            or _read_unsigned(fields[4], 8) != authority_term
            or _read_unsigned(fields[5], 8) != connector_generation
            or _read_unsigned(fields[9], 8) != 0
        ):
            raise RelayProtocolError("relay accepted different connector authority")
        heartbeat = _read_unsigned(fields[7], 4)
        idle = _read_unsigned(fields[8], 4)
        if not 1_000 <= heartbeat <= 60_000 or not max(3_000, heartbeat * 3) <= idle <= 300_000:
            raise RelayProtocolError("relay returned invalid connector timing")
        return ConnectorAccepted(
            lease_expires_at_unix_millis=_read_unsigned(fields[6], 8),
            heartbeat_millis=heartbeat,
            idle_millis=idle,
        )
    if kind == OPEN_CONNECTION_KIND:
        _require_exact_fields(fields, 2)
        _require_length(fields[1], 16, "connection identifier")
        timeout = _read_unsigned(fields[2], 4)
        if not 1_000 <= timeout <= 30_000:
            raise RelayProtocolError("relay returned invalid connection timeout")
        return OpenConnection(connection_id=fields[1], timeout_millis=timeout)
    if kind == HEARTBEAT_ACK_KIND:
        _require_exact_fields(fields, 2)
        _read_unsigned(fields[2], 8)
        return HeartbeatAcknowledgement(sequence=_read_unsigned(fields[1], 8))
    if kind == DRAIN_KIND:
        _require_exact_fields(fields, 1)
        return DrainRequest(deadline_unix_millis=_read_unsigned(fields[1], 8))
    if kind in {
        CONNECTOR_REJECTED_KIND,
        LEASE_REPLACED_KIND,
        CONNECTOR_REVOKED_KIND,
        GO_AWAY_KIND,
        PROTOCOL_ERROR_KIND,
    }:
        raise RelayProtocolError("relay terminated the connector authority")
    raise RelayProtocolError("relay sent an unexpected control message")


def _build_lease(
    *,
    private_key: ec.EllipticCurvePrivateKey,
    authority_key_id: bytes,
    routing_locator: bytes,
    region_id: bytes,
    connector_id: bytes,
    authority_epoch: bytes,
    authority_term: int,
    connector_generation: int,
    now_unix_millis: int,
) -> ConnectorLease:
    """Build and sign a five-minute authority statement with bounded clock skew."""

    _require_length(authority_key_id, 32, "authority key identifier")
    _require_length(routing_locator, 32, "routing locator")
    _require_length(region_id, 8, "region identifier")
    _require_length(connector_id, 16, "connector identifier")
    _require_length(authority_epoch, 16, "authority epoch")
    not_before = max(0, now_unix_millis - 5_000)
    expires_at = now_unix_millis + 295_000
    statement = b"".join(
        (
            _ADMISSION_MAGIC,
            bytes((_FORMAT_VERSION,)),
            struct.pack(">HHH", 1, 0, 0),
            routing_locator,
            region_id,
            connector_id,
            authority_epoch,
            _unsigned_64(authority_term),
            _unsigned_64(connector_generation),
            _unsigned_64(not_before),
            _unsigned_64(expires_at),
            os.urandom(32),
            _unsigned_64(0),
            _unsigned_64(0),
            authority_key_id,
        )
    )
    if len(statement) != _ADMISSION_BYTES:
        raise AssertionError("canonical admission statement length changed")
    signature_der = private_key.sign(statement, ec.ECDSA(hashes.SHA256()))
    signature_r, signature_s = decode_dss_signature(signature_der)
    signature_s = min(signature_s, _P256_ORDER - signature_s)
    # Re-encoding verifies both scalar bounds before fixed-width P1363 export.
    decode_dss_signature(encode_dss_signature(signature_r, signature_s))
    signature = signature_r.to_bytes(32, "big") + signature_s.to_bytes(32, "big")
    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    proof = b"".join(
        (
            _PROOF_MAGIC,
            bytes((_FORMAT_VERSION, 1)),
            struct.pack(">HHH", len(public_key), len(statement), len(signature)),
            public_key,
            statement,
            signature,
        )
    )
    if len(proof) > _MAXIMUM_PROOF_BYTES:
        raise RelayProtocolError("connector admission proof exceeds its bound")
    return ConnectorLease(not_before, expires_at, proof)


def _encode_message(kind: int, fields: tuple[tuple[int, bytes], ...]) -> bytes:
    """Encode one exact SKRL-TLV1 message with required fields."""

    encoded_fields = bytearray()
    previous_identifier = 0
    for identifier, value in fields:
        if identifier <= previous_identifier or not value or len(value) > _MAXIMUM_FIELD_BYTES:
            raise RelayProtocolError("connector control field is invalid")
        previous_identifier = identifier
        encoded_fields.extend(struct.pack(">HBBI", identifier, 1, 0, len(value)))
        encoded_fields.extend(value)
    payload = bytes(encoded_fields)
    encoded = _ENVELOPE_MAGIC + struct.pack(">BBHI", 1, kind, 0, len(payload)) + payload
    if len(fields) > _MAXIMUM_FIELD_COUNT or len(encoded) > _MAXIMUM_CONTROL_MESSAGE_BYTES:
        raise RelayProtocolError("connector control message exceeds its bound")
    return encoded


def _decode_message(payload: bytes) -> tuple[int, dict[int, bytes]]:
    """Decode exact SKRL-TLV1 framing and canonical required fields."""

    if len(payload) < _ENVELOPE_HEADER_BYTES or len(payload) > _MAXIMUM_CONTROL_MESSAGE_BYTES:
        raise RelayProtocolError("relay control message has invalid length")
    magic, version, kind, reserved, declared_length = cast(
        tuple[bytes, int, int, int, int],
        struct.unpack(">4sBBHI", payload[:12]),
    )
    if magic != _ENVELOPE_MAGIC or version != 1 or reserved != 0:
        raise RelayProtocolError("relay control message has invalid framing")
    if declared_length != len(payload) - _ENVELOPE_HEADER_BYTES:
        raise RelayProtocolError("relay control message length does not match")
    fields: dict[int, bytes] = {}
    offset = _ENVELOPE_HEADER_BYTES
    previous_identifier = 0
    while offset < len(payload):
        if len(fields) >= _MAXIMUM_FIELD_COUNT or offset + _FIELD_HEADER_BYTES > len(payload):
            raise RelayProtocolError("relay control fields exceed their bound")
        identifier, flags, field_reserved, value_length = cast(
            tuple[int, int, int, int],
            struct.unpack(">HBBI", payload[offset : offset + _FIELD_HEADER_BYTES]),
        )
        offset += _FIELD_HEADER_BYTES
        if (
            identifier <= previous_identifier
            or flags not in {0, 1}
            or field_reserved != 0
            or value_length == 0
            or value_length > _MAXIMUM_FIELD_BYTES
            or offset + value_length > len(payload)
        ):
            raise RelayProtocolError("relay control field is malformed")
        previous_identifier = identifier
        value = payload[offset : offset + value_length]
        offset += value_length
        if flags == 1:
            fields[identifier] = value
    if offset != len(payload):
        raise RelayProtocolError("relay control message has trailing data")
    return kind, fields


def _require_exact_fields(fields: dict[int, bytes], maximum_identifier: int) -> None:
    """Require every base field and reject unknown critical fields."""

    if set(fields) != set(range(1, maximum_identifier + 1)):
        raise RelayProtocolError("relay control message has invalid fields")


def _unsigned_64(value: int) -> bytes:
    """Encode one checked unsigned 64-bit scalar."""

    if not 0 <= value <= (2**64) - 1:
        raise RelayProtocolError("connector control scalar is out of range")
    return value.to_bytes(8, "big")


def _read_unsigned(value: bytes, length: int) -> int:
    """Read one exact-width unsigned scalar."""

    _require_length(value, length, "control scalar")
    return int.from_bytes(value, "big")


def _require_length(value: bytes, length: int, name: str) -> None:
    """Require one exact fixed-width protocol value."""

    if len(value) != length:
        raise RelayProtocolError(f"{name} has invalid length")
