"""Frozen-wire and signature tests for the on-demand relay connector."""

from __future__ import annotations

import hashlib
import struct
from typing import cast

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

from skulk.operator.relay_protocol import (
    ConnectorAccepted,
    OpenConnection,
    RelayProtocolError,
    build_connector_hello,
    decode_server_message,
    encode_connection_accepted,
    encode_heartbeat,
)

_P256_ORDER = int(
    "FFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551",
    16,
)


def _field(identifier: int, value: bytes) -> bytes:
    """Encode one required field independently for frozen test messages."""

    return struct.pack(">HBBI", identifier, 1, 0, len(value)) + value


def _message(kind: int, *fields: bytes) -> bytes:
    """Encode one independent frozen server message fixture."""

    payload = b"".join(fields)
    return b"SKRL" + struct.pack(">BBHI", 1, kind, 0, len(payload)) + payload


def test_small_client_messages_match_frozen_relay_vectors() -> None:
    """Heartbeat and data-preface bytes remain identical to relay vectors."""

    assert encode_heartbeat(21, 1_000_000).hex() == (
        "534b524c010700000000002000010100000000080000000000000015"
        "000201000000000800000000000f4240"
    )
    assert encode_connection_accepted(bytes([0x44]) * 16).hex() == (
        "534b524c010a0000000000180001010000000010"
        "44444444444444444444444444444444"
    )


def test_server_acceptance_and_open_are_strictly_decoded() -> None:
    """The connector accepts only its exact authority and bounded open request."""

    connector_id = bytes([0x22]) * 16
    epoch = bytes([0x55]) * 16
    accepted_bytes = _message(
        2,
        _field(1, struct.pack(">HH", 1, 0)),
        _field(2, connector_id),
        _field(3, epoch),
        _field(4, (8).to_bytes(8, "big")),
        _field(5, (13).to_bytes(8, "big")),
        _field(6, (1_030_000).to_bytes(8, "big")),
        _field(7, (5_000).to_bytes(4, "big")),
        _field(8, (20_000).to_bytes(4, "big")),
        _field(9, (0).to_bytes(8, "big")),
    )
    accepted = decode_server_message(
        accepted_bytes,
        connector_id=connector_id,
        authority_epoch=epoch,
        authority_term=8,
        connector_generation=13,
    )
    assert accepted == ConnectorAccepted(1_030_000, 5_000, 20_000)

    connection_id = bytes([0x44]) * 16
    opened = decode_server_message(
        _message(
            9,
            _field(1, connection_id),
            _field(2, (5_000).to_bytes(4, "big")),
        ),
        connector_id=connector_id,
        authority_epoch=epoch,
        authority_term=8,
        connector_generation=13,
    )
    assert opened == OpenConnection(connection_id, 5_000)

    with pytest.raises(RelayProtocolError, match="different connector authority"):
        decode_server_message(
            accepted_bytes,
            connector_id=bytes(16),
            authority_epoch=epoch,
            authority_term=8,
            connector_generation=13,
        )


def test_connector_hello_contains_a_low_s_verifiable_admission_proof() -> None:
    """Generated SKRP1 proves the exact hello authority without malleability."""

    private_key = ec.generate_private_key(ec.SECP256R1())
    public_der = private_key.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    key_id = hashlib.sha256(public_der).digest()
    hello, lease = build_connector_hello(
        private_key=private_key,
        authority_key_id=key_id,
        routing_locator=bytes([0x11]) * 32,
        region_id=bytes([0x33]) * 8,
        connector_id=bytes([0x22]) * 16,
        authority_epoch=bytes([0x55]) * 16,
        authority_term=8,
        connector_generation=13,
        now_unix_millis=1_000_000,
    )

    assert hello.startswith(b"SKRL\x01\x01")
    assert lease.not_before_unix_millis == 995_000
    assert lease.expires_at_unix_millis == 1_295_000
    proof = lease.proof
    assert proof[:6] == b"SKRP\x01\x01"
    issuer_length, payload_length, signature_length = cast(
        tuple[int, int, int],
        struct.unpack(">HHH", proof[6:12]),
    )
    assert (issuer_length, payload_length, signature_length) == (91, 195, 64)
    statement_start = 12 + issuer_length
    statement = proof[statement_start : statement_start + payload_length]
    signature = proof[-signature_length:]
    signature_r = int.from_bytes(signature[:32], "big")
    signature_s = int.from_bytes(signature[32:], "big")
    assert signature_s <= _P256_ORDER // 2
    private_key.public_key().verify(
        encode_dss_signature(signature_r, signature_s),
        statement,
        ec.ECDSA(hashes.SHA256()),
    )
    assert statement[:5] == b"SKRA\x01"
    assert statement[163:195] == key_id
