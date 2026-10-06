"""Channel-level tests for the authority router adapter."""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from skulk.operator.consensus import (
    AuthorityCatchUpRequestMessage,
    AuthorityNetworkEnvelope,
    create_authority_envelope,
)
from skulk.operator.transport import AuthorityChannelTransport
from skulk.utils.channels import channel


def _private_key() -> bytes:
    """Return one raw signing key for a transport envelope."""

    return Ed25519PrivateKey.generate().private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )


def _envelope(source: UUID, target: UUID) -> AuthorityNetworkEnvelope:
    """Return one valid signed public consensus envelope."""

    payload = AuthorityCatchUpRequestMessage(
        cluster_id=uuid4(),
        request_id=uuid4(),
        after_commit_index=1,
    )
    return create_authority_envelope(payload, source, target, _private_key())


async def test_transport_sends_only_for_its_bound_installation() -> None:
    """One adapter cannot inject an envelope claiming another source."""

    local_id = uuid4()
    outbound_send, outbound_receive = channel[AuthorityNetworkEnvelope]()
    inbound_send, inbound_receive = channel[AuthorityNetworkEnvelope]()
    transport = AuthorityChannelTransport(
        local_id,
        outbound_send,
        inbound_receive,
    )
    valid = _envelope(local_id, uuid4())

    await transport.send(valid)
    assert await outbound_receive.receive() == valid
    with pytest.raises(ValueError, match="another member"):
        await transport.send(_envelope(uuid4(), local_id))
    inbound_send.close()


async def test_transport_filters_broadcasts_for_other_installations() -> None:
    """Shared-topic broadcasts reach consensus only at their stable target."""

    local_id = uuid4()
    remote_id = uuid4()
    outbound_send, _ = channel[AuthorityNetworkEnvelope]()
    inbound_send, inbound_receive = channel[AuthorityNetworkEnvelope]()
    transport = AuthorityChannelTransport(
        local_id,
        outbound_send,
        inbound_receive,
    )
    for_other_member = _envelope(remote_id, remote_id)
    for_local_member = _envelope(remote_id, local_id)

    await inbound_send.send(for_other_member)
    await inbound_send.send(for_local_member)

    assert await transport.receive() == for_local_member
