"""Channel adapter between authority consensus and Skulk's message router."""

from __future__ import annotations

from typing import final
from uuid import UUID

from skulk.operator.consensus import AuthorityNetworkEnvelope
from skulk.utils.channels import Receiver, Sender


@final
class AuthorityChannelTransport:
    """Deliver signed authority envelopes over injected router channels.

    The adapter deliberately knows nothing about libp2p, retries, elections, or
    secret payloads. The caller supplies the sender and receiver created for the
    ``AUTHORITY_MESSAGES`` topic. Because that topic currently broadcasts over
    gossipsub, the receive boundary discards envelopes addressed to other stable
    installations before consensus sees them.
    """

    def __init__(
        self,
        node_install_id: UUID,
        sender: Sender[AuthorityNetworkEnvelope],
        receiver: Receiver[AuthorityNetworkEnvelope],
    ) -> None:
        """Bind one stable installation to its injected topic channels.

        Args:
            node_install_id: Stable installation identity served by this adapter.
            sender: Authority topic sender obtained from the Skulk router.
            receiver: Authority topic receiver obtained from the Skulk router.
        """

        self._node_install_id = node_install_id
        self._sender = sender
        self._receiver = receiver

    async def send(self, envelope: AuthorityNetworkEnvelope) -> None:
        """Send one envelope whose authenticated source is this installation.

        Args:
            envelope: Signed public consensus message to deliver.

        Raises:
            ValueError: The envelope claims another source installation.
        """

        if UUID(str(envelope.source_node_install_id)) != self._node_install_id:
            raise ValueError("authority transport cannot send for another member")
        await self._sender.send(envelope)

    async def receive(self) -> AuthorityNetworkEnvelope:
        """Return the next envelope addressed to this stable installation."""

        while True:
            envelope = await self._receiver.receive()
            if UUID(str(envelope.target_node_install_id)) == self._node_install_id:
                return envelope
