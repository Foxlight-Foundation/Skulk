"""A node that cannot take its peer port says which port and why.

libp2p's own error for a port it could not take prints as nothing, so a node
used to end with a bare ``RuntimeError``. Seen on a Windows node whose first
two starts failed that way with another program on the port. The network
layer now names the port and passes libp2p's reason on, and the router adds
what the owner can do.
"""

import re
import socket

import pytest
from skulk_pyo3_bindings import Keypair

from skulk.routing import router as router_module
from skulk.routing.router import (
    PEER_LISTENER_REFUSED,
    Router,
    peer_listener_refused,
)


def _held() -> tuple[socket.socket, int]:
    """A listener on a port of the system's choosing, as another program holds one."""
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("0.0.0.0", 0))
    holder.listen()
    return holder, holder.getsockname()[1]


def test_the_owner_is_told_the_reason_and_what_to_do() -> None:
    said = (
        PEER_LISTENER_REFUSED + "TCP port 52416: Address already in use (os error 48)"
    )
    message = peer_listener_refused(said, 52416)
    assert message.startswith(f"Skulk {said}. ")
    assert "--libp2p-port" in message and "SKULK_LIBP2P_PORT" in message


def test_a_port_of_the_systems_choosing_is_not_one_to_change() -> None:
    said = PEER_LISTENER_REFUSED + "a TCP port of the system's choosing: no buffers"
    assert peer_listener_refused(said, 0) == f"Skulk {said}."


def test_the_router_adds_what_to_do_to_the_network_layers_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    said = PEER_LISTENER_REFUSED + "TCP port 52416: the reason libp2p gave"

    def refuses(*_arguments: object) -> None:
        raise RuntimeError(said)

    monkeypatch.setattr(router_module, "NetworkingHandle", refuses)
    with pytest.raises(RuntimeError, match="the reason libp2p gave") as caught:
        Router.create(Keypair.generate(), listen_port=52416)
    assert str(caught.value) == peer_listener_refused(said, 52416)
    # The original failure is kept, for whoever reads the traceback.
    assert str(caught.value.__cause__) == said


@pytest.mark.parametrize("said", ["the key is of a kind libp2p does not accept", ""])
def test_any_other_error_is_passed_on_as_it_is(
    monkeypatch: pytest.MonkeyPatch, said: str
) -> None:
    """Also one that says nothing: silence is not taken for a port in use."""
    failure = RuntimeError(said)

    def fails(*_arguments: object) -> None:
        raise failure

    monkeypatch.setattr(router_module, "NetworkingHandle", fails)
    with pytest.raises(RuntimeError) as caught:
        Router.create(Keypair.generate(), listen_port=52416)
    assert caught.value is failure


def test_the_real_network_layer_cannot_take_a_held_port_and_the_node_says_so() -> None:
    """With the real bindings: the port, the system's own reason, and what to do."""
    holder, port = _held()
    try:
        with pytest.raises(RuntimeError) as caught:
            Router.create(Keypair.generate(), listen_port=port)
    finally:
        holder.close()
    message = str(caught.value)
    named = f"Skulk {PEER_LISTENER_REFUSED}TCP port {port}: "
    assert message.startswith(named)
    # The operating system's own error, whatever language it speaks. Which
    # error is the platform's business: Windows refuses libp2p's bind with
    # 10013 where it refuses a plain bind with 10048.
    assert re.match(r".+ \(os error \d+\)\. ", message[len(named) :])
    assert "--libp2p-port" in message
    # The network layer's own words, which the sentence was made from.
    assert str(caught.value.__cause__).startswith(PEER_LISTENER_REFUSED)
