"""A node that cannot take its peer port says which port and why.

libp2p's own error for a port it could not take has no text, so a node used
to end with a bare ``RuntimeError``. Seen on a Windows node whose first two
starts failed that way with another program on the port.
"""

import socket

import pytest
from skulk_pyo3_bindings import Keypair

from skulk.routing import router as router_module
from skulk.routing.router import Router, why_no_peer_listener


def _held() -> tuple[socket.socket, int]:
    """A listener on a port of the system's choosing, as another program holds one."""
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.bind(("0.0.0.0", 0))
    holder.listen()
    return holder, holder.getsockname()[1]


def test_a_held_port_is_named_with_what_the_system_says_about_it() -> None:
    holder, port = _held()
    try:
        message = why_no_peer_listener(port)
        with pytest.raises(OSError) as refused:
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).bind(("0.0.0.0", port))
    finally:
        holder.close()
    assert f"TCP port {port}" in message
    # The operating system's own words, in whatever language it speaks.
    assert str(refused.value) in message
    assert "Another program" in message
    assert "--libp2p-port" in message and "SKULK_LIBP2P_PORT" in message


def test_a_port_that_is_free_again_is_said_to_be_free() -> None:
    holder, port = _held()
    holder.close()
    message = why_no_peer_listener(port)
    assert f"TCP port {port}" in message
    assert "free now" in message and "gave no reason" in message
    assert "--libp2p-port" in message


def test_a_port_of_the_systems_choosing_has_no_number_to_name() -> None:
    message = why_no_peer_listener(0)
    assert "gave no reason" in message
    assert "TCP port 0" not in message


def test_the_router_gives_a_reason_where_the_network_layer_gave_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    holder, port = _held()

    def says_nothing(*_arguments: object) -> None:
        # How the bindings report a port libp2p could not take.
        raise RuntimeError

    monkeypatch.setattr(router_module, "NetworkingHandle", says_nothing)
    try:
        with pytest.raises(RuntimeError, match=f"TCP port {port}") as caught:
            Router.create(Keypair.generate(), listen_port=port)
    finally:
        holder.close()
    assert "Another program" in str(caught.value)
    # The original failure is kept, for whoever reads the traceback.
    assert isinstance(caught.value.__cause__, RuntimeError)


def test_an_error_that_says_something_is_passed_on_as_it_is(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def says_why(*_arguments: object) -> None:
        raise RuntimeError("the key is of a kind libp2p does not accept")

    monkeypatch.setattr(router_module, "NetworkingHandle", says_why)
    with pytest.raises(RuntimeError, match="does not accept") as caught:
        Router.create(Keypair.generate(), listen_port=52416)
    assert caught.value.__cause__ is None


def test_the_real_network_layer_cannot_take_a_held_port_and_the_node_says_so() -> None:
    """With the real bindings: this is the failure the sentence exists for."""
    holder, port = _held()
    try:
        with pytest.raises(RuntimeError, match=f"TCP port {port}") as caught:
            Router.create(Keypair.generate(), listen_port=port)
    finally:
        holder.close()
    assert str(caught.value.__cause__) == ""
