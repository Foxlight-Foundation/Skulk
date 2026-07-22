"""Session-edge ingress and probe backoff units (#662)."""

from collections import defaultdict

from skulk_pyo3_bindings import PyFromSwarm

from skulk.routing.connection_message import ConnectionMessage


def test_connection_message_carries_remote_endpoint() -> None:
    """The bindings' observed remote endpoint reaches the Python message.

    The session-edge path annotates the topology edge with the connection's
    real endpoint; dropping it here would leave nothing to record.
    """
    connected = ConnectionMessage.from_update(
        PyFromSwarm.Connection("12D3KooWTestPeer", True, "100.95.14.7", 52416)
    )
    assert connected.remote_ip == "100.95.14.7"
    assert connected.remote_tcp_port == 52416

    disconnected = ConnectionMessage.from_update(
        PyFromSwarm.Connection("12D3KooWTestPeer", False, "", 0)
    )
    assert disconnected.remote_ip is None
    assert disconnected.remote_tcp_port is None


def test_probe_backoff_schedule() -> None:
    """Failing addresses drop to the slow cadence; success resets.

    Mirrors the worker's sweep bookkeeping: an address enters backoff after
    three consecutive failed sweeps and is then probed only every sixth
    sweep, so a permanently dead advertised address (the WAN-member shape)
    stops flooding warnings while a path that comes back is rediscovered
    within about a minute.
    """
    backoff_after = 3
    retry_rounds = 6
    probe_failures: defaultdict[str, int] = defaultdict(int)

    def due(ip: str, sweep: int) -> bool:
        if probe_failures[ip] < backoff_after:
            return True
        return sweep % retry_rounds == 0

    dead = "203.0.113.9"
    probed_sweeps: list[int] = []
    for sweep in range(1, 19):
        if due(dead, sweep):
            probed_sweeps.append(sweep)
            probe_failures[dead] += 1  # the probe fails every time
    # Full rate for the first three sweeps, then only multiples of six.
    assert probed_sweeps == [1, 2, 3, 6, 12, 18]

    # A successful probe resets the address to full cadence.
    probe_failures.pop(dead, None)
    assert due(dead, 19)
