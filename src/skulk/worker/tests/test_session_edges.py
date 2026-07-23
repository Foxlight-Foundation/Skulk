"""Session-edge ingress and probe backoff units (#662)."""

from collections import defaultdict

from skulk_pyo3_bindings import PyFromSwarm

from skulk.routing.connection_message import ConnectionMessage
from skulk.worker.main import Worker


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


def test_session_snapshot_carries_live_connection_count() -> None:
    """The router snapshot reports the true per-peer connection count.

    A worker recreated mid-session seeds its session-edge refcounts from
    this snapshot; collapsing a multi-homed peer to one ref would let the
    first later disconnect delete the edge while other connections remain
    (PR #668 review). Exercised on a stub so the test needs no live swarm.
    """
    from skulk.routing.router import Router

    class _RouterStub:
        def __init__(self) -> None:
            self._session_connections: dict[
                str, tuple[int, tuple[str, int] | None]
            ] = {}

        _track = Router._track_session_connection  # pyright: ignore[reportPrivateUsage]
        snapshot = Router.current_session_connections

    stub = _RouterStub()

    def connect(ip: str, port: int = 52416) -> None:
        stub._track(  # pyright: ignore[reportPrivateUsage]
            ConnectionMessage.from_update(
                PyFromSwarm.Connection("12D3KooWTestPeer", True, ip, port)
            )
        )

    def disconnect() -> None:
        # Disconnect updates from the bindings carry no endpoint: an empty
        # string and port 0, which from_update maps to None.
        stub._track(  # pyright: ignore[reportPrivateUsage]
            ConnectionMessage.from_update(
                PyFromSwarm.Connection("12D3KooWTestPeer", False, "", 0)
            )
        )

    connect("100.95.14.7")
    connect("10.99.0.2")  # second, multi-homed connection
    assert stub.snapshot() == {"12D3KooWTestPeer": ("100.95.14.7", 52416, 2)}

    disconnect()
    assert stub.snapshot() == {"12D3KooWTestPeer": ("100.95.14.7", 52416, 1)}

    disconnect()
    assert stub.snapshot() == {}


def test_session_edge_multiaddr_normalizes_mapped_ipv6() -> None:
    """IPv4-mapped IPv6 endpoints become plain /ip4 multiaddrs.

    Dual-stack listeners can report an IPv4 peer as ``::ffff:a.b.c.d``; a
    dotted literal inside ``/ip6/`` fails multiaddr validation and would
    crash session-edge ingestion, and the same host must not appear under
    two address families (PR #668 review).
    """
    mapped = Worker._session_edge_multiaddr("::ffff:203.0.113.7", 52416)  # pyright: ignore[reportPrivateUsage]
    assert mapped.address == "/ip4/203.0.113.7/tcp/52416"

    plain_v6 = Worker._session_edge_multiaddr("fd00::7", 52416)  # pyright: ignore[reportPrivateUsage]
    assert plain_v6.address == "/ip6/fd00::7/tcp/52416"

    plain_v4 = Worker._session_edge_multiaddr("203.0.113.7", 52416)  # pyright: ignore[reportPrivateUsage]
    assert plain_v4.address == "/ip4/203.0.113.7/tcp/52416"


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
