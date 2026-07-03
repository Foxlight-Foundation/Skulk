"""Large file-body downloads must not be capped by a total wall-clock timeout.

A fixed ``total`` timeout on the ``long`` profile caps a download by elapsed
time regardless of progress, so a multi-GB GGUF that is downloading fine fails
partway through once it outlasts the cap (a 17 GB model hit the old 1800 s cap
at ~80%, surfacing as an empty-string ``TimeoutError``). The ``long`` profile
must instead rely on ``sock_read`` / ``sock_connect`` so a stalled connection
still times out while a slow-but-alive transfer of any size completes.
"""

from skulk.download.download_utils import create_http_session


async def test_long_profile_has_no_total_timeout() -> None:
    session = create_http_session(timeout_profile="long")
    try:
        assert session.timeout.total is None
        # Progress is still policed by per-read / connect inactivity timeouts.
        assert session.timeout.sock_read == 60
        assert session.timeout.sock_connect == 60
    finally:
        await session.close()


async def test_short_profile_keeps_a_total_cap() -> None:
    # Small metadata/HEAD requests should still fail fast overall.
    session = create_http_session(timeout_profile="short")
    try:
        assert session.timeout.total == 30
    finally:
        await session.close()
