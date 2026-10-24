"""Actual native-only Core mirrors, recipient verification and abrupt owner loss."""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
import socket
from pathlib import Path
from typing import Literal

import psutil
import pytest
from foxlight_model_preservation.handoffs import read_handoff
from foxlight_model_preservation.kits import export_kit
from foxlight_model_preservation.mirror_native import export_reference
from foxlight_model_preservation.mirror_state import MirrorLedger
from foxlight_model_preservation.peer_invitations import (
    PeerInvitation,
    fetch_peer_invitation,
)
from foxlight_model_preservation.peer_runtime import PeerClient, executable
from foxlight_model_preservation.storage import manifest_path, verify_store
from foxlight_model_preservation.torrent_binding import torrent_engine

from skulk.store.mirror_contracts import MirrorRequest
from skulk.store.peer_contracts import PeerSetup
from skulk.store.tests.test_mirror_host import complete
from skulk.store.tests.test_mirror_worker import archive, configuration
from skulk.store.tests.test_peer_discovery import native_node
from skulk.store.tests.test_preservation_sharing import empty_host, import_copy


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "transport,crash", [("torrent", False), ("ipfs", False), ("ipfs", True)]
)
async def test_native_only_core_recipient_and_owned_descendant_drain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    transport: Literal["torrent", "ipfs"],
    crash: bool,
) -> None:
    binary = None
    if transport == "ipfs":
        selected = os.environ.get("KUBO_BINARY")
        if not selected:
            pytest.skip("set KUBO_BINARY to authenticated stock Kubo 0.43.1")
        binary = executable(selected)
        monkeypatch.setenv(
            "PATH", str(Path(binary).parent) + os.pathsep + os.environ["PATH"]
        )
    else:
        pytest.importorskip("libtorrent")
    host = empty_host(tmp_path)
    occupied = socket.socket()
    occupied.bind(("127.0.0.1", 0))
    occupied.listen(1)
    receiver = handle = None
    try:
        original, identity, trust = archive(tmp_path)
        wrapper = (original / manifest_path(identity)).read_bytes()
        kit = tmp_path / "complete.foxkit"
        export_kit(original, identity, trust, kit)
        copy = await import_copy(host, kit.read_bytes())
        setup_id = None
        # The peer setup is deliberately an offline numeric hint. The mirror
        # creates its own isolated publisher and never adopts Core's receive repo.
        if binary:
            with native_node(tmp_path / "bootstrap", binary) as (_, contact):
                setup = await host.peers.configure(
                    PeerSetup(bootstrap=(contact,), allow_private=True, confirmed=True)
                )
                setup_id = setup.setup_id
        selected_config = configuration().model_copy(
            update={
                "http": False,
                "port": occupied.getsockname()[1],
                "torrent": transport == "torrent",
                "ipfs": transport == "ipfs",
                "peer_setup_id": setup_id,
                "session_seconds": 120,
                "session_bytes": 16 * 1024**2,
            }
        )
        await complete(
            host,
            MirrorRequest(
                owner_id="library",
                operation_id="configure",
                kind="configure",
                configuration=selected_config,
            ),
        )
        page = await host.mirrors.page()
        await complete(
            host,
            MirrorRequest(
                owner_id="library",
                operation_id="retain",
                kind="retain",
                policy_id=page.policy_id,
                copy_id=copy.copy_id,
                manifest_id=identity,
                confirm_redistribution=True,
            ),
        )
        shutil.rmtree(original)
        shutil.rmtree(tmp_path / "model")
        shutil.rmtree(tmp_path / "tokenizer")
        page = await host.mirrors.page()
        run = await host.mirrors.start(
            MirrorRequest(
                owner_id="library",
                operation_id="native-run",
                kind="run",
                policy_id=page.policy_id,
                public_participation=True,
                seconds=120,
            )
        )
        ledger = MirrorLedger(host.root / "managed-mirror")
        async with asyncio.timeout(40):
            while True:
                observed = host.mirrors.job(run.job_id)
                assert observed.state in {"queued", "running"}, observed
                record = ledger.read().native_models.get(identity)
                if (
                    record
                    and (record.cid if binary else record.last_torrent_port)
                    and observed.activity
                    and observed.activity.native_workers
                ):
                    break
                await asyncio.sleep(0.05)
        assert observed.activity.http_port is None and not ledger.read().http_enabled
        assert record.error is None
        destination = tmp_path / "received"
        if binary:
            handoff = read_handoff(
                await asyncio.to_thread(export_reference, ledger, identity, "peer")
            )
            assert isinstance(handoff.reference, PeerInvitation)
            reference = handoff.reference
            receiver_setup = await host.peers.configure(
                PeerSetup(bootstrap=record.contacts, allow_private=True, confirmed=True)
            )
            assert receiver_setup.setup_id is not None
            lease = await host.peers.open(receiver_setup.setup_id, 120, lambda: False)
            try:
                client = PeerClient(
                    lease.repository, lease.binary, allow_network=True, seconds=120
                )
                await asyncio.to_thread(
                    fetch_peer_invitation,
                    reference,
                    trust,
                    destination,
                    client,
                    maximum_bytes=16 * 1024**2,
                )
            finally:
                await lease.close()
        else:
            engine = torrent_engine()
            metadata = await asyncio.to_thread(
                export_reference, ledger, identity, "torrent"
            )
            destination.mkdir()
            receiver = engine.session(
                {
                    "listen_interfaces": "127.0.0.1:0",
                    "enable_dht": False,
                    "enable_lsd": False,
                    "enable_upnp": False,
                    "enable_natpmp": False,
                    "alert_mask": 0,
                }
            )
            handle = receiver.add_torrent(
                {
                    "ti": engine.torrent_info(engine.bdecode(metadata)),
                    "save_path": str(destination),
                    "flags": 0,
                }
            )
            handle.connect_peer(("127.0.0.1", record.last_torrent_port))
            async with asyncio.timeout(30):
                while not handle.status().is_seeding:
                    await asyncio.sleep(0.05)
            destination = destination / identity
        manifest = await asyncio.to_thread(verify_store, destination, identity, trust)
        assert (
            len(manifest.members) == 2
            and (destination / manifest_path(identity)).read_bytes() == wrapper
        )
        descendants: tuple[tuple[int, float], ...] = ()
        if crash:
            owned = host.mirrors._processes[run.job_id]  # pyright: ignore[reportPrivateUsage] - precise owned-process fault injection
            descendants = tuple(
                (child.pid, child.create_time())
                for child in psutil.Process(owned.process.pid).children(recursive=True)
            )
            assert descendants
            os.kill(owned.process.pid, signal.SIGKILL)
            async with asyncio.timeout(20):
                while host.mirrors.job(run.job_id).state in {
                    "queued",
                    "running",
                    "cancelling",
                }:
                    await asyncio.sleep(0.05)
            assert host.mirrors.job(run.job_id).state == "failed"
        else:
            assert (await host.mirrors.cancel(run.job_id)).state == "cancelled"
        async with asyncio.timeout(20):
            while True:
                live = False
                for identifier, started in descendants:
                    try:
                        process = psutil.Process(identifier)
                        live |= (
                            process.create_time() == started
                            and process.is_running()
                            and process.status() != psutil.STATUS_ZOMBIE
                        )
                    except psutil.NoSuchProcess:
                        continue
                if not live:
                    break
                await asyncio.sleep(0.05)
        assert (await host.mirrors.page()).activity is None and not ledger.running()
        assert verify_store(ledger.path / "store", identity, trust) == manifest
        assert ledger.read().egress_month_used >= 16 * 1024**2
    finally:
        await host.close()
        if receiver is not None:
            if handle is not None:
                receiver.remove_torrent(handle)
            receiver.pause()
        del handle
        del receiver
        occupied.close()
