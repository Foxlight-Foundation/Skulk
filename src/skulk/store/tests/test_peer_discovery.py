"""Real Core peer ownership and complete recovery without original services."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest
from foxlight_model_preservation.discovery import (
    bootstrap_identity,
    export_source,
    follow_source,
    refresh_source,
)
from foxlight_model_preservation.discovery_models import SourceBootstrap
from foxlight_model_preservation.models import TrustProfile
from foxlight_model_preservation.peer import (
    PeerSource,
    follow_peer_source,
    peer_source_identity,
    publish_peer_model,
    publish_peer_source,
)
from foxlight_model_preservation.peer_runtime import (
    PeerClient,
    executable,
    initialize_peer,
)
from foxlight_model_preservation.publication import (
    PUBLISHER_NOTICE_PATH,
    PublicationRequest,
    PublicMemberTerms,
    PublisherNotice,
)
from foxlight_model_preservation.signing import encode_record, strict_json
from foxlight_model_preservation.storage import verify_store

from skulk.store.discovery_contracts import (
    DiscoveryQuery,
    DiscoveryRequest,
    DiscoveryResults,
    SourceConfirmation,
    SourceReview,
)
from skulk.store.model_store import ModelStore
from skulk.store.peer_contracts import PeerSetup, PeerState
from skulk.store.preservation_contracts import (
    PreservationPlanRequest,
    PreservationStart,
)
from skulk.store.preservation_host import PreservationHost
from skulk.store.preservation_journal import PreservationHostError
from skulk.store.tests.test_discovery import (
    application,
    metadata_settled,
    source_tree,
    submit,
)
from skulk.store.tests.test_preservation_delivery import metadata_upload
from skulk.store.tests.test_preservation_export import make_preservation_installation
from skulk.store.tests.test_preservation_sharing import settled


@pytest.fixture
def native_binary(monkeypatch: pytest.MonkeyPatch) -> str:
    value = os.environ.get("KUBO_BINARY")
    if not value:
        pytest.skip("set KUBO_BINARY to checksum-verified Kubo 0.43.1")
    binary = executable(value)
    monkeypatch.setenv(
        "PATH", str(Path(binary).parent) + os.pathsep + os.environ["PATH"]
    )
    return binary


@contextmanager
def native_node(
    root: Path, binary: str, bootstrap: tuple[str, ...] = ()
) -> Iterator[tuple[PeerClient, str]]:
    initialize_peer(
        root, binary, mode="mirror", bootstrap=bootstrap, allow_private_peers=True
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "foxlight_model_preservation.cli",
            "peer-run",
            "--repo",
            str(root),
            "--ipfs",
            binary,
            "--allow-peer-network",
            "--share-public",
            "--seconds",
            "300",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        end = time.monotonic() + 25
        while time.monotonic() < end:
            if process.poll() is not None:
                raise AssertionError("native mirror stopped during startup")
            try:
                client = PeerClient(root, binary, allow_network=True, seconds=180)
                identity = strict_json(client.command("id"))
                if not isinstance(identity, dict):
                    raise ValueError("invalid native identity")
                addresses = identity.get("Addresses")
                if not isinstance(addresses, list):
                    raise ValueError("missing native addresses")
                contact = next(
                    value
                    for value in addresses
                    if isinstance(value, str) and value.startswith("/ip4/127.0.0.1/")
                )
                break
            except (ValueError, OSError):
                time.sleep(0.05)
        else:
            raise AssertionError("native startup deadline")
        yield client, contact
    finally:
        if process.poll() is None:
            process.send_signal(2)
            try:
                process.wait(timeout=12)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        if process.stderr:
            process.stderr.close()
        end = time.monotonic() + 10
        while (root / "api").exists() and time.monotonic() < end:
            time.sleep(0.05)
        assert not (root / "api").exists()


def test_http_request_replay_shape_and_peer_consent_are_separate() -> None:
    """Additive peer controls cannot change old HTTP replay identities."""
    http = DiscoveryRequest(
        owner_id="library", operation_id="old", kind="refresh", source_id="a" * 64
    )
    assert set(http.model_dump(mode="json")) == {
        "owner_id",
        "operation_id",
        "kind",
        "source_id",
        "entry_id",
        "manifest_id",
        "accept_source_keys",
        "allow_http",
        "allow_private",
    }
    for overrides in (
        {"transport": "peer"},
        {"transport": "peer", "allow_peer_network": True},
        {"historical": True},
        {"allow_peer_network": True},
    ):
        with pytest.raises(ValueError):
            DiscoveryRequest.model_validate_json(
                json.dumps(
                    {**http.model_dump(mode="json"), "operation_id": "bad", **overrides}
                )
            )
    assert (
        DiscoveryRequest(
            owner_id="library",
            operation_id="peer",
            kind="refresh",
            source_id="a" * 64,
            transport="peer",
            allow_peer_network=True,
            peer_setup_id="b" * 64,
        ).transport
        == "peer"
    )


@pytest.mark.asyncio
async def test_peer_setup_is_operator_gated_and_contains_no_executable_path(
    tmp_path: Path,
) -> None:
    """Settings/read/search remain local and an imported file cannot start peers."""
    host = PreservationHost(ModelStore(tmp_path / "models"), tmp_path / "core")
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=application(host)),
            base_url="http://localhost",
        ) as client:
            refused = await client.post(
                "/preservation/discovery/peer",
                json={
                    "bootstrap": ["https://gateway.invalid"],
                    "confirmed": True,
                    "binary": "/arbitrary/program",
                },
            )
            assert refused.status_code == 422
            with pytest.raises(ValueError):
                PeerSetup(
                    bootstrap=("/ip4/127.0.0.1/tcp/4001/p2p/" + "1" * 40,),
                    confirmed=True,
                )
            response = await client.get(
                "/preservation/discovery/peer",
                headers={"Origin": "https://untrusted.example"},
            )
            assert response.status_code == 403
        assert not (host.root / "peer-cache").exists()
    finally:
        await host.close()


async def public_assistant(
    root: Path,
) -> tuple[PreservationHost, Path, str, TrustProfile]:
    """Produce a complete signed public copy through ordinary Core reviewed jobs."""
    installed = make_preservation_installation(
        root / "original", "assistant", code=True
    )
    producer = PreservationHost(ModelStore(installed.primary.parent), root / "producer")
    await producer.setup_device()
    await producer.setup_publisher()
    model = (await producer.page()).models[0]
    plan = await producer.plan(
        PreservationPlanRequest(
            kind="preserve",
            location_id="device",
            model_id=model.model_id,
            generation_id=model.generation_id,
        )
    )
    kept = await settled(
        producer,
        producer.start(
            PreservationStart(plan_id=plan.plan_id, operation_id="keep")
        ).job_id,
    )
    assert kept.state == "succeeded" and kept.copy_id
    source = await producer.publication_source(kept.copy_id)
    terms = PublicationRequest(
        name="Peer community assistant",
        workloads=source.workloads,
        runtime_requirements=source.runtime_requirements,
        members=tuple(
            PublicMemberTerms(
                member_id=member.member_id,
                license_id="test-only",
                terms_reference="LICENSE",
                decision="Synthetic fixture redistribution",
                notice_paths=member.notice_paths or (PUBLISHER_NOTICE_PATH,),
            )
            for member in source.members
        ),
        notices=tuple(
            PublisherNotice(member_id=member.member_id, text="Synthetic fixture notice")
            for member in source.members
            if not member.notice_paths
        ),
    )
    upload = metadata_upload(producer, encode_record(terms), "publication")
    publish = await producer.plan(
        PreservationPlanRequest(
            kind="publish", location_id="device", copy_id=kept.copy_id, upload_id=upload
        )
    )
    job = await settled(
        producer,
        producer.start(
            PreservationStart(
                plan_id=publish.plan_id,
                operation_id="publish",
                public_participation=True,
            )
        ).job_id,
    )
    assert job.state == "succeeded" and job.copy_id
    copy = producer.journal.copy(job.copy_id)
    location = producer.journal.setting("location:" + copy.location_id)
    assert isinstance(location, dict) and isinstance(location["path"], str)
    return (
        producer,
        Path(location["path"]),
        copy.manifest_id,
        TrustProfile.model_validate_json(json.dumps(location["trust"])),
    )


@pytest.mark.asyncio
async def test_cold_peer_source_to_complete_core_restore(
    tmp_path: Path, native_binary: str
) -> None:
    """Cold native peer lookup, protected jobs, original evidence and code gate."""
    producer, archive, manifest_id, trust = await public_assistant(tmp_path)
    original = verify_store(archive, manifest_id, trust)
    with (
        native_node(tmp_path / "routing", native_binary) as (_, contact),
        native_node(tmp_path / "retainer", native_binary, (contact,)) as (retainer, _),
    ):
        with native_node(tmp_path / "original-peer", native_binary, (contact,)) as (
            publisher,
            _,
        ):
            first = publish_peer_model(
                archive, manifest_id, trust, publisher, public_model=True
            )
            second = publish_peer_model(
                archive, manifest_id, trust, retainer, public_model=True
            )
            assert first["cid"] == second["cid"]
            tree = tmp_path / "metadata-tree"
            root = source_tree(tree, manifest_id, trust, "ipfs://" + str(first["cid"]))
            local = SourceBootstrap(
                name="Community peer source",
                root=base64.b64encode(root).decode(),
                endpoints=(str(tree),),
            )
            cache = tmp_path / "publisher-cache"
            status = follow_source(
                cache, encode_record(local), bootstrap_identity(local), allow_local=True
            )
            refresh_source(cache, status.source_id)
            source = PeerSource(
                source=local.model_copy(
                    update={"endpoints": ("https://gone.invalid/source",)}
                ),
                ipns=publisher.publisher_name(),
            )
            follow_peer_source(
                cache, encode_record(source), peer_source_identity(source)
            )
            feed = publish_peer_source(
                cache, status.source_id, publisher, public_metadata=True
            )
            backup = tmp_path / "retainer-cache"
            follow_source(
                backup,
                export_source(cache, status.source_id),
                bootstrap_identity(source.source),
            )
            follow_peer_source(
                backup, encode_record(source), peer_source_identity(source)
            )
            assert (
                publish_peer_source(
                    backup,
                    status.source_id,
                    retainer,
                    public_metadata=True,
                    publish_name=False,
                )["cid"]
                == feed["cid"]
            )
        await producer.close()
        shutil.rmtree(tmp_path / "producer")
        shutil.rmtree(tmp_path / "original")
        shutil.rmtree(tmp_path / "original-peer")
        shutil.rmtree(tree)
        (tmp_path / "models").mkdir(mode=0o700)
        receiver = PreservationHost(
            ModelStore(tmp_path / "models"), tmp_path / "receiver"
        )
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=application(receiver)),
                base_url="http://localhost",
            ) as client:
                setup = await client.post(
                    "/preservation/discovery/peer",
                    json={
                        "bootstrap": [contact],
                        "allow_private": True,
                        "confirmed": True,
                    },
                )
                assert setup.status_code == 200, setup.text
                settings_id = PeerState.model_validate_json(setup.text).setup_id
                assert settings_id is not None
                encoded = encode_record(source)
                created = receiver.uploads.create(len(encoded), "source")
                upload_id = str(created["upload_id"])
                receiver.uploads.append(upload_id, 0, encoded)
                response = await client.post(
                    "/preservation/discovery/reviews", json={"upload_id": upload_id}
                )
                assert response.status_code == 200, response.text
                review = SourceReview.model_validate_json(response.text)
                assert review.transport == "peer" and not review.endpoints
                assert not (receiver.root / "peer-cache").exists()
                followed = await client.post(
                    "/preservation/discovery/sources",
                    json={"review_id": review.review_id, "confirmed": True},
                )
                assert followed.status_code == 200, followed.text
                wrong_transport = await submit(
                    client,
                    DiscoveryRequest(
                        owner_id="library-peer",
                        operation_id="wrong-transport",
                        kind="refresh",
                        source_id=review.source_id,
                    ),
                )
                rejected = await metadata_settled(client, wrong_transport.job_id)
                assert (
                    rejected.state == "failed"
                    and rejected.error_code == "source_transport_changed"
                )
                assert not (receiver.root / "peer-cache").exists()
                refresh = await submit(
                    client,
                    DiscoveryRequest(
                        owner_id="library-peer",
                        operation_id="refresh",
                        kind="refresh",
                        source_id=review.source_id,
                        transport="peer",
                        allow_peer_network=True,
                        peer_setup_id=settings_id,
                    ),
                )
                done = await metadata_settled(client, refresh.job_id)
                assert done.state == "succeeded", done
                assert not (await receiver.peers.status()).active
                found = await client.post(
                    "/preservation/discovery/search", json={"query": "offline"}
                )
                assert (
                    DiscoveryResults.model_validate_json(found.text)
                    .results[0]
                    .suppliers[0]
                    .transport
                    == "peer"
                )
                prep = await submit(
                    client,
                    DiscoveryRequest(
                        owner_id="library-peer",
                        operation_id="metadata",
                        kind="model_metadata",
                        source_id=review.source_id,
                        entry_id="community-model",
                        manifest_id=manifest_id,
                        accept_source_keys=True,
                        transport="peer",
                        allow_peer_network=True,
                        peer_setup_id=settings_id,
                    ),
                )
                metadata = await metadata_settled(client, prep.job_id)
                assert metadata.state == "succeeded" and metadata.upload_id, metadata
                preview = receiver.inspect_invitation(metadata.upload_id)
                assert preview.transport == "peer"
                with pytest.raises(
                    PreservationHostError, match="peer_network_confirmation_required"
                ):
                    await receiver.plan(
                        PreservationPlanRequest(
                            kind="receive",
                            location_id="invitation",
                            upload_id=metadata.upload_id,
                            accepted_key_ids=preview.publisher_key_ids,
                        )
                    )
                plan = await receiver.plan(
                    PreservationPlanRequest(
                        kind="receive",
                        location_id="invitation",
                        upload_id=metadata.upload_id,
                        accepted_key_ids=preview.publisher_key_ids,
                        allow_peer_network=True,
                        peer_setup_id=settings_id,
                    )
                )
                # Even a queued cancellation needs a new review of the same
                # trust/allowance, and changed contact scope cannot weaken it.
                cancelled = receiver.start(
                    PreservationStart(
                        plan_id=plan.plan_id,
                        operation_id="cancel-before-receive",
                        accept_owner_capture=True,
                    )
                )
                await receiver.cancel(cancelled.job_id)
                assert (await settled(receiver, cancelled.job_id)).state == "cancelled"
                public_scope = await receiver.peers.configure(
                    PeerSetup(
                        bootstrap=(contact.replace("127.0.0.1", "8.8.8.8"),),
                        confirmed=True,
                    )
                )
                with pytest.raises(PreservationHostError, match="peer_setup_changed"):
                    await receiver.plan(
                        PreservationPlanRequest(
                            kind="receive",
                            location_id="invitation",
                            resume_job_id=cancelled.job_id,
                            allow_peer_network=True,
                            peer_setup_id=settings_id,
                        )
                    )
                with pytest.raises(PreservationHostError, match="peer_scope_changed"):
                    await receiver.plan(
                        PreservationPlanRequest(
                            kind="receive",
                            location_id="invitation",
                            resume_job_id=cancelled.job_id,
                            allow_peer_network=True,
                            peer_setup_id=public_scope.setup_id,
                        )
                    )
                await receiver.peers.configure(
                    PeerSetup(bootstrap=(contact,), allow_private=True, confirmed=True)
                )
                with pytest.raises(
                    PreservationHostError, match="receive_resume_inputs_fixed"
                ):
                    await receiver.plan(
                        PreservationPlanRequest(
                            kind="receive",
                            location_id="invitation",
                            resume_job_id=cancelled.job_id,
                            allow_peer_network=True,
                            peer_setup_id=settings_id,
                            download_limit_bytes=plan.receiving.max_bytes + 1
                            if plan.receiving
                            else 1,
                        )
                    )
                resumed_review = await receiver.plan(
                    PreservationPlanRequest(
                        kind="receive",
                        location_id="invitation",
                        resume_job_id=cancelled.job_id,
                        allow_peer_network=True,
                        peer_setup_id=settings_id,
                    )
                )
                assert resumed_review.receiving == plan.receiving
                received = await settled(
                    receiver,
                    receiver.start(
                        PreservationStart(
                            plan_id=plan.plan_id,
                            operation_id="receive",
                            accept_owner_capture=True,
                        )
                    ).job_id,
                )
                assert received.state == "succeeded" and received.copy_id, received
                assert not (await receiver.peers.status()).active
                copy_id = received.copy_id
        finally:
            await receiver.close()
    # Every native supplier is stopped before cached search and full restore.
    reopened = PreservationHost(ModelStore(tmp_path / "models"), tmp_path / "receiver")
    try:
        assert (
            await reopened.discovery.search(DiscoveryQuery(query="offline"))
        ).results
        copy = reopened.journal.copy(copy_id)
        restore = await reopened.plan(
            PreservationPlanRequest(
                kind="recover",
                location_id=copy.location_id,
                copy_id=copy.copy_id,
                historical=True,
            )
        )
        assert restore.member_count == 2 and restore.requires_code_permission
        with pytest.raises(
            PreservationHostError, match="repository_code_permission_required"
        ):
            reopened.start(
                PreservationStart(
                    plan_id=restore.plan_id,
                    operation_id="restore",
                    accept_owner_capture=True,
                )
            )
        restored = await settled(
            reopened,
            reopened.start(
                PreservationStart(
                    plan_id=restore.plan_id,
                    operation_id="restore",
                    accept_owner_capture=True,
                    allow_repository_code=True,
                )
            ).job_id,
        )
        assert restored.state == "succeeded", restored
        from skulk.store.installed_cards import (
            read_installed_card,
            verify_installed_card,
        )
        from skulk.store.preservation_provenance import retained_recovery_card

        for member in original.members:
            if member.retained_card is None:
                continue
            path = reopened.store.get_store_path(member.origin_repository)
            assert path is not None
            record = read_installed_card(path)
            assert record is not None and verify_installed_card(
                path, record, verify_hashes=True
            )
            retained = retained_recovery_card(path, record)
            assert (
                retained is not None
                and retained.model_dump(mode="json") == member.retained_card
            )
    finally:
        await reopened.close()


@pytest.mark.asyncio
async def test_waiting_peer_job_cancels_without_stopping_another_owner(
    tmp_path: Path, native_binary: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stop drains its own lease; queued cancellation never waits for another job."""
    from skulk.utils.process_groups import terminate_process_group

    stopped: list[int] = []

    def owned_stop(process: subprocess.Popen[str]) -> None:
        assert process.returncode is None
        terminate_process_group(process)
        assert process.poll() is not None
        stopped.append(process.pid)

    monkeypatch.setattr("skulk.store.peer_host.terminate_process_group", owned_stop)
    with native_node(tmp_path / "router", native_binary) as (_, contact):
        host = PreservationHost(ModelStore(tmp_path / "models"), tmp_path / "core")
        try:
            state = await host.peers.configure(
                PeerSetup(bootstrap=(contact,), allow_private=True, confirmed=True)
            )
            assert state.setup_id
            first = await host.peers.open(state.setup_id, 60, lambda: False)
            cancelled = asyncio.Event()
            queued = asyncio.create_task(
                host.peers.open(state.setup_id, 60, cancelled.is_set)
            )
            await asyncio.sleep(0.05)
            cancelled.set()
            with pytest.raises(PreservationHostError, match="cancelled"):
                await asyncio.wait_for(queued, 2)
            assert (await host.peers.status()).active
            with pytest.raises(PreservationHostError, match="peer_busy"):
                await host.peers.configure(
                    PeerSetup(bootstrap=(contact,), allow_private=True, confirmed=True)
                )
            await first.close()
            assert not (await host.peers.status()).active
            assert not (first.repository / "api").exists()
            assert stopped == [first.process.pid]
            with pytest.raises(PreservationHostError, match="peer_setup_changed"):
                await host.peers.open("a" * 64, 60, lambda: False)
            assert not (await host.peers.status()).active
        finally:
            await host.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["receive", "mirror"])
async def test_core_caller_loss_stops_real_peer_and_restart_never_replays(
    tmp_path: Path, native_binary: str, mode: str
) -> None:
    """Kill the owning Core process and observe native EOF cleanup, not just asyncio."""
    with native_node(tmp_path / "router", native_binary) as (_, contact):
        root = tmp_path / "child-core"
        script = r"""
import asyncio,sys
from pathlib import Path
from skulk.store.discovery_contracts import DiscoveryResults
from skulk.store.model_store import ModelStore
from skulk.store.preservation_host import PreservationHost
from skulk.store.peer_contracts import PeerSetup, PeerState
async def run():
    root,contact = Path(sys.argv[1]),sys.argv[2]
    host=PreservationHost(ModelStore(root/'models'),root/'state')
    state=await host.peers.configure(PeerSetup(bootstrap=(contact,),allow_private=True,confirmed=True))
    if sys.argv[3] == 'mirror':
        lease=await host.peers.open_sharing(state.setup_id,60,1024**3,root/'state/shared-peer',lambda:False)
    else:
        lease=await host.peers.open(state.setup_id,60,lambda:False)
    print((lease.repository/'api').read_text().strip(),flush=True)
    await asyncio.sleep(120)
asyncio.run(run())
"""
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            script,
            str(root),
            contact,
            mode,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            assert process.stdout
            address = await asyncio.wait_for(process.stdout.readline(), 25)
            assert b"/ip4/127.0.0.1/tcp/" in address
            process.kill()
            await process.wait()
            api = root / (
                "state/shared-peer/api"
                if mode == "mirror"
                else "state/peer-cache/private/api"
            )
            async with asyncio.timeout(12):
                while api.exists():
                    await asyncio.sleep(0.1)
            host = PreservationHost(ModelStore(root / "models"), root / "state")
            try:
                assert not (await host.peers.status()).active
                assert not (host.root / "peer-request.json").exists()
                await asyncio.sleep(0.2)
                assert not api.exists()
            finally:
                await host.close()
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()


@pytest.mark.asyncio
async def test_active_receive_cancel_then_resume_with_replacement_contacts(
    tmp_path: Path, native_binary: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Actual stalled object transfer drains before terminal; resume binds new peers."""
    from foxlight_model_preservation.ledger import TransferReceipt
    from foxlight_model_preservation.peer_invitations import (
        peer_invitation_from_envelope,
    )

    from skulk.store.preservation_contracts import PreservationJob

    producer, archive, manifest_id, trust = await public_assistant(tmp_path)
    try:
        with native_node(tmp_path / "gone-supplier", native_binary) as (supplier, _):
            published = publish_peer_model(
                archive, manifest_id, trust, supplier, public_model=True
            )
        reference = peer_invitation_from_envelope(
            (archive / "manifests" / (manifest_id + ".json")).read_bytes(),
            manifest_id,
            trust,
            ("ipfs://" + str(published["cid"]),),
        )
        (tmp_path / "model-store").mkdir(mode=0o700)
        receiver = PreservationHost(
            ModelStore(tmp_path / "model-store"), tmp_path / "receiver"
        )
        try:
            with native_node(tmp_path / "router", native_binary) as (_, contact):
                setup = await receiver.peers.configure(
                    PeerSetup(bootstrap=(contact,), allow_private=True, confirmed=True)
                )
                upload = metadata_upload(
                    receiver, encode_record(reference), "invitation"
                )
                plan = await receiver.plan(
                    PreservationPlanRequest(
                        kind="receive",
                        location_id="invitation",
                        upload_id=upload,
                        accepted_key_ids=tuple(
                            key.key_id for key in reference.candidate_trust.keys
                        ),
                        allow_peer_network=True,
                        peer_setup_id=setup.setup_id,
                    )
                )
                original_update = receiver.journal.update_job
                terminal: list[str] = []
                receipts: list[TransferReceipt] = []

                def assert_native_stopped(job: PreservationJob) -> None:
                    if (
                        job.kind == "receive"
                        and job.expected_manifest_id is not None
                        and not receipts
                    ):
                        receipts.append(
                            TransferReceipt.model_validate_json(
                                (
                                    work / "receive-accounting" / "receive.json"
                                ).read_bytes()
                            )
                        )
                        assert staged.stat().st_ino == inode
                    if job.kind == "receive" and job.state in {
                        "failed",
                        "interrupted",
                        "cancelled",
                        "succeeded",
                    }:
                        assert not (
                            receiver.root / "peer-cache" / "private" / "api"
                        ).exists()
                        terminal.append(job.state)
                    original_update(job)

                monkeypatch.setattr(
                    receiver.journal, "update_job", assert_native_stopped
                )
                job = receiver.start(
                    PreservationStart(
                        plan_id=plan.plan_id,
                        operation_id="original",
                        accept_owner_capture=True,
                    )
                )
                work = receiver.root / "work" / job.job_id
                staged = work / "public-copy"
                ledger = staged / ".operations" / "receive.json"
                end = time.monotonic() + 30
                while not ledger.exists() and time.monotonic() < end:
                    await asyncio.sleep(0.05)
                assert ledger.exists(), receiver.journal.job(job.job_id)
                before = TransferReceipt.model_validate_json(ledger.read_bytes())
                inode = staged.stat().st_ino
                await receiver.cancel(job.job_id)
                stopped = await settled(receiver, job.job_id)
                assert stopped.state == "cancelled" and terminal == ["cancelled"]
                with native_node(
                    tmp_path / "replacement", native_binary, (contact,)
                ) as (replacement, new_contact):
                    assert (
                        publish_peer_model(
                            archive, manifest_id, trust, replacement, public_model=True
                        )["cid"]
                        == published["cid"]
                    )
                    new_setup = await receiver.peers.configure(
                        PeerSetup(
                            bootstrap=(new_contact,), allow_private=True, confirmed=True
                        )
                    )
                    assert new_setup.setup_id != setup.setup_id
                    review = await receiver.plan(
                        PreservationPlanRequest(
                            kind="receive",
                            location_id="invitation",
                            resume_job_id=job.job_id,
                            allow_peer_network=True,
                            peer_setup_id=new_setup.setup_id,
                        )
                    )
                    assert review.receiving == plan.receiving
                    resumed = await receiver.resume_receive(
                        job.job_id,
                        PreservationStart(
                            plan_id=review.plan_id,
                            operation_id="resume",
                            accept_owner_capture=True,
                        ),
                    )
                    assert (
                        resumed.job_id == job.job_id and resumed.plan_id == plan.plan_id
                    )
                    assert (
                        receiver.journal.setting("peer-attempt:" + job.job_id)
                        == new_setup.setup_id
                    )
                    finished = await settled(receiver, job.job_id)
                    assert finished.state == "succeeded" and finished.attempts == 2, (
                        finished
                    )
                    after = receipts[0]
                    assert (
                        after.max_bytes == before.max_bytes
                        and after.charged_bytes >= before.charged_bytes
                    )
                    assert terminal == ["cancelled", "succeeded"]
                    assert not staged.exists() or staged.stat().st_ino == inode
        finally:
            await receiver.close()
    finally:
        await producer.close()


@pytest.mark.asyncio
async def test_peer_unfollow_then_http_reenrollment_refuses_before_mutation(
    tmp_path: Path,
) -> None:
    """No false HTTP enrollment or verifier-floor loss for a retained peer authority."""
    from skulk.store.tests.test_preservation_sharing import public_kit

    source_path = tmp_path / "source"
    source_path.mkdir()
    _, identity, trust = public_kit(source_path)
    initial = source_tree(
        tmp_path / "metadata", identity, trust, "https://mirror.invalid/"
    )
    bootstrap = SourceBootstrap(
        name="Fixture source",
        root=base64.b64encode(initial).decode(),
        endpoints=("https://source.invalid/",),
    )
    peer = PeerSource(
        source=bootstrap,
        ipns="k51qzi5uqu5dl0ppxz7s1b5ou99zdf5ridzo2lvjti880d6t0tqyamyhwglmxf",
    )
    host = PreservationHost(ModelStore(tmp_path / "models"), tmp_path / "core")

    def source_upload(content: bytes) -> str:
        status = host.uploads.create(len(content), "source")
        upload_id = str(status["upload_id"])
        host.uploads.append(upload_id, 0, content)
        return upload_id

    try:
        reviewed = await host.discovery.review(source_upload(encode_record(peer)))
        followed = await host.discovery.follow(
            SourceConfirmation(review_id=reviewed.review_id, confirmed=True)
        )
        await host.discovery.remove(followed.source_id)
        cache = host.root / "discovery-cache"
        before = {
            str(path.relative_to(cache)): path.read_bytes()
            for path in cache.rglob("*")
            if path.is_file()
        }
        http = await host.discovery.review(source_upload(encode_record(bootstrap)))
        with pytest.raises(
            PreservationHostError, match="source_transport_switch_unavailable"
        ):
            await host.discovery.follow(
                SourceConfirmation(review_id=http.review_id, confirmed=True)
            )
        after = {
            str(path.relative_to(cache)): path.read_bytes()
            for path in cache.rglob("*")
            if path.is_file()
        }
        assert after == before
        assert not (await host.discovery.page("library", 0)).sources[0].enabled
        assert not (host.root / "peer-cache").exists()
    finally:
        await host.close()


@pytest.mark.asyncio
async def test_one_file_core_sharing_to_cold_receive_and_complete_restore(
    tmp_path: Path, native_binary: str
) -> None:
    """An exported handoff outlives its original Core without granting trust/code."""
    from foxlight_model_preservation.handoffs import read_handoff
    from foxlight_model_preservation.peer_invitations import PeerInvitation

    from skulk.store.preservation_contracts import HandoffRequest
    from skulk.store.tests.test_preservation_sharing import ready

    producer, archive, identity, trust = await public_assistant(tmp_path)
    original = verify_store(archive, identity, trust)
    copy = next(c for c in producer.journal.copies() if c.audience == "public")
    (tmp_path / "recipient-models").mkdir()
    receiver = PreservationHost(
        ModelStore(tmp_path / "recipient-models"), tmp_path / "recipient"
    )
    try:
        with (
            native_node(tmp_path / "handoff-router", native_binary) as (_, contact),
            native_node(tmp_path / "handoff-retainer", native_binary, (contact,)) as (
                retainer,
                _,
            ),
        ):
            settings = await producer.peers.configure(
                PeerSetup(bootstrap=(contact,), allow_private=True, confirmed=True)
            )
            with pytest.raises(
                PreservationHostError, match="peer_network_confirmation_required"
            ):
                await producer.plan(
                    PreservationPlanRequest(
                        kind="peer_share",
                        copy_id=copy.copy_id,
                        location_id=copy.location_id,
                        public_network=True,
                        peer_setup_id=settings.setup_id,
                    )
                )
            plan = await producer.plan(
                PreservationPlanRequest(
                    kind="peer_share",
                    copy_id=copy.copy_id,
                    location_id=copy.location_id,
                    public_network=True,
                    allow_peer_network=True,
                    peer_setup_id=settings.setup_id,
                    duration_seconds=120,
                )
            )
            with pytest.raises(
                PreservationHostError, match="public_participation_required"
            ):
                producer.start(
                    PreservationStart(plan_id=plan.plan_id, operation_id="share")
                )
            sharing = producer.start(
                PreservationStart(
                    plan_id=plan.plan_id,
                    operation_id="share",
                    public_participation=True,
                )
            )
            active = await ready(producer, sharing.job_id)
            assert active.sharing and active.sharing.handoff_sha256
            assert active.sharing.accounting == "observed-peer-wire"
            assert (await producer.peers.status()).active
            with pytest.raises(PreservationHostError, match="peer_busy_try_again"):
                await producer.peers.configure(
                    PeerSetup(bootstrap=(contact,), allow_private=True, confirmed=True)
                )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=application(producer)),
                base_url="http://localhost",
            ) as client:
                denied = await client.post(
                    f"/preservation/public-copies/{copy.copy_id}/handoff",
                    json={"sharing_job_id": sharing.job_id},
                    headers={"Origin": "https://stranger.invalid"},
                )
                assert denied.status_code == 403
                response = await client.post(
                    f"/preservation/public-copies/{copy.copy_id}/handoff",
                    json={"sharing_job_id": sharing.job_id},
                )
                assert response.status_code == 200, response.text
                assert response.headers["content-disposition"].endswith('.foxhandoff"')
                data = response.content
            handoff = read_handoff(data)
            assert isinstance(handoff.reference, PeerInvitation)
            assert handoff.reference.manifest_id == identity
            assert handoff.reference.manifest == original
            assert contact in handoff.contacts
            retained = publish_peer_model(
                archive, identity, trust, retainer, public_model=True
            )
            assert handoff.reference.mirrors == (retained["mirror"],)
            await producer.cancel(sharing.job_id)
            stopped = await settled(producer, sharing.job_id)
            assert (
                stopped.state == "cancelled"
                and stopped.sharing
                and not stopped.sharing.ready
            )
            assert not (await producer.peers.status()).active
            native_cache = producer.root / "work" / sharing.job_id / "peer"
            assert native_cache.is_dir() and not (native_cache / "api").exists()
            assert verify_store(archive, identity, trust) == original
            assert (
                await producer.public_handoff(
                    copy.copy_id, HandoffRequest(sharing_job_id=sharing.job_id)
                )
                == data
            )
            pending = await producer.temporary_work(sharing.job_id, remove=False)
            assert pending.found and pending.logical_bytes > 0
            cleaned = await producer.temporary_work(sharing.job_id, remove=True)
            assert cleaned.removed and not native_cache.exists()
            assert (
                await producer.public_handoff(
                    copy.copy_id, HandoffRequest(sharing_job_id=sharing.job_id)
                )
                == data
            )
            assert verify_store(archive, identity, trust) == original
            await producer.close()
            shutil.rmtree(tmp_path / "producer")
            shutil.rmtree(tmp_path / "original")
            # Only the same file and a separately chosen fingerprint enter the
            # recipient. Contacts remain an unsaved draft until explicit setup.
            upload = metadata_upload(receiver, data, "invitation")
            preview = receiver.inspect_invitation(upload)
            assert preview.contacts == handoff.contacts and preview.member_count == 2
            assert (await receiver.peers.status()).setup_id is None
            assert not (receiver.root / "peer-cache").exists()
            accepted = tuple(k.key_id for k in trust.keys)
            with pytest.raises(
                PreservationHostError, match="peer_network_confirmation_required"
            ):
                await receiver.plan(
                    PreservationPlanRequest(
                        kind="receive",
                        location_id="invitation",
                        upload_id=upload,
                        accepted_key_ids=accepted,
                    )
                )
            settings = await receiver.peers.configure(
                PeerSetup(
                    bootstrap=preview.contacts,
                    allow_private=True,
                    confirmed=True,
                )
            )
            receiving = await receiver.plan(
                PreservationPlanRequest(
                    kind="receive",
                    location_id="invitation",
                    upload_id=upload,
                    accepted_key_ids=accepted,
                    allow_peer_network=True,
                    peer_setup_id=settings.setup_id,
                )
            )
            received = await settled(
                receiver,
                receiver.start(
                    PreservationStart(
                        plan_id=receiving.plan_id,
                        operation_id="receive-file",
                        accept_owner_capture=True,
                    )
                ).job_id,
            )
            assert received.state == "succeeded" and received.copy_id, received
            assert not (await receiver.peers.status()).active
        retained_copy = receiver.journal.copy(received.copy_id)
        location = receiver.journal.setting("location:" + retained_copy.location_id)
        assert isinstance(location, dict) and isinstance(location["path"], str)
        assert verify_store(Path(location["path"]), identity, trust) == original
        recover = await receiver.plan(
            PreservationPlanRequest(
                kind="recover",
                location_id=retained_copy.location_id,
                copy_id=retained_copy.copy_id,
                historical=True,
            )
        )
        assert recover.member_count == 2 and recover.requires_code_permission
        with pytest.raises(
            PreservationHostError, match="repository_code_permission_required"
        ):
            receiver.start(
                PreservationStart(
                    plan_id=recover.plan_id,
                    operation_id="recover-file",
                    accept_owner_capture=True,
                )
            )
        restored = await settled(
            receiver,
            receiver.start(
                PreservationStart(
                    plan_id=recover.plan_id,
                    operation_id="recover-file",
                    accept_owner_capture=True,
                    allow_repository_code=True,
                )
            ).job_id,
        )
        assert restored.state == "succeeded", restored
        from skulk.store.installed_cards import (
            read_installed_card,
            verify_installed_card,
        )
        from skulk.store.preservation_provenance import retained_recovery_card

        for member in original.members:
            if member.retained_card is None:
                continue
            path = receiver.store.get_store_path(member.origin_repository)
            assert path is not None
            record = read_installed_card(path)
            assert record is not None and verify_installed_card(
                path, record, verify_hashes=True
            )
            evidence = retained_recovery_card(path, record)
            assert (
                evidence is not None
                and evidence.model_dump(mode="json") == member.retained_card
            )
    finally:
        await producer.close()
        await receiver.close()


@pytest.mark.asyncio
async def test_source_handoff_keeps_inner_pin_and_contacts_are_only_drafts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import socket

    from foxlight_model_preservation.handoffs import Handoff
    from multiformats import CID, multihash

    from skulk.store.tests.test_preservation_sharing import public_kit

    fixture = tmp_path / "fixture"
    fixture.mkdir()
    _, manifest, trust = public_kit(fixture)
    initial = source_tree(
        tmp_path / "metadata", manifest, trust, "https://offline.invalid/models"
    )
    source = PeerSource(
        source=SourceBootstrap(
            name="Community source",
            root=base64.b64encode(initial).decode(),
            endpoints=("https://offline.invalid/source",),
        ),
        ipns=str(
            CID("base36", 1, "libp2p-key", multihash.digest(b"synthetic", "sha2-256"))
        ),
    )
    contact = "/ip4/127.0.0.1/tcp/4001/p2p/" + "1" * 40
    encoded = encode_record(Handoff(reference=source, contacts=(contact,)))
    host = PreservationHost(ModelStore(tmp_path / "models"), tmp_path / "core")
    try:

        def forbidden(*args: object, **kwargs: object) -> None:
            raise AssertionError("review/follow must remain offline")

        monkeypatch.setattr(socket, "getaddrinfo", forbidden)
        reservation = host.uploads.create(len(encoded), "source")
        upload = str(reservation["upload_id"])
        host.uploads.append(upload, 0, encoded)
        review = await host.discovery.review(upload)
        assert review.bootstrap_id == peer_source_identity(source)
        assert review.contacts == (contact,) and review.transport == "peer"
        assert host.journal.setting("peer-setup") is None
        followed = await host.discovery.follow(
            SourceConfirmation(review_id=review.review_id, confirmed=True)
        )
        assert followed.freshness == "empty"
        assert host.journal.setting("peer-setup") is None
        assert not (host.root / "peer-cache").exists()
        saved = host.discovery.cache / "peer-sources" / (followed.source_id + ".json")
        assert saved.read_bytes() == encode_record(source)
    finally:
        await host.close()


@pytest.mark.asyncio
async def test_mirror_supervisor_loss_drains_native_group_without_waiting_for_job(
    tmp_path: Path, native_binary: str
) -> None:
    import signal
    import socket

    import psutil

    host = PreservationHost(ModelStore(tmp_path / "models"), tmp_path / "core")
    try:
        with native_node(tmp_path / "router", native_binary) as (_, contact):
            setup = await host.peers.configure(
                PeerSetup(bootstrap=(contact,), allow_private=True, confirmed=True)
            )
            assert setup.setup_id is not None
            lease = await host.peers.open_sharing(
                setup.setup_id,
                60,
                1024**3,
                host.root / "fresh-mirror",
                lambda: False,
            )
            assert (lease.repository / "api").exists()
            address = (lease.repository / "api").read_text().strip()
            port = int(address.rsplit("/", 1)[-1])
            children = psutil.Process(lease.process.pid).children(recursive=True)
            assert children
            os.kill(lease.process.pid, signal.SIGKILL)
            async with asyncio.timeout(12):
                while (await host.peers.status()).active:
                    await asyncio.sleep(0.05)
            with socket.socket() as connection:
                connection.settimeout(0.5)
                assert connection.connect_ex(("127.0.0.1", port)) != 0
            for child in children:
                assert not child.is_running() or child.status() == psutil.STATUS_ZOMBIE
            assert lease.failed()
            assert lease.repository.is_dir()
            # A forced group kill may leave Kubo's stale API marker. The socket
            # and actual descendants, not deletion of metadata, prove shutdown.
            await lease.close()
            await lease.close()
    finally:
        await host.close()
