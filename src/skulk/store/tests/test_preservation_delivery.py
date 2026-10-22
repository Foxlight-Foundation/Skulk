"""Real publication, recipient custody and recovery without the original Core."""

from __future__ import annotations

import asyncio
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Literal

import pytest
from foxlight_model_preservation.invitations import read_invitation
from foxlight_model_preservation.models import Manifest
from foxlight_model_preservation.publication import (
    PUBLISHER_NOTICE_PATH,
    PublicationRequest,
    PublicMemberTerms,
    PublisherNotice,
)
from foxlight_model_preservation.signing import encode_record
from foxlight_model_preservation.storage import verify_store

from skulk.store.installed_cards import read_installed_card
from skulk.store.model_store import ModelStore
from skulk.store.preservation_contracts import (
    InvitationRequest,
    PreservationPlanRequest,
    PreservationStart,
)
from skulk.store.preservation_host import PreservationHost
from skulk.store.preservation_journal import PreservationHostError
from skulk.store.tests.test_preservation_export import make_preservation_installation
from skulk.store.tests.test_preservation_sharing import ready, settled


@pytest.mark.parametrize("boundary", ["before", "after"])
def test_process_exit_at_atomic_receive_commit_can_be_confirmed_offline(
    tmp_path: Path, boundary: str
) -> None:
    """A real process exit cannot strand an empty unregistered final copy."""
    from foxlight_model_preservation.invitations import create_invitation
    from foxlight_model_preservation.operations import OperationControl

    from skulk.store.preservation_contracts import ReceivingPolicy
    from skulk.store.preservation_delivery import receive_copy
    from skulk.store.tests.test_preservation_sharing import public_kit

    source = tmp_path / "source"
    source.mkdir()
    _, identity, trust = public_kit(source)
    invitation = create_invitation(
        source / "archive", identity, trust, ("https://unavailable.invalid",)
    )
    work = tmp_path / "work"
    work.mkdir(mode=0o700)
    stage = work / "public-copy"
    shutil.copytree(source / "archive", stage)
    inode = stage.stat().st_ino
    shutil.rmtree(source)
    parent = tmp_path / "public"
    parent.mkdir(mode=0o700)
    destination = parent / "copy"
    (tmp_path / "invitation.json").write_bytes(encode_record(invitation))
    (tmp_path / "trust.json").write_bytes(encode_record(trust))
    # This child uses real verification and native exclusive rename. It exits
    # without exception unwinding immediately on either side of that boundary.
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import os
import sys
from pathlib import Path
from foxlight_model_preservation.invitations import read_invitation
from foxlight_model_preservation.models import TrustProfile
from foxlight_model_preservation.operations import OperationControl
from skulk.store.preservation_contracts import ReceivingPolicy
import skulk.store.preservation_delivery as delivery

root, boundary = Path(sys.argv[1]), sys.argv[2]
original = delivery.publish_directory
def interrupted(source, name, parent):
    if boundary == "before":
        os._exit(97)
    original(source, name, parent)
    os._exit(97)
delivery.publish_directory = interrupted
invitation = read_invitation((root / "invitation.json").read_bytes())
trust = TrustProfile.model_validate_json((root / "trust.json").read_bytes())
parent = root / "public"
info = parent.stat()
delivery.receive_copy(
    invitation, trust,
    ReceivingPolicy(mirrors=invitation.mirrors, max_bytes=1024**2, deadline_seconds=60),
    parent / "copy", root / "work", OperationControl(), OperationControl,
    lambda identity: None, parent_identity=(info.st_dev, info.st_ino),
)
raise AssertionError("crash boundary was not exercised")
""",
            str(tmp_path),
            boundary,
        ],
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert child.returncode == 97, child.stderr.decode()
    survivor = stage if boundary == "before" else destination
    assert survivor.stat().st_ino == inode
    assert verify_store(survivor, identity, trust)
    assert not (destination if boundary == "before" else stage).exists()
    info = parent.stat()
    result = receive_copy(
        invitation,
        trust,
        ReceivingPolicy(
            mirrors=invitation.mirrors, max_bytes=1024**2, deadline_seconds=60
        ),
        destination,
        work,
        OperationControl(),
        OperationControl,
        lambda identity: None,
        parent_identity=(info.st_dev, info.st_ino),
    )
    assert result == verify_store(destination, identity, trust)
    assert destination.stat().st_ino == inode and not stage.exists()


def metadata_upload(
    host: PreservationHost,
    content: bytes,
    purpose: Literal["kit", "invitation", "publication"],
) -> str:
    status = host.uploads.create(len(content), purpose)
    upload_id = str(status["upload_id"])
    host.uploads.append(upload_id, 0, content)
    return upload_id


@pytest.mark.asyncio
async def test_publish_receive_and_complete_restore_after_both_original_cores_are_gone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installed = make_preservation_installation(tmp_path, "assistant", notices=False)
    publisher = PreservationHost(
        ModelStore(installed.primary.parent), tmp_path / "publisher"
    )
    receiver = PreservationHost(
        ModelStore(tmp_path / "receiver-models"),
        tmp_path / "receiver",
        custom_cards=tmp_path / "receiver-cards",
    )
    try:
        device = await publisher.setup_device()
        public_identity = await publisher.setup_publisher()
        assert public_identity.publisher_key_id != device.publisher_key_id
        page = await publisher.page()
        assert tuple(location.location_id for location in page.locations) == ("device",)
        assert (
            page.publishing
            and page.publishing.publisher_key_id == public_identity.publisher_key_id
        )
        assert (
            await publisher.setup_publisher()
        ).publisher_key_id == public_identity.publisher_key_id
        model = page.models[0]
        plan = await publisher.plan(
            PreservationPlanRequest(
                kind="preserve",
                location_id="device",
                model_id=model.model_id,
                generation_id=model.generation_id,
            )
        )
        captured = publisher.start(
            PreservationStart(plan_id=plan.plan_id, operation_id="keep")
        )
        captured = await settled(publisher, captured.job_id)
        assert captured.state == "succeeded" and captured.copy_id
        copy = publisher.journal.copy(captured.copy_id)
        private_id = copy.manifest_id
        # Publication uses the saved archive after the original installation is gone.
        shutil.rmtree(installed.primary.parent)
        source = await publisher.publication_source(copy.copy_id)
        assert len(source.members) == 2 and all(
            not m.notice_paths for m in source.members
        )
        request = PublicationRequest(
            name="A public complete assistant",
            workloads=source.workloads,
            runtime_requirements=source.runtime_requirements,
            members=tuple(
                PublicMemberTerms(
                    member_id=m.member_id,
                    license_id="test-only",
                    terms_reference="Synthetic publisher notice",
                    decision="Synthetic redistribution choice",
                    notice_paths=(PUBLISHER_NOTICE_PATH,),
                )
                for m in source.members
            ),
            notices=tuple(
                PublisherNotice(
                    member_id=m.member_id,
                    text="Synthetic fixture license and attribution.\n",
                )
                for m in source.members
            ),
        )
        upload = metadata_upload(publisher, encode_record(request), "publication")

        def no_dns(*args: object, **kwargs: object) -> None:
            pytest.fail("offline review contacted a source")

        with monkeypatch.context() as patch:
            patch.setattr(socket, "getaddrinfo", no_dns)
            plan = await publisher.plan(
                PreservationPlanRequest(
                    kind="publish",
                    location_id="device",
                    copy_id=copy.copy_id,
                    upload_id=upload,
                )
            )
            public_metadata = Manifest.model_validate_json(
                await publisher.public_manifest(plan.plan_id)
            )
        assert plan.manifest_id != private_id
        assert all(m.member_id != "source-evidence" for m in public_metadata.members)
        with pytest.raises(
            PreservationHostError, match="public_participation_required"
        ):
            publisher.start(
                PreservationStart(plan_id=plan.plan_id, operation_id="publish")
            )
        published = publisher.start(
            PreservationStart(
                plan_id=plan.plan_id, operation_id="publish", public_participation=True
            )
        )
        published = await settled(publisher, published.job_id)
        assert published.state == "succeeded" and published.copy_id, published
        public_copy = publisher.journal.copy(published.copy_id)
        assert public_copy.audience == "public" and public_copy.member_count == 2
        (publisher.root / "custody/publisher.pem").unlink()
        retry_plan = await publisher.plan(
            PreservationPlanRequest(
                kind="publish",
                location_id="device",
                copy_id=copy.copy_id,
                upload_id=upload,
            )
        )
        retried = publisher.start(
            PreservationStart(
                plan_id=retry_plan.plan_id,
                operation_id="confirm-publication-without-private-key",
                public_participation=True,
            )
        )
        retried = await settled(publisher, retried.job_id)
        assert (
            retried.state == "succeeded" and retried.copy_id == public_copy.copy_id
        ), retried
        assert publisher.journal.copy(copy.copy_id).manifest_id == private_id
        plan = await publisher.plan(
            PreservationPlanRequest(
                kind="http_share",
                location_id=public_copy.location_id,
                copy_id=public_copy.copy_id,
                duration_seconds=60,
                upload_limit_bytes=1024**2,
            )
        )
        serving = publisher.start(
            PreservationStart(
                plan_id=plan.plan_id, operation_id="serve", public_participation=True
            )
        )
        serving = await ready(publisher, serving.job_id)
        assert serving.sharing is not None
        invitation = await publisher.public_invitation(
            public_copy.copy_id,
            InvitationRequest(
                mirrors=(f"http://127.0.0.1:{serving.sharing.listen_port}",)
            ),
        )
        assert read_invitation(invitation).manifest_id == public_copy.manifest_id
        uid = metadata_upload(receiver, invitation, "invitation")
        with monkeypatch.context() as patch:
            patch.setattr(socket, "getaddrinfo", no_dns)
            preview = receiver.inspect_invitation(uid)
        assert preview.member_count == 2 and len(preview.members) == 2
        plan = await receiver.plan(
            PreservationPlanRequest(
                kind="receive",
                location_id="invitation",
                upload_id=uid,
                accepted_key_ids=preview.publisher_key_ids,
                download_limit_bytes=1024**2,
                allow_http=True,
                allow_private=True,
            )
        )
        with pytest.raises(
            PreservationHostError, match="owner_capture_acceptance_required"
        ):
            receiver.start(
                PreservationStart(plan_id=plan.plan_id, operation_id="receive")
            )
        received = receiver.start(
            PreservationStart(
                plan_id=plan.plan_id, operation_id="receive", accept_owner_capture=True
            )
        )
        received = await settled(receiver, received.job_id)
        assert received.state == "succeeded" and received.copy_id, received
        assert not receiver.store.list_models()
        assert received.bytes_received > 0
        kept = receiver.journal.copy(received.copy_id)
        assert kept.manifest_id == public_copy.manifest_id
        public_store = next((receiver.root / "public-copies").iterdir())
        verified = verify_store(
            public_store, kept.manifest_id, receiver.public_trust(kept.copy_id)
        )
        assert len(verified.members) == 2
        from foxlight_model_preservation.signing import load_trust

        retained_public_keys = load_trust(
            (public_store / "bundle-trust.json").read_bytes()
        )
        assert retained_public_keys == receiver.public_trust(kept.copy_id)
        assert not any(p.suffix == ".pem" for p in public_store.rglob("*"))
        assert (
            not (public_store / ".operations").exists()
            and not (public_store / ".partial").exists()
        )
        await publisher.close()
        shutil.rmtree(publisher.root)
        plan = await receiver.plan(
            PreservationPlanRequest(
                kind="kit_export", location_id=kept.location_id, copy_id=kept.copy_id
            )
        )
        kit = receiver.start(
            PreservationStart(plan_id=plan.plan_id, operation_id="keep-recovery-kit")
        )
        kit = await settled(receiver, kit.job_id)
        assert kit.state == "succeeded", kit
        kit_bytes = receiver.kit_path(kit.job_id).read_bytes()
        await receiver.close()
        shutil.rmtree(receiver.root)
    finally:
        await publisher.close()
        await receiver.close()
    (tmp_path / "fresh-models").mkdir()
    fresh = PreservationHost(
        ModelStore(tmp_path / "fresh-models"),
        tmp_path / "fresh",
        custom_cards=tmp_path / "fresh-cards",
    )
    try:
        uid = metadata_upload(fresh, kit_bytes, "kit")
        plan = await fresh.plan(
            PreservationPlanRequest(
                kind="recover", location_id="kit", upload_id=uid, historical=True
            )
        )
        assert plan.member_count == 2
        recovered = fresh.start(
            PreservationStart(
                plan_id=plan.plan_id, operation_id="restore", accept_owner_capture=True
            )
        )
        recovered = await settled(fresh, recovered.job_id)
        assert recovered.state == "succeeded", recovered
        for alias in ("example/base", "example/companion"):
            path = fresh.store.get_store_path(alias)
            assert path is not None
            record = read_installed_card(path)
            assert (
                record
                and record.preservation
                and record.preservation.manifest_id == kept.manifest_id
            )
    finally:
        await fresh.close()


@pytest.mark.asyncio
async def test_explicit_receive_resume_reuses_partial_bytes_and_cumulative_allowance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading
    import time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from foxlight_model_preservation.invitations import create_invitation
    from foxlight_model_preservation.ledger import TransferReceipt
    from foxlight_model_preservation.models import Recipe, TrustProfile
    from foxlight_model_preservation.signing import public_record
    from foxlight_model_preservation.storage import pack

    from skulk.store.preservation_contracts import PreservationJob
    from skulk.store.tests.test_preservation_sharing import public_kit

    source = tmp_path / "source-owner"
    source.mkdir()
    _, small_id, small_trust = public_kit(source)
    small = verify_store(source / "archive", small_id, small_trust)
    (source / "source/weights.bin").write_bytes(b"x" * (8 * 1024**2))
    recipe = Recipe.model_validate(
        {
            **small.model_dump(exclude={"members", "attestation"}),
            "members": tuple(
                {
                    **m.model_dump(exclude={"files"}),
                    "files": tuple({"path": f.path} for f in m.files),
                }
                for m in small.members
            ),
        }
    )
    key = Ed25519PrivateKey.generate()
    trust = TrustProfile(
        schema_version="foxlight.model-preservation.trust.v1",
        keys=(public_record(key),),
        threshold=1,
        as_of="synthetic interrupted transfer",
    )
    archive = source / "large-archive"
    identity = pack(recipe, {"primary": source / "source"}, archive, (key,))
    blocked, release = threading.Event(), threading.Event()
    ranges: list[str] = []
    payloads = {
        "/" + str(p.relative_to(archive)): p.read_bytes()
        for p in archive.rglob("*")
        if p.is_file() and (p.parent.name == "sha256" or p.name == identity + ".json")
    }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            body = payloads.get(self.path)
            if body is None:
                self.send_error(404)
                return
            header = self.headers.get("Range")
            start = (
                int(header.removeprefix("bytes=").removesuffix("-")) if header else 0
            )
            if header:
                ranges.append(header)
            self.send_response(206 if header else 200)
            self.send_header("Content-Length", str(len(body) - start))
            if header:
                self.send_header(
                    "Content-Range", f"bytes {start}-{len(body) - 1}/{len(body)}"
                )
            self.end_headers()
            try:
                for offset in range(start, len(body), 65536):
                    self.wfile.write(body[offset : offset + 65536])
                    self.wfile.flush()
                    if len(body) > 1024**2 and not release.is_set():
                        if offset + 65536 >= 2 * 1024**2:
                            blocked.set()
                            release.wait(20)
                        time.sleep(0.01)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args: object, **kwargs: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root = tmp_path / "receiver"
    model_root = tmp_path / "models"
    model_root.mkdir()
    host = PreservationHost(ModelStore(model_root), root)
    try:
        invitation = create_invitation(
            archive, identity, trust, (f"http://127.0.0.1:{server.server_port}",)
        )
        uid = metadata_upload(host, encode_record(invitation), "invitation")
        limit = invitation.manifest.total_bytes + 2 * 1024**2
        plan = await host.plan(
            PreservationPlanRequest(
                kind="receive",
                location_id="invitation",
                upload_id=uid,
                accepted_key_ids=(trust.keys[0].key_id,),
                download_limit_bytes=limit,
                allow_http=True,
                allow_private=True,
            )
        )
        submission = PreservationStart(
            plan_id=plan.plan_id,
            operation_id="receive-original",
            accept_owner_capture=True,
        )
        job = host.start(submission)
        assert await asyncio.to_thread(blocked.wait, 10)
        stopping = asyncio.create_task(host.cancel(job.job_id))
        async with asyncio.timeout(5):
            while host.journal.job(job.job_id).state != "cancelling":
                await asyncio.sleep(0.01)
        release.set()
        await stopping
        stopped = await settled(host, job.job_id)
        assert stopped.state == "cancelled" and stopped.attempts == 1, stopped
        work = root / "work" / job.job_id
        staged = work / "public-copy"
        receipt = TransferReceipt.model_validate_json(
            (staged / ".operations/receive.json").read_bytes()
        )
        assert 0 < receipt.charged_bytes <= limit
        partials = list((staged / ".partial").iterdir())
        assert partials and max(p.stat().st_size for p in partials) > 0
        staged_inode = staged.stat().st_ino
        # The original upload is not needed for review or resumption.
        host.uploads.remove(uid)
        await host.close()
        host = PreservationHost(ModelStore(model_root), root)
        assert host.journal.job(job.job_id).attempts == 1
        assert host.start(submission).attempts == 1
        assert host.receive_invitation(job.job_id).manifest_id == identity
        with pytest.raises(PreservationHostError, match="receive_resume_inputs_fixed"):
            await host.plan(
                PreservationPlanRequest(
                    kind="receive",
                    location_id="invitation",
                    resume_job_id=job.job_id,
                    download_limit_bytes=limit + 1,
                )
            )
        review = await host.plan(
            PreservationPlanRequest(
                kind="receive", location_id="invitation", resume_job_id=job.job_id
            )
        )
        assert review.receiving == plan.receiving
        resumption = PreservationStart(
            plan_id=review.plan_id, operation_id="resume-one", accept_owner_capture=True
        )
        with pytest.raises(
            PreservationHostError, match="receive_resume_action_required"
        ):
            host.start(resumption)
        observed: list[TransferReceipt] = []
        original_update = host.journal.update_job

        def record_outcome(current: PreservationJob) -> None:
            if current.expected_manifest_id is not None and not observed:
                observed.append(
                    TransferReceipt.model_validate_json(
                        (work / "receive-accounting/receive.json").read_bytes()
                    )
                )
                assert staged.stat().st_ino == staged_inode
                assert not (staged / ".operations").exists()
            original_update(current)

        monkeypatch.setattr(host.journal, "update_job", record_outcome)
        resumed = await host.resume_receive(job.job_id, resumption)
        assert resumed.job_id == job.job_id and resumed.plan_id == plan.plan_id
        assert resumed.operation_id == "receive-original"
        assert (await host.resume_receive(job.job_id, resumption)).job_id == job.job_id
        finished = await settled(host, job.job_id)
        assert (
            finished.state == "succeeded"
            and finished.attempts == 2
            and finished.copy_id
        ), finished
        assert (
            len(observed) == 1
            and receipt.charged_bytes < observed[0].charged_bytes <= limit
        )
        assert observed[0].request_digest == receipt.request_digest
        assert any(h != "bytes=0-" for h in ranges)
        kept = host.journal.copy(finished.copy_id)
        completed = next((root / "public-copies").iterdir())
        result = verify_store(completed, identity, host.public_trust(kept.copy_id))
        assert result.members[0].retained_card == small.members[0].retained_card
        assert not host.store.list_models()
        assert not work.exists()
        assert (await host.resume_receive(job.job_id, resumption)).state == "succeeded"
        assert host.start(submission).attempts == 2
    finally:
        release.set()
        await host.close()
        server.shutdown()
        server.server_close()
        thread.join(5)
        assert not thread.is_alive()


@pytest.mark.parametrize("replace_parent", [False, True])
def test_receive_finishes_verified_local_objects_offline_and_fences_changed_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replace_parent: bool
) -> None:
    from foxlight_model_preservation.invitations import create_invitation
    from foxlight_model_preservation.operations import OperationControl
    from foxlight_model_preservation.storage import manifest_path

    from skulk.store.preservation_contracts import ReceivingPolicy
    from skulk.store.preservation_delivery import receive_copy
    from skulk.store.tests.test_preservation_sharing import public_kit

    source = tmp_path / "source"
    source.mkdir()
    _, identity, trust = public_kit(source)
    invitation = create_invitation(
        source / "archive", identity, trust, ("https://gone.invalid",)
    )
    work = tmp_path / "work"
    work.mkdir(mode=0o700)
    stage = work / "public-copy"
    shutil.copytree(source / "archive", stage)
    (stage / manifest_path(identity)).unlink()
    shutil.rmtree(source)
    parent = tmp_path / "public"
    parent.mkdir(mode=0o700)
    info = parent.stat()
    destination = parent / "copy"
    intents: list[str] = []

    def before_publish(manifest_id: str) -> None:
        intents.append(manifest_id)
        if replace_parent:
            parent.rename(tmp_path / "disconnected-parent")
            parent.mkdir(mode=0o700)

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("complete local files required a gone mirror")

    monkeypatch.setattr(socket, "getaddrinfo", forbidden)

    def run() -> Manifest:
        return receive_copy(
            invitation,
            trust,
            ReceivingPolicy(
                mirrors=invitation.mirrors, max_bytes=1024**2, deadline_seconds=60
            ),
            destination,
            work,
            OperationControl(),
            OperationControl,
            before_publish,
            parent_identity=(info.st_dev, info.st_ino),
        )

    if replace_parent:
        with pytest.raises(PreservationHostError, match="public_destination_changed"):
            run()
        assert list(parent.iterdir()) == []
        assert list((tmp_path / "disconnected-parent").iterdir()) == []
        assert verify_store(stage, identity, trust)
    else:
        assert run() == verify_store(destination, identity, trust)
        assert not stage.exists()
        assert not (destination / ".operations").exists()
    assert intents == [identity]


@pytest.mark.asyncio
async def test_failed_public_parent_check_still_leaves_authenticated_reclaimable_work(
    tmp_path: Path,
) -> None:
    installed = make_preservation_installation(tmp_path)
    host = PreservationHost(ModelStore(installed.primary.parent), tmp_path / "host")
    try:
        await host.setup_device()
        await host.setup_publisher()
        model = (await host.page()).models[0]
        plan = await host.plan(
            PreservationPlanRequest(
                kind="preserve",
                location_id="device",
                model_id=model.model_id,
                generation_id=model.generation_id,
            )
        )
        kept = host.start(PreservationStart(plan_id=plan.plan_id, operation_id="keep"))
        kept = await settled(host, kept.job_id)
        assert kept.copy_id
        source = await host.publication_source(kept.copy_id)
        request = PublicationRequest(
            name="Public fixture",
            workloads=source.workloads,
            members=tuple(
                PublicMemberTerms(
                    member_id=m.member_id,
                    license_id="test-only",
                    terms_reference="LICENSE",
                    decision="Synthetic fixture public review",
                    notice_paths=m.notice_paths,
                )
                for m in source.members
            ),
        )
        uid = metadata_upload(host, encode_record(request), "publication")
        plan = await host.plan(
            PreservationPlanRequest(
                kind="publish",
                location_id="device",
                copy_id=kept.copy_id,
                upload_id=uid,
            )
        )
        parent = host.root / "public-copies"
        parent.rename(host.root / "disconnected-public")
        parent.mkdir(mode=0o700)
        started = host.start(
            PreservationStart(
                plan_id=plan.plan_id, operation_id="publish", public_participation=True
            )
        )
        stopped = await settled(host, started.job_id)
        assert stopped.state == "failed" and stopped.expected_manifest_id is None
        assert not (await host.page()).public_copies and list(parent.iterdir()) == []
        inspection = await host.temporary_work(started.job_id)
        assert inspection.found and inspection.logical_bytes == 0
        cleanup = await host.temporary_work(started.job_id, remove=True)
        assert cleanup.removed and not (host.root / "work" / started.job_id).exists()
        assert (host.root / "custody/publisher.pem").is_file()
        assert host.journal.copy(kept.copy_id).audience == "private"
    finally:
        await host.close()


@pytest.mark.asyncio
async def test_receive_deadline_stops_a_stalled_mirror_and_cleared_work_never_refills(
    tmp_path: Path,
) -> None:
    import threading
    import time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from foxlight_model_preservation.invitations import create_invitation

    from skulk.store.tests.test_preservation_sharing import empty_host, public_kit

    source = tmp_path / "publisher"
    source.mkdir()
    _, identity, trust = public_kit(source)
    release = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            release.wait(20)

        def log_message(self, *args: object, **kwargs: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host = empty_host(tmp_path)
    try:
        invitation = create_invitation(
            source / "archive",
            identity,
            trust,
            (f"http://127.0.0.1:{server.server_port}",),
        )
        uid = metadata_upload(host, encode_record(invitation), "invitation")
        plan = await host.plan(
            PreservationPlanRequest(
                kind="receive",
                location_id="invitation",
                upload_id=uid,
                accepted_key_ids=(trust.keys[0].key_id,),
                download_deadline_seconds=5,
                allow_http=True,
                allow_private=True,
            )
        )
        submission = PreservationStart(
            plan_id=plan.plan_id, operation_id="stalled", accept_owner_capture=True
        )
        started = time.monotonic()
        job = host.start(submission)
        stopped = await settled(host, job.job_id)
        assert (
            stopped.state == "failed" and stopped.error_code == "deadline_exceeded"
        ), stopped
        assert time.monotonic() - started < 12
        assert (
            stopped.expected_manifest_id is None
            and not (await host.page()).public_copies
        )
        # Inspection is refused until both task and process ownership have drained.
        inspection = await host.temporary_work(job.job_id)
        assert inspection.found
        await host.temporary_work(job.job_id, remove=True)
        review = await host.plan(
            PreservationPlanRequest(
                kind="receive", location_id="invitation", resume_job_id=job.job_id
            )
        )
        with pytest.raises(PreservationHostError, match="receive_work_unavailable"):
            await host.resume_receive(
                job.job_id,
                PreservationStart(
                    plan_id=review.plan_id,
                    operation_id="resume-cleared",
                    accept_owner_capture=True,
                ),
            )
        assert not (host.root / "work" / job.job_id).exists()
        assert host.start(submission).attempts == 1
    finally:
        release.set()
        await host.close()
        server.shutdown()
        server.server_close()
        thread.join(5)
        assert not thread.is_alive()
