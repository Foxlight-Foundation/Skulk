"""Public-only data workers; they never admit model cards or execute model code."""

from __future__ import annotations

import hashlib
import os
import shutil
import threading
import time
from collections.abc import Callable
from pathlib import Path

from foxlight_model_preservation.handoffs import write_handoff
from foxlight_model_preservation.kits import import_kit
from foxlight_model_preservation.models import Manifest, PreservationError, TrustProfile
from foxlight_model_preservation.operations import DeadlineExceeded, OperationControl
from foxlight_model_preservation.peer_runtime import PeerClient
from foxlight_model_preservation.peer_sharing import prepare_peer_handoff
from foxlight_model_preservation.seed import SeedServer
from foxlight_model_preservation.signing import digest, encode_record
from foxlight_model_preservation.storage import directory, verify_store
from foxlight_model_preservation.torrent import (
    TorrentSeeder,
    TorrentSeedPolicy,
    export_torrent,
)

from skulk.store.preservation_contracts import SharingPolicy, SharingStatus
from skulk.utils.directory_publication import publish_directory


def check_peer_space(
    archive: Path, repository: Path, required_bytes: int, *, staged: bool = False
) -> None:
    """Budget staging and native import on their actual volumes before writing.

    Combine demands on a shared device. Once staging exists, recheck the remaining
    native allocation immediately before import. Metadata/block overhead has a
    conservative allowance; this is an admission check, not a storage quota.
    """
    demands: dict[int, tuple[int, int]] = {}
    native = required_bytes + max(required_bytes // 10, 32 * 1024 * 1024)
    for path, demand in (
        (archive, (0 if staged else required_bytes) + 32 * 1024 * 1024),
        (repository, native),
    ):
        with directory(path) as descriptor:
            device = os.fstat(descriptor).st_dev
            free = shutil.disk_usage(descriptor).free
        previous, available = demands.get(device, (0, free))
        demands[device] = previous + demand, min(available, free)
    if any(required > free for required, free in demands.values()):
        raise PreservationError("insufficient_space")


class _SpaceCheckedPeerClient(PeerClient):
    """Recheck native-cache headroom after the adapter has finished staging."""

    required_bytes: int = 0

    def command(self, *arguments: str, maximum: int = 65536) -> bytes:
        """Guard the adapter's fixed import command before it reaches Kubo."""
        if arguments and arguments[0] == "add":
            check_peer_space(Path("."), self.repo, self.required_bytes, staged=True)
        return super().command(*arguments, maximum=maximum)


def require_public_bundle(manifest: Manifest) -> None:
    """Refuse any private member before an explicit mirror import or serving.

    Args:
        manifest: The caller's signature-verified selected manifest.

    Raises:
        PreservationError: A member declares a non-public audience.

    This inspects declarations only; it does not establish redistribution rights,
    registry provenance, anonymity or inference compatibility, and performs no IO.
    """
    if any(member.intake.audience != "public" for member in manifest.members):
        raise PreservationError("public_bundle_required")


def import_public_bundle(
    kit: Path,
    destination: Path,
    manifest_id: str,
    trust: TrustProfile,
    work: Path,
    control: OperationControl,
    import_control: OperationControl,
    before_publish: Callable[[str], None],
    *,
    parent_identity: tuple[int, int] | None = None,
) -> Manifest:
    """Verify a public kit before recording intent or exposing its complete copy.

    Args:
        kit: Host-selected completed ingress file.
        destination: Host-selected immutable public-copy directory.
        manifest_id: Independently reviewed exact signed identity.
        trust: Public keys accepted for this selected copy.
        work: Host-owned job staging directory on the destination filesystem.
        control: Live parent cancellation, separate from the one-use importer.
        import_control: One-use byte verification/copy progress and cancellation.
        before_publish: Durable parent intent/acknowledgement after verification.
        parent_identity: Optional host-reviewed destination-parent device/inode.

    Returns:
        The complete byte-verified public manifest.

    Corrupt data fails before publication intent. Existing exact copies are
    reverified and never replaced. New verified staging is atomically renamed;
    a late cancellation can lose to completed publication. This uses no signing
    key, runtime card parsing, model admission or network.
    """
    if destination.exists() or destination.is_symlink():
        manifest = verify_store(destination, manifest_id, trust, control=import_control)
        require_public_bundle(manifest)
        control.checkpoint()
        before_publish(manifest_id)
        return manifest
    with directory(work) as source_parent:
        verified = work / "public-copy"
        imported = import_kit(
            kit,
            verified,
            expected_manifest_id=manifest_id,
            trust=trust,
            control=import_control,
        )
        manifest = imported.manifest
        if not isinstance(manifest, Manifest):
            raise PreservationError("public PMF sharing requires a supported adapter")
        require_public_bundle(manifest)
        control.checkpoint()
        parent = os.open(
            destination.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        )
        try:
            info = os.fstat(parent)
            if (
                parent_identity is not None
                and (info.st_dev, info.st_ino) != parent_identity
            ):
                raise PreservationError("public_destination_changed")
            # Intent can now represent a real publication uncertainty. No ordinary
            # bad-object/length/signature failure consumes the uncertainty allowance.
            before_publish(manifest_id)
            control.checkpoint()
            current = destination.parent.lstat()
            staging = work.lstat()
            pinned = os.fstat(source_parent)
            if (current.st_dev, current.st_ino) != (info.st_dev, info.st_ino) or (
                staging.st_dev,
                staging.st_ino,
            ) != (pinned.st_dev, pinned.st_ino):
                raise PreservationError("public_destination_changed")
            publish_directory(verified.name, source_parent, destination.name, parent)
            control.mark_committed()
        finally:
            os.close(parent)
    return manifest


def serve_public_http(
    manifest_id: str,
    trust: TrustProfile,
    policy: SharingPolicy,
    control: OperationControl,
    observe: Callable[[SharingStatus], None],
) -> SharingStatus:
    """Own a bounded HTTP mirror until cancellation, duration or allowance ends.

    Args:
        manifest_id: Exact selected public manifest in the pinned process cwd.
        trust: Accepted public-copy verification keys.
        policy: Explicit reviewed binding, duration and total response allowance.
        control: The parent's live cancellation control; never marked committed.
        observe: Bounded safe status sink without peer addresses/history.

    Returns:
        Final stopped observation, after the listener has closed.

    Construction rehashes all selected files before binding. HTTP has a total
    response allowance, not BitTorrent's rate shaping. The caller owns the process
    and must observe its exit before confirming that all request workers stopped.
    """
    control.checkpoint()
    server = SeedServer(
        ("0.0.0.0" if policy.public_network else "127.0.0.1", policy.listen_port),
        Path("."),
        (manifest_id,),
        trust,
        public=True,
        max_upload_bytes=policy.upload_limit_bytes,
        concurrency=4,
    )
    # Hashing in the constructor can outlast a stop request. Never begin
    # accepting file requests after cancellation arrived during that check.
    try:
        control.checkpoint()
    except BaseException:
        server.server_close()
        raise
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    started = time.monotonic()
    latest = SharingStatus(accounting="reserved-response")
    thread.start()
    try:
        while True:
            latest = SharingStatus(
                ready=True,
                listen_port=server.server_port,
                charged_bytes=server.reserved_upload_bytes,
                accounting="reserved-response",
            )
            observe(latest)
            if (
                time.monotonic() - started >= policy.duration_seconds
                or latest.charged_bytes >= policy.upload_limit_bytes
            ):
                break
            control.checkpoint()
            time.sleep(0.25)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        latest = latest.model_copy(
            update={
                "ready": False,
                "listen_port": 0,
                "charged_bytes": server.reserved_upload_bytes,
            }
        )
        observe(latest)
    return latest


def serve_public_torrent(
    manifest_id: str,
    trust: TrustProfile,
    policy: SharingPolicy,
    work: Path,
    metainfo: Path,
    control: OperationControl,
    export_control: OperationControl,
    observe: Callable[[SharingStatus], None],
) -> SharingStatus:
    """Export and own one seed-only native session; receiving is never enabled.

    Args:
        manifest_id: Exact public source identity in the pinned process cwd.
        trust: Accepted exact-copy public keys.
        policy: Explicit discovery, listener and session allowances.
        work: Host-owned ephemeral metainfo/export directory parent.
        metainfo: Unique host-owned durable standard torrent artifact path.
        control: Live parent cancellation for verification and seeding.
        export_control: Separate single-use export progress/cancellation control.
        observe: Safe aggregate status sink, never peer addresses or identifiers.

    Returns:
        A stopped observation after native session closure.

    Export starts no network. Native seeding begins only after exact public-byte
    checks and metainfo binding. Public discovery is explicit, UPnP/LAN discovery
    stays disabled, and the payload stop threshold can overshoot in flight.
    """
    exported = work / "torrent"
    receipt = export_torrent(
        Path("."),
        manifest_id,
        trust,
        exported,
        control=export_control,
    )
    control.checkpoint()
    encoded = (exported / (manifest_id + ".torrent")).read_bytes()
    if hashlib.sha256(encoded).hexdigest() != receipt.metainfo_sha256:
        raise PreservationError("torrent_metadata_changed")
    descriptor = os.open(
        metainfo, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600
    )
    with os.fdopen(descriptor, "wb") as stream:
        _ = stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    parent = os.open(metainfo.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)
    observe(
        SharingStatus(
            accounting="observed-torrent-payload",
            metainfo_sha256=receipt.metainfo_sha256,
        )
    )
    control.checkpoint()
    seeder = TorrentSeeder(
        exported,
        receipt,
        trust,
        TorrentSeedPolicy(
            public_swarm=policy.public_network,
            listen_host="0.0.0.0" if policy.public_network else "127.0.0.1",
            listen_port=policy.listen_port,
            upload_rate_bytes=policy.upload_rate_bytes,
            max_seconds=policy.duration_seconds,
            stop_after_payload_bytes=policy.upload_limit_bytes,
        ),
        control=control,
    )
    latest = SharingStatus(
        accounting="observed-torrent-payload", metainfo_sha256=receipt.metainfo_sha256
    )
    try:
        while True:
            status = seeder.observe()
            latest = SharingStatus(
                metainfo_sha256=receipt.metainfo_sha256,
                ready=status.ready,
                listen_port=seeder.port,
                charged_bytes=status.uploaded_payload_bytes,
                connected_peers=status.connected_peers,
                accounting="observed-torrent-payload",
            )
            observe(latest)
            if status.stopped:
                break
            control.checkpoint()
            time.sleep(0.25)
    finally:
        seeder.close()
        status = seeder.observe()
        latest = latest.model_copy(
            update={
                "ready": False,
                "listen_port": 0,
                "connected_peers": 0,
                "charged_bytes": status.uploaded_payload_bytes,
            }
        )
        observe(latest)
    return latest


def serve_public_peer(
    manifest_id: str,
    trust: TrustProfile,
    policy: SharingPolicy,
    repository: Path,
    binary: str,
    contacts: tuple[str, ...],
    output: Path,
    control: OperationControl,
    observe: Callable[[SharingStatus], None],
) -> SharingStatus:
    """Prepare one handoff under Core's separately owned finite mirror lease.

    Core holds the native runtime gate and actual process group through drain.
    That peer worker monitors its wire threshold independently during this slow
    preparation. This data worker cannot adopt an old repo or authorize serving;
    its caller supplies the already reviewed mirror lease and public copy.
    Stopping keeps the source archive, handoff and native cache for the operator.
    """
    manifest = verify_store(Path("."), manifest_id, trust)
    check_peer_space(Path("."), repository, manifest.total_bytes)
    client = _SpaceCheckedPeerClient(
        repository,
        binary,
        allow_network=True,
        control=control.fork(),
        seconds=policy.duration_seconds,
    )
    client.required_bytes = manifest.total_bytes
    if (
        not policy.public_network
        or client.profile.mode != "mirror"
        or client.profile.allow_private_peers != policy.peer_private
    ):
        raise PreservationError("reviewed public peer mirror required")
    handoff = prepare_peer_handoff(
        Path("."),
        manifest_id,
        trust,
        client,
        public_model=True,
        bootstrap=contacts,
    )
    control.checkpoint()
    write_handoff(output, handoff)
    identity = digest(encode_record(handoff))
    latest = SharingStatus(
        ready=True, handoff_sha256=identity, accounting="observed-peer-wire"
    )
    observe(latest)
    while time.monotonic() < client.deadline:
        control.checkpoint()
        try:
            uploaded = client.wire_usage()
        except (OSError, DeadlineExceeded):
            break
        latest = latest.model_copy(update={"charged_bytes": uploaded})
        observe(latest)
        if uploaded >= policy.upload_limit_bytes:
            break
        time.sleep(0.25)
    # The data worker owns no listener. Core reports stopped only after draining
    # the actual PeerLease, never just because this client loop has returned.
    return latest
