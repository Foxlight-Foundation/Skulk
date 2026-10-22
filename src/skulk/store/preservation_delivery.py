"""Exact public publication and receive-only delivery under the durable host."""

from __future__ import annotations

import os
import shutil
import stat
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from foxlight_model_preservation.invitations import (
    Invitation,
    authenticate_invitation,
    fetch_invitation,
)
from foxlight_model_preservation.models import Manifest, PreservationError, TrustProfile
from foxlight_model_preservation.operations import OperationControl, OperationError
from foxlight_model_preservation.publication import PublicationRequest, publish_public
from foxlight_model_preservation.signing import encode_record, verify_manifest
from foxlight_model_preservation.storage import (
    atomic_bytes,
    directory,
    manifest_path,
    read_metadata,
    unique_objects,
    verify_object,
    verify_store,
)
from foxlight_model_preservation.transfer import TransferPolicy

from skulk.store.preservation_contracts import (
    InvitationPreview,
    PublicationMember,
    ReceivingPolicy,
)
from skulk.store.preservation_journal import PreservationHostError, secure_directory
from skulk.utils.directory_publication import publish_directory


def publication_members(manifest: Manifest) -> tuple[PublicationMember, ...]:
    """Describe all supported retained members without bytes or private bookkeeping.

    Args:
        manifest: Signature-authenticated private source metadata.

    Returns:
        Existing terms and notice hints for explicit publisher review. These do
        not authorize redistribution. Unknown retained cards are not narrowed.
    """
    members: list[PublicationMember] = []
    for member in manifest.members:
        if member.role == "notices" and member.member_id == "source-evidence":
            continue
        files = tuple(
            f for f in member.files if f.path != ".foxlight-capture-notice.txt"
        )
        members.append(
            PublicationMember(
                member_id=member.member_id,
                role=member.role,
                repository=member.origin_repository,
                revision=member.origin_revision,
                format=member.format,
                size_bytes=sum(f.size for f in files),
                file_count=len(files),
                license_id=member.intake.license_id,
                terms_reference=member.intake.terms_reference,
                notice_paths=tuple(
                    p
                    for p in member.intake.notice_paths
                    if p != ".foxlight-capture-notice.txt"
                ),
            )
        )
    if not 1 <= len(members) <= 16:
        raise PreservationHostError("selection_needs_paged_review")
    return tuple(members)


def invitation_preview(invitation: Invitation) -> InvitationPreview:
    """Derive a complete offline preview, without accepting the candidate keys.

    The caller uses the bounded strict invitation reader first. This performs
    candidate-signature checks only, no network, runtime card parsing or IO.
    """
    manifest = invitation.manifest
    return InvitationPreview(
        manifest_id=invitation.manifest_id,
        name=manifest.name,
        size_bytes=manifest.total_bytes,
        member_count=len(manifest.members),
        members=tuple(f"{m.member_id}: {m.role}" for m in manifest.members),
        publisher_key_ids=tuple(key.key_id for key in invitation.candidate_trust.keys),
        threshold=invitation.candidate_trust.threshold,
        mirrors=invitation.mirrors,
    )


def transfer_policy(policy: ReceivingPolicy) -> TransferPolicy:
    """Map an exact host review to the portable finite direct-fetch controls."""
    return TransferPolicy(
        max_bytes=policy.max_bytes,
        deadline_seconds=policy.deadline_seconds,
        allow_private=policy.allow_private,
        allow_http=policy.allow_http,
    )


def _parent_current(destination: Path, parent: int) -> None:
    pinned = os.fstat(parent)
    current = destination.parent.lstat()
    if not stat.S_ISDIR(current.st_mode) or (current.st_dev, current.st_ino) != (
        pinned.st_dev,
        pinned.st_ino,
    ):
        raise PreservationHostError("public_destination_changed")


@contextmanager
def _output_parent(destination: Path, expected: tuple[int, int]) -> Generator[int]:
    with directory(destination.parent) as parent:
        info = os.fstat(parent)
        if (
            (info.st_dev, info.st_ino) != expected
            or info.st_uid != os.getuid()
            or info.st_mode & 0o077
        ):
            raise PreservationHostError("public_destination_changed")
        yield parent


def _commit_copy(
    verified: Path,
    destination: Path,
    manifest_id: str,
    control: OperationControl,
    before_publish: Callable[[str], None],
    parent: int,
    source_parent: int,
) -> None:
    control.checkpoint()
    _parent_current(destination, parent)
    _parent_current(verified, source_parent)
    before_publish(manifest_id)
    control.checkpoint()
    _parent_current(destination, parent)
    _parent_current(verified, source_parent)
    publish_directory(verified.name, source_parent, destination.name, parent)
    control.mark_committed()


def publish_copy(
    source_id: str,
    source_trust: TrustProfile,
    request: PublicationRequest,
    destination: Path,
    expected_id: str,
    publishing_key: Callable[[], Ed25519PrivateKey],
    publisher_trust: TrustProfile,
    work: Path,
    control: OperationControl,
    publication_control: OperationControl,
    before_publish: Callable[[str], None],
    *,
    parent_identity: tuple[int, int],
) -> Manifest:
    """Verify and atomically retain a separately reviewed public publication.

    Source is the worker's pinned archive cwd. All paths/keys are host authored.
    The portable publisher rechecks source bytes, distinct signing keys, terms,
    notices and the exact reviewed public identity. A complete existing outcome
    is independently reverified, never replaced. No listener or model admission
    occurs; the durable parent acknowledgement precedes global publication.
    """
    with (
        _output_parent(destination, parent_identity) as parent,
        directory(work) as source_parent,
    ):
        if destination.exists() or destination.is_symlink():
            manifest = verify_store(
                destination, expected_id, publisher_trust, control=publication_control
            )
            if any(m.intake.audience != "public" for m in manifest.members):
                raise PreservationError("public_bundle_required")
            control.checkpoint()
            _parent_current(destination, parent)
            before_publish(expected_id)
            return manifest
        staged = work / "public-copy"
        identity = publish_public(
            Path("."),
            source_id,
            source_trust,
            request,
            staged,
            (publishing_key(),),
            expected_manifest_id=expected_id,
            control=publication_control,
        )
        # The portable publisher just checked every output object. Authenticate
        # its completed envelope here without an uninterruptible second hash pass.
        with directory(staged) as root:
            manifest = verify_manifest(
                read_metadata(root, manifest_path(identity)), publisher_trust, identity
            )
            atomic_bytes(root, "bundle-trust.json", encode_record(publisher_trust))
        _commit_copy(
            staged, destination, identity, control, before_publish, parent, source_parent
        )
        return manifest


def _complete_received_stage(
    staged: Path, invitation: Invitation, manifest: Manifest, control: OperationControl
) -> bool:
    # The invitation already authenticates the original envelope. If a crash
    # left every object but no completion envelope, finish offline by rehashing
    # the entire closure. Neither source availability nor old flags substitute
    # for this actual byte check.
    control.phase("checking")
    with directory(staged) as root:
        try:
            for file in unique_objects(manifest):
                verify_object(root, file, control=control)
        except OperationError:
            raise
        except (FileNotFoundError, PreservationError):
            return False
        control.checkpoint()
        atomic_bytes(
            root,
            manifest_path(invitation.manifest_id),
            encode_record(invitation.envelope),
        )
    return True


def _move_accounting(work: Path, staged: Path, *, restore: bool) -> None:
    with directory(work) as parent, directory(staged) as target:
        source_fd, name, destination_fd, destination = (
            (parent, "receive-accounting", target, ".operations")
            if restore
            else (target, ".operations", parent, "receive-accounting")
        )
        try:
            info = os.stat(name, dir_fd=source_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.getuid()
            or info.st_mode & 0o077
        ):
            raise PreservationError("unsafe_receive_accounting")
        try:
            os.stat(destination, dir_fd=destination_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise PreservationError("receive_accounting_collision")
        os.rename(name, destination, src_dir_fd=source_fd, dst_dir_fd=destination_fd)
        os.fsync(source_fd)
        os.fsync(destination_fd)


def receive_copy(
    invitation: Invitation,
    trust: TrustProfile,
    policy: ReceivingPolicy,
    destination: Path,
    work: Path,
    control: OperationControl,
    byte_control: Callable[[], OperationControl],
    before_publish: Callable[[str], None],
    *,
    parent_identity: tuple[int, int],
) -> Manifest:
    """Receive and verify a public closure, keeping retry accounting private.

    Authentication precedes networking. A fresh explicit retry uses this same
    stage inode, exact sources/policy/trust and operation ID, preserving charged
    allowance. Verified local bytes can complete with all mirrors gone. Only
    after full checks is accounting moved outside the public stage and durable
    intent acknowledged before atomic publication. Partial files remain private.
    No card interpretation, model admission, repository code or seeding occurs.
    """
    with (
        _output_parent(destination, parent_identity) as parent,
        directory(work) as source_parent,
    ):
        manifest = authenticate_invitation(invitation, trust)
        transfer_policy(policy).validate()
        if sum(file.size for file in unique_objects(manifest)) > policy.max_bytes:
            raise PreservationError("signed bundle exceeds the receiving allowance")
        if policy.mirrors != invitation.mirrors:
            raise PreservationError("receive_sources_changed")
        if destination.exists() or destination.is_symlink():
            manifest = verify_store(
                destination, invitation.manifest_id, trust, control=byte_control()
            )
            control.checkpoint()
            _parent_current(destination, parent)
            before_publish(invitation.manifest_id)
            return manifest
        staged = work / "public-copy"
        secure_directory(staged)
        _move_accounting(work, staged, restore=True)
        if not _complete_received_stage(staged, invitation, manifest, byte_control()):
            manifest = fetch_invitation(
                invitation,
                trust,
                staged,
                transfer_policy(policy),
                control=byte_control(),
                operation_id="receive",
            )
        control.checkpoint()
        # The accounting receipt survives outside the immutable public copy until
        # the host records completion. A crash/retry restores it before any fetch.
        _move_accounting(work, staged, restore=False)
        partial = staged / ".partial"
        if partial.exists():
            shutil.rmtree(partial)
        (staged / ".write.lock").unlink(missing_ok=True)
        with directory(staged) as root:
            atomic_bytes(root, "bundle-trust.json", encode_record(trust))
        _commit_copy(
            staged,
            destination,
            invitation.manifest_id,
            control,
            before_publish,
            parent,
            source_parent,
        )
        return manifest
