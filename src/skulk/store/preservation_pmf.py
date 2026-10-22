"""Native PMF adapter; runtime admission always uses independently verified source cards."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from foxlight_model_preservation.archives import verify_archive, verify_archive_manifest
from foxlight_model_preservation.kits import KitInspection, read_kit_document
from foxlight_model_preservation.models import (
    FileIdentity,
    Manifest,
    PreservationError,
    TrustProfile,
)
from foxlight_model_preservation.operations import OperationControl
from foxlight_model_preservation.pmf import restore_pmf, sign_pmf_manifest
from foxlight_model_preservation.pmf_models import PMFManifest
from foxlight_model_preservation.pmf_preservation import (
    PRESERVATION_PROFILE,
    build_preservation_pmf,
    inspect_preservation_pmf,
)
from foxlight_model_preservation.storage import (
    atomic_bytes,
    copy_verified_object,
    directory,
    manifest_path,
    object_path,
    read_metadata,
    restore,
    verify_object,
    writer_lock,
)


class PMFProfileUnavailableError(PreservationError):
    """The valid portable archive needs a compatible runtime adapter for admission."""


def source_manifest(
    manifest: Manifest | PMFManifest, source_envelope: bytes | None, trust: TrustProfile
) -> Manifest:
    """Project only a lossless supported PMF profile with independently trusted source signatures.

    Native captures use the enrolled archive keys for both signing domains.
    An imported conversion signed by different keys needs separately enrolled
    source trust; this adapter refuses it rather than substituting converter trust.
    """
    if isinstance(manifest, Manifest):
        return manifest
    if source_envelope is None or not any(
        profile.profile_id == PRESERVATION_PROFILE and profile.member_id is None
        for profile in manifest.profiles
    ):
        raise PMFProfileUnavailableError(
            "PMF has no supported original preservation profile"
        )
    return inspect_preservation_pmf(manifest, source_envelope, source_trust=trust)[1]


def read_capture(root: int, identity: str, trust: TrustProfile) -> Manifest:
    """Authenticate exact metadata from a pinned archive without hashing weights."""
    manifest = verify_archive_manifest(
        read_metadata(root, manifest_path(identity)), trust, identity
    )
    if isinstance(manifest, Manifest):
        return manifest
    document = next(
        (
            item
            for item in manifest.documents
            if item.document_id == "preservation-envelope"
        ),
        None,
    )
    if document is None:
        raise PMFProfileUnavailableError(
            "PMF has no supported original preservation profile"
        )
    return source_manifest(
        manifest, read_metadata(root, object_path(document.sha256)), trust
    )


def read_kit_capture(path: Path, info: KitInspection, trust: TrustProfile) -> Manifest:
    """Inspect signed PMF source metadata by seeking over weights in a selected kit."""
    original = (
        read_kit_document(
            path,
            "preservation-envelope",
            trust=trust,
            expected_manifest_id=info.manifest_id,
        )
        if isinstance(info.manifest, PMFManifest)
        else None
    )
    return source_manifest(info.manifest, original, trust)


def read_recovered_capture(
    evidence: Path, identity: str, trust: TrustProfile
) -> Manifest:
    """Revalidate retained PMF and source signatures at the live store commit boundary."""
    with directory(evidence) as root:
        manifest = verify_archive_manifest(
            read_metadata(root, "envelope.json"), trust, identity
        )
        original = (
            read_metadata(root, "documents/preservation-envelope")
            if isinstance(manifest, PMFManifest)
            else None
        )
        return source_manifest(manifest, original, trust)


def verify_capture(
    store: Path,
    identity: str,
    trust: TrustProfile,
    *,
    control: OperationControl | None = None,
) -> Manifest:
    """Verify all PMF/legacy objects before projecting the exact retained capture."""
    verify_archive(store, identity, trust, control=control)
    with directory(store) as root:
        return read_capture(root, identity, trust)


def restore_capture(
    store: Path, identity: str, trust: TrustProfile, destination: Path
) -> None:
    """Restore exact members plus signed evidence without granting runtime permissions."""
    with directory(store) as root:
        manifest = verify_archive_manifest(
            read_metadata(root, manifest_path(identity)), trust, identity
        )
    if isinstance(manifest, Manifest):
        restore(store, identity, trust, destination)
        return
    restore_pmf(store, identity, trust, destination)
    # This is private preparation inside the caller's unpublished recovery home.
    # Keep every PMF document, including opaque records, beside its exact envelope.
    (destination / ".pmf").rename(destination / ".preservation")
    (destination / "documents").rename(destination / ".preservation" / "documents")


def pack_capture(
    original: Manifest,
    sources: Mapping[str, Path],
    destination: Path,
    keys: tuple[Ed25519PrivateKey, ...],
    control: OperationControl,
    before_publish: Callable[[str], None] | None,
) -> str:
    """Publish a native signed PMF generation last in an existing enrolled archive.

    Capture and conversion use the same protected owner custody. All selected
    snapshot files and source records verify before durable commit intent and
    manifest publication; partial objects never constitute a complete generation.
    """
    manifest, records = build_preservation_pmf(original, keys)
    identity, envelope = sign_pmf_manifest(manifest, keys)
    if set(sources) != {member.member_id for member in original.members}:
        raise PreservationError("provide exactly the selected member sources")
    with (
        control.operation("pack") as active,
        directory(destination) as target,
        writer_lock(target),
    ):
        for member in original.members:
            with directory(sources[member.member_id]) as source:
                for file in member.files:
                    copy_verified_object(
                        source,
                        file.path,
                        target,
                        file,
                        active,
                        verify_reused_source=True,
                    )
        for document in manifest.documents:
            active.checkpoint()
            atomic_bytes(
                target, object_path(document.sha256), records[document.document_id]
            )
            verify_object(
                target,
                FileIdentity(
                    path=document.document_id,
                    size=document.size,
                    sha256=document.sha256,
                ),
                control=active,
            )
        active.checkpoint()
        if before_publish is not None:
            before_publish(identity)
        active.phase("publishing")
        atomic_bytes(target, manifest_path(identity), envelope)
        active.mark_committed()
    return identity
