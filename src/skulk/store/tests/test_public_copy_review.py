"""Original all-member terms, independent trust and metadata-only public review."""

from __future__ import annotations

import shutil
import socket
from pathlib import Path
from typing import Literal

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from foxlight_model_preservation.kits import export_kit
from foxlight_model_preservation.models import Manifest
from foxlight_model_preservation.pmf_preservation import import_preservation
from foxlight_model_preservation.signing import (
    digest,
    encode_record,
    public_record,
    sign_manifest,
)
from foxlight_model_preservation.storage import (
    atomic_bytes,
    directory,
    manifest_path,
    object_path,
    verify_store,
)

from skulk.store.preservation_contracts import PreservationCopy
from skulk.store.preservation_host import PreservationHost
from skulk.store.preservation_journal import PreservationHostError
from skulk.store.tests.test_preservation_sharing import (
    empty_host,
    import_copy,
    public_kit,
)


def all_member_kit(root: Path) -> tuple[bytes, Manifest, Ed25519PrivateKey]:
    """Build signed public primary, companion and notice-only future-card fixtures."""
    _, identity, trust = public_kit(root)
    archive = root / "archive"
    original = verify_store(archive, identity, trust)
    primary = original.members[0]
    companion = primary.model_copy(
        update={
            "member_id": "companion",
            "role": "assistant",
            "intake": primary.intake.model_copy(
                update={"license_id": "companion-terms", "terms_reference": "LICENSE"}
            ),
        }
    )
    notice = next(file for file in primary.files if file.path == "LICENSE").model_copy(
        update={"path": ".foxlight-capture-notice.txt"}
    )
    evidence = primary.model_copy(
        update={
            "member_id": "source-evidence",
            "role": "notices",
            "files": (notice,),
            "intake": primary.intake.model_copy(
                update={
                    "license_id": "notice-terms",
                    "terms_reference": ".foxlight-capture-notice.txt",
                    "notice_paths": (".foxlight-capture-notice.txt",),
                }
            ),
        }
    )
    manifest = Manifest.model_validate_json(
        original.model_copy(
            update={
                "members": (
                    primary.model_copy(
                        update={"dependencies": ("companion", "source-evidence")}
                    ),
                    companion,
                    evidence,
                )
            }
        ).model_dump_json()
    )
    key = Ed25519PrivateKey.generate()
    accepted_trust = trust.model_copy(update={"keys": (public_record(key),)})
    identity = digest(encode_record(manifest))
    envelope = encode_record(sign_manifest(manifest, (key,)))
    with directory(archive) as descriptor:
        atomic_bytes(descriptor, manifest_path(identity), envelope)
    output = root / "all-members.foxkit"
    export_kit(archive, identity, accepted_trust, output)
    return output.read_bytes(), manifest, key


def accepted_archive(host: PreservationHost, copy: PreservationCopy) -> Path:
    """Find the actual enrolled test archive instead of guessing copy filenames."""
    location = host.journal.setting("location:" + copy.location_id)
    assert isinstance(location, dict)
    path = location["path"]
    assert isinstance(path, str)
    return Path(path)


@pytest.mark.asyncio
async def test_public_review_keeps_every_signed_member_without_weights_or_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review survives origin and weight loss without asserting a fresh byte check."""
    source = tmp_path / "source"
    source.mkdir()
    content, original, _ = all_member_kit(source)
    host = empty_host(tmp_path)
    try:
        copy = await import_copy(host, content)
        # Import really verified all bytes. Subsequent review is deliberately
        # metadata-only and must not mistake the prior check for current health.
        archive = accepted_archive(host, copy)
        assert archive.is_dir()
        for member in original.members:
            for file in member.files:
                (archive / object_path(file.sha256)).unlink(missing_ok=True)
        shutil.rmtree(source)
        jobs_before = host.journal.jobs()
        checked_at = copy.checked_at

        def forbidden(*args: object, **kwargs: object) -> None:
            pytest.fail("public terms review requested network access")

        monkeypatch.setattr(socket, "getaddrinfo", forbidden)
        review = await host.public_copy_review(copy.copy_id)
        assert review.copy_id == copy.copy_id
        assert review.manifest_id == copy.manifest_id
        assert review.archive_format == "preservation-v1"
        assert review.size_bytes == original.total_bytes
        assert tuple(member.member_id for member in review.members) == (
            "primary",
            "companion",
            "source-evidence",
        )
        for actual, expected in zip(review.members, original.members, strict=True):
            assert actual.license_id == expected.intake.license_id
            assert actual.terms_reference == expected.intake.terms_reference
            assert actual.notice_paths == expected.intake.notice_paths
            assert actual.file_count == len(expected.files)
            assert actual.size_bytes == sum(file.size for file in expected.files)
        assert host.journal.jobs() == jobs_before
        assert host.journal.copy(copy.copy_id).checked_at == checked_at
        assert not host.store.list_models()
        assert str(tmp_path) not in review.model_dump_json()
    finally:
        await host.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["signature", "identity", "private"])
async def test_public_review_refuses_changed_trust_identity_and_audience(
    tmp_path: Path, change: Literal["signature", "identity", "private"]
) -> None:
    """Require both enrolled signatures, exact identity and public declarations."""
    content, manifest, key = all_member_kit(tmp_path)
    host = empty_host(tmp_path)
    try:
        copy = await import_copy(host, content)
        if change == "identity":
            manifest = manifest.model_copy(
                update={"name": "Different signed selection"}
            )
        elif change == "private":
            member = manifest.members[0]
            manifest = manifest.model_copy(
                update={
                    "members": (
                        member.model_copy(
                            update={
                                "intake": member.intake.model_copy(
                                    update={"audience": "private"}
                                )
                            }
                        ),
                    )
                }
            )
        # A new unaccepted signature cannot become trusted merely by replacing
        # metadata. Exact identity also binds fields even with a valid signature.
        identity = digest(encode_record(manifest))
        envelope = encode_record(
            sign_manifest(
                manifest,
                (Ed25519PrivateKey.generate() if change == "signature" else key,),
            )
        )
        archive = accepted_archive(host, copy)
        if change == "private":
            copy = copy.model_copy(
                update={"copy_id": "private-declaration", "manifest_id": identity}
            )
            host.journal.add_copy(copy)
        with directory(archive) as descriptor:
            atomic_bytes(descriptor, manifest_path(copy.manifest_id), envelope)
        with pytest.raises(ValueError):
            await host.public_copy_review(copy.copy_id)
    finally:
        await host.close()


@pytest.mark.asyncio
async def test_public_review_rejects_private_handle_before_archive_io(
    tmp_path: Path,
) -> None:
    """A private journal handle cannot expose metadata through the public route."""
    content, _, _ = public_kit(tmp_path)
    host = empty_host(tmp_path)
    try:
        copy = await import_copy(host, content)
        host.journal.add_copy(copy.model_copy(update={"audience": "private"}))
        shutil.rmtree(accepted_archive(host, copy))
        with pytest.raises(PreservationHostError, match="public_copy_required"):
            await host.public_copy_review(copy.copy_id)
    finally:
        await host.close()


@pytest.mark.asyncio
async def test_public_review_authenticates_pmf_source_without_flattening(
    tmp_path: Path,
) -> None:
    """Qualify the metadata adapter, not the still-unavailable public PMF intake."""
    content, original, key = all_member_kit(tmp_path)
    host = empty_host(tmp_path)
    try:
        legacy = await import_copy(host, content)
        archive = accepted_archive(host, legacy)
        trust = host.public_trust(legacy.copy_id)
        converted = tmp_path / "converted"
        conversion = import_preservation(
            archive, legacy.manifest_id, trust, converted, (key,)
        )
        # Seed an exact authenticated adapter fixture in already enrolled custody.
        # Public PMF transport/admission is a separate gate; this does not add it.
        shutil.copytree(converted, archive, dirs_exist_ok=True)
        copy = legacy.model_copy(
            update={"copy_id": "pmf-review", "manifest_id": conversion.output_identity}
        )
        host.journal.add_copy(copy)
        before = (archive / manifest_path(legacy.manifest_id)).read_bytes()
        for member in original.members:
            for file in member.files:
                (archive / object_path(file.sha256)).unlink(missing_ok=True)
        review = await host.public_copy_review(copy.copy_id)
        assert review.archive_format == "pmf-v0.1"
        assert review.manifest_id == conversion.output_identity
        assert len(review.members) == len(original.members)
        assert review.members[-1].notice_paths == (".foxlight-capture-notice.txt",)
        assert (archive / manifest_path(legacy.manifest_id)).read_bytes() == before
        source_document = archive / object_path(conversion.source_envelope_sha256)
        source_document.write_bytes(b"changed original signed envelope")
        with pytest.raises(ValueError):
            await host.public_copy_review(copy.copy_id)
    finally:
        await host.close()


@pytest.mark.asyncio
async def test_public_review_refuses_oversized_terms_without_partial_results(
    tmp_path: Path,
) -> None:
    """Large signed selections refuse instead of returning incomplete terms."""
    content, original, key = all_member_kit(tmp_path)
    host = empty_host(tmp_path)
    try:
        copy = await import_copy(host, content)
        primary = original.members[0]
        large = original.model_copy(
            update={
                "members": tuple(
                    primary.model_copy(
                        update={
                            "member_id": "member-" + str(index),
                            "dependencies": tuple(
                                "member-" + str(item) for item in range(1, 30)
                            )
                            if index == 0
                            else (),
                            "intake": primary.intake.model_copy(
                                update={"terms_reference": "x" * 2048}
                            ),
                        }
                    )
                    for index in range(30)
                ),
                "root_member": "member-0",
            }
        )
        identity = digest(encode_record(large))
        envelope = encode_record(sign_manifest(large, (key,)))
        with directory(accepted_archive(host, copy)) as descriptor:
            atomic_bytes(descriptor, manifest_path(identity), envelope)
        copy = copy.model_copy(
            update={"copy_id": "large-review", "manifest_id": identity}
        )
        host.journal.add_copy(copy)
        with pytest.raises(PreservationHostError, match="metadata_preview_too_large"):
            await host.public_copy_review(copy.copy_id)
    finally:
        await host.close()
