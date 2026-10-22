"""PMF native recovery retains evidence and existing admission boundaries."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from foxlight_model_preservation.archives import verify_archive
from foxlight_model_preservation.models import PreservationError
from foxlight_model_preservation.pmf_models import PMFManifest
from foxlight_model_preservation.signing import load_key
from foxlight_model_preservation.storage import object_path

import skulk.store.preservation as receiver
from skulk.store.installed_cards import read_installed_card, verify_installed_card
from skulk.store.model_store import ModelStore
from skulk.store.preservation_export import preserve_installed
from skulk.store.preservation_pmf import read_recovered_capture, verify_capture
from skulk.store.preservation_provenance import retained_recovery_card
from skulk.store.tests.test_preservation_export import (
    Role,
    make_preservation_installation,
)


@pytest.mark.parametrize(
    "role",
    ["none", "assistant", "served_draft", "mtp_sidecar", "vision_weights", "projector"],
)
def test_native_pmf_survives_original_loss_and_recapture(
    tmp_path: Path, role: Role
) -> None:
    original = tmp_path / "original"
    original.mkdir()
    installed = make_preservation_installation(original, role)
    key = load_key(installed.key, None)
    archive = tmp_path / "portable"
    archive.mkdir()
    identity = preserve_installed(installed.primary, archive, (key,), portable=True)
    manifest = verify_archive(archive, identity, installed.trust)
    assert isinstance(manifest, PMFManifest)
    source = verify_capture(archive, identity, installed.trust)
    retained = source.members[0].retained_card
    shutil.rmtree(original)
    home = tmp_path / "home"
    report = receiver.recover_home(
        archive,
        identity,
        installed.trust,
        home,
        accept_owner_capture=True,
        allow_repository_code=role in {"vision_weights", "projector"},
    )
    assert report.manifest_id == identity
    assert len(report.members) == (1 if role in {"none", "projector"} else 2)
    store = ModelStore(home / "model-store")
    for alias in report.members:
        path = store.get_store_path(alias)
        assert path is not None
        record = read_installed_card(path)
        assert record is not None and record.preservation is not None
        assert record.preservation.manifest_id == identity
        assert verify_installed_card(path, record, verify_hashes=True)
    primary = store.get_store_path("example/base")
    assert primary is not None
    record = read_installed_card(primary)
    assert record is not None
    reused = retained_recovery_card(primary, record)
    assert reused is not None and reused.model_dump(mode="json") == retained
    next_archive = tmp_path / "next-archive"
    next_archive.mkdir()
    again = preserve_installed(
        primary, next_archive, (key,), artifact_roots=(store.store_path,), portable=True
    )
    assert (
        verify_capture(next_archive, again, installed.trust).members[0].retained_card
        == retained
    )
    assert (
        read_recovered_capture(home / "preservation", identity, installed.trust)
        == source
    )


def test_corrupt_pmf_document_never_publishes_recovery(tmp_path: Path) -> None:
    installed = make_preservation_installation(tmp_path)
    identity = preserve_installed(
        installed.primary,
        installed.archive,
        (load_key(installed.key, None),),
        portable=True,
    )
    manifest = verify_archive(installed.archive, identity, installed.trust)
    assert isinstance(manifest, PMFManifest)
    document = manifest.documents[0]
    path = installed.archive / object_path(document.sha256)
    path.write_bytes(b"x" * document.size)
    with pytest.raises(PreservationError):
        receiver.recover_home(
            installed.archive,
            identity,
            installed.trust,
            tmp_path / "home",
            accept_owner_capture=True,
        )
    assert not (tmp_path / "home").exists()


def test_generic_pmf_keeps_standalone_recovery_without_runtime_admission(
    tmp_path: Path,
) -> None:
    from foxlight_model_preservation.pmf import restore_pmf, sign_pmf_manifest
    from foxlight_model_preservation.storage import (
        atomic_bytes,
        directory,
        manifest_path,
    )

    from skulk.store.preservation_pmf import PMFProfileUnavailableError

    installed = make_preservation_installation(tmp_path)
    key = load_key(installed.key, None)
    identity = preserve_installed(
        installed.primary, installed.archive, (key,), portable=True
    )
    original = verify_archive(installed.archive, identity, installed.trust)
    assert isinstance(original, PMFManifest)
    generic = original.model_copy(
        update={"documents": (), "profiles": (), "annotations": {}}
    )
    identity, envelope = sign_pmf_manifest(generic, (key,))
    with directory(installed.archive) as root:
        atomic_bytes(root, manifest_path(identity), envelope)
    assert verify_archive(installed.archive, identity, installed.trust) == generic
    with pytest.raises(PMFProfileUnavailableError):
        receiver.recover_home(
            installed.archive,
            identity,
            installed.trust,
            tmp_path / "runtime",
            accept_owner_capture=True,
        )
    assert not (tmp_path / "runtime").exists()
    restore_pmf(installed.archive, identity, installed.trust, tmp_path / "files")
    assert (tmp_path / "files/members/primary/model.gguf").is_file()
