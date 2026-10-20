"""Installed export and complete closure recovery through ordinary offline stores."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import pytest
from foxlight_model_preservation.cards import (
    ArtifactFile,
    bundle_identity,
    identity_json,
)
from foxlight_model_preservation.models import PreservationError, TrustProfile
from foxlight_model_preservation.signing import create_key, load_key
from foxlight_model_preservation.storage import verify_store
from pydantic import JsonValue

import skulk.store.preservation as receiver
from skulk.download.download_utils import write_source_revision_marker
from skulk.shared.models.model_cards import ModelCard, ModelId
from skulk.store.installed_cards import (
    InstalledArtifactRole,
    build_installed_card_record,
    read_installed_card,
    verify_installed_card,
    write_installed_card,
)
from skulk.store.model_store import ModelStore
from skulk.store.preservation_cli import main
from skulk.store.preservation_export import plan_preservation, preserve_installed

Role = Literal[
    "none", "assistant", "served_draft", "mtp_sidecar", "vision_weights", "projector"
]


@dataclass(frozen=True)
class PreservationInstallation:
    primary: Path
    companion: Path | None
    archive: Path
    key: Path
    trust: TrustProfile
    card: ModelCard


def make_preservation_installation(
    root: Path, role: Role = "none", *, notices: bool = True, code: bool = False
) -> PreservationInstallation:
    primary = root / "installed" / "example--base"
    primary.mkdir(parents=True)
    gguf = role in {"none", "served_draft", "projector"}
    if gguf:
        (primary / "model.gguf").write_bytes(b"base-weight-fixture")
    else:
        (primary / "config.json").write_text('{"model_type":"llama"}')
        (primary / "model.safetensors").write_bytes(b"base-weight-fixture")
    if notices:
        (primary / "LICENSE").write_text("Synthetic test-only notice")
    body: dict[str, JsonValue] = {
        "model_id": "example/base",
        "source_revision": "a" * 40,
        "storage_size": {"in_bytes": 19},
        "n_layers": 1,
        "hidden_size": 16,
        "supports_tensor": False,
        "tasks": ["TextGeneration"],
        "is_custom": True,
        "trust_remote_code": code,
    }
    if gguf:
        body["gguf_file"] = "model.gguf"
    if role == "projector":
        (primary / "mmproj.gguf").write_bytes(b"projector-weight-fixture")
        # The fixture's declared exact projector length is independently checked.
        body["vision"] = {
            "projector_file": "mmproj.gguf",
            "projector_size": (primary / "mmproj.gguf").stat().st_size,
        }
    elif role == "vision_weights":
        body["vision"] = {
            "weights_repo": "example/companion",
            "weights_revision": "b" * 40,
            "image_token_id": 9,
        }
    elif role == "assistant":
        body["runtime"] = {
            "assistant_model_repo": "example/companion",
            "assistant_model_revision": "b" * 40,
        }
    elif role == "served_draft":
        body["runtime"] = {
            "served_spec_draft_repo": "example/companion",
            "served_spec_draft_revision": "b" * 40,
            "served_spec_draft_file": "draft.gguf",
        }
    elif role == "mtp_sidecar":
        body["runtime"] = {
            "mtp_sidecar_repo": "example/companion",
            "mtp_sidecar_revision": "b" * 40,
            "mtp_heads": True,
        }
    card = ModelCard.model_validate_json(identity_json(body))
    write_source_revision_marker(primary, card.source_revision)
    write_installed_card(primary, build_installed_card_record(primary, card))
    companion = None
    if role not in {"none", "projector"}:
        companion = primary.parent / "example--companion"
        companion.mkdir()
        if role == "served_draft":
            (companion / "draft.gguf").write_bytes(b"draft-weight-fixture")
        else:
            if role != "mtp_sidecar":
                (companion / "config.json").write_text('{"model_type":"llama"}')
            (
                companion
                / ("mtp.safetensors" if role == "mtp_sidecar" else "model.safetensors")
            ).write_bytes(b"companion-weight-fixture")
        write_source_revision_marker(companion, "b" * 40)
        roles: dict[str, InstalledArtifactRole] = {
            "assistant": "assistant",
            "served_draft": "served_draft",
            "mtp_sidecar": "mtp_sidecar",
            "vision_weights": "vision_weights",
        }
        child_role = roles[role]
        record = build_installed_card_record(
            companion,
            card,
            artifact_role=child_role,
            artifact_model_id="example/companion",
            owner_model_id="example/base",
            artifact_repository="example/companion",
            artifact_revision="b" * 40,
            artifact_format="gguf" if role == "served_draft" else "mlx",
        )
        write_installed_card(companion, record)
    archive = root / "archive"
    archive.mkdir()
    key = root / "owner.pem"
    trust = create_key(key, None)
    return PreservationInstallation(primary, companion, archive, key, trust, card)


def _preserve(installed: PreservationInstallation) -> str:
    return preserve_installed(
        installed.primary, installed.archive, (load_key(installed.key, None),)
    )


@pytest.mark.parametrize(
    "role",
    ["none", "assistant", "served_draft", "mtp_sidecar", "vision_weights", "projector"],
)
def test_complete_round_trip_and_normal_offline_companion_resolution(
    tmp_path: Path, role: Role
) -> None:
    installed = make_preservation_installation(tmp_path, role)
    plan = plan_preservation(installed.primary)
    assert plan.audience == "private" and not plan.retains_registry_proof
    assert len(plan.artifacts) == (1 if installed.companion is None else 2)
    manifest_id = _preserve(installed)
    manifest = verify_store(installed.archive, manifest_id, installed.trust)
    assert all(member.intake.audience == "private" for member in manifest.members)
    home = tmp_path / "home"
    report = receiver.recover_home(
        installed.archive,
        manifest_id,
        installed.trust,
        home,
        accept_owner_capture=True,
        allow_repository_code=installed.card.vision is not None,
    )
    assert report.members == plan.artifacts
    store = ModelStore(home / "model-store")
    assert len(store.list_models()) == len(plan.artifacts)
    for alias in plan.artifacts:
        entry = store.get_entry(alias)
        assert entry and entry.installed_card
        assert entry.installed_card.verification == "custom"
        assert verify_installed_card(
            home / "model-store" / entry.store_path,
            entry.installed_card,
            verify_hashes=True,
        )
    evidence = (
        home
        / "preservation"
        / "source-evidence"
        / "source-evidence"
        / "installed-records.json"
    )
    assert (
        evidence.is_file() and "repository_code_authorized" not in evidence.read_text()
    )
    script = """
import asyncio, json, socket
from pathlib import Path
def denied(*args, **kwargs):
    raise AssertionError('network attempted')
socket.socket.connect = denied
socket.create_connection = denied
from skulk.shared.constants import add_model_search_path, set_offline_mode
from skulk.shared.models.model_cards import ModelCard, ModelId, get_all_model_cards
from skulk.download.download_utils import model_companions_present_on_disk, build_sidecar_path, build_companion_model_path
from skulk.store.model_store import ModelStore
set_offline_mode()
add_model_search_path(Path.cwd() / 'model-store')
async def check():
    card = await ModelCard.load(ModelId('example/base'))
    assert model_companions_present_on_disk(card)
    store = ModelStore(Path.cwd() / 'model-store')
    assert store.is_in_store('example/base')
    cards = await get_all_model_cards()
    assert any(item.model_id == card.model_id for item in cards)
    assert not any(item.model_id == 'example/companion' for item in cards)
    print(json.dumps({'complete': True}))
asyncio.run(check())
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=home,
        env={
            **os.environ,
            "SKULK_HOME": str(home),
            "SKULK_MODELS_DIR": str(home / "models"),
            "SKULK_MODELS_PATH": "",
            "SKULK_OFFLINE": "true",
        },
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == {"complete": True}
    # Rebuilding the ordinary index preserves companion ownership, not new aliases.
    (home / "model-store" / "registry.json").unlink()
    assert len(ModelStore(home / "model-store").list_models()) == len(plan.artifacts)


def test_missing_companion_never_publishes_partial_copy(tmp_path: Path) -> None:
    installed = make_preservation_installation(tmp_path, "vision_weights")
    assert installed.companion
    (installed.companion / ".skulk" / "installed-card.json").unlink()
    with pytest.raises(PreservationError, match="required companions"):
        _preserve(installed)
    assert not list(installed.archive.glob("manifests/*"))


@pytest.mark.parametrize("role", ["assistant", "projector"])
def test_missing_load_bearing_file_never_publishes(tmp_path: Path, role: Role) -> None:
    installed = make_preservation_installation(tmp_path, role)
    path = installed.companion or installed.primary
    selected = "model.safetensors" if installed.companion else "mmproj.gguf"
    (path / selected).unlink()
    record = read_installed_card(path)
    assert record
    # Even a newly recorded incomplete installation cannot assert completeness.
    write_installed_card(
        path,
        record.model_copy(
            update={
                "files": tuple(file for file in record.files if file.path != selected)
            }
        ),
    )
    with pytest.raises(PreservationError, match="layout|projector"):
        _preserve(installed)
    assert not list(installed.archive.glob("manifests/*"))


def test_same_size_source_tampering_is_not_signed(tmp_path: Path) -> None:
    installed = make_preservation_installation(tmp_path)
    path = installed.primary / "model.gguf"
    path.write_bytes(b"x" * path.stat().st_size)
    with pytest.raises(PreservationError, match="recorded identity"):
        _preserve(installed)
    assert not list(installed.archive.glob("manifests/*"))
    assert not list(installed.archive.glob(".skulk-preserve-*"))


def test_corrupt_companion_blocks_whole_recovery(tmp_path: Path) -> None:
    installed = make_preservation_installation(tmp_path, "assistant")
    manifest_id = _preserve(installed)
    digest = hashlib.sha256(b"companion-weight-fixture").hexdigest()
    path = installed.archive / "objects" / "sha256" / digest
    path.write_bytes(b"x" * path.stat().st_size)
    with pytest.raises(PreservationError):
        receiver.recover_home(
            installed.archive,
            manifest_id,
            installed.trust,
            tmp_path / "home",
            accept_owner_capture=True,
        )
    assert not (tmp_path / "home").exists()


def test_late_companion_failure_is_atomic_and_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installed = make_preservation_installation(tmp_path, "assistant")
    manifest_id = _preserve(installed)
    original = receiver.validate_recovery_layout

    def fail_companion(
        directory: Path, item: receiver.RecoveryMember, owner: ModelCard
    ) -> None:
        if item.role != "base":
            raise OSError("interrupted companion admission")
        original(directory, item, owner)

    with monkeypatch.context() as patch:
        patch.setattr(receiver, "validate_recovery_layout", fail_companion)
        with pytest.raises(OSError, match="interrupted"):
            receiver.recover_home(
                installed.archive,
                manifest_id,
                installed.trust,
                tmp_path / "home",
                accept_owner_capture=True,
            )
    assert not (tmp_path / "home").exists()
    assert not list(tmp_path.glob(".skulk-recovery-*"))
    receiver.recover_home(
        installed.archive,
        manifest_id,
        installed.trust,
        tmp_path / "home",
        accept_owner_capture=True,
    )


def test_capture_notice_and_cli_without_recipe(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    installed = make_preservation_installation(tmp_path, notices=False)
    assert main(["plan", "--installed", str(installed.primary)]) == 0
    assert '"audience": "private"' in capsys.readouterr().out
    assert (
        main(
            [
                "preserve",
                "--installed",
                str(installed.primary),
                "--store",
                str(installed.archive),
                "--key",
                str(installed.key),
            ]
        )
        == 0
    )
    result = cast(JsonValue, json.loads(capsys.readouterr().out))
    assert isinstance(result, dict)
    manifest_id = result["manifest_id"]
    assert isinstance(manifest_id, str)
    manifest = verify_store(installed.archive, manifest_id, installed.trust)
    assert manifest.members[0].intake.notice_paths == (".foxlight-capture-notice.txt",)


def test_repository_code_acceptance_remains_separate(tmp_path: Path) -> None:
    installed = make_preservation_installation(tmp_path, "assistant", code=True)
    manifest_id = _preserve(installed)
    with pytest.raises(PreservationError, match="code permission"):
        receiver.recover_home(
            installed.archive,
            manifest_id,
            installed.trust,
            tmp_path / "home",
            accept_owner_capture=True,
        )
    assert not (tmp_path / "home").exists()


def test_unknown_revision_and_missing_archive_are_actionable(tmp_path: Path) -> None:
    installed = make_preservation_installation(tmp_path)
    with pytest.raises(OSError):
        preserve_installed(
            installed.primary,
            tmp_path / "disconnected" / "archive",
            (load_key(installed.key, None),),
        )
    assert not (tmp_path / "disconnected").exists()
    record = read_installed_card(installed.primary)
    assert record
    write_installed_card(
        installed.primary,
        record.model_copy(
            update={
                "artifact_revision": None,
                "model_card": record.model_card.model_copy(
                    update={"source_revision": None}
                ),
            }
        ),
    )
    with pytest.raises(PreservationError, match="exact source revision"):
        plan_preservation(installed.primary)


def _compositemake_preservation_installation(
    root: Path, kind: Literal["image", "video"]
) -> PreservationInstallation:
    installed = make_preservation_installation(root)
    body = installed.card.model_dump(mode="json")
    child = installed.primary.parent / "example--components"
    child.mkdir()
    (child / "LICENSE").write_text("Synthetic test-only notice")
    companion_path = (
        "encoder.gguf" if kind == "image" else "checkpoints/pose.safetensors"
    )
    (child / companion_path).parent.mkdir(parents=True, exist_ok=True)
    (child / companion_path).write_bytes(b"component")
    if kind == "image":
        (child / "mmproj.gguf").write_bytes(b"image-projector")
        (installed.primary / "vae.safetensors").write_bytes(b"vae")
        body["storage_size"] = {"in_bytes": 46}
        body["tasks"] = ["TextToImage", "ImageToImage"]
        body["image"] = {
            "family": "qwen_image_21",
            "default_steps": 20,
            "max_reference_images": 2,
            "components": [
                {
                    "role": role,
                    "repository": repo,
                    "revision": revision,
                    "entry_file": path,
                    "files": [
                        {
                            "path": path,
                            "size_bytes": len(data),
                            "object_id": "sha256:" + hashlib.sha256(data).hexdigest(),
                        }
                    ],
                }
                for role, repo, revision, path, data in (
                    (
                        "diffusion",
                        "example/base",
                        "a" * 40,
                        "model.gguf",
                        b"base-weight-fixture",
                    ),
                    ("vae", "example/base", "a" * 40, "vae.safetensors", b"vae"),
                    (
                        "text_encoder",
                        "example/components",
                        "b" * 40,
                        "encoder.gguf",
                        b"component",
                    ),
                    (
                        "text_encoder_vision",
                        "example/components",
                        "b" * 40,
                        "mmproj.gguf",
                        b"image-projector",
                    ),
                )
            ],
        }
    else:
        body["tasks"] = ["TextToVideo"]
        body["video"] = {
            "modes": ["t2va"],
            "companions": [
                {
                    "kind": "preprocessor",
                    "name": "pose",
                    "role": "pose_estimator",
                    "path": companion_path,
                    "repo": "example/components",
                    "revision": "b" * 40,
                    "size_bytes": 9,
                }
            ],
        }
    files = tuple(
        ArtifactFile(
            path=path.name,
            size_bytes=path.stat().st_size,
            object_id="sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in sorted(installed.primary.iterdir())
        if path.name in {"LICENSE", "model.gguf", "vae.safetensors"}
    )
    body["artifact_bundle"] = {
        "bundle_id": bundle_identity(files),
        "download_size": sum(file.size_bytes for file in files),
        "files": [file.model_dump(mode="json") for file in files],
    }
    card = ModelCard.model_validate_json(identity_json(body))
    write_installed_card(
        installed.primary, build_installed_card_record(installed.primary, card)
    )
    write_source_revision_marker(child, "b" * 40)
    write_installed_card(
        child,
        build_installed_card_record(
            child,
            card,
            artifact_role="image_component" if kind == "image" else "video_companion",
            artifact_model_id="example/components",
            owner_model_id="example/base",
            artifact_repository="example/components",
            artifact_revision="b" * 40,
            artifact_format="safetensors",
        ),
    )
    return PreservationInstallation(
        installed.primary,
        child,
        installed.archive,
        installed.key,
        installed.trust,
        card,
    )


@pytest.mark.parametrize("kind", ["image", "video"])
def test_manifest_selected_composite_closure_round_trip(
    tmp_path: Path, kind: Literal["image", "video"]
) -> None:
    installed = _compositemake_preservation_installation(tmp_path, kind)
    manifest_id = _preserve(installed)
    report = receiver.recover_home(
        installed.archive,
        manifest_id,
        installed.trust,
        tmp_path / "home",
        accept_owner_capture=True,
    )
    assert report.members == ("example/base", "example/components")
    for alias in report.members:
        entry = ModelStore(tmp_path / "home" / "model-store").get_entry(alias)
        assert entry and entry.installed_card
        assert verify_installed_card(
            tmp_path / "home" / "model-store" / entry.store_path,
            entry.installed_card,
            verify_hashes=True,
        )


def test_missing_primary_image_component_is_not_a_complete_capture(
    tmp_path: Path,
) -> None:
    installed = _compositemake_preservation_installation(tmp_path, "image")
    (installed.primary / "vae.safetensors").unlink()
    write_installed_card(
        installed.primary,
        build_installed_card_record(installed.primary, installed.card),
    )
    with pytest.raises(PreservationError, match="required primary image|omits files"):
        _preserve(installed)
    assert not list(installed.archive.glob("manifests/*"))


@pytest.mark.parametrize("role", ["vision_weights", "projector"])
def test_vision_recovery_requires_separate_code_permission_even_when_card_declines_code(
    tmp_path: Path, role: Role
) -> None:
    installed = make_preservation_installation(tmp_path, role)
    assert not installed.card.trust_remote_code
    manifest_id = _preserve(installed)
    report = receiver.inspect_recovery(installed.archive, manifest_id, installed.trust)
    assert report.requires_owner_acceptance and report.requires_code_permission
    with pytest.raises(PreservationError, match="code permission"):
        receiver.recover_home(
            installed.archive,
            manifest_id,
            installed.trust,
            tmp_path / "home",
            accept_owner_capture=True,
        )
    assert not (tmp_path / "home").exists()


@pytest.mark.parametrize("indexed", [False, True])
def test_companion_records_determine_paths_not_directory_spelling(
    tmp_path: Path, indexed: bool
) -> None:
    installed = make_preservation_installation(tmp_path, "assistant")
    assert installed.companion
    renamed = installed.companion.with_name(
        "opaque--revision-"
        + "b" * 40
        + "--source-123456789abc--bundle-123456789abcdef0"
    )
    installed.companion.rename(renamed)
    if indexed:
        store = ModelStore(installed.primary.parent, recover_installed_cards=False)
        for model_id, path in (
            ("example/base", installed.primary),
            ("example/companion", renamed),
        ):
            record = read_installed_card(path)
            assert record
            store.register_model(
                model_id,
                path,
                [file.path for file in record.files],
                sum(file.size_bytes for file in record.files),
                installed_card=record,
                source_revision=record.artifact_revision,
                source_repository=record.artifact_repository,
            )
        # A retained but inactive same-identity copy must not make the active
        # registered generation ambiguous.
        import shutil

        shutil.copytree(renamed, renamed.with_name("old-inactive-copy"))
    assert plan_preservation(installed.primary).artifacts == (
        "example/base",
        "example/companion",
    )
    manifest_id = _preserve(installed)
    assert receiver.recover_home(
        installed.archive,
        manifest_id,
        installed.trust,
        tmp_path / "home",
        accept_owner_capture=True,
    ).members == ("example/base", "example/companion")


async def test_preserve_companion_downloaded_through_normal_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio

    import skulk.download.download_utils as downloads
    from skulk.shared.types.worker.downloads import FileListEntry

    installed = make_preservation_installation(tmp_path, "assistant")
    assert installed.companion
    record = read_installed_card(installed.companion)
    assert record
    upstream = {
        file.path: (installed.companion / file.path).read_bytes()
        for file in record.files
    }
    installed.companion.rename(tmp_path / "upstream-fixture")

    async def files(
        model_id: ModelId, revision: str, recursive: bool
    ) -> list[FileListEntry]:
        assert str(model_id) == "example/companion" and revision == "b" * 40
        return [
            FileListEntry(type="file", path=path, size=len(data))
            for path, data in upstream.items()
        ]

    async def download(
        model_id: ModelId,
        revision: str,
        path: str,
        target_dir: Path,
        *_args: object,
        **_kwargs: object,
    ) -> Path:
        target = target_dir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(upstream[path])
        return target

    monkeypatch.setattr(downloads, "fetch_file_list_with_cache", files)
    monkeypatch.setattr(downloads, "download_file_with_retry", download)
    store = ModelStore(installed.primary.parent, recover_installed_cards=False)
    base_record = read_installed_card(installed.primary)
    assert base_record
    store.register_model(
        "example/base",
        installed.primary,
        [file.path for file in base_record.files],
        sum(file.size_bytes for file in base_record.files),
        source_revision=base_record.artifact_revision,
        source_repository=base_record.artifact_repository,
        installed_card=base_record,
    )
    status = await store.request_download(
        "example/companion",
        source_revision="b" * 40,
        source_repository="example/companion",
        model_card=installed.card,
        artifact_role="assistant",
        owner_model_id="example/base",
    )
    for _ in range(200):
        if status.status in {"complete", "failed", "cancelled"}:
            break
        await asyncio.sleep(0.01)
    assert status.status == "complete"
    entry = store.get_entry("example/companion")
    assert entry and "--source-" in entry.store_path
    manifest_id = await asyncio.to_thread(_preserve, installed)
    report = await asyncio.to_thread(
        receiver.recover_home,
        installed.archive,
        manifest_id,
        installed.trust,
        tmp_path / "home",
        accept_owner_capture=True,
    )
    assert report.members == ("example/base", "example/companion")
