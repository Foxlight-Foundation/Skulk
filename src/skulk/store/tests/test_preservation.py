"""Receiver trust, atomic publication and ordinary offline catalog recovery."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, tzinfo
from pathlib import Path
from typing import Literal, Self, cast

import foxlight_model_preservation.registry_proof as proof_verifier
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from foxlight_model_preservation.assurance import retain_card
from foxlight_model_preservation.cards import (
    ArtifactFile,
    ModelCardEnvelope,
    bundle_identity,
    create_card,
    identity_json,
)
from foxlight_model_preservation.models import (
    Intake,
    Manifest,
    PreservationError,
    Recipe,
    RecipeMember,
    SourceFile,
    TrustProfile,
)
from foxlight_model_preservation.registry_proof import (
    RegistryProof,
    create_registry_proof,
)
from foxlight_model_preservation.signing import (
    create_key,
    load_key,
)
from foxlight_model_preservation.storage import pack, restore
from pydantic import JsonValue

import skulk.download.download_utils as download_utils
import skulk.store.preservation as preservation
from skulk.shared.models.registry import EMBEDDED_REGISTRY_ROOT
from skulk.shared.types.worker.downloads import FileListEntry
from skulk.store.installed_cards import build_installed_card_record, read_installed_card
from skulk.store.model_store import ModelStore
from skulk.store.preservation_cli import main


@dataclass(frozen=True)
class _Capture:
    store: Path
    manifest_id: str
    trust: TrustProfile
    trust_path: Path


def _registry_proof(
    card: ModelCardEnvelope, root_path: Path, *, expired: bool
) -> RegistryProof:
    signer = Ed25519PrivateKey.generate()
    expires = (datetime.now(UTC) + timedelta(days=-1 if expired else 30)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    key: dict[str, JsonValue] = {
        "keytype": "ed25519",
        "scheme": "ed25519",
        "keyval": {"public": signer.public_key().public_bytes_raw().hex()},
    }
    key_id = hashlib.sha256(identity_json(key)).hexdigest()

    def signed(role: str, payload: dict[str, JsonValue]) -> bytes:
        content: dict[str, JsonValue] = {
            "_type": role,
            "spec_version": "1.0.31",
            "version": 1,
            "expires": expires,
            **payload,
        }
        return identity_json(
            {
                "signed": content,
                "signatures": [
                    {"keyid": key_id, "sig": signer.sign(identity_json(content)).hex()}
                ],
            }
        )

    def binding(data: bytes) -> dict[str, JsonValue]:
        return {
            "version": 1,
            "length": len(data),
            "hashes": {"sha256": hashlib.sha256(data).hexdigest()},
        }

    root = signed(
        "root",
        {
            "consistent_snapshot": True,
            "keys": {key_id: key},
            "roles": {
                role: {"keyids": [key_id], "threshold": 1}
                for role in ("root", "timestamp", "snapshot", "targets")
            },
        },
    )
    target_path = f"v1/cards/{card.card_id}.json"
    target = identity_json(card.model_dump(mode="json"))
    targets = signed(
        "targets",
        {
            "targets": {
                target_path: {
                    "length": len(target),
                    "hashes": {"sha256": hashlib.sha256(target).hexdigest()},
                }
            }
        },
    )
    snapshot = signed("snapshot", {"meta": {"targets.json": binding(targets)}})
    timestamp = signed("timestamp", {"meta": {"snapshot.json": binding(snapshot)}})
    root_path.write_bytes(root)
    return create_registry_proof(
        trusted_root=root,
        root_chain=(),
        timestamp=timestamp,
        snapshot=snapshot,
        targets=targets,
        target_path=target_path,
        target=target,
    )


def _capture(
    temporary: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    registry: bool = False,
    code: bool = False,
    expired: bool = False,
    invalid_proof: bool = False,
    incomplete: bool = False,
    companion: bool = False,
    split: bool = False,
    complete_split: bool = False,
    composite: bool = False,
    reserved_metadata: bool = False,
    mlx_layout: Literal[
        "none",
        "unindexed",
        "indexed",
        "missing",
        "escape",
        "malformed",
        "split-missing",
        "split-complete",
        "no-weights",
    ] = "none",
) -> _Capture:
    source = temporary / "source"
    source.mkdir()
    (source / "model.gguf").write_bytes(b"synthetic-weights")
    (source / "LICENSE").write_text("Synthetic test fixture only")
    files = ("LICENSE",) if incomplete else ("LICENSE", "model.gguf")
    selected = "model.gguf"
    if reserved_metadata:
        nested = source / "weights" / ".skulk"
        nested.mkdir(parents=True)
        (nested / "installed-card.json").write_text("{}")
        files += ("weights/.skulk/installed-card.json",)
    if split:
        selected = "model-00001-of-00002.gguf"
        (source / selected).write_bytes(b"first")
        (source / "a.gguf").write_bytes(b"unrelated-complete-quant")
        files = ("LICENSE", "a.gguf", selected)
        if complete_split:
            (source / "model-00002-of-00002.gguf").write_bytes(b"second")
            files += ("model-00002-of-00002.gguf",)
    if mlx_layout != "none":
        (source / "config.json").write_text('{"model_type":"llama"}')
        (source / "model.safetensors").write_bytes(b"synthetic-weights")
        files = ("LICENSE", "config.json", "model.safetensors")
        if mlx_layout == "no-weights":
            files = ("LICENSE", "config.json")
        elif mlx_layout.startswith("split-"):
            (source / "model-00001-of-00002.safetensors").write_bytes(b"first")
            files = ("LICENSE", "config.json", "model-00001-of-00002.safetensors")
            if mlx_layout == "split-complete":
                (source / "model-00002-of-00002.safetensors").write_bytes(b"second")
                files += ("model-00002-of-00002.safetensors",)
        elif mlx_layout != "unindexed":
            reference = (
                "../outside.safetensors"
                if mlx_layout == "escape"
                else "absent.safetensors"
                if mlx_layout == "missing"
                else "model.safetensors"
            )
            (temporary / "outside.safetensors").write_bytes(b"outside")
            (source / "model.safetensors.index.json").write_text(
                "invalid"
                if mlx_layout == "malformed"
                else json.dumps(
                    {
                        "metadata": {"total_size": 16},
                        "weight_map": {"tensor": reference},
                    }
                )
            )
            files += ("model.safetensors.index.json",)
    body: dict[str, JsonValue] = {
        "model_id": "example/recovered",
        "source_revision": "a" * 40,
        "storage_size": {"in_bytes": 16},
        "n_layers": 1,
        "hidden_size": 16,
        "supports_tensor": False,
        "tasks": ["TextGeneration"],
        "trust_remote_code": code,
        "gguf_file": selected if not incomplete and mlx_layout == "none" else None,
    }
    if composite:
        body["components"] = [
            {
                "component_name": "encoder",
                "component_path": "encoder",
                "storage_size": {"in_bytes": 16},
                "can_shard": False,
            }
        ]
    if companion:
        body["runtime"] = {
            "assistant_model_repo": "example/assistant",
            "assistant_model_revision": "b" * 40,
        }
    if registry:
        declared = tuple(
            ArtifactFile(
                path=name,
                size_bytes=(source / name).stat().st_size,
                object_id="sha256:"
                + hashlib.sha256((source / name).read_bytes()).hexdigest(),
            )
            for name in files
        )
        body["artifact_bundle"] = {
            "bundle_id": bundle_identity(declared),
            "files": [item.model_dump(mode="json") for item in declared],
            "download_size": sum(item.size_bytes for item in declared),
        }
    card = create_card(body)
    proof = None
    if registry:
        root_path = temporary / "root.json"
        proof = _registry_proof(card, root_path, expired=expired)
        monkeypatch.setattr(preservation, "EMBEDDED_REGISTRY_ROOT", root_path)
        if invalid_proof:
            proof = proof.model_copy(update={"anchor_sha256": "0" * 64})
    retained = retain_card(card, registry_proof=proof)
    member = RecipeMember(
        member_id="primary",
        role="base",
        format=card.artifact.format,
        origin_repository=card.artifact.repository,
        origin_revision=card.artifact.revision,
        artifact_identity=card.card_id,
        retained_card=cast(dict[str, JsonValue], retained.model_dump(mode="json")),
        intake=Intake(
            audience="private",
            license_id="test",
            terms_reference="fixture",
            decision="synthetic fixture",
            notice_paths=("LICENSE",),
        ),
        files=tuple(SourceFile(path=name) for name in files),
    )
    recipe = Recipe(
        schema_version="foxlight.model-preservation.v1",
        name="Synthetic",
        root_member="primary",
        workloads=("text",),
        runtime_requirements=(),
        members=(member,),
    )
    key_path, store = temporary / "owner.pem", temporary / "archive"
    trust = create_key(key_path, None)
    manifest = pack(recipe, {"primary": source}, store, (load_key(key_path, None),))
    trust_path = temporary / "owner-trust.json"
    trust_path.write_text(trust.model_dump_json())
    return _Capture(store, manifest, trust, trust_path)


def test_owner_warning_and_separate_code_permission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture = _capture(tmp_path, monkeypatch, code=True)
    home = tmp_path / "home"
    report = preservation.inspect_recovery(
        capture.store, capture.manifest_id, capture.trust
    )
    assert report.publication == "owner-only"
    assert report.requires_owner_acceptance and report.requires_code_permission
    with pytest.raises(PreservationError, match="accept this owner capture"):
        preservation.recover_home(
            capture.store, capture.manifest_id, capture.trust, home
        )
    with pytest.raises(PreservationError, match="code permission"):
        preservation.recover_home(
            capture.store,
            capture.manifest_id,
            capture.trust,
            home,
            accept_owner_capture=True,
        )
    assert not home.exists()
    preservation.recover_home(
        capture.store,
        capture.manifest_id,
        capture.trust,
        home,
        accept_owner_capture=True,
        allow_repository_code=True,
    )
    record = read_installed_card(home / "model-store" / "example--recovered")
    assert record is not None and record.verification == "custom"
    assert record.model_card.registry_card_id is None
    assert (
        record.preservation is not None
        and record.preservation.repository_code_authorized
    )


@pytest.mark.parametrize("registry", [False, True])
def test_fresh_process_normal_catalog_and_store_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, registry: bool
) -> None:
    capture = _capture(tmp_path, monkeypatch, registry=registry)
    home = tmp_path / "home"
    report = preservation.recover_home(
        capture.store,
        capture.manifest_id,
        capture.trust,
        home,
        accept_owner_capture=not registry,
    )
    assert report.publication == ("registry-fresh" if registry else "owner-only")
    script = """
import asyncio, json, socket
from pathlib import Path
def denied(*args, **kwargs):
    raise AssertionError("network attempted")
socket.socket.connect = denied
socket.create_connection = denied
from skulk.shared.constants import add_model_search_path, set_offline_mode
from skulk.shared.models.model_cards import get_all_model_cards, ModelCard, ModelId
from skulk.shared.models.remote_code_approval import require_remote_code_approval
from skulk.store.config import load_skulk_config
from skulk.store.model_store import ModelStore
from skulk.store.installed_cards import verify_installed_card
set_offline_mode()
config = load_skulk_config()
assert config and config.model_store and not config.model_store.download.allow_hf_fallback
root = Path(config.model_store.store_path)
assert Path(config.model_store.staging.node_cache_path) == Path.cwd() / "staging"
add_model_search_path(root)
async def check():
    cards = await get_all_model_cards()
    card = await ModelCard.load(ModelId("example/recovered"))
    assert any(item.model_id == card.model_id for item in cards)
    require_remote_code_approval(card)
    store = ModelStore(root)
    entry = store.get_entry(str(card.model_id))
    assert entry and entry.installed_card
    assert verify_installed_card(root / entry.store_path, entry.installed_card, verify_hashes=True)
    assert store.get_store_path(str(card.model_id)) == root / entry.store_path
    print(json.dumps({"selectable": True, "verification": entry.installed_card.verification, "custom": card.is_custom}))
asyncio.run(check())
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=home,
        env={
            **os.environ,
            "SKULK_HOME": str(home),
            "SKULK_OFFLINE": "true",
            "SKULK_MODELS_DIR": str(home / "models"),
            "SKULK_MODELS_PATH": "",
        },
        check=True,
        capture_output=True,
        text=True,
    )
    observation = cast(object, json.loads(result.stdout))
    assert observation == {
        "selectable": True,
        "verification": "registry_verified" if registry else "custom",
        "custom": not registry,
    }
    index = home / "model-store" / "registry.json"
    index.unlink()
    rebuilt = ModelStore(home / "model-store")
    assert rebuilt.is_in_store("example/recovered")


def test_invalid_proof_never_downgrades(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture = _capture(tmp_path, monkeypatch, registry=True, invalid_proof=True)
    with pytest.raises(PreservationError):
        preservation.recover_home(
            capture.store,
            capture.manifest_id,
            capture.trust,
            tmp_path / "home",
            accept_owner_capture=True,
            allow_repository_code=True,
        )
    assert not (tmp_path / "home").exists()


def test_bundle_cannot_enroll_its_own_registry_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture = _capture(tmp_path, monkeypatch, registry=True)
    monkeypatch.setattr(preservation, "EMBEDDED_REGISTRY_ROOT", EMBEDDED_REGISTRY_ROOT)
    with pytest.raises(PreservationError):
        preservation.recover_home(
            capture.store,
            capture.manifest_id,
            capture.trust,
            tmp_path / "home",
            accept_owner_capture=True,
            allow_repository_code=True,
        )
    assert not (tmp_path / "home").exists()


@pytest.mark.parametrize("boundary", ["restore", "flush"])
def test_registry_expiry_during_copy_blocks_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: Literal["restore", "flush"],
) -> None:
    capture = _capture(tmp_path, monkeypatch, registry=True)
    copied = False

    class Clock(datetime):
        @classmethod
        def now(cls, tz: tzinfo | None = None) -> Self:
            shift = timedelta(days=60 if copied else 0).total_seconds()
            return cls.fromtimestamp(datetime.now(tz).timestamp() + shift, tz)

    def delayed_restore(
        store: Path, manifest_id: str, trust: TrustProfile, destination: Path
    ) -> Manifest:
        nonlocal copied
        result = restore(store, manifest_id, trust, destination)
        copied = boundary == "restore"
        return result

    def delayed_flush(root: Path) -> None:
        nonlocal copied
        assert root.is_dir()
        copied = True

    if boundary == "flush":
        monkeypatch.setattr(preservation, "_flush_tree", delayed_flush)
    monkeypatch.setattr(proof_verifier, "datetime", Clock)
    monkeypatch.setattr(preservation, "restore", delayed_restore)
    with pytest.raises(PreservationError):
        preservation.recover_home(
            capture.store, capture.manifest_id, capture.trust, tmp_path / "home"
        )
    assert copied and not (tmp_path / "home").exists()


def test_concurrent_destination_is_never_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture = _capture(tmp_path, monkeypatch)
    home = tmp_path / "home"

    def competing_creator(staging: Path) -> None:
        assert staging.is_dir()
        home.mkdir()
        (home / "keep").write_text("existing operator data")

    monkeypatch.setattr(preservation, "_flush_tree", competing_creator)
    with pytest.raises(FileExistsError):
        preservation.recover_home(
            capture.store,
            capture.manifest_id,
            capture.trust,
            home,
            accept_owner_capture=True,
        )
    assert (home / "keep").read_text() == "existing operator data"


def test_historical_is_explicit_custom_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture = _capture(tmp_path, monkeypatch, registry=True, expired=True)
    with pytest.raises(PreservationError):
        preservation.inspect_recovery(capture.store, capture.manifest_id, capture.trust)
    report = preservation.recover_home(
        capture.store,
        capture.manifest_id,
        capture.trust,
        tmp_path / "home",
        mode="historical",
        accept_owner_capture=True,
    )
    assert report.publication == "registry-historical"
    assert report.requires_owner_acceptance
    record = read_installed_card(
        tmp_path / "home" / "model-store" / "example--recovered"
    )
    assert (
        record
        and record.verification == "custom"
        and record.model_card.registry_card_id is None
    )


def test_tamper_blocks_even_with_all_acceptances(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture = _capture(tmp_path, monkeypatch)
    weight = b"synthetic-weights"
    # Locate by content without relying on the worker's internal path format.
    selected = next(
        path
        for path in capture.store.rglob("*")
        if path.is_file() and path.read_bytes() == weight
    )
    selected.write_bytes(b"tampered-weights")
    with pytest.raises(PreservationError):
        preservation.recover_home(
            capture.store,
            capture.manifest_id,
            capture.trust,
            tmp_path / "home",
            accept_owner_capture=True,
            allow_repository_code=True,
        )
    assert not (tmp_path / "home").exists()


def test_failure_keeps_destination_unpublished_and_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture = _capture(tmp_path, monkeypatch)

    def interrupted(root: Path) -> None:
        assert root.is_dir()
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(preservation, "_flush_tree", interrupted)
        with pytest.raises(KeyboardInterrupt):
            preservation.recover_home(
                capture.store,
                capture.manifest_id,
                capture.trust,
                tmp_path / "home",
                accept_owner_capture=True,
            )
        assert not (tmp_path / "home").exists()
        assert not list(tmp_path.glob(".skulk-recovery-*"))
    preservation.recover_home(
        capture.store,
        capture.manifest_id,
        capture.trust,
        tmp_path / "home",
        accept_owner_capture=True,
    )
    with pytest.raises(PreservationError, match="new home"):
        preservation.recover_home(
            capture.store,
            capture.manifest_id,
            capture.trust,
            tmp_path / "home",
            accept_owner_capture=True,
        )


@pytest.mark.parametrize("companion,incomplete", [(True, False), (False, True)])
def test_unsupported_closure_or_incomplete_weights_never_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, companion: bool, incomplete: bool
) -> None:
    capture = _capture(
        tmp_path, monkeypatch, companion=companion, incomplete=incomplete
    )
    with pytest.raises(PreservationError, match="companions|weight layout"):
        preservation.recover_home(
            capture.store,
            capture.manifest_id,
            capture.trust,
            tmp_path / "home",
            accept_owner_capture=True,
        )
    assert not (tmp_path / "home").exists()


def test_cli_and_wrong_owner_trust(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    capture = _capture(tmp_path, monkeypatch)
    assert (
        main(
            [
                "inspect",
                capture.manifest_id,
                "--store",
                str(capture.store),
                "--trust",
                str(capture.trust_path),
            ]
        )
        == 0
    )
    assert '"publication": "owner-only"' in capsys.readouterr().out
    other = create_key(tmp_path / "other.pem", None)
    with pytest.raises(PreservationError):
        preservation.inspect_recovery(capture.store, capture.manifest_id, other)
    assert (
        main(
            [
                "recover",
                capture.manifest_id,
                "--store",
                str(capture.store),
                "--trust",
                str(capture.trust_path),
                "--destination",
                str(tmp_path / "home"),
            ]
        )
        == 1
    )
    assert "accept this owner capture" in capsys.readouterr().err


@pytest.mark.parametrize("complete", [False, True])
def test_another_quant_cannot_mask_missing_selected_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, complete: bool
) -> None:
    capture = _capture(tmp_path, monkeypatch, split=True, complete_split=complete)
    home = tmp_path / "home"
    if complete:
        preservation.recover_home(
            capture.store,
            capture.manifest_id,
            capture.trust,
            home,
            accept_owner_capture=True,
        )
        assert home.exists()
    else:
        with pytest.raises(PreservationError, match="weight layout"):
            preservation.recover_home(
                capture.store,
                capture.manifest_id,
                capture.trust,
                home,
                accept_owner_capture=True,
            )
        assert not home.exists()


def test_composite_model_requires_the_closure_importer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture = _capture(tmp_path, monkeypatch, composite=True)
    with pytest.raises(PreservationError, match="exact artifact bundle"):
        preservation.recover_home(
            capture.store,
            capture.manifest_id,
            capture.trust,
            tmp_path / "home",
            accept_owner_capture=True,
        )


def test_nested_copied_sidecar_cannot_become_receiver_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    capture = _capture(tmp_path, monkeypatch, reserved_metadata=True)
    with pytest.raises(PreservationError, match="reserved Skulk"):
        preservation.recover_home(
            capture.store,
            capture.manifest_id,
            capture.trust,
            tmp_path / "home",
            accept_owner_capture=True,
        )
    assert not (tmp_path / "home").exists()


@pytest.mark.parametrize(
    "layout,accepted",
    [
        ("unindexed", True),
        ("indexed", True),
        ("missing", False),
        ("escape", False),
        ("malformed", False),
        ("split-missing", False),
        ("split-complete", True),
        ("no-weights", False),
    ],
)
def test_mlx_layouts_are_checked_before_issuing_a_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    layout: Literal[
        "unindexed",
        "indexed",
        "missing",
        "escape",
        "malformed",
        "split-missing",
        "split-complete",
        "no-weights",
    ],
    accepted: bool,
) -> None:
    capture = _capture(tmp_path, monkeypatch, mlx_layout=layout)
    home = tmp_path / "home"
    if accepted:
        preservation.recover_home(
            capture.store,
            capture.manifest_id,
            capture.trust,
            home,
            accept_owner_capture=True,
        )
        record = read_installed_card(home / "model-store" / "example--recovered")
        assert record and record.schema_version == 3
    else:
        with pytest.raises(PreservationError, match="weight layout"):
            preservation.recover_home(
                capture.store,
                capture.manifest_id,
                capture.trust,
                home,
                accept_owner_capture=True,
            )
        assert not home.exists()


@pytest.mark.parametrize("publication", ["owner", "fresh", "historical"])
@pytest.mark.parametrize("format_name", ["mlx", "gguf"])
async def test_normal_complete_store_request_retains_receiving_evidence_without_hub(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    publication: Literal["owner", "fresh", "historical"],
    format_name: Literal["mlx", "gguf"],
) -> None:
    capture = _capture(
        tmp_path,
        monkeypatch,
        registry=publication != "owner",
        expired=publication == "historical",
        mlx_layout="unindexed" if format_name == "mlx" else "none",
    )
    home = tmp_path / "home"
    await asyncio.to_thread(
        preservation.recover_home,
        capture.store,
        capture.manifest_id,
        capture.trust,
        home,
        mode="historical" if publication == "historical" else "fresh",
        accept_owner_capture=publication != "fresh",
    )
    directory = home / "model-store" / "example--recovered"
    before = read_installed_card(directory)
    assert before and before.preservation
    probes: list[str] = []

    async def forbidden_probe(*_args: object, **_kwargs: object) -> list[FileListEntry]:
        probes.append("hub")
        raise AssertionError("a complete recovered model must not contact the Hub")

    monkeypatch.setattr(download_utils, "fetch_file_list_with_cache", forbidden_probe)
    store = ModelStore(home / "model-store")
    status = await store.request_download(
        before.artifact_model_id,
        pinned_gguf=before.artifact_file,
        source_revision=before.artifact_revision,
        source_repository=before.artifact_repository,
        model_card=before.model_card,
    )
    assert status.status == "complete" and probes == []
    assert read_installed_card(directory) == before
    entry = store.get_entry(before.artifact_model_id)
    assert entry and entry.installed_card == before


@pytest.mark.parametrize("change", ["snapshot", "card", "bytes"])
def test_receiving_observations_are_scoped_to_unchanged_card_and_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: Literal["snapshot", "card", "bytes"],
) -> None:
    capture = _capture(tmp_path, monkeypatch, registry=True, mlx_layout="unindexed")
    home = tmp_path / "home"
    preservation.recover_home(capture.store, capture.manifest_id, capture.trust, home)
    directory = home / "model-store" / "example--recovered"
    before = read_installed_card(directory)
    assert before and before.preservation
    card = before.model_card
    if change == "snapshot":
        card = card.model_copy(
            update={"registry_snapshot_id": "new-observation-of-same-card"}
        )
    elif change == "card":
        card = card.model_copy(update={"trust_remote_code": True})
    else:
        path = directory / "model.safetensors"
        path.write_bytes(b"x" * path.stat().st_size)
    after = build_installed_card_record(directory, card, previous_record=before)
    if change == "snapshot":
        assert after.preservation == before.preservation and after.schema_version == 3
        assert after.captured_at == before.captured_at
    else:
        assert after.preservation is None and after.schema_version == 2


def test_preserve_retains_explicit_original_registry_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from foxlight_model_preservation.assurance import load_retained_card
    from foxlight_model_preservation.storage import verify_store

    import skulk.store.preservation_export as exporter

    capture = _capture(tmp_path, monkeypatch, registry=True)
    source_home = tmp_path / "first-home"
    original_report = preservation.recover_home(
        capture.store, capture.manifest_id, capture.trust, source_home
    )
    manifest = verify_store(capture.store, capture.manifest_id, capture.trust)
    original = manifest.members[0].retained_card
    assert original is not None
    retained = load_retained_card(identity_json(original))
    monkeypatch.setattr(
        exporter, "EMBEDDED_REGISTRY_ROOT", preservation.EMBEDDED_REGISTRY_ROOT
    )
    archive = tmp_path / "second-archive"
    archive.mkdir()
    exported = exporter.preserve_installed(
        source_home / "model-store" / "example--recovered",
        archive,
        (load_key(tmp_path / "owner.pem", None),),
        retained_card=retained,
    )
    report = preservation.inspect_recovery(archive, exported, capture.trust)
    assert report.publication == "registry-fresh"
    assert report.card_id == original_report.card_id
    assert (
        report.file_coverage == "complete-sha256"
        and not report.requires_owner_acceptance
    )
    assert retained.registry_proof
    automatic_archive = tmp_path / "automatic-archive"
    automatic_archive.mkdir()
    automatic = exporter.preserve_installed(
        source_home / "model-store" / "example--recovered",
        automatic_archive,
        (load_key(tmp_path / "owner.pem", None),),
    )
    automatic_manifest = verify_store(automatic_archive, automatic, capture.trust)
    assert automatic_manifest.members[0].retained_card == original
    forged = retained.model_copy(
        update={
            "registry_proof": retained.registry_proof.model_copy(
                update={"anchor_sha256": "0" * 64}
            )
        }
    )
    with pytest.raises(PreservationError):
        exporter.preserve_installed(
            source_home / "model-store" / "example--recovered",
            archive,
            (load_key(tmp_path / "owner.pem", None),),
            retained_card=forged,
        )
    assert len(list((archive / "manifests").glob("*.json"))) == 1


def test_future_card_properties_remain_verifiable_without_unsafe_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import base64

    from foxlight_model_preservation.assurance import load_retained_card
    from foxlight_model_preservation.cards import create_card
    from foxlight_model_preservation.signing import digest, encode_record, sign_manifest
    from foxlight_model_preservation.storage import (
        atomic_bytes,
        directory,
        manifest_path,
        verify_store,
    )

    capture = _capture(tmp_path, monkeypatch)
    manifest = verify_store(capture.store, capture.manifest_id, capture.trust)
    original = manifest.members[0].retained_card
    assert original is not None
    retained = load_retained_card(identity_json(original))
    raw_card = dict(retained.envelope.card)
    raw_card["future_runtime_contract"] = {"required": True, "version": 99}
    # Use the ordinary public writer: additive metadata is identity-bound and
    # preservable even though this runtime cannot interpret the new requirement.
    future_card = create_card(raw_card, alias=retained.envelope.alias)
    envelope = future_card.model_dump(mode="json")
    future = {
        **original,
        "card_base64": base64.b64encode(identity_json(envelope)).decode(),
    }
    member = manifest.members[0].model_copy(update={"retained_card": future})
    updated = manifest.model_copy(update={"members": (member, *manifest.members[1:])})
    identity = digest(encode_record(updated))
    signed = sign_manifest(updated, (load_key(tmp_path / "owner.pem", None),))
    with directory(capture.store) as root:
        atomic_bytes(root, manifest_path(identity), encode_record(signed))
    assert (
        verify_store(capture.store, identity, capture.trust).members[0].retained_card
        == future
    )
    restore(capture.store, identity, capture.trust, tmp_path / "future-files")
    with pytest.raises(
        preservation.RecoveryCompatibilityError, match="compatible reader"
    ):
        preservation.inspect_recovery(capture.store, identity, capture.trust)


def test_derived_runtime_view_keeps_original_signed_source_across_recapture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from foxlight_model_preservation.assurance import load_retained_card
    from foxlight_model_preservation.storage import object_path, verify_store

    import skulk.store.preservation_export as exporter
    import skulk.store.preservation_provenance as provenance
    from skulk.store.installed_cards import write_installed_card

    capture = _capture(tmp_path, monkeypatch, registry=True)
    first_home = tmp_path / "first-view"
    preservation.recover_home(
        capture.store, capture.manifest_id, capture.trust, first_home
    )
    original_manifest = verify_store(capture.store, capture.manifest_id, capture.trust)
    original = original_manifest.members[0].retained_card
    assert original is not None
    primary = first_home / "model-store/example--recovered"
    record = read_installed_card(primary)
    assert record is not None
    # Runtime-derived header facts can differ from the original signed view.
    write_installed_card(
        primary,
        record.model_copy(
            update={
                "model_card": record.model_card.model_copy(
                    update={"hidden_size": record.model_card.hidden_size + 16}
                )
            }
        ),
    )
    monkeypatch.setattr(
        exporter, "EMBEDDED_REGISTRY_ROOT", preservation.EMBEDDED_REGISTRY_ROOT
    )

    def no_cached_card(card_id: str) -> None:
        return None

    monkeypatch.setattr(provenance, "cached_registry_card", no_cached_card)
    for number in (1, 2):
        archive = tmp_path / f"derived-archive-{number}"
        archive.mkdir()
        identity = exporter.preserve_installed(
            primary, archive, (load_key(tmp_path / "owner.pem", None),)
        )
        manifest = verify_store(archive, identity, capture.trust)
        root = load_retained_card(identity_json(manifest.members[0].retained_card))
        assert root.registry_proof is None
        assert (
            preservation.inspect_recovery(archive, identity, capture.trust).publication
            == "owner-only"
        )
        evidence = next(
            member
            for member in manifest.members
            if member.member_id == "source-evidence"
        )
        source = next(
            file for file in evidence.files if file.path == "registry-source-card.json"
        )
        assert (
            load_retained_card(
                (archive / object_path(source.sha256)).read_bytes()
            ).model_dump(mode="json")
            == original
        )
        if number == 1:
            second_home = tmp_path / "second-view"
            preservation.recover_home(
                archive, identity, capture.trust, second_home, accept_owner_capture=True
            )
            primary = second_home / "model-store/example--recovered"
    retained_file = (
        primary.parent.parent
        / "preservation/source-evidence/source-evidence/registry-source-card.json"
    )
    retained_file.write_bytes(retained_file.read_bytes() + b"x")
    rejected = tmp_path / "bad-source-archive"
    rejected.mkdir()
    with pytest.raises(PreservationError, match="integrity"):
        exporter.preserve_installed(
            primary, rejected, (load_key(tmp_path / "owner.pem", None),)
        )
