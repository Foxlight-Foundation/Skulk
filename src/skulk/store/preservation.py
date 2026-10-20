"""Offline public-bundle inspection and atomic recovery into a fresh Skulk home.

Live store mutation belongs to the store's serialized host service. This local
recovery entry point instead publishes a *new* home, including its normal custom
card authorization records and model-store index, in one directory rename.
"""

from __future__ import annotations

import os
import re
import shutil
import socket
import uuid
from pathlib import Path
from typing import Literal, final

import anyio
from foxlight_model_preservation.assurance import (
    inspect_member_evidence,
    load_retained_card,
)
from foxlight_model_preservation.cards import identity_json
from foxlight_model_preservation.models import (
    MAX_METADATA_BYTES,
    Manifest,
    PreservationError,
    TrustProfile,
    canonical_path,
)
from foxlight_model_preservation.operations import OperationControl
from foxlight_model_preservation.registry_proof import ProofMode, ProofReport
from foxlight_model_preservation.signing import load_trust
from foxlight_model_preservation.storage import directory, restore, verify_store
from pydantic import BaseModel, ConfigDict, Field

from skulk.download.download_utils import (
    selected_gguf_group_complete,
    write_source_revision_marker,
)
from skulk.shared.models.model_cards import (
    ModelCard,
    ModelTask,
    card_requires_newer_skulk,
)
from skulk.shared.models.registry import EMBEDDED_REGISTRY_ROOT
from skulk.shared.types.worker.downloads import ModelSafetensorsIndex
from skulk.store.config import (
    DownloadStoreConfig,
    ModelStoreConfig,
    SkulkConfig,
    StagingNodeConfig,
)
from skulk.store.installed_cards import (
    PreservationEvidence,
    build_installed_card_record,
    verify_installed_card,
)
from skulk.store.model_store import ModelStore


@final
class RecoveryInspection(BaseModel):
    """Bounded summary of verified bytes and independently checked card evidence."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    manifest_id: str = Field(
        description="Exact SHA-256 manifest selected by the receiver."
    )
    model_id: str = Field(description="Normal Skulk model alias to recover.")
    card_id: str = Field(
        description="Original public card identity, without implied registry authority."
    )
    publication: Literal["owner-only", "registry-fresh", "registry-historical"] = Field(
        description="Verified publication assurance."
    )
    file_coverage: Literal["complete-sha256", "partial-or-missing"] = Field(
        description="Original card coverage, separate from verified capture bytes."
    )
    unbound_file_count: int = Field(
        description="Files not bound to SHA-256 by the original card."
    )
    total_bytes: int = Field(description="Logical bytes required by this recovery.")
    requires_owner_acceptance: bool = Field(
        description="Whether recovered bytes need explicit custom/operator admission."
    )
    requires_code_permission: bool = Field(
        description="Whether the card may execute repository code."
    )
    registry_proof: ProofReport | None = Field(
        description="Checked versions, expiry and supplied-chain revocation scope; no latest-registry claim."
    )
    runtime_requirements: tuple[str, ...] = Field(
        description="Owner statements about required runtimes; not a qualification result."
    )


def read_recovery_metadata(path: Path) -> bytes:
    """Read bounded local metadata without accepting a bundle's embedded keys as trust."""
    with path.open("rb") as stream:
        data = stream.read(MAX_METADATA_BYTES + 1)
    if len(data) > MAX_METADATA_BYTES:
        raise PreservationError("recovery metadata exceeds the size limit")
    return data


def _runtime_card(manifest: Manifest) -> ModelCard:
    if len(manifest.members) != 1:
        raise PreservationError(
            "this importer supports one self-contained text model; companion sets need the closure importer"
        )
    member = manifest.members[0]
    if member.retained_card is None:
        raise PreservationError("recovery requires a retained public model card")
    envelope = load_retained_card(identity_json(member.retained_card)).envelope
    payload = dict(envelope.card)
    payload["model_id"] = envelope.alias
    payload["source_repository"] = envelope.artifact.repository
    card = ModelCard.model_validate_json(identity_json(payload))
    card.require_immutable_external_companions(context="preservation recovery")
    if reason := card_requires_newer_skulk(card):
        raise PreservationError(reason)
    if (
        card.tasks != [ModelTask.TextGeneration]
        or card.vision is not None
        or card.image is not None
        or card.video is not None
        or card.audio is not None
        or card.music is not None
        or card.components is not None
        or member.dependencies
        or member.role != "base"
        or envelope.artifact.format not in {"mlx", "gguf"}
    ):
        raise PreservationError(
            "recovery currently supports self-contained MLX/GGUF text models only"
        )
    if card.runtime is not None and any(
        (
            card.runtime.mtp_sidecar_repo,
            card.runtime.assistant_model_repo,
            card.runtime.served_spec_draft_repo,
            card.runtime.vllm_spec_draft_repo,
        )
    ):
        raise PreservationError(
            "recovery of separate runtime companions is not supported yet"
        )
    # These names belong to the receiving store. Captured sidecars must never
    # be confused with a freshly issued receiver authorization record.
    reserved = {
        ".skulk",
        ".skulk-source-revision",
        ".skulk-source-revision-staging",
        ".last_used",
    }
    if any(
        any(part.casefold() in reserved for part in file.path.split("/"))
        for file in member.files
    ):
        raise PreservationError("capture contains reserved Skulk installation metadata")
    return card


def _inspect_verified(
    manifest: Manifest, manifest_id: str, mode: ProofMode
) -> tuple[RecoveryInspection, ModelCard]:
    card = _runtime_card(manifest)
    evidence = inspect_member_evidence(
        manifest.members[0], trusted_root=EMBEDDED_REGISTRY_ROOT.read_bytes(), mode=mode
    )
    owner_acceptance = (
        evidence.publication != "registry-fresh"
        or evidence.file_coverage != "complete-sha256"
    )
    # Project registry identity only after independent publication AND complete
    # original SHA-256 binding. A card signature alone does not bind v1 bytes.
    payload = card.model_dump(mode="json")
    payload["is_custom"] = owner_acceptance
    if not owner_acceptance:
        payload["registry_card_id"] = evidence.card_id
        payload["registry_snapshot_id"] = "retained-proof:" + manifest_id
        payload["registry_artifact_format"] = manifest.members[0].format
        payload["registry_provenance"] = (
            evidence.proof.provenance if evidence.proof else None
        )
    card = ModelCard.model_validate(payload)
    return RecoveryInspection(
        manifest_id=manifest_id,
        model_id=str(card.model_id),
        card_id=evidence.card_id,
        publication=evidence.publication,
        file_coverage=evidence.file_coverage,
        unbound_file_count=len(evidence.unbound_paths),
        total_bytes=manifest.total_bytes,
        requires_owner_acceptance=owner_acceptance,
        requires_code_permission=card.trust_remote_code,
        registry_proof=evidence.proof,
        runtime_requirements=manifest.runtime_requirements,
    ), card


def inspect_recovery(
    store: Path,
    manifest_id: str,
    trust: TrustProfile,
    *,
    mode: ProofMode = "fresh",
    control: OperationControl | None = None,
) -> RecoveryInspection:
    """Hash the actual bundle and validate it against this runtime and packaged registry root.

    This performs no network access or writes. Invalid registry proofs always
    fail, including when the caller intends to accept a community capture.
    """
    manifest = verify_store(store, manifest_id, trust, control=control)
    return _inspect_verified(manifest, manifest_id, mode)[0]


def _flush_tree(root: Path) -> None:
    for folder, _, files in os.walk(root, topdown=False):
        for name in files:
            with (Path(folder) / name).open("rb") as stream:
                os.fsync(stream.fileno())
        descriptor = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _mlx_layout_complete(directory: Path) -> bool:
    """Check indexed and unindexed weights before issuing a receiver sidecar."""
    if not (directory / "config.json").is_file():
        return False
    weights = {
        path.name for path in directory.glob("model*.safetensors") if path.is_file()
    }
    if not weights:
        return False
    for index in directory.rglob("*.safetensors.index.json"):
        try:
            parsed = ModelSafetensorsIndex.model_validate_json(
                read_recovery_metadata(index)
            )
            if not parsed.weight_map:
                return False
            for selected in parsed.weight_map.values():
                if not (index.parent / canonical_path(selected)).is_file():
                    return False
        except ValueError:
            return False
    for name in weights:
        split = re.fullmatch(r"(.+)-(\d+)-of-(\d+)\.safetensors", name)
        if split is None:
            continue
        base, index_token, total_token = split.groups()
        total = int(total_token)
        if not 1 <= int(index_token) <= total <= len(weights):
            return False
        if any(
            f"{base}-{number:0{len(index_token)}d}-of-{total_token}.safetensors"
            not in weights
            for number in range(1, total + 1)
        ):
            return False
    return True


def recover_home(
    store: Path,
    manifest_id: str,
    trust: TrustProfile,
    destination: Path,
    *,
    mode: ProofMode = "fresh",
    accept_owner_capture: bool = False,
    allow_repository_code: bool = False,
) -> RecoveryInspection:
    """Publish verified model bytes, normal store records and card admission atomically.

    ``destination`` must not exist, and its parent must already be mounted and
    writable. Existing homes and live stores are never modified. Owner-capture
    acceptance and repository-code permission are independent, exact-invocation
    decisions. The bundle never supplies either grant. Failure before publication
    leaves no published home. After an interruption near publication, inspect the
    selected destination; a completed home is never overwritten by a retry.
    """
    parent = destination.parent.resolve(strict=True)
    with directory(parent) as parent_descriptor:
        return _recover_home(
            store,
            manifest_id,
            trust,
            parent / destination.name,
            parent_descriptor,
            mode=mode,
            accept_owner_capture=accept_owner_capture,
            allow_repository_code=allow_repository_code,
        )


def _recover_home(
    store: Path,
    manifest_id: str,
    trust: TrustProfile,
    destination: Path,
    parent_descriptor: int,
    *,
    mode: ProofMode,
    accept_owner_capture: bool,
    allow_repository_code: bool,
) -> RecoveryInspection:
    parent = destination.parent
    parent_identity = os.fstat(parent_descriptor)

    def parent_unchanged() -> bool:
        try:
            current = parent.stat()
        except FileNotFoundError:
            return False
        return (current.st_dev, current.st_ino) == (
            parent_identity.st_dev,
            parent_identity.st_ino,
        )

    if destination.exists() or destination.is_symlink():
        raise PreservationError("recovery destination must be a new home")
    manifest = verify_store(store, manifest_id, trust)
    inspection, card = _inspect_verified(manifest, manifest_id, mode)
    if inspection.requires_owner_acceptance and not accept_owner_capture:
        raise PreservationError(
            "original registry/file assurance is incomplete or historical; inspect and explicitly accept this owner capture"
        )
    if inspection.requires_code_permission and not allow_repository_code:
        raise PreservationError(
            "this card may execute repository code; separate explicit code permission is required"
        )
    if shutil.disk_usage(parent).free < inspection.total_bytes + 16 * 1024 * 1024:
        raise PreservationError("not enough free space for the recovery home")
    pending = parent / (".skulk-recovery-" + uuid.uuid4().hex)
    if not parent_unchanged():
        raise PreservationError("recovery parent changed before staging")
    os.mkdir(pending.name, mode=0o700, dir_fd=parent_descriptor)
    try:
        # The standalone restore rehashes the copied bytes and publishes only
        # its private staging child. It is not the host's final commit point.
        restore(store, manifest_id, trust, pending / "capture")
        # Large copies can outlive signed metadata. Do not admit a formerly
        # fresh proof after it expires during materialization.
        inspection, card = _inspect_verified(manifest, manifest_id, mode)
        store_path = pending / "model-store"
        store_path.mkdir()
        model_path = store_path / card.model_id.normalize()
        (pending / "capture" / "members" / manifest.root_member).rename(model_path)
        (pending / "capture" / ".preservation").rename(pending / "preservation")
        shutil.rmtree(pending / "capture")
        load_path = model_path / (manifest.members[0].loader_root or "")
        layout_complete = (
            selected_gguf_group_complete(model_path, card.gguf_file)
            if card.gguf_file is not None
            else _mlx_layout_complete(load_path)
        )
        if not layout_complete:
            raise PreservationError(
                "capture lacks the normal runtime's complete weight layout"
            )
        write_source_revision_marker(model_path, card.source_revision)
        record = build_installed_card_record(model_path, card)
        record = record.model_copy(
            update={
                "schema_version": 3,
                "preservation": PreservationEvidence(
                    manifest_id=manifest_id,
                    card_id=inspection.card_id,
                    publication=inspection.publication,
                    file_coverage=inspection.file_coverage,
                    repository_code_authorized=allow_repository_code
                    and inspection.requires_code_permission,
                ),
            }
        )
        if not verify_installed_card(model_path, record, verify_hashes=True):
            raise PreservationError(
                "normal store verification rejected the recovered installation"
            )
        ModelStore(store_path, recover_installed_cards=False).register_model(
            str(card.model_id),
            model_path,
            [file.path for file in record.files],
            inspection.total_bytes,
            source_revision=card.source_revision,
            source_repository=str(card.artifact_repository),
            installed_card=record,
        )
        if card.is_custom:
            custom_directory = pending / "custom_model_cards"
            custom_directory.mkdir()
            anyio.run(
                card.save,
                anyio.Path(
                    str(custom_directory / (card.model_id.normalize() + ".toml"))
                ),
            )
        configuration = SkulkConfig(
            model_store=ModelStoreConfig(
                store_host=socket.gethostname().split(".", 1)[0],
                store_http_host="127.0.0.1",
                store_path=str(destination / "model-store"),
                download=DownloadStoreConfig(allow_hf_fallback=False),
                staging=StagingNodeConfig(node_cache_path=str(destination / "staging")),
            )
        )
        # JSON is valid YAML and avoids an untyped serializer at this boundary.
        (pending / "skulk.yaml").write_text(
            configuration.model_dump_json(exclude_none=True, indent=2)
        )
        (pending / "preservation" / "inspection.json").write_text(
            inspection.model_dump_json(indent=2)
        )
        (pending / "preservation" / "owner-trust.json").write_text(
            trust.model_dump_json(indent=2)
        )
        _flush_tree(pending)
        # Exclusive reservation prevents replacing an existing operator home.
        if not parent_unchanged():
            raise PreservationError("recovery parent changed before publication")
        os.mkdir(destination.name, mode=0o700, dir_fd=parent_descriptor)
        try:
            os.rename(
                pending.name,
                destination.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
        except BaseException:
            os.rmdir(destination.name, dir_fd=parent_descriptor)
            raise
        os.fsync(parent_descriptor)
        return inspection
    finally:
        if parent_unchanged() and pending.exists():
            shutil.rmtree(pending)


def load_owner_trust(path: Path) -> TrustProfile:
    """Load an explicitly selected receiver trust file, with a bounded read."""
    return load_trust(read_recovery_metadata(path))
