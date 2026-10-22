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
from pydantic import BaseModel, ConfigDict, Field, ValidationError

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
from skulk.store.preservation_closure import RecoveryMember, map_recovery_members
from skulk.utils.directory_publication import publish_directory


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
    members: tuple[str, ...] = Field(
        default=(),
        description="Exact base and companion store aliases restored together.",
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


class RecoveryCompatibilityError(PreservationError):
    """Captured bytes remain recoverable, but this runtime cannot project the card."""


def recovery_card(manifest: Manifest) -> ModelCard:
    """Validate the public root card and complete supported loader mapping."""
    member = next(
        item for item in manifest.members if item.member_id == manifest.root_member
    )
    if member.retained_card is None:
        raise PreservationError("recovery requires a retained public model card")
    try:
        retained = load_retained_card(identity_json(member.retained_card))
        envelope = retained.envelope
        payload = dict(envelope.card)
        payload["model_id"] = envelope.alias
        if (
            retained.registry_proof is not None
            or envelope.artifact.repository != envelope.alias
            or payload.get("source_repository") is not None
        ):
            payload["source_repository"] = envelope.artifact.repository
        else:
            payload.pop("source_repository", None)
        card = ModelCard.model_validate_json(identity_json(payload))
    except ValidationError as error:
        raise RecoveryCompatibilityError(
            "this reader cannot interpret the retained card; keep the kit and use a compatible reader"
        ) from error
    card.require_immutable_external_companions(context="preservation recovery")
    if reason := card_requires_newer_skulk(card):
        raise RecoveryCompatibilityError(reason)
    if card.artifact_bundle is None and (
        card.tasks != [ModelTask.TextGeneration]
        or card.image is not None
        or card.video is not None
        or card.audio is not None
        or card.music is not None
        or card.components is not None
        or member.format not in {"mlx", "gguf"}
    ):
        raise PreservationError(
            "non-text or composite recovery requires an exact artifact bundle"
        )
    map_recovery_members(manifest, card)
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
        for member in manifest.members
        if member.role != "notices"
        for file in member.files
    ):
        raise PreservationError("capture contains reserved Skulk installation metadata")
    return card


def inspect_verified_manifest(
    manifest: Manifest, manifest_id: str, mode: ProofMode
) -> tuple[RecoveryInspection, ModelCard]:
    """Inspect authenticated metadata; the caller must separately verify weights.

    Keeping this projection separate from exact retained bytes allows receiving
    runtimes to evolve without rewriting original card/provenance identities.
    """
    card = recovery_card(manifest)
    root = next(
        item for item in manifest.members if item.member_id == manifest.root_member
    )
    closure = map_recovery_members(manifest, card)
    evidence = inspect_member_evidence(
        root, trusted_root=EMBEDDED_REGISTRY_ROOT.read_bytes(), mode=mode
    )
    for item in closure[1:]:
        if item.member.retained_card is not None:
            inspect_member_evidence(
                item.member, trusted_root=EMBEDDED_REGISTRY_ROOT.read_bytes(), mode=mode
            )
    owner_acceptance = (
        evidence.publication != "registry-fresh"
        or evidence.file_coverage != "complete-sha256"
        or len(closure) > 1
    )
    # Project registry identity only after independent publication AND complete
    # original SHA-256 binding. A card signature alone does not bind v1 bytes.
    payload = card.model_dump(mode="json")
    payload["is_custom"] = owner_acceptance
    if not owner_acceptance:
        payload["registry_card_id"] = evidence.card_id
        payload["registry_snapshot_id"] = "retained-proof:" + manifest_id
        payload["registry_artifact_format"] = root.format
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
        # Vision processors share the runtime's repository-code authorization
        # boundary, including cards whose primary tokenizer declines code.
        requires_code_permission=card.trust_remote_code or card.vision is not None,
        registry_proof=evidence.proof,
        members=tuple(item.model_id for item in closure),
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
    return inspect_verified_manifest(manifest, manifest_id, mode)[0]


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


def validate_recovery_layout(
    directory: Path, item: RecoveryMember, owner: ModelCard
) -> None:
    """Check the runtime-selected layout before generating any receiving sidecar."""
    selection = item.selection
    bundle = selection.artifact_bundle
    load_path = directory / (item.member.loader_root or "")
    if bundle is not None:
        actual = {file.path: file for file in item.member.files}
        for file in bundle.files:
            captured = actual.get(file.path)
            if captured is None or captured.size != file.size_bytes:
                raise PreservationError(
                    "capture lacks the complete selected artifact bundle"
                )
            if (
                file.object_id is not None
                and file.object_id.startswith("sha256:")
                and file.object_id != "sha256:" + captured.sha256
            ):
                raise PreservationError(
                    "capture differs from the selected artifact bundle"
                )
    if item.role == "mtp_sidecar":
        # The runner reads the pinned tensor file directly; real sidecar
        # repositories need not contain a model configuration or tokenizer.
        if not (load_path / "mtp.safetensors").is_file():
            raise PreservationError("capture lacks the complete MTP companion layout")
    elif item.role == "video_companion":
        for companion in owner.video.companions if owner.video is not None else ():
            if (
                companion.repo == item.member.origin_repository
                and companion.revision == item.member.origin_revision
            ):
                path = directory / canonical_path(companion.path)
                if not path.is_file() or (
                    companion.size_bytes is not None
                    and path.stat().st_size != companion.size_bytes
                ):
                    raise PreservationError("capture lacks a required video companion")
    elif (
        bundle is None
        or item.role in {"assistant", "vision_weights", "served_draft"}
        or (item.role == "base" and owner.tasks == [ModelTask.TextGeneration])
    ):
        complete = (
            selected_gguf_group_complete(directory, selection.gguf_file)
            if selection.gguf_file is not None
            else _mlx_layout_complete(load_path)
        )
        if not complete:
            raise PreservationError(
                "capture lacks the normal runtime's complete weight layout"
            )
    if item.role == "base":
        if owner.image is not None:
            captured = {file.path: file for file in item.member.files}
            for component in owner.image.components:
                if component.repository != str(owner.artifact_repository):
                    continue
                for expected in component.files:
                    actual = captured.get(expected.path)
                    if (
                        actual is None
                        or actual.size != expected.size_bytes
                        or (
                            expected.object_id.startswith("sha256:")
                            and expected.object_id != "sha256:" + actual.sha256
                        )
                    ):
                        raise PreservationError(
                            "capture lacks a required primary image component"
                        )
        if owner.video is not None:
            for companion in owner.video.companions:
                if companion.repo is not None and str(companion.repo) != str(
                    owner.artifact_repository
                ):
                    continue
                path = directory / canonical_path(companion.path)
                if not path.is_file() or (
                    companion.size_bytes is not None
                    and path.stat().st_size != companion.size_bytes
                ):
                    raise PreservationError(
                        "capture lacks a required primary video companion"
                    )
        vision = owner.vision
        if owner.gguf_file is not None and vision is not None:
            projector = vision.projector_file
            if projector is None:
                raise PreservationError(
                    "vision recovery requires an explicitly selected projector file"
                )
            path = directory / canonical_path(projector)
            if not path.is_file() or (
                vision.projector_size is not None
                and path.stat().st_size != vision.projector_size
            ):
                raise PreservationError("capture lacks the required vision projector")
        runtime = owner.runtime
        if runtime is not None:
            if (
                runtime.mtp_sidecar_repo == str(owner.artifact_repository)
                and runtime.mtp_heads
                and not (directory / "mtp.safetensors").is_file()
            ):
                raise PreservationError(
                    "capture lacks the required same-repository MTP companion"
                )
            if (
                runtime.served_spec_draft_repo == str(owner.artifact_repository)
                and runtime.served_spec_draft_file
                and not selected_gguf_group_complete(
                    directory, runtime.served_spec_draft_file
                )
            ):
                raise PreservationError(
                    "capture lacks the required same-repository draft companion"
                )


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
    inspection, card = inspect_verified_manifest(manifest, manifest_id, mode)
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
        inspection, card = inspect_verified_manifest(manifest, manifest_id, mode)
        store_path = pending / "model-store"
        store_path.mkdir()
        receiver_store = ModelStore(store_path, recover_installed_cards=False)
        closure = map_recovery_members(manifest, card)
        for item in closure:
            model_path = store_path / item.selection.model_id.normalize()
            (pending / "capture" / "members" / item.member.member_id).rename(model_path)
            validate_recovery_layout(model_path, item, card)
            write_source_revision_marker(model_path, item.member.origin_revision)
            record = build_installed_card_record(
                model_path,
                card,
                artifact_role=item.role,
                artifact_model_id=item.model_id,
                owner_model_id=str(card.model_id) if item.role != "base" else None,
                owner_card_id=card.registry_card_id if item.role != "base" else None,
                artifact_repository=item.member.origin_repository,
                artifact_revision=item.member.origin_revision,
                artifact_format=item.member.format,
                artifact_file=item.selection.gguf_file,
            )
            record = record.model_copy(
                update={
                    "schema_version": 3,
                    "preservation": PreservationEvidence(
                        manifest_id=manifest_id,
                        card_id=inspection.card_id,
                        publication=inspection.publication
                        if item.role == "base"
                        else "owner-only",
                        file_coverage=inspection.file_coverage
                        if item.role == "base"
                        else "partial-or-missing",
                        repository_code_authorized=allow_repository_code
                        and inspection.requires_code_permission,
                    ),
                }
            )
            if not verify_installed_card(model_path, record, verify_hashes=True):
                raise PreservationError(
                    "normal store verification rejected the recovered installation"
                )
            receiver_store.register_model(
                item.model_id,
                model_path,
                [file.path for file in record.files],
                sum(file.size_bytes for file in record.files),
                repo_has_projector=card.vision is not None
                and card.gguf_file is not None
                if item.role == "base"
                else False,
                source_revision=item.member.origin_revision,
                source_repository=item.member.origin_repository,
                installed_card=record,
            )
        # Original notices/installed observations remain signed evidence only;
        # they are never copied over the receiver's authorization sidecars.
        (pending / "capture" / ".preservation").rename(pending / "preservation")
        if (pending / "capture" / "members").exists():
            (pending / "capture" / "members").rename(
                pending / "preservation" / "source-evidence"
            )
        shutil.rmtree(pending / "capture")
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
        # Flushing a large closure can outlive the previously checked proof.
        # Fresh publication must still be valid at the final commit boundary.
        inspect_verified_manifest(manifest, manifest_id, mode)
        # No intermediate reservation can strand an empty operator home after
        # process exit; existing destinations still cannot be replaced.
        if not parent_unchanged():
            raise PreservationError("recovery parent changed before publication")
        publish_directory(pending, destination.name, parent_descriptor)
        return inspection
    finally:
        if parent_unchanged() and pending.exists():
            shutil.rmtree(pending)


def load_owner_trust(path: Path) -> TrustProfile:
    """Load an explicitly selected receiver trust file, with a bounded read."""
    return load_trust(read_recovery_metadata(path))
