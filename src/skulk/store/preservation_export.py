"""Native private preservation from exact installed records, without a recipe."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast, final

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from foxlight_model_preservation.assurance import (
    RetainedCard,
    inspect_member_evidence,
    retain_card,
)
from foxlight_model_preservation.cards import create_card, identity_json
from foxlight_model_preservation.models import (
    FileIdentity,
    Intake,
    Manifest,
    ManifestMember,
    PreservationError,
    Recipe,
    RecipeMember,
    SourceFile,
    canonical_path,
)
from foxlight_model_preservation.operations import OperationControl
from foxlight_model_preservation.signing import digest, encode_record
from foxlight_model_preservation.storage import copy_exact, directory, open_file, pack
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from skulk.shared.models.model_cards import ModelCard
from skulk.shared.models.registry import EMBEDDED_REGISTRY_ROOT
from skulk.store.installed_cards import (
    InstalledCardRecord,
    companion_owner_matches,
    read_installed_card,
)
from skulk.store.model_store import StoreModelEntry, StoreRegistryIndex
from skulk.store.preservation import recovery_card, validate_recovery_layout
from skulk.store.preservation_closure import companion_selections, map_recovery_members


class PreservationCapacityError(PreservationError):
    """Refuse capture before copying when snapshot/archive space is insufficient."""


@final
class PreservePlan(BaseModel):
    """Read-only selected installation summary; not a grant or a completed copy."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    model_id: str = Field(description="Exact installed base model alias.")
    artifacts: tuple[str, ...] = Field(
        description="Base and all selected companion aliases."
    )
    source_bytes: int = Field(
        description="Logical installed file bytes before capture metadata."
    )
    audience: Literal["private"] = Field(
        default="private", description="Native exports never enable public serving."
    )
    retains_registry_proof: bool = Field(
        description="Whether explicit original registry evidence was supplied."
    )


CAPTURE_NOTICE = b"Private owner capture. Original license/attribution may be incomplete. No redistribution rights, upstream authorship or runtime qualification are granted.\n"


@dataclass(frozen=True)
class _Source:
    path: Path
    record: InstalledCardRecord
    member_id: str


def public_card_body(card: ModelCard) -> dict[str, JsonValue]:
    """Project public model semantics while withholding receiver authority fields."""
    # Local authorization and verified registry projections are receiver state,
    # not fields an owner may assert in an original public card.
    excluded = {name for name in ModelCard.model_fields if name.startswith("registry_")}
    excluded.update({"is_custom", "qualification_only"})
    return cast(
        dict[str, JsonValue],
        card.model_dump(mode="json", exclude=excluded, exclude_none=True),
    )


def _retained(
    record: InstalledCardRecord, original: RetainedCard | None
) -> RetainedCard:
    if original is None:
        return retain_card(create_card(public_card_body(record.model_card)))
    envelope = original.envelope
    payload = dict(envelope.card)
    payload["model_id"] = envelope.alias
    payload["source_repository"] = envelope.artifact.repository
    selected = ModelCard.model_validate_json(identity_json(payload))
    installed = record.model_card.model_copy(
        update={"source_repository": record.model_card.artifact_repository}
    )
    if public_card_body(selected) != public_card_body(installed):
        raise PreservationError(
            "retained original card differs from the selected installed card"
        )
    return original


def _artifact_directories(root: Path, model_id: str) -> tuple[Path, ...]:
    index_path = root / "registry.json"
    if index_path.is_file():
        # Read the index without constructing another ModelStore (which can
        # recover/rewrite state). Canonical paths carry source/bundle suffixes
        # and can change independently of a repository's human-facing alias.
        adapter: TypeAdapter[StoreRegistryIndex | dict[str, StoreModelEntry]] = (
            TypeAdapter(StoreRegistryIndex | dict[str, StoreModelEntry])
        )
        registry = adapter.validate_json(index_path.read_bytes())
        entries = (
            registry.entries if isinstance(registry, StoreRegistryIndex) else registry
        )
        entry = entries.get(model_id)
        if entry is None:
            return ()
        if entry.model_id != model_id:
            raise PreservationError(
                "store index alias differs from its registered artifact"
            )
        relative = canonical_path(entry.store_path)
        path = root / relative
        if not path.resolve().is_relative_to(root.resolve()):
            raise PreservationError(
                "registered artifact path escapes its selected store"
            )
        return (path,)
    if not root.is_dir():
        return ()
    # Standalone/staged roots need no index. Discover only adjacent installed
    # records, then validate exact ownership/revision below; names grant no trust.
    return tuple(
        path
        for path in root.iterdir()
        if not path.name.startswith(".") and not path.is_symlink() and path.is_dir()
    )


def _sources(
    model_directory: Path, artifact_roots: tuple[Path, ...]
) -> tuple[_Source, ...]:
    record = read_installed_card(model_directory)
    if record is None or record.artifact_role != "base":
        raise PreservationError("Preserve requires a base installed-card record")
    if (
        record.artifact_revision != record.model_card.source_revision
        or record.artifact_repository != str(record.model_card.artifact_repository)
    ):
        raise PreservationError("installed base identity disagrees with its card")
    sources = [_Source(model_directory, record, "primary")]
    roots = tuple(dict.fromkeys((model_directory.parent, *artifact_roots)))
    for index, (role, selection) in enumerate(companion_selections(record.model_card)):
        matches: dict[Path, _Source] = {}
        for root in roots:
            for candidate in _artifact_directories(root, str(selection.model_id)):
                if not candidate.is_dir():
                    continue
                child = read_installed_card(candidate)
                if child is not None and (
                    child.artifact_role == role
                    and child.artifact_model_id == str(selection.model_id)
                    and child.artifact_repository == str(selection.artifact_repository)
                    and child.artifact_revision == selection.source_revision
                    and companion_owner_matches(child, record.model_card)
                ):
                    matches[candidate.resolve()] = _Source(
                        candidate, child, f"companion-{index}"
                    )
        if len(matches) != 1:
            raise PreservationError(
                f"required companions are missing or ambiguous: {role} {selection.model_id}"
            )
        sources.append(next(iter(matches.values())))
    for source in sources:
        if source.record.artifact_revision is None:
            raise PreservationError(
                "portable v1 requires an exact source revision; unknown origins need a versioned extension"
            )
        if not source.record.files:
            raise PreservationError("installed source has no selected files")
    return tuple(sources)


def plan_preservation(
    model_directory: Path,
    *,
    artifact_roots: tuple[Path, ...] = (),
    retained_card: RetainedCard | None = None,
) -> PreservePlan:
    """Resolve installed closure without writes, networking or repository execution.

    This inventory plan is not full byte verification. Execution hashes the
    copied snapshot against each installed file identity before publication.
    """
    sources = _sources(model_directory, artifact_roots)
    retained = _retained(sources[0].record, retained_card)
    return PreservePlan(
        model_id=str(sources[0].record.model_card.model_id),
        artifacts=tuple(source.record.artifact_model_id for source in sources),
        source_bytes=sum(
            file.size_bytes for source in sources for file in source.record.files
        ),
        retains_registry_proof=retained.registry_proof is not None,
    )


def preservation_generation(
    model_directory: Path, *, artifact_roots: tuple[Path, ...] = ()
) -> str:
    """Bind the complete installed record selection without hashing model bytes.

    The authoritative store owner holds its publication fence while planning or
    executing. Execution independently hashes every selected byte before signing.
    """
    sources = _sources(model_directory, artifact_roots)
    return hashlib.sha256(
        identity_json(
            {
                source.member_id: source.record.model_dump(mode="json")
                for source in sources
            }
        )
    ).hexdigest()


def preserve_installed(
    model_directory: Path,
    destination: Path,
    keys: tuple[Ed25519PrivateKey, ...],
    *,
    artifact_roots: tuple[Path, ...] = (),
    retained_card: RetainedCard | None = None,
    control: OperationControl | None = None,
    snapshot_parent: Path | None = None,
    before_publish: Callable[[str], None] | None = None,
) -> str:
    """Snapshot and sign a complete private installed closure into an existing archive.

    Local CLI authority only: paths/keys must never be accepted from a remote
    browser as a host grant. No source files or receiver grants are modified.
    Copies reject changed/truncated/symlinked source files. A private snapshot
    avoids signing a mixed live generation; a future host integration must also
    hold the actual store's eviction/publication lease. This function owns no
    durable background job and must not run on an HTTP event loop. A host may
    supply private local snapshot storage and a durable expected-outcome callback
    that must return successfully before archive publication begins.
    """
    sources = _sources(model_directory, artifact_roots)
    registry_source: RetainedCard | None = None
    if retained_card is None:
        from skulk.store.preservation_provenance import (
            cached_registry_card,
            retained_recovery_card,
            retained_registry_source,
        )

        registry_source = retained_registry_source(model_directory, sources[0].record)
        candidate = retained_recovery_card(model_directory, sources[0].record)
        if candidate is not None:
            try:
                retained_card = _retained(sources[0].record, candidate)
            except PreservationError:
                registry_source = candidate
        if (
            retained_card is None
            and sources[0].record.model_card.registry_card_id is not None
        ):
            candidate = cached_registry_card(
                sources[0].record.model_card.registry_card_id
            )
            if candidate is not None:
                # Auxiliary runtime projections can differ from an original card.
                # Never falsely bind a different installed selection to that proof.
                try:
                    retained_card = _retained(sources[0].record, candidate)
                except PreservationError:
                    registry_source = candidate
                    retained_card = None
    retained = _retained(sources[0].record, retained_card)
    active = control or OperationControl()
    with directory(destination) as archive_descriptor:
        identity = os.fstat(archive_descriptor)
        source_bytes = sum(
            file.size_bytes for source in sources for file in source.record.files
        )
        snapshot_location = snapshot_parent or destination
        snapshot_device = snapshot_location.stat().st_dev
        reserve = 16 * 1024 * 1024
        snapshot_free = shutil.disk_usage(snapshot_location).free
        archive_free = shutil.disk_usage(archive_descriptor).free
        if snapshot_device == identity.st_dev:
            if min(snapshot_free, archive_free) < 2 * source_bytes + reserve:
                raise PreservationCapacityError(
                    "not enough space for a private snapshot and preservation copy"
                )
        else:
            if snapshot_free < source_bytes + reserve:
                raise PreservationCapacityError(
                    "not enough space on private snapshot storage"
                )
            if archive_free < source_bytes + reserve:
                raise PreservationCapacityError(
                    "not enough space on preservation archive storage"
                )
        with tempfile.TemporaryDirectory(
            prefix=".skulk-preserve-", dir=snapshot_parent or destination
        ) as temporary:
            snapshot = Path(temporary)
            source_paths: dict[str, Path] = {}
            members: list[RecipeMember] = []
            observed: list[ManifestMember] = []
            records: dict[str, JsonValue] = {}
            for source in sources:
                active.checkpoint()
                target = snapshot / source.member_id
                target.mkdir(mode=0o700)
                captured: list[FileIdentity] = []
                with directory(source.path) as origin, directory(target) as output:
                    source_identity = os.fstat(origin)
                    for file in source.record.files:
                        path = canonical_path(file.path)
                        with (
                            open_file(origin, path) as incoming,
                            open_file(output, path, write=True) as outgoing,
                        ):
                            copy_exact(
                                incoming, outgoing, file.size_bytes, control=active
                            )
                            outgoing.flush()
                        hasher = hashlib.sha256()
                        with open_file(output, path) as copied:
                            while chunk := copied.read(1024 * 1024):
                                active.checkpoint()
                                hasher.update(chunk)
                        if hasher.hexdigest() != file.sha256:
                            raise PreservationError(
                                "installed bytes changed or do not match their recorded identity"
                            )
                        captured.append(
                            FileIdentity(
                                path=path, size=file.size_bytes, sha256=file.sha256
                            )
                        )
                    now = source.path.stat()
                    if (now.st_dev, now.st_ino) != (
                        source_identity.st_dev,
                        source_identity.st_ino,
                    ) or read_installed_card(source.path) != source.record:
                        raise PreservationError(
                            "installed source generation changed during capture"
                        )
                notices = tuple(
                    file.path
                    for file in captured
                    if Path(file.path).name.lower()
                    in {
                        *(
                            (".foxlight-capture-notice.txt",)
                            if source.record.preservation is not None
                            else ()
                        ),
                        "license",
                        "license.txt",
                        "license.md",
                        "notice",
                        "notice.txt",
                        "readme.md",
                    }
                )
                if not notices:
                    name = ".foxlight-capture-notice.txt"
                    if (target / name).exists():
                        raise PreservationError(
                            "source conflicts with the generated capture notice"
                        )
                    notice = CAPTURE_NOTICE
                    (target / name).write_bytes(notice)
                    captured.append(
                        FileIdentity(
                            path=name,
                            size=len(notice),
                            sha256=hashlib.sha256(notice).hexdigest(),
                        )
                    )
                    notices = (name,)
                record = source.record
                metadata: dict[str, JsonValue] = cast(
                    dict[str, JsonValue],
                    record.model_dump(mode="json", exclude={"preservation"}),
                )
                metadata["schema_version"] = 2 if record.artifact_bundle_id else 1
                records[source.member_id] = {
                    "installed_record": metadata,
                    "prior_recovery": cast(
                        dict[str, JsonValue],
                        record.preservation.model_dump(
                            mode="json", exclude={"repository_code_authorized"}
                        ),
                    )
                    if record.preservation is not None
                    else None,
                }
                member = RecipeMember(
                    member_id=source.member_id,
                    role=record.artifact_role,
                    format=retained.envelope.artifact.format
                    if source.member_id == "primary"
                    else record.artifact_format,
                    origin_repository=record.artifact_repository,
                    origin_revision=record.artifact_revision or "",
                    artifact_identity=retained.envelope.card_id
                    if source.member_id == "primary"
                    else record.installed_identity,
                    loader_root=retained.envelope.artifact.bundle.root
                    if source.member_id == "primary"
                    and retained.envelope.artifact.bundle is not None
                    else None,
                    dependencies=tuple(item.member_id for item in sources[1:])
                    + ("source-evidence",)
                    if source.member_id == "primary"
                    else (),
                    retained_card=cast(
                        dict[str, JsonValue], retained.model_dump(mode="json")
                    )
                    if source.member_id == "primary"
                    else None,
                    intake=Intake(
                        audience="private",
                        license_id="unresolved-owner-capture",
                        terms_reference="retained source notices and original installed observations",
                        decision="Private preservation only; public redistribution requires a separate terms decision.",
                        notice_paths=notices,
                    ),
                    files=tuple(
                        SourceFile(path=file.path, expected_sha256=file.sha256)
                        for file in captured
                    ),
                )
                source_paths[source.member_id] = target
                members.append(member)
                observed.append(
                    ManifestMember.model_validate(
                        {
                            **member.model_dump(exclude={"files"}),
                            "files": tuple(captured),
                        }
                    )
                )
            evidence = snapshot / "source-evidence"
            evidence.mkdir(mode=0o700)
            (evidence / "installed-records.json").write_bytes(identity_json(records))
            (evidence / "NOTICE.txt").write_text(
                "Source installed records are historical observations, not receiver trust or code permission.\n"
            )
            evidence_names = ["installed-records.json", "NOTICE.txt"]
            if registry_source is not None:
                (evidence / "registry-source-card.json").write_bytes(
                    identity_json(registry_source.model_dump(mode="json"))
                )
                evidence_names.append("registry-source-card.json")
            if retained.registry_proof is not None or registry_source is not None:
                # Transport this public anchor as evidence only. A receiver must
                # still independently recognize it before granting registry trust.
                (evidence / "registry-root.json").write_bytes(
                    EMBEDDED_REGISTRY_ROOT.read_bytes()
                )
                evidence_names.append("registry-root.json")
            root = sources[0].record
            members.append(
                RecipeMember(
                    member_id="source-evidence",
                    role="notices",
                    format="text",
                    origin_repository=root.artifact_repository,
                    origin_revision=root.artifact_revision or "",
                    artifact_identity=root.installed_identity + ":observations",
                    intake=Intake(
                        audience="private",
                        license_id="unresolved-owner-capture",
                        terms_reference="original installed record",
                        decision="Retained evidence only; not transferable authorization.",
                        notice_paths=("NOTICE.txt",),
                    ),
                    files=tuple(SourceFile(path=name) for name in evidence_names),
                )
            )
            source_paths["source-evidence"] = evidence
            recipe = Recipe(
                schema_version="foxlight.model-preservation.v1",
                name=str(root.model_card.model_id),
                root_member="primary",
                workloads=tuple(task.value for task in root.model_card.tasks),
                runtime_requirements=(),
                members=tuple(members),
            )
            # Validate runtime mappings/layout before the public packer can publish.
            # Evidence carries no loader mapping and is retained only for inspection.
            evidence_identities = tuple(
                FileIdentity(
                    path=name,
                    size=(evidence / name).stat().st_size,
                    sha256=hashlib.sha256((evidence / name).read_bytes()).hexdigest(),
                )
                for name in evidence_names
            )
            observed.append(
                ManifestMember.model_validate(
                    {
                        **members[-1].model_dump(exclude={"files"}),
                        "files": evidence_identities,
                    }
                )
            )
            manifest = Manifest.model_validate(
                {**recipe.model_dump(exclude={"members"}), "members": tuple(observed)}
            )
            card = recovery_card(manifest)
            inspect_member_evidence(
                observed[0],
                trusted_root=EMBEDDED_REGISTRY_ROOT.read_bytes(),
                mode="historical",
            )
            for item in map_recovery_members(manifest, card):
                validate_recovery_layout(
                    source_paths[item.member.member_id], item, card
                )
            current = destination.stat()
            if (current.st_dev, current.st_ino) != (identity.st_dev, identity.st_ino):
                raise PreservationError(
                    "archive changed during capture; reconnect the selected storage"
                )
            expected = digest(encode_record(manifest))
            if before_publish is not None:
                before_publish(expected)
            active.checkpoint()
            return pack(
                recipe,
                source_paths,
                destination,
                keys,
                control=active,
                expected_manifest_id=expected,
            )
