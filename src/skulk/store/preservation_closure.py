"""Map a signed preservation closure to the normal runtime's artifact identities."""

from __future__ import annotations

from dataclasses import dataclass

from foxlight_model_preservation.models import (
    Manifest,
    ManifestMember,
    PreservationError,
)

from skulk.download.download_utils import companion_download_specs
from skulk.shared.models.model_cards import ModelCard
from skulk.store.installed_cards import InstalledArtifactRole, companion_artifact_role


@dataclass(frozen=True)
class RecoveryMember:
    """One receiver-selected loader mapping, derived from card truth, not a path hint."""

    member: ManifestMember
    role: InstalledArtifactRole
    model_id: str
    selection: ModelCard


def companion_selections(
    card: ModelCard,
) -> tuple[tuple[InstalledArtifactRole, ModelCard], ...]:
    """Resolve every supported declared companion, refusing unsupported offline paths.

    This first native export includes optional speculation companions as well.
    Missing optional bytes are therefore a preparation error, not a silent
    reduction of the selected preservation workload.
    """
    card.require_immutable_external_companions(context="preservation companions")
    if card.vision is not None and card.vision.processor_repo:
        # The current vision processor loader still addresses the Hub directly;
        # placing a folder in the model store would not make that path offline.
        raise PreservationError(
            "separate vision processor repositories need an offline loader adapter before preservation recovery"
        )
    if card.runtime is not None and card.runtime.vllm_spec_draft_repo:
        raise PreservationError(
            "vLLM draft repositories need an offline loader adapter before preservation recovery"
        )
    selected_list: list[tuple[InstalledArtifactRole, ModelCard]] = []
    for shard, _, _ in companion_download_specs(card):
        selected_list.append(
            (
                companion_artifact_role(
                    card, str(shard.model_card.artifact_repository)
                ),
                shard.model_card,
            )
        )
    selected = tuple(selected_list)
    aliases = [str(selection.model_id) for _, selection in selected]
    if len(set(aliases)) != len(aliases) or str(card.model_id) in aliases:
        raise PreservationError(
            "companions with overlapping store identities need a distinct closure mapping"
        )
    return selected


def map_recovery_members(
    manifest: Manifest, card: ModelCard
) -> tuple[RecoveryMember, ...]:
    """Require exactly the card's selected data closure and refuse hidden loader members."""
    root = next(
        member
        for member in manifest.members
        if member.member_id == manifest.root_member
    )
    if root.role not in {"base", "primary"}:
        raise PreservationError("the recovery root must be a base model")
    result = [RecoveryMember(root, "base", str(card.model_id), card)]
    used = {root.member_id}
    for role, selection in companion_selections(card):
        matches = [
            member
            for member in manifest.members
            if member.role == role
            and member.origin_repository == str(selection.artifact_repository)
            and member.origin_revision == selection.source_revision
        ]
        if len(matches) != 1:
            raise PreservationError(
                f"required companions are missing or ambiguous: {role} {selection.artifact_repository}"
            )
        member = matches[0]
        expected_format = "gguf" if selection.gguf_file is not None else None
        if expected_format is not None and member.format != expected_format:
            raise PreservationError(
                "companion format disagrees with its selected artifact"
            )
        if member.loader_root != (
            selection.artifact_bundle.root
            if selection.artifact_bundle is not None
            else None
        ):
            raise PreservationError(
                "companion loader root differs from the card's selection"
            )
        used.add(member.member_id)
        result.append(RecoveryMember(member, role, str(selection.model_id), selection))
    for member in manifest.members:
        if member.member_id not in used and member.role != "notices":
            raise PreservationError(
                "capture contains an undeclared companion or unsupported closure member"
            )
    normalized = [item.selection.model_id.normalize().casefold() for item in result]
    if len(set(normalized)) != len(normalized):
        raise PreservationError("recovery members have colliding store paths")
    return tuple(result)
