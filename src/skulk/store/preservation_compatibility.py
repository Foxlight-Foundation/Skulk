"""Exact reviewed reuse of an already installed owned companion."""

from __future__ import annotations

import hashlib

from skulk.store.installed_cards import InstalledCardRecord
from skulk.store.preservation_export import CAPTURE_NOTICE, public_card_body


def compatible_companion(
    existing: InstalledCardRecord, selected: InstalledCardRecord
) -> bool:
    """Require the same owner, artifact, bytes and intrinsic runtime contract.

    Receiver publication/permission observations are rebuilt for the complete
    generation. An unrelated alias or another model's companion cannot be taken.
    """
    if (
        existing.artifact_role == "base"
        or existing.artifact_role != selected.artifact_role
        or existing.owner_model_id != selected.owner_model_id
    ):
        return False
    if (
        existing.artifact_repository,
        existing.artifact_revision,
        existing.artifact_file,
        existing.artifact_bundle_id,
    ) != (
        selected.artifact_repository,
        selected.artifact_revision,
        selected.artifact_file,
        selected.artifact_bundle_id,
    ):
        return False
    if not compatible_file_tables(
        {(file.path, file.size_bytes, file.sha256) for file in existing.files},
        {(file.path, file.size_bytes, file.sha256) for file in selected.files},
    ):
        return False
    old = existing.model_card.model_copy(
        update={"source_repository": existing.model_card.artifact_repository}
    )
    new = selected.model_card.model_copy(
        update={"source_repository": selected.model_card.artifact_repository}
    )
    return public_card_body(old) == public_card_body(new)


def compatible_file_tables(
    existing: set[tuple[str, int, str]], selected: set[tuple[str, int, str]]
) -> bool:
    """Allow only the exact host-generated missing-attribution notice to be added."""
    notice = (
        ".foxlight-capture-notice.txt",
        len(CAPTURE_NOTICE),
        hashlib.sha256(CAPTURE_NOTICE).hexdigest(),
    )
    return existing == selected or existing | {notice} == selected
