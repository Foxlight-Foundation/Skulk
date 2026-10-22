"""Recover available original signed evidence without network access or enrollment."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from filelock import FileLock, Timeout
from foxlight_model_preservation.assurance import (
    RetainedCard,
    load_retained_card,
    retain_card,
)
from foxlight_model_preservation.cards import identity_json
from foxlight_model_preservation.registry_proof import (
    create_registry_proof,
    verify_registry_proof,
)
from foxlight_model_preservation.signing import (
    encode_record,
    load_trust,
    strict_json,
)

from skulk.shared.constants import SKULK_MODEL_REGISTRY_CACHE_DIR
from skulk.shared.models.registry import EMBEDDED_REGISTRY_ROOT
from skulk.store.installed_cards import InstalledCardRecord
from skulk.store.preservation import read_recovery_metadata
from skulk.store.preservation_pmf import read_recovered_capture


def retained_recovery_card(
    model_directory: Path, record: InstalledCardRecord
) -> RetainedCard | None:
    """Reuse exact retained source evidence when re-preserving a recovered model.

    No runtime re-serialization replaces the signed original. A peer installation
    without the original evidence directory cannot invent a registry proof.
    """
    if record.preservation is None:
        return None
    evidence = model_directory.parent.parent / "preservation"
    envelope_path = evidence / "envelope.json"
    trust_path = evidence / "owner-trust.json"
    if not envelope_path.is_file() or not trust_path.is_file():
        return None
    # Recovery retains materialized members, not a portable object archive.

    trust = load_trust(read_recovery_metadata(trust_path))
    manifest = read_recovered_capture(evidence, record.preservation.manifest_id, trust)
    member = next(
        item for item in manifest.members if item.member_id == manifest.root_member
    )
    if member.retained_card is None:
        return None
    return load_retained_card(identity_json(member.retained_card))


def cached_registry_card(
    card_id: str, *, cache: Path | None = None, root: Path | None = None
) -> RetainedCard | None:
    """Retain a matching exact cached catalog and TUF chain, or report absence.

    Cache labels are not authority: every returned proof is independently checked
    against the packaged receiver root. Missing/stale/mismatched cache evidence
    cannot prevent owner-only preservation or be promoted to registry provenance.
    """
    cache = cache or Path(str(SKULK_MODEL_REGISTRY_CACHE_DIR))
    root = root or EMBEDDED_REGISTRY_ROOT
    if not cache.is_dir():
        return None
    try:
        with FileLock(str(cache / "refresh.lock"), timeout=2):
            anchor = read_recovery_metadata(root)
            parsed = strict_json(anchor)
            if not isinstance(parsed, dict):
                return None
            signed = parsed.get("signed")
            if not isinstance(signed, dict):
                return None
            version = signed.get("version")
            if not isinstance(version, int):
                return None
            metadata = cache / "metadata"
            roots: list[tuple[int, Path]] = []
            history = metadata / "root_history"
            if history.is_dir():
                for path in history.iterdir():
                    match = re.fullmatch(r"([0-9]+)\.root\.json", path.name)
                    if match and int(match[1]) > version:
                        roots.append((int(match[1]), path))
            if len(roots) > 32:
                return None
            proof = create_registry_proof(
                trusted_root=anchor,
                root_chain=tuple(
                    read_recovery_metadata(path) for _, path in sorted(roots)
                ),
                timestamp=read_recovery_metadata(metadata / "timestamp.json"),
                snapshot=read_recovery_metadata(metadata / "snapshot.json"),
                targets=read_recovery_metadata(metadata / "targets.json"),
                target_path="v1/catalog.json",
                target=read_recovery_metadata(cache / "last-known-good-catalog.json"),
            )
            checked = verify_registry_proof(
                encode_record(proof), anchor, card_id, mode="historical"
            )
            return retain_card(checked.card, registry_proof=proof)
    except (OSError, ValueError, Timeout):
        return None


def retained_registry_source(
    model_directory: Path, record: InstalledCardRecord
) -> RetainedCard | None:
    """Carry an original card alongside a later derived runtime view, unchanged.

    Additional original evidence is read only when the retained signed manifest
    binds that exact file. It grants no automatic publication status to the view.
    """
    if record.preservation is None:
        return None
    evidence = model_directory.parent.parent / "preservation"
    if (
        not (evidence / "envelope.json").is_file()
        or not (evidence / "owner-trust.json").is_file()
    ):
        return None
    trust = load_trust(read_recovery_metadata(evidence / "owner-trust.json"))
    manifest = read_recovered_capture(evidence, record.preservation.manifest_id, trust)
    for member in manifest.members:
        file = next(
            (item for item in member.files if item.path == "registry-source-card.json"),
            None,
        )
        if member.member_id != "source-evidence" or file is None:
            continue
        path = evidence / "source-evidence" / member.member_id / file.path
        if not path.is_file():
            return None
        data = read_recovery_metadata(path)
        if len(data) != file.size or hashlib.sha256(data).hexdigest() != file.sha256:
            from foxlight_model_preservation.models import PreservationError

            raise PreservationError(
                "retained original source evidence failed integrity verification"
            )
        return load_retained_card(data)
    return None
