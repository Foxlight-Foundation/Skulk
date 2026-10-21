"""Local restore consent receipts; never accepted from a portable bundle."""

from __future__ import annotations

import os
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from skulk.store.installed_cards import verify_installed_card

if TYPE_CHECKING:
    from skulk.store.model_store import StoreModelEntry


class RecoveryAdmission(BaseModel):
    """Receiver-created consent bound to exact deletion stamps and a whole closure."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    schema_version: Literal[1] = Field(default=1, description="Local receipt format.")
    manifest_id: str = Field(
        pattern=r"^[0-9a-f]{64}$", description="Exact restored capture."
    )
    members: tuple[str, ...] = Field(
        min_length=1, max_length=16, description="Every committed artifact alias."
    )
    deletions: dict[str, str] = Field(
        max_length=16,
        description="Exact receiver deletion stamps explicitly acknowledged.",
    )


def write_recovery_admission(
    home: Path, manifest_id: str, members: tuple[str, ...], deletions: dict[str, str]
) -> None:
    """Write only host-authored consent inside a private prepared home."""
    receipt = RecoveryAdmission(
        manifest_id=manifest_id, members=members, deletions=deletions
    )
    descriptor = os.open(
        home / "recovery-admission.json",
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
    with os.fdopen(descriptor, "w") as stream:
        _ = stream.write(receipt.model_dump_json())
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(home, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def committed_deletion_acknowledgements(
    store: Path, entries: Mapping[str, StoreModelEntry], deletions: Mapping[str, str]
) -> frozenset[str]:
    """Reconcile consent only after every member appears in the committed index.

    A newly deleted alias has a new stamp and cannot reuse an older receipt.
    Unindexed staging, peer sidecars and evidence carried in kits grant nothing.
    """
    acknowledged: set[str] = set()
    generations: set[str] = set()
    for entry in entries.values():
        parts = Path(entry.store_path).parts
        if (
            len(parts) == 4
            and parts[0] == ".recoveries"
            and parts[2] == "model-store"
            and len(parts[1]) == 64
            and all(c in "0123456789abcdef" for c in parts[1])
        ):
            generations.add(parts[1])
    for generation in generations:
        home = store / ".recoveries" / generation
        try:
            descriptor = os.open(
                home / "recovery-admission.json",
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            )
            with os.fdopen(descriptor, "rb") as stream:
                info = os.fstat(stream.fileno())
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_uid != os.getuid()
                    or info.st_mode & 0o077
                ):
                    continue
                data = stream.read(1024 * 1024 + 1)
            if len(data) > 1024 * 1024:
                continue
            receipt = RecoveryAdmission.model_validate_json(data)
            if not set(receipt.deletions).issubset(receipt.members):
                continue
            complete = True
            for alias in receipt.members:
                current = entries.get(alias)
                if (
                    current is None
                    or current.installed_card is None
                    or current.installed_card.preservation is None
                    or current.installed_card.preservation.manifest_id
                    != receipt.manifest_id
                ):
                    complete = False
                    break
                path = store / current.store_path
                if path.parent != home / "model-store" or not verify_installed_card(
                    path, current.installed_card
                ):
                    complete = False
                    break
            if complete:
                acknowledged.update(
                    alias
                    for alias, stamp in receipt.deletions.items()
                    if deletions.get(alias) == stamp
                )
        except (OSError, ValueError):
            continue
    return frozenset(acknowledged)
