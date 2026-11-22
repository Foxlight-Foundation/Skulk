"""What engine packages a node holds, read from disk without starting anything.

For every engine lane under the packages root this reports the active
generation after the installer's quick check (names, sizes and executable
bits against the install-time inventory; no hashing, no trust document, no
clock). A lane whose active generation is missing or damaged is reported as
such instead of being dropped, so an operator can see why a backend is not
offered.

A backend tag is ``<engine name>-<lane>``. The engine name is only the last
part of the engine ID, so two publishers could claim one tag. A tag claimed
twice is offered by neither; both lanes are reported as ambiguous.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, cast, final, get_args

from pydantic import TypeAdapter, ValidationError

from skulk.provisioning.engine_install import (
    ENGINE_PACKAGES_DIR,
    InstalledEngine,
    active_engine,
    verify_installed_engine,
)
from skulk.provisioning.engine_packages import (
    EngineId,
    EngineLane,
    EnginePackageRefusal,
    EnginePackageRefusedError,
)
from skulk.shared.backends import engine_packages_disabled, package_engine_of

UnusableReason = EnginePackageRefusal | Literal["backend_tag_ambiguous"]
"""Why a lane's active generation is not offered: the installer's refusal, or
a backend tag that is reserved, built in, or claimed by another engine ID."""

_ENGINE_ID: Final[TypeAdapter[str]] = TypeAdapter(EngineId)
_LANES: Final[frozenset[str]] = frozenset(get_args(EngineLane))
_MAX_ENGINES: Final = 64
_DIGEST: Final = re.compile(r"^[0-9a-f]{64}$")


@final
@dataclass(frozen=True)
class UnusableEnginePackage:
    """An engine lane whose active generation cannot be offered.

    Attributes:
        engine_id: The ``publisher.engine`` identity.
        lane: The accelerator class.
        reason: The installer's refusal code, or ``backend_tag_ambiguous``
            when a built-in engine or another engine ID on this node claims
            the same backend tag.
    """

    engine_id: str
    lane: EngineLane
    reason: UnusableReason


@final
@dataclass(frozen=True)
class EnginePackageInventory:
    """The engine packages on one node.

    Attributes:
        active: The usable active generation of each backend tag.
        unusable: Lanes that have an active generation which cannot be used.
    """

    active: dict[str, InstalledEngine]
    unusable: tuple[UnusableEnginePackage, ...]

    @property
    def builds(self) -> dict[str, str]:
        """Each offered backend tag's build identity (its manifest digest)."""
        return {tag: installed.digest for tag, installed in self.active.items()}


def _valid_engine_id(name: str) -> bool:
    try:
        _ENGINE_ID.validate_python(name)
    except ValidationError:
        return False
    return True


def read_engine_package_inventory(
    *, root: Path = ENGINE_PACKAGES_DIR
) -> EnginePackageInventory:
    """Read the active generation of every installed engine lane.

    Directories that are not a valid engine ID or lane are ignored: they are
    not packages. Lanes with generations but no active one are not reported;
    nothing is offered from them and nothing is wrong with them.

    With engine packages turned off (``SKULK_ENGINE_PACKAGES_DISABLE=1``) the
    root is not read and the inventory is empty, whatever is installed. Every
    reader of a node's packages goes through here, so the node then offers
    and runs none.
    """
    if engine_packages_disabled():
        return EnginePackageInventory(active={}, unusable=())
    found: list[InstalledEngine] = []
    unusable: list[UnusableEnginePackage] = []
    if root.is_dir():
        engine_directories = sorted(
            entry
            for entry in root.iterdir()
            if entry.is_dir()
            and not entry.is_symlink()
            and _valid_engine_id(entry.name)
        )[:_MAX_ENGINES]
        for engine_directory in engine_directories:
            for lane_directory in sorted(engine_directory.iterdir()):
                if (
                    lane_directory.name not in _LANES
                    or not lane_directory.is_dir()
                    or lane_directory.is_symlink()
                ):
                    continue
                lane = cast(EngineLane, lane_directory.name)
                try:
                    installed = active_engine(engine_directory.name, lane, root=root)
                except EnginePackageRefusedError as refused:
                    unusable.append(
                        UnusableEnginePackage(engine_directory.name, lane, refused.code)
                    )
                    continue
                if installed is None:
                    continue
                claims = installed.claims
                if claims.engine_id != engine_directory.name or claims.lane != lane:
                    # A generation moved by hand into another engine's lane.
                    unusable.append(
                        UnusableEnginePackage(
                            engine_directory.name, lane, "installation_damaged"
                        )
                    )
                    continue
                found.append(installed)

    claimed: dict[str, list[InstalledEngine]] = {}
    for installed in found:
        claimed.setdefault(installed.claims.backend_tag, []).append(installed)
    active: dict[str, InstalledEngine] = {}
    for tag, claimants in claimed.items():
        # A package may never take a built-in engine's tag or a reserved name.
        if len(claimants) == 1 and package_engine_of(tag) is not None:
            active[tag] = claimants[0]
            continue
        unusable.extend(
            UnusableEnginePackage(
                claimant.claims.engine_id, claimant.claims.lane, "backend_tag_ambiguous"
            )
            for claimant in claimants
        )
    return EnginePackageInventory(active=active, unusable=tuple(unusable))


def installed_engine_for_backend(
    backend_tag: str,
    build_digest: str | None,
    *,
    root: Path = ENGINE_PACKAGES_DIR,
) -> InstalledEngine | None:
    """Find the installed generation a stamped placement names.

    A placement is stamped with a backend tag and the manifest digest that was
    active when it was made. Activating another generation later must not move
    a placement onto a build it was not admitted for, so a stamped digest is
    looked up exactly, among every generation of the lane, and given the same
    quick check as an active one. Without a stamp the active generation is
    used.

    Returns ``None`` when no installed package offers the tag or the stamped
    generation is no longer installed.

    Raises:
        EnginePackageRefusedError: ``installation_damaged`` when the named
            generation is present but differs from its inventory.
    """
    offered = read_engine_package_inventory(root=root).active.get(backend_tag)
    if offered is None:
        return None
    if build_digest is None or build_digest == offered.digest:
        return offered
    if _DIGEST.fullmatch(build_digest) is None:
        return None
    generation = offered.root.parent / build_digest
    if not generation.is_dir() or generation.is_symlink():
        return None
    return verify_installed_engine(generation, deep=False)
