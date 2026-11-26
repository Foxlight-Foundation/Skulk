"""The three changes a person may make to an engine package on one node.

Activate makes an installed build the one new launches use, roll back returns
to the build that was active before, and remove deletes one build or the whole
package. Each is a change to the node's own packages folder, made by that
node's worker, which is the only thing that knows what is running there.

Nothing here installs anything, starts an engine or touches a running one. A
launch that is already running keeps the build it was started from, which is
why a build something still runs on cannot be removed.
"""

from collections.abc import Container
from pathlib import Path

from skulk.engines.inventory import engine_build_fit_refusal
from skulk.provisioning.engine_install import (
    ENGINE_PACKAGES_DIR,
    InstalledEngine,
    activate_engine_generation,
    active_engine_generation,
    installed_engine_generation,
    installed_engine_generations,
    remove_engine_generation,
    remove_engine_package,
    rollback_engine,
    rollback_engine_generation,
)
from skulk.provisioning.engine_packages import (
    EngineLane,
    EnginePackageRefusedError,
)
from skulk.shared.backends import (
    engine_packages_disabled,
    engine_packages_off_phrase,
)
from skulk.shared.types.engine_packages import EnginePackageAction


def _require_runnable(
    installed: InstalledEngine, engine_id: str, lane: EngineLane
) -> None:
    """Refuse a build that would leave the lane with nothing a launch can use."""
    claims = installed.claims
    if claims.engine_id != engine_id or claims.lane != lane:
        raise EnginePackageRefusedError(
            "installation_damaged",
            "that build belongs to another engine or accelerator class",
        )
    unfit = engine_build_fit_refusal(claims)
    if unfit is not None:
        raise EnginePackageRefusedError(
            unfit, "this Skulk version cannot run that build"
        )


def change_engine_package(
    engine_id: str,
    lane: EngineLane,
    action: EnginePackageAction,
    build: str | None,
    *,
    builds_in_use: Container[str],
    root: Path = ENGINE_PACKAGES_DIR,
) -> None:
    """Activate, roll back or remove an engine package build on this node.

    Args:
        engine_id: The ``publisher.engine`` identity.
        lane: The accelerator class of the package.
        action: What to do.
        build: The manifest digest to activate or remove. Remove without
            one removes the whole package. For a roll back it is the build
            the caller expects to return to: the request is then safe to
            read twice, and is refused if the node's earlier build is
            another one. A roll back without it returns to whatever the
            node kept.
        builds_in_use: The builds of this lane that something on the node is
            running on right now.
        root: Root of installed engine packages.

    Raises:
        EnginePackageRefusedError: ``packages_disabled`` when engine packages
            are turned off here; ``generation_missing`` when the named build,
            an earlier build or the package is not installed;
            ``installation_damaged`` when the build is not intact;
            ``skulk_version_unsupported``, ``protocol_unsupported`` or
            ``adapter_python_unsupported`` when this Skulk cannot run the
            build asked for; ``generation_active`` when asked to remove the
            build new launches use; ``generation_in_use`` when something is
            running on a build that would be removed. Nothing is changed in
            any of these cases.

    Side effects:
        Rewrites the lane's active-build record or deletes build directories
        under ``root``, holding the lane's install lock.
    """
    if engine_packages_disabled():
        raise EnginePackageRefusedError(
            "packages_disabled",
            f"engine packages are turned off on this node ({engine_packages_off_phrase()})",
        )
    missing = EnginePackageRefusedError(
        "generation_missing", "that engine build is not installed on this node"
    )
    in_use = EnginePackageRefusedError(
        "generation_in_use", "a model is running on that engine build"
    )
    if action == "activate":
        if build is None:
            raise missing
        _require_runnable(
            installed_engine_generation(engine_id, lane, build, root=root),
            engine_id,
            lane,
        )
        activate_engine_generation(engine_id, lane, build, root=root)
    elif action == "rollback":
        if build is not None and build == active_engine_generation(
            engine_id, lane, root=root
        ):
            # The same request read a second time: the swap was already made,
            # and making it again would undo it.
            return
        earlier = rollback_engine_generation(engine_id, lane, root=root)
        if earlier is None or (build is not None and build != earlier):
            raise EnginePackageRefusedError(
                "generation_missing", "there is no such earlier build to return to"
            )
        _require_runnable(
            installed_engine_generation(engine_id, lane, earlier, root=root),
            engine_id,
            lane,
        )
        rollback_engine(engine_id, lane, root=root)
    elif build is not None:
        if build in builds_in_use:
            raise in_use
        remove_engine_generation(engine_id, lane, build, root=root)
    else:
        held = installed_engine_generations(engine_id, lane, root=root)
        if any(digest in builds_in_use for digest in held):
            raise in_use
        remove_engine_package(engine_id, lane, root=root)
