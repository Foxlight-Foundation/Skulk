"""Activate, roll back and remove an engine package build on one node's disk."""

from __future__ import annotations

import re
from pathlib import Path
from typing import get_args

import pytest
from pydantic import TypeAdapter, ValidationError

from skulk.engines.actions import change_engine_package
from skulk.engines.inventory import read_engine_package_inventory
from skulk.engines.tests.package_fixtures import install_test_package
from skulk.provisioning.engine_install import (
    active_engine,
    installed_engine_generations,
    remove_engine_package,
    rollback_engine_generation,
)
from skulk.provisioning.engine_packages import (
    EngineId,
    EngineLane,
    EnginePackageRefusedError,
)
from skulk.shared.backends import ENGINE_PACKAGES_DISABLE_ENV
from skulk.shared.types.engine_packages import (
    ENGINE_ID_PATTERN,
    EnginePackageAction,
    EnginePackageLane,
)

ENGINE = "foxlight.tensorfold"
LANE: EngineLane = "rocm"
NOTHING: frozenset[str] = frozenset()


@pytest.fixture
def packages(tmp_path: Path) -> Path:
    return tmp_path / "packages"


def _two_builds(tmp_path: Path, packages: Path) -> tuple[str, str]:
    """Install an active first build and a second one that is only on disk."""
    first = install_test_package(tmp_path, packages, generation=1)
    second = install_test_package(tmp_path, packages, generation=2, activate=False)
    return first.digest, second.digest


def _active(packages: Path) -> str | None:
    installed = active_engine(ENGINE, LANE, root=packages)
    return None if installed is None else installed.digest


def _change(
    packages: Path,
    action: EnginePackageAction,
    build: str | None = None,
    *,
    in_use: frozenset[str] = NOTHING,
) -> None:
    change_engine_package(
        ENGINE, LANE, action, build, builds_in_use=in_use, root=packages
    )


def _refused(
    packages: Path,
    action: EnginePackageAction,
    build: str | None = None,
    *,
    in_use: frozenset[str] = NOTHING,
) -> str:
    with pytest.raises(EnginePackageRefusedError) as refused:
        _change(packages, action, build, in_use=in_use)
    return refused.value.code


def test_activating_a_build_keeps_the_earlier_one_for_rollback(
    tmp_path: Path, packages: Path
) -> None:
    first, second = _two_builds(tmp_path, packages)

    _change(packages, "activate", second)

    assert _active(packages) == second
    assert rollback_engine_generation(ENGINE, LANE, root=packages) == first
    assert read_engine_package_inventory(root=packages).builds == {
        "tensorfold-rocm": second
    }


def test_a_rollback_swaps_the_two_builds_and_a_second_one_undoes_it(
    tmp_path: Path, packages: Path
) -> None:
    first, second = _two_builds(tmp_path, packages)
    _change(packages, "activate", second)

    _change(packages, "rollback")
    assert _active(packages) == first
    assert rollback_engine_generation(ENGINE, LANE, root=packages) == second

    _change(packages, "rollback")
    assert _active(packages) == second


def test_a_rollback_that_names_its_build_is_safe_to_read_twice(
    tmp_path: Path, packages: Path
) -> None:
    """A request replayed from the log must not swap the builds back."""
    first, second = _two_builds(tmp_path, packages)
    _change(packages, "activate", second)

    _change(packages, "rollback", first)
    _change(packages, "rollback", first)

    assert _active(packages) == first
    assert rollback_engine_generation(ENGINE, LANE, root=packages) == second
    # The node's earlier build is no longer the one a stale caller expects.
    assert _refused(packages, "rollback", "f" * 64) == "generation_missing"
    assert _active(packages) == first


def test_a_build_that_is_not_there_changes_nothing(
    tmp_path: Path, packages: Path
) -> None:
    first, _ = _two_builds(tmp_path, packages)

    assert _refused(packages, "activate", "f" * 64) == "generation_missing"
    assert _refused(packages, "activate") == "generation_missing"
    assert _refused(packages, "rollback") == "generation_missing"
    assert _refused(packages, "remove", "f" * 64) == "generation_missing"
    assert _active(packages) == first


def test_a_build_this_skulk_cannot_run_is_not_activated(
    tmp_path: Path, packages: Path
) -> None:
    """Activating it would leave the lane with nothing a launch can use."""
    first = install_test_package(tmp_path, packages, generation=1)
    unfit = install_test_package(
        tmp_path, packages, generation=2, protocol=(9, 9), activate=False
    )

    assert _refused(packages, "activate", unfit.digest) == "protocol_unsupported"
    assert _active(packages) == first.digest
    assert rollback_engine_generation(ENGINE, LANE, root=packages) is None


def test_a_build_from_another_lane_is_not_activated(
    tmp_path: Path, packages: Path
) -> None:
    first, _ = _two_builds(tmp_path, packages)
    other = install_test_package(tmp_path, packages, lane="cuda", generation=3)
    moved = packages / ENGINE / LANE / other.digest
    other.root.rename(moved)

    assert _refused(packages, "activate", other.digest) == "installation_damaged"
    assert _active(packages) == first


def test_one_build_is_removed_unless_it_is_active_or_running(
    tmp_path: Path, packages: Path
) -> None:
    first, second = _two_builds(tmp_path, packages)

    assert _refused(packages, "remove", first) == "generation_active"
    assert (
        _refused(packages, "remove", second, in_use=frozenset({second}))
        == "generation_in_use"
    )
    assert installed_engine_generations(ENGINE, LANE, root=packages) == tuple(
        sorted((first, second))
    )

    _change(packages, "remove", second)

    assert installed_engine_generations(ENGINE, LANE, root=packages) == (first,)
    assert _active(packages) == first


def test_removing_the_rollback_build_clears_it_from_the_record(
    tmp_path: Path, packages: Path
) -> None:
    first, second = _two_builds(tmp_path, packages)
    _change(packages, "activate", second)

    _change(packages, "remove", first)

    assert rollback_engine_generation(ENGINE, LANE, root=packages) is None
    assert _refused(packages, "rollback") == "generation_missing"


def test_the_whole_package_is_removed_and_the_node_then_holds_none(
    tmp_path: Path, packages: Path
) -> None:
    _two_builds(tmp_path, packages)
    other_lane = install_test_package(tmp_path, packages, lane="cuda", generation=3)

    _change(packages, "remove")

    assert installed_engine_generations(ENGINE, LANE, root=packages) == ()
    assert _active(packages) is None
    # Another lane of the same engine is a different package and stays.
    assert read_engine_package_inventory(root=packages).builds == {
        "tensorfold-cuda": other_lane.digest
    }
    assert _refused(packages, "remove") == "generation_missing"


def test_the_whole_package_stays_while_a_model_runs_on_any_of_its_builds(
    tmp_path: Path, packages: Path
) -> None:
    first, second = _two_builds(tmp_path, packages)

    assert _refused(packages, "remove", in_use=frozenset({second})) == "generation_in_use"

    assert _active(packages) == first
    assert len(installed_engine_generations(ENGINE, LANE, root=packages)) == 2


def test_a_package_with_an_unreadable_record_can_still_be_removed(
    tmp_path: Path, packages: Path
) -> None:
    """Removing is how a damaged package is cleared, so it must not need a good record."""
    _two_builds(tmp_path, packages)
    (packages / ENGINE / LANE / "active.json").write_text("not a record")

    remove_engine_package(ENGINE, LANE, root=packages)

    assert installed_engine_generations(ENGINE, LANE, root=packages) == ()
    assert read_engine_package_inventory(root=packages).unusable == ()


def test_a_package_that_was_never_installed_is_not_there_to_remove(
    packages: Path,
) -> None:
    assert _refused(packages, "remove") == "generation_missing"
    assert not (packages / ENGINE).exists()


@pytest.mark.parametrize("action", get_args(EnginePackageAction))
def test_nothing_changes_while_packages_are_turned_off(
    tmp_path: Path,
    packages: Path,
    monkeypatch: pytest.MonkeyPatch,
    action: EnginePackageAction,
) -> None:
    first, second = _two_builds(tmp_path, packages)
    monkeypatch.setenv(ENGINE_PACKAGES_DISABLE_ENV, "1")

    assert _refused(packages, action, second) == "packages_disabled"

    monkeypatch.delenv(ENGINE_PACKAGES_DISABLE_ENV)
    assert _active(packages) == first
    assert len(installed_engine_generations(ENGINE, LANE, root=packages)) == 2


def test_the_shared_names_match_what_a_signed_manifest_admits() -> None:
    """Commands and events restate the manifest's vocabulary; they must not drift."""
    assert get_args(EnginePackageLane) == get_args(EngineLane)
    manifest_engine_id: TypeAdapter[str] = TypeAdapter(EngineId)
    for name in (
        "foxlight.tensorfold",
        "acme-labs.other_engine2",
        "foxlight",
        "foxlight.tensor-fold",
        "Foxlight.tensorfold",
        "foxlight..tensorfold",
        "../foxlight.tensorfold",
        "foxlight.tensorfold/x",
    ):
        try:
            manifest_engine_id.validate_python(name)
        except ValidationError:
            admitted = False
        else:
            admitted = True
        assert (re.fullmatch(ENGINE_ID_PATTERN, name) is not None) is admitted, name
