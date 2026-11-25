# pyright: reportPrivateUsage=false
"""The doctor's add-on engines check, and what ``--fix`` does about a finding."""

from __future__ import annotations

from pathlib import Path

import pytest

from skulk.doctor.checks import (
    REGISTRY,
    CheckResult,
    _check_engine_packages,
    _fix_engine_packages,
)
from skulk.engines.inventory import node_engine_package_report
from skulk.engines.tests.package_fixtures import install_test_package
from skulk.facts.testing import make_facts
from skulk.provisioning.engine_install import (
    InstalledEngine,
    active_engine_generation,
    installed_engine_generations,
)
from skulk.shared.backends import ENGINE_PACKAGES_DISABLE_ENV
from skulk.shared.types.engine_packages import NodeEnginePackage
from skulk.shared.types.node_facts import NodeFacts

ENGINE = "foxlight.tensorfold"
NAME = "foxlight.tensorfold (rocm)"


@pytest.fixture
def packages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """This node's packages folder, which the doctor is pointed at."""
    root = tmp_path / "packages"
    monkeypatch.setattr("skulk.doctor.checks._engine_packages_root", lambda: root)
    monkeypatch.delenv(ENGINE_PACKAGES_DISABLE_ENV, raising=False)
    return root


def _facts(packages: Path) -> NodeFacts:
    """The facts a node gathers about the packages folder as it is now."""
    return make_facts().model_copy(
        update={"engine_packages": node_engine_package_report(root=packages).packages}
    )


def _only(results: list[CheckResult] | tuple[CheckResult, ...]) -> CheckResult:
    assert len(results) == 1, results
    return results[0]


def _check(packages: Path) -> list[CheckResult]:
    return list(_check_engine_packages(_facts(packages)))


def _lose_a_file(installed: InstalledEngine) -> None:
    """Damage a build so the quick check notices."""
    installed.adapter.unlink()


def _change_a_file(installed: InstalledEngine) -> None:
    """Damage a build so only hashing notices: same name, size and mode."""
    content = installed.executable.read_bytes()
    mode = installed.executable.stat().st_mode
    _ = installed.executable.write_bytes(bytes(reversed(content)))
    installed.executable.chmod(mode)


def test_the_check_is_registered_with_a_repair() -> None:
    check = next(check for check in REGISTRY if check.check_id == "engine-packages")
    assert check.run is _check_engine_packages
    assert check.fix is _fix_engine_packages


def test_a_node_with_no_add_on_engine_passes(packages: Path) -> None:
    result = _only(_check(packages))

    assert (result.verdict, result.detail) == ("ok", "none installed")
    assert _fix_engine_packages(_facts(packages)) is None
    assert not packages.exists()


def test_an_engine_that_is_only_offered_is_not_a_finding(packages: Path) -> None:
    offered = NodeEnginePackage(
        engine_id=ENGINE,
        lane="rocm",
        backend_tag="tensorfold-rocm",
        status="offered",
        build="a" * 64,
        channel="stable",
        download_bytes=1024,
    )
    facts = make_facts().model_copy(update={"engine_packages": (offered,)})

    result = _only(list(_check_engine_packages(facts)))

    assert result.verdict == "ok"
    assert result.detail == (
        "none installed; 1 more can be installed the first time a model needs one"
    )
    assert _fix_engine_packages(facts) is None


def test_a_usable_engine_is_named_with_its_version(
    tmp_path: Path, packages: Path
) -> None:
    installed = install_test_package(tmp_path, packages)

    result = _only(_check(packages))

    assert result.verdict == "ok"
    assert result.detail == f"usable: {NAME} {installed.claims.engine_version}"
    assert _fix_engine_packages(_facts(packages)) is None


def test_a_damaged_engine_with_an_earlier_build_is_switched_to_it(
    tmp_path: Path, packages: Path
) -> None:
    first = install_test_package(tmp_path, packages, generation=1)
    second = install_test_package(tmp_path, packages, generation=2)
    _lose_a_file(second)

    result = _only(_check(packages))

    assert result.verdict == "degraded"
    assert result.title == f"Add-on engine: {NAME}"
    assert result.detail == (
        f"{NAME}: the build new launches use has missing or changed files"
    )
    assert "a model that can only run on it is not placed here" in result.consequence
    assert result.fix_available
    assert f"build {first.digest[:12]}" in result.remediation

    action = _fix_engine_packages(_facts(packages))

    assert action == (
        f"{NAME}: new launches now use build {first.digest[:12]} "
        f"(version {first.claims.engine_version})"
    )
    assert active_engine_generation(ENGINE, "rocm", root=packages) == first.digest
    # Nothing is deleted: the damaged build is still there for a person to remove.
    assert installed_engine_generations(ENGINE, "rocm", root=packages) == tuple(
        sorted((first.digest, second.digest))
    )
    assert _only(_check(packages)).verdict == "ok"
    # Asked again, there is nothing left to do.
    assert _fix_engine_packages(_facts(packages)) is None


def test_a_damaged_engine_with_no_other_build_is_left_for_a_person(
    tmp_path: Path, packages: Path
) -> None:
    installed = install_test_package(tmp_path, packages)
    _lose_a_file(installed)

    result = _only(_check(packages))

    assert result.verdict == "degraded"
    assert not result.fix_available
    assert f"skulk engines remove {ENGINE} --lane rocm" in result.remediation
    assert _fix_engine_packages(_facts(packages)) is None
    assert active_engine_generation(ENGINE, "rocm", root=packages) == installed.digest
    assert installed.root.is_dir()


def test_a_build_this_skulk_cannot_run_is_switched_away_from(
    tmp_path: Path, packages: Path
) -> None:
    """A Skulk update can leave the active build behind; an older one may still fit."""
    fits = install_test_package(tmp_path, packages, generation=1)
    _ = install_test_package(tmp_path, packages, generation=2, protocol=(9, 9))

    result = _only(_check(packages))

    assert result.detail == (
        f"{NAME}: this Skulk version cannot run the build new launches use"
    )
    assert result.fix_available

    assert _fix_engine_packages(_facts(packages)) is not None
    assert active_engine_generation(ENGINE, "rocm", root=packages) == fits.digest
    assert _only(_check(packages)).verdict == "ok"


def test_a_build_this_skulk_cannot_run_is_never_the_replacement(
    tmp_path: Path, packages: Path
) -> None:
    _ = install_test_package(
        tmp_path, packages, generation=1, protocol=(9, 9), activate=False
    )
    damaged = install_test_package(tmp_path, packages, generation=2)
    _lose_a_file(damaged)

    assert not _only(_check(packages)).fix_available
    assert _fix_engine_packages(_facts(packages)) is None
    assert active_engine_generation(ENGINE, "rocm", root=packages) == damaged.digest


def test_the_earlier_build_is_preferred_over_a_newer_one_never_used(
    tmp_path: Path, packages: Path
) -> None:
    earlier = install_test_package(tmp_path, packages, generation=1)
    _ = install_test_package(tmp_path, packages, generation=3, activate=False)
    damaged = install_test_package(tmp_path, packages, generation=2)
    _lose_a_file(damaged)

    assert f"build {earlier.digest[:12]}" in _only(_check(packages)).remediation
    assert _fix_engine_packages(_facts(packages)) is not None
    assert active_engine_generation(ENGINE, "rocm", root=packages) == earlier.digest


def test_a_candidate_that_fails_its_full_check_is_passed_over(
    tmp_path: Path, packages: Path
) -> None:
    """The quick check cannot see changed bytes; activation hashes and refuses them."""
    earlier = install_test_package(tmp_path, packages, generation=1)
    spare = install_test_package(tmp_path, packages, generation=3, activate=False)
    damaged = install_test_package(tmp_path, packages, generation=2)
    _lose_a_file(damaged)
    _change_a_file(earlier)

    action = _fix_engine_packages(_facts(packages))

    assert action is not None and f"build {spare.digest[:12]}" in action
    assert active_engine_generation(ENGINE, "rocm", root=packages) == spare.digest


def test_when_no_candidate_passes_its_full_check_nothing_changes(
    tmp_path: Path, packages: Path
) -> None:
    earlier = install_test_package(tmp_path, packages, generation=1)
    damaged = install_test_package(tmp_path, packages, generation=2)
    _lose_a_file(damaged)
    _change_a_file(earlier)

    action = _fix_engine_packages(_facts(packages))

    assert action == f"{NAME}: no other build on disk passed a full check"
    assert active_engine_generation(ENGINE, "rocm", root=packages) == damaged.digest
    assert _only(_check(packages)).verdict == "degraded"


def test_one_lanes_problem_does_not_hide_a_usable_engine(
    tmp_path: Path, packages: Path
) -> None:
    good = install_test_package(tmp_path, packages, lane="cuda")
    damaged = install_test_package(tmp_path, packages, lane="rocm")
    _lose_a_file(damaged)

    results = _check(packages)

    assert [result.verdict for result in results] == ["ok", "degraded"]
    assert results[0].detail == (
        f"usable: {ENGINE} (cuda) {good.claims.engine_version}"
    )
    assert results[1].title == f"Add-on engine: {NAME}"


def test_two_engines_claiming_one_name_are_reported_and_not_changed(
    packages: Path,
) -> None:
    ambiguous = NodeEnginePackage(
        engine_id=ENGINE,
        lane="rocm",
        backend_tag="tensorfold-rocm",
        status="unusable",
        installed_builds=("a" * 64, "b" * 64),
        problem="backend_tag_ambiguous",
    )
    facts = make_facts().model_copy(update={"engine_packages": (ambiguous,)})

    result = _only(list(_check_engine_packages(facts)))

    assert result.verdict == "degraded"
    assert "tensorfold-rocm is reserved or claimed by another installed engine" in (
        result.detail
    )
    assert not result.fix_available
    assert "remove one of the engines" in result.remediation
    assert _fix_engine_packages(facts) is None


def test_with_add_on_engines_turned_off_nothing_is_reported_or_changed(
    tmp_path: Path, packages: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ = install_test_package(tmp_path, packages, generation=1)
    second = install_test_package(tmp_path, packages, generation=2)
    _lose_a_file(second)
    stale = _facts(packages)
    monkeypatch.setenv(ENGINE_PACKAGES_DISABLE_ENV, "1")

    result = _only(list(_check_engine_packages(stale)))

    assert result.verdict == "ok"
    assert result.detail.startswith(f"turned off ({ENGINE_PACKAGES_DISABLE_ENV}=1)")
    assert _fix_engine_packages(stale) is None
    assert active_engine_generation(ENGINE, "rocm", root=packages) == second.digest
