"""What a node reports about the engine packages it holds."""

from pathlib import Path

import pytest

from skulk.engines.inventory import (
    installed_engine_for_backend,
    read_engine_package_inventory,
)
from skulk.engines.tests.package_fixtures import install_test_package
from skulk.shared.backends import ENGINE_PACKAGES_DISABLE_ENV


def test_a_node_without_packages_reports_nothing(tmp_path: Path) -> None:
    """No packages root, or an empty one, is an empty inventory."""
    assert read_engine_package_inventory(root=tmp_path / "absent").active == {}
    (tmp_path / "packages").mkdir()
    inventory = read_engine_package_inventory(root=tmp_path / "packages")
    assert inventory.active == {} and inventory.unusable == ()


def test_only_an_active_generation_is_offered(tmp_path: Path) -> None:
    """An installed generation is not offered until it is activated."""
    packages = tmp_path / "packages"
    waiting = install_test_package(tmp_path, packages, activate=False)
    inventory = read_engine_package_inventory(root=packages)
    assert inventory.active == {} and inventory.unusable == ()

    active = install_test_package(tmp_path, packages)
    assert active.digest == waiting.digest
    inventory = read_engine_package_inventory(root=packages)
    assert inventory.builds == {"tensorfold-rocm": active.digest}
    assert inventory.unusable == ()


def test_lanes_of_one_engine_are_offered_separately(tmp_path: Path) -> None:
    """Each lane has its own active generation and its own backend tag."""
    packages = tmp_path / "packages"
    rocm = install_test_package(tmp_path, packages, lane="rocm")
    cuda = install_test_package(tmp_path, packages, lane="cuda")
    assert read_engine_package_inventory(root=packages).builds == {
        "tensorfold-rocm": rocm.digest,
        "tensorfold-cuda": cuda.digest,
    }


def test_a_damaged_active_generation_is_reported_not_offered(tmp_path: Path) -> None:
    """A lost file withdraws the backend and says why."""
    packages = tmp_path / "packages"
    installed = install_test_package(tmp_path, packages)
    installed.adapter.unlink()

    inventory = read_engine_package_inventory(root=packages)

    assert inventory.active == {}
    assert [
        (entry.engine_id, entry.lane, entry.reason) for entry in inventory.unusable
    ] == [("foxlight.tensorfold", "rocm", "installation_damaged")]


def test_a_tag_claimed_by_two_engines_is_offered_by_neither(tmp_path: Path) -> None:
    """Two publishers' engines with one name cannot both own a backend tag."""
    packages = tmp_path / "packages"
    install_test_package(tmp_path, packages, engine_id="foxlight.tensorfold")
    install_test_package(tmp_path, packages, engine_id="another.tensorfold")

    inventory = read_engine_package_inventory(root=packages)

    assert inventory.active == {}
    assert sorted((entry.engine_id, entry.reason) for entry in inventory.unusable) == [
        ("another.tensorfold", "backend_tag_ambiguous"),
        ("foxlight.tensorfold", "backend_tag_ambiguous"),
    ]


def test_a_package_cannot_take_a_built_in_engines_tag(tmp_path: Path) -> None:
    """A package named after a built-in engine never replaces that engine."""
    packages = tmp_path / "packages"
    install_test_package(
        tmp_path, packages, engine_id="foxlight.llama_server", lane="cuda"
    )

    inventory = read_engine_package_inventory(root=packages)

    assert inventory.active == {}
    assert [entry.reason for entry in inventory.unusable] == ["backend_tag_ambiguous"]


def test_directories_that_are_not_packages_are_ignored(tmp_path: Path) -> None:
    """Stray directories under the packages root are neither offered nor reported."""
    packages = tmp_path / "packages"
    installed = install_test_package(tmp_path, packages)
    (packages / "Not A Package").mkdir()
    (packages / "foxlight.tensorfold" / "quantum").mkdir()

    inventory = read_engine_package_inventory(root=packages)

    assert inventory.builds == {"tensorfold-rocm": installed.digest}
    assert inventory.unusable == ()


def test_a_stamped_placement_keeps_its_own_generation(tmp_path: Path) -> None:
    """Activating a newer generation does not move a placement stamped with an older one."""
    packages = tmp_path / "packages"
    first = install_test_package(tmp_path, packages, generation=1)
    second = install_test_package(tmp_path, packages, generation=2)
    assert first.digest != second.digest
    tag = "tensorfold-rocm"

    stamped = installed_engine_for_backend(tag, first.digest, root=packages)
    unstamped = installed_engine_for_backend(tag, None, root=packages)

    assert stamped is not None and stamped.digest == first.digest
    assert unstamped is not None and unstamped.digest == second.digest


def test_an_unknown_stamp_or_tag_finds_nothing(tmp_path: Path) -> None:
    """A removed generation, a malformed stamp and an unheld tag all return nothing."""
    packages = tmp_path / "packages"
    install_test_package(tmp_path, packages)
    tag = "tensorfold-rocm"

    assert installed_engine_for_backend(tag, "f" * 64, root=packages) is None
    assert installed_engine_for_backend(tag, "../../outside", root=packages) is None
    assert installed_engine_for_backend("tensorfold-cuda", None, root=packages) is None


def test_packages_turned_off_report_nothing_whatever_is_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A good package and a damaged one both vanish; clearing the switch brings them back."""
    packages = tmp_path / "packages"
    good = install_test_package(tmp_path, packages)
    damaged = install_test_package(tmp_path, packages, lane="cuda")
    damaged.adapter.unlink()
    tag = "tensorfold-rocm"
    before = read_engine_package_inventory(root=packages)
    assert before.builds == {tag: good.digest} and len(before.unusable) == 1

    monkeypatch.setenv(ENGINE_PACKAGES_DISABLE_ENV, "1")
    empty = read_engine_package_inventory(root=tmp_path / "absent")
    turned_off = read_engine_package_inventory(root=packages)

    # Exactly what a node with no packages reports.
    assert turned_off == empty
    assert turned_off.active == {} and turned_off.unusable == ()
    assert installed_engine_for_backend(tag, None, root=packages) is None
    assert installed_engine_for_backend(tag, good.digest, root=packages) is None

    monkeypatch.delenv(ENGINE_PACKAGES_DISABLE_ENV)
    assert read_engine_package_inventory(root=packages) == before


@pytest.mark.parametrize("value", ["", "0", "true", "yes", "on", "2"])
def test_only_the_value_one_turns_packages_off(
    value: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packages = tmp_path / "packages"
    good = install_test_package(tmp_path, packages)

    monkeypatch.setenv(ENGINE_PACKAGES_DISABLE_ENV, value)

    assert read_engine_package_inventory(root=packages).builds == {
        "tensorfold-rocm": good.digest
    }
