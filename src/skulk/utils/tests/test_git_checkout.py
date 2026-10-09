"""Reading a checkout's commit from its files, without running git."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from skulk.utils import git_checkout
from skulk.utils.git_checkout import find_checkout_root, read_checkout_head
from skulk.utils.info_gatherer import info_gatherer

COMMIT = "0123456789abcdef0123456789abcdef01234567"
OTHER = "fedcba9876543210fedcba9876543210fedcba98"


def _clone(root: Path, head: str = "ref: refs/heads/main\n") -> Path:
    git = root / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text(head)
    return git


def test_an_ordinary_clone_reads_its_branch_commit(tmp_path: Path) -> None:
    git = _clone(tmp_path)
    (git / "refs" / "heads" / "main").write_text(COMMIT + "\n")
    assert read_checkout_head(tmp_path) == COMMIT


def test_a_ref_only_in_packed_refs_is_found(tmp_path: Path) -> None:
    git = _clone(tmp_path)
    (git / "packed-refs").write_text(
        "# pack-refs with: peeled fully-peeled sorted\n"
        f"{OTHER} refs/heads/other\n"
        f"{COMMIT} refs/heads/main\n"
        f"^{OTHER}\n"
    )
    assert read_checkout_head(tmp_path) == COMMIT


def test_a_detached_head_is_its_own_commit(tmp_path: Path) -> None:
    _clone(tmp_path, head=COMMIT + "\n")
    assert read_checkout_head(tmp_path) == COMMIT


def test_a_linked_worktree_resolves_refs_through_its_common_dir(tmp_path: Path) -> None:
    main = tmp_path / "main"
    common = _clone(main)
    (common / "refs" / "heads" / "feature").write_text(COMMIT + "\n")
    worktree_git = common / "worktrees" / "feature"
    worktree_git.mkdir(parents=True)
    (worktree_git / "HEAD").write_text("ref: refs/heads/feature\n")
    (worktree_git / "commondir").write_text("../..\n")
    worktree = tmp_path / "feature"
    worktree.mkdir()
    (worktree / ".git").write_text(f"gitdir: {worktree_git}\n")
    assert read_checkout_head(worktree) == COMMIT


def test_a_relative_gitdir_is_resolved_from_the_checkout(tmp_path: Path) -> None:
    real = tmp_path / "store" / "repo.git"
    (real / "refs" / "heads").mkdir(parents=True)
    (real / "HEAD").write_text("ref: refs/heads/main\n")
    (real / "refs" / "heads" / "main").write_text(COMMIT + "\n")
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / ".git").write_text("gitdir: ../store/repo.git\n")
    assert read_checkout_head(checkout) == COMMIT


@pytest.mark.parametrize(
    "head",
    [
        "",
        "not a ref\n",
        "ref: refs/heads/missing\n",
        "ref: refs/../../../etc/passwd\n",
        "ref: HEAD\n",
        "0123456789abcdef\n",
    ],
)
def test_unreadable_heads_yield_none(tmp_path: Path, head: str) -> None:
    _clone(tmp_path, head=head)
    assert read_checkout_head(tmp_path) is None


def test_a_ref_with_an_embedded_nul_yields_none(tmp_path: Path) -> None:
    _clone(tmp_path, head="ref: refs/heads/ma\x00in\n")
    assert read_checkout_head(tmp_path) is None


def test_node_identity_reports_unknown_when_the_checkout_cannot_be_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unreadable(_: Path) -> Path | None:
        raise PermissionError("an ancestor directory is not readable")

    monkeypatch.setattr(info_gatherer, "find_checkout_root", unreadable)
    assert info_gatherer._get_git_commit(tmp_path) == "unknown"  # pyright: ignore[reportPrivateUsage]


def test_a_directory_that_is_not_a_checkout_yields_none(tmp_path: Path) -> None:
    assert read_checkout_head(tmp_path) is None
    (tmp_path / ".git").write_text("not a gitdir pointer\n")
    assert read_checkout_head(tmp_path) is None


def test_the_checkout_root_is_found_from_a_nested_package(tmp_path: Path) -> None:
    _clone(tmp_path)
    package = tmp_path / "src" / "skulk"
    package.mkdir(parents=True)
    assert find_checkout_root(package) == tmp_path


def test_node_identity_reads_the_commit_without_starting_a_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    git = _clone(tmp_path)
    (git / "refs" / "heads" / "main").write_text(COMMIT + "\n")
    package = tmp_path / "src" / "skulk"
    package.mkdir(parents=True)

    def refuse(*_: object, **__: object) -> object:
        raise AssertionError("node identity must not start a process to read its commit")

    monkeypatch.setattr(subprocess, "run", refuse)
    monkeypatch.setattr(subprocess, "Popen", refuse)
    assert info_gatherer._get_git_commit(package) == COMMIT[:8]  # pyright: ignore[reportPrivateUsage]


def test_node_identity_without_a_checkout_reports_unknown(tmp_path: Path) -> None:
    package = tmp_path / "site-packages" / "skulk"
    package.mkdir(parents=True)
    assert info_gatherer._get_git_commit(package) == "unknown"  # pyright: ignore[reportPrivateUsage]


def test_this_repository_matches_git_when_git_is_available() -> None:
    """Cross-check the reader against git itself on the real checkout, when there is one."""
    root = find_checkout_root(Path(git_checkout.__file__).resolve().parent)
    git = shutil.which("git")
    if root is None or git is None:
        pytest.skip("not running from a git checkout with git available")
    expected = subprocess.run(
        [git, "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if expected.returncode != 0:
        pytest.skip("git could not read this checkout")
    assert read_checkout_head(root) == expected.stdout.strip()
