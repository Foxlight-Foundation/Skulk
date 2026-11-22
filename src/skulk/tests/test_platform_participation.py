"""Windows starts as a management node; other platforms are left alone."""

import ast
from pathlib import Path

import pytest

import skulk
from skulk.main import (
    NODE_PARTICIPATION_ENV_VAR,
    Args,
    UnsupportedParticipationError,
    apply_platform_participation,
    is_master_candidate,
)

_POSIX_ONLY_MODULES = frozenset({"fcntl", "grp", "pwd", "resource", "termios"})


def _args(*, no_worker: bool = False) -> Args:
    return Args(libp2p_port=0, no_worker=no_worker)


@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_other_platforms_keep_their_arguments_and_environment(platform: str) -> None:
    environ: dict[str, str] = {}
    args = _args()

    assert apply_platform_participation(args, platform, environ) is args
    assert environ == {}


@pytest.mark.parametrize("declared", [None, "", "management", " Management "])
def test_windows_runs_without_a_worker_as_a_management_node(
    declared: str | None,
) -> None:
    environ: dict[str, str] = {}
    if declared is not None:
        environ[NODE_PARTICIPATION_ENV_VAR] = declared

    args = apply_platform_participation(_args(), "win32", environ)

    assert args.no_worker is True
    assert environ[NODE_PARTICIPATION_ENV_VAR] == "management"


@pytest.mark.parametrize("declared", ["full", "ffn_only", "FULL"])
def test_windows_refuses_a_participation_that_needs_an_engine(declared: str) -> None:
    environ = {NODE_PARTICIPATION_ENV_VAR: declared}

    with pytest.raises(UnsupportedParticipationError, match="not available on Windows"):
        apply_platform_participation(_args(), "win32", environ)

    assert environ == {NODE_PARTICIPATION_ENV_VAR: declared}


@pytest.mark.parametrize("declared", [None, "", "full", "ffn_only", "unknown"])
def test_an_inference_node_is_an_ordinary_master_candidate(
    declared: str | None,
) -> None:
    environ = {} if declared is None else {NODE_PARTICIPATION_ENV_VAR: declared}

    assert is_master_candidate(_args(), environ) is True


@pytest.mark.parametrize("declared", ["management", " Management "])
def test_a_declared_management_node_is_a_last_resort_candidate(declared: str) -> None:
    environ = {NODE_PARTICIPATION_ENV_VAR: declared}

    assert is_master_candidate(_args(), environ) is False


def test_a_node_without_a_worker_is_a_last_resort_candidate() -> None:
    assert is_master_candidate(_args(no_worker=True), {}) is False


def test_a_windows_node_is_a_last_resort_candidate() -> None:
    environ: dict[str, str] = {}
    args = apply_platform_participation(_args(), "win32", environ)

    assert is_master_candidate(args, environ) is False


def test_force_master_keeps_a_management_node_an_ordinary_candidate() -> None:
    args = Args(libp2p_port=0, no_worker=True, force_master=True)
    environ = {NODE_PARTICIPATION_ENV_VAR: "management"}

    assert is_master_candidate(args, environ) is True


def test_no_module_imports_a_posix_only_module_unconditionally() -> None:
    """A module-level POSIX-only import stops every Windows node at startup.

    The node entry point imports most of the package, so one unconditional
    ``import fcntl`` anywhere in that graph is fatal on Windows. Such imports
    belong inside the function that needs them, or under a platform check.
    """
    package_root = Path(skulk.__file__).parent
    offenders: list[str] = []
    for source in sorted(package_root.rglob("*.py")):
        if "tests" in source.parts:
            continue
        for statement in ast.parse(source.read_text(encoding="utf-8")).body:
            if isinstance(statement, ast.Import):
                names = {alias.name.split(".")[0] for alias in statement.names}
            elif isinstance(statement, ast.ImportFrom) and statement.level == 0:
                names = {(statement.module or "").split(".")[0]}
            else:
                continue
            for name in sorted(names & _POSIX_ONLY_MODULES):
                offenders.append(f"{source.relative_to(package_root)}: {name}")

    assert offenders == []


def test_no_module_uses_the_tuf_client_that_needs_a_symbolic_link() -> None:
    """python-tuf's own client cannot keep its trusted root on Windows.

    It points ``root.json`` at the current root with a symbolic link, which an
    ordinary Windows account may not create, so the signed model catalog and
    the capability store would both be unreadable there. The model-preservation
    package's ``TufUpdater`` writes a copy on Windows and is the same client
    everywhere else; that is the one to import.
    """
    package_root = Path(skulk.__file__).parent
    offenders: list[str] = []
    for source in sorted(package_root.rglob("*.py")):
        if "tests" in source.parts:
            continue
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.ImportFrom)
                and (node.module or "").startswith("tuf.ngclient")
                and any(alias.name == "Updater" for alias in node.names)
            ):
                offenders.append(str(source.relative_to(package_root)))

    assert offenders == []
