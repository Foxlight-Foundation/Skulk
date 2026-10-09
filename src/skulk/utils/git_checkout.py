"""Read a git checkout's current commit from its files, without running git.

On a Mac without the Xcode command line tools, ``/usr/bin/git`` is only a shim:
running it opens a dialog offering to install the tools. A node that ran git
just to report which build it is would show that dialog on every fresh Mac,
and a packaged install has no checkout to report anyway. Reading the commit
from the checkout's own files answers the same question with no process.
"""

from __future__ import annotations

import re
from pathlib import Path

_SHA = re.compile(r"[0-9a-f]{40}")
_SYMBOLIC = "ref: "
_MAX_FILE_BYTES = 1 << 20


def find_checkout_root(start: Path) -> Path | None:
    """Return the nearest directory at or above ``start`` that holds a ``.git`` entry.

    Args:
        start: A path inside the checkout, such as an installed package's
            directory.

    Returns:
        The checkout's root, or None when no ancestor is a git checkout.
    """
    for directory in (start, *start.parents):
        if (directory / ".git").exists():
            return directory
    return None


def read_checkout_head(checkout: Path) -> str | None:
    """Return the full commit the checkout's HEAD points at, or None.

    Handles an ordinary clone (``.git`` directory), a linked worktree or
    submodule (``.git`` file naming its git directory, with refs shared
    through ``commondir``), a detached HEAD, and refs that live only in
    ``packed-refs``. Anything unreadable or malformed yields None rather than
    a guess.

    Args:
        checkout: The checkout's root directory.

    Returns:
        The 40-character lowercase commit, or None.
    """
    git_dir = _git_dir(checkout)
    if git_dir is None:
        return None
    head = _read_text(git_dir / "HEAD")
    if head is None:
        return None
    head = head.strip()
    if _SHA.fullmatch(head):
        return head
    if not head.startswith(_SYMBOLIC):
        return None
    return _resolve_ref(git_dir, head[len(_SYMBOLIC) :].strip())


def _git_dir(checkout: Path) -> Path | None:
    entry = checkout / ".git"
    if entry.is_dir():
        return entry
    if not entry.is_file():
        return None
    text = _read_text(entry)
    if text is None or not text.startswith("gitdir:"):
        return None
    target = Path(text[len("gitdir:") :].strip())
    if not target.is_absolute():
        target = checkout / target
    return target if target.is_dir() else None


def _common_dir(git_dir: Path) -> Path:
    """Where a worktree's shared refs live; an ordinary git directory is its own."""
    text = _read_text(git_dir / "commondir")
    if text is None or not text.strip():
        return git_dir
    common = Path(text.strip())
    if not common.is_absolute():
        common = git_dir / common
    return common if common.is_dir() else git_dir


def _resolve_ref(git_dir: Path, ref: str) -> str | None:
    if not ref.startswith("refs/") or ".." in ref.split("/"):
        return None
    common = _common_dir(git_dir)
    # A worktree's own refs (such as its HEAD's per-worktree refs) shadow the
    # shared ones, so its git directory is read first.
    for directory in dict.fromkeys((git_dir, common)):
        loose = _read_text(directory / ref)
        if loose is not None and _SHA.fullmatch(loose.strip()):
            return loose.strip()
    packed = _read_text(common / "packed-refs")
    if packed is None:
        return None
    for line in packed.splitlines():
        if not line or line.startswith(("#", "^")):
            continue
        sha, _, name = line.partition(" ")
        if name.strip() == ref and _SHA.fullmatch(sha):
            return sha
    return None


def _read_text(path: Path) -> str | None:
    try:
        with path.open("rb") as handle:
            data = handle.read(_MAX_FILE_BYTES + 1)
    except OSError:
        return None
    if len(data) > _MAX_FILE_BYTES:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None
