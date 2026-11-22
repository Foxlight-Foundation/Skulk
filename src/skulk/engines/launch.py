"""Start an installed engine package's adapter: the launch contract.

The contract is the same for every package, so core holds no engine name:

* A Python adapter is run by this node's interpreter in isolated mode with
  bytecode writing off; a native adapter is run directly.
* The environment is built here from nothing. It carries the package's own
  locations and identity, a private home, cache and temporary directory under
  a Skulk-owned state directory, and a minimal ``PATH``. No library search
  path and no host variable is inherited. An adapter that needs more for its
  engine sets it itself, for the engine process only.
* The working directory is the state directory. The generation directory is
  never written.
* The adapter's standard error goes to a bounded private log in the state
  directory.

The protocol versions offered are the ones this build speaks that the
package's manifest also declares. The interpreter range is checked again at
every launch, because Skulk's interpreter can change under an installed
package.
"""

import contextlib
import os
import platform
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final, final

from skulk.engines.adapter import START_DEADLINE_SECONDS, AdapterSession
from skulk.engines.protocol import PROTOCOL_VERSIONS_SUPPORTED
from skulk.provisioning.engine_install import InstalledEngine
from skulk.provisioning.engine_packages import (
    EnginePackageRefusedError,
    adapter_python_supported,
)
from skulk.shared.constants import SKULK_ENGINES_DIR

ENGINE_STATE_DIR: Final = SKULK_ENGINES_DIR / "package-state"
"""Root of what running packages may write: ``<manifest digest>/``."""

ADAPTER_PATH: Final = "/usr/bin:/bin"
"""The whole ``PATH`` an adapter starts with."""

_LOG_NAME: Final = "adapter.log"
_LOG_LIMIT_BYTES: Final = 8 * 1024 * 1024


@final
@dataclass(frozen=True)
class AdapterLaunch:
    """Everything needed to start one adapter process.

    Attributes:
        command: The complete argument vector.
        environment: The complete environment; nothing else is passed on.
        working_directory: Where the adapter runs, inside the state directory.
        log_path: The private file that receives the adapter's standard error.
        versions: Control-protocol versions to offer in ``hello``.
        build_digest: The installed generation's manifest digest.
    """

    command: tuple[str, ...]
    environment: dict[str, str]
    working_directory: Path
    log_path: Path
    versions: tuple[int, ...]
    build_digest: str


def engine_state_directory(digest: str, *, root: Path = ENGINE_STATE_DIR) -> Path:
    """The Skulk-owned directory one package build may write to."""
    return root / digest


def _refuse_python(message: str) -> EnginePackageRefusedError:
    return EnginePackageRefusedError("adapter_python_unsupported", message)


def plan_adapter_launch(
    installed: InstalledEngine,
    *,
    state_directory: Path,
    interpreter: Path | None = None,
    python_version: str | None = None,
) -> AdapterLaunch:
    """Build the command and environment for an installed generation's adapter.

    Creates the private state directories. Starts nothing.

    Args:
        installed: A generation that passed ``verify_installed_engine``.
        state_directory: The Skulk-owned directory this build may write to.
        interpreter: The interpreter for a Python adapter. Defaults to the
            one running Skulk.
        python_version: That interpreter's version. Defaults to the running
            one.

    Raises:
        EnginePackageRefusedError: ``protocol_unsupported`` when this build
            speaks no control-protocol version the manifest declares, and
            ``adapter_python_unsupported`` when the interpreter is outside
            the adapter's range or this is a packaged Skulk whose own
            executable cannot run a script.
    """
    claims = installed.claims
    versions = tuple(
        version
        for version in PROTOCOL_VERSIONS_SUPPORTED
        if claims.control_protocol.minimum <= version <= claims.control_protocol.maximum
    )
    if not versions:
        raise EnginePackageRefusedError(
            "protocol_unsupported",
            f"the adapter speaks control protocol {claims.control_protocol.minimum} "
            f"to {claims.control_protocol.maximum}; this Skulk speaks "
            f"{PROTOCOL_VERSIONS_SUPPORTED[0]} to {PROTOCOL_VERSIONS_SUPPORTED[-1]}",
        )
    if claims.adapter.kind == "python":
        if interpreter is None:
            if getattr(sys, "frozen", False):
                # A packaged Skulk's executable is the application launcher,
                # not an interpreter that can run an adapter file.
                raise _refuse_python(
                    "this packaged Skulk has no interpreter for a Python adapter"
                )
            interpreter = Path(sys.executable)
        running = python_version or platform.python_version()
        if not adapter_python_supported(claims, running):
            raise _refuse_python(
                f"the adapter needs Python {claims.adapter.python_requires}; "
                f"this node runs {running}"
            )
        command = (str(interpreter), "-I", "-B", str(installed.adapter))
    else:
        command = (str(installed.adapter),)

    home = state_directory / "home"
    cache = state_directory / "cache"
    temporary = state_directory / "tmp"
    for directory in (state_directory, home, cache, temporary):
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    environment = {
        "PATH": ADAPTER_PATH,
        "HOME": str(home),
        "XDG_CACHE_HOME": str(cache),
        "TMPDIR": str(temporary),
        "LC_ALL": "C",
        "SKULK_ENGINE_ID": claims.engine_id,
        "SKULK_ENGINE_LANE": claims.lane,
        "SKULK_ENGINE_BUILD_DIGEST": installed.digest,
        "SKULK_ENGINE_GENERATION": str(installed.root),
        "SKULK_ENGINE_EXECUTABLE": str(installed.executable),
        "SKULK_ENGINE_STATE": str(state_directory),
    }
    if installed.requirements:
        first = next(iter(installed.requirements.values()))
        environment["SKULK_ENGINE_REQUIREMENTS"] = str(first.parent)
    return AdapterLaunch(
        command=command,
        environment=environment,
        working_directory=state_directory,
        log_path=state_directory / _LOG_NAME,
        versions=versions,
        build_digest=installed.digest,
    )


def _open_log(path: Path) -> int:
    """Open the adapter log for appending, starting over once it is large."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW
    with contextlib.suppress(OSError):
        if path.stat().st_size > _LOG_LIMIT_BYTES:
            flags |= os.O_TRUNC
    return os.open(path, flags, 0o600)


@contextlib.contextmanager
def adapter_session(
    launch: AdapterLaunch, *, start_deadline_seconds: float = START_DEADLINE_SECONDS
) -> Iterator[AdapterSession]:
    """Start an adapter, yield its negotiated session, and always stop it.

    Leaving the block terminates the adapter's whole process group, whether
    the block ended normally or not.

    Raises:
        AdapterFailedError: When the adapter cannot be started or does not
            complete the version handshake.
    """
    with os.fdopen(_open_log(launch.log_path), "ab") as log:
        session = AdapterSession(
            command=launch.command,
            environment=launch.environment,
            working_directory=launch.working_directory,
            stderr=log,
            build_digest=launch.build_digest,
        )
        try:
            session.start(
                versions=launch.versions, deadline_seconds=start_deadline_seconds
            )
            yield session
        finally:
            session.close()
