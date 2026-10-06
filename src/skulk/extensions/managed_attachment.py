"""One process-lifetime local bridge to an independently supervised manager."""

import asyncio
import os
import re
import time
from pathlib import Path
from typing import final

import psutil
from pydantic import JsonValue, TypeAdapter

from skulk.connectivity.tailscale import query_tailscale_status
from skulk.extensions.runtime_artifacts import measure_host
from skulk.extensions.runtime_attachment import AttachmentRequest, ServeAddress
from skulk.extensions.runtime_files import RuntimeLock, read_private
from skulk.extensions.runtime_manager import (
    MANAGER_BUILD_DIFFERS,
    ManagerBuildMismatchError,
    manager_request,
)

SERVE_ADDRESS_REFRESH_SECONDS = 60.0
"""How often the bridge looks at Tailscale for the node's serve address."""

_SERVE_ADDRESS: TypeAdapter[str] = TypeAdapter(ServeAddress)


@final
class ManagedAttachment:
    """Share a local attachment fence across one API's installed plugin adapters."""

    def __init__(self, root: Path, profile_id: str) -> None:
        """Bind locally provisioned service storage and its stable profile identity."""
        if not root.is_absolute():
            raise ValueError("manager state root must be absolute")
        self.root = root
        self.profile_id = profile_id
        self.lock: RuntimeLock | None = None
        self.guard = asyncio.Lock()
        self.transport_node_id: str | None = None
        self.build: str | None = None
        self.observed = 0.0
        self.users = 0
        # The node's Tailscale address, kept once seen for this API lifetime.
        self.serve_host: str | None = None
        self.serve_checked: float | None = None
        self.serve_task: asyncio.Task[None] | None = None

    def retain(self, transport_node_id: str) -> None:
        """Retain the bridge for one adapter in the same live API process."""
        if (
            self.transport_node_id is not None
            and self.transport_node_id != transport_node_id
        ):
            raise ValueError("bridge already belongs to another Skulk lifetime")
        self.transport_node_id = transport_node_id
        self.users += 1

    async def ensure(self) -> None:
        """Refresh local attachment and compatibility; refuse competing API lifetimes."""
        async with self.guard:
            if self.users == 0 or self.transport_node_id is None:
                raise ValueError("bridge is not active")
            if self.lock is None:
                self.lock = RuntimeLock(self.root, "attachment.lock")
            self._observe_serve_host()
            if time.monotonic() - self.observed < 1:
                return
            if self.build is None:
                host = await asyncio.to_thread(measure_host)
                self.build = host.skulk_build_sha256
            request = AttachmentRequest(
                profile_id=self.profile_id,
                transport_node_id=self.transport_node_id,
                skulk_build_sha256=self.build,
                serve_host=self.serve_host,
            )
            result = await manager_request(self.root, request)
            if result.get("error") == MANAGER_BUILD_DIFFERS:
                raise ManagerBuildMismatchError(
                    str(result.get("manager")), str(result.get("live"))
                )
            if "error" in result:
                # A manager from before the named refusal answers generically;
                # its selected generation says which build it runs.
                # A manager the keep-alive started before a selection moved
                # runs the previous generation while the pointer names the
                # new one; the process it runs from tells the two apart.
                running = running_manager_build(self.root)
                if running is not None and running != self.build:
                    raise ManagerBuildMismatchError(running, self.build)
                selected = selected_manager_build(self.root)
                if selected is not None and selected != self.build:
                    raise ManagerBuildMismatchError(selected, self.build)
            if result != {
                "result": {
                    "transport_node_id": self.transport_node_id,
                    "skulk_build_sha256": self.build,
                }
            }:
                raise ValueError("local attachment unavailable or incompatible")
            self.observed = time.monotonic()

    def _observe_serve_host(self) -> None:
        """Look at Tailscale in the background, at most once a minute.

        The query is a subprocess that can take seconds, so an attachment
        never waits on it. Until the first answer, attachments carry no
        address, which the manager reads as "keep the one the installations
        already have".
        """
        if self.serve_task is not None and not self.serve_task.done():
            return
        now = time.monotonic()
        if (
            self.serve_checked is not None
            and now - self.serve_checked < SERVE_ADDRESS_REFRESH_SECONDS
        ):
            return
        self.serve_checked = now
        self.serve_task = asyncio.create_task(self._refresh_serve_host())

    async def _refresh_serve_host(self) -> None:
        """Record the node's Tailscale address when Tailscale reports a new one."""
        try:
            status = await query_tailscale_status()
            if not status.running or status.self_ip is None:
                # An address stays once seen. A slow or failed query must not
                # restart every owner, and a node that left its tailnet only
                # leaves its children a bind they fall back from.
                return
            address = _SERVE_ADDRESS.validate_python(status.self_ip)
        except (AttributeError, TypeError, ValueError):
            # Output this build cannot read is no observation at all.
            return
        if address != self.serve_host:
            self.serve_host = address
            # Attach again at the next refresh, which restarts the owners
            # once so every child is told the new address.
            self.observed = 0.0

    async def release(self) -> None:
        """Release the API fence after its last adapter stops; leave cleanup running."""
        async with self.guard:
            self.users -= 1
            if self.users == 0 and self.lock is not None:
                self.lock.close()
                self.lock = None
                self.observed = 0.0
                if self.serve_task is not None:
                    self.serve_task.cancel()
                    self.serve_task = None
                self.serve_checked = None


def manager_processes(root: Path) -> list[tuple[int, list[str]]]:
    """Manager processes serving ``root`` with their command lines; they run as this user."""
    found: list[tuple[int, list[str]]] = []
    arguments = TypeAdapter(list[str])
    for process in psutil.process_iter():
        try:
            cmdline = arguments.validate_python(process.cmdline())
        except (psutil.Error, ValueError):
            continue
        if (
            "skulk.extensions.runtime_manager" in cmdline
            and "serve" in cmdline
            and str(root) in cmdline
        ):
            found.append((int(process.pid), cmdline))
    return found


def manager_pids(root: Path) -> list[int]:
    """Processes serving the manager at ``root``; they run as this user."""
    return [pid for pid, _ in manager_processes(root)]


def runs_generation(cmdline: list[str], root: Path, generation: str) -> bool:
    """Whether a manager command line runs from ``generation``'s runtime under ``root``.

    The OS service starts the manager through the interpreter of the selected
    generation, so the interpreter path names the generation it runs.
    """
    prefix = str(root / "core-runtimes" / generation) + os.sep
    return bool(cmdline) and cmdline[0].startswith(prefix)


def running_generations(root: Path) -> set[str]:
    """Generations any manager serving ``root`` runs from.

    The OS service starts the manager through a generation's interpreter, so
    the interpreter path of every running manager names the generation it
    still needs, including a manager that has not yet restarted onto a newer
    selection.
    """
    prefix = str(root / "core-runtimes") + os.sep
    found: set[str] = set()
    for _, cmdline in manager_processes(root):
        if not cmdline or not cmdline[0].startswith(prefix):
            continue
        generation = cmdline[0][len(prefix) :].split(os.sep, 1)[0]
        if re.fullmatch(r"[a-f0-9]{32}", generation):
            found.add(generation)
    return found


def stale_manager_pids(root: Path, generation: str) -> list[int]:
    """Managers at ``root`` not running from ``generation``.

    A keep-alive restart that began before the selection runs the previous
    generation; it must be stopped again so the next start reads the pointer.
    """
    return [
        pid
        for pid, cmdline in manager_processes(root)
        if not runs_generation(cmdline, root, generation)
    ]


def running_manager_build(root: Path) -> str | None:
    """The Skulk build of the generation a running manager was started from.

    The OS service starts the manager through the selected generation's
    interpreter; a manager started before the selection moved still runs the
    previous generation, and its interpreter path names it.
    """
    generations = root / "core-runtimes"
    prefix = str(generations) + os.sep
    for _, cmdline in manager_processes(root):
        if not cmdline or not cmdline[0].startswith(prefix):
            continue
        generation = cmdline[0][len(prefix) :].split(os.sep, 1)[0]
        if not re.fullmatch(r"[a-f0-9]{32}", generation):
            continue
        try:
            staged = TypeAdapter(dict[str, JsonValue]).validate_json(
                read_private(generations / generation / "staged.json", 131072)
            )
        except (OSError, ValueError):
            continue
        build = staged.get("skulk_build_sha256")
        return build if isinstance(build, str) else None
    return None


def selected_manager_build(root: Path) -> str | None:
    """The Skulk build the manager's selected generation was staged from."""
    document = TypeAdapter(dict[str, JsonValue])
    try:
        pointer = document.validate_json(read_private(root / "core-runtime.json", 4096))
        generation = pointer.get("generation")
        if not isinstance(generation, str) or not re.fullmatch(
            r"[a-f0-9]{32}", generation
        ):
            return None
        staged = document.validate_json(
            read_private(root / "core-runtimes" / generation / "staged.json", 131072)
        )
    except (OSError, ValueError):
        return None
    build = staged.get("skulk_build_sha256")
    return build if isinstance(build, str) else None
