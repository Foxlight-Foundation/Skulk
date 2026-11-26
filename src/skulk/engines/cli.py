"""``skulk engines``: list engine packages and change the ones a node holds.

Every command talks to the Skulk node running on this machine, over its
ordinary API, and so does exactly what the same routes do for any other
caller: ``list`` reads ``GET /v1/engines``; ``activate``, ``rollback`` and
``remove`` call the routes under ``/v1/engines/{engine_id}``. Nothing is read
from or written to this machine's packages folder directly, so a change is
made by the node that holds the package, which may be another machine.

Exit codes: 0 when the command did what was asked, 1 when the node refused
or could not be reached, 2 for a command line that cannot be understood.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from typing import Final, cast

import httpx

DEFAULT_API_PORT: Final = 52415

_READ_TIMEOUT_SECONDS: Final = 15.0
# A change waits for the node to hash a build and answer; the route itself
# gives up after 100 seconds and says so.
_CHANGE_TIMEOUT_SECONDS: Final = 130.0

Json = dict[str, object]
Call = Callable[[str, str, Json | None, float], httpx.Response]
"""``(method, path, body, timeout) -> response``. Injected by tests."""


class EnginesCommandError(Exception):
    """A command that could not do what was asked, with a sentence for a person."""


def _local_call(api_port: int) -> Call:
    """Build the caller that reaches the node on this machine."""
    origin = f"http://127.0.0.1:{api_port}"

    def call(method: str, path: str, body: Json | None, timeout: float) -> httpx.Response:
        return httpx.request(method, f"{origin}{path}", json=body, timeout=timeout)

    return call


def _ask(call: Call, method: str, path: str, body: Json | None, timeout: float) -> object:
    """Call the node and return its JSON answer, or raise with what it said."""
    try:
        response = call(method, path, body, timeout)
    except httpx.HTTPError as error:
        raise EnginesCommandError(
            "The Skulk node on this machine did not answer "
            f"({type(error).__name__}). Is it running?"
        ) from None
    try:
        payload = cast(object, response.json())
    except ValueError:
        payload = None
    if response.status_code == 200:
        return payload
    error = cast(Json, payload).get("error") if isinstance(payload, dict) else None
    message = cast(Json, error).get("message") if isinstance(error, dict) else None
    raise EnginesCommandError(
        message
        if isinstance(message, str) and message
        else f"The node answered {response.status_code}."
    )


def _engines(call: Call) -> list[Json]:
    """Read the cluster's engine packages report."""
    report = _ask(call, "GET", "/v1/engines", None, _READ_TIMEOUT_SECONDS)
    engines = cast(Json, report).get("engines") if isinstance(report, dict) else None
    if not isinstance(engines, list):
        raise EnginesCommandError("The node's engines report could not be read.")
    return [
        cast(Json, engine)
        for engine in cast(list[object], engines)
        if isinstance(engine, dict)
    ]


def _lanes(engines: list[Json], engine_id: str, node_id: str) -> list[Json]:
    """The package lanes one node holds for one engine."""
    held: list[Json] = []
    for engine in engines:
        if engine.get("engineId") != engine_id:
            continue
        for entry in cast(list[Json], engine.get("nodes") or []):
            package = cast(Json, entry.get("package") or {})
            if entry.get("nodeId") == node_id and package.get("status") != "offered":
                held.append(package)
    return held


def _short(build: object) -> str:
    return f"{build[:12]}" if isinstance(build, str) and build else "unknown"


def _describe(package: Json) -> str:
    """One line for one package lane."""
    parts = [str(package.get("lane")), str(package.get("status"))]
    if package.get("build"):
        parts.append(f"build {_short(package.get('build'))}")
    if package.get("engineVersion"):
        parts.append(f"version {package.get('engineVersion')}")
    if package.get("problem"):
        parts.append(f"problem {package.get('problem')}")
    if package.get("rollbackBuild"):
        parts.append(f"earlier build {_short(package.get('rollbackBuild'))}")
    on_disk = cast(list[object], package.get("installedBuilds") or [])
    if len(on_disk) > 1:
        parts.append(f"{len(on_disk)} builds on disk")
    for result in cast(list[Json], package.get("loadResults") or [])[:1]:
        outcome = str(result.get("outcome"))
        if result.get("code"):
            outcome = f"{outcome} ({result.get('code')})"
        parts.append(f"last load {outcome}: {result.get('modelId')}")
    return ", ".join(parts)


def _render(engines: list[Json]) -> str:
    """The engines report as text."""
    if not engines:
        return "No node holds or is offered an engine package."
    lines: list[str] = []
    for engine in engines:
        lines.append(str(engine.get("engineId")))
        for entry in cast(list[Json], engine.get("nodes") or []):
            package = cast(Json, entry.get("package") or {})
            lines.append(f"  {entry.get('nodeId')}: {_describe(package)}")
            for build in cast(list[object], package.get("installedBuilds") or []):
                lines.append(f"      on disk: {build}")
    return "\n".join(lines)


def _target(
    call: Call, engine_id: str, node: str | None, lane: str | None
) -> tuple[str, str]:
    """Settle which node and accelerator class a change is for.

    Without ``--node`` the node on this machine is meant. Without ``--lane``
    the node's only package for the engine is meant; a node holding the
    engine for more than one accelerator class must be told which.
    """
    if node is None:
        answer = _ask(call, "GET", "/node_id", None, _READ_TIMEOUT_SECONDS)
        if not isinstance(answer, str) or not answer:
            raise EnginesCommandError("This machine's node did not say which node it is.")
        node = answer
    if lane is not None:
        return node, lane
    held = _lanes(_engines(call), engine_id, node)
    if len(held) == 1:
        return node, str(held[0].get("lane"))
    if not held:
        raise EnginesCommandError(
            f"Node {node} holds no {engine_id} package. Run `skulk engines list` "
            "to see what each node holds."
        )
    choices = ", ".join(sorted(str(package.get("lane")) for package in held))
    raise EnginesCommandError(
        f"Node {node} holds {engine_id} for {choices}. Say which with --lane."
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="skulk engines",
        description="List engine packages and change the ones a node holds.",
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=DEFAULT_API_PORT,
        help="Port of the Skulk node on this machine (default: 52415).",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="Show every node's engine packages.")
    listing.add_argument("--json", action="store_true", help="Print the report as JSON.")

    def change(name: str, summary: str) -> argparse.ArgumentParser:
        command = commands.add_parser(name, help=summary, description=summary)
        command.add_argument("engine_id", help="The engine, such as foxlight.tensorfold.")
        command.add_argument(
            "--node", help="The node to change (default: the node on this machine)."
        )
        command.add_argument(
            "--lane",
            choices=("metal", "cuda", "rocm", "vulkan", "cpu"),
            help="The accelerator class (default: the node's only one for the engine).",
        )
        return command

    change(
        "activate", "Make a build on a node's disk the one new launches use."
    ).add_argument("build", help="Manifest digest of the build, as `list` shows it.")
    change("rollback", "Return to the build that was active before.")
    change(
        "remove", "Delete one build, or the whole package when no build is named."
    ).add_argument("build", nargs="?", help="Manifest digest of the build to delete.")
    return parser


def main(argv: list[str] | None = None, *, call: Call | None = None) -> int:
    """Entry point for ``skulk engines``.

    Args:
        argv: Arguments after the ``engines`` subcommand, or ``None`` for
            ``sys.argv[1:]``.
        call: How to reach the node. Tests inject one; otherwise the node on
            this machine is called over HTTP.

    Returns:
        The process exit code.

    Side effects:
        ``activate``, ``rollback`` and ``remove`` change the packages a node
        holds, through that node's own API routes.
    """
    arguments = _parser().parse_args(argv)
    reach = call if call is not None else _local_call(cast(int, arguments.api_port))
    command = cast(str, arguments.command)
    try:
        if command == "list":
            engines = _engines(reach)
            print(
                json.dumps(engines, indent=2)
                if cast(bool, arguments.json)
                else _render(engines)
            )
            return 0
        engine_id = cast(str, arguments.engine_id)
        node, lane = _target(
            reach,
            engine_id,
            cast("str | None", arguments.node),
            cast("str | None", arguments.lane),
        )
        body: Json = {"nodeId": node, "lane": lane}
        build = cast("str | None", getattr(arguments, "build", None))
        if build is not None:
            body["build"] = build
        answer = _ask(
            reach,
            "POST",
            f"/v1/engines/{engine_id}/{command}",
            body,
            _CHANGE_TIMEOUT_SECONDS,
        )
    except EnginesCommandError as error:
        print(str(error), file=sys.stderr)
        return 1
    package = cast(Json, answer).get("package") if isinstance(answer, dict) else None
    if isinstance(package, dict):
        print(f"{engine_id} on {node}: {_describe(cast(Json, package))}")
    else:
        print(f"{engine_id} on {node}: the node no longer holds this package.")
    return 0
