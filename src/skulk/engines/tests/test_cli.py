"""``skulk engines``: the terminal form of the engines report and its three changes."""

from __future__ import annotations

import json

import httpx
import pytest

from skulk.engines.cli import Json, main

BUILD = "c" * 64
OLDER = "b" * 64
HELD: Json = {
    "engineId": "foxlight.tensorfold",
    "lane": "rocm",
    "backendTag": "tensorfold-rocm",
    "status": "active",
    "build": BUILD,
    "engineVersion": "1.3.2",
    "installedBuilds": [OLDER, BUILD],
    "rollbackBuild": OLDER,
    "loadResults": [
        {
            "build": BUILD,
            "outcome": "refused",
            "code": "load_refused",
            "modelId": "org/model",
            "atUnixMs": 1,
        }
    ],
}
OFFERED: Json = {**HELD, "lane": "cuda", "status": "offered", "installedBuilds": []}


def _report(*entries: tuple[str, Json]) -> Json:
    return {
        "packagesEnabled": True,
        "engines": [
            {
                "engineId": "foxlight.tensorfold",
                "name": "tensorfold",
                "nodes": [{"nodeId": node, "package": package} for node, package in entries],
            }
        ]
        if entries
        else [],
    }


class _Node:
    """A stand-in for the local node's API that records what it was asked."""

    def __init__(self, report: Json, change: tuple[int, Json] | None = None) -> None:
        self.report = report
        self.change = change if change is not None else (200, {"package": HELD})
        self.asked: list[tuple[str, str, Json | None]] = []

    def __call__(
        self, method: str, path: str, body: Json | None, timeout: float
    ) -> httpx.Response:
        self.asked.append((method, path, body))
        if path == "/node_id":
            return httpx.Response(200, json="node-a")
        if path == "/v1/engines":
            return httpx.Response(200, json=self.report)
        return httpx.Response(self.change[0], json=self.change[1])


def test_list_shows_each_node_its_builds_and_its_last_load(
    capsys: pytest.CaptureFixture[str],
) -> None:
    node = _Node(_report(("node-a", HELD), ("node-b", OFFERED)))

    assert main(["list"], call=node) == 0

    shown = capsys.readouterr().out
    assert shown.splitlines()[0] == "foxlight.tensorfold"
    assert f"node-a: rocm, active, build {BUILD[:12]}, version 1.3.2" in shown
    assert f"earlier build {OLDER[:12]}, 2 builds on disk" in shown
    assert "last load refused (load_refused): org/model" in shown
    # The full digest is what activate and remove take.
    assert f"on disk: {OLDER}" in shown
    assert "node-b: cuda, offered" in shown


def test_list_says_so_when_there_is_nothing_and_can_print_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["list"], call=_Node(_report())) == 0
    assert "No node holds or is offered an engine package." in capsys.readouterr().out

    assert main(["list", "--json"], call=_Node(_report(("node-a", HELD)))) == 0
    assert json.loads(capsys.readouterr().out)[0]["nodes"][0]["package"] == HELD


def test_a_change_names_the_node_and_lane_it_was_given(
    capsys: pytest.CaptureFixture[str],
) -> None:
    node = _Node(_report(("node-a", HELD)))

    code = main(
        ["activate", "foxlight.tensorfold", OLDER, "--node", "node-b", "--lane", "cuda"],
        call=node,
    )

    assert code == 0
    assert node.asked == [
        (
            "POST",
            "/v1/engines/foxlight.tensorfold/activate",
            {"nodeId": "node-b", "lane": "cuda", "build": OLDER},
        )
    ]
    assert "foxlight.tensorfold on node-b: rocm, active" in capsys.readouterr().out


def test_without_a_node_or_lane_this_machines_only_package_is_meant() -> None:
    node = _Node(_report(("node-a", HELD), ("node-a", OFFERED), ("node-b", HELD)))

    assert main(["rollback", "foxlight.tensorfold"], call=node) == 0
    assert main(["remove", "foxlight.tensorfold"], call=node) == 0
    assert main(["remove", "foxlight.tensorfold", OLDER], call=node) == 0

    changes = [(path, body) for method, path, body in node.asked if method == "POST"]
    assert changes == [
        ("/v1/engines/foxlight.tensorfold/rollback", {"nodeId": "node-a", "lane": "rocm"}),
        ("/v1/engines/foxlight.tensorfold/remove", {"nodeId": "node-a", "lane": "rocm"}),
        (
            "/v1/engines/foxlight.tensorfold/remove",
            {"nodeId": "node-a", "lane": "rocm", "build": OLDER},
        ),
    ]


def test_a_node_with_two_lanes_or_none_must_be_told_which(
    capsys: pytest.CaptureFixture[str],
) -> None:
    two = _Node(_report(("node-a", HELD), ("node-a", {**HELD, "lane": "cpu"})))
    assert main(["rollback", "foxlight.tensorfold"], call=two) == 1
    assert "holds foxlight.tensorfold for cpu, rocm. Say which with --lane." in (
        capsys.readouterr().err
    )

    none = _Node(_report(("node-b", HELD)))
    assert main(["rollback", "foxlight.tensorfold"], call=none) == 1
    assert "Node node-a holds no foxlight.tensorfold package" in capsys.readouterr().err
    assert all(method == "GET" for method, _, _ in (*two.asked, *none.asked))


def test_a_removed_package_and_a_refusal_are_both_said_plainly(
    capsys: pytest.CaptureFixture[str],
) -> None:
    gone = _Node(_report(("node-a", HELD)), change=(200, {"package": None}))
    assert main(["remove", "foxlight.tensorfold"], call=gone) == 0
    assert "the node no longer holds this package" in capsys.readouterr().out

    refused = _Node(
        _report(("node-a", HELD)),
        change=(
            409,
            {
                "error": {
                    "message": "A model on the node is running on a build this would delete. Stop it first.",
                    "type": "Conflict",
                    "code": 409,
                }
            },
        ),
    )
    assert main(["remove", "foxlight.tensorfold"], call=refused) == 1
    assert capsys.readouterr().err.strip() == (
        "A model on the node is running on a build this would delete. Stop it first."
    )

    malformed = _Node(_report(("node-a", HELD)), change=(422, {"detail": [{"loc": ["body"]}]}))
    assert main(["remove", "foxlight.tensorfold"], call=malformed) == 1
    assert capsys.readouterr().err.strip() == "The node answered 422."


def test_a_node_that_is_not_running_is_said_so(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def unreachable(method: str, path: str, body: Json | None, timeout: float) -> httpx.Response:
        raise httpx.ConnectError("refused")

    assert main(["list"], call=unreachable) == 1
    assert "did not answer (ConnectError). Is it running?" in capsys.readouterr().err


def test_an_unknown_lane_is_a_command_line_error() -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["rollback", "foxlight.tensorfold", "--lane", "gpu"], call=_Node(_report()))
    assert exit_info.value.code == 2
