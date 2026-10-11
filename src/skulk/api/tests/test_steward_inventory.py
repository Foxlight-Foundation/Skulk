"""Regression cases for operator-facing identity and capability-node evidence."""

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from skulk.api.steward import (
    MAX_TOOL_RESULT_CHARS,
    StewardChatMessage,
    StewardHarness,
    steward_operator_tool_result,
)
from skulk.api.steward_inventory import (
    bounded_inventory,
    capability_inventory,
    internal_service_question,
    inventory_question,
    render_installed_inventory,
    render_inventory,
)
from skulk.shared.types.chunks import TokenChunk
from skulk.shared.types.worker.instances import InstanceId

if TYPE_CHECKING:
    from skulk.api.main import API


def test_installed_answer_reports_declarations_and_requirements() -> None:
    """A deterministic inventory answer retains the facts needed to understand use."""
    from skulk.extensions.capabilities import (
        InstalledCapabilityRegistration,
        RegisteredCapabilitySummary,
    )

    registration = InstalledCapabilityRegistration(
        capabilities=(
            RegisteredCapabilitySummary(
                qualified_id="capacity@1.0.0",
                revision="a" * 16,
                title="Capacity",
                purpose="Estimate external offers under owner policy.",
                io_mode="unary",
            ),
        ),
        skulk_requires=">=2.0.3",
        python_requires=">=3.12",
        platforms=("linux",),
        configuration_required=True,
        required_credential_ids=("provider_key",),
    )
    payload: dict[str, object] = {
        "installations": [
            {
                "bundle": "example.plugin",
                "version": "1.0.0",
                "enabled": False,
                "reachable": None,
                "ready_capabilities": [],
                "registration": registration.model_dump(mode="json"),
            }
        ]
    }
    answer = render_installed_inventory(payload)
    assert "capacity@1.0.0" in answer and "Estimate external offers" in answer
    assert (
        "Skulk >=2.0.3" in answer
        and "Python >=3.12" in answer
        and "platforms linux" in answer
    )
    assert "ordinary configuration required" in answer and "provider_key" in answer
    assert "disabled" in answer and "reachability unknown" in answer
    assert "does not prove allocated inference capacity" in answer


def test_installed_answer_discloses_unknown_or_omitted_registration() -> None:
    """Missing or bounded-away declarations are never an empty installed inventory."""
    missing: dict[str, object] = {
        "installations": [{"plugin_id": "managed.fixture", "registration": None}]
    }
    assert "unavailable or omitted" in render_installed_inventory(missing)
    omitted: dict[str, object] = {
        "installations": [],
        "installedCount": 5,
        "omittedInstallations": 5,
    }
    answer = render_installed_inventory(omitted)
    assert "5 additional installations" in answer
    assert "No entries were returned" not in answer


@pytest.mark.parametrize(
    "question",
    ["What version are you", "What skulk version?", "What version are you running?"],
)
def test_version_questions_select_observed_builds(question: str) -> None:
    assert inventory_question(question) == "versions"


@pytest.mark.parametrize(
    "question",
    ["What capabilities are available?", "Which capability nodes are installed?"],
)
def test_capabilities_mean_capability_nodes(question: str) -> None:
    assert inventory_question(question) == "capability_nodes"


@pytest.mark.parametrize(
    "question",
    [
        "What version of Python should I install?",
        "What capabilities are available and how do I add one?",
        "What GPUs are available?",
    ],
)
def test_unrelated_and_compound_questions_keep_investigation(question: str) -> None:
    assert inventory_question(question) is None


@pytest.mark.parametrize(
    "question,expected",
    [
        ("Are your models ready?", False),
        ("What is in your model store?", False),
        ("How are you?", False),
        ("Which node hosts the steward?", True),
        ("Show internal services", True),
    ],
)
def test_only_explicit_internal_subjects_enable_service_details(
    question: str, expected: bool
) -> None:
    assert internal_service_question(question) is expected


def test_internal_service_details_are_opt_in_including_compacted_context() -> None:
    payload: dict[str, object] = {
        "topology": {"nodes": ["a"]},
        "instances": {
            "system": {
                "MlxRingInstance": {
                    "instanceId": "system",
                    "systemRole": "steward",
                    "shardAssignments": {
                        "modelId": "org/private-brain",
                        "nodeToRunner": {"a": "r"},
                        "runnerToShard": {},
                    },
                }
            }
        },
    }
    internal_failure = {
        "modelId": "org/private-brain",
        "systemRole": "steward",
        "errorMessage": "internal runner failed",
    }
    payload["instanceFailures"] = [internal_failure]
    for bulky in (False, True):
        if bulky:
            payload["instanceFailures"] = [internal_failure] + [
                {"errorMessage": "historical " * 100} for _ in range(100)
            ]
        ordinary = steward_operator_tool_result(payload)
        assert "fabricSystemInstances" not in ordinary
        assert "org/private-brain" not in ordinary
        assert "org/private-brain" in steward_operator_tool_result(
            payload, include_internal_services=True
        )
        assert len(ordinary) <= MAX_TOOL_RESULT_CHARS


def test_capability_inventory_distinguishes_absent_and_unknown() -> None:
    empty = capability_inventory(
        {"capabilityNodes": {}, "nodeResources": {"a": {"backends": ["mlx"]}}},
        {"a": "host"},
    )
    answer = render_inventory(empty, "capability_nodes")
    assert "No capability nodes are currently advertised" in answer
    assert "does not prove none are installed" in answer
    assert "mlx" not in answer
    unknown = capability_inventory({}, {})
    assert unknown["advertisedCount"] is None
    assert "Coverage is incomplete" in render_inventory(unknown, "capability_nodes")
    assert "No capability nodes" not in render_inventory(unknown, "capability_nodes")


def test_capability_owner_state_and_bundle_version_remain_distinct() -> None:
    payload: dict[str, object] = {
        "capabilityNodes": {
            "a": [
                {
                    "pluginId": "example",
                    "nodeId": "service",
                    "bundleId": "example.service",
                    "version": "0.1.0",
                    "title": "Example service",
                    "status": "disabled",
                    "ownerAvailable": False,
                    "observedAt": "2026-09-16T00:00:00Z",
                }
            ]
        }
    }
    inventory = capability_inventory(payload, {"a": "host"})
    answer = render_inventory(inventory, "capability_nodes")
    assert (
        "Example service on host: disabled, owner unavailable; bundle version 0.1.0"
        in answer
    )
    assert "permission" in answer
    assert "Skulk version 0.1.0" not in answer
    result = bounded_inventory(inventory, 300)
    parsed = cast("dict[str, object]", json.loads(result))
    assert len(result) <= 300
    assert parsed["omittedRows"] == 1
    assert "omitted" in render_inventory(parsed, "capability_nodes")


@pytest.mark.parametrize(
    "question,expected",
    [
        ("What skulk version?", "1.5.2; commit build123"),
        (
            "What capabilities are available?",
            "No capability nodes are currently advertised",
        ),
    ],
)
async def test_inventory_answers_do_not_generate_invented_values(
    question: str, expected: str
) -> None:
    class _Api:
        async def get_cluster_state(self) -> dict[str, object]:
            return {
                "topology": {"nodes": ["a"]},
                "downloads": {},
                "capabilityNodes": {},
                "nodeIdentities": {"a": {"friendlyName": "host"}},
            }

        async def get_cluster_diagnostics(self) -> SimpleNamespace:
            return SimpleNamespace(
                generated_at="2026-09-16T00:00:00Z",
                version_status="consistent",
                nodes=[
                    SimpleNamespace(
                        node_id="a",
                        ok=True,
                        version_status="current",
                        diagnostics=SimpleNamespace(
                            runtime=SimpleNamespace(
                                skulk_version="1.5.2", skulk_commit="build123"
                            )
                        ),
                    )
                ],
            )

    harness = StewardHarness(cast("API", cast(object, _Api())))
    harness.steward_instance = lambda: (InstanceId(), "org/private-brain")
    chunks = [
        chunk
        async for chunk in harness.run_turn_chunks(
            [
                StewardChatMessage(
                    role="assistant",
                    content="Skulk version 0.1.0 is the latest. My internal brain is on host.",
                ),
                StewardChatMessage(role="user", content=question),
            ]
        )
    ]
    answer = "".join(
        chunk.text
        for chunk in chunks
        if isinstance(chunk, TokenChunk) and not chunk.is_thinking
    )
    assert expected in answer
    assert "0.1.0" not in answer
    assert "org/private-brain" not in answer
    assert isinstance(chunks[-1], TokenChunk) and chunks[-1].finish_reason == "stop"


def test_partial_versions_do_not_claim_all_nodes_current() -> None:
    payload: dict[str, object] = {
        "nodes": [{"name": "host", "skulkVersion": None, "skulkCommit": None}],
        "coverageComplete": False,
    }
    answer = render_inventory(payload, "versions")
    assert "version unavailable" in answer
    assert "Coverage is incomplete" in answer
    assert "not checked" in answer
    assert "consistent" not in answer


@pytest.mark.parametrize(
    "version,commit,missing",
    [
        ("unknown", "build123", "version unavailable"),
        ("1.5.2", "Unknown", "commit unavailable"),
        (" UNKNOWN ", "", "version unavailable"),
        ("None", "build123", "version unavailable"),
        ("1.5.2", "null", "commit unavailable"),
    ],
)
async def test_runtime_sentinels_remain_missing_in_version_answers(
    version: str, commit: str, missing: str
) -> None:
    class _Api:
        async def get_cluster_state(self) -> dict[str, object]:
            return {
                "topology": {"nodes": ["a"]},
                "downloads": {},
                "nodeIdentities": {
                    "a": {"skulkVersion": version, "skulkCommit": commit}
                },
            }

        async def get_cluster_diagnostics(self) -> SimpleNamespace:
            return SimpleNamespace(
                generated_at="2026-09-16T00:00:00Z",
                version_status="unknown",
                nodes=[
                    SimpleNamespace(
                        node_id="a",
                        ok=True,
                        version_status="unknown",
                        diagnostics=SimpleNamespace(
                            runtime=SimpleNamespace(
                                skulk_version=version, skulk_commit=commit
                            )
                        ),
                    )
                ],
            )

    api = _Api()
    # General answers see the same normalized identity evidence as the dedicated
    # version tool, so switching paths cannot turn a sentinel into a build.
    state_result = steward_operator_tool_result(await api.get_cluster_state())
    assert '"unknown"' not in state_result.lower()
    harness = StewardHarness(cast("API", cast(object, api)))
    harness.steward_instance = lambda: (InstanceId(), "org/brain")
    chunks = [
        chunk
        async for chunk in harness.run_turn_chunks(
            [StewardChatMessage(role="user", content="What skulk version?")]
        )
    ]
    answer = "".join(
        chunk.text
        for chunk in chunks
        if isinstance(chunk, TokenChunk) and not chunk.is_thinking
    )
    assert missing in answer
    assert "Coverage is incomplete" in answer
    assert "commit Unknown" not in answer


async def test_capability_only_hosts_keep_advertisements_without_inflating_nodes() -> (
    None
):
    class _Api:
        async def get_cluster_state(self) -> dict[str, object]:
            return {
                "topology": {"nodes": ["physical"]},
                "downloads": {},
                "capabilityNodes": {
                    "telemetry-only-routing-id": [
                        {
                            "pluginId": "example",
                            "nodeId": "service",
                            "bundleId": "example.service",
                            "version": "0.1.0",
                            "title": "Example service",
                            "status": "disabled",
                            "ownerAvailable": False,
                        }
                    ]
                },
            }

    api = _Api()
    state_result = cast(
        "dict[str, object]",
        json.loads(steward_operator_tool_result(await api.get_cluster_state())),
    )
    assert state_result["nodeCount"] == 1
    harness = StewardHarness(cast("API", cast(object, api)))
    harness.steward_instance = lambda: (InstanceId(), "org/brain")
    chunks = [
        chunk
        async for chunk in harness.run_turn_chunks(
            [
                StewardChatMessage(
                    role="user", content="What capabilities are available?"
                )
            ]
        )
    ]
    answer = "".join(
        chunk.text
        for chunk in chunks
        if isinstance(chunk, TokenChunk) and not chunk.is_thinking
    )
    assert "Example service on Node 2: disabled, owner unavailable" in answer
    assert "telemetry-only-routing-id" not in answer
    assert "Coverage is incomplete" not in answer
    assert "No capability nodes" not in answer

    for tool_name in ("get_node_diagnostics", "run_doctor"):
        result = cast(
            "dict[str, str]",
            json.loads(await harness.execute_tool(tool_name, {"node_name": "Node 2"})),
        )
        assert "error" in result
        assert "tool failed" not in result["error"]
