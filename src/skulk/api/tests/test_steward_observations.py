"""Coverage and non-generative answers for Steward's protected inventory reads."""

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, cast

import pytest

from skulk.api.steward import (
    MAX_TOOL_RESULT_CHARS,
    StewardChatMessage,
    StewardHarness,
    steward_operator_summary,
    steward_operator_tool_result,
)
from skulk.api.steward_observations import (
    factual_question,
    observe_state,
    render_observations,
)
from skulk.shared.types.chunks import TokenChunk
from skulk.shared.types.worker.instances import InstanceId

if TYPE_CHECKING:
    from skulk.api.main import API

NOW = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)


_INVALID_DOWNLOADS: list[object] = [
    None,
    [],
    {"a": None},
    {"a": [{}]},
    {"a": [{"FutureStatus": {}}]},
    {"a": [{"DownloadOngoing": None}]},
]


@pytest.mark.parametrize("downloads", _INVALID_DOWNLOADS)
def test_missing_or_malformed_download_coverage_is_unknown(downloads: object) -> None:
    facts = observe_state({"downloads": downloads}, read_at=NOW)
    assert facts.transferring is None
    assert facts.queued is None
    assert "incomplete" in render_observations(facts, "downloads")
    assert facts.telemetry_observed_at is None


def test_topology_scope_and_empty_download_inventory() -> None:
    facts = observe_state(
        {
            "topology": {"nodes": ["a", "b", "a"]},
            "nodeIdentities": {"c": {}},
            "capabilityNodes": {"a": [{}, {}]},
            "downloads": {},
        },
        read_at=NOW,
    )
    assert facts.node_count == 2
    assert facts.transferring == 0
    assert "transport peers" in render_observations(facts, "nodes")
    assert "0 transferring" in render_observations(facts, "downloads")
    assert observe_state({}, read_at=NOW).node_count is None


@pytest.mark.parametrize(
    "question,topic",
    [
        ("Good morning how many nodes do you currently have", "nodes"),
        ("What's the current node count?", "nodes"),
        ("How many nodes?", "nodes"),
        ("How many nodes are in this cluster?", "nodes"),
        ("How many nodes do we currently have?", "nodes"),
        ("How many nodes are connected right now?", "nodes"),
        ("Are there any downloads in flight?", "downloads"),
        ("What's downloading right now?", "downloads"),
        ("Current download status", "downloads"),
        ("How many nodes do you have and why is one unhealthy?", None),
        ("Cancel any active downloads", None),
        ("How many downloads are running for org/model?", None),
        ("Do you have enough nodes to run this model?", None),
        ("What is a node count?", None),
        ("What was downloading yesterday?", None),
    ],
)
def test_only_standalone_inventory_requests_are_routed(
    question: str, topic: str | None
) -> None:
    assert factual_question(question) == topic


@pytest.mark.parametrize("vendor", ["amd", "nvidia", "apple"])
def test_hardware_does_not_invent_backend_support(vendor: str) -> None:
    payload: dict[str, object] = {
        "topology": {"nodes": ["a"]},
        "nodeSystem": {"a": {"accelerator": {"vendor": vendor}}},
        "nodeResources": {
            "a": {"backends": ["llama_server-vulkan"], "hardwareClasses": [vendor]}
        },
    }
    nodes = cast("list[dict[str, object]]", steward_operator_summary(payload)["nodes"])
    assert nodes[0]["supports"] == {"cuda": False, "rocm": False, "mlx": False}
    payload["nodeResources"] = {"a": {"hardwareClasses": [vendor]}}
    nodes = cast("list[dict[str, object]]", steward_operator_summary(payload)["nodes"])
    assert nodes[0]["supports"] == {"cuda": None, "rocm": None, "mlx": None}


def test_compaction_preserves_counts_and_active_download_before_history() -> None:
    payload: dict[str, object] = {
        "topology": {"nodes": ["a", "b"]},
        "downloads": {
            "a": [{"DownloadCompleted": {}} for _ in range(100)],
            "b": [
                {"DownloadOngoing": {}},
                {"DownloadPending": {}},
                {"DownloadFailed": {}},
            ],
        },
        "instanceFailures": [{"errorMessage": "old failure " * 20} for _ in range(50)],
    }
    result = steward_operator_tool_result(payload)
    assert len(result) <= MAX_TOOL_RESULT_CHARS
    parsed = cast("dict[str, object]", json.loads(result))
    observations = cast("dict[str, object]", parsed["observations"])
    downloads = cast("dict[str, list[dict[str, object]]]", parsed["downloads"])
    coverage = cast("dict[str, dict[str, int]]", parsed["coverage"])
    assert observations["transferring"] == 1
    assert observations["queued"] == 1
    assert observations["completed"] == 100
    assert any(
        row["lifecycle"] == "downloading" for rows in downloads.values() for row in rows
    )
    assert coverage["downloads"]["included"] < coverage["downloads"]["total"]


@pytest.mark.parametrize(
    "question,expected",
    [
        ("Good morning how many nodes do you currently have", "3 nodes"),
        ("Are there any downloads in flight?", "0 transferring and 1 queued"),
    ],
)
async def test_inventory_answers_cannot_stream_model_hallucinations(
    question: str, expected: str
) -> None:
    class _Api:
        async def get_cluster_state(self) -> dict[str, object]:
            return {
                "topology": {"nodes": ["a", "b", "c"]},
                "downloads": {
                    "a": [{"DownloadCompleted": {}}, {"DownloadPending": {}}]
                },
            }

    harness = StewardHarness(cast("API", cast(object, _Api())))
    harness.steward_instance = lambda: (InstanceId(), "org/brain")
    # No generation API is supplied: entering generation fails the test.
    chunks = [
        chunk
        async for chunk in harness.run_turn_chunks(
            [
                StewardChatMessage(
                    role="assistant", content="Four nodes and six active downloads."
                ),
                StewardChatMessage(role="user", content=question),
            ]
        )
    ]
    assert all(isinstance(chunk, TokenChunk) for chunk in chunks)
    content = "".join(
        chunk.text
        for chunk in chunks
        if isinstance(chunk, TokenChunk) and not chunk.is_thinking
    )
    assert expected in content
    assert "Four" not in content
    assert "six" not in content
    assert isinstance(chunks[-1], TokenChunk) and chunks[-1].finish_reason == "stop"


async def test_repeated_inventory_question_reads_new_state() -> None:
    class _Api:
        def __init__(self) -> None:
            self.count = 0

        async def get_cluster_state(self) -> dict[str, object]:
            self.count += 1
            return {
                "topology": {"nodes": [str(index) for index in range(self.count)]},
                "downloads": {},
            }

    api = _Api()
    harness = StewardHarness(cast("API", cast(object, api)))
    harness.steward_instance = lambda: (InstanceId(), "org/brain")
    for count in (1, 2):
        chunks = [
            chunk
            async for chunk in harness.run_turn_chunks(
                [
                    StewardChatMessage(
                        role="assistant", content="There are four nodes."
                    ),
                    StewardChatMessage(role="user", content="How many nodes?"),
                ]
            )
        ]
        content = "".join(
            chunk.text
            for chunk in chunks
            if isinstance(chunk, TokenChunk) and not chunk.is_thinking
        )
        assert f"{count} nodes" in content
    assert api.count == 2


def test_missing_topology_does_not_become_zero_in_model_evidence() -> None:
    result = cast(
        "dict[str, object]", json.loads(steward_operator_tool_result({"downloads": {}}))
    )
    assert result["nodeCount"] is None
