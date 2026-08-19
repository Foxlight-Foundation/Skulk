"""Regression coverage for the fabric resident's factual operator projection."""

import json
from typing import cast

from skulk.api.steward import (
    MAX_TOOL_RESULT_CHARS,
    steward_operator_summary,
    steward_operator_tool_result,
)


def _instance(
    *, model_id: str, node_to_runner: dict[str, str], system_role: str | None = None
) -> dict[str, object]:
    return {
        "MlxRingInstance": {
            "instanceId": f"instance-{model_id}",
            "shardAssignments": {
                "modelId": model_id,
                "nodeToRunner": node_to_runner,
                "runnerToShard": {},
            },
            "systemRole": system_role,
        }
    }


def test_operator_summary_preserves_heterogeneous_node_truth() -> None:
    payload: dict[str, object] = {
        "topology": {"nodes": ["apple-node", "amd-node", "cuda-node"]},
        "nodeIdentities": {
            "apple-node": {
                "friendlyName": "Apple",
                "modelId": "Mac mini",
                "chipId": "Apple M4",
                "osVersion": "macOS",
            },
            "amd-node": {
                "friendlyName": "AMD",
                "modelId": "AI workstation",
                "chipId": "AMD Ryzen AI MAX+ 395",
                "osVersion": "Linux",
            },
            "cuda-node": {
                "friendlyName": "CUDA",
                "modelId": "GX10",
                "chipId": "NVIDIA GB10",
                "osVersion": "Linux",
            },
        },
        "nodeMemory": {
            "apple-node": {
                "ramTotal": {"inBytes": 16 * 1024**3},
                "ramAvailable": {"inBytes": 8 * 1024**3},
            },
            "amd-node": {
                "ramTotal": {"inBytes": 128 * 1024**3},
                "ramAvailable": {"inBytes": 96 * 1024**3},
            },
            "cuda-node": {
                "ramTotal": {"inBytes": 128 * 1024**3},
                "ramAvailable": {"inBytes": 112 * 1024**3},
            },
        },
        "nodeSystem": {
            "apple-node": {"accelerator": {"vendor": "apple", "name": "M4"}},
            "amd-node": {"accelerator": {"vendor": "amd", "name": "Radeon 8060S"}},
            "cuda-node": {
                "accelerator": {
                    "vendor": "nvidia",
                    "name": "NVIDIA GB10",
                    "computeCapability": "12.1",
                    "nativeFp4": True,
                    "nativeFp8": True,
                }
            },
        },
        "nodeResources": {
            "apple-node": {
                "backends": ["mlx"],
                "hardwareClasses": ["apple", "apple:m4"],
                "participation": "full",
            },
            "amd-node": {
                "backends": ["llama_server-rocm"],
                "hardwareClasses": ["amd", "amd:amd-gpu"],
                "participation": "full",
            },
            "cuda-node": {
                "backends": ["llama_server-cuda"],
                "hardwareClasses": ["nvidia", "nvidia:nvidia-gb10", "nvidia:sm-12.1"],
                "participation": "full",
            },
        },
        "nodeHealth": {
            "apple-node": {"level": "healthy"},
            "amd-node": {"level": "healthy"},
            "cuda-node": {"level": "healthy"},
        },
    }

    summary = steward_operator_summary(payload)

    assert summary["nodeCount"] == 3
    nodes = cast("list[dict[str, object]]", summary["nodes"])
    by_name = {cast(str, node["name"]): node for node in nodes}
    assert by_name["Apple"]["chip"] == "Apple M4"
    assert by_name["AMD"]["chip"] == "AMD Ryzen AI MAX+ 395"
    assert by_name["CUDA"]["chip"] == "NVIDIA GB10"
    assert cast("dict[str, object]", by_name["AMD"]["memory"])["ramTotalGiB"] == 128.0
    assert cast("dict[str, object]", by_name["CUDA"]["supports"]) == {
        "cuda": True,
        "rocm": False,
        "mlx": False,
    }
    assert cast("dict[str, object]", by_name["AMD"]["supports"])["rocm"] is True
    assert cast("dict[str, object]", by_name["Apple"]["supports"])["mlx"] is True
    assert "nodeId" not in nodes[0]


def test_operator_summary_does_not_invent_negative_capability_truth() -> None:
    payload: dict[str, object] = {
        "topology": {"nodes": ["joining-node"]},
        "nodeIdentities": {
            "joining-node": {
                "friendlyName": "Joining",
                "modelId": "Unknown workstation",
            }
        },
    }

    summary = steward_operator_summary(payload)

    nodes = cast("list[dict[str, object]]", summary["nodes"])
    assert cast("dict[str, object]", nodes[0]["supports"]) == {
        "cuda": None,
        "rocm": None,
        "mlx": None,
    }


def test_operator_tools_name_telemetry_only_nodes_without_changing_node_count() -> None:
    topology_node = "12D3KooWTopologyNode123456789"
    telemetry_node = "12D3KooWTelemetryOnlyNode123456789"
    payload: dict[str, object] = {
        "topology": {"nodes": [topology_node]},
        "nodeIdentities": {
            topology_node: {"friendlyName": "Worker"},
            telemetry_node: {"friendlyName": "Operator API"},
        },
        "nodeResources": {
            topology_node: {"backends": ["mlx"]},
            telemetry_node: {"backends": ["management"]},
        },
        "nodeDisk": {
            telemetry_node: {"freeBytes": 10 * 1024**3},
        },
    }

    summary = steward_operator_summary(payload)
    rendered = steward_operator_tool_result(payload)

    assert summary["nodeCount"] == 1
    assert cast("dict[str, object]", summary["nodeDisk"]) == {
        "Operator API": {"freeBytes": 10 * 1024**3}
    }
    assert "Operator API" in rendered
    assert telemetry_node not in rendered


def test_operator_summary_separates_active_placement_from_ready_instances() -> None:
    payload: dict[str, object] = {
        "topology": {"nodes": ["node-a", "node-b"]},
        "instances": {
            "placing": _instance(
                model_id="org/placing-model", node_to_runner={"node-a": "runner-loading"}
            ),
            "ready": _instance(
                model_id="org/ready-model", node_to_runner={"node-b": "runner-ready"}
            ),
        },
        "runners": {
            "runner-loading": {"RunnerLoading": {"layersLoaded": 2, "totalLayers": 20}},
            "runner-ready": {"RunnerReady": {}},
        },
    }

    summary = steward_operator_summary(payload)

    active = cast("list[dict[str, object]]", summary["operatorActivePlacements"])
    ready = cast(
        "list[dict[str, object]]", summary["operatorReadyOrRunningInstances"]
    )
    assert [(row["modelId"], row["lifecycle"]) for row in active] == [
        ("org/placing-model", "loading")
    ]
    assert [(row["modelId"], row["lifecycle"]) for row in ready] == [
        ("org/ready-model", "ready")
    ]


def test_operator_summary_separates_internal_services_and_historical_failures() -> None:
    payload: dict[str, object] = {
        "topology": {"nodes": ["node-a", "node-b"]},
        "instances": {
            "operator-ready": _instance(
                model_id="org/operator-model",
                node_to_runner={"node-a": "runner-operator"},
            ),
            "fabric-ready": _instance(
                model_id="org/steward-brain",
                node_to_runner={"node-b": "runner-steward"},
                system_role="steward",
            ),
        },
        "runners": {
            "runner-operator": {"RunnerReady": {}},
            "runner-steward": {"RunnerReady": {}},
        },
        "instanceFailures": [
            {
                "instanceId": "vanished-instance",
                "modelId": "org/old-model",
                "errorCode": "runner_crashed",
                "errorMessage": "Runner exited.",
                "affectedNodeIds": ["node-a"],
                "recordedAt": "2026-08-17T12:00:00Z",
            }
        ],
    }

    summary = steward_operator_summary(payload)

    operator_ready = cast(
        "list[dict[str, object]]", summary["operatorReadyOrRunningInstances"]
    )
    system_instances = cast(
        "list[dict[str, object]]", summary["fabricSystemInstances"]
    )
    failures = cast(
        "list[dict[str, object]]", summary["historicalTerminalFailures"]
    )
    assert [row["modelId"] for row in operator_ready] == ["org/operator-model"]
    assert [row["modelId"] for row in system_instances] == ["org/steward-brain"]
    assert system_instances[0]["systemRole"] == "steward"
    assert failures == [
        {
            "historical": True,
            "currentInstance": False,
            "modelId": "org/old-model",
            "errorCode": "runner_crashed",
            "errorMessage": "Runner exited.",
            "affectedNodes": ["Node 1"],
            "recordedAt": "2026-08-17T12:00:00Z",
        }
    ]


def test_operator_summary_keeps_replacement_instance_current_with_same_model_id() -> None:
    payload: dict[str, object] = {
        "topology": {"nodes": ["node-a"]},
        "instances": {
            "replacement-instance": _instance(
                model_id="org/recovered-model",
                node_to_runner={"node-a": "replacement-runner"},
            )
        },
        "runners": {"replacement-runner": {"RunnerReady": {}}},
        "instanceFailures": [
            {
                "instanceId": "failed-instance",
                "modelId": "org/recovered-model",
                "errorCode": "runner_crashed",
                "errorMessage": "Earlier runner exited.",
                "affectedNodeIds": ["node-a"],
                "recordedAt": "2026-08-17T12:00:00Z",
            }
        ],
    }

    summary = steward_operator_summary(payload)

    ready = cast(
        "list[dict[str, object]]", summary["operatorReadyOrRunningInstances"]
    )
    failures = cast(
        "list[dict[str, object]]", summary["historicalTerminalFailures"]
    )
    assert [(row["modelId"], row["lifecycle"]) for row in ready] == [
        ("org/recovered-model", "ready")
    ]
    assert [row["modelId"] for row in failures] == ["org/recovered-model"]
    assert "instanceId" not in ready[0]
    assert "instanceId" not in failures[0]


def test_operator_summary_marks_terminal_downloads_inactive() -> None:
    payload: dict[str, object] = {
        "topology": {"nodes": ["node-a"]},
        "downloads": {
            "node-a": [
                {
                    "DownloadCompleted": {
                        "shardMetadata": {
                            "MlxShardMetadata": {
                                "modelCard": {"modelId": "org/already-downloaded"}
                            }
                        }
                    }
                },
                {
                    "DownloadOngoing": {
                        "shardMetadata": {
                            "MlxShardMetadata": {
                                "modelCard": {"modelId": "org/downloading-now"}
                            }
                        },
                        "downloadProgress": {
                            "downloaded": {"inBytes": 5},
                            "total": {"inBytes": 10},
                            "etaMs": 1000,
                        },
                    }
                },
            ]
        },
    }

    summary = steward_operator_summary(payload)
    downloads = cast("dict[str, list[dict[str, object]]]", summary["downloads"])
    assert [(row["modelId"], row["lifecycle"], row["active"]) for row in downloads["Node 1"]] == [
        ("org/already-downloaded", "completed", False),
        ("org/downloading-now", "downloading", True),
    ]


def test_operator_tool_result_preserves_lifecycle_truth_when_nodes_are_large() -> None:
    node_ids = [f"node-{index}" for index in range(12)]
    payload: dict[str, object] = {
        "topology": {"nodes": node_ids},
        "nodeIdentities": {
            node_id: {
                "friendlyName": f"Detailed node {index} " + "x" * 180,
                "modelId": "Workstation " + "y" * 180,
                "chipId": "Heterogeneous accelerator " + "z" * 180,
                "osVersion": "Linux distribution with extensive build metadata " + "v" * 180,
            }
            for index, node_id in enumerate(node_ids)
        },
        "nodeResources": {
            node_id: {
                "backends": ["llama_server-cuda", "llama_cpp-cpu"],
                "hardwareClasses": ["nvidia", "nvidia:sm-12.1"],
                "participation": "full",
            }
            for node_id in node_ids
        },
        "instances": {
            "placing": _instance(
                model_id="org/model-being-placed",
                node_to_runner={"node-0": "runner-loading"},
            ),
            "ready": _instance(
                model_id="org/model-ready",
                node_to_runner={"node-1": "runner-ready"},
            ),
        },
        "runners": {
            "runner-loading": {"RunnerLoading": {}},
            "runner-ready": {"RunnerReady": {}},
        },
    }

    rendered = steward_operator_tool_result(payload)
    result = cast("dict[str, object]", json.loads(rendered))

    assert len(rendered) <= MAX_TOOL_RESULT_CHARS
    assert result["detailState"] == "compacted"
    assert result["nodeCount"] == 12
    active = cast("list[dict[str, object]]", result["operatorActivePlacements"])
    ready = cast(
        "list[dict[str, object]]", result["operatorReadyOrRunningInstances"]
    )
    assert [(row["modelId"], row["lifecycle"]) for row in active] == [
        ("org/model-being-placed", "loading")
    ]
    assert [(row["modelId"], row["lifecycle"]) for row in ready] == [
        ("org/model-ready", "ready")
    ]
    coverage = cast("dict[str, dict[str, int]]", result["coverage"])
    assert coverage["nodes"] == {"included": 12, "total": 12}


def test_operator_tool_result_keeps_downloads_from_multiple_nodes() -> None:
    payload: dict[str, object] = {
        "topology": {"nodes": ["node-a", "node-b"]},
        "nodeIdentities": {
            "node-a": {"friendlyName": "Node A"},
            "node-b": {"friendlyName": "Node B"},
        },
        "downloads": {
            "node-a": [{"DownloadPending": {}}],
            "node-b": [{"DownloadPending": {}}],
        },
        # Force the bounded representation without competing with the compact
        # rows whose cross-node behavior this regression covers.
        "nodeDisk": {"diagnostic": "x" * MAX_TOOL_RESULT_CHARS},
    }

    rendered = steward_operator_tool_result(payload)
    result = cast("dict[str, object]", json.loads(rendered))

    downloads = cast("dict[str, list[dict[str, object]]]", result["downloads"])
    assert sorted(downloads) == ["Node A", "Node B"]
    coverage = cast("dict[str, dict[str, int]]", result["coverage"])
    assert coverage["downloads"] == {"included": 2, "total": 2}


def test_operator_tool_result_never_exposes_internal_identifiers() -> None:
    node_id = "12D3KooWPrivateRoutingIdentity123456789"
    instance_id = "b25203ab-6eb8-44b1-8b3d-9df19e751906"
    payload: dict[str, object] = {
        "topology": {"nodes": [node_id]},
        "nodeIdentities": {node_id: {"friendlyName": "Studio"}},
        "instances": {
            instance_id: _instance(
                model_id="org/current-model", node_to_runner={node_id: "runner-private"}
            )
        },
        "runners": {"runner-private": {"RunnerReady": {}}},
        "instanceFailures": [
            {
                "instanceId": "old-private-instance",
                "modelId": "org/old-model",
                "errorCode": "node_unavailable",
                "errorMessage": f"Assigned node {node_id} timed out.",
                "affectedNodeIds": [node_id],
                "recordedAt": "2026-08-19T12:00:00Z",
            }
        ],
    }

    rendered = steward_operator_tool_result(payload)

    assert "Studio" in rendered
    assert node_id not in rendered
    assert instance_id not in rendered
    assert "runner-private" not in rendered
