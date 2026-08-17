"""Regression coverage for the fabric resident's factual operator projection."""

from typing import cast

from skulk.api.steward import steward_operator_summary


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
    by_id = {cast(str, node["nodeId"]): node for node in nodes}
    assert by_id["apple-node"]["chip"] == "Apple M4"
    assert by_id["amd-node"]["chip"] == "AMD Ryzen AI MAX+ 395"
    assert by_id["cuda-node"]["chip"] == "NVIDIA GB10"
    assert cast("dict[str, object]", by_id["amd-node"]["memory"])["ramTotalGiB"] == 128.0
    assert cast("dict[str, object]", by_id["cuda-node"]["supports"]) == {
        "cuda": True,
        "rocm": False,
        "mlx": False,
    }
    assert cast("dict[str, object]", by_id["amd-node"]["supports"])["rocm"] is True
    assert cast("dict[str, object]", by_id["apple-node"]["supports"])["mlx"] is True


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

    active = cast("list[dict[str, object]]", summary["activePlacements"])
    ready = cast("list[dict[str, object]]", summary["readyOrRunningInstances"])
    assert [(row["modelId"], row["lifecycle"]) for row in active] == [
        ("org/placing-model", "loading")
    ]
    assert [(row["modelId"], row["lifecycle"]) for row in ready] == [
        ("org/ready-model", "ready")
    ]


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
    assert [(row["modelId"], row["lifecycle"], row["active"]) for row in downloads["node-a"]] == [
        ("org/already-downloaded", "completed", False),
        ("org/downloading-now", "downloading", True),
    ]
