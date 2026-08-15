from skulk.api.steward import (
    _instance_failures_summary,  # pyright: ignore[reportPrivateUsage] - unit under test
)


def test_steward_receives_recent_instance_failure_truth() -> None:
    payload: dict[str, object] = {
        "instanceFailures": [
            {
                "instanceId": "instance-a",
                "modelId": "org/model",
                "errorCode": "runner_crashed",
                "errorMessage": "runner exited while serving the model",
                "affectedNodeIds": ["node-a", "node-b"],
                "recordedAt": "2026-08-15T14:00:00Z",
            }
        ]
    }

    assert _instance_failures_summary(payload) == [
        {
            "instanceId": "instance-a",
            "modelId": "org/model",
            "errorCode": "runner_crashed",
            "errorMessage": "runner exited while serving the model",
            "affectedNodeIds": ["node-a", "node-b"],
            "recordedAt": "2026-08-15T14:00:00Z",
        }
    ]
