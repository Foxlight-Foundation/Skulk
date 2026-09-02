"""Steward node-scoped diagnostics use friendly names and remote bundles."""

import json
from typing import TYPE_CHECKING, Literal, cast

import pytest

from skulk.api.steward import StewardHarness
from skulk.shared.types.diagnostics import DoctorCheckDiagnostics

if TYPE_CHECKING:
    from skulk.api.main import API


class _Diagnostics:
    def __init__(self) -> None:
        self.doctor = [
            DoctorCheckDiagnostics(
                check_id="engine",
                title="Serving engine",
                verdict="ok",
                detail="vLLM is available",
            )
        ]

    def model_dump(
        self,
        *,
        by_alias: bool = False,
        mode: Literal["json", "python"] = "python",
    ) -> dict[str, object]:
        del by_alias, mode
        return {
            "doctor": [
                result.model_dump(mode="json", by_alias=True)
                for result in self.doctor
            ],
            "warnings": [],
        }


class _Api:
    def __init__(self) -> None:
        self.requested_node_ids: list[str] = []

    async def get_cluster_state(self) -> dict[str, object]:
        return {
            "topology": {"nodes": ["internal-node-id"]},
            "nodeIdentities": {
                "internal-node-id": {"friendlyName": "GPU Worker"}
            },
        }

    async def get_cluster_node_diagnostics(self, node_id: str) -> _Diagnostics:
        self.requested_node_ids.append(node_id)
        return _Diagnostics()


@pytest.mark.asyncio
async def test_get_node_diagnostics_routes_by_friendly_name() -> None:
    api = _Api()
    result = cast(
        "dict[str, object]",
        json.loads(
            await StewardHarness(cast("API", cast(object, api))).execute_tool(
            "get_node_diagnostics", {"node_name": "GPU Worker"}
            )
        ),
    )

    assert api.requested_node_ids == ["internal-node-id"]
    assert result["node"] == "GPU Worker"
    assert "internal-node-id" not in json.dumps(result)


@pytest.mark.asyncio
async def test_run_doctor_returns_the_selected_nodes_findings() -> None:
    api = _Api()
    result = cast(
        "dict[str, object]",
        json.loads(
            await StewardHarness(cast("API", cast(object, api))).execute_tool(
            "run_doctor", {"node_name": "GPU Worker"}
            )
        ),
    )

    assert api.requested_node_ids == ["internal-node-id"]
    assert result == {
        "node": "GPU Worker",
        "results": [
            {
                "checkId": "engine",
                "title": "Serving engine",
                "verdict": "ok",
                "detail": "vLLM is available",
                "consequence": "",
                "remediation": "",
                "fixAvailable": False,
            }
        ],
    }
