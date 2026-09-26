# pyright: reportPrivateUsage=false, reportAny=false
"""Preparation reports success only for a live, build-verified audio.cpp lane."""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any, Literal
from unittest.mock import AsyncMock

import pytest

import skulk.facts as facts
import skulk.provisioning.audio_cpp as provisioning
from skulk.shared.types.common import CommandId, NodeId
from skulk.shared.types.events import AudioCppPreparationRequested
from skulk.shared.types.profiling import NodeResources
from skulk.worker.main import Worker


@pytest.mark.parametrize(
    ("variant", "lane", "builds", "expected_success"),
    [
        ("cpu", "audio_cpp-metal", {"audio_cpp-metal": "verified-build"}, True),
        ("cpu", "audio_cpp-cuda", {"audio_cpp-cuda": "verified-build"}, True),
        ("cpu", "audio_cpp-rocm", {"audio_cpp-rocm": "verified-build"}, True),
        ("cpu", "audio_cpp-metal", {}, False),
        ("vulkan", "audio_cpp-vulkan", {"audio_cpp-vulkan": "verified-build"}, True),
        ("vulkan", "audio_cpp-cpu", {"audio_cpp-cpu": "verified-build"}, False),
    ],
)
async def test_prepared_gpu_only_lane_requires_matching_build(
    monkeypatch: pytest.MonkeyPatch, variant: Literal["cpu", "vulkan"], lane: str,
    builds: dict[str, str], expected_success: bool,
) -> None:
    """A GPU-only override can mount, while an unverified lane cannot."""
    def prepared(*, allow_download: bool, variant: Literal["cpu", "vulkan"]) -> None:
        assert allow_download
        assert variant == requested_variant

    requested_variant = variant

    monkeypatch.setattr(provisioning, "prepare_audio_cpp", prepared)
    monkeypatch.setattr(facts, "refresh_node_facts", lambda: None)
    resources = NodeResources(
        backends=frozenset({"audio_cpp", lane}),
        engine_builds=builds,
    )
    monkeypatch.setattr(NodeResources, "gather", AsyncMock(return_value=resources))
    worker: Any = object.__new__(Worker)
    worker.node_id = NodeId("worker-node")
    worker._offline = False
    worker._zenoh_peer_sampler = None
    worker._api_available = True
    worker._data_transport = "gossipsub"
    worker._telemetry_sender = SimpleNamespace(send=AsyncMock())
    worker.event_sender = SimpleNamespace(send=AsyncMock())
    request = AudioCppPreparationRequested(
        request_id=CommandId("prepare-music"),
        target_node=worker.node_id,
        owner_node=NodeId("api-node"),
        variant=variant,
        expires_at=time.time() + 60,
    )

    await worker._prepare_audio_cpp_engine(request)

    completed = worker.event_sender.send.await_args.args[0]
    assert completed.success is expected_success
    if expected_success:
        worker._telemetry_sender.send.assert_awaited_once()
        assert completed.resources == resources
    else:
        worker._telemetry_sender.send.assert_not_awaited()
        assert completed.resources is None
