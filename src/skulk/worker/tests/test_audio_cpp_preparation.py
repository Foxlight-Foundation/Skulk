# pyright: reportPrivateUsage=false, reportAny=false
"""Preparation reports success only for a live, build-verified audio.cpp lane."""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

import skulk.facts as facts
import skulk.provisioning.audio_cpp as provisioning
from skulk.shared.types.common import CommandId, NodeId
from skulk.shared.types.events import AudioCppPreparationRequested
from skulk.shared.types.profiling import NodeResources
from skulk.worker.main import Worker


@pytest.mark.parametrize(
    ("builds", "expected_success"),
    [({"audio_cpp-metal": "verified-build"}, True), ({}, False)],
)
async def test_prepared_gpu_only_lane_requires_matching_build(
    monkeypatch: pytest.MonkeyPatch, builds: dict[str, str], expected_success: bool,
) -> None:
    """A GPU-only override can mount, while an unverified lane cannot."""
    def prepared(*, allow_download: bool) -> None:
        assert allow_download

    monkeypatch.setattr(provisioning, "prepare_audio_cpp", prepared)
    monkeypatch.setattr(facts, "refresh_node_facts", lambda: None)
    resources = NodeResources(
        backends=frozenset({"audio_cpp", "audio_cpp-metal"}),
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
        expires_at=time.time() + 60,
    )

    await worker._prepare_audio_cpp_engine(request)

    completed = worker.event_sender.send.await_args.args[0]
    assert completed.success is expected_success
    if expected_success:
        worker._telemetry_sender.send.assert_awaited_once()
    else:
        worker._telemetry_sender.send.assert_not_awaited()
