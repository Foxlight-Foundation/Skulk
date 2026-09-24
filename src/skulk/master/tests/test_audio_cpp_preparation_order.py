# pyright: reportPrivateUsage=false
"""Ordered preparation completion closes the telemetry race before placement."""

import pytest

import skulk.master.main as master_module
from skulk.master.main import Master
from skulk.shared.models.model_cards import ModelCard, ModelId, ModelTask
from skulk.shared.types.commands import PlaceInstance
from skulk.shared.types.common import CommandId, NodeId
from skulk.shared.types.events import AudioCppPreparationCompleted, IndexedEvent
from skulk.shared.types.memory import Memory
from skulk.shared.types.profiling import NodeResources
from skulk.shared.types.state import State
from skulk.shared.types.telemetry import TelemetryView
from skulk.shared.types.worker.instances import InstanceMeta
from skulk.shared.types.worker.shards import Sharding


def test_preparation_completion_makes_build_ready_on_master(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Placement sees the worker-verified build when completion is indexed."""
    node = NodeId("music-node")
    master = object.__new__(Master)
    master.state = State()
    master.state.topology.add_node(node)
    master._telemetry_view = TelemetryView()
    master._runner_loaded_at = {}
    resources = NodeResources(
        backends=frozenset({"audio_cpp", "audio_cpp-cpu"}),
        engine_builds={"audio_cpp-cpu": "verified-build"},
    )
    completion = AudioCppPreparationCompleted(
        request_id=CommandId("prepare-music"),
        target_node=node,
        owner_node=NodeId("api-node"),
        success=True,
        resources=resources,
    )

    master._apply_indexed_event(IndexedEvent(event=completion, idx=0))

    assert master._telemetry_view.node_resources[node] == resources

    # A delayed pre-preparation telemetry packet can arrive before the next
    # command. Its scoped snapshot must still win during that placement.
    master._telemetry_view.node_resources[node] = NodeResources()
    card = ModelCard.model_validate({
        "model_id": ModelId("audio-cpp/test-music"),
        "storage_size": Memory.from_mb(128),
        "source_revision": "a" * 40,
        "gguf_file": "language_model_q4_0.gguf",
        "n_layers": 1, "hidden_size": 1, "supports_tensor": False,
        "tasks": [ModelTask.TextToMusic],
        "music": {
            "family": "minimax_music3", "lyrics": "required",
            "min_seconds": 10, "max_seconds": 60,
            "language_model_gguf": "language_model_q4_0.gguf",
            "rvq_depth_decoder_gguf": "rvq_depth_decoder_q8_0.gguf",
            "flow_transformer_gguf": "transformer_q4_0.gguf",
        },
        "artifact_bundle": {
            "bundle_id": "bundle_" + "a" * 52,
            "files": [
                {"path": name, "size_bytes": 1}
                for name in (
                    "language_model_q4_0.gguf",
                    "rvq_depth_decoder_q8_0.gguf",
                    "transformer_q4_0.gguf",
                    "condition_encoder.gguf",
                    "vocoder.gguf",
                    "config.json",
                    "config/condition_encoder.json",
                    "config/language_model.json",
                    "config/rvq_depth_decoder.json",
                    "config/transformer.json",
                    "config/vocoder.json",
                    "tokenizer/tokenizer.json",
                    "tokenizer/tokenizer_config.json",
                )
            ],
            "download_size": 13,
        },
    })
    master._ordered_placement_model_card = lambda model_id: card
    command = PlaceInstance(
        model_card=card, sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing, min_nodes=1,
        prepared_node_resources={node: resources},
    )

    assert master._placement_resources_for_command(command)[node] == resources
    assert master._telemetry_view.node_resources[node] != resources

    # The prepared host is a placement constraint, not merely a refreshed
    # resource hint. A second ready host must not win the planner's scoring.
    other = NodeId("other-ready-node")
    master.state.topology.add_node(other)
    master._telemetry_view.node_resources[other] = resources
    def accept_card(_command: PlaceInstance) -> None:
        pass

    def empty_memory(**_kwargs: object) -> tuple[dict[object, object], dict[object, object]]:
        return {}, {}

    def empty_downloads() -> dict[object, object]:
        return {}

    def no_context_default() -> None:
        return None

    monkeypatch.setattr(master, "_require_ordered_place_instance_card", accept_card)
    monkeypatch.setattr(master, "_placement_memory_inputs", empty_memory)
    monkeypatch.setattr(master, "_effective_downloads", empty_downloads)
    monkeypatch.setattr(master, "_served_context_default", no_context_default)
    master._model_trust_approvals = set()
    seen: list[object] = []

    def capture_placement(*_args: object, **kwargs: object) -> dict[object, object]:
        seen.append(kwargs["required_nodes"])
        return {}

    monkeypatch.setattr(master_module, "place_instance", capture_placement)
    master._place_requested_instance(command)
    assert seen == [{node}]


def test_preparation_completion_does_not_restore_timed_out_node() -> None:
    """A late completion cannot resurrect a node absent from topology."""
    node = NodeId("departed-node")
    master = object.__new__(Master)
    master.state = State()
    master._telemetry_view = TelemetryView()
    master._runner_loaded_at = {}
    completion = AudioCppPreparationCompleted(
        request_id=CommandId("prepare-music"),
        target_node=node,
        owner_node=NodeId("api-node"),
        success=True,
        resources=NodeResources(
            backends=frozenset({"audio_cpp-cpu"}),
            engine_builds={"audio_cpp-cpu": "verified-build"},
        ),
    )

    master._apply_indexed_event(IndexedEvent(event=completion, idx=0))

    assert node not in master._telemetry_view.node_resources
