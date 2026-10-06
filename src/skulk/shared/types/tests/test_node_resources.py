"""Wire and TOML round-trip coverage for NodeResources (#149).

The frozenset fields must survive the gossip path (model_dump(mode="json")
-> array -> model_validate under strict mode) or node_resources never
populates and the planner filter is inert. Unit tests that construct the
model in-process do not exercise this, which is how the original slice
shipped a round-trip bug; these tests lock it.
"""

import threading

import pytest

from skulk.shared.types.node_facts import CapabilityConflict
from skulk.shared.types.profiling import NodeResources


def test_node_resources_survives_json_wire_round_trip() -> None:
    original = NodeResources(
        backends=frozenset({"mlx"}),
        engine_builds={"mlx": "mlx@0.29.1"},
        hardware_classes=frozenset({"apple", "apple:m4-max"}),
        participation="management",
        api_available=False,
        data_transport="zenoh",
    )
    restored = NodeResources.model_validate(original.model_dump(mode="json"))
    assert restored == original
    assert restored.backends == frozenset({"mlx"})
    assert restored.engine_builds == {"mlx": "mlx@0.29.1"}
    assert restored.hardware_classes == frozenset({"apple", "apple:m4-max"})
    assert restored.participation == "management"
    assert restored.api_available is False
    assert restored.data_transport == "zenoh"


def test_node_resources_coerces_list_backends() -> None:
    # A JSON array (how the wire and any list-shaped input arrive).
    restored = NodeResources.model_validate(
        {"backends": ["mlx", "llama_cpp"], "participation": "full"}
    )
    assert restored.backends == frozenset({"mlx", "llama_cpp"})


def test_node_resources_defaults_are_full_mlx() -> None:
    nr = NodeResources()
    assert nr.backends == frozenset({"mlx"})
    assert nr.participation == "full"
    assert nr.api_available is True
    assert nr.data_transport == "gossipsub"


async def test_node_resources_gather_uses_resolved_data_transport() -> None:
    resources = await NodeResources.gather(
        api_available=False,
        data_transport="zenoh",
    )
    assert resources.api_available is False
    assert resources.data_transport == "zenoh"


async def test_node_resources_gather_probes_engine_builds_off_the_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The engine-build inventory runs blocking subprocesses in a worker thread.

    A hung ComfyUI interpreter would otherwise hold the worker loop for its
    whole timeout, stalling heartbeats past the fleet's pruning window.
    """
    import skulk.facts as facts_module

    loop_thread = threading.current_thread()
    probe_threads: list[threading.Thread] = []

    def recording_inventory(
        backends: frozenset[str], facts: object, **_kwargs: object
    ) -> dict[str, str]:
        probe_threads.append(threading.current_thread())
        return {"comfy": "comfy@" + "a" * 40 + "/torch@2.9.1+cu130"}

    monkeypatch.setattr(facts_module, "engine_build_inventory", recording_inventory)

    resources = await NodeResources.gather(api_available=False, data_transport="zenoh")

    assert resources.engine_builds == {"comfy": "comfy@" + "a" * 40 + "/torch@2.9.1+cu130"}
    assert probe_threads and all(thread is not loop_thread for thread in probe_threads)


def test_node_resources_capability_conflicts_survive_wire_round_trip() -> None:
    # Conflicts from backend derivation (#614) ride the same gossip path; the
    # tuple-of-models field must survive json dump -> validate under strict mode.
    original = NodeResources(
        backends=frozenset({"llama_server", "llama_server-cpu"}),
        capability_conflicts=(
            CapabilityConflict(
                code="gpu_serving_disabled",
                message="A GPU is visible but serving resolved cpu-only.",
                remediation="Configure a GPU engine and restart skulk.",
            ),
        ),
    )
    restored = NodeResources.model_validate(original.model_dump(mode="json"))
    assert restored == original
    assert restored.capability_conflicts[0].code == "gpu_serving_disabled"


def test_node_resources_default_has_no_conflicts() -> None:
    assert NodeResources().capability_conflicts == ()
