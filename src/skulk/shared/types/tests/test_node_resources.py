"""Wire and TOML round-trip coverage for NodeResources (#149).

The frozenset fields must survive the gossip path (model_dump(mode="json")
-> array -> model_validate under strict mode) or node_resources never
populates and the planner filter is inert. Unit tests that construct the
model in-process do not exercise this, which is how the original slice
shipped a round-trip bug; these tests lock it.
"""

from skulk.shared.types.node_facts import CapabilityConflict
from skulk.shared.types.profiling import NodeResources


def test_node_resources_survives_json_wire_round_trip() -> None:
    original = NodeResources(
        backends=frozenset({"mlx"}),
        participation="management",
        data_transport="zenoh",
    )
    restored = NodeResources.model_validate(original.model_dump(mode="json"))
    assert restored == original
    assert restored.backends == frozenset({"mlx"})
    assert restored.participation == "management"
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
    assert nr.data_transport == "gossipsub"


async def test_node_resources_gather_uses_resolved_data_transport() -> None:
    resources = await NodeResources.gather(data_transport="zenoh")
    assert resources.data_transport == "zenoh"


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
