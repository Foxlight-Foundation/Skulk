"""Wire and TOML round-trip coverage for NodeResources (#149).

The frozenset fields must survive the gossip path (model_dump(mode="json")
-> array -> model_validate under strict mode) or node_resources never
populates and the planner filter is inert. Unit tests that construct the
model in-process do not exercise this, which is how the original slice
shipped a round-trip bug; these tests lock it.
"""

from skulk.shared.types.profiling import NodeResources


def test_node_resources_survives_json_wire_round_trip() -> None:
    original = NodeResources(
        backends=frozenset({"mlx"}), participation="management"
    )
    restored = NodeResources.model_validate(original.model_dump(mode="json"))
    assert restored == original
    assert restored.backends == frozenset({"mlx"})
    assert restored.participation == "management"


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
