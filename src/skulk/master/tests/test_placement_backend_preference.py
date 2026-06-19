# pyright: reportPrivateUsage=false
"""Tests for backend-preference cycle ranking (separable-engine routing)."""

from skulk.master.placement import _cycle_backend_preference_score
from skulk.shared.types.common import NodeId
from skulk.shared.types.profiling import NodeResources
from skulk.shared.types.topology import Cycle


def _node(*tags: str) -> NodeResources:
    return NodeResources(backends=frozenset(tags), participation="full")


def test_no_preference_is_neutral() -> None:
    cycle = Cycle(node_ids=[NodeId("a")])
    resources = {NodeId("a"): _node("llama_cpp-vulkan")}
    assert _cycle_backend_preference_score(cycle, resources, ()) == 0


def test_top_preference_outranks_fallback() -> None:
    preference = ("llama_cpp-vulkan", "llama_cpp-rocm")
    vulkan_cycle = Cycle(node_ids=[NodeId("v")])
    rocm_cycle = Cycle(node_ids=[NodeId("r")])
    resources = {
        NodeId("v"): _node("llama_cpp", "llama_cpp-vulkan"),
        NodeId("r"): _node("llama_cpp", "llama_cpp-rocm"),
    }
    vulkan_score = _cycle_backend_preference_score(vulkan_cycle, resources, preference)
    rocm_score = _cycle_backend_preference_score(rocm_cycle, resources, preference)
    assert vulkan_score > rocm_score > 0


def test_unsatisfied_preference_scores_zero() -> None:
    # Eligible (compatible_backends filtered upstream) but serves neither
    # preferred tag -> ranks below any cycle that serves one, never excluded.
    preference = ("llama_cpp-vulkan", "llama_cpp-rocm")
    cycle = Cycle(node_ids=[NodeId("c")])
    resources = {NodeId("c"): _node("llama_cpp", "llama_cpp-cpu")}
    assert _cycle_backend_preference_score(cycle, resources, preference) == 0


def test_preference_requires_all_nodes_in_cycle() -> None:
    # A multi-node cycle only earns a preference rank if EVERY node with a
    # resources entry can serve that tag (the whole ring runs one backend).
    preference = ("llama_cpp-vulkan",)
    cycle = Cycle(node_ids=[NodeId("a"), NodeId("b")])
    mixed = {
        NodeId("a"): _node("llama_cpp-vulkan"),
        NodeId("b"): _node("llama_cpp-rocm"),
    }
    assert _cycle_backend_preference_score(cycle, mixed, preference) == 0
    both = {
        NodeId("a"): _node("llama_cpp-vulkan"),
        NodeId("b"): _node("llama_cpp-vulkan"),
    }
    assert _cycle_backend_preference_score(cycle, both, preference) == 1


def test_warming_up_node_without_entry_does_not_block_preference() -> None:
    # A node not yet in node_resources (gossip warming up) is not counted
    # against a preference, matching the optimistic eligibility default.
    preference = ("llama_cpp-vulkan",)
    cycle = Cycle(node_ids=[NodeId("a"), NodeId("missing")])
    resources = {NodeId("a"): _node("llama_cpp-vulkan")}
    assert _cycle_backend_preference_score(cycle, resources, preference) == 1
