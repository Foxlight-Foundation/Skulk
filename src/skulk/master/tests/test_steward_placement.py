"""System-role (steward) placement stamping and repair-survival tests."""

from skulk.master.placement import (
    fallback_command_for_refused_instance,
    place_instance,
    replacement_command_for_download_failed_instance,
    replacement_command_for_refused_instance,
)
from skulk.master.tests.test_placement import fully_connected_three_nodes
from skulk.shared.models.model_cards import (
    ModelCard,
    ModelId,
    ModelTask,
    PlacementCardConfig,
)
from skulk.shared.types.commands import PlaceInstance
from skulk.shared.types.memory import Memory
from skulk.shared.types.profiling import NodeResources
from skulk.shared.types.worker.instances import InstanceMeta, MlxRingInstance
from skulk.shared.types.worker.shards import Sharding


def _steward_card() -> ModelCard:
    return ModelCard(
        model_id=ModelId("steward-brain-model"),
        storage_size=Memory.from_gb(3),
        n_layers=12,
        hidden_size=30,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
    )


def _place_steward() -> MlxRingInstance:
    topology, node_memory, node_network, _node_ids = fully_connected_three_nodes(
        (10.0, 10.0, 10.0)
    )
    command = PlaceInstance(
        model_card=_steward_card(),
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        min_nodes=1,
        system_role="steward",
    )
    placed = place_instance(command, topology, {}, node_memory, node_network)
    instance = next(iter(placed.values()))
    assert isinstance(instance, MlxRingInstance)
    return instance


def test_place_instance_stamps_system_role() -> None:
    """The minted instance carries the command's system-role marker."""
    instance = _place_steward()
    assert instance.system_role == "steward"


def test_default_placement_has_no_system_role() -> None:
    """Ordinary placements stay unmarked (and old event logs replay as None)."""
    topology, node_memory, node_network, _node_ids = fully_connected_three_nodes(
        (10.0, 10.0, 10.0)
    )
    command = PlaceInstance(
        model_card=_steward_card(),
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        min_nodes=1,
    )
    placed = place_instance(command, topology, {}, node_memory, node_network)
    instance = next(iter(placed.values()))
    assert instance.system_role is None


def test_repair_commands_preserve_system_role() -> None:
    """Every repair builder re-stamps the steward flag (failover survival).

    Repair reconstructs placement intent from the instance's shards, which
    have no channel for the flag; without explicit re-stamping a repaired
    steward would silently demote to an ordinary instance and the invariant
    pass would then place a second steward.
    """
    instance = _place_steward()

    wider = replacement_command_for_refused_instance(instance)
    assert wider.system_role == "steward"

    refusing = next(iter(instance.shard_assignments.node_to_runner))
    fallback = fallback_command_for_refused_instance(instance, refusing)
    assert fallback.system_role == "steward"

    failed = replacement_command_for_download_failed_instance(instance, {refusing})
    assert failed.system_role == "steward"


def test_repair_commands_keep_none_for_ordinary_instances() -> None:
    """Ordinary instances never acquire a system role through repair."""
    topology, node_memory, node_network, _node_ids = fully_connected_three_nodes(
        (10.0, 10.0, 10.0)
    )
    command = PlaceInstance(
        model_card=_steward_card(),
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        min_nodes=1,
    )
    placed = place_instance(command, topology, {}, node_memory, node_network)
    instance = next(iter(placed.values()))
    assert replacement_command_for_refused_instance(instance).system_role is None


def _gguf_steward_card() -> ModelCard:
    """A card shaped like the GGUF steward brains: served lanes, no mlx."""
    return ModelCard(
        model_id=ModelId("org/steward-brain-GGUF"),
        storage_size=Memory.from_gb(3),
        n_layers=12,
        hidden_size=30,
        num_key_value_heads=2,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        family="qwen",
        quantization="Q4_K_M",
        gguf_file="steward-brain-Q4_K_M.gguf",
        context_length=262144,
        placement=PlacementCardConfig(
            compatible_backends=frozenset(
                {"llama_server-cuda", "llama_cpp-cuda"}
            ),
            backend_preference=("llama_server-cuda", "llama_cpp-cuda"),
        ),
    )


def test_mlx_ring_meta_is_benign_for_a_single_node_gguf_steward() -> None:
    """The invariant's hardcoded InstanceMeta.MlxRing must not break GGUF.

    ``_maintain_steward_placement`` always asks for ``InstanceMeta.MlxRing``,
    which is the same default every dashboard placement sends. For a
    single-node cycle the planner rewrites the request to Pipeline/Ring
    regardless of engine, and a multi-node GGUF cycle resolves to the RPC
    shape on its own, so the meta never has to describe the engine. This
    test pins that so a future planner change cannot make the invariant
    place the GGUF brain wrongly (or refuse to place it) unnoticed.
    """
    topology, node_memory, node_network, node_ids = fully_connected_three_nodes(
        (10.0, 10.0, 10.0)
    )
    command = PlaceInstance(
        model_card=_gguf_steward_card(),
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        min_nodes=1,
        system_role="steward",
    )
    placed = place_instance(
        command,
        topology,
        {},
        node_memory,
        node_network,
        node_resources={
            node_id: NodeResources(backends=frozenset({"llama_server-cuda"}))
            for node_id in node_ids
        },
    )
    instance = next(iter(placed.values()))
    assert isinstance(instance, MlxRingInstance)
    assert instance.system_role == "steward"
    assert len(instance.shard_assignments.node_to_runner) == 1


def test_steward_candidates_require_text_and_tool_capability() -> None:
    """Only tool-calling text cards may serve as steward brains (#830).

    The harness always dispatches ``TextGeneration`` with server tools, so
    an operator override naming any other card shape must be skipped, not
    placed.
    """
    from skulk.master.main import steward_candidate_is_servable
    from skulk.shared.models.model_cards import ToolingCardConfig

    def _card(tasks: list[ModelTask], *, tools: bool) -> ModelCard:
        return ModelCard(
            model_id=ModelId("candidate-model"),
            storage_size=Memory.from_gb(3),
            n_layers=12,
            hidden_size=30,
            supports_tensor=True,
            tasks=tasks,
            tooling=ToolingCardConfig(supports_tool_calling=True) if tools else None,
        )

    assert steward_candidate_is_servable(
        _card([ModelTask.TextGeneration], tools=True)
    )
    # A speech card cannot serve steward turns no matter what it declares.
    assert not steward_candidate_is_servable(
        _card([ModelTask.TextToSpeech], tools=True)
    )
    # A text card without tool calling can never investigate.
    assert not steward_candidate_is_servable(
        _card([ModelTask.TextGeneration], tools=False)
    )


def test_vllm_only_steward_candidates_require_a_pinned_parser() -> None:
    """Backend truth joins model truth in the servability gate (#833).

    A tool-declaring card whose only platform-servable engine is vllm still
    fails every steward turn unless the card pins
    ``runtime.vllm_tool_call_parser``: the launched server rejects
    tools-bearing requests loudly, so such a candidate must be skipped.
    """
    from skulk.master.main import steward_candidate_is_servable
    from skulk.shared.models.model_cards import (
        RuntimeCapabilityCardConfig,
        ToolingCardConfig,
    )

    def _vllm_card(parser: str | None) -> ModelCard:
        return ModelCard(
            model_id=ModelId("vllm-only-model"),
            storage_size=Memory.from_gb(3),
            n_layers=12,
            hidden_size=30,
            supports_tensor=True,
            tasks=[ModelTask.TextGeneration],
            tooling=ToolingCardConfig(supports_tool_calling=True),
            runtime=(
                RuntimeCapabilityCardConfig(vllm_tool_call_parser=parser)
                if parser is not None
                else None
            ),
            placement=PlacementCardConfig(
                compatible_backends=frozenset({"vllm-cuda"}),
            ),
        )

    assert not steward_candidate_is_servable(_vllm_card(None))
    assert steward_candidate_is_servable(_vllm_card("hermes"))


def test_stamped_vllm_placement_without_parser_is_rejected() -> None:
    """The post-placement gate catches multi-engine cards that resolve to vllm.

    A card listing vllm alongside another engine passes the card-level
    servability gate, but a fleet whose hardware makes placement stamp
    vllm still needs the parser pin; the walk must skip the candidate when
    the minted shards resolved to vllm without one (#833).
    """
    from skulk.master.main import placement_may_select_parserless_vllm
    from skulk.shared.models.model_cards import (
        RuntimeCapabilityCardConfig,
        ToolingCardConfig,
    )
    from skulk.shared.types.worker.instances import Instance, InstanceId

    def _card(parser: str | None) -> ModelCard:
        return ModelCard(
            model_id=ModelId("multi-engine-model"),
            storage_size=Memory.from_gb(3),
            n_layers=12,
            hidden_size=30,
            supports_tensor=True,
            tasks=[ModelTask.TextGeneration],
            tooling=ToolingCardConfig(supports_tool_calling=True),
            runtime=(
                RuntimeCapabilityCardConfig(vllm_tool_call_parser=parser)
                if parser is not None
                else None
            ),
            placement=PlacementCardConfig(
                compatible_backends=frozenset({"vllm-cuda", "llama_server-cuda"}),
                backend_preference=("vllm-cuda", "llama_server-cuda"),
            ),
        )

    topology, node_memory, node_network, node_ids = fully_connected_three_nodes(
        (10.0, 10.0, 10.0)
    )

    def _place(card: ModelCard) -> "dict[InstanceId, Instance]":
        command = PlaceInstance(
            model_card=card,
            sharding=Sharding.Pipeline,
            instance_meta=InstanceMeta.MlxRing,
            min_nodes=1,
            system_role="steward",
        )
        return place_instance(
            command,
            topology,
            {},
            node_memory,
            node_network,
            node_resources={
                node_id: NodeResources(
                    backends=frozenset({"vllm-cuda", "llama_server-cuda"})
                )
                for node_id in node_ids
            },
        )

    parserless = _place(_card(None))
    stamped = {
        shard.resolved_backend
        for instance in parserless.values()
        for shard in instance.shard_assignments.runner_to_shard.values()
    }
    assert stamped == {"vllm-cuda"}, "test premise: placement resolves vllm"
    assert placement_may_select_parserless_vllm(_card(None), parserless)
    assert not placement_may_select_parserless_vllm(
        _card("hermes"), _place(_card("hermes"))
    )

    # Telemetry warm-up: no NodeResources means placement stamps no
    # backend, and the worker's local fallback could still pick vllm, so
    # an unstamped parserless placement must read as unsafe too.
    unstamped_command = PlaceInstance(
        model_card=_card(None),
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        min_nodes=1,
        system_role="steward",
    )
    unstamped = place_instance(
        unstamped_command, topology, {}, node_memory, node_network
    )
    stamped_backends = {
        shard.resolved_backend
        for instance in unstamped.values()
        for shard in instance.shard_assignments.runner_to_shard.values()
    }
    assert stamped_backends == {None}, "test premise: warm-up leaves no stamp"
    assert placement_may_select_parserless_vllm(_card(None), unstamped)
