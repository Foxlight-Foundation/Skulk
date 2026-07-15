from skulk.shared.models.memory_estimate import (
    GPU_WORKING_SET_FRACTION,
    LLAMA_CPP_MEMORY_OVERHEAD_FACTOR,
    MEMORY_OVERHEAD_FACTOR,
    MEMORY_OVERHEAD_FLOOR,
    estimate_kv_cache_bytes,
    estimate_shard_footprint,
    gpu_working_set_ceiling,
    memory_overhead_factor,
)
from skulk.shared.models.model_cards import ModelCard, ModelId, ModelTask
from skulk.shared.types.memory import Memory


def _card(
    storage_gb: float,
    *,
    kv_heads: int | None = None,
    n_layers: int = 32,
    gguf_file: str | None = None,
) -> ModelCard:
    return ModelCard(
        model_id=ModelId("test-model"),
        storage_size=Memory.from_gb(storage_gb),
        n_layers=n_layers,
        hidden_size=1000,
        supports_tensor=True,
        num_key_value_heads=kv_heads,
        gguf_file=gguf_file,
        tasks=[ModelTask.TextGeneration],
    )


def test_memory_overhead_factor_is_engine_aware():
    """GGUF (llama.cpp C++ runtime) gets the lighter factor; MLX gets 1.30."""
    assert memory_overhead_factor(_card(8)) == MEMORY_OVERHEAD_FACTOR
    assert (
        memory_overhead_factor(_card(8, gguf_file="model.gguf"))
        == LLAMA_CPP_MEMORY_OVERHEAD_FACTOR
    )


def test_shard_footprint_uses_lighter_factor_for_gguf():
    """A GGUF card's footprint uses the 1.10 factor, so it is smaller than the
    same weights under the MLX 1.30 factor."""
    mlx = estimate_shard_footprint(_card(40), 1.0)
    gguf = estimate_shard_footprint(_card(40, gguf_file="model.gguf"), 1.0)
    assert gguf.in_bytes < mlx.in_bytes
    expected = (
        Memory.from_gb(40) * LLAMA_CPP_MEMORY_OVERHEAD_FACTOR + MEMORY_OVERHEAD_FLOOR
    )
    assert gguf.in_bytes == expected.in_bytes


def test_kv_is_zero_without_kv_heads():
    assert estimate_kv_cache_bytes(_card(8), 32, 8192).in_bytes == 0


def test_kv_zero_for_nonpositive_args():
    card = _card(8, kv_heads=8)
    assert estimate_kv_cache_bytes(card, 0, 8192).in_bytes == 0
    assert estimate_kv_cache_bytes(card, 16, 0).in_bytes == 0


def test_kv_scales_linearly_with_layers_and_context():
    card = _card(8, kv_heads=8)
    base = estimate_kv_cache_bytes(card, 16, 8192).in_bytes
    assert base > 0
    assert estimate_kv_cache_bytes(card, 32, 8192).in_bytes == 2 * base
    assert estimate_kv_cache_bytes(card, 16, 16384).in_bytes == 2 * base


def test_shard_footprint_whole_model_is_weights_times_factor_plus_floor():
    card = _card(8)  # no KV
    expected = Memory.from_gb(8) * MEMORY_OVERHEAD_FACTOR + MEMORY_OVERHEAD_FLOOR
    assert estimate_shard_footprint(card, 1.0).in_bytes == expected.in_bytes


def test_shard_footprint_zero_fraction_is_zero():
    assert estimate_shard_footprint(_card(8), 0.0).in_bytes == 0


def test_shard_preallocates_kv_upfront_uses_resolved_backend():
    from skulk.shared.models.memory_estimate import shard_preallocates_kv_upfront
    from skulk.shared.models.model_cards import PlacementCardConfig

    def _shard(card: ModelCard, backend: str | None) -> PipelineShardMetadata:
        s = _pipeline_shard(card, start=0, end=4)
        return s.model_copy(update={"resolved_backend": backend})

    def _backends(*tags: str) -> PlacementCardConfig:
        return PlacementCardConfig(compatible_backends=frozenset(tags))

    hybrid = _card(1).model_copy(
        update={"placement": _backends("mlx", "llama_cpp-cuda")}
    )
    # Resolved backend is the precise signal: a HYBRID card resolving to llama.cpp
    # on a non-MLX node IS preallocating (the reviewer's bypass case); the same
    # card resolving to mlx is NOT (grows KV lazily, no force-clamp).
    assert shard_preallocates_kv_upfront(_shard(hybrid, "llama_cpp-cuda"))
    assert not shard_preallocates_kv_upfront(_shard(hybrid, "mlx-metal"))
    assert shard_preallocates_kv_upfront(_shard(_card(1), "llama_server-cuda"))
    assert not shard_preallocates_kv_upfront(_shard(_card(1), "vllm-cuda"))
    # Fallback when the backend is unresolved (node absent from telemetry): pinned
    # gguf and unpinned llama.cpp-only cards are preallocating; mlx-capable is not.
    assert shard_preallocates_kv_upfront(_shard(_card(1, gguf_file="m.gguf"), None))
    assert shard_preallocates_kv_upfront(
        _shard(_card(1).model_copy(update={"placement": _backends("llama_cpp-cuda")}), None)
    )
    assert not shard_preallocates_kv_upfront(_shard(_card(1), None))  # mlx-only
    # Unresolved fallback is CONSERVATIVE: a hybrid card (mlx + llama.cpp) counts as
    # preallocating when we can't prove it will run MLX (non-Darwin nodes won't
    # advertise MLX and may dispatch it to llama.cpp -> OOM if unclamped).
    assert shard_preallocates_kv_upfront(_shard(hybrid, None))


def test_instance_limit_unpinned_gguf_non_vram_clamps_to_floor():
    # An UNPINNED gguf card (no gguf_file, llama.cpp backends) on a non-VRAM node
    # is still clamped to the floor -- the safety must not be bypassable by leaving
    # the gguf pin unset. (P2 review, #585.)
    from skulk.shared.models.model_cards import PlacementCardConfig

    card = _card(1, kv_heads=8, n_layers=32).model_copy(
        update={
            "context_length": 131072,
            "placement": PlacementCardConfig(
                compatible_backends=frozenset({"llama_cpp-cuda"})
            ),
        }
    )
    assignments = _assignments(
        card, {"r0": (_pipeline_shard(card, start=0, end=32), "n0")}
    )
    assert (
        instance_context_token_limit(assignments, {NodeId("n0"): Memory.from_gb(64)})
        == KV_CONTEXT_BUDGET_TOKENS
    )


def test_shard_footprint_scales_with_context_budget():
    # The worker's pre-load fit guard now sizes KV to the stamped served window
    # (context_token_limit) for gguf instead of a fixed 8192, so a larger window
    # yields a larger footprint -- the guard refuses if live memory cannot hold the
    # window the runner will actually preallocate up front. (P1 review, #585.)
    card = _card(4, kv_heads=8, n_layers=32, gguf_file="m.gguf")
    small = estimate_shard_footprint(card, 1.0, context_budget=8192)
    large = estimate_shard_footprint(card, 1.0, context_budget=32768)
    assert large.in_bytes > small.in_bytes


def test_shard_footprint_fraction_scales_weights_and_kv_not_floor():
    card = _card(8, kv_heads=8)
    half = estimate_shard_footprint(card, 0.5)
    whole = estimate_shard_footprint(card, 1.0)
    assert half.in_bytes < whole.in_bytes
    # Weights and KV both scale with the fraction; only the flat floor does not,
    # so (footprint - floor) is linear in the fraction.
    floor = MEMORY_OVERHEAD_FLOOR.in_bytes
    assert abs((half.in_bytes - floor) * 2 - (whole.in_bytes - floor)) <= 4


def test_gpu_working_set_ceiling_is_fraction_of_total():
    expected = Memory.from_gb(16) * GPU_WORKING_SET_FRACTION
    assert gpu_working_set_ceiling(Memory.from_gb(16)).in_bytes == expected.in_bytes


# --- Context-admission ceiling (#145) ---------------------------------------

from skulk.shared.models.memory_estimate import (  # noqa: E402
    KV_CONTEXT_BUDGET_TOKENS,
    KV_DTYPE_BYTES,
    KV_HEAD_DIM_FALLBACK,
    instance_context_token_limit,
    per_token_kv_bytes,
    shard_fraction_of_model,
)
from skulk.shared.types.common import NodeId  # noqa: E402
from skulk.shared.types.worker.runners import RunnerId, ShardAssignments  # noqa: E402
from skulk.shared.types.worker.shards import (  # noqa: E402
    PipelineShardMetadata,
    TensorShardMetadata,
)


def _assignments(
    card: ModelCard,
    shards: dict[str, tuple[PipelineShardMetadata | TensorShardMetadata, str]],
) -> ShardAssignments:
    """Build ShardAssignments from {runner_name: (shard, node_name)}."""
    return ShardAssignments(
        model_id=card.model_id,
        runner_to_shard={
            RunnerId(runner): shard for runner, (shard, _node) in shards.items()
        },
        node_to_runner={
            NodeId(node): RunnerId(runner) for runner, (_shard, node) in shards.items()
        },
    )


def _pipeline_shard(
    card: ModelCard, *, start: int, end: int, rank: int = 0, world: int = 1
) -> PipelineShardMetadata:
    return PipelineShardMetadata(
        model_card=card,
        device_rank=rank,
        world_size=world,
        start_layer=start,
        end_layer=end,
        n_layers=card.n_layers,
    )


def test_per_token_kv_bytes_formula():
    card = _card(4, kv_heads=8, n_layers=32)
    assert (
        per_token_kv_bytes(card) == 2 * 32 * 8 * KV_HEAD_DIM_FALLBACK * KV_DTYPE_BYTES
    )


def test_per_token_kv_bytes_zero_without_kv_heads():
    assert per_token_kv_bytes(_card(4)) == 0


def test_shard_fraction_tensor_is_inverse_world_size():
    card = _card(4, kv_heads=8)
    shard = TensorShardMetadata(
        model_card=card,
        device_rank=0,
        world_size=2,
        start_layer=0,
        end_layer=32,
        n_layers=32,
    )
    assert shard_fraction_of_model(shard) == 0.5


def test_shard_fraction_pipeline_is_layer_share():
    card = _card(4, kv_heads=8, n_layers=32)
    assert shard_fraction_of_model(_pipeline_shard(card, start=0, end=8)) == 0.25


def test_instance_limit_single_node_matches_independent_arithmetic():
    card = _card(4, kv_heads=8, n_layers=32)
    assignments = _assignments(
        card, {"r0": (_pipeline_shard(card, start=0, end=32), "n0")}
    )
    limit = instance_context_token_limit(
        assignments, {NodeId("n0"): Memory.from_gb(16)}
    )
    gib = 1024**3
    budget = (
        round(16 * gib * GPU_WORKING_SET_FRACTION)
        - round(4 * gib * MEMORY_OVERHEAD_FACTOR)
        - MEMORY_OVERHEAD_FLOOR.in_bytes
    )
    assert limit == int(budget / (2 * 32 * 8 * KV_HEAD_DIM_FALLBACK * KV_DTYPE_BYTES))


def test_instance_limit_is_min_across_nodes():
    card = _card(4, kv_heads=8, n_layers=32)
    half_a = _pipeline_shard(card, start=0, end=16, rank=0, world=2)
    half_b = _pipeline_shard(card, start=16, end=32, rank=1, world=2)
    assignments = _assignments(card, {"r0": (half_a, "big"), "r1": (half_b, "small")})
    both_big = instance_context_token_limit(
        assignments,
        {NodeId("big"): Memory.from_gb(32), NodeId("small"): Memory.from_gb(32)},
    )
    constrained = instance_context_token_limit(
        assignments,
        {NodeId("big"): Memory.from_gb(32), NodeId("small"): Memory.from_gb(8)},
    )
    assert both_big is not None and constrained is not None
    assert constrained < both_big


def test_instance_limit_capped_by_card_context_length():
    card = _card(1, kv_heads=8, n_layers=32).model_copy(update={"context_length": 1000})
    assignments = _assignments(
        card, {"r0": (_pipeline_shard(card, start=0, end=32), "n0")}
    )
    assert (
        instance_context_token_limit(assignments, {NodeId("n0"): Memory.from_gb(64)})
        == 1000
    )


def test_instance_limit_missing_node_memory_falls_back_to_card():
    card = _card(4, kv_heads=8, n_layers=32).model_copy(update={"context_length": 4096})
    assignments = _assignments(
        card, {"r0": (_pipeline_shard(card, start=0, end=32), "n0")}
    )
    assert instance_context_token_limit(assignments, {}) == 4096


def test_instance_limit_none_when_nothing_enforceable():
    card = _card(4, kv_heads=8, n_layers=32)  # context_length defaults to 0
    assignments = _assignments(
        card, {"r0": (_pipeline_shard(card, start=0, end=32), "n0")}
    )
    assert instance_context_token_limit(assignments, {}) is None


def test_instance_limit_gguf_not_capped_to_kv_budget_on_vram_node():
    # The old fixed 8192 clamp on GGUF/llama.cpp instances is gone on a discrete-
    # VRAM node: the model is sized to the memory/card fit (the runner serves that
    # window via serving_n_ctx, and placement admits the node against the same VRAM
    # the ceiling is derived from). Generous VRAM + a large advertised context ->
    # well above the old budget, capped only by the card max. A fixed 8192 made
    # served models unusable for real-context work.
    card = _card(1, kv_heads=8, n_layers=32, gguf_file="m-Q4_K_M.gguf").model_copy(
        update={"context_length": 131072}
    )
    # The shard must resolve to a GPU-offload backend for the lift: only then does
    # the runner preallocate KV in VRAM (a -cpu/bare backend runs -ngl 0 in RAM).
    gpu_shard = _pipeline_shard(card, start=0, end=32).model_copy(
        update={"resolved_backend": "llama_server-cuda"}
    )
    assignments = _assignments(card, {"r0": (gpu_shard, "n0")})
    limit = instance_context_token_limit(
        assignments,
        {NodeId("n0"): Memory.from_gb(64)},
        node_vram={NodeId("n0"): Memory.from_gb(48)},
    )
    assert limit is not None and limit > KV_CONTEXT_BUDGET_TOKENS
    assert limit <= 131072  # never above the card's advertised max


def test_instance_limit_gguf_cpu_resolved_on_vram_node_clamps_to_floor():
    # A gguf shard that resolves to a CPU/bare backend on a node WITH discrete VRAM
    # runs -ngl 0 and preallocates KV in SYSTEM RAM, so the VRAM-sized lift is
    # unsafe -- it clamps to the floor even though the node reports VRAM. (P2
    # review, #585.)
    card = _card(1, kv_heads=8, n_layers=32, gguf_file="m-Q4_K_M.gguf").model_copy(
        update={"context_length": 131072}
    )
    cpu_shard = _pipeline_shard(card, start=0, end=32).model_copy(
        update={"resolved_backend": "llama_server-cpu"}
    )
    assignments = _assignments(card, {"r0": (cpu_shard, "n0")})
    limit = instance_context_token_limit(
        assignments,
        {NodeId("n0"): Memory.from_gb(64)},
        node_vram={NodeId("n0"): Memory.from_gb(48)},
    )
    assert limit == KV_CONTEXT_BUDGET_TOKENS


def test_instance_limit_gguf_non_vram_node_clamps_to_floor():
    # The gguf lift applies only to discrete-VRAM nodes. On a node WITHOUT discrete
    # VRAM the fit is derived from static ram_total, but llama.cpp preallocates the
    # window up front against live ram_available (which placement admits on and can
    # be far lower under memory pressure) -- so preallocating a ram_total-sized
    # window could OOM. gguf on a non-VRAM node stays at the budget floor. (P1
    # review, #585.)
    card = _card(1, kv_heads=8, n_layers=32, gguf_file="m-Q4_K_M.gguf").model_copy(
        update={"context_length": 131072}
    )
    assignments = _assignments(
        card, {"r0": (_pipeline_shard(card, start=0, end=32), "n0")}
    )
    # Plenty of system RAM but NO discrete VRAM reported for the node.
    assert (
        instance_context_token_limit(assignments, {NodeId("n0"): Memory.from_gb(64)})
        == KV_CONTEXT_BUDGET_TOKENS
    )


def test_instance_limit_mlx_non_vram_node_keeps_memory_fit():
    # The non-VRAM clamp is gguf-only: MLX grows KV lazily (no up-front
    # preallocation), so an MLX card on a non-VRAM node keeps the full memory fit.
    card = _card(1, kv_heads=8, n_layers=32).model_copy(
        update={"context_length": 131072}
    )
    assignments = _assignments(
        card, {"r0": (_pipeline_shard(card, start=0, end=32), "n0")}
    )
    limit = instance_context_token_limit(
        assignments, {NodeId("n0"): Memory.from_gb(64)}
    )
    assert limit is not None and limit > KV_CONTEXT_BUDGET_TOKENS


def test_instance_limit_gguf_no_info_clamps_to_floor():
    # A bare GGUF card with neither node memory nor an advertised context has an
    # uncomputable memory fit. llama.cpp preallocates KV up front, so the stamp
    # must be a SAFE preallocation size: it clamps to the KV_CONTEXT_BUDGET_TOKENS
    # floor rather than None (which the runner would also floor) or the card max.
    card = _card(4, kv_heads=8, n_layers=32, gguf_file="m-Q4_K_M.gguf")
    assignments = _assignments(
        card, {"r0": (_pipeline_shard(card, start=0, end=32), "n0")}
    )
    assert instance_context_token_limit(assignments, {}) == KV_CONTEXT_BUDGET_TOKENS


def test_instance_limit_gguf_uncomputable_kv_cost_clamps_to_floor():
    # A GGUF card with no num_key_value_heads => per_token_kv_bytes is 0 => the
    # memory fit is uncomputable (memory_limit None) even with plenty of node
    # memory and a large advertised context. Without the clamp the stamp would be
    # the card's 128k max and llama.cpp would preallocate a fictitious window and
    # OOM on load; instead it clamps to the budget floor. (P1 review catch, #585.)
    card = _card(1, n_layers=32, gguf_file="m-Q4_K_M.gguf").model_copy(
        update={"context_length": 131072}
    )
    assignments = _assignments(
        card, {"r0": (_pipeline_shard(card, start=0, end=32), "n0")}
    )
    assert (
        instance_context_token_limit(assignments, {NodeId("n0"): Memory.from_gb(64)})
        == KV_CONTEXT_BUDGET_TOKENS
    )


def test_instance_limit_non_gguf_uncomputable_kv_cost_keeps_card_limit():
    # The uncomputable-fit clamp is gguf-only: MLX grows KV lazily (no up-front
    # preallocation), so a non-gguf card with unknown KV cost keeps its advertised
    # max rather than being floored.
    card = _card(1, n_layers=32).model_copy(update={"context_length": 131072})
    assignments = _assignments(
        card, {"r0": (_pipeline_shard(card, start=0, end=32), "n0")}
    )
    assert (
        instance_context_token_limit(assignments, {NodeId("n0"): Memory.from_gb(64)})
        == 131072
    )


def test_instance_limit_gguf_below_budget_unchanged():
    # A GGUF card whose advertised context is already below the budget keeps that
    # smaller limit (the cap is a ceiling, never raises a smaller real limit).
    card = _card(1, kv_heads=8, n_layers=32, gguf_file="m-Q4_K_M.gguf").model_copy(
        update={"context_length": 2048}
    )
    assignments = _assignments(
        card, {"r0": (_pipeline_shard(card, start=0, end=32), "n0")}
    )
    assert (
        instance_context_token_limit(assignments, {NodeId("n0"): Memory.from_gb(64)})
        == 2048
    )


def test_instance_limit_mlx_not_capped_to_kv_budget():
    # The cap is GGUF-only: an MLX card (no gguf_file) keeps its memory/card
    # ceiling, which grows the KV cache per request rather than allocating up front.
    card = _card(1, kv_heads=8, n_layers=32).model_copy(
        update={"context_length": 131072}
    )
    assignments = _assignments(
        card, {"r0": (_pipeline_shard(card, start=0, end=32), "n0")}
    )
    limit = instance_context_token_limit(
        assignments, {NodeId("n0"): Memory.from_gb(64)}
    )
    assert limit is not None and limit > KV_CONTEXT_BUDGET_TOKENS


def test_instance_limit_unknown_kv_cost_falls_back_to_card():
    card = _card(4, n_layers=32).model_copy(update={"context_length": 2048})
    assignments = _assignments(
        card, {"r0": (_pipeline_shard(card, start=0, end=32), "n0")}
    )
    assert (
        instance_context_token_limit(assignments, {NodeId("n0"): Memory.from_gb(16)})
        == 2048
    )


def test_instance_limit_zero_when_weights_leave_no_kv_room():
    card = _card(64, kv_heads=8, n_layers=32)
    assignments = _assignments(
        card, {"r0": (_pipeline_shard(card, start=0, end=32), "n0")}
    )
    assert (
        instance_context_token_limit(assignments, {NodeId("n0"): Memory.from_gb(16)})
        == 0
    )


def test_placement_card_config_round_trips_through_toml_and_json() -> None:
    """frozenset compatible_backends must survive TOML save and JSON wire;
    without coercion+list serialization, explicit [placement] cards are
    unloadable and ModelCard.save() crashes (#149)."""
    import tomlkit

    from skulk.shared.models.model_cards import PlacementCardConfig

    # TOML provides a list; strict mode must accept it.
    cfg = PlacementCardConfig.model_validate({"compatible_backends": ["mlx"]})
    assert cfg.compatible_backends == frozenset({"mlx"})

    # model_dump must emit a list tomlkit can encode (ModelCard.save path).
    dumped = cfg.model_dump(exclude_none=True)
    assert dumped["compatible_backends"] == ["mlx"]
    tomlkit.dumps(dumped)  # pyright: ignore[reportUnknownMemberType]  # no raise

    # JSON wire round-trip preserves the value.
    restored = PlacementCardConfig.model_validate(cfg.model_dump(mode="json"))
    assert restored.compatible_backends == frozenset({"mlx"})


def test_backend_preference_round_trips_and_preserves_order() -> None:
    """backend_preference is an ORDERED tuple: list input must coerce, order
    must survive TOML/JSON round-trips, and default is empty (no preference)."""
    import tomlkit

    from skulk.shared.models.model_cards import PlacementCardConfig

    assert PlacementCardConfig().backend_preference == ()

    cfg = PlacementCardConfig.model_validate(
        {"backend_preference": ["llama_cpp-vulkan", "llama_cpp-rocm"]}
    )
    assert cfg.backend_preference == ("llama_cpp-vulkan", "llama_cpp-rocm")

    dumped = cfg.model_dump(exclude_none=True)
    assert dumped["backend_preference"] == ["llama_cpp-vulkan", "llama_cpp-rocm"]
    tomlkit.dumps(dumped)  # pyright: ignore[reportUnknownMemberType]  # no raise

    restored = PlacementCardConfig.model_validate(cfg.model_dump(mode="json"))
    assert restored.backend_preference == ("llama_cpp-vulkan", "llama_cpp-rocm")
