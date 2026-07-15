from enum import Enum

from pydantic import model_validator

from skulk.shared.models.memory_estimate import (
    KV_CONTEXT_BUDGET_TOKENS,
    shard_preallocates_kv_upfront,
)
from skulk.shared.models.model_cards import ModelTask
from skulk.shared.types.common import Host, Id, NodeId
from skulk.shared.types.worker.runners import RunnerId, ShardAssignments, ShardMetadata
from skulk.utils.pydantic_ext import CamelCaseModel, TaggedModel


class InstanceId(Id):
    pass


class InstanceMeta(str, Enum):
    MlxRing = "MlxRing"
    MlxJaccl = "MlxJaccl"
    # Multi-node llama.cpp via the served engine's RPC backend (#328): one
    # driver runs llama-server --rpc, donors run ggml-rpc-server. Requesting it
    # explicitly is optional -- placement resolves a multi-node GGUF cycle to
    # this shape automatically when llama_server is the cycle's only common
    # multi-node engine.
    LlamaRpc = "LlamaRpc"


class BaseInstance(TaggedModel):
    instance_id: InstanceId
    shard_assignments: ShardAssignments
    # Context-admission ceiling (#145/#279 slice 2): the master computes this
    # ONCE at placement time and stamps it here, in the event-sourced placement
    # decision. Every rank then reads the identical value off replicated state
    # instead of recomputing from per-node memory — required now that node
    # memory lives in the (last-write-wins, unordered) telemetry plane rather
    # than the ordered event log, where divergent per-rank ceilings would
    # deadlock the collectives. ``None`` means no enforceable ceiling.
    context_token_limit: int | None = None

    def shard(self, runner_id: RunnerId) -> ShardMetadata | None:
        return self.shard_assignments.runner_to_shard.get(runner_id, None)

    @model_validator(mode="after")
    def _backfill_context_token_limit(self) -> "BaseInstance":
        """Give legacy/hydrated instances a context-admission ceiling.

        Instances created before #279 slice 2 (replayed from an old event log
        or snapshot, or accepted via an older ``CreateInstance`` payload) carry
        ``context_token_limit=None``, which both the API pre-flight and the
        runner treat as "no ceiling" — so an upgraded cluster would serve those
        existing placements without the #145 admission guard, letting oversized
        requests reach MLX and OOM instead of getting a clean
        ``context_length_exceeded``. Fall back to the card's advertised
        ``context_length`` (static, deterministic across ranks). Freshly placed
        instances are already stamped by the master (placement's
        ``instance_context_token_limit`` itself returns the card limit when no
        memory-derived ceiling applies), so this only fills the legacy gap.

        For a window-committing served engine (llama.cpp / llama-server / vLLM, per
        ``shard_preallocates_kv_upfront``) the fallback is bounded to
        ``KV_CONTEXT_BUDGET_TOKENS``: the engine commits its load-time context
        window from this value (``serving_n_ctx``), so backfilling a large
        advertised context (e.g. 128k) would commit a fictitious window and OOM /
        fail to load. The budget floor is the safe legacy default; a freshly placed
        instance gets the real memory-fit ceiling instead. MLX (lazy KV) keeps the
        full advertised context.
        """
        if self.context_token_limit is None:
            # All shards of an instance share one model card, so any shard's
            # context_length is authoritative; take the first.
            shard = next(iter(self.shard_assignments.runner_to_shard.values()), None)
            if shard is not None:
                card = shard.model_card
                if shard_preallocates_kv_upfront(shard):
                    # A legacy served-engine instance ALWAYS needs a ceiling, even
                    # when the card's context_length is unknown (0): the engine still
                    # commits a window at KV_CONTEXT_BUDGET_TOKENS (serving_n_ctx
                    # falls back to the floor for a None ceiling), so leaving
                    # context_token_limit=None would let the API admit requests
                    # larger than the runner serves. Fall back to the floor (a known
                    # context_length is additionally bounded by it).
                    self.context_token_limit = (
                        min(card.context_length, KV_CONTEXT_BUDGET_TOKENS)
                        if card.context_length > 0
                        else KV_CONTEXT_BUDGET_TOKENS
                    )
                elif card.context_length > 0:
                    self.context_token_limit = card.context_length
        return self


class MlxRingInstance(BaseInstance):
    hosts_by_node: dict[NodeId, list[Host]]
    ephemeral_port: int


class MlxJacclInstance(BaseInstance):
    jaccl_devices: list[list[str | None]]
    jaccl_coordinators: dict[NodeId, str]


class LlamaRpcInstance(BaseInstance):
    """Multi-node llama.cpp placement: one driver plus RPC memory donors (#328).

    The driver node runs the served engine (``llama-server``) with
    ``--rpc <endpoints>`` and holds the model file; each donor node runs
    ``ggml-rpc-server`` bound to its stamped endpoint and lends GPU memory.
    llama.cpp splits weights/KV across the pooled devices proportional to
    their free memory, so the shard assignments carry no meaningful layer
    ranges (the driver's shard nominally spans all layers; donors carry the
    degenerate ``RpcDonorShardMetadata``). Endpoints are chosen at placement
    time from the observed connectivity between the pair, preferring the
    fastest interconnect (Thunderbolt first, #265), and must be static
    routable addresses (never link-local; see the multi-node GGUF design
    record's asymmetric-routing trap).
    """

    driver_node: NodeId
    """The node that runs llama-server and reads the model file."""

    donor_endpoints: dict[NodeId, str]
    """Per-donor ``ip:port`` the donor's ggml-rpc-server binds and the driver
    dials, keyed by donor node id."""


# TODO: Single node instance
Instance = MlxRingInstance | MlxJacclInstance | LlamaRpcInstance


def instance_meta_of(instance: Instance) -> InstanceMeta:
    """The ``InstanceMeta`` a concrete instance actually embodies.

    Placement normalizes and resolves shapes (a single-node cycle becomes a
    ring regardless of the requested meta; an RPC-capable cycle mints a
    ``LlamaRpcInstance`` even from a ring request), so consumers reporting or
    recovering intent must derive the meta from the instance TYPE, never echo
    the requested value. Single source of truth for that derivation.
    """
    if isinstance(instance, MlxJacclInstance):
        return InstanceMeta.MlxJaccl
    if isinstance(instance, LlamaRpcInstance):
        return InstanceMeta.LlamaRpc
    return InstanceMeta.MlxRing


class BoundInstance(CamelCaseModel):
    instance: Instance
    bound_runner_id: RunnerId
    bound_node_id: NodeId

    @property
    def bound_shard(self) -> ShardMetadata:
        shard = self.instance.shard(self.bound_runner_id)
        assert shard is not None
        return shard

    @property
    def is_image_model(self) -> bool:
        return (
            ModelTask.TextToImage in self.bound_shard.model_card.tasks
            or ModelTask.ImageToImage in self.bound_shard.model_card.tasks
        )

    @property
    def is_embedding_model(self) -> bool:
        return ModelTask.TextEmbedding in self.bound_shard.model_card.tasks

    @property
    def is_speech_model(self) -> bool:
        return (
            ModelTask.TextToSpeech in self.bound_shard.model_card.tasks
            or ModelTask.SpeechToText in self.bound_shard.model_card.tasks
            or ModelTask.SpeechTranslation in self.bound_shard.model_card.tasks
        )

    @model_validator(mode="after")
    def validate_shard_exists(self) -> "BoundInstance":
        assert (
            self.bound_runner_id in self.instance.shard_assignments.runner_to_shard
        ), (
            "Bound Instance must be constructed with a runner_id that is in the instances assigned shards"
        )
        return self
