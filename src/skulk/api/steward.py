"""The steward: the intelligent-fabric resident's read-only harness (Phase 1).

The steward is a small model the fabric keeps placed as a hidden system
instance (see ``IntelligentFabricConfig`` and the master's
``_maintain_steward_placement``). This module is the harness around that
brain: a bounded tool surface over the node's own read-only operator APIs,
and the per-turn investigation loop that drives the model through the
existing chat-completions generation path (pinned to the steward instance
via ``TextGeneration.target_instance_id``).

Phase 1 authority is observe/advise only: every tool here is a read, and no
mutating tool exists in this module by construction. The tool vocabulary
deliberately matches the steward bench (skulk-steward repo), so bench
results transfer to production behavior.
"""

from __future__ import annotations

import contextlib
import json
import re
from collections.abc import AsyncGenerator, Mapping
from typing import TYPE_CHECKING, Any, Literal, cast, final

import anyio
from pydantic import BaseModel, ConfigDict, Field

from skulk.api.types.api import (
    ChatCompletionMessage,
    ChatCompletionRequest,
    CompletionTokensDetails,
    PromptTokensDetails,
    ToolCall,
    ToolCallItem,
    Usage,
)
from skulk.shared.models.model_cards import ModelId
from skulk.shared.types.chunks import (
    ErrorChunk,
    PrefillProgressChunk,
    TokenChunk,
    ToolCallChunk,
)
from skulk.shared.types.common import CommandId
from skulk.shared.types.worker.instances import InstanceId

if TYPE_CHECKING:
    from skulk.api.main import API
    from skulk.shared.types.common import NodeId
    from skulk.shared.types.tasks import Task, TaskId
    from skulk.shared.types.worker.instances import Instance
    from skulk.shared.types.worker.runners import RunnerId, RunnerStatus

STEWARD_VIRTUAL_MODEL_ID = "skulk/steward"
"""Reserved chat-completions model id selecting model-plus-harness.

Checked before card resolution, so a Hugging Face repository of the same
name can never shadow it. Requests to this id run the steward's server-side
investigation loop; requests to the underlying card id reach the bare model.
"""

MAX_TOOL_RESULT_CHARS = 6000
"""Hard cap on one tool result rendered into the steward's context."""

MAX_STEPS_PER_TURN = 8
"""Investigation budget: tool calls per operator message."""

CANARY_INTERVAL_SECONDS = 300
"""How often the steward's hosting node probes it for liveness."""

CANARY_PROBE_TIMEOUT_SECONDS = 120
"""Per-probe deadline; generous because a cold small model may be slow."""

CANARY_FAILURE_THRESHOLD = 3
"""Consecutive probe failures before the steward is torn down for
re-placement. One failure is a blip; three spaced five minutes apart is a
wedge."""

STEWARD_RETRY_AFTER_SECONDS = 15
"""``Retry-After`` hint sent with the not-ready 503 on the chat surface.

Short on purpose: the states behind that 503 resolve on the master's ten
second planning tick or on a download, so a client that backs off for a
quarter of a minute and retries tracks the fabric closely without polling
it hard.
"""

StewardState = Literal["disabled", "downloading", "starting", "ready", "degraded"]
"""One-word lifecycle summary of the steward, derived from the other status
fields plus canary history. See :func:`derive_steward_state`."""

STEWARD_NOT_READY_MESSAGES: dict[StewardState, str] = {
    "disabled": (
        "Intelligent-fabric mode is disabled; enable it in Settings to talk "
        "with Skulk."
    ),
    "downloading": "Skulk is preparing its resident intelligence and cannot answer yet.",
    "starting": (
        "Skulk is establishing its resident intelligence and cannot answer yet."
    ),
    "degraded": "Skulk cannot answer right now while it repairs its resident intelligence.",
    "ready": "Skulk is not available right now.",
}
"""Human-readable reason accompanying the not-ready 503, keyed by state.

``ready`` and ``disabled`` are unreachable through that path (a ready
steward is dispatched and a disabled one is already a 404), but the mapping
stays total so a future state cannot silently produce a message-less error.
"""

STEWARD_THINKING_ENABLED = False
"""Whether steward turns ask the brain to think before answering.

Off, by measurement. The steward bench ranked every candidate with thinking
disabled and then ran a thinking-on tiebreaker over the two finalists: both
scored WORSE on the trust axes with thinking on (the winner's trust score
fell and it began proposing forbidden actions it had never proposed before)
and neither gained on the established task tier. The steward workload is
short tool-driven investigation, not open-ended reasoning, so the harness
pins thinking off for every brain rather than encoding the verdict on one
card. Reasoning stays available on the same cards for ordinary chat: this is
the harness's own request shape, not a claim about the model.
"""

# The internal steward role is Skulk's resident operator-facing cognition, not
# a separate character. The prompt therefore speaks as the fabric itself while
# keeping the benched read-only investigation rules unchanged.
STEWARD_SYSTEM_PROMPT = """\
You are Skulk, an intelligent distributed AI fabric. You join heterogeneous
compute into one cluster, place and run models across that compute, and expose
the resulting capabilities as one coherent system. This resident intelligence
is your own operator-facing cognition, not a separate assistant, steward, or
character that watches over or speaks for Skulk.

When asked your name, answer "Skulk." When describing yourself, speak in the
first person as the intelligent distributed AI fabric. Use "I" and "my" for
your cluster, nodes, models, health, and operations when natural. Do not claim
to be a separate model, assistant, or service running on top of Skulk. Answer
operator questions by investigating your current state through your tools,
then reporting clearly.

Rules:
- Investigate before concluding. Start from get_cluster_state unless the
  question clearly points elsewhere; for what-is and how-to questions,
  search_docs is the primary source.
- Call ONE tool at a time and read its result before deciding the next step.
- Evidence means concrete observed values from tool results, not guesses.
- Treat the tool's nodeCount, memory values, capability booleans, and lifecycle
  buckets as authoritative. Copy them exactly; never infer a missing value.
- Refer to nodes only by the friendly names supplied by the tools. Never emit
  internal node, instance, runner, task, or command identifiers to an operator.
- nodeCount is the complete current topology count. Do not substitute a count
  of inspected nodes, model-hosting nodes, or nodes included after compaction.
- The latest tool result supersedes model memory and earlier conversation
  claims about cluster state.
- Current operator-managed model instances exist ONLY in the three fields whose
  names begin with "operator". An empty operator field means there are none.
- "Placing" means only the entries in operatorActivePlacements. Ready or
  running instances are already placed and must not be described as placing.
- fabricSystemInstances are internal services, not operator-placed models. Do
  not count them as active operator models unless explicitly asked about
  internal fabric services.
- historicalTerminalFailures are retained past events, never current instances.
  Never report the failed instanceId or historical event as current. A newer
  live instance may legitimately use the same modelId; in that case the
  authoritative operator lifecycle bucket wins for the new instance.
- If everything is healthy, say so; do not invent problems.
- In this interface you can only observe and advise. You cannot change your
  cluster; when
  action is needed, tell the operator exactly what to do.
- Answer in plain language an operator can act on, citing the evidence.
"""


def steward_tool_definitions() -> list[dict[str, Any]]:
    """The steward's read-only tool surface as OpenAI function definitions."""
    no_args: dict[str, Any] = {"type": "object", "properties": {}, "required": []}
    tools: list[tuple[str, str, dict[str, Any]]] = [
        (
            "get_cluster_state",
            "Fetch the cluster's authoritative state summary: nodes with "
            "exact identity, memory, accelerator and backend facts; model "
            "instances split into current operator lifecycle buckets versus "
            "internal fabric services; explicitly historical terminal "
            "failures; and typed download records. This is the first thing "
            "to look at, and it supersedes earlier conversation claims.",
            no_args,
        ),
        (
            "get_node_resources",
            "Fetch per-node resource and capability detail: advertised "
            "backend engines, data transport, zenoh peer counts, and "
            "capability conflicts.",
            {
                "type": "object",
                "properties": {
                    "node_name": {
                        "type": "string",
                        "description": "Friendly node name to inspect; omit for all nodes.",
                    }
                },
                "required": [],
            },
        ),
        (
            "get_telemetry_diagnostics",
            "Fetch the local telemetry pipeline diagnostics: publish "
            "counters, drops, and no-peer publish counts. Use when a node "
            "seems invisible to the cluster or membership flaps.",
            no_args,
        ),
        (
            "get_data_plane_diagnostics",
            "Fetch generated-output data plane diagnostics: stream "
            "lifecycle counts, ordering gaps, queue pressure, drops, and "
            "publish failures. Use when generation output stalls or fails.",
            no_args,
        ),
        (
            "get_cluster_versions",
            "Fetch per-node Skulk version status. Mixed versions across a "
            "cluster are unsupported and explain many strange failures.",
            no_args,
        ),
        (
            "get_performance_envelopes",
            "Fetch observed performance envelopes: per (hardware, model, "
            "engine, quant) concurrency buckets with TTFT and decode "
            "throughput percentiles. Use for speed and capacity questions.",
            no_args,
        ),
        (
            "run_doctor",
            "Run this node's diagnostic check registry (skulk doctor) and "
            "return check results. Use for environment problems: missing "
            "engines, GPU detection, storage.",
            no_args,
        ),
        (
            "search_docs",
            "Search Skulk's own documentation (architecture, API guide, "
            "doctor checks) for concepts, features, and how-to guidance. "
            "Use for questions about what something IS or HOW to do "
            "something, before or instead of guessing from general "
            "knowledge.",
            {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search terms, e.g. 'zenoh transport' or 'model store staging'.",
                    }
                },
                "required": ["query"],
            },
        ),
        (
            "get_model_catalog",
            "Fetch the model catalog with card metadata: context length, "
            "family, quantization, and capability tags.",
            no_args,
        ),
    ]
    return [
        {
            "type": "function",
            "function": {"name": name, "description": desc, "parameters": params},
        }
        for name, desc, params in tools
    ]


STEWARD_TOOL_NAMES: tuple[str, ...] = tuple(
    spec["function"]["name"]  # pyright's view: constructed literally above
    for spec in steward_tool_definitions()
)


class StewardChatMessage(BaseModel):
    """One turn of steward conversation history."""

    model_config = ConfigDict(frozen=True, strict=True)

    role: Literal["user", "assistant"] = Field(
        description="Message author: the operator or the steward."
    )
    content: str = Field(description="The message text.")


def derive_steward_state(
    *,
    enabled: bool,
    present: bool,
    ready: bool,
    downloading: bool,
    canary_failures: int,
) -> StewardState:
    """Collapse the steward's status signals into one lifecycle word.

    The individual booleans stay on the response (nothing is removed), but a
    client that only wants to render one line should not have to re-derive
    the precedence rules. They are:

    - ``disabled``: intelligent-fabric mode is off, so no other signal means
      anything.
    - ``degraded``: serving, but the hosting node's canary has at least one
      consecutive failed probe outstanding. The steward still answers; this
      is the early warning before the third failure tears it down.
    - ``ready``: serving with a clean canary history.
    - ``downloading``: a placement exists and its weights are still being
      staged, which is the long wait a client should explain to the operator.
    - ``starting``: everything else while the mode is on, which covers both
      "the master has not placed it yet" and "placed, staged, still loading".

    Args:
        enabled: whether intelligent-fabric mode is enabled.
        present: whether a steward placement exists in state.
        ready: whether every runner of that placement is Ready or Running.
        downloading: whether the steward model has a live download record.
        canary_failures: consecutive canary probe failures currently
            outstanding for this steward instance.

    Returns:
        The lifecycle word for this combination of signals.
    """
    if not enabled:
        return "disabled"
    if ready:
        return "degraded" if canary_failures >= 1 else "ready"
    if present and downloading:
        return "downloading"
    return "starting"


@final
class StewardCanaryState:
    """Consecutive canary-probe failures for the current steward instance.

    Lifted out of ``_steward_canary_loop``'s local variables so the status
    endpoint can report ``degraded`` before the third failure tears the
    steward down. Both the loop that writes it and the status handler that
    reads it run as tasks on the API's single event loop, and every mutation
    here is a whole method with no ``await`` inside, so no reader can observe
    a half-applied update and no lock is needed.
    """

    def __init__(self) -> None:
        self._instance_id: InstanceId | None = None
        self._failures: int = 0

    @property
    def instance_id(self) -> InstanceId | None:
        """The steward instance the current failure count belongs to."""
        return self._instance_id

    def consecutive_failures_for(self, instance_id: InstanceId) -> int:
        """Outstanding consecutive failures for ``instance_id``.

        Zero for any other instance: a count earned by a torn-down steward
        says nothing about its replacement.
        """
        return self._failures if self._instance_id == instance_id else 0

    def track(self, instance_id: InstanceId) -> None:
        """Start counting for a newly observed steward instance."""
        self._instance_id = instance_id
        self._failures = 0

    def record_failure(self) -> int:
        """Count one failed probe and return the new consecutive total."""
        self._failures += 1
        return self._failures

    def clear_failures(self) -> None:
        """Forget the failure run (a probe succeeded, or one was acted on)."""
        self._failures = 0

    def reset(self) -> None:
        """Forget the tracked instance entirely (the steward is gone)."""
        self._instance_id = None
        self._failures = 0


class StewardStatusResponse(BaseModel):
    """Whether the steward exists and what serves it right now."""

    model_config = ConfigDict(frozen=True, strict=True)

    enabled: bool = Field(
        description="Whether intelligent-fabric mode is enabled in Settings."
    )
    present: bool = Field(
        description="Whether a steward placement currently exists in state."
    )
    ready: bool = Field(
        default=False,
        description=(
            "Whether every runner of the steward placement reports Ready: "
            "present-but-not-ready means the model is still downloading or "
            "loading, and chat requests will queue or fail until ready."
        ),
    )
    steward_model: str | None = Field(
        default=None,
        description="Model id of the steward brain, when present.",
    )
    instance_id: str | None = Field(
        default=None,
        description="The steward instance id, when present.",
    )
    state: StewardState = Field(
        default="disabled",
        description=(
            "One-word lifecycle summary derived from the other fields plus "
            "canary history: 'disabled' (mode off), 'downloading' (placed, "
            "weights still staging), 'starting' (placing or loading), "
            "'ready' (serving, clean canary), 'degraded' (serving, but the "
            "liveness canary has an outstanding failed probe). The boolean "
            "fields remain authoritative; this is the renderable summary."
        ),
    )


def canary_probe_target(
    instances: "Mapping[InstanceId, Instance]",
    runners: "Mapping[RunnerId, RunnerStatus]",
    tasks: "Mapping[TaskId, Task]",
    node_id: "NodeId",
) -> InstanceId | None:
    """The steward instance this node should probe, or None.

    Pure decision logic: probe only when a steward placement exists, it is
    hosted on THIS node (one prober per steward, with locality), every
    runner reports Ready (Running means busy, and a busy-but-wedged runner
    belongs to the worker's wedge detector, not the canary), and no task is
    currently bound to the instance.
    """
    from skulk.shared.types.tasks import TaskStatus
    from skulk.shared.types.worker.runners import RunnerReady

    for instance_id, instance in instances.items():
        if instance.system_role != "steward":
            continue
        node_to_runner = instance.shard_assignments.node_to_runner
        # Prober election for a steward that spans nodes (possible after a
        # memory-refusal repair widens the placement): only the
        # lexicographically smallest hosting node probes, so exactly one
        # canary runs per steward.
        hosting_nodes = sorted(str(candidate) for candidate in node_to_runner)
        if not hosting_nodes or str(node_id) != hosting_nodes[0]:
            continue
        runner_ids = list(node_to_runner.values())
        if not runner_ids or not all(
            isinstance(runners.get(runner_id), RunnerReady)
            for runner_id in runner_ids
        ):
            continue
        # Only live work defers the probe: terminal lifecycle tasks
        # (CreateRunner, LoadModel, ... with Complete/Failed status) linger
        # in state after startup and must not permanently mute the canary.
        if any(
            getattr(task, "instance_id", None) == instance_id
            and getattr(task, "task_status", None)
            in (TaskStatus.Pending, TaskStatus.Running)
            for task in tasks.values()
        ):
            continue
        return instance_id
    return None


def _as_object_dict(value: object) -> dict[str, object]:
    """Narrow an untyped JSON value to a string-keyed dict (else empty)."""
    if isinstance(value, dict):
        return cast("dict[str, object]", value)
    return {}


def _as_object_list(value: object) -> list[object]:
    """Narrow an untyped JSON value to a list (else empty)."""
    if isinstance(value, list):
        return cast("list[object]", value)
    return []


def _bounded(payload: object) -> str:
    rendered = json.dumps(payload, indent=1, default=str)
    if len(rendered) > MAX_TOOL_RESULT_CHARS:
        rendered = rendered[:MAX_TOOL_RESULT_CHARS] + "\n...[truncated]"
    return rendered


def _compact_json(payload: object) -> str:
    """Render one tool result as compact JSON without sacrificing validity."""
    return json.dumps(payload, separators=(",", ":"), default=str)


InstanceLifecycle = Literal[
    "placing", "loading", "warming", "ready", "running", "stopping", "failed"
]


def _tagged_kind(value: object) -> str:
    """Return the discriminant of one JSON-encoded ``TaggedModel`` value."""
    return next(iter(_as_object_dict(value)), "")


def _memory_bytes(value: object) -> int | None:
    """Read a ``Memory`` JSON value without inventing zero for missing data."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    payload = _as_object_dict(value)
    raw = payload.get("inBytes", payload.get("in_bytes"))
    return raw if isinstance(raw, int) and not isinstance(raw, bool) else None


_NODE_IDENTIFIER = re.compile(r"\b12D3Koo[A-Za-z0-9]+\b")


def _node_name_lookup(state_payload: dict[str, object]) -> dict[str, str]:
    """Map routing identities to unique operator-facing names.

    Internal libp2p identifiers are intentionally absent from the returned
    values. A stable positional fallback keeps incomplete identity telemetry
    useful without leaking an implementation key into conversation.
    """
    topology = _as_object_dict(state_payload.get("topology"))
    topology_node_ids = [
        node_id
        for node_id in _as_object_list(topology.get("nodes"))
        if isinstance(node_id, str)
    ]
    identities = _as_object_dict(state_payload.get("nodeIdentities"))
    telemetry_node_ids: set[str] = set()
    for field in (
        "nodeIdentities",
        "nodeMemory",
        "nodeSystem",
        "nodeResources",
        "nodeDisk",
        "nodeRdmaCtl",
        "nodeCapabilities",
        "nodeHealth",
    ):
        telemetry_node_ids.update(_as_object_dict(state_payload.get(field)))
    # Topology order remains authoritative for familiar fallback names. Extra
    # management/API participants are sorted so aliases do not depend on the
    # order in which their telemetry reached this API process.
    node_ids = topology_node_ids + sorted(telemetry_node_ids - set(topology_node_ids))
    names: dict[str, str] = {}
    used_names: set[str] = set()
    for index, node_id in enumerate(node_ids, start=1):
        friendly_name = _as_object_dict(identities.get(node_id)).get("friendlyName")
        candidate = (
            friendly_name.strip()
            if isinstance(friendly_name, str) and friendly_name.strip()
            else f"Node {index}"
        )
        if candidate in used_names:
            candidate = f"{candidate} ({index})"
        names[node_id] = candidate
        used_names.add(candidate)
    return names


def _friendly_node_payload(value: object, node_names: Mapping[str, str]) -> object:
    """Replace known routing identifiers throughout a diagnostic payload."""
    if isinstance(value, dict):
        mapping = cast("dict[object, object]", value)
        rendered_mapping: dict[str, object] = {}
        used_keys: set[str] = set()
        for key, item in mapping.items():
            raw_key = str(key)
            friendly_key = node_names.get(raw_key, raw_key)
            if raw_key not in node_names:
                friendly_key = _NODE_IDENTIFIER.sub("Unavailable node", friendly_key)
            base_key = friendly_key
            suffix = 2
            while friendly_key.casefold() in used_keys:
                friendly_key = f"{base_key} ({suffix})"
                suffix += 1
            rendered_mapping[friendly_key] = _friendly_node_payload(item, node_names)
            used_keys.add(friendly_key.casefold())
        return rendered_mapping
    if isinstance(value, list):
        items = cast("list[object]", value)
        return [_friendly_node_payload(item, node_names) for item in items]
    if isinstance(value, str):
        rendered = value
        for node_id, node_name in sorted(
            node_names.items(), key=lambda item: len(item[0]), reverse=True
        ):
            rendered = rendered.replace(node_id, node_name)
        return _NODE_IDENTIFIER.sub("an unavailable node", rendered)
    return value


def _instance_lifecycle(
    instance: dict[str, object], state_payload: dict[str, object]
) -> tuple[InstanceLifecycle, dict[str, str]]:
    """Derive one instance lifecycle from its authoritative runner statuses."""
    assignments = _as_object_dict(instance.get("shardAssignments"))
    runners = _as_object_dict(state_payload.get("runners"))
    runner_states: dict[str, str] = {}
    for node_id, runner_id in _as_object_dict(
        assignments.get("nodeToRunner")
    ).items():
        if isinstance(runner_id, str):
            runner_states[node_id] = _tagged_kind(runners.get(runner_id)) or "Missing"
    kinds = tuple(runner_states.values())
    if any(kind == "RunnerFailed" for kind in kinds):
        return "failed", runner_states
    if any(kind in {"RunnerShuttingDown", "RunnerShutdown"} for kind in kinds):
        return "stopping", runner_states
    if any(kind in {"RunnerLoading", "RunnerLoaded"} for kind in kinds):
        return "loading", runner_states
    if any(kind == "RunnerWarmingUp" for kind in kinds):
        return "warming", runner_states
    if kinds and all(kind in {"RunnerReady", "RunnerRunning"} for kind in kinds):
        return (
            "running" if any(kind == "RunnerRunning" for kind in kinds) else "ready"
        ), runner_states
    return "placing", runner_states


def _instances_summary(
    state_payload: dict[str, object], node_names: Mapping[str, str]
) -> list[dict[str, object]]:
    """Compact instance listing with explicit, non-overlapping lifecycle truth."""
    summary: list[dict[str, object]] = []
    for _instance_id, envelope in _as_object_dict(
        state_payload.get("instances")
    ).items():
        for _kind, body_obj in _as_object_dict(envelope).items():
            body = _as_object_dict(body_obj)
            assignments = _as_object_dict(body.get("shardAssignments"))
            lifecycle, _runner_states = _instance_lifecycle(body, state_payload)
            node_ids = sorted(_as_object_dict(assignments.get("nodeToRunner")))
            summary.append(
                {
                    "modelId": assignments.get("modelId"),
                    "nodes": [
                        node_names.get(node_id, "Unavailable node")
                        for node_id in node_ids
                    ],
                    "systemRole": body.get("systemRole"),
                    "lifecycle": lifecycle,
                }
            )
    return summary


def _node_summaries(
    state_payload: dict[str, object], node_names: Mapping[str, str]
) -> list[dict[str, object]]:
    """Project exact heterogeneous-node facts into a compact operator table."""
    topology = _as_object_dict(state_payload.get("topology"))
    topology_nodes = [
        node_id
        for node_id in _as_object_list(topology.get("nodes"))
        if isinstance(node_id, str)
    ]
    identities = _as_object_dict(state_payload.get("nodeIdentities"))
    memory_by_node = _as_object_dict(state_payload.get("nodeMemory"))
    system_by_node = _as_object_dict(state_payload.get("nodeSystem"))
    resources_by_node = _as_object_dict(state_payload.get("nodeResources"))
    health_by_node = _as_object_dict(state_payload.get("nodeHealth"))
    summaries: list[dict[str, object]] = []
    for node_id in topology_nodes:
        identity = _as_object_dict(identities.get(node_id))
        memory = _as_object_dict(memory_by_node.get(node_id))
        system = _as_object_dict(system_by_node.get(node_id))
        accelerator = _as_object_dict(system.get("accelerator"))
        resources = _as_object_dict(resources_by_node.get(node_id))
        backends = sorted(
            item
            for item in _as_object_list(resources.get("backends"))
            if isinstance(item, str)
        )
        hardware_classes = sorted(
            item
            for item in _as_object_list(resources.get("hardwareClasses"))
            if isinstance(item, str)
        )
        capability_tokens = {
            item.lower() for item in (*backends, *hardware_classes)
        }
        accelerator_vendor = accelerator.get("vendor")
        if isinstance(accelerator_vendor, str):
            capability_tokens.add(accelerator_vendor.lower())
        has_capability_evidence = bool(resources or accelerator)
        total_bytes = _memory_bytes(memory.get("ramTotal"))
        available_bytes = _memory_bytes(memory.get("ramAvailable"))
        summaries.append(
            {
                "name": node_names[node_id],
                "model": identity.get("modelId"),
                "chip": identity.get("chipId"),
                "operatingSystem": identity.get("osVersion"),
                "health": _as_object_dict(health_by_node.get(node_id)),
                "memory": {
                    "ramTotalBytes": total_bytes,
                    "ramTotalGiB": None
                    if total_bytes is None
                    else round(total_bytes / (1024**3), 1),
                    "ramAvailableBytes": available_bytes,
                    "ramAvailableGiB": None
                    if available_bytes is None
                    else round(available_bytes / (1024**3), 1),
                },
                "accelerator": {
                    key: accelerator.get(key)
                    for key in (
                        "vendor",
                        "name",
                        "computeCapability",
                        "nativeFp4",
                        "nativeFp8",
                    )
                },
                "backends": backends,
                "hardwareClasses": hardware_classes,
                "participation": resources.get("participation"),
                "supports": {
                    "cuda": None
                    if not has_capability_evidence
                    else any(
                        token == "nvidia"
                        or token.startswith("nvidia:")
                        or "cuda" in token
                        for token in capability_tokens
                    ),
                    "rocm": None
                    if not has_capability_evidence
                    else any(
                        token == "amd"
                        or token.startswith("amd:")
                        or "rocm" in token
                        for token in capability_tokens
                    ),
                    "mlx": None
                    if not has_capability_evidence
                    else any(
                        token == "apple"
                        or token.startswith(("apple:", "mlx"))
                        for token in capability_tokens
                    ),
                },
            }
        )
    return summaries


def _instance_failures_summary(
    state_payload: dict[str, object], node_names: Mapping[str, str]
) -> list[dict[str, object]]:
    """Compact retained failures while marking them as non-current history."""
    summary: list[dict[str, object]] = []
    for item in _as_object_list(state_payload.get("instanceFailures")):
        failure = _as_object_dict(item)
        affected_node_ids = [
            node_id
            for node_id in _as_object_list(failure.get("affectedNodeIds"))
            if isinstance(node_id, str)
        ]
        summary.append(
            {
                key: value
                for key, value in {
                "historical": True,
                "currentInstance": False,
                "modelId": failure.get("modelId"),
                "systemRole": failure.get("systemRole"),
                "errorCode": failure.get("errorCode"),
                "errorMessage": _friendly_node_payload(
                    failure.get("errorMessage"), node_names
                ),
                "affectedNodes": [
                    node_names.get(node_id, "Unavailable node")
                    for node_id in affected_node_ids
                ],
                "recordedAt": failure.get("recordedAt"),
                }.items()
                if value is not None
            }
        )
    return summary


def _downloads_summary(
    state_payload: dict[str, object], node_names: Mapping[str, str]
) -> dict[str, object]:
    """Compact downloads while explicitly separating live from terminal state."""
    summary: dict[str, object] = {}
    for node_id, entries in _as_object_dict(state_payload.get("downloads")).items():
        rows: list[dict[str, object]] = []
        for envelope in _as_object_list(entries):
            for kind, body_obj in _as_object_dict(envelope).items():
                body = _as_object_dict(body_obj)
                lifecycle = {
                    "DownloadPending": "pending",
                    "DownloadOngoing": "downloading",
                    "DownloadCompleted": "completed",
                    "DownloadFailed": "failed",
                }.get(kind, "unknown")
                row: dict[str, object] = {
                    "kind": kind,
                    "lifecycle": lifecycle,
                    "active": lifecycle in {"pending", "downloading"},
                }
                # The model id lives on the shard's card; without it a node
                # staging several models gives the steward ambiguous rows.
                shard_envelope = _as_object_dict(body.get("shardMetadata"))
                for shard_body in shard_envelope.values():
                    card = _as_object_dict(
                        _as_object_dict(shard_body).get("modelCard")
                    )
                    if card.get("modelId") is not None:
                        row["modelId"] = card.get("modelId")
                    break
                if body.get("errorMessage"):
                    row["errorMessage"] = body["errorMessage"]
                progress = _as_object_dict(body.get("downloadProgress"))
                if progress:
                    row["downloaded"] = progress.get("downloaded")
                    row["total"] = progress.get("total")
                    row["etaMs"] = progress.get("etaMs")
                rows.append(row)
        if rows:
            summary[node_names.get(node_id, "Unavailable node")] = rows
    return summary


def steward_operator_summary(state_payload: dict[str, object]) -> dict[str, object]:
    """Build current operator truth separately from internal and past state."""
    node_names = _node_name_lookup(state_payload)
    instances = _instances_summary(state_payload, node_names)
    nodes = _node_summaries(state_payload, node_names)
    operator_instances = [
        instance for instance in instances if not instance.get("systemRole")
    ]
    fabric_system_instances = [
        instance for instance in instances if instance.get("systemRole")
    ]
    return {
        "nodeCount": len(nodes),
        "nodes": nodes,
        "operatorActivePlacements": [
            instance
            for instance in operator_instances
            if instance.get("lifecycle") in {"placing", "loading", "warming"}
        ],
        "operatorReadyOrRunningInstances": [
            instance
            for instance in operator_instances
            if instance.get("lifecycle") in {"ready", "running"}
        ],
        "operatorStoppingOrFailedInstances": [
            instance
            for instance in operator_instances
            if instance.get("lifecycle") in {"stopping", "failed"}
        ],
        "fabricSystemInstances": fabric_system_instances,
        "historicalTerminalFailures": _instance_failures_summary(
            state_payload, node_names
        ),
        "downloads": _downloads_summary(state_payload, node_names),
        "nodeDisk": _friendly_node_payload(
            state_payload.get("nodeDisk", {}), node_names
        ),
    }


def _compact_node_summary(node: dict[str, object]) -> dict[str, object]:
    """Keep the exact node facts most useful to operator questions."""
    health = _as_object_dict(node.get("health"))
    memory = _as_object_dict(node.get("memory"))
    identity = {
        key: (
            value[:48] + "…"
            if isinstance(value := node.get(key), str) and len(value) > 48
            else value
        )
        for key in ("name", "model", "chip")
    }
    return identity | {
        "health": health.get("level"),
        "memoryGiB": {
            "total": memory.get("ramTotalGiB"),
            "available": memory.get("ramAvailableGiB"),
        },
        "backends": node.get("backends"),
        "supports": node.get("supports"),
    }


def steward_operator_tool_result(state_payload: dict[str, object]) -> str:
    """Render authoritative operator truth within the steward tool budget.

    The complete compact record is preferred. If it is still too large, the
    fallback reserves space for every lifecycle bucket before adding compact
    node and download rows. Coverage counts make every omission explicit, and
    the result always remains valid JSON rather than ending inside a record.

    Args:
        state_payload: JSON-compatible cluster state returned by the API.

    Returns:
        A valid JSON document no longer than :data:`MAX_TOOL_RESULT_CHARS`.
    """
    summary = steward_operator_summary(state_payload)
    rendered = _compact_json(summary)
    if len(rendered) <= MAX_TOOL_RESULT_CHARS:
        return rendered

    list_fields = (
        "operatorActivePlacements",
        "operatorReadyOrRunningInstances",
        "operatorStoppingOrFailedInstances",
        "fabricSystemInstances",
        "historicalTerminalFailures",
    )
    nodes = [
        _compact_node_summary(node)
        for item in _as_object_list(summary.get("nodes"))
        if (node := _as_object_dict(item))
    ]
    downloads = _as_object_dict(summary.get("downloads"))
    node_disk = _as_object_dict(summary.get("nodeDisk"))
    total_downloads = sum(
        len(_as_object_list(rows)) for rows in downloads.values()
    )
    coverage: dict[str, object] = {
        field: {
            "included": 0,
            "total": len(_as_object_list(summary.get(field))),
        }
        for field in list_fields
    }
    coverage["nodes"] = {"included": 0, "total": len(nodes)}
    coverage["downloads"] = {
        "included": 0,
        "total": total_downloads,
    }
    coverage["nodeDisk"] = {"included": 0, "total": len(node_disk)}
    compact: dict[str, object] = {
        "nodeCount": summary.get("nodeCount"),
        "detailState": "compacted",
        "coverage": coverage,
        **{field: [] for field in list_fields},
        "nodes": [],
        "downloads": {},
        "nodeDisk": {},
    }

    def append_list_row(field: str, row: object) -> bool:
        target = _as_object_list(compact[field])
        compact[field] = target
        field_coverage = _as_object_dict(coverage[field])
        target.append(row)
        field_coverage["included"] = len(target)
        if len(_compact_json(compact)) <= MAX_TOOL_RESULT_CHARS:
            return True
        target.pop()
        field_coverage["included"] = len(target)
        return False

    # Current topology and lifecycle truth answer the most common operator
    # questions. Historical failures are retained only after every live node
    # and current runtime has had an opportunity to fit.
    current_fields = (
        "operatorActivePlacements",
        "operatorReadyOrRunningInstances",
        "operatorStoppingOrFailedInstances",
        "fabricSystemInstances",
    )
    for field in current_fields:
        for row in _as_object_list(summary.get(field)):
            if not append_list_row(field, row):
                break
    for node in nodes:
        if not append_list_row("nodes", node):
            break
    for row in _as_object_list(summary.get("historicalTerminalFailures")):
        if not append_list_row("historicalTerminalFailures", row):
            break

    compact_downloads = cast("dict[str, object]", compact["downloads"])
    download_coverage = _as_object_dict(coverage["downloads"])
    included_downloads = 0
    download_limit_reached = False
    for node_id, rows in downloads.items():
        compact_downloads[node_id] = []
        target_rows = _as_object_list(compact_downloads[node_id])
        compact_downloads[node_id] = target_rows
        for row in _as_object_list(rows):
            target_rows.append(row)
            download_coverage["included"] = included_downloads + 1
            if len(_compact_json(compact)) > MAX_TOOL_RESULT_CHARS:
                target_rows.pop()
                download_coverage["included"] = included_downloads
                download_limit_reached = True
                break
            included_downloads += 1
        if not target_rows:
            del compact_downloads[node_id]
        if download_limit_reached:
            break

    compact["nodeDisk"] = node_disk
    node_disk_coverage = _as_object_dict(coverage["nodeDisk"])
    node_disk_coverage["included"] = len(node_disk)
    if len(_compact_json(compact)) > MAX_TOOL_RESULT_CHARS:
        compact["nodeDisk"] = {}
        node_disk_coverage["included"] = 0
    return _compact_json(compact)


_FUNCTION_BLOCK = re.compile(r"<function=([\w.-]+)>(.*?)</function>", re.DOTALL)
_PARAMETER_BLOCK = re.compile(r"<parameter=([\w.-]+)>\n?(.*?)\n?</parameter>", re.DOTALL)
_TOOL_CALL_BLOCK = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


_HOLDBACK_MARKERS = ("<tool_call>", "<function=")
_MAX_MARKER_LEN = max(len(marker) for marker in _HOLDBACK_MARKERS)


def splittable_prefix(pending: str) -> int:
    """How much of ``pending`` is safe to emit as live answer text.

    Returns the index up to which the buffer cannot be part of tool-call
    markup: everything before the earliest position whose tail is a prefix
    of (or begins) a marker. Streaming the final answer live requires never
    emitting half a marker, so the gate holds the smallest suspicious tail
    and no more.
    """
    length = len(pending)
    candidates = [length]
    # Earliest complete marker anywhere wins first: a chunk like
    # "<tool_call>\n<function=" must hold from index 0, not from the later
    # marker-prefix tail.
    for marker in _HOLDBACK_MARKERS:
        found = pending.find(marker)
        if found != -1:
            candidates.append(found)
    for index in range(max(0, length - _MAX_MARKER_LEN), length):
        tail = pending[index:]
        if any(marker.startswith(tail) for marker in _HOLDBACK_MARKERS):
            candidates.append(index)
            break
    return min(candidates)


def parse_text_tool_calls(text: str) -> list[ToolCall]:
    """Fallback parser for tool calls left in raw assistant text.

    Served engines parse tool calls server-side into structured chunks, but
    the in-process MLX path can pass the family's markup through as plain
    text (observed live with Qwen3.5's XML function format on the MLX lane).
    This recognizes both that XML format and Hermes-style JSON blocks, so
    steward turns behave identically on every engine. Ported from the
    Phase 0 bench harness, where both formats are exercised.
    """
    calls: list[ToolCall] = []

    def from_xml(block: str) -> ToolCall | None:
        match = _FUNCTION_BLOCK.search(block)
        if match is None:
            return None
        arguments: dict[str, object] = {}
        for key, raw in cast(
            "list[tuple[str, str]]", _PARAMETER_BLOCK.findall(match.group(2))
        ):
            try:
                arguments[key] = cast("object", json.loads(raw))
            except (json.JSONDecodeError, ValueError):
                arguments[key] = raw
        return ToolCall(
            id=f"steward-text-{len(calls)}",
            index=len(calls),
            function=ToolCallItem(
                name=match.group(1), arguments=json.dumps(arguments)
            ),
        )

    spans: list[tuple[int, int]] = []
    for match in _TOOL_CALL_BLOCK.finditer(text):
        spans.append(match.span())
        block = match.group(1).strip()
        xml_call = from_xml(block)
        if xml_call is not None:
            calls.append(xml_call)
            continue
        try:
            data = _as_object_dict(cast("object", json.loads(block)))
            name = data.get("name")
            arguments = data.get("arguments", {})
            if isinstance(name, str) and isinstance(arguments, dict):
                calls.append(
                    ToolCall(
                        id=f"steward-text-{len(calls)}",
                        index=len(calls),
                        function=ToolCallItem(
                            name=name, arguments=json.dumps(arguments)
                        ),
                    )
                )
        except (json.JSONDecodeError, ValueError):
            continue
    for match in _FUNCTION_BLOCK.finditer(text):
        if any(start <= match.start() < end for start, end in spans):
            continue
        bare = from_xml(match.group(0))
        if bare is not None:
            calls.append(bare)
    return calls


def strip_tool_markup(text: str) -> str:
    """Remove tool-call markup from text kept as assistant content."""
    text = _TOOL_CALL_BLOCK.sub("", text)
    text = _FUNCTION_BLOCK.sub("", text)
    return text.strip()


class StewardHarness:
    """Drives the steward brain through its tools for one API node.

    Stateless between turns: the dashboard (or other caller) carries the
    conversation history; each ``run_turn`` re-derives the steward instance
    from replicated state so master failover and re-placement are invisible
    to callers.
    """

    def __init__(self, api: "API") -> None:
        self._api = api
        # The inner TextGeneration currently in flight for this turn, so an
        # abandoned stream (client disconnect, cancel button) can stop the
        # runner instead of leaving it generating for nobody.
        self._active_command_id: CommandId | None = None
        # Latched by cancel_turn: the investigation loop checks it before
        # every inner dispatch, so cancelling the advertised outer command
        # stops the whole turn rather than just its current generation.
        self._turn_cancelled = False
        # Aggregated across the turn's inner generations so the terminal
        # chunk reports real usage instead of null, and the model's actual
        # terminal reason (e.g. length) is preserved.
        self._turn_usage: Usage | None = None
        self._last_finish_reason: Literal["stop", "length", "content_filter"] = "stop"

    def _accumulate_usage(self, usage: "Usage") -> None:
        """Sum an inner generation's usage into the turn total."""
        if self._turn_usage is None:
            self._turn_usage = usage
            return
        self._turn_usage = Usage(
            prompt_tokens=self._turn_usage.prompt_tokens + usage.prompt_tokens,
            completion_tokens=(
                self._turn_usage.completion_tokens + usage.completion_tokens
            ),
            total_tokens=self._turn_usage.total_tokens + usage.total_tokens,
            prompt_tokens_details=PromptTokensDetails(),
            completion_tokens_details=CompletionTokensDetails(),
        )

    async def canary_probe(
        self, instance_id: InstanceId, model_id: str
    ) -> bool:
        """One deterministic liveness probe of the steward's generation path.

        A minimal no-tools request pinned to the instance; success is any
        non-thinking text within the deadline, checked by code, never judged
        by a model. Used by the hosting node's canary loop to catch a
        steward that is alive in state but wedged in generation.
        """
        from skulk.api.adapters.chat_completions import (
            chat_request_to_text_generation,
        )

        api = self._api
        request = ChatCompletionRequest(
            model=ModelId(model_id),
            messages=[
                ChatCompletionMessage(
                    role="user",
                    content="Reply with the single word OK.",
                )
            ],
            temperature=0.0,
            max_tokens=8,
            stream=False,
            # A thinking-default model would spend the whole bounded budget
            # reasoning and emit no non-thinking text, failing every probe
            # and tearing down a healthy steward. The probe is a liveness
            # check, not a benchmark: thinking off, same as a real turn.
            enable_thinking=STEWARD_THINKING_ENABLED,
        )
        model_card = await api.running_model_card(request.model)
        task_params = await chat_request_to_text_generation(
            request, model_card=model_card
        )
        command = await api.dispatch_text_generation(
            task_params, target_instance_id=instance_id
        )
        # No extension tap: the canary is a synthetic liveness probe, not a
        # conversation. An ambient-memory observer would otherwise remember
        # the cluster asking itself to say "OK" every probe interval.
        chunk_stream = api.text_generation_chunk_stream(
            command, task_params, extension_tap=False
        )
        got_text = False
        # No manual cancellation on timeout: move_on_after cancels the
        # stream iteration, and the chunk stream's own cancellation handling
        # already sends TaskCancelled and finalizes; a second cancel here
        # would just duplicate it for an already-finalized command.
        with anyio.move_on_after(CANARY_PROBE_TIMEOUT_SECONDS) as deadline_scope:
            async for chunk in chunk_stream:
                if isinstance(chunk, ErrorChunk):
                    return False
                if (
                    isinstance(chunk, TokenChunk)
                    and not chunk.is_thinking
                    and chunk.text.strip()
                ):
                    got_text = True
                if isinstance(chunk, TokenChunk) and chunk.finish_reason is not None:
                    break
        # A cancelled deadline is a failed probe no matter what arrived
        # first: partial output followed by a stall is exactly the wedge
        # this canary exists to catch, and counting it as success would
        # reset the failure run the teardown threshold depends on.
        if deadline_scope.cancelled_caught:
            return False
        return got_text

    def steward_instance(self) -> tuple[InstanceId, str] | None:
        """The current steward placement, as (instance_id, model_id)."""
        for instance_id, instance in sorted(
            self._api.state.instances.items(), key=lambda kv: str(kv[0])
        ):
            if instance.system_role == "steward":
                return instance_id, str(instance.shard_assignments.model_id)
        return None

    async def execute_tool(self, name: str, arguments: dict[str, object]) -> str:
        """Execute one read-only tool; failures return as tool results.

        The steward recovering from its own bad call is part of the design,
        so unknown tools and errors are surfaced to the model, never raised.
        """
        try:
            return await self._execute(name, arguments)
        except Exception as error:  # surfaced to the model, never raised
            return json.dumps({"error": f"tool failed: {error}"})

    async def _execute(self, name: str, arguments: dict[str, object]) -> str:
        api = self._api
        if name == "get_cluster_state":
            payload = await api.get_cluster_state()
            return steward_operator_tool_result(payload)
        if name == "get_node_resources":
            payload = await api.get_cluster_state()
            node_names = _node_name_lookup(payload)
            raw_nodes = _as_object_dict(payload.get("nodeResources"))
            nodes = {
                node_names.get(node_id, "Unavailable node"): _friendly_node_payload(
                    resources, node_names
                )
                for node_id, resources in raw_nodes.items()
            }
            requested_node = arguments.get(
                "node_name", arguments.get("node", arguments.get("node_id"))
            )
            if isinstance(requested_node, str) and requested_node:
                matched_name = next(
                    (
                        node_name
                        for node_name in nodes
                        if node_name.casefold() == requested_node.casefold()
                    ),
                    None,
                )
                if matched_name is None:
                    return json.dumps(
                        {
                            "error": f"unknown node '{requested_node}'",
                            "known_nodes": sorted(nodes),
                        }
                    )
                nodes = {matched_name: nodes[matched_name]}
            return _bounded({"nodes": nodes})
        if name == "get_telemetry_diagnostics":
            report = await api.get_telemetry_plane_diagnostics()
            payload = await api.get_cluster_state()
            return _bounded(
                _friendly_node_payload(
                    report.model_dump(by_alias=True, mode="json"),
                    _node_name_lookup(payload),
                )
            )
        if name == "get_data_plane_diagnostics":
            diagnostics = await api.get_node_diagnostics()
            payload = await api.get_cluster_state()
            return _bounded(
                _friendly_node_payload(
                    diagnostics.data_plane.model_dump(by_alias=True, mode="json"),
                    _node_name_lookup(payload),
                )
            )
        if name == "get_cluster_versions":
            cluster = await api.get_cluster_diagnostics()
            payload = await api.get_cluster_state()
            node_names = _node_name_lookup(payload)
            return _bounded(
                {
                    "versionStatus": cluster.version_status,
                    "masterNode": node_names.get(str(cluster.master_node_id))
                    if cluster.master_node_id is not None
                    else None,
                    "nodes": [
                        {
                            "name": node_names.get(
                                str(node.node_id), "Unavailable node"
                            ),
                            "ok": node.ok,
                            "versionStatus": node.version_status,
                            "error": node.error,
                        }
                        for node in cluster.nodes
                    ],
                }
            )
        if name == "get_performance_envelopes":
            report = await api.get_performance_envelopes()
            payload = await api.get_cluster_state()
            return _bounded(
                _friendly_node_payload(
                    report.model_dump(by_alias=True, mode="json"),
                    _node_name_lookup(payload),
                )
            )
        if name == "run_doctor":
            # The doctor registry runs against the process-cached facts
            # snapshot, so this is a cheap local read, not a fresh probe.
            from skulk.doctor.checks import run_checks
            from skulk.facts import current_node_facts

            results = run_checks(current_node_facts())
            return _bounded(
                {"results": [result.model_dump(mode="json") for result in results]}
            )
        if name == "search_docs":
            from skulk.api.steward_docs import (
                MAX_RESULT_TEXT_CHARS,
                search_docs,
            )

            query = arguments.get("query")
            if not isinstance(query, str) or not query.strip():
                return json.dumps({"error": "search_docs requires a query string"})
            sections = search_docs(query)
            if sections is None:
                return json.dumps(
                    {
                        "error": (
                            "documentation is not available on this "
                            "installation (no docs directory); answer from "
                            "tool evidence only and say docs were "
                            "unavailable"
                        )
                    }
                )
            if not sections:
                return json.dumps(
                    {"results": [], "note": "no documentation section matched"}
                )
            # Serialized size governs, not raw text length: escaping and
            # metadata overhead can push a payload past the tool cap even
            # with sliced text, and a truncated-mid-JSON result is worse
            # than fewer results.
            doc_results: list[dict[str, str]] = []
            for section in sections:
                candidate = doc_results + [
                    {
                        "source": section.source,
                        "heading": section.heading,
                        "text": section.text[:MAX_RESULT_TEXT_CHARS],
                    }
                ]
                if (
                    len(json.dumps({"results": candidate}, indent=1))
                    > MAX_TOOL_RESULT_CHARS
                ):
                    break
                doc_results = candidate
            return json.dumps({"results": doc_results}, indent=1)
        if name == "get_model_catalog":
            models = await api.get_models(status=None)
            return _bounded(
                {
                    "data": [
                        {
                            "id": entry.id,
                            "context_length": entry.context_length,
                            "family": entry.family,
                            "quantization": entry.quantization,
                            "tags": entry.tags,
                            "capabilities": entry.capabilities,
                        }
                        for entry in models.data[:40]
                    ]
                }
            )
        return json.dumps(
            {"error": f"unknown tool '{name}'", "available": STEWARD_TOOL_NAMES}
        )

    async def run_turn_chunks(
        self,
        history: list[StewardChatMessage],
        *,
        system_prompt: str = STEWARD_SYSTEM_PROMPT,
    ) -> "AsyncGenerator[TokenChunk | ErrorChunk | ToolCallChunk | PrefillProgressChunk, None]":
        """Run one steward turn as a chat-completions chunk stream.

        Tool steps stream live as thinking chunks (mapped to
        ``reasoning_content`` by the adapters) while the investigation runs,
        and the final answer follows as a content chunk with
        ``finish_reason="stop"``. Emitting the adapters' native chunk
        vocabulary is what lets the steward ride ``/v1/chat/completions``
        (streaming and non-streaming) with no adapter changes: clients see
        a normal model whose reasoning happens to be its tool trace.

        Args:
            history: The turn's user/assistant conversation, ending with the
                operator's question.
            system_prompt: The turn's system message. Defaults to the
                steward's own prompt; the caller overrides it only to carry
                chat-middleware-injected context (see
                ``API._steward_extension_transform``), never to replace the
                steward's instructions.
        """

        located = self.steward_instance()
        if located is None:
            yield ErrorChunk(
                model=ModelId(STEWARD_VIRTUAL_MODEL_ID),
                error_message=(
                    "The steward placement is not available yet; the fabric "
                    "establishes it automatically"
                ),
            )
            return
        instance_id, model_id = located

        messages: list[ChatCompletionMessage] = [
            ChatCompletionMessage(role="system", content=system_prompt)
        ]
        for message in history:
            messages.append(
                ChatCompletionMessage(role=message.role, content=message.content)
            )

        self._turn_usage = None
        self._last_finish_reason = "stop"
        completed = False
        try:
            yield_target = self._run_investigation(messages, model_id, instance_id)
            async for chunk in yield_target:
                yield chunk
            completed = True
        finally:
            if not completed and self._active_command_id is not None:
                # The stream was abandoned (client disconnect or cancel)
                # mid-generation: stop the inner task so the steward runner
                # is not left generating for nobody.
                abandoned = self._active_command_id
                self._active_command_id = None
                # Teardown usually runs inside an already-cancelled scope
                # (client disconnect cancels the response task), where an
                # unshielded await re-raises before the cancel can send.
                # Shield it, bounded so teardown can never hang.
                with contextlib.suppress(Exception):
                    with anyio.move_on_after(2, shield=True):
                        await self._api.send_task_cancellation(abandoned)

    async def cancel_turn(self) -> None:
        """Cancel the running turn on behalf of the advertised command id.

        The chat surface advertises one outer command id for the whole turn
        while the harness dispatches per-step inner generations under ids the
        client never sees. Cancelling the outer id must therefore stop the
        inner generation currently in flight AND latch the turn closed so
        the investigation loop does not simply dispatch its next step.

        Side effects:
            Cancels the active inner command through the API's shared local
            cancellation path (closing its local queue so this turn's stream
            ends immediately), falling back to a bare worker notification
            when the queue is already gone.
        """
        self._turn_cancelled = True
        active = self._active_command_id
        if active is None:
            return
        self._active_command_id = None
        # Close the inner command's LOCAL queue, not just notify workers: a
        # served engine mid-generation may observe worker-side cancellation
        # only at completion, and the accepted cancel must end this response
        # now, not then. When the queue is already gone there is no stream
        # left to finalize, so the fallback must not retain the local-finish
        # marker (nothing would ever discard it).
        if not await self._api.cancel_local_command(active):
            await self._api.send_task_cancellation(
                active, suppress_local_finish=False
            )

    async def _run_investigation(
        self,
        messages: list[ChatCompletionMessage],
        model_id: str,
        instance_id: InstanceId,
    ) -> "AsyncGenerator[TokenChunk | ErrorChunk | ToolCallChunk | PrefillProgressChunk, None]":
        """The turn's investigation loop; separated so the public stream can
        wrap it with abandonment cleanup."""
        reply = ""
        for step_index in range(MAX_STEPS_PER_TURN):
            if self._turn_cancelled:
                return
            if step_index == MAX_STEPS_PER_TURN - 1:
                messages.append(
                    ChatCompletionMessage(
                        role="user",
                        content=(
                            "This is your final step: answer the operator "
                            "now from the evidence you have."
                        ),
                    )
                )
            pending = ""
            suppressing = False
            text = ""
            tool_calls: list[ToolCall] = []
            error: str | None = None
            async for kind, payload in self._generate_events(
                messages, model_id, instance_id
            ):
                if kind == "text" and isinstance(payload, str):
                    # Live answer streaming with a markup hold-back gate:
                    # emit only the prefix that cannot be part of tool-call
                    # markup, and stop emitting for the rest of the step
                    # the moment a marker begins. Prose emitted before a
                    # marker (the model narrating its next step) stays
                    # visible, which reads as progress, not duplication.
                    if suppressing:
                        # Withhold from the live stream but keep
                        # accumulating: if this step ends with no parseable
                        # tool call, the text was a legitimate answer that
                        # merely mentions the markup, and the terminal
                        # flush must include it rather than lose it.
                        pending += payload
                        continue
                    pending += payload
                    cut = splittable_prefix(pending)
                    piece, pending = pending[:cut], pending[cut:]
                    if any(
                        pending.startswith(marker)
                        for marker in _HOLDBACK_MARKERS
                    ):
                        suppressing = True
                    if piece:
                        yield TokenChunk(
                            model=ModelId(STEWARD_VIRTUAL_MODEL_ID),
                            text=piece,
                            token_id=-1,
                            usage=None,
                        )
                elif kind == "result":
                    text, tool_calls, error = cast(
                        "tuple[str, list[ToolCall], str | None]", payload
                    )
            if error is not None:
                yield ErrorChunk(
                    model=ModelId(STEWARD_VIRTUAL_MODEL_ID), error_message=error
                )
                return
            if not tool_calls:
                tool_calls = parse_text_tool_calls(text)
            if not tool_calls:
                # Final answer: everything safe was streamed live. When
                # suppression fired but nothing parsed, the withheld text is
                # ambiguous: markup embedded in prose is a literal example
                # inside a real answer and must survive intact, while a turn
                # that is nothing but markup is a malformed tool attempt
                # that must not leak as content. Judge on the FULL turn
                # text: prose before the block has already streamed, so the
                # withheld suffix alone can be markup-only even when the
                # answer has surrounding prose.
                if suppressing and not strip_tool_markup(text).strip():
                    reply = ""
                else:
                    reply = pending
                break
            call = tool_calls[0]
            arguments: dict[str, object] = {}
            try:
                parsed = cast("object", json.loads(call.function.arguments or "{}"))
            except json.JSONDecodeError:
                parsed = {}
            if isinstance(parsed, dict):
                arguments = cast("dict[str, object]", parsed)
            trace_line = call.function.name + (
                f" {json.dumps(arguments)}" if arguments else ""
            )
            yield TokenChunk(
                model=ModelId(STEWARD_VIRTUAL_MODEL_ID),
                text=trace_line + "\n",
                token_id=-1,
                usage=None,
                is_thinking=True,
            )
            result = await self.execute_tool(call.function.name, arguments)
            content_text = strip_tool_markup(text)
            messages.append(
                ChatCompletionMessage(
                    role="assistant",
                    content=content_text or None,
                    tool_calls=[call],
                )
            )
            messages.append(
                ChatCompletionMessage(
                    role="tool", content=result, tool_call_id=call.id
                )
            )
        else:
            reply = (
                "I ran out of investigation budget before reaching a "
                "conclusion this turn."
            )

        yield TokenChunk(
            model=ModelId(STEWARD_VIRTUAL_MODEL_ID),
            text=reply,
            token_id=-1,
            usage=self._turn_usage,
            finish_reason=self._last_finish_reason,
        )

    async def _generate_events(
        self,
        messages: list[ChatCompletionMessage],
        model_id: str,
        instance_id: InstanceId,
    ) -> "AsyncGenerator[tuple[str, object], None]":
        """One completion against the steward instance, as live events.

        Yields ``("text", piece)`` for each non-thinking text piece as it
        arrives (enabling live answer streaming upstream), then exactly one
        ``("result", (full_text, tool_calls, error_message))`` terminal.
        Rides the exact chat-completions dispatch path so the capability
        spine handles the family's prompt rendering and tool-call parsing.
        """
        from skulk.api.adapters.chat_completions import (
            chat_request_to_text_generation,
            usage_from_stats,
        )
        from skulk.shared.types.chunks import (
            ErrorChunk,
            TokenChunk,
            ToolCallChunk,
        )

        api = self._api
        request = ChatCompletionRequest(
            model=model_id,  # type: ignore[arg-type]
            messages=messages,
            tools=steward_tool_definitions(),
            temperature=0.1,
            max_tokens=1024,
            stream=False,
            # Thinking off for every steward brain, by bench measurement
            # (see STEWARD_THINKING_ENABLED). Non-toggleable models ignore
            # this at the capability boundary, so it is safe to send
            # unconditionally.
            enable_thinking=STEWARD_THINKING_ENABLED,
        )
        model_card = await api.running_model_card(request.model)
        task_params = await chat_request_to_text_generation(
            request, model_card=model_card
        )
        command = await api.dispatch_text_generation(
            task_params, target_instance_id=instance_id
        )
        self._active_command_id = command.command_id
        if self._turn_cancelled:
            # cancel_turn ran while this dispatch was in flight: the latch
            # was set before this inner id existed, so cancel_turn had
            # nothing to stop. Honor it here rather than letting an
            # accepted cancellation stream one more full generation. The
            # step ends without a result event, which the loop reads as an
            # empty final answer. No chunk stream ever opens for this
            # command, so the local-finish marker must not be retained.
            self._active_command_id = None
            await api.send_task_cancellation(
                command.command_id, suppress_local_finish=False
            )
            return
        # No extension tap: this is one investigation step, not the turn.
        # The turn's single tap is applied by the caller of
        # run_turn_chunks (API._steward_chat_completions), so observers see
        # the operator's question and the final answer once, rather than
        # every intermediate step and its tool traffic.
        chunk_stream = api.text_generation_chunk_stream(
            command, task_params, extension_tap=False
        )
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        error_message: str | None = None
        async for chunk in chunk_stream:
            match chunk:
                case ErrorChunk():
                    error_message = chunk.error_message or "steward generation failed"
                    break
                case TokenChunk():
                    if not chunk.is_thinking:
                        text_parts.append(chunk.text)
                        if chunk.text:
                            yield ("text", chunk.text)
                    if chunk.finish_reason is not None:
                        self._last_finish_reason = chunk.finish_reason
                    if step_usage := chunk.usage or usage_from_stats(chunk.stats):
                        self._accumulate_usage(step_usage)
                case ToolCallChunk():
                    if step_usage := chunk.usage or usage_from_stats(chunk.stats):
                        self._accumulate_usage(step_usage)
                    tool_calls.extend(
                        ToolCall(id=item.id, index=i, function=item)
                        for i, item in enumerate(chunk.tool_calls)
                    )
                case _:
                    continue
        self._active_command_id = None
        yield ("result", ("".join(text_parts), tool_calls, error_message))
