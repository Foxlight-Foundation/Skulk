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

import json
import re
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from skulk.api.types.api import (
    ChatCompletionMessage,
    ChatCompletionRequest,
    ToolCall,
    ToolCallItem,
)
from skulk.shared.models.model_cards import ModelId
from skulk.shared.types.chunks import (
    ErrorChunk,
    PrefillProgressChunk,
    TokenChunk,
    ToolCallChunk,
)
from skulk.shared.types.worker.instances import InstanceId

if TYPE_CHECKING:
    from skulk.api.main import API

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

STEWARD_SYSTEM_PROMPT = """\
You are the steward: the resident operator intelligence of this Skulk
cluster. Skulk is a distributed AI inference fabric that runs models across
multiple machines. You answer operator questions by investigating the
cluster through your tools, then reporting clearly.

Rules:
- Investigate before concluding. Start from get_cluster_state unless the
  question clearly points elsewhere.
- Call ONE tool at a time and read its result before deciding the next step.
- Evidence means concrete observed values from tool results, not guesses.
- If everything is healthy, say so; do not invent problems.
- You can only observe and advise. You cannot change the cluster; when
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
            "health reasons, model instances and their placement, and "
            "download records. This is the first thing to look at.",
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
                    "node_id": {
                        "type": "string",
                        "description": "Node to inspect; omit for all nodes.",
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


class StewardStatusResponse(BaseModel):
    """Whether the steward exists and what serves it right now."""

    model_config = ConfigDict(frozen=True, strict=True)

    enabled: bool = Field(
        description="Whether intelligent-fabric mode is enabled in Settings."
    )
    present: bool = Field(
        description="Whether a steward placement currently exists in state."
    )
    steward_model: str | None = Field(
        default=None,
        description="Model id of the steward brain, when present.",
    )
    instance_id: str | None = Field(
        default=None,
        description="The steward instance id, when present.",
    )


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


def _instances_summary(state_payload: dict[str, object]) -> list[dict[str, object]]:
    """Compact instance listing for the steward's context window."""
    summary: list[dict[str, object]] = []
    for instance_id, envelope in _as_object_dict(
        state_payload.get("instances")
    ).items():
        for kind, body_obj in _as_object_dict(envelope).items():
            body = _as_object_dict(body_obj)
            assignments = _as_object_dict(body.get("shardAssignments"))
            summary.append(
                {
                    "instanceId": instance_id,
                    "kind": kind,
                    "modelId": assignments.get("modelId"),
                    "nodes": sorted(_as_object_dict(assignments.get("nodeToRunner"))),
                    "systemRole": body.get("systemRole"),
                }
            )
    return summary


def _downloads_summary(state_payload: dict[str, object]) -> dict[str, object]:
    """Compact download listing keeping error and progress essentials."""
    summary: dict[str, object] = {}
    for node_id, entries in _as_object_dict(state_payload.get("downloads")).items():
        rows: list[dict[str, object]] = []
        for envelope in _as_object_list(entries):
            for kind, body_obj in _as_object_dict(envelope).items():
                body = _as_object_dict(body_obj)
                row: dict[str, object] = {"kind": kind}
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
            summary[node_id] = rows
    return summary


_FUNCTION_BLOCK = re.compile(r"<function=([\w.-]+)>(.*?)</function>", re.DOTALL)
_PARAMETER_BLOCK = re.compile(r"<parameter=([\w.-]+)>\n?(.*?)\n?</parameter>", re.DOTALL)
_TOOL_CALL_BLOCK = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


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
            return _bounded(
                {
                    "nodeHealth": payload.get("nodeHealth", {}),
                    "instances": _instances_summary(payload),
                    "downloads": _downloads_summary(payload),
                    "nodeIdentities": payload.get("nodeIdentities", {}),
                    "nodeDisk": payload.get("nodeDisk", {}),
                    "topologyNodes": _as_object_dict(payload.get("topology")).get(
                        "nodes", []
                    ),
                }
            )
        if name == "get_node_resources":
            payload = await api.get_cluster_state()
            nodes = _as_object_dict(payload.get("nodeResources"))
            node_id = arguments.get("node_id")
            if isinstance(node_id, str) and node_id:
                if node_id not in nodes:
                    return json.dumps(
                        {
                            "error": f"unknown node_id '{node_id}'",
                            "known_nodes": sorted(nodes),
                        }
                    )
                nodes = {node_id: nodes[node_id]}
            return _bounded({"nodes": nodes})
        if name == "get_telemetry_diagnostics":
            report = await api.get_telemetry_plane_diagnostics()
            return _bounded(report.model_dump(by_alias=True, mode="json"))
        if name == "get_data_plane_diagnostics":
            diagnostics = await api.get_node_diagnostics()
            return _bounded(
                diagnostics.data_plane.model_dump(by_alias=True, mode="json")
            )
        if name == "get_cluster_versions":
            cluster = await api.get_cluster_diagnostics()
            return _bounded(
                {
                    "versionStatus": cluster.version_status,
                    "masterNodeId": str(cluster.master_node_id)
                    if cluster.master_node_id is not None
                    else None,
                    "nodes": [
                        {
                            "nodeId": str(node.node_id),
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
            return _bounded(report.model_dump(by_alias=True, mode="json"))
        if name == "run_doctor":
            # The doctor registry runs against the process-cached facts
            # snapshot, so this is a cheap local read, not a fresh probe.
            from skulk.doctor.checks import run_checks
            from skulk.facts import current_node_facts

            results = run_checks(current_node_facts())
            return _bounded(
                {"results": [result.model_dump(mode="json") for result in results]}
            )
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
        self, history: list[StewardChatMessage]
    ) -> "AsyncGenerator[TokenChunk | ErrorChunk | ToolCallChunk | PrefillProgressChunk, None]":
        """Run one steward turn as a chat-completions chunk stream.

        Tool steps stream live as thinking chunks (mapped to
        ``reasoning_content`` by the adapters) while the investigation runs,
        and the final answer follows as a content chunk with
        ``finish_reason="stop"``. Emitting the adapters' native chunk
        vocabulary is what lets the steward ride ``/v1/chat/completions``
        (streaming and non-streaming) with no adapter changes: clients see
        a normal model whose reasoning happens to be its tool trace.
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
            ChatCompletionMessage(role="system", content=STEWARD_SYSTEM_PROMPT)
        ]
        for message in history:
            messages.append(
                ChatCompletionMessage(role=message.role, content=message.content)
            )

        reply = ""
        for step_index in range(MAX_STEPS_PER_TURN):
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
            text, tool_calls, error = await self._generate(
                messages, model_id, instance_id
            )
            if error is not None:
                yield ErrorChunk(
                    model=ModelId(STEWARD_VIRTUAL_MODEL_ID), error_message=error
                )
                return
            if not tool_calls:
                tool_calls = parse_text_tool_calls(text)
                if tool_calls:
                    text = strip_tool_markup(text)
            if not tool_calls:
                reply = text.strip()
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
                token_id=0,
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
            token_id=0,
            usage=None,
            finish_reason="stop",
        )

    async def _generate(
        self,
        messages: list[ChatCompletionMessage],
        model_id: str,
        instance_id: InstanceId,
    ) -> tuple[str, list[ToolCall], str | None]:
        """One completion against the steward instance.

        Returns (text, tool_calls, error_message). Rides the exact
        chat-completions dispatch path so the capability spine handles the
        family's prompt rendering and tool-call parsing.
        """
        from skulk.api.adapters.chat_completions import (
            chat_request_to_text_generation,
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
        )
        model_card = await api.running_model_card(request.model)
        task_params = await chat_request_to_text_generation(
            request, model_card=model_card
        )
        command = await api.dispatch_text_generation(
            task_params, target_instance_id=instance_id
        )
        chunk_stream = api.text_generation_chunk_stream(command, task_params)
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
                case ToolCallChunk():
                    tool_calls.extend(
                        ToolCall(id=item.id, index=i, function=item)
                        for i, item in enumerate(chunk.tool_calls)
                    )
                case _:
                    continue
        return "".join(text_parts), tool_calls, error_message
