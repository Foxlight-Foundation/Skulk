"""The engine-package chat dialect's limits, as the API and the runner both read them."""

import pytest

from skulk.engines.dialect import (
    engine_package_refusal,
    served_only_by_engine_packages,
    unsupported_request_reason,
)
from skulk.shared.models.model_cards import ModelCard, ModelTask
from skulk.shared.types.common import ModelId, NodeId
from skulk.shared.types.memory import Memory
from skulk.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from skulk.shared.types.worker.instances import InstanceId, MlxRingInstance
from skulk.shared.types.worker.runners import RunnerId, ShardAssignments
from skulk.shared.types.worker.shards import PipelineShardMetadata

_MODEL = ModelId("org/chat")
_TOOL = {"type": "function", "function": {"name": "get_time", "parameters": {}}}


def _params(**fields: object) -> TextGenerationTaskParams:
    return TextGenerationTaskParams.model_validate(
        {
            "model": _MODEL,
            "input": [InputMessage(role="user", content="hello")],
            **fields,
        }
    )


def _instance(name: str, backend: str | None) -> MlxRingInstance:
    """Build a single-rank instance whose shard is stamped with ``backend``."""
    card = ModelCard(
        model_id=_MODEL,
        storage_size=Memory.from_bytes(1024),
        n_layers=1,
        hidden_size=16,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
    )
    runner = RunnerId(name)
    return MlxRingInstance(
        instance_id=InstanceId(name),
        shard_assignments=ShardAssignments(
            model_id=_MODEL,
            node_to_runner={NodeId(name): runner},
            runner_to_shard={
                runner: PipelineShardMetadata(
                    model_card=card,
                    device_rank=0,
                    world_size=1,
                    start_layer=0,
                    end_layer=1,
                    n_layers=1,
                    resolved_backend=backend,
                )
            },
        ),
        hosts_by_node={NodeId(name): []},
        ephemeral_port=52416,
    )


@pytest.mark.parametrize(
    ("fields", "word"),
    [
        ({"tools": [_TOOL]}, "tool calls"),
        ({"images": ["aGVsbG8="]}, "images"),
        ({"image_count": 1}, "images"),
        ({"logprobs": True}, "logprobs"),
        ({"top_logprobs": 3}, "logprobs"),
    ],
)
def test_each_request_kind_outside_the_dialect_is_named(
    fields: dict[str, object], word: str
) -> None:
    reason = unsupported_request_reason(_params(**fields))
    assert reason is not None and word in reason


def test_a_plain_text_request_is_inside_the_dialect() -> None:
    assert unsupported_request_reason(_params()) is None


@pytest.mark.parametrize(
    ("backends", "only_packages"),
    [
        ([], False),
        (["tensorfold-rocm"], True),
        (["tensorfold-rocm", "tensorfold-cuda"], True),
        (["mlx-metal"], False),
        (["llama_server-vulkan"], False),
        (["tensorfold-rocm", "mlx-metal"], False),
        ([None], False),
    ],
)
def test_only_instances_wholly_on_packages_count(
    backends: list[str | None], only_packages: bool
) -> None:
    instances = [_instance(f"i{n}", backend) for n, backend in enumerate(backends)]
    assert served_only_by_engine_packages(instances) is only_packages


def test_a_refusal_needs_both_an_unsupported_request_and_only_packages() -> None:
    package = [_instance("package", "tensorfold-rocm")]
    built_in = [_instance("built-in", "mlx-metal")]
    tools = _params(tools=[_TOOL])
    reason = engine_package_refusal(package, tools)
    assert reason is not None and "tool calls" in reason
    assert engine_package_refusal(package, _params()) is None
    assert engine_package_refusal(built_in, tools) is None
    assert engine_package_refusal(package + built_in, tools) is None
    assert engine_package_refusal([], tools) is None
