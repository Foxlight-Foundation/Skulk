# pyright: reportPrivateUsage=false
"""``embed_texts`` without a model id picks one embedding model deterministically.

Vectors from different models live in different spaces. With several embedding
models mounted, the old first-found choice followed the state dict's ordering,
so two calls could embed into different spaces and a caller comparing them got
garbage. The choice must depend only on which models are mounted.
"""

from types import SimpleNamespace
from typing import cast

from skulk.api.main import API
from skulk.shared.models.model_cards import ModelTask


def _instance(model_id: str, tasks: list[ModelTask]) -> SimpleNamespace:
    shard = SimpleNamespace(model_card=SimpleNamespace(tasks=tasks))
    return SimpleNamespace(
        shard_assignments=SimpleNamespace(model_id=model_id, runner_to_shard={"r": shard})
    )


def _choose(instances: list[SimpleNamespace]) -> str | None:
    api = SimpleNamespace(
        state=SimpleNamespace(instances={str(index): item for index, item in enumerate(instances)})
    )
    return API._find_running_embedding_model(cast(API, cast(object, api)))


def test_choice_ignores_instance_ordering() -> None:
    embedders = [
        _instance("sentence-transformers/all-MiniLM-L6-v2", [ModelTask.TextEmbedding]),
        _instance("BAAI/bge-small-en-v1.5", [ModelTask.TextEmbedding]),
    ]
    chat = _instance("mlx-community/Qwen3.5-4B-MLX-4bit", [ModelTask.TextGeneration])
    assert _choose([chat, *embedders]) == "BAAI/bge-small-en-v1.5"
    assert _choose([*reversed(embedders), chat]) == "BAAI/bge-small-en-v1.5"


def test_single_or_no_embedding_model() -> None:
    assert _choose([_instance("a/embed", [ModelTask.TextEmbedding])]) == "a/embed"
    assert _choose([_instance("a/chat", [ModelTask.TextGeneration])]) is None
