from typing import cast

from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.models.model_cards import ModelCard, ModelTask
from exo.shared.types.common import ModelId
from exo.shared.types.memory import Memory
from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from exo.worker.engines.mlx.utils_mlx import apply_chat_template


class _FakeTokenizer:
    def __init__(self) -> None:
        self.messages: list[dict[str, object]] | None = None
        self.kwargs: dict[str, object] | None = None
        self.chat_template = "{{ messages }}"

    def apply_chat_template(
        self,
        messages: list[dict[str, object]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        tools: list[dict[str, object]] | None = None,
        **kwargs: object,
    ) -> str:
        self.messages = messages
        self.kwargs = {
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
            "tools": tools,
            **kwargs,
        }
        return "rendered prompt"


def _nemotron_card(model_id: str) -> ModelCard:
    return ModelCard(
        model_id=ModelId(model_id),
        storage_size=Memory.from_mb(100),
        n_layers=10,
        hidden_size=1024,
        supports_tensor=True,
        tasks=[ModelTask.TextGeneration],
        family="nemotron",
        capabilities=["text", "thinking", "thinking_toggle"],
    )


def test_apply_chat_template_injects_no_think_system_control_for_nemotron() -> None:
    tokenizer = _FakeTokenizer()
    task = TextGenerationTaskParams(
        model=ModelId("mlx-community/NVIDIA-Nemotron-Nano-9B-v2-4bits"),
        input=[InputMessage(role="user", content="How are you?")],
        enable_thinking=False,
    )

    prompt = apply_chat_template(
        cast(TokenizerWrapper, cast(object, tokenizer)),
        task,
        model_card=_nemotron_card("mlx-community/NVIDIA-Nemotron-Nano-9B-v2-4bits"),
    )

    assert prompt == "rendered prompt"
    assert tokenizer.messages == [
        {"role": "system", "content": "/no_think"},
        {"role": "user", "content": "How are you?"},
    ]
    assert tokenizer.kwargs is not None
    assert tokenizer.kwargs["enable_thinking"] is False
    assert tokenizer.kwargs["thinking"] is False


def test_apply_chat_template_prefixes_existing_system_message_for_nemotron_think() -> None:
    tokenizer = _FakeTokenizer()
    task = TextGenerationTaskParams(
        model=ModelId("mlx-community/NVIDIA-Nemotron-Nano-9B-v2-4bits"),
        input=[InputMessage(role="user", content="Hello")],
        instructions="You are helpful.",
        enable_thinking=True,
    )

    apply_chat_template(
        cast(TokenizerWrapper, cast(object, tokenizer)),
        task,
        model_card=_nemotron_card("mlx-community/NVIDIA-Nemotron-Nano-9B-v2-4bits"),
    )

    assert tokenizer.messages == [
        {"role": "system", "content": "/think\nYou are helpful."},
        {"role": "user", "content": "Hello"},
    ]


def test_apply_chat_template_does_not_duplicate_existing_nemotron_control() -> None:
    tokenizer = _FakeTokenizer()
    task = TextGenerationTaskParams(
        model=ModelId("mlx-community/NVIDIA-Nemotron-Nano-9B-v2-4bits"),
        input=[InputMessage(role="user", content="Hello")],
        instructions="/no_think\nYou are helpful.",
        enable_thinking=False,
    )

    apply_chat_template(
        cast(TokenizerWrapper, cast(object, tokenizer)),
        task,
        model_card=_nemotron_card("mlx-community/NVIDIA-Nemotron-Nano-9B-v2-4bits"),
    )

    assert tokenizer.messages == [
        {"role": "system", "content": "/no_think\nYou are helpful."},
        {"role": "user", "content": "Hello"},
    ]
