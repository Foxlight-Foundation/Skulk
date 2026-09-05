from pathlib import Path
from typing import Callable, cast

import pytest
from mlx_lm.tokenizer_utils import TokenizerWrapper

from skulk.shared.models.model_cards import (
    ModelCard,
    ModelId,
    ModelTask,
    ToolCallFormat,
    ToolingCardConfig,
)
from skulk.shared.types.memory import Memory
from skulk.shared.types.worker.shards import PipelineShardMetadata
from skulk.worker.engines.mlx import utils_mlx as utils_mlx_module
from skulk.worker.engines.mlx.utils_mlx import (
    get_tokenizer,
    load_tokenizer_for_model_id,
)


class _FakeTokenizer:
    def __init__(self) -> None:
        self.eos_token_ids: list[int] | None = None
        self._tool_call_start: str | None = None
        self._tool_call_end: str | None = None
        self._tool_parser: Callable[[str], list[dict[str, object]]] | None = None

    @property
    def tool_call_start(self) -> str | None:
        return self._tool_call_start

    @property
    def tool_call_end(self) -> str | None:
        return self._tool_call_end

    @property
    def tool_parser(self) -> Callable[[str], list[dict[str, object]]] | None:
        return self._tool_parser


class _TokenizerBackend:
    def __init__(self) -> None:
        self.eos_token_id = 1
        self.chat_template = None

    def get_vocab(self) -> dict[str, int]:
        return {
            "<|tool_call>": 10,
            "<tool_call|>": 11,
            "<bos>": 0,
        }

    def apply_chat_template(
        self, messages: object, **kwargs: object
    ) -> str:
        """Faithful-fake requirement of mlx-lm 0.32.

        The wrapper's ``_infer_thinking_kwarg`` compares
        ``type(tokenizer).apply_chat_template`` against the transformers
        base method and then inspects its signature; a fake without the
        method reads as a "custom renderer" whose class attribute lookup
        then raises. Every real tokenizer has this method, so the fake
        must too. Rendering is irrelevant to these tests.
        """
        return ""


def _gemma4_tool_parser() -> Callable[[str], list[dict[str, object]]]:
    module_dict = cast(dict[str, object], utils_mlx_module.__dict__)
    return cast(
        Callable[[str], list[dict[str, object]]],
        module_dict["_parse_gemma4_tool_calls"],
    )


def test_load_tokenizer_for_model_id_uses_explicit_model_card_when_cache_is_empty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    card = ModelCard(
        model_id=ModelId("custom/tool-model"),
        storage_size=Memory.from_mb(100),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        capabilities=["text"],
        tooling=ToolingCardConfig(tool_call_format=ToolCallFormat.Gemma4),
    )

    def _missing_card(_model_id: ModelId) -> None:
        return None

    def _fake_load_tokenizer(*_args: object, **_kwargs: object) -> _FakeTokenizer:
        return _FakeTokenizer()

    monkeypatch.setattr(
        "skulk.worker.engines.mlx.utils_mlx.get_card",
        _missing_card,
    )
    monkeypatch.setattr(
        "skulk.worker.engines.mlx.utils_mlx.load_tokenizer",
        _fake_load_tokenizer,
    )

    tokenizer = cast(
        _FakeTokenizer,
        cast(
            object,
            load_tokenizer_for_model_id(
                card.model_id,
                tmp_path,
                model_card=card,
            ),
        ),
    )

    assert tokenizer.tool_call_start == "<|tool_call>"
    assert tokenizer.tool_call_end == "<tool_call|>"
    assert tokenizer.tool_parser is _gemma4_tool_parser()


def test_load_tokenizer_for_model_id_patches_real_tokenizer_wrapper_for_gemma4(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    card = ModelCard(
        model_id=ModelId("mlx-community/gemma-4-26b-a4b-it-4bit"),
        storage_size=Memory.from_mb(100),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        capabilities=["text"],
        family="gemma4",
        tooling=ToolingCardConfig(tool_call_format=ToolCallFormat.Gemma4),
    )

    def _fake_load_tokenizer(*_args: object, **_kwargs: object) -> TokenizerWrapper:
        return TokenizerWrapper(_TokenizerBackend())

    monkeypatch.setattr(
        "skulk.worker.engines.mlx.utils_mlx.load_tokenizer",
        _fake_load_tokenizer,
    )

    tokenizer = load_tokenizer_for_model_id(
        card.model_id,
        tmp_path,
        model_card=card,
    )
    tool_parser = cast(
        Callable[[str], list[dict[str, object]]] | None,
        tokenizer.tool_parser,
    )

    assert tokenizer.tool_call_start == "<|tool_call>"
    assert tokenizer.tool_call_end == "<tool_call|>"
    assert tool_parser is _gemma4_tool_parser()


def test_get_tokenizer_passes_shard_model_card_to_tokenizer_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    card = ModelCard(
        model_id=ModelId("custom/tool-model"),
        storage_size=Memory.from_mb(100),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        capabilities=["text"],
        tooling=ToolingCardConfig(tool_call_format=ToolCallFormat.Gemma4),
    )
    shard_metadata = PipelineShardMetadata(
        model_card=card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=1,
        n_layers=1,
    )

    captured: dict[str, object] = {}

    def _fake_loader(
        model_id: ModelId,
        model_path: Path,
        *,
        model_card: ModelCard | None = None,
        trust_remote_code: bool = False,
    ) -> _FakeTokenizer:
        captured["model_id"] = model_id
        captured["model_path"] = model_path
        captured["model_card"] = model_card
        captured["trust_remote_code"] = trust_remote_code
        return _FakeTokenizer()

    monkeypatch.setattr(
        "skulk.worker.engines.mlx.utils_mlx.load_tokenizer_for_model_id",
        _fake_loader,
    )

    get_tokenizer(tmp_path, shard_metadata)

    assert captured["model_id"] == card.model_id
    assert captured["model_path"] == tmp_path
    assert captured["model_card"] == card
    assert captured["trust_remote_code"] == card.trust_remote_code


class _QwenTemplateBackend(_TokenizerBackend):
    """Backend whose chat template speaks the <tool_call> dialect but that
    arrives without an inferred tool parser (the mlx-vlm vision-path shape,
    #728)."""

    def __init__(self) -> None:
        super().__init__()
        self.chat_template = (
            "{% if tools %}<tool_call>\n<function=example>\n</function>\n"
            "</tool_call>{% endif %}"
        )


def test_generic_format_wires_text_parser_when_template_speaks_tool_call(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """#728: a Generic-format card whose tokenizer lacks an inferred parser
    but whose template uses <tool_call> gets the shared text parser wired, so
    the MLX lane (including vision loaders) emits structured tool calls."""
    card = ModelCard(
        model_id=ModelId("mlx-community/Qwen3.5-4B-MLX-4bit"),
        storage_size=Memory.from_mb(100),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        capabilities=["text"],
        family="qwen",
        tooling=ToolingCardConfig(tool_call_format=ToolCallFormat.Generic),
    )

    def _fake_load_tokenizer(*_args: object, **_kwargs: object) -> TokenizerWrapper:
        return TokenizerWrapper(_QwenTemplateBackend())

    monkeypatch.setattr(
        "skulk.worker.engines.mlx.utils_mlx.load_tokenizer",
        _fake_load_tokenizer,
    )

    tokenizer = load_tokenizer_for_model_id(
        card.model_id,
        tmp_path,
        model_card=card,
    )
    assert tokenizer.tool_call_start == "<tool_call>"
    assert tokenizer.tool_call_end == "</tool_call>"
    parser = cast(
        Callable[[str], list[dict[str, object]]] | None,
        tokenizer.tool_parser,
    )
    assert parser is not None
    parsed = parser(
        "\n<function=get_node_resources>\n<parameter=node_id>\nmac-den\n"
        "</parameter>\n</function>\n"
    )
    assert parsed == [
        {"name": "get_node_resources", "arguments": '{"node_id": "mac-den"}'}
    ]


def test_generic_format_leaves_other_template_dialects_alone(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    card = ModelCard(
        model_id=ModelId("mlx-community/some-other-model"),
        storage_size=Memory.from_mb(100),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        capabilities=["text"],
        family="other",
        tooling=ToolingCardConfig(tool_call_format=ToolCallFormat.Generic),
    )

    def _fake_load_tokenizer(*_args: object, **_kwargs: object) -> TokenizerWrapper:
        return TokenizerWrapper(_TokenizerBackend())

    monkeypatch.setattr(
        "skulk.worker.engines.mlx.utils_mlx.load_tokenizer",
        _fake_load_tokenizer,
    )

    tokenizer = load_tokenizer_for_model_id(
        card.model_id,
        tmp_path,
        model_card=card,
    )
    assert cast(object, tokenizer.tool_parser) is None


def _mistral_tool_parser() -> Callable[[str], list[dict[str, object]]]:
    module_dict = cast(dict[str, object], utils_mlx_module.__dict__)
    return cast(
        Callable[[str], list[dict[str, object]]],
        module_dict["_parse_mistral_tool_calls"],
    )


def test_mistral_template_overrides_the_preinstalled_parser(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A [TOOL_CALLS] template gets the Mistral whole-block parser.

    Production-faithful: the pinned mlx-lm auto-assigns its own Mistral
    parser (which reads the `NAME[ARGS]{...}` form) to any template
    containing [TOOL_CALLS], so the wiring must REPLACE an existing parser,
    not only fill an empty slot. The override reads the JSON-array form the
    2410-era instruct models emit, and delegates back to the displaced
    parser when the array form does not parse, so upstream-form models keep
    working. The end marker is an impossible NUL-containing sentinel (never
    present in detokenized text; an EOS literal like "</s>" can occur inside
    generated arguments and would close the block early); the block closes at
    end of generation like Llama's unmarked dialect.
    """
    card = ModelCard(
        model_id=ModelId("mlx-community/Ministral-8B-Instruct-2410-4bit"),
        storage_size=Memory.from_mb(100),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        capabilities=["text"],
        family="mistral",
        tooling=ToolingCardConfig(tool_call_format=ToolCallFormat.Generic),
    )

    upstream_calls: list[str] = []

    def _upstream_mistral(text: str) -> list[dict[str, object]]:
        upstream_calls.append(text)
        return [{"name": "upstream_form", "arguments": "{}"}]

    def _fake_load_tokenizer(*_args: object, **_kwargs: object) -> TokenizerWrapper:
        backend = _TokenizerBackend()
        backend.chat_template = (  # pyright: ignore[reportAttributeAccessIssue]
            "{% if tools %}[AVAILABLE_TOOLS]{{ tools }}[/AVAILABLE_TOOLS]"
            "{% endif %}[TOOL_CALLS]"
        )
        wrapper = TokenizerWrapper(backend)
        # Replicate mlx-lm's auto-assignment for [TOOL_CALLS] templates.
        object.__setattr__(wrapper, "_tool_parser", _upstream_mistral)
        return wrapper

    monkeypatch.setattr(
        "skulk.worker.engines.mlx.utils_mlx.load_tokenizer",
        _fake_load_tokenizer,
    )

    tokenizer = load_tokenizer_for_model_id(
        card.model_id,
        tmp_path,
        model_card=card,
    )
    assert tokenizer.tool_call_start == "[TOOL_CALLS]"
    end = tokenizer.tool_call_end
    assert end is not None and "\x00" in end
    parser = cast(
        Callable[[str], list[dict[str, object]]],
        cast(object, tokenizer.tool_parser),
    )
    assert parser is not _upstream_mistral

    # The array form the 2410-era models emit is read by our dialect...
    calls = parser(' [{"name": "get_weather", "arguments": {"location": "X"}}]')
    assert [call["name"] for call in calls] == ["get_weather"]
    assert upstream_calls == []

    # ...and a block ours cannot read is delegated to the displaced parser.
    calls = parser("get_weather[ARGS]{}")
    assert [call["name"] for call in calls] == ["upstream_form"]
    assert upstream_calls == ["get_weather[ARGS]{}"]


def test_mistral_parser_reads_a_tool_calls_array() -> None:
    """The callable digests the marker-stripped block via the shared parser."""
    parser = _mistral_tool_parser()
    calls = parser(' [{"name": "get_weather", "arguments": {"location": "Denver"}}]')
    assert [call["name"] for call in calls] == ["get_weather"]
    with pytest.raises(ValueError):
        parser(" this is not a call ")


def test_mistral_tool_call_id_normalization() -> None:
    """UUID-style ids map deterministically onto nine alphanumerics.

    Mistral's template raises on any other shape, which killed the runner at
    the first tool-result round trip; both the assistant's recorded call and
    the echoing tool message must map to the same value.
    """
    module_dict = cast(dict[str, object], utils_mlx_module.__dict__)
    to_id = cast(
        Callable[[str], str],
        module_dict["_mistral_tool_call_id"],
    )
    normalize = cast(
        Callable[[list[dict[str, object]]], None],
        module_dict["_normalize_mistral_tool_call_ids"],
    )

    raw = "99cf7e1d-3708-4063-a647-e27843a8b15f"
    mapped = to_id(raw)
    assert len(mapped) == 9 and mapped.isalnum()
    assert to_id(raw) == mapped
    # Ids differing only past the ninth character must not collide (parallel
    # calls with sequential caller-minted ids).
    assert to_id("call_000000001") != to_id("call_000000002")

    messages: list[dict[str, object]] = [
        {"role": "assistant", "tool_calls": [{"id": raw, "function": {}}]},
        {"role": "tool", "tool_call_id": raw, "content": "68F"},
    ]
    normalize(messages)
    calls = cast(list[dict[str, object]], messages[0]["tool_calls"])
    assert calls[0]["id"] == mapped
    assert messages[1]["tool_call_id"] == mapped
