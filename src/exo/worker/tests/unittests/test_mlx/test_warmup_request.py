from typing import cast

import pytest

import exo.worker.engines.mlx.generator.generate as generate_mod
from exo.shared.types.common import ModelId
from exo.shared.types.text_generation import TextGenerationTaskParams


class _FakeGroup:
    def size(self) -> int:
        return 3


class _SingleNodeGroup:
    def size(self) -> int:
        return 1


def test_warmup_inference_uses_safe_default_user_content_without_instructions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_apply_chat_template(*, tokenizer: object, task_params: object, model_card: object):
        del tokenizer, model_card
        captured["task_params"] = task_params
        return "warmup prompt"

    def fake_mx_barrier(_group: object) -> None:
        return None

    def fake_mlx_generate(**_kwargs: object):
        if False:
            yield None
        return

    def fake_all_gather(array: object, *, group: object):
        del group
        return array

    def fake_log_request_shape(
        label: str,
        task_params: object,
        prompt: str,
        *,
        extra: object = None,
    ) -> None:
        captured["label"] = label
        captured["logged_task_params"] = task_params
        captured["logged_prompt"] = prompt
        captured["logged_extra"] = extra

    monkeypatch.setattr(generate_mod, "apply_chat_template", fake_apply_chat_template)
    monkeypatch.setattr(generate_mod, "mx_barrier", fake_mx_barrier)
    monkeypatch.setattr(generate_mod, "mlx_generate", fake_mlx_generate)
    monkeypatch.setattr(generate_mod, "log_request_shape", fake_log_request_shape)
    monkeypatch.setattr(generate_mod.mx.distributed, "all_gather", fake_all_gather)

    check_every = generate_mod.warmup_inference(
        model=object(),  # type: ignore[arg-type]
        tokenizer=object(),  # type: ignore[arg-type]
        group=cast(object, _FakeGroup()),  # type: ignore[arg-type]
        model_id=ModelId("mlx-community/gemma-4-26b-a4b-it-4bit"),
        model_card=None,
    )

    task_params = cast(TextGenerationTaskParams, captured["task_params"])
    assert check_every == 0
    assert task_params.instructions is None
    assert task_params.enable_thinking is False
    assert task_params.temperature == 1.0
    assert task_params.top_p == 0.95
    assert task_params.top_k == 64
    assert task_params.max_output_tokens == 1024
    first_message = task_params.input[0]
    assert first_message.content == "hello"
    assert captured["label"] == "warmup"
    assert captured["logged_task_params"] == task_params
    assert captured["logged_prompt"] == "warmup prompt"
    assert captured["logged_extra"] == {
        "group_size": 3,
        "model_id": "mlx-community/gemma-4-26b-a4b-it-4bit",
    }


def test_warmup_inference_ignores_repeat_count_override_for_pipeline_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_apply_chat_template(*, tokenizer: object, task_params: object, model_card: object):
        del tokenizer, model_card
        captured["task_params"] = task_params
        return "warmup prompt"

    def fake_mx_barrier(_group: object) -> None:
        return None

    def fake_mlx_generate(**_kwargs: object):
        if False:
            yield None
        return

    def fake_all_gather(array: object, *, group: object):
        del group
        return array

    monkeypatch.setenv("SKULK_DEBUG_WARMUP_REPEAT_COUNT", "4")
    monkeypatch.setattr(generate_mod, "apply_chat_template", fake_apply_chat_template)
    monkeypatch.setattr(generate_mod, "mx_barrier", fake_mx_barrier)
    monkeypatch.setattr(generate_mod, "mlx_generate", fake_mlx_generate)
    monkeypatch.setattr(generate_mod.mx.distributed, "all_gather", fake_all_gather)

    generate_mod.warmup_inference(
        model=object(),  # type: ignore[arg-type]
        tokenizer=object(),  # type: ignore[arg-type]
        group=cast(object, _FakeGroup()),  # type: ignore[arg-type]
        model_id=ModelId("mlx-community/gemma-4-26b-a4b-it-4bit"),
        model_card=None,
    )

    task_params = cast(TextGenerationTaskParams, captured["task_params"])
    assert task_params.input[0].content == "hello"


def test_warmup_inference_ignores_instruction_override_for_pipeline_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_apply_chat_template(*, tokenizer: object, task_params: object, model_card: object):
        del tokenizer, model_card
        captured["task_params"] = task_params
        return "warmup prompt"

    def fake_mx_barrier(_group: object) -> None:
        return None

    def fake_mlx_generate(**_kwargs: object):
        if False:
            yield None
        return

    def fake_all_gather(array: object, *, group: object):
        del group
        return array

    monkeypatch.setenv("SKULK_DEBUG_WARMUP_INCLUDE_INSTRUCTIONS", "1")
    monkeypatch.setattr(generate_mod, "apply_chat_template", fake_apply_chat_template)
    monkeypatch.setattr(generate_mod, "mx_barrier", fake_mx_barrier)
    monkeypatch.setattr(generate_mod, "mlx_generate", fake_mlx_generate)
    monkeypatch.setattr(generate_mod.mx.distributed, "all_gather", fake_all_gather)

    generate_mod.warmup_inference(
        model=object(),  # type: ignore[arg-type]
        tokenizer=object(),  # type: ignore[arg-type]
        group=cast(object, _FakeGroup()),  # type: ignore[arg-type]
        model_id=ModelId("mlx-community/gemma-4-26b-a4b-it-4bit"),
        model_card=None,
    )

    task_params = cast(TextGenerationTaskParams, captured["task_params"])
    assert task_params.instructions is None


def test_warmup_inference_honors_repeat_and_instruction_overrides_for_single_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_apply_chat_template(*, tokenizer: object, task_params: object, model_card: object):
        del tokenizer, model_card
        captured["task_params"] = task_params
        return "warmup prompt"

    def fake_mx_barrier(_group: object) -> None:
        return None

    def fake_mlx_generate(**_kwargs: object):
        if False:
            yield None
        return

    def fake_all_gather(array: object, *, group: object):
        del group
        return array

    monkeypatch.setenv("SKULK_DEBUG_WARMUP_REPEAT_COUNT", "4")
    monkeypatch.setenv("SKULK_DEBUG_WARMUP_INCLUDE_INSTRUCTIONS", "1")
    monkeypatch.setattr(generate_mod, "apply_chat_template", fake_apply_chat_template)
    monkeypatch.setattr(generate_mod, "mx_barrier", fake_mx_barrier)
    monkeypatch.setattr(generate_mod, "mlx_generate", fake_mlx_generate)
    monkeypatch.setattr(generate_mod.mx.distributed, "all_gather", fake_all_gather)

    generate_mod.warmup_inference(
        model=object(),  # type: ignore[arg-type]
        tokenizer=object(),  # type: ignore[arg-type]
        group=cast(object, _SingleNodeGroup()),  # type: ignore[arg-type]
        model_id=ModelId("mlx-community/gemma-4-26b-a4b-it-4bit"),
        model_card=None,
    )

    task_params = cast(TextGenerationTaskParams, captured["task_params"])
    assert task_params.input[0].content == "hello hello hello hello"
    assert (
        task_params.instructions
        == "You are a helpful assistant. Answer the user in one short sentence."
    )


def test_warmup_inference_stops_after_first_generated_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated_tokens = 0

    def fake_apply_chat_template(*, tokenizer: object, task_params: object, model_card: object):
        del tokenizer, task_params, model_card
        return "warmup prompt"

    def fake_mx_barrier(_group: object) -> None:
        return None

    def fake_mlx_generate(**_kwargs: object):
        nonlocal generated_tokens
        for token in ("first", "second", "third"):
            generated_tokens += 1
            yield token

    def fake_all_gather(array: object, *, group: object):
        del group
        return array

    monkeypatch.setattr(generate_mod, "apply_chat_template", fake_apply_chat_template)
    monkeypatch.setattr(generate_mod, "mx_barrier", fake_mx_barrier)
    monkeypatch.setattr(generate_mod, "mlx_generate", fake_mlx_generate)
    monkeypatch.setattr(generate_mod.mx.distributed, "all_gather", fake_all_gather)

    check_every = generate_mod.warmup_inference(
        model=object(),  # type: ignore[arg-type]
        tokenizer=object(),  # type: ignore[arg-type]
        group=cast(object, _SingleNodeGroup()),  # type: ignore[arg-type]
        model_id=ModelId("mlx-community/gemma-4-26b-a4b-it-4bit"),
        model_card=None,
    )

    assert generated_tokens == 1
    assert check_every == 100
