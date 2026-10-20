"""Exercise relay only through the public clone-and-object-patch boundary."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Barrier
from typing import cast

import pytest
import torch
from skulk_comfy_ltx.relay_binding import patch_prompt_relay
from skulk_comfy_ltx.relay_encoding import RELAY_CONDITION_KEY, RelayLayout
from torch import Tensor

LAYOUT = RelayLayout(((1, 3), (3, 5)), (1, 1))


@dataclass(frozen=True)
class _Constant:
    value: object


class _Patcher:
    def __init__(self, objects: dict[str, object], patches: dict[str, object] | None = None) -> None:
        self.objects = objects
        self.patches = {} if patches is None else patches

    def clone(self) -> _Patcher:
        return _Patcher(self.objects, self.patches.copy())

    def get_model_object(self, name: str) -> object:
        return self.patches.get(name, self.objects.get(name))

    def add_object_patch(self, name: str, value: object) -> None:
        self.patches[name] = value


def _fixture(**methods: Callable[..., object]) -> tuple[_Patcher, _Patcher]:
    def original(*_args: object, **_kwargs: object) -> object:
        return None

    names = ["extra_conds", "diffusion_model._forward", "diffusion_model._process_input", *[f"diffusion_model.transformer_blocks.{index}.attn2.forward" for index in range(2)]]
    objects: dict[str, object] = dict.fromkeys(names, original)
    objects.update({"diffusion_model.vae_scale_factors": (8, 32, 32), "diffusion_model.patchifier.patch_size": (1, 1, 1), "diffusion_model.causal_temporal_positioning": True, "diffusion_model.transformer_blocks": [object(), object()]})
    objects.update(methods)
    model = _Patcher(objects)
    return model, cast(_Patcher, patch_prompt_relay(model, constant_factory=_Constant))


def _method(model: _Patcher, name: str) -> Callable[..., object]:
    value = model.get_model_object(name)
    assert callable(value)
    return value


def _processed() -> object:
    coordinates = torch.zeros(1, 3, 4, 2)
    coordinates[:, 0, :, 0] = torch.tensor([0, 9, 17, 0])
    return ([torch.zeros(1, 4, 2), torch.zeros(1, 2, 2)], [coordinates, torch.zeros(1, 1)], {"num_guide_tokens": 1})


def _prepare(model: _Patcher) -> object:
    return _method(model, "diffusion_model._process_input")([torch.zeros(1, 1, 5, 1, 1), torch.zeros(1, 1)], torch.zeros(1, 3, 1, 2), None)


def test_graph_clones_share_weights_without_installing_live_methods() -> None:
    """A second graph stage gets independent patches without mutating the source."""
    source, first = _fixture()
    second = cast(_Patcher, patch_prompt_relay(first, constant_factory=_Constant))
    assert not source.patches
    assert first.objects is source.objects is second.objects
    assert first.patches is not second.patches
    assert first.patches == second.patches
    assert len(first.patches) == 5


def test_positive_constant_keeps_connector_once_and_negative_dictionary_unchanged() -> None:
    """Only positive conditioning acquires a constant; connector metadata survives."""
    calls: list[dict[str, object]] = []
    original_result: dict[str, object] = {"c_crossattn": object(), "audio": object()}

    def conditions(**kwargs: object) -> object:
        calls.append(kwargs)
        return original_result

    _, model = _fixture(extra_conds=conditions)
    patched = _method(model, "extra_conds")
    positive = patched(**{RELAY_CONDITION_KEY: LAYOUT})
    assert isinstance(positive, dict)
    assert positive[RELAY_CONDITION_KEY] == _Constant(LAYOUT)
    assert positive["c_crossattn"] is original_result["c_crossattn"]
    assert RELAY_CONDITION_KEY not in original_result
    assert patched() is original_result
    assert len(calls) == 2


@pytest.mark.parametrize("failure", [RuntimeError("sampler failed"), KeyboardInterrupt("cancelled")])
def test_failed_or_cancelled_forward_clears_hook_and_can_recover(failure: BaseException) -> None:
    """Cancellation cannot leak a job's temporal routing into a later attention call."""
    fail = True
    observed: list[bool] = []
    processed = _processed()

    def process(*_args: object, **_kwargs: object) -> object:
        return processed

    def attention(x: Tensor, **kwargs: object) -> Tensor:
        options = kwargs["transformer_options"]
        assert isinstance(options, dict)
        observed.append("optimized_attention_override" in options)
        return x

    def forward(**_kwargs: object) -> object:
        assert _prepare(model) is processed
        if fail:
            raise failure
        return call_attention(torch.zeros(1, 4, 8), context=torch.zeros(1, 5, 8))

    _, model = _fixture(**{"diffusion_model._forward": forward, "diffusion_model._process_input": process, "diffusion_model.transformer_blocks.0.attn2.forward": attention})
    call_attention = _method(model, "diffusion_model.transformer_blocks.0.attn2.forward")
    call_forward = _method(model, "diffusion_model._forward")
    with pytest.raises(type(failure)):
        call_forward(**{RELAY_CONDITION_KEY: LAYOUT})
    call_attention(torch.zeros(1, 4, 8), context=torch.zeros(1, 5, 8))
    fail = False
    call_forward(**{RELAY_CONDITION_KEY: LAYOUT})
    call_attention(torch.zeros(1, 4, 8), context=torch.zeros(1, 5, 8))
    assert observed == [False, True, False]


def test_nested_negative_pass_restores_only_its_own_positive_caller() -> None:
    """An unmarked nested CFG pass temporarily suspends positive scene routing."""
    observed: list[bool] = []

    def process(*_args: object, **_kwargs: object) -> object:
        return _processed()

    def attention(x: Tensor, **kwargs: object) -> Tensor:
        options = kwargs["transformer_options"]
        assert isinstance(options, dict)
        observed.append("optimized_attention_override" in options)
        return x

    def forward(**kwargs: object) -> object:
        if RELAY_CONDITION_KEY in kwargs:
            _prepare(model)
            call_attention(torch.zeros(1, 4, 8), context=torch.zeros(1, 5, 8))
            call_forward()
        return call_attention(torch.zeros(1, 4, 8), context=torch.zeros(1, 5, 8))

    _, model = _fixture(**{"diffusion_model._forward": forward, "diffusion_model._process_input": process, "diffusion_model.transformer_blocks.0.attn2.forward": attention})
    call_attention = _method(model, "diffusion_model.transformer_blocks.0.attn2.forward")
    call_forward = _method(model, "diffusion_model._forward")
    call_forward(**{RELAY_CONDITION_KEY: LAYOUT})
    assert observed == [True, False, True]


def test_concurrent_positive_and_negative_calls_do_not_share_scope() -> None:
    """The shared clone's methods can run distinct forwards in separate contexts."""
    barrier = Barrier(2)

    def process(*_args: object, **_kwargs: object) -> object:
        return _processed()

    def attention(x: Tensor, **kwargs: object) -> Tensor:
        options = kwargs["transformer_options"]
        assert isinstance(options, dict)
        return x + int("optimized_attention_override" in options)

    def forward(**kwargs: object) -> object:
        if RELAY_CONDITION_KEY in kwargs:
            _prepare(model)
        _ = barrier.wait(timeout=10)
        return call_attention(torch.zeros(1, 4, 8), context=torch.zeros(1, 5, 8))

    _, model = _fixture(**{"diffusion_model._forward": forward, "diffusion_model._process_input": process, "diffusion_model.transformer_blocks.0.attn2.forward": attention})
    call_attention = _method(model, "diffusion_model.transformer_blocks.0.attn2.forward")
    call_forward = _method(model, "diffusion_model._forward")

    def job(index: int) -> object:
        return call_forward(**({RELAY_CONDITION_KEY: LAYOUT} if index % 2 == 0 else {}))

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(job, range(4)))
    for index, result in enumerate(results):
        assert isinstance(result, Tensor)
        torch.testing.assert_close(result, torch.full_like(result, int(index % 2 == 0)))


def test_unmarked_and_self_attention_delegate_without_mutating_options() -> None:
    """Video-only relay leaves self attention and unmarked conditioning untouched."""
    calls: list[dict[str, object]] = []

    def attention(x: Tensor, **kwargs: object) -> Tensor:
        calls.append(kwargs)
        return x

    def forward(**_kwargs: object) -> object:
        return call_attention(values, transformer_options=options)

    _, model = _fixture(**{"diffusion_model._forward": forward, "diffusion_model.transformer_blocks.0.attn2.forward": attention})
    call_attention = _method(model, "diffusion_model.transformer_blocks.0.attn2.forward")
    options: dict[str, object] = {"sentinel": object()}
    values = torch.ones(1, 3, 4)
    assert call_attention(values, context=values, transformer_options=options) is values
    assert _method(model, "diffusion_model._forward")(**{RELAY_CONDITION_KEY: LAYOUT}) is values
    assert all(call["transformer_options"] is options for call in calls)
    assert set(options) == {"sentinel"}


def test_active_cross_attention_requires_prepared_geometry() -> None:
    """An incompatible Comfy forward fails instead of silently ignoring scene timing."""
    def forward(**_kwargs: object) -> object:
        return call_attention(torch.zeros(1, 2, 4), context=torch.zeros(1, 3, 4))

    _, model = _fixture(**{"diffusion_model._forward": forward})
    call_attention = _method(model, "diffusion_model.transformer_blocks.0.attn2.forward")
    with pytest.raises(ValueError, match="actual video grid"):
        _method(model, "diffusion_model._forward")(**{RELAY_CONDITION_KEY: LAYOUT})
