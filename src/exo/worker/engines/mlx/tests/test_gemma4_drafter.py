# pyright: reportPrivateUsage=false, reportUnknownLambdaType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportArgumentType=false, reportUnknownMemberType=false
# pyright: reportUnknownParameterType=false, reportMissingParameterType=false
"""Unit tests for the Gemma 4 assistant drafter (gemma4-mtp Phase C)."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
from mlx_lm.models.cache import KVCache, RotatingKVCache

from exo.worker.engines.mlx.drafters import build_drafter
from exo.worker.engines.mlx.drafters.gemma4_assistant import (
    Gemma4AssistantDrafter,
    _extract_shared_kv,
    build_gemma4_assistant_drafter,
    load_assistant_model,
)

HIDDEN = 8
VOCAB = 16


class _FakeLayer:
    def __init__(self, layer_type: str) -> None:
        self.layer_type = layer_type


def _kv_cache_with(n: int, fill: float) -> KVCache:
    cache = KVCache()
    cache.update_and_fetch(
        mx.full((1, 1, n, 4), fill), mx.full((1, 1, n, 4), fill)
    )
    return cache


class TestExtractSharedKV:
    def test_last_layer_of_each_type_wins(self) -> None:
        layers = [
            _FakeLayer("sliding_attention"),
            _FakeLayer("full_attention"),
            _FakeLayer("sliding_attention"),
            _FakeLayer("full_attention"),
        ]
        caches = [
            _kv_cache_with(3, 1.0),
            _kv_cache_with(3, 2.0),
            _kv_cache_with(3, 3.0),
            _kv_cache_with(3, 4.0),
        ]
        shared = _extract_shared_kv(layers, caches)
        assert set(shared) == {"sliding_attention", "full_attention"}
        # Last sliding layer carried fill=3.0; last full carried fill=4.0.
        assert float(shared["sliding_attention"][0][0, 0, 0, 0].item()) == 3.0
        assert float(shared["full_attention"][0][0, 0, 0, 0].item()) == 4.0

    def test_skips_layers_without_type_or_state(self) -> None:
        class _Untyped:
            pass

        layers = [_Untyped(), _FakeLayer("full_attention")]
        caches = [_kv_cache_with(2, 1.0), KVCache()]  # second has empty state
        shared = _extract_shared_kv(layers, caches)
        assert shared == {}

    def test_rotating_cache_restored_to_temporal_order(self) -> None:
        layer = _FakeLayer("sliding_attention")
        cache = RotatingKVCache(max_size=4)
        # Overfill so the ring buffer wraps; values encode insertion order.
        for i in range(6):
            cache.update_and_fetch(
                mx.full((1, 1, 1, 4), float(i)), mx.full((1, 1, 1, 4), float(i))
            )
        shared = _extract_shared_kv([layer], [cache])
        keys = shared["sliding_attention"][0]
        flat = [float(keys[0, 0, i, 0].item()) for i in range(keys.shape[2])]
        # Temporal order: strictly non-decreasing insertion stamps.
        assert flat == sorted(flat), f"not temporally ordered: {flat}"


class _FakeAssistant:
    """Protocol-satisfying fake recording every adapter interaction."""

    def __init__(self, vocab: int = VOCAB) -> None:
        self._vocab = vocab
        self.reset_calls = 0
        self.bind_calls = 0
        self.set_shared_kv_calls: list[tuple[set[str], int]] = []
        self.call_positions: list[int] = []
        self._input_embed = lambda toks: mx.ones((1, 1, HIDDEN))
        self._input_embed_scale = 2.0

    def bind(self, target_model: object) -> "_FakeAssistant":
        self.bind_calls += 1
        return self

    def reset(self, target_model: object) -> list[object]:
        self.reset_calls += 1
        return []

    def set_shared_kv(self, shared_kv_states: dict[str, tuple[mx.array, mx.array]], kv_offset: int) -> None:
        self.set_shared_kv_calls.append((set(shared_kv_states), kv_offset))

    def __call__(
        self,
        inputs_embeds: mx.array,
        shared_kv_states: dict[str, tuple[mx.array, mx.array]],
        position_ids: mx.array,
    ) -> tuple[mx.array, mx.array]:
        self.call_positions.append(int(position_ids[0, 0].item()))
        # Draft token = current call count, encoded as one-hot logits.
        step = len(self.call_positions)
        logits = mx.where(
            mx.arange(self._vocab)[None, None, :] == step,
            mx.array(100.0),
            mx.zeros((1, 1, self._vocab)),
        )
        return mx.ones((1, 1, HIDDEN)), logits

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        return weights

    def load_weights(
        self, weights: list[tuple[str, mx.array]], strict: bool = True
    ) -> object:
        return self

    def parameters(self) -> dict[str, object]:
        return {}


class _FakeGemmaTrunk:
    def __init__(self) -> None:
        self.layers = [_FakeLayer("sliding_attention"), _FakeLayer("full_attention")]


class _FakeGemmaModel:
    def __init__(self) -> None:
        self.language_model = type("LM", (), {"model": _FakeGemmaTrunk()})()


class TestGemma4AssistantDrafter:
    def _drafter(self) -> tuple[Gemma4AssistantDrafter, _FakeAssistant, list[object]]:
        assistant = _FakeAssistant()
        model = _FakeGemmaModel()
        drafter = Gemma4AssistantDrafter(assistant=assistant, target_model=model)
        caches: list[object] = [_kv_cache_with(5, 1.0), _kv_cache_with(5, 2.0)]
        drafter.begin_request(caches)
        return drafter, assistant, caches

    def test_begin_request_resets_assistant(self) -> None:
        _, assistant, _ = self._drafter()
        assert assistant.reset_calls == 1

    def test_draft_chains_to_depth_with_constant_position(self) -> None:
        drafter, assistant, _ = self._drafter()
        rows = drafter.draft(mx.zeros(HIDDEN), next_token=7, depth=3)
        assert rows.shape == (3, VOCAB)
        assert rows.dtype == mx.float32
        # Position held constant at the cache offset (5) across all steps.
        assert assistant.call_positions == [5, 5, 5]
        # Shared KV carried both layer types; kv_offset = cache offset.
        assert assistant.set_shared_kv_calls == [
            ({"sliding_attention", "full_attention"}, 5)
        ]

    def test_observe_is_noop(self) -> None:
        drafter, _, _ = self._drafter()
        drafter.observe(mx.zeros((3, HIDDEN)), mx.array([1, 2, 3]))  # must not raise

    def test_kv_shared_prefix_mapping(self) -> None:
        """Fewer caches than layers = KV-shared model: caches map to the
        layer prefix (gemma4 make_cache builds layers[:first_kv_shared])."""
        assistant = _FakeAssistant()
        drafter = Gemma4AssistantDrafter(
            assistant=assistant, target_model=_FakeGemmaModel()
        )
        # 1 cache vs 2 layers -> only the first (sliding) layer participates.
        drafter.begin_request([_kv_cache_with(2, 1.0)])
        drafter.draft(mx.zeros(HIDDEN), next_token=1)
        assert assistant.set_shared_kv_calls == [({"sliding_attention"}, 2)]

    def test_more_caches_than_layers_raises(self) -> None:
        assistant = _FakeAssistant()
        drafter = Gemma4AssistantDrafter(
            assistant=assistant, target_model=_FakeGemmaModel()
        )
        drafter.begin_request(
            [_kv_cache_with(2, 1.0), _kv_cache_with(2, 1.0), _kv_cache_with(2, 1.0)]
        )
        try:
            drafter.draft(mx.zeros(HIDDEN), next_token=1)
        except RuntimeError as error:
            assert "more caches" in str(error)
        else:
            raise AssertionError("expected RuntimeError on cache surplus")


class TestBuilderDispatch:
    def test_assistant_takes_precedence(self) -> None:
        drafter = build_drafter(
            _FakeGemmaModel(), None, assistant_model=_FakeAssistant()
        )
        assert isinstance(drafter, Gemma4AssistantDrafter)

    def test_non_gemma_trunk_returns_none(self) -> None:
        class _NoTypeLayerTrunk:
            layers = [object()]

        class _Model:
            language_model = type("LM", (), {"model": _NoTypeLayerTrunk()})()

        drafter = build_gemma4_assistant_drafter(_Model(), _FakeAssistant())
        assert drafter is None

    def test_bind_failure_returns_none(self) -> None:
        class _ExplodingAssistant(_FakeAssistant):
            def bind(self, target_model: object) -> "_FakeAssistant":
                raise ValueError("no embeddings")

        drafter = build_gemma4_assistant_drafter(
            _FakeGemmaModel(), _ExplodingAssistant()
        )
        assert drafter is None


class TestLoadAssistantModel:
    def test_missing_files_returns_none(self, tmp_path: Path) -> None:
        assert load_assistant_model(tmp_path) is None


class TestPrenormGating:
    def test_non_qwen_trunk_skips_prenorm_wrapper(self) -> None:
        """gemma4-shaped trunks (no fa_idx, no is_linear layers) must NOT get
        the qwen-specific pre-norm wrapper — its manual masks mirror the
        qwen ssm/full split and would be wrong for sliding/full layers."""
        from exo.worker.engines.mlx.generator.generate import _make_prenorm_trunk_fn

        class _GemmaishTrunk:
            def embed_tokens(self, x: mx.array) -> mx.array:
                return mx.zeros((1, x.shape[-1], HIDDEN))

            layers = [_FakeLayer("sliding_attention"), _FakeLayer("full_attention")]

        assert _make_prenorm_trunk_fn(_GemmaishTrunk()) is None
