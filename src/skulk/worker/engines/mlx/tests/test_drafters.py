# pyright: reportPrivateUsage=false, reportUnknownLambdaType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportArgumentType=false
"""Unit tests for the speculative-decoding drafters package.

Tests cover:
- build_drafter: layout dispatch (Qwen / DeepSeek / unrecognised) and
  model-card convention overrides
- QwenSidecarDrafter: +1.0 zero-centered norm shift, embed_first concat
  order, block strict-load against a real tiny qwen3_5 decoder layer,
  private KV cache advancement through observe/draft, failure fallbacks
- DeepseekSidecarDrafter: build paths, quantized eh_proj, draft shape/dtype
- introspection: tied-embedding head discovery
- _stream_generate_with_mtp: begin_request/observe pair-stream symmetry on
  the prompt, accept, and reject paths; SSM state restoration on reject

The Qwen tests use a REAL one-layer qwen3_5 text model at tiny dimensions
(not mocks) so block instantiation, strict weight loading, and attention
cache behavior are exercised against genuine mlx-lm modules.
"""

from __future__ import annotations

from typing import Callable, cast
from unittest.mock import MagicMock

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models import qwen3_5

from skulk.shared.models.model_cards import RuntimeCapabilityCardConfig
from skulk.worker.engines.mlx.drafters import Drafter, build_drafter
from skulk.worker.engines.mlx.drafters.deepseek_sidecar import (
    DeepseekSidecarDrafter,
    build_deepseek_sidecar_drafter,
)
from skulk.worker.engines.mlx.drafters.introspection import get_head_fn
from skulk.worker.engines.mlx.drafters.qwen_sidecar import (
    QwenSidecarDrafter,
    build_qwen_sidecar_drafter,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

HIDDEN = 16
VOCAB = 32
GROUP = 4


def _tiny_args() -> qwen3_5.TextModelArgs:
    """Args for a one-layer, all-full-attention qwen3_5 model at toy size."""
    # mlx-lm's BaseModelArgs dataclass machinery is unannotated, so pyright
    # cannot see the field defaults.
    return qwen3_5.TextModelArgs(  # pyright: ignore[reportCallIssue]
        model_type="qwen3_5",
        hidden_size=HIDDEN,
        intermediate_size=2 * HIDDEN,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=VOCAB,
        full_attention_interval=1,
        tie_word_embeddings=True,
    )


class _TinyModel:
    """Wrapper mimicking the loaded-model shape (`.language_model`)."""

    def __init__(self) -> None:
        self.language_model = qwen3_5.TextModel(_tiny_args())


def _make_qwen35_weights(
    *,
    with_block: bool = True,
    with_norm: bool = True,
    norm_value: float = 1.0,
) -> dict[str, mx.array]:
    """Return a Qwen3.5-style sidecar dict shaped for the tiny model."""
    w: dict[str, mx.array] = {
        "mtp.pre_fc_norm_hidden.weight": mx.full((HIDDEN,), norm_value),
        "mtp.pre_fc_norm_embedding.weight": mx.full((HIDDEN,), norm_value),
        "mtp.fc.weight": mx.zeros((HIDDEN, 2 * HIDDEN)),
    }
    if with_norm:
        w["mtp.norm.weight"] = mx.full((HIDDEN,), norm_value)
    if with_block:
        # Borrow exact parameter names/shapes from a real decoder layer so
        # strict-load is guaranteed to match.
        layer = qwen3_5.DecoderLayer(_tiny_args(), layer_idx=0)
        for name, value in tree_flatten(layer.parameters()):
            w[f"mtp.layers.0.{name}"] = value
    return w


def _make_deepseek_weights(
    *,
    prefix: str = "mtp.0.",
    quantized: bool = False,
    add_shared_norm: bool = False,
) -> dict[str, mx.array]:
    """Return a minimal DeepSeek-style sidecar weight dict."""
    w: dict[str, mx.array] = {}
    w[f"{prefix}hnorm.weight"] = mx.ones(HIDDEN)
    w[f"{prefix}enorm.weight"] = mx.ones(HIDDEN)
    if quantized:
        w[f"{prefix}eh_proj.weight"] = mx.zeros((HIDDEN, 2 * HIDDEN // 8), dtype=mx.uint32)
        w[f"{prefix}eh_proj.weight_scales"] = mx.zeros(
            (HIDDEN, 2 * HIDDEN // GROUP), dtype=mx.float16
        )
        w[f"{prefix}eh_proj.weight_biases"] = mx.zeros(
            (HIDDEN, 2 * HIDDEN // GROUP), dtype=mx.float16
        )
    else:
        w[f"{prefix}eh_proj.weight"] = mx.zeros((HIDDEN, 2 * HIDDEN))
    if add_shared_norm:
        w[f"{prefix}shared_head.norm.weight"] = mx.ones(HIDDEN)
    return w


# ---------------------------------------------------------------------------
# build_drafter — layout dispatch and overrides
# ---------------------------------------------------------------------------


class TestBuildDrafterDispatch:
    def test_returns_none_for_empty_weights(self) -> None:
        assert build_drafter(_TinyModel(), {}) is None

    def test_returns_none_for_unrecognised_layout(self) -> None:
        weights = {"other.norm.weight": mx.ones(HIDDEN)}
        assert build_drafter(_TinyModel(), weights) is None

    def test_qwen_layout_builds_qwen_drafter(self) -> None:
        drafter = build_drafter(_TinyModel(), _make_qwen35_weights())
        assert isinstance(drafter, QwenSidecarDrafter)
        assert isinstance(drafter, Drafter)

    def test_deepseek_layout_builds_deepseek_drafter(self) -> None:
        drafter = build_drafter(_TinyModel(), _make_deepseek_weights())
        assert isinstance(drafter, DeepseekSidecarDrafter)
        assert isinstance(drafter, Drafter)

    def test_deepseek_model_prefix(self) -> None:
        drafter = build_drafter(_TinyModel(), _make_deepseek_weights(prefix="model.mtp.0."))
        assert isinstance(drafter, DeepseekSidecarDrafter)

    def test_qwen_default_conventions(self) -> None:
        # zero_centered default: norms stored as 0.0 become effective 1.0.
        weights = _make_qwen35_weights(norm_value=0.0)
        drafter = build_drafter(_TinyModel(), weights)
        assert isinstance(drafter, QwenSidecarDrafter)
        assert mx.allclose(drafter._hnorm_w, mx.ones(HIDDEN))
        assert drafter._concat_order == "embed_first"

    def test_card_overrides_take_precedence(self) -> None:
        runtime = RuntimeCapabilityCardConfig(
            mtp_norm_convention="actual_scale",
            mtp_concat_order="hidden_first",
        )
        weights = _make_qwen35_weights(norm_value=0.5)
        drafter = build_drafter(_TinyModel(), weights, runtime=runtime)
        assert isinstance(drafter, QwenSidecarDrafter)
        # actual_scale: stored value used verbatim, no +1 shift.
        assert mx.allclose(drafter._hnorm_w, mx.full((HIDDEN,), 0.5))
        assert drafter._concat_order == "hidden_first"


# ---------------------------------------------------------------------------
# QwenSidecarDrafter
# ---------------------------------------------------------------------------


class TestQwenSidecarDrafter:
    def _build(self, **kwargs: object) -> QwenSidecarDrafter:
        drafter = build_qwen_sidecar_drafter(
            _TinyModel(),
            _make_qwen35_weights(**kwargs),
            norm_convention="zero_centered",
            concat_order="embed_first",
        )
        assert drafter is not None
        return drafter

    def test_norm_shift_applied_to_all_norm_weights(self) -> None:
        drafter = self._build(norm_value=0.0)
        assert mx.allclose(drafter._hnorm_w, mx.ones(HIDDEN))
        assert mx.allclose(drafter._enorm_w, mx.ones(HIDDEN))
        assert mx.allclose(drafter._final_norm_w, mx.ones(HIDDEN))
        # Block norms shift too: input_layernorm initialises to 1.0 in the
        # real layer, so the shifted load must read 2.0.
        block_norm = drafter._block.input_layernorm.weight  # type: ignore[union-attr]
        assert mx.allclose(block_norm, mx.full((HIDDEN,), 2.0))

    def test_missing_block_returns_none(self) -> None:
        drafter = build_qwen_sidecar_drafter(
            _TinyModel(),
            _make_qwen35_weights(with_block=False),
            norm_convention="zero_centered",
            concat_order="embed_first",
        )
        assert drafter is None

    def test_missing_final_norm_returns_none(self) -> None:
        drafter = build_qwen_sidecar_drafter(
            _TinyModel(),
            _make_qwen35_weights(with_norm=False),
            norm_convention="zero_centered",
            concat_order="embed_first",
        )
        assert drafter is None

    def test_corrupt_block_shape_returns_none(self) -> None:
        weights = _make_qwen35_weights()
        weights["mtp.layers.0.self_attn.q_proj.weight"] = mx.zeros((3, 3))
        drafter = build_qwen_sidecar_drafter(
            _TinyModel(),
            weights,
            norm_convention="zero_centered",
            concat_order="embed_first",
        )
        assert drafter is None

    def test_quantized_sidecar_rejected(self) -> None:
        weights = _make_qwen35_weights()
        weights["mtp.fc.weight_scales"] = mx.zeros((HIDDEN, 2 * HIDDEN // GROUP))
        weights["mtp.fc.weight_biases"] = mx.zeros((HIDDEN, 2 * HIDDEN // GROUP))
        drafter = build_qwen_sidecar_drafter(
            _TinyModel(),
            weights,
            norm_convention="zero_centered",
            concat_order="embed_first",
        )
        assert drafter is None

    def test_draft_output_shape_and_dtype(self) -> None:
        drafter = self._build()
        drafter.begin_request([])
        logits = drafter.draft(mx.zeros(HIDDEN), next_token=1)
        assert logits.shape == (1, VOCAB)
        assert logits.dtype == mx.float32

    def test_chained_draft_shape_and_cache_rollback(self) -> None:
        """A depth-3 chain returns 3 rows but persists only the input pair —
        chained entries use block-output hiddens, not canonical pairs, and
        must be rolled back (the pair-stream contract)."""
        drafter = self._build()
        drafter.begin_request([])
        logits = drafter.draft(mx.zeros(HIDDEN), next_token=1, depth=3)
        assert logits.shape == (3, VOCAB)
        assert drafter._cache.offset == 1

    def test_observe_and_draft_advance_private_cache(self) -> None:
        drafter = self._build()
        drafter.begin_request([])
        assert drafter._cache.offset == 0
        drafter.observe(mx.zeros((3, HIDDEN)), mx.array([1, 2, 3]))
        assert drafter._cache.offset == 3
        drafter.draft(mx.zeros(HIDDEN), next_token=4)
        assert drafter._cache.offset == 4

    def test_begin_request_resets_private_cache(self) -> None:
        drafter = self._build()
        drafter.begin_request([])
        drafter.observe(mx.zeros((3, HIDDEN)), mx.array([1, 2, 3]))
        drafter.begin_request([])
        assert drafter._cache.offset == 0


class TestQwenConcatOrder:
    """Behavioral check that concat order controls which half fc consumes."""

    def _project_with_order(self, order: str) -> mx.array:
        # fc = [I | 0]: output equals the FIRST half of the concat input.
        fc = mx.concatenate([mx.eye(HIDDEN), mx.zeros((HIDDEN, HIDDEN))], axis=1)
        drafter = QwenSidecarDrafter(
            hnorm_w=mx.ones(HIDDEN),
            enorm_w=mx.ones(HIDDEN),
            fc_w=fc,
            final_norm_w=mx.ones(HIDDEN),
            block=MagicMock(),
            embed_fn=lambda tokens: mx.ones((tokens.shape[0], HIDDEN)),
            head_fn=lambda x: x,
            concat_order=order,
            eps=1e-6,
        )
        # rms_norm(ones) = ones; rms_norm(-ones) = -ones.
        return drafter._project(mx.full((1, HIDDEN), -1.0), mx.array([0]))

    def test_embed_first_picks_embedding_half(self) -> None:
        out = self._project_with_order("embed_first")
        assert mx.allclose(out[0, 0], mx.ones(HIDDEN), atol=1e-2)

    def test_hidden_first_picks_hidden_half(self) -> None:
        out = self._project_with_order("hidden_first")
        assert mx.allclose(out[0, 0], -mx.ones(HIDDEN), atol=1e-2)


# ---------------------------------------------------------------------------
# DeepseekSidecarDrafter
# ---------------------------------------------------------------------------


class TestDeepseekSidecarDrafter:
    def test_float_weights_top_level_prefix(self) -> None:
        drafter = build_deepseek_sidecar_drafter(_TinyModel(), _make_deepseek_weights())
        assert isinstance(drafter, DeepseekSidecarDrafter)

    def test_unrecognised_layout_returns_none(self) -> None:
        weights = {"other.0.hnorm.weight": mx.ones(HIDDEN)}
        assert build_deepseek_sidecar_drafter(_TinyModel(), weights) is None

    def test_quantized_eh_proj(self) -> None:
        drafter = build_deepseek_sidecar_drafter(
            _TinyModel(), _make_deepseek_weights(quantized=True)
        )
        assert drafter is not None
        assert drafter._eh_proj_scales is not None
        assert drafter._eh_proj_biases is not None

    def test_shared_norm_loaded(self) -> None:
        drafter = build_deepseek_sidecar_drafter(
            _TinyModel(), _make_deepseek_weights(add_shared_norm=True)
        )
        assert drafter is not None
        assert drafter._shared_norm_w is not None

    def test_draft_output_dtype(self) -> None:
        drafter = build_deepseek_sidecar_drafter(_TinyModel(), _make_deepseek_weights())
        assert drafter is not None
        drafter.begin_request([])
        logits = drafter.draft(mx.zeros(HIDDEN), next_token=5)
        assert logits.dtype == mx.float32

    def test_observe_is_noop(self) -> None:
        drafter = build_deepseek_sidecar_drafter(_TinyModel(), _make_deepseek_weights())
        assert drafter is not None
        drafter.observe(mx.zeros((3, HIDDEN)), mx.array([1, 2, 3]))  # must not raise


# ---------------------------------------------------------------------------
# introspection
# ---------------------------------------------------------------------------


class TestTiedEmbeddingsHead:
    def test_head_located_via_as_linear(self) -> None:
        """Tied-embedding models (mlx-lm >= 0.31.3 qwen3_5 TextModel) expose
        no lm_head; the head is embed_tokens.as_linear."""

        class _Embed:
            def as_linear(self, x: mx.array) -> mx.array:
                return x

        class _Trunk:
            embed_tokens = _Embed()

        class _TextModel:
            model = _Trunk()
            # no lm_head attribute

        class _Model:
            language_model = _TextModel()

        head = get_head_fn(_Model())
        assert head is not None
        probe = mx.array([1.0])
        assert mx.array_equal(head(probe), probe)


# ---------------------------------------------------------------------------
# _get_trunk_and_head
# ---------------------------------------------------------------------------


class TestGetTrunkAndHead:
    def test_qwen_style(self) -> None:
        from skulk.worker.engines.mlx.generator.generate import _get_trunk_and_head

        trunk = MagicMock()
        lm_head = MagicMock()
        lm = MagicMock()
        lm.model = trunk
        lm.lm_head = lm_head
        model = MagicMock()
        model.language_model = lm

        result = _get_trunk_and_head(model)
        assert result is not None
        t, h = result
        assert t is trunk
        assert h is lm_head

    def test_deepseek_style(self) -> None:
        from skulk.worker.engines.mlx.generator.generate import _get_trunk_and_head

        trunk = MagicMock()
        lm_head = MagicMock()
        model = MagicMock(spec=[])  # no attributes
        model.model = trunk
        model.lm_head = lm_head

        result = _get_trunk_and_head(model)
        assert result is not None
        t, h = result
        assert t is trunk
        assert h is lm_head

    def test_unsupported_returns_none(self) -> None:
        from skulk.worker.engines.mlx.generator.generate import _get_trunk_and_head

        model = MagicMock(spec=[])  # no relevant attributes
        result = _get_trunk_and_head(model)
        assert result is None


class TestPrenormTrunk:
    def test_trunk_fn_returns_prenorm_hiddens(self) -> None:
        """The MTP trunk_fn must return PRE-final-norm hiddens.

        Regression test for the live 0%-acceptance finding: feeding the
        drafter post-norm hiddens silently breaks drafting while keeping
        main-path logits correct. Pin the contract on a real tiny model:
        norm(trunk_fn(x)) == full trunk forward, and head_fn folds the norm.
        """
        from skulk.worker.engines.mlx.generator.generate import _get_trunk_and_head

        model = _TinyModel()
        text_model = model.language_model
        result = _get_trunk_and_head(model)
        assert result is not None
        trunk_fn, head_fn = result

        tokens = mx.array([[1, 2, 3, 4]])
        prenorm = trunk_fn(tokens, cache=None)
        full = text_model.model(tokens)  # applies the final norm
        assert not mx.allclose(prenorm, full)
        assert mx.allclose(text_model.model.norm(prenorm), full, atol=1e-5)
        # head_fn must reproduce the model's own logits from prenorm hiddens.
        assert mx.allclose(head_fn(prenorm), text_model(tokens), atol=1e-4)


# ---------------------------------------------------------------------------
# _stream_generate_with_mtp — pair-stream symmetry and token sequences
# ---------------------------------------------------------------------------


class _FakeDrafter:
    """Protocol-satisfying fake that records every loop interaction.

    Drafts the configured chain (a one-hot row per chain entry, capped by
    the requested depth), so depth tests can stage exact prefix outcomes.
    """

    def __init__(
        self, draft_token_id: int | list[int], vocab_size: int = VOCAB
    ) -> None:
        self._chain = (
            [draft_token_id] if isinstance(draft_token_id, int) else draft_token_id
        )
        self._v = vocab_size
        self.begin_request_count = 0
        self.observe_calls: list[tuple[int, list[int]]] = []
        self.draft_calls: list[int] = []

    def begin_request(self, prompt_cache: object) -> None:
        self.begin_request_count += 1

    def observe(self, hiddens: mx.array, next_tokens: mx.array) -> None:
        tokens = [int(t) for t in cast("list[int]", next_tokens.tolist())]
        self.observe_calls.append((int(hiddens.shape[0]), tokens))

    def draft(self, hidden: mx.array, next_token: int, depth: int = 1) -> mx.array:
        self.draft_calls.append(next_token)
        rows: list[mx.array] = []
        for token in self._chain[: max(depth, 1)]:
            row = mx.zeros(self._v)
            rows.append(mx.where(mx.arange(self._v) == token, mx.array(100.0), row))
        return mx.stack(rows).astype(mx.float32)


def _build_fake_stream_env(
    *,
    vocab_size: int = VOCAB,
    hidden_size: int = HIDDEN,
    main_token_ids: list[int],  # what the "trunk+head" return each call
    draft_token_id: int,
):
    """Build minimal fakes for testing _stream_generate_with_mtp."""
    from mlx_lm.tokenizer_utils import TokenizerWrapper

    _main_token_iter = iter(main_token_ids)

    def fake_trunk(tokens: mx.array, cache: object = None) -> mx.array:
        seq_len = tokens.shape[1] if tokens.ndim == 2 else 1
        return mx.zeros((1, seq_len, hidden_size))

    def fake_head(hidden: mx.array) -> mx.array:
        seq_len = hidden.shape[1] if hidden.ndim == 3 else 1
        out = mx.zeros((1, seq_len, vocab_size))
        try:
            tok = next(_main_token_iter)
        except StopIteration:
            tok = 0  # EOS or fallback
        return mx.where(
            mx.arange(vocab_size)[None, None, :] == tok,
            mx.array(100.0),
            out,
        )

    tokenizer = MagicMock(spec=TokenizerWrapper)
    detokenizer = MagicMock()
    detokenizer.last_segment = "hi"
    tokenizer.detokenizer = detokenizer
    tokenizer.eos_token_ids = []

    model = MagicMock()
    drafter = _FakeDrafter(draft_token_id=draft_token_id, vocab_size=vocab_size)

    class _FakeCache:
        def __init__(self):
            self.state = []
            self.offset = 0
            self._trimmed = 0

        def trim(self, n: int) -> None:
            self._trimmed += n

    fake_cache = [_FakeCache()]

    return model, tokenizer, drafter, fake_trunk, fake_head, fake_cache


class TestStreamGenerateWithMTP:
    def _run(
        self,
        *,
        main_token_ids: list[int],
        draft_token_id: int | list[int],
        max_tokens: int = 10,
        depth: int = 1,
    ):
        from skulk.worker.engines.mlx.generator.generate import (
            _stream_generate_with_mtp,
        )

        model, tokenizer, drafter, trunk_fn, head_fn, cache = _build_fake_stream_env(
            main_token_ids=main_token_ids,
            draft_token_id=draft_token_id,
        )

        sampler = lambda lp: mx.argmax(lp, axis=-1)  # noqa: E731
        prompt = mx.array([1, 2, 3])

        outputs = list(
            _stream_generate_with_mtp(
                model=model,
                tokenizer=tokenizer,
                drafter=drafter,
                trunk_fn=trunk_fn,
                head_fn=head_fn,
                prompt=prompt,
                max_tokens=max_tokens,
                sampler=sampler,
                logits_processors=[],
                prompt_cache=cache,
                kv_group_size=None,
                kv_bits=None,
                depth=depth,
            )
        )
        return outputs, drafter, cache

    def test_yields_responses(self) -> None:
        outputs, _, _ = self._run(
            main_token_ids=[5, 0],
            draft_token_id=5,
            max_tokens=5,
        )
        assert len(outputs) >= 1

    def test_begin_request_called_once(self) -> None:
        _, drafter, _ = self._run(
            main_token_ids=[5, 0],
            draft_token_id=5,
            max_tokens=5,
        )
        assert drafter.begin_request_count == 1

    def test_prompt_pairs_bulk_observed(self) -> None:
        _, drafter, _ = self._run(
            main_token_ids=[5, 0],
            draft_token_id=5,
            max_tokens=5,
        )
        # Prompt [1, 2, 3]: positions 0..1 pair with tokens [2, 3].
        assert drafter.observe_calls[0] == (2, [2, 3])

    def test_accept_observes_skipped_draft_pair(self) -> None:
        # main 5, draft 5, verify target 5 → accept. The skipped position's
        # pair carries the accepted draft token.
        _, drafter, _ = self._run(
            main_token_ids=[5, 5, 0],
            draft_token_id=5,
            max_tokens=4,
        )
        assert (1, [5]) in drafter.observe_calls[1:]

    def test_reject_observes_nothing_and_drafts_from_correction(self) -> None:
        # Bonus-driven rounds: on a reject the correction becomes the next
        # bonus — consumed by the next draft() call, never observed. The
        # rejected draft token must never enter the pair stream.
        _, drafter, _ = self._run(
            main_token_ids=[5, 3, 0],
            draft_token_id=7,
            max_tokens=4,
        )
        assert all(7 not in tokens for _, tokens in drafter.observe_calls)
        # Post-prefill, rejects contribute no observes (only accepted drafts do).
        assert all(count == 2 for count, _ in drafter.observe_calls[:1])
        # The correction (3) is consumed as the next draft's bonus pair.
        assert 3 in drafter.draft_calls

    def test_draft_calls_track_main_tokens(self) -> None:
        _outputs, drafter, _ = self._run(
            main_token_ids=[5, 6, 0],
            draft_token_id=5,
            max_tokens=3,
        )
        assert 5 in drafter.draft_calls

    def test_max_tokens_respected(self) -> None:
        limit = 4
        outputs, _, _ = self._run(
            main_token_ids=[5] * 20,
            draft_token_id=5,
            max_tokens=limit,
        )
        # Accept chains can yield 2 per pass so allow a small window.
        assert len(outputs) <= limit + 2

    def test_terminal_response_carries_finalized_segment(self) -> None:
        """Break-path terminal yields must finalize the detokenizer first —
        sentencepiece-backed tokenizers buffer tail bytes until finalize()
        (#180 item 4). The fake reveals its buffered tail only on finalize."""
        from skulk.worker.engines.mlx.generator.generate import (
            _stream_generate_with_mtp,
        )

        model, tokenizer, drafter, trunk_fn, head_fn, cache = _build_fake_stream_env(
            main_token_ids=[5] * 20,
            draft_token_id=5,
        )

        class _BufferingDetokenizer:
            def __init__(self) -> None:
                self._finalized = False
                self.finalize_calls = 0

            @property
            def last_segment(self) -> str:
                return "tail" if self._finalized else "hi"

            def reset(self) -> None:
                self._finalized = False

            def add_token(self, token: int) -> None:
                del token

            def finalize(self) -> None:
                self._finalized = True
                self.finalize_calls += 1

        buffering_detokenizer = _BufferingDetokenizer()
        tokenizer.detokenizer = buffering_detokenizer
        sampler = lambda lp: mx.argmax(lp, axis=-1)  # noqa: E731
        outputs = list(
            _stream_generate_with_mtp(
                model=model,
                tokenizer=tokenizer,
                drafter=drafter,
                trunk_fn=trunk_fn,
                head_fn=head_fn,
                prompt=mx.array([1, 2, 3]),
                max_tokens=4,  # terminal via max_tokens break path
                sampler=sampler,
                logits_processors=[],
                prompt_cache=cache,
                kv_group_size=None,
                kv_bits=None,
            )
        )
        terminal = [o for o in outputs if o.finish_reason is not None]
        assert terminal, "loop must yield a terminal response"
        assert terminal[-1].text == "tail", (
            "terminal response was built before detokenizer.finalize()"
        )
        # Exactly once: the post-loop tail must not double-finalize after a
        # break path already finalized via the terminal response.
        assert buffering_detokenizer.finalize_calls == 1

    def test_drafter_exception_disables_speculation_not_generation(self) -> None:
        """Speculation must never abort generation: a drafter that raises
        disables drafting for the request and decode continues plain."""
        from skulk.worker.engines.mlx.generator.generate import (
            _stream_generate_with_mtp,
        )

        model, tokenizer, _drafter, trunk_fn, head_fn, cache = _build_fake_stream_env(
            main_token_ids=[5, 6, 7, 0],
            draft_token_id=5,
        )

        class _ExplodingDrafter:
            def __init__(self) -> None:
                self.draft_attempts = 0

            def begin_request(self, prompt_cache: object) -> None:
                pass

            def observe(self, hiddens: mx.array, next_tokens: mx.array) -> None:
                pass

            def draft(
                self, hidden: mx.array, next_token: int, depth: int = 1
            ) -> mx.array:
                self.draft_attempts += 1
                raise RuntimeError("upstream API drift")

        exploding = _ExplodingDrafter()
        sampler = lambda lp: mx.argmax(lp, axis=-1)  # noqa: E731
        outputs = list(
            _stream_generate_with_mtp(
                model=model,
                tokenizer=tokenizer,
                drafter=exploding,
                trunk_fn=trunk_fn,
                head_fn=head_fn,
                prompt=mx.array([1, 2, 3]),
                max_tokens=4,
                sampler=sampler,
                logits_processors=[],
                prompt_cache=cache,
                kv_group_size=None,
                kv_bits=None,
            )
        )
        # Generation completed despite the drafter failing...
        assert len(outputs) >= 3
        # ...and drafting was attempted exactly once, then disabled.
        assert exploding.draft_attempts == 1

    def test_reject_path_does_not_crash(self) -> None:
        outputs, _drafter, _cache = self._run(
            main_token_ids=[5, 3, 0],
            draft_token_id=7,  # draft != verify → reject
            max_tokens=3,
        )
        assert len(outputs) >= 1


class TestStreamGenerateSampled(TestStreamGenerateWithMTP):
    """T>0 ratio-acceptance paths. The fake head/drafter emit near-one-hot
    distributions, so accept (p(x)≈1/q(x)≈1) and reject (p(x)≈0) outcomes
    are deterministic even under sampling."""

    def _run_sampled(
        self,
        *,
        main_token_ids: list[int],
        draft_token_id: int | list[int],
        max_tokens: int = 6,
        depth: int = 1,
    ):
        from skulk.worker.engines.mlx.generator.generate import (
            _stream_generate_with_mtp,
        )
        from skulk.worker.engines.mlx.generator.speculative_sampling import (
            SamplingParams,
        )

        model, tokenizer, drafter, trunk_fn, head_fn, cache = _build_fake_stream_env(
            main_token_ids=main_token_ids,
            draft_token_id=draft_token_id,
        )
        sampler = lambda lp: mx.argmax(lp, axis=-1)  # noqa: E731

        outputs = list(
            _stream_generate_with_mtp(
                model=model,
                tokenizer=tokenizer,
                drafter=drafter,
                trunk_fn=trunk_fn,
                head_fn=head_fn,
                prompt=mx.array([1, 2, 3]),
                max_tokens=max_tokens,
                sampler=sampler,
                logits_processors=[],
                prompt_cache=cache,
                kv_group_size=None,
                kv_bits=None,
                depth=depth,
                sampling=SamplingParams(temperature=0.7, top_p=1.0, min_p=0.0),
            )
        )
        return outputs, drafter

    def test_sampled_accept_emits_draft(self) -> None:
        mx.random.seed(11)
        outputs, _ = self._run_sampled(
            main_token_ids=[5] * 8, draft_token_id=5, max_tokens=5
        )
        assert any(o.from_draft for o in outputs)

    def test_sampled_reject_emits_residual_not_draft(self) -> None:
        mx.random.seed(11)
        outputs, _ = self._run_sampled(
            main_token_ids=[5, 3, 3, 0], draft_token_id=7, max_tokens=4
        )
        emitted = [o.token for o in outputs]
        assert 7 not in emitted  # p(7) ≈ 0 → always rejected
        assert any(o.token == 3 for o in outputs)  # residual ≈ p → target

    def test_sampled_forces_depth_one(self) -> None:
        mx.random.seed(11)
        _, drafter = self._run_sampled(
            main_token_ids=[5] * 8, draft_token_id=[5, 5], max_tokens=5, depth=2
        )
        # The fake records depth via rows returned; with depth forced to 1
        # the configured 2-token chain is truncated to a single draft per
        # call — observe payloads therefore never exceed one token.
        assert all(count == 1 for count, _ in drafter.observe_calls[1:])


class TestStreamGenerateDepth(TestStreamGenerateWithMTP):
    """Depth-K prefix-acceptance outcomes (fake head emits the same token at
    every verify position, so chain entries equal to the current main id are
    accepted and others reject at that position)."""

    def test_full_accept_emits_all_drafts(self) -> None:
        # mains all 5, chain [5, 5] → both drafts match every verify row.
        outputs, drafter, _ = self._run(
            main_token_ids=[5] * 8,
            draft_token_id=[5, 5],
            max_tokens=6,
            depth=2,
        )
        assert sum(int(o.from_draft) for o in outputs) >= 2
        # Full accept observes one pair per accepted draft.
        assert (2, [5, 5]) in drafter.observe_calls[1:]

    def test_partial_accept_commits_prefix_plus_correction(self) -> None:
        # Chain [5, 9]: first draft matches the verifier (5), second rejects;
        # the correction becomes the next bonus.
        outputs, drafter, _ = self._run(
            main_token_ids=[5] * 8,
            draft_token_id=[5, 9],
            max_tokens=6,
            depth=2,
        )
        assert any(o.from_draft for o in outputs)
        # Only the ACCEPTED prefix is observed (the correction rides as the
        # next bonus into draft()).
        assert (1, [5]) in drafter.observe_calls[1:]
        assert all(9 not in tokens for _, tokens in drafter.observe_calls)

    def test_full_reject_observes_nothing(self) -> None:
        outputs, drafter, _ = self._run(
            main_token_ids=[5, 3, 3, 0],
            draft_token_id=[7, 9],
            max_tokens=4,
            depth=2,
        )
        assert len(outputs) >= 1
        # Rejected drafts never enter the pair stream; no observes beyond
        # the prefill bulk-ingest on a full reject.
        assert all(
            7 not in tokens and 9 not in tokens
            for _, tokens in drafter.observe_calls
        )
        assert len(drafter.observe_calls) == 1  # prefill only

    def test_depth_reject_restores_ssm_state(self) -> None:
        """A depth-2 reject must trim all three verify positions and restore
        the SSM snapshot."""
        from mlx_lm.models.cache import ArraysCache, KVCache

        from skulk.worker.engines.mlx.generator.generate import (
            _stream_generate_with_mtp,
        )

        model, tokenizer, drafter, trunk_fn, head_fn, _unused = (
            _build_fake_stream_env(
                main_token_ids=[5, 3, 0],
                draft_token_id=[7, 9],  # always rejected at position 0
            )
        )
        # Without this, MagicMock auto-provides a callable
        # rollback_speculative_cache and the snapshot path under test is
        # silently bypassed.
        model.language_model = None
        ssm_cache = ArraysCache(size=1)
        sentinel = mx.ones((1, 4))
        ssm_cache.state = [sentinel]
        kv_cache = KVCache()
        _ = kv_cache.update_and_fetch(  # pyright: ignore[reportUnknownMemberType]
            mx.zeros((1, 1, 8, 4)), mx.zeros((1, 1, 8, 4))
        )
        cache = [ssm_cache, kv_cache]

        sampler = lambda lp: mx.argmax(lp, axis=-1)  # noqa: E731
        outputs = list(
            _stream_generate_with_mtp(
                model=model,
                tokenizer=tokenizer,
                drafter=drafter,
                trunk_fn=trunk_fn,
                head_fn=head_fn,
                prompt=mx.array([1, 2, 3]),
                max_tokens=3,
                sampler=sampler,
                logits_processors=[],
                prompt_cache=cache,
                kv_group_size=None,
                kv_bits=None,
                depth=2,
            )
        )
        assert len(outputs) >= 1
        assert ssm_cache.state[0] is not None, "depth reject zeroed the SSM state"
        assert mx.array_equal(ssm_cache.state[0], sentinel)


class TestRejectPathSSMState:
    def test_reject_restores_ssm_state(self) -> None:
        """A reject must restore SSM (ArraysCache) state, not zero it.

        Regression test: trim_cache without a snapshot sets ArraysCache
        state to [None, ...], silently wiping the recurrent state of hybrid
        models (Qwen3.5 GDN) on every rejected draft and degenerating the
        output. The reject path must snapshot before the verify pass and
        restore on reject.
        """
        from mlx_lm.models.cache import ArraysCache, KVCache

        from skulk.worker.engines.mlx.generator.generate import (
            _stream_generate_with_mtp,
        )

        model, tokenizer, drafter, trunk_fn, head_fn, _fake_cache = (
            _build_fake_stream_env(
                main_token_ids=[5, 3, 0],
                draft_token_id=7,  # head verifies 5/3 → draft 7 always rejected
            )
        )
        # Without this, MagicMock auto-provides a callable
        # rollback_speculative_cache and the snapshot path under test is
        # silently bypassed.
        model.language_model = None

        ssm_cache = ArraysCache(size=1)
        sentinel = mx.ones((1, 4))
        ssm_cache.state = [sentinel]
        # Seed the KV cache: in production it always holds prompt positions
        # by the time the MTP loop runs, and trim on an empty KVCache throws.
        kv_cache = KVCache()
        _ = kv_cache.update_and_fetch(  # pyright: ignore[reportUnknownMemberType]
            mx.zeros((1, 1, 8, 4)), mx.zeros((1, 1, 8, 4))
        )
        cache = [ssm_cache, kv_cache]

        sampler = lambda lp: mx.argmax(lp, axis=-1)  # noqa: E731
        outputs = list(
            _stream_generate_with_mtp(
                model=model,
                tokenizer=tokenizer,
                drafter=drafter,
                trunk_fn=trunk_fn,
                head_fn=head_fn,
                prompt=mx.array([1, 2, 3]),
                max_tokens=3,
                sampler=sampler,
                logits_processors=[],
                prompt_cache=cache,
                kv_group_size=None,
                kv_bits=None,
            )
        )

        assert len(outputs) >= 1
        # The fake trunk never advances SSM state, so after snapshot+restore
        # the sentinel must survive verbatim. The pre-fix code left this
        # zeroed ([None]) after the first reject.
        assert ssm_cache.state[0] is not None, "reject zeroed the SSM state"
        assert mx.array_equal(ssm_cache.state[0], sentinel)


# ---------------------------------------------------------------------------
# Bonus EOS detokenization — special-token text must not leak into output
# ---------------------------------------------------------------------------


class TestBonusEosDetokenization:
    """EOS bonus tokens must never enter the detokenizer.

    The accepted-draft emit checks `eos_token_ids` before `add_token`; the
    three bonus emits (first bonus, plain-decode fallback, post-verify) must
    match, or a stop-terminated generation leaks the EOS token's decoded
    special-token text into the terminal segment (PR #204 review).
    """

    EOS = 9

    def _run_with_eos(
        self,
        *,
        main_token_ids: list[int],
        draft_token_id: int,
        break_drafter: bool = False,
    ):
        from skulk.worker.engines.mlx.generator.generate import (
            _stream_generate_with_mtp,
        )

        model, tokenizer, drafter, trunk_fn, head_fn, cache = (
            _build_fake_stream_env(
                main_token_ids=main_token_ids,
                draft_token_id=draft_token_id,
            )
        )
        tokenizer.eos_token_ids = [self.EOS]
        if break_drafter:
            # Force the speculation_disabled plain-decode fallback.
            drafter.draft = MagicMock(side_effect=RuntimeError("boom"))

        sampler = lambda lp: mx.argmax(lp, axis=-1)  # noqa: E731
        outputs = list(
            _stream_generate_with_mtp(
                model=model,
                tokenizer=tokenizer,
                drafter=drafter,
                trunk_fn=trunk_fn,
                head_fn=head_fn,
                prompt=mx.array([1, 2, 3]),
                max_tokens=8,
                sampler=sampler,
                logits_processors=[],
                prompt_cache=cache,
                kv_group_size=None,
                kv_bits=None,
            )
        )
        add_token_mock: MagicMock = cast(
            MagicMock,
            tokenizer.detokenizer.add_token,  # pyright: ignore[reportAny]
        )
        added = cast(
            "list[int]",
            [call.args[0] for call in add_token_mock.call_args_list],
        )
        return outputs, added

    def test_first_bonus_eos_not_detokenized(self) -> None:
        outputs, added = self._run_with_eos(main_token_ids=[9], draft_token_id=5)
        assert outputs[-1].finish_reason == "stop"
        assert self.EOS not in added

    def test_post_verify_bonus_eos_not_detokenized(self) -> None:
        # First bonus 5; round 1 verify targets 9 → draft 5 rejected, the
        # correction bonus IS the EOS.
        outputs, added = self._run_with_eos(main_token_ids=[5, 9], draft_token_id=5)
        assert outputs[-1].finish_reason == "stop"
        assert self.EOS not in added
        assert 5 in added  # non-EOS tokens still detokenize

    def test_plain_decode_bonus_eos_not_detokenized(self) -> None:
        outputs, added = self._run_with_eos(
            main_token_ids=[5, 9], draft_token_id=5, break_drafter=True
        )
        assert outputs[-1].finish_reason == "stop"
        assert self.EOS not in added


# ---------------------------------------------------------------------------
# Distributed draft exchange (#201 Track 2b)
# ---------------------------------------------------------------------------


class TestExchangeDrafts:
    """Payload round-trip for last-rank drafting: the drafting rank encodes
    [chain_len, toks..., (q row under sampling)], non-drafting ranks decode
    the broadcast. The all_sum is monkeypatched to identity (a one-rank sum)
    so both directions are exercised without a real ring."""

    def _patch_broadcast(
        self, monkeypatch: pytest.MonkeyPatch, captured: list[mx.array]
    ):
        from skulk.worker.engines.mlx.generator import generate as generate_module

        def fake_broadcast(
            group: object, payload: mx.array, *, detail: str
        ) -> mx.array:
            del group, detail
            captured.append(payload)
            return payload

        monkeypatch.setattr(
            generate_module, "_broadcast_via_all_sum", fake_broadcast
        )
        return generate_module

    def test_greedy_drafting_rank_round_trip(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from skulk.worker.engines.mlx.generator.speculative_sampling import (
            SamplingParams,
        )

        captured: list[mx.array] = []
        generate_module = self._patch_broadcast(monkeypatch, captured)

        drafter = _FakeDrafter(draft_token_id=[7, 9])
        toks, probs = generate_module._exchange_drafts(
            draft_group=cast("mx.distributed.Group", cast(object, MagicMock())),
            drafter=drafter,
            hidden=mx.zeros(HIDDEN),
            bonus=5,
            depth=2,
            sampling=SamplingParams(temperature=0.0),
            vocab_size=VOCAB,
            draft_key=3,
        )
        assert toks == [7, 9]
        assert probs is None
        # Greedy payload: 1 + depth slots, no vocab row.
        assert captured[0].shape == (3,)

    def test_greedy_receiving_rank_decodes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from skulk.worker.engines.mlx.generator import generate as generate_module
        from skulk.worker.engines.mlx.generator.speculative_sampling import (
            SamplingParams,
        )

        def fake_broadcast(
            group: object, payload: mx.array, *, detail: str
        ) -> mx.array:
            del group, detail
            # Receiving rank contributed zeros of the right shape; the "sum"
            # is the drafting rank's payload.
            assert payload.shape == (3,)
            assert float(mx.sum(mx.abs(payload)).item()) == 0.0
            return mx.array([2.0, 7.0, 9.0])

        monkeypatch.setattr(
            generate_module, "_broadcast_via_all_sum", fake_broadcast
        )
        toks, probs = generate_module._exchange_drafts(
            draft_group=cast("mx.distributed.Group", cast(object, MagicMock())),
            drafter=None,
            hidden=mx.zeros(HIDDEN),
            bonus=5,
            depth=2,
            sampling=SamplingParams(temperature=0.0),
            vocab_size=VOCAB,
            draft_key=3,
        )
        assert toks == [7, 9]
        assert probs is None

    def test_sampled_payload_carries_q_row(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from skulk.worker.engines.mlx.generator.speculative_sampling import (
            SamplingParams,
        )

        captured: list[mx.array] = []
        generate_module = self._patch_broadcast(monkeypatch, captured)

        drafter = _FakeDrafter(draft_token_id=[7])
        toks, probs = generate_module._exchange_drafts(
            draft_group=cast("mx.distributed.Group", cast(object, MagicMock())),
            drafter=drafter,
            hidden=mx.zeros(HIDDEN),
            bonus=5,
            depth=1,
            sampling=SamplingParams(temperature=0.7),
            vocab_size=VOCAB,
            draft_key=3,
        )
        assert len(toks) == 1
        assert probs is not None
        assert probs.shape == (VOCAB,)
        # The drafter's one-hot logits warp to a near-one-hot distribution;
        # the sampled token must come from it.
        assert toks[0] == 7
        assert float(probs[7].item()) > 0.99
        # Sampled payload: 1 + depth slot + vocab row.
        assert captured[0].shape == (1 + 1 + VOCAB,)


class TestDeferredReplay:
    """On the hybrid-SSM path a reject must NOT pay a dedicated replay
    forward: restored-but-committed tokens are deferred and prepended to the
    next verify input (extra verify width is free on memory-bound decode).
    """

    def _run_ssm_logged(
        self,
        *,
        main_token_ids: list[int],
        draft_token_id: int | list[int],
        max_tokens: int,
        depth: int = 1,
    ) -> tuple[list[int], object]:
        """Run the loop on a real ArraysCache+KVCache pair, logging every
        trunk forward's sequence width."""
        from mlx_lm.models.cache import ArraysCache, KVCache

        from skulk.worker.engines.mlx.generator.generate import (
            _stream_generate_with_mtp,
        )

        model, tokenizer, drafter, trunk_fn, head_fn, _unused = (
            _build_fake_stream_env(
                main_token_ids=main_token_ids,
                draft_token_id=draft_token_id,
            )
        )
        # MagicMock would auto-provide language_model.rollback_speculative_cache
        # as a callable, silently routing the loop down the native-rollback
        # branch instead of the snapshot path under test.
        model.language_model = None

        widths: list[int] = []

        def logging_trunk(tokens: mx.array, cache: object = None) -> mx.array:
            widths.append(int(tokens.shape[1] if tokens.ndim == 2 else 1))
            return trunk_fn(tokens, cache=cache)

        ssm_cache = ArraysCache(size=1)
        ssm_cache.state = [mx.ones((1, 4))]
        kv_cache = KVCache()
        # Generous seed: reject trims grow with the pending window and the
        # fake trunk never advances offsets.
        _ = kv_cache.update_and_fetch(  # pyright: ignore[reportUnknownMemberType]
            mx.zeros((1, 1, 128, 4)), mx.zeros((1, 1, 128, 4))
        )
        cache = [ssm_cache, kv_cache]

        sampler = lambda lp: mx.argmax(lp, axis=-1)  # noqa: E731
        outputs = list(
            _stream_generate_with_mtp(
                model=model,
                tokenizer=tokenizer,
                drafter=drafter,
                trunk_fn=logging_trunk,
                head_fn=head_fn,
                prompt=mx.array([1, 2, 3]),
                max_tokens=max_tokens,
                sampler=sampler,
                logits_processors=[],
                prompt_cache=cache,
                kv_group_size=None,
                kv_bits=None,
                depth=depth,
            )
        )
        assert len(outputs) >= 1
        return widths, outputs

    def test_rejects_pay_no_dedicated_replay_forward(self) -> None:
        # Draft 31 never matches the verify targets → every round rejects.
        # Old shape per reject: verify + a 1-wide replay forward. New shape:
        # the restored [bonus] rides the next verify, growing its width.
        widths, _ = self._run_ssm_logged(
            main_token_ids=[5, 3, 9, 11, 0],
            draft_token_id=31,
            max_tokens=4,
        )
        # prefill(2), first-bonus(1), verify(2), verify(3: 1 pending),
        # verify(4: 2 pending), end flush(3).
        assert widths == [2, 1, 2, 3, 4, 3]

    def test_full_accept_clears_pending(self) -> None:
        # Round 1 rejects (target 3 ≠ draft 7) → pending [bonus]. Round 2
        # accepts (target 7 == draft 7) → the verify commits the pending
        # token, so round 3's verify is back to minimum width.
        widths, _ = self._run_ssm_logged(
            main_token_ids=[5, 3, 7, 9, 0],
            draft_token_id=7,
            max_tokens=5,
        )
        # prefill(2), first-bonus(1), verify(2: reject), verify(3: 1
        # pending, full accept), verify(2: pending cleared), ...
        assert widths[:5] == [2, 1, 2, 3, 2]

    def test_pending_window_is_capped(self) -> None:
        from skulk.worker.engines.mlx.generator.generate import (
            _MTP_MAX_PENDING_REPLAY,
        )

        widths, _ = self._run_ssm_logged(
            main_token_ids=list(range(3, 18)),
            draft_token_id=31,  # never matches → reject streak
            max_tokens=14,
        )
        # The verify window may grow to cap+1 (cap pending + bonus + draft)
        # but a reject streak must flush rather than grow without bound.
        assert max(widths) <= _MTP_MAX_PENDING_REPLAY + 2
        assert _MTP_MAX_PENDING_REPLAY in widths  # the mid-stream flush


# ---------------------------------------------------------------------------
# Per-expert MoE sidecar key stacking
# ---------------------------------------------------------------------------


class TestPerExpertStacking:
    """SWP sidecars for MoE backbones preserve raw per-expert keys
    (``mlp.experts.N.*``); mlx-lm decoder layers hold stacked SwitchGLU
    tensors. The builder must normalize on load (found on the 35B-A3B
    sidecar; the family sanitize that normally does this never runs on the
    sidecar strict-load path)."""

    def test_stacks_per_expert_keys(self) -> None:
        from skulk.worker.engines.mlx.drafters.qwen_sidecar import (
            _stack_per_expert_block_weights,
        )

        pairs = [
            ("input_layernorm.weight", mx.ones(4)),
            ("mlp.experts.1.gate_proj.weight", mx.full((8, 4), 1.0)),
            ("mlp.experts.0.gate_proj.weight", mx.full((8, 4), 0.0)),
            ("mlp.experts.0.down_proj.weight", mx.full((4, 8), 0.0)),
            ("mlp.experts.1.down_proj.weight", mx.full((4, 8), 1.0)),
            ("mlp.gate.weight", mx.zeros((2, 4))),
        ]
        out = dict(_stack_per_expert_block_weights(pairs))
        assert "mlp.switch_mlp.gate_proj.weight" in out
        gate = out["mlp.switch_mlp.gate_proj.weight"]
        # (num_experts, out, in), expert order by index regardless of input
        # order.
        assert gate.shape == (2, 8, 4)
        assert mx.allclose(gate[0], mx.zeros((8, 4)))
        assert mx.allclose(gate[1], mx.ones((8, 4)))
        assert out["mlp.switch_mlp.down_proj.weight"].shape == (2, 4, 8)
        # Router and norms pass through untouched.
        assert "mlp.gate.weight" in out
        assert "input_layernorm.weight" in out
        assert not any("experts." in k for k in out)

    def test_dense_pairs_pass_through(self) -> None:
        from skulk.worker.engines.mlx.drafters.qwen_sidecar import (
            _stack_per_expert_block_weights,
        )

        pairs = [("self_attn.q_proj.weight", mx.zeros((4, 4)))]
        assert _stack_per_expert_block_weights(pairs) == pairs

    def test_expert_gap_left_unstacked_for_loud_failure(self) -> None:
        from skulk.worker.engines.mlx.drafters.qwen_sidecar import (
            _stack_per_expert_block_weights,
        )

        pairs = [
            ("mlp.experts.0.gate_proj.weight", mx.zeros((8, 4))),
            ("mlp.experts.2.gate_proj.weight", mx.ones((8, 4))),  # 1 missing
        ]
        out = dict(_stack_per_expert_block_weights(pairs))
        # Truncated sidecar: keys stay per-expert so the strict load fails
        # on the missing stacked key instead of stacking a wrong tensor.
        assert "mlp.switch_mlp.gate_proj.weight" not in out
        assert len(out) == 2


# ---------------------------------------------------------------------------
# Sibling-layer construction under pipeline slicing
# ---------------------------------------------------------------------------


def _hybrid_args() -> qwen3_5.TextModelArgs:
    """Toy hybrid args: full attention every 4th layer (like real Qwen3.5)."""
    return qwen3_5.TextModelArgs(  # pyright: ignore[reportCallIssue]
        model_type="qwen3_5",
        hidden_size=HIDDEN,
        intermediate_size=2 * HIDDEN,
        num_hidden_layers=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        vocab_size=VOCAB,
        full_attention_interval=4,
        tie_word_embeddings=True,
    )


class TestSiblingLayerSliceAlignment:
    """Hybrid constructors derive layer TYPE from layer_idx; a pipeline
    slice whose offset is not a multiple of the attention interval makes the
    local index of a full-attention layer map to a LINEAR position — the
    naive ``type(layer)(args, layer_idx=<local>)`` silently builds a GDN
    sibling and the strict-load fails (#201 Track 2a flagship run). The
    builder must self-validate that the constructed sibling carries
    ``self_attn``.
    """

    def test_aligned_slice_builds_attention_sibling(self) -> None:
        from skulk.worker.engines.mlx.drafters.introspection import (
            build_sibling_attention_layer,
        )

        class _Model:
            def __init__(self) -> None:
                self.language_model = qwen3_5.TextModel(_hybrid_args())

        sibling = build_sibling_attention_layer(_Model())
        assert sibling is not None
        assert getattr(sibling, "self_attn", None) is not None

    def test_misaligned_slice_builds_attention_sibling(self) -> None:
        # Simulate a pipeline shard: slice the trunk so the first
        # full-attention layer (global 3) lands at local index 1 — a linear
        # position in args terms. The 27B's 21-layer thirds hit exactly
        # this; the 2B's 12-layer halves dodged it (12 % 4 == 0).
        from skulk.worker.engines.mlx.drafters.introspection import (
            build_sibling_attention_layer,
        )

        class _Model:
            def __init__(self) -> None:
                self.language_model = qwen3_5.TextModel(_hybrid_args())

        model = _Model()
        trunk = model.language_model.model
        trunk.layers = trunk.layers[2:]  # offset 2: 2 % 4 != 0

        sibling = build_sibling_attention_layer(model)
        assert sibling is not None
        assert getattr(sibling, "self_attn", None) is not None


# ---------------------------------------------------------------------------
# Quantize-on-load — sidecar matches the target's precision
# ---------------------------------------------------------------------------


def _tiny_args_quantizable() -> qwen3_5.TextModelArgs:
    """Toy args big enough for mlx quantization (group size 32)."""
    return qwen3_5.TextModelArgs(  # pyright: ignore[reportCallIssue]
        model_type="qwen3_5",
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        vocab_size=64,
        full_attention_interval=1,
        tie_word_embeddings=True,
    )


class TestQuantizeOnLoad:
    """The builder quantizes the sidecar to the target's (group_size, bits).

    Published sidecars ship bf16; leaving them unquantized on a quantized
    target makes the draft forward read several times more weight bytes than
    the verifier's own layers (measured 10.7ms/draft vs 1.2ms observe on
    Qwen3.5-9B-4bit).
    """

    def _quantized_target(self) -> object:
        from skulk.worker.engines.mlx.drafters.introspection import get_trunk

        class _Model:
            def __init__(self) -> None:
                self.language_model = qwen3_5.TextModel(_tiny_args_quantizable())

        model = _Model()
        trunk = get_trunk(model)
        assert isinstance(trunk, nn.Module)
        quantize_module = cast("Callable[..., object]", nn.quantize)
        quantize_module(trunk, group_size=32, bits=4)
        return model

    def _sidecar_weights(self) -> dict[str, mx.array]:
        hidden = 64
        w: dict[str, mx.array] = {
            "mtp.pre_fc_norm_hidden.weight": mx.ones((hidden,)),
            "mtp.pre_fc_norm_embedding.weight": mx.ones((hidden,)),
            "mtp.fc.weight": mx.zeros((hidden, 2 * hidden)),
            "mtp.norm.weight": mx.ones((hidden,)),
        }
        layer = qwen3_5.DecoderLayer(_tiny_args_quantizable(), layer_idx=0)
        for name, value in tree_flatten(layer.parameters()):
            w[f"mtp.layers.0.{name}"] = value
        return w

    def test_detect_quantization(self) -> None:
        from skulk.worker.engines.mlx.drafters.introspection import (
            detect_quantization,
        )

        assert detect_quantization(self._quantized_target()) == (32, 4)
        assert detect_quantization(_TinyModel()) is None

    def test_builder_quantizes_sidecar_to_match(self) -> None:
        drafter = build_qwen_sidecar_drafter(
            self._quantized_target(),
            self._sidecar_weights(),
            norm_convention="actual_scale",
            concat_order="embed_first",
        )
        assert drafter is not None
        assert isinstance(drafter._fc, nn.QuantizedLinear)
        block_modules = cast(
            "list[tuple[str, nn.Module]]",
            drafter._block.named_modules(),  # pyright: ignore[reportUnknownMemberType]
        )
        assert any(
            isinstance(module, nn.QuantizedLinear) for _name, module in block_modules
        )
        # The quantized drafter still drafts end-to-end.
        drafter.begin_request([])
        out = drafter.draft(mx.zeros(64), next_token=1)
        assert out.shape == (1, 64)

    def test_bf16_target_keeps_bf16_sidecar(self) -> None:
        drafter = build_qwen_sidecar_drafter(
            _TinyModel(),
            _make_qwen35_weights(),
            norm_convention="actual_scale",
            concat_order="embed_first",
        )
        assert drafter is not None
        assert isinstance(drafter._fc, nn.Linear)
        assert not isinstance(drafter._fc, nn.QuantizedLinear)
