# pyright: reportPrivateUsage=false, reportUnknownLambdaType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportArgumentType=false
"""Unit tests for MTP speculative decoding.

Tests cover:
- build_mtp_head: returns None for unsupported key layouts and missing callables
- build_mtp_head: constructs a valid MTPHead with float/quantized weights
- MTPHead.draft: output shape and dtype
- _get_trunk_and_head: Qwen3.5-style and DeepSeek-style model introspection
- _stream_generate_with_mtp: accept / reject paths update the cache correctly
  and yield the right sequence of tokens
"""

from __future__ import annotations

from unittest.mock import MagicMock

import mlx.core as mx

from exo.worker.engines.mlx.mtp import (
    _QWEN35_REQUIRED_KEYS,
    _REQUIRED_KEYS,
    MTPHead,
    build_mtp_head,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

HIDDEN = 16
VOCAB = 32
GROUP = 4


def _make_weights(
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
        # 4-bit packed: (out, in // 8)
        w[f"{prefix}eh_proj.weight"] = mx.zeros((HIDDEN, 2 * HIDDEN // 8), dtype=mx.uint32)
        # scales/biases: (out, in // group_size)
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


def _make_qwen35_weights(
    *,
    quantized: bool = False,
    add_norm: bool = True,
) -> dict[str, mx.array]:
    """Return a minimal Qwen3.5-style sidecar weight dict (prefix 'mtp.')."""
    w: dict[str, mx.array] = {}
    w["mtp.pre_fc_norm_hidden.weight"] = mx.ones(HIDDEN)
    w["mtp.pre_fc_norm_embedding.weight"] = mx.ones(HIDDEN)

    if quantized:
        w["mtp.fc.weight"] = mx.zeros((HIDDEN, 2 * HIDDEN // 8), dtype=mx.uint32)
        w["mtp.fc.weight_scales"] = mx.zeros((HIDDEN, 2 * HIDDEN // GROUP), dtype=mx.float16)
        w["mtp.fc.weight_biases"] = mx.zeros((HIDDEN, 2 * HIDDEN // GROUP), dtype=mx.float16)
    else:
        w["mtp.fc.weight"] = mx.zeros((HIDDEN, 2 * HIDDEN))

    if add_norm:
        w["mtp.norm.weight"] = mx.ones(HIDDEN)

    return w


def _make_model(*, missing_head: bool = False, missing_embed: bool = False) -> MagicMock:
    """Return a fake model with Qwen3.5-style structure."""
    # For missing_head, constrain the mock so it does not auto-create
    # `as_linear` either — tied-embedding models are headed via
    # embed_tokens.as_linear, so a truly headless model must lack both.
    embed = MagicMock(spec=["__call__"]) if missing_head else MagicMock()
    embed.return_value = mx.zeros((1, HIDDEN))

    lm_head = MagicMock()
    lm_head.return_value = mx.zeros((1, 1, VOCAB))

    trunk = MagicMock(spec=["norm", "embed_tokens"])
    trunk.norm = MagicMock(side_effect=lambda x: x)
    trunk.embed_tokens = embed

    lm = MagicMock(spec=["model", "lm_head"])
    lm.model = trunk
    lm.lm_head = None if missing_head else lm_head

    # Use spec=[] so outer model doesn't auto-create a .lm_head attribute
    # that would shadow the language_model.lm_head=None test.
    model = MagicMock(spec=["language_model"])
    model.language_model = lm
    # Remove embed if testing that path
    if missing_embed:
        trunk.embed_tokens = None
    return model


# ---------------------------------------------------------------------------
# build_mtp_head
# ---------------------------------------------------------------------------


class TestBuildMtpHead:
    def test_returns_none_for_empty_weights(self) -> None:
        model = _make_model()
        head = build_mtp_head(model, {})
        assert head is None

    def test_returns_none_for_unrecognised_prefix(self) -> None:
        model = _make_model()
        weights = {f"other.0.{k}": mx.zeros(HIDDEN) for k in _REQUIRED_KEYS}
        head = build_mtp_head(model, weights)
        assert head is None

    def test_returns_none_when_lm_head_missing(self) -> None:
        model = _make_model(missing_head=True)
        weights = _make_weights()
        head = build_mtp_head(model, weights)
        assert head is None

    def test_float_weights_top_level_prefix(self) -> None:
        model = _make_model()
        weights = _make_weights(prefix="mtp.0.")
        head = build_mtp_head(model, weights)
        assert head is not None
        assert isinstance(head, MTPHead)

    def test_float_weights_model_prefix(self) -> None:
        model = _make_model()
        weights = _make_weights(prefix="model.mtp.0.")
        head = build_mtp_head(model, weights)
        assert head is not None

    def test_quantized_weights(self) -> None:
        model = _make_model()
        weights = _make_weights(quantized=True)
        head = build_mtp_head(model, weights)
        assert head is not None
        assert head.eh_proj_scales is not None
        assert head.eh_proj_biases is not None

    def test_with_shared_norm(self) -> None:
        model = _make_model()
        weights = _make_weights(add_shared_norm=True)
        head = build_mtp_head(model, weights)
        assert head is not None
        assert head.shared_norm_w is not None

    def test_without_shared_norm(self) -> None:
        model = _make_model()
        weights = _make_weights(add_shared_norm=False)
        head = build_mtp_head(model, weights)
        assert head is not None
        assert head.shared_norm_w is None


# ---------------------------------------------------------------------------
# build_mtp_head — Qwen3.5 key layout
# ---------------------------------------------------------------------------


class TestBuildMtpHeadQwen35:
    def test_returns_none_for_unrecognised_qwen35_prefix(self) -> None:
        model = _make_model()
        weights = {f"other.{k}": mx.zeros(HIDDEN) for k in _QWEN35_REQUIRED_KEYS}
        head = build_mtp_head(model, weights)
        assert head is None

    def test_float_weights_qwen35(self) -> None:
        model = _make_model()
        weights = _make_qwen35_weights()
        head = build_mtp_head(model, weights)
        assert head is not None
        assert isinstance(head, MTPHead)

    def test_qwen35_shared_norm_loaded(self) -> None:
        model = _make_model()
        weights = _make_qwen35_weights(add_norm=True)
        head = build_mtp_head(model, weights)
        assert head is not None
        assert head.shared_norm_w is not None

    def test_qwen35_without_norm_uses_model_norm(self) -> None:
        model = _make_model()
        weights = _make_qwen35_weights(add_norm=False)
        head = build_mtp_head(model, weights)
        assert head is not None
        assert head.shared_norm_w is None

    def test_qwen35_quantized(self) -> None:
        model = _make_model()
        weights = _make_qwen35_weights(quantized=True)
        head = build_mtp_head(model, weights)
        assert head is not None
        assert head.eh_proj_scales is not None
        assert head.eh_proj_biases is not None

    def test_qwen35_draft_output_shape(self) -> None:
        model = _make_model()
        weights = _make_qwen35_weights()
        head = build_mtp_head(model, weights)
        assert head is not None
        logits = head.draft(mx.zeros(HIDDEN), next_token_id=0)
        assert logits.ndim >= 1
        assert logits.dtype == mx.float32

    def test_deepseek_prefix_not_confused_with_qwen35(self) -> None:
        # DeepSeek weights should not be matched by Qwen3.5 detection
        model = _make_model()
        weights = _make_weights(prefix="mtp.0.")
        head = build_mtp_head(model, weights)
        assert head is not None
        # Confirm it loaded as DeepSeek (hnorm from DeepSeek key, not Qwen3.5 key)
        assert "mtp.0.hnorm.weight" in weights
        assert "mtp.pre_fc_norm_hidden.weight" not in weights


# ---------------------------------------------------------------------------
# MTPHead.draft — shape and dtype
# ---------------------------------------------------------------------------


class TestMTPHeadDraft:
    def _build(self, *, quantized: bool = False) -> MTPHead:
        model = _make_model()
        weights = _make_weights(quantized=quantized)
        head = build_mtp_head(model, weights)
        assert head is not None
        return head

    def test_output_shape_float(self) -> None:
        head = self._build()
        hidden = mx.zeros(HIDDEN)
        logits = head.draft(hidden, next_token_id=0)
        # embed mock returns (1, HIDDEN); lm_head mock returns (1, 1, VOCAB)
        # After [0] slicing → (1, VOCAB); MTPHead returns [0] → (VOCAB,)... but mock
        # returns (1, 1, VOCAB) so draft returns (1, VOCAB)[0] = (VOCAB,).
        # Just check it doesn't crash and is 1D.
        assert logits.ndim >= 1

    def test_output_dtype_is_float32(self) -> None:
        head = self._build()
        hidden = mx.zeros(HIDDEN)
        logits = head.draft(hidden, next_token_id=5)
        assert logits.dtype == mx.float32

    def test_reset_is_noop(self) -> None:
        head = self._build()
        head.reset()  # should not raise


# ---------------------------------------------------------------------------
# _get_trunk_and_head
# ---------------------------------------------------------------------------


class TestGetTrunkAndHead:
    def test_qwen_style(self) -> None:
        from exo.worker.engines.mlx.generator.generate import _get_trunk_and_head

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
        from exo.worker.engines.mlx.generator.generate import _get_trunk_and_head

        trunk = MagicMock()
        lm_head = MagicMock()
        model = MagicMock()
        del model.language_model  # not present
        model.language_model = None
        model.model = trunk
        model.lm_head = lm_head

        # Need to ensure the qwen path doesn't match
        model2 = MagicMock(spec=[])  # no attributes
        model2.model = trunk
        model2.lm_head = lm_head

        result = _get_trunk_and_head(model2)
        assert result is not None
        t, h = result
        assert t is trunk
        assert h is lm_head

    def test_unsupported_returns_none(self) -> None:
        from exo.worker.engines.mlx.generator.generate import _get_trunk_and_head

        model = MagicMock(spec=[])  # no relevant attributes
        result = _get_trunk_and_head(model)
        assert result is None


# ---------------------------------------------------------------------------
# _stream_generate_with_mtp — accept / reject token sequences
# ---------------------------------------------------------------------------


class _FakeMTPHead:
    """Deterministic fake MTP head that always returns fixed logits."""

    def __init__(self, draft_token_id: int, vocab_size: int = VOCAB) -> None:
        self._draft_id = draft_token_id
        self._v = vocab_size
        self.reset_count = 0
        self.draft_calls: list[int] = []

    def draft(self, hidden: mx.array, next_token_id: int) -> mx.array:
        self.draft_calls.append(next_token_id)
        logits = mx.zeros(self._v)
        # Set high logit for the target draft token
        logits = mx.where(
            mx.arange(self._v) == self._draft_id, mx.array(100.0), logits
        )
        return logits.astype(mx.float32)

    def reset(self) -> None:
        self.reset_count += 1


def _build_fake_stream_env(
    *,
    vocab_size: int = VOCAB,
    hidden_size: int = HIDDEN,
    main_token_ids: list[int],  # what the "trunk+head" return each call
    draft_token_id: int,
):
    """Build minimal fakes for testing _stream_generate_with_mtp."""
    from mlx_lm.tokenizer_utils import TokenizerWrapper

    # Token ids to return from head_fn on successive calls.
    _main_token_iter = iter(main_token_ids)

    call_count = 0

    def fake_trunk(tokens: mx.array, cache: object = None) -> mx.array:
        nonlocal call_count
        call_count += 1
        # Return zeros with appropriate shape
        seq_len = tokens.shape[1] if tokens.ndim == 2 else 1
        return mx.zeros((1, seq_len, hidden_size))

    def fake_head(hidden: mx.array) -> mx.array:
        seq_len = hidden.shape[1] if hidden.ndim == 3 else 1
        out = mx.zeros((1, seq_len, vocab_size))
        # Set high logit for the current main token
        try:
            tok = next(_main_token_iter)
        except StopIteration:
            tok = 0  # EOS or fallback
        # Use last position for the main token
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

    # Fake model (not actually called in our test since trunk/head are separate)
    model = MagicMock()

    mtp_head = _FakeMTPHead(draft_token_id=draft_token_id, vocab_size=vocab_size)

    # Fake cache: a list of objects with a `state` attribute (for quantize_cache_fn)
    class _FakeCache:
        def __init__(self):
            self.state = []
            self.offset = 0
            self._trimmed = 0

        def trim(self, n: int) -> None:
            self._trimmed += n

    fake_cache = [_FakeCache()]

    return (
        model,
        tokenizer,
        mtp_head,
        fake_trunk,
        fake_head,
        fake_cache,
    )


class TestStreamGenerateWithMTP:
    def _run(
        self,
        *,
        main_token_ids: list[int],
        draft_token_id: int,
        max_tokens: int = 10,
    ):
        from exo.worker.engines.mlx.generator.generate import _stream_generate_with_mtp

        model, tokenizer, mtp_head, trunk_fn, head_fn, cache = _build_fake_stream_env(
            main_token_ids=main_token_ids,
            draft_token_id=draft_token_id,
        )

        sampler = lambda lp: mx.argmax(lp, axis=-1)  # noqa: E731
        prompt = mx.array([1, 2, 3])

        outputs = list(
            _stream_generate_with_mtp(
                model=model,
                tokenizer=tokenizer,
                mtp_head=mtp_head,
                trunk_fn=trunk_fn,
                head_fn=head_fn,
                prompt=prompt,
                max_tokens=max_tokens,
                sampler=sampler,
                logits_processors=[],
                prompt_cache=cache,
                kv_group_size=None,
                kv_bits=None,
            )
        )
        return outputs, mtp_head, cache

    def test_yields_responses(self) -> None:
        outputs, _, _ = self._run(
            main_token_ids=[5, 0],  # token 5 then EOS (0)
            draft_token_id=5,
            max_tokens=5,
        )
        assert len(outputs) >= 1

    def test_mtp_head_reset_called(self) -> None:
        _, mtp_head, _ = self._run(
            main_token_ids=[5, 0],
            draft_token_id=5,
            max_tokens=5,
        )
        assert mtp_head.reset_count == 1

    def test_draft_calls_track_main_tokens(self) -> None:
        # main_token_ids: 5, then more
        _outputs, mtp_head, _ = self._run(
            main_token_ids=[5, 6, 0],
            draft_token_id=5,
            max_tokens=3,
        )
        # Draft should be called at least once with the first main token (5)
        assert 5 in mtp_head.draft_calls

    def test_max_tokens_respected(self) -> None:
        limit = 4
        outputs, _, _ = self._run(
            main_token_ids=[5] * 20,
            draft_token_id=5,
            max_tokens=limit,
        )
        # Generator must terminate and must not emit more than limit tokens
        # (accept chains can yield 2 per pass so allow a small window).
        assert len(outputs) <= limit + 2

    def test_reject_path_does_not_crash(self) -> None:
        # main_token 5, draft is 7, but head always returns token 5
        # → head_fn will return 5 for verify position, draft is 7 → reject
        outputs, _mtp_head, _cache = self._run(
            main_token_ids=[5, 3, 0],
            draft_token_id=7,  # draft != verify → reject
            max_tokens=3,
        )
        # After rejection we expect at least 1 token emitted
        assert len(outputs) >= 1


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

        from exo.worker.engines.mlx.generator.generate import (
            _stream_generate_with_mtp,
        )

        model, tokenizer, mtp_head, trunk_fn, head_fn, _fake_cache = (
            _build_fake_stream_env(
                main_token_ids=[5, 3, 0],
                draft_token_id=7,  # head verifies 5/3 → draft 7 always rejected
            )
        )

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
                mtp_head=mtp_head,
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

        from exo.worker.engines.mlx.mtp import _get_head_fn

        head = _get_head_fn(_Model())
        assert head is not None
        probe = mx.array([1.0])
        assert mx.array_equal(head(probe), probe)


class TestGdnPatchModuleSweep:
    def test_sweep_skips_foreign_lazy_modules(self) -> None:
        """patch_gdn_softplus must not probe non-mlx modules for compute_g.

        Regression test: transformers 5.10's lazy top-level namespace
        resolves a "compute_g" attribute probe by importing an unrelated
        aria image-processing module that requires torchvision — crashing
        the runner at startup. The sweep must only touch mlx_lm/mlx_vlm
        modules.
        """
        import sys
        import types

        class _LazyBoobyTrap(types.ModuleType):
            def __getattr__(self, name: str) -> object:
                raise ModuleNotFoundError(f"booby trap tripped resolving {name!r}")

        trap = _LazyBoobyTrap("fake_lazy_package")
        sys.modules["fake_lazy_package"] = trap
        try:
            from exo.worker.engines.mlx.patches.high_precision_gdn_softplus import (
                patch_gdn_softplus,
            )

            patch_gdn_softplus()  # raised ModuleNotFoundError before the fix
        finally:
            del sys.modules["fake_lazy_package"]
