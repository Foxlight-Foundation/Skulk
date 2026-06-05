# pyright: reportPrivateUsage=false
"""Unit tests for distribution-preserving speculative sampling (#180)."""

from __future__ import annotations

import math

import mlx.core as mx

from exo.worker.engines.mlx.generator.speculative_sampling import (
    SamplingParams,
    ratio_accept,
    residual_sample,
    warp_to_probs,
)


def _logprobs(probs: list[float]) -> mx.array:
    return mx.log(mx.array(probs))


class TestSamplingParams:
    def test_greedy_detection(self) -> None:
        assert SamplingParams(temperature=0.0).is_greedy
        assert not SamplingParams(temperature=0.7).is_greedy


class TestWarpToProbs:
    def test_temp_one_no_filters_is_softmax(self) -> None:
        lp = _logprobs([0.5, 0.3, 0.15, 0.05])
        params = SamplingParams(temperature=1.0, top_p=1.0, min_p=0.0, top_k=0)
        probs = warp_to_probs(lp, params)
        assert mx.allclose(probs, mx.exp(lp), atol=1e-5)

    def test_top_k_zeroes_tail(self) -> None:
        lp = _logprobs([0.5, 0.3, 0.15, 0.05])
        params = SamplingParams(temperature=1.0, top_p=1.0, min_p=0.0, top_k=2)
        probs = warp_to_probs(lp, params)
        assert float(probs[2].item()) == 0.0
        assert float(probs[3].item()) == 0.0
        assert math.isclose(float(mx.sum(probs).item()), 1.0, rel_tol=1e-5)

    def test_min_p_filters_relative_to_top(self) -> None:
        lp = _logprobs([0.90, 0.08, 0.015, 0.005])
        # min_p=0.1: threshold = 0.1 * 0.90 = 0.09 -> only token 0 survives.
        params = SamplingParams(temperature=1.0, top_p=1.0, min_p=0.1, top_k=0)
        probs = warp_to_probs(lp, params)
        assert math.isclose(float(probs[0].item()), 1.0, rel_tol=1e-4)

    def test_temperature_flattens(self) -> None:
        lp = _logprobs([0.7, 0.2, 0.07, 0.03])
        params_sharp = SamplingParams(temperature=0.5, top_p=1.0, min_p=0.0)
        params_flat = SamplingParams(temperature=2.0, top_p=1.0, min_p=0.0)
        sharp = warp_to_probs(lp, params_sharp)
        flat = warp_to_probs(lp, params_flat)
        assert float(sharp[0].item()) > float(flat[0].item())


class TestRatioAccept:
    def test_always_accepts_when_target_dominates(self) -> None:
        q = mx.array([0.5, 0.5, 0.0, 0.0])
        p = mx.array([0.8, 0.2, 0.0, 0.0])
        # p(0)/q(0) = 1.6 >= 1 -> deterministic accept.
        assert ratio_accept(0, q, p) is True

    def test_always_rejects_when_target_excludes_token(self) -> None:
        q = mx.array([0.5, 0.5, 0.0, 0.0])
        p = mx.array([0.0, 1.0, 0.0, 0.0])
        assert ratio_accept(0, q, p) is False

    def test_defensive_reject_on_zero_q(self) -> None:
        q = mx.array([0.0, 1.0, 0.0, 0.0])
        p = mx.array([0.5, 0.5, 0.0, 0.0])
        assert ratio_accept(0, q, p) is False


class TestResidualSample:
    def test_samples_only_where_p_exceeds_q(self) -> None:
        q = mx.array([0.6, 0.4, 0.0, 0.0])
        p = mx.array([0.1, 0.1, 0.8, 0.0])
        # residual = max(p-q, 0) = [0, 0, 0.8, 0] -> token 2 always.
        mx.random.seed(7)
        for _ in range(8):
            assert residual_sample(q, p) == 2

    def test_degenerate_falls_back_to_target(self) -> None:
        q = mx.array([0.5, 0.5])
        p = mx.array([0.5, 0.5])
        mx.random.seed(7)
        token = residual_sample(q, p)
        assert token in (0, 1)


class TestDistributionPreservation:
    def test_committed_tokens_follow_target_distribution(self) -> None:
        """The whole point of #180: accept-with-ratio + residual-resample
        must produce exact samples from p regardless of q.

        Vectorized simulation of the per-token procedure (mirroring
        ratio_accept / residual_sample math) over a deliberately skewed
        draft distribution; the committed histogram must match p within
        sampling error.
        """
        mx.random.seed(1234)
        n = 40_000
        q = mx.array([0.70, 0.15, 0.10, 0.05])  # overconfident drafter
        p = mx.array([0.25, 0.25, 0.25, 0.25])  # uniform target

        draws = mx.random.categorical(mx.log(q), num_samples=n)  # (n,)
        ratios = p[draws] / q[draws]
        accept = mx.random.uniform(shape=(n,)) < mx.minimum(ratios, 1.0)

        residual = mx.maximum(p - q, 0.0)
        residual = residual / mx.sum(residual)
        resamples = mx.random.categorical(mx.log(residual + 1e-30), num_samples=n)

        committed = mx.where(accept, draws, resamples)
        counts = mx.array(
            [
                int(mx.sum(mx.array(committed == i)).item())
                for i in range(4)
            ]
        ).astype(mx.float32)
        freqs = counts / n
        # 4 cells, n=40k: 3-sigma sampling error ~0.007 per cell.
        assert mx.allclose(freqs, p, atol=0.015), f"freqs={freqs.tolist()} vs p={p.tolist()}"
