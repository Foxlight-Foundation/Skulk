"""Speculative-decoding drafters for the MLX engine.

A *drafter* is any mechanism that proposes candidate tokens cheaply so the
main model can verify them in a batched forward pass. Skulk supports (and
plans to support) several mechanisms with very different state needs:

- Qwen3.5/3.6 sidecar MTP heads (projection + one transformer block,
  private KV cache) — :mod:`.qwen_sidecar`
- DeepSeek V3/R1 sidecar MTP heads (projection-only) — :mod:`.deepseek_sidecar`
- Gemma 4 assistant models (separate 4-layer drafter attending over the
  *target's* KV cache) — planned, see the gemma4-mtp initiative
- Nemotron and future families — unknown shapes, deliberately unconstrained

The generation loop in :mod:`exo.worker.engines.mlx.generator.generate` talks
only to the :class:`~exo.worker.engines.mlx.drafters.protocol.Drafter`
protocol; everything family-specific lives behind it. Family-specific *facts*
(norm conventions, concat orders, key layouts) are declarative data resolved
by :mod:`.builder` from layout-keyed defaults plus model-card overrides —
never constants buried in drafter code.
"""

from exo.worker.engines.mlx.drafters.builder import build_drafter
from exo.worker.engines.mlx.drafters.protocol import Drafter

__all__ = ["Drafter", "build_drafter"]
