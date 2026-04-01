"""Runtime patches applied to mlx-lm before inference.

These patches fix correctness and performance issues in upstream mlx-lm that
haven't been merged yet, or that require exo-specific behaviour. Every patch
is idempotent and applied once via :func:`apply_mlx_patches`.
"""

from exo.worker.engines.mlx.patches.high_precision_gdn_softplus import (
    patch_gdn_softplus,
)
from exo.worker.engines.mlx.patches.opt_batch_gen import apply_batch_gen_patch
from exo.worker.engines.mlx.patches.standard_yarn_rope import patch_yarn_rope

_applied = False


def apply_mlx_patches() -> None:
    """Apply all mlx-lm runtime patches (idempotent)."""
    global _applied
    if _applied:
        return
    _applied = True
    patch_yarn_rope()
    patch_gdn_softplus()
    apply_batch_gen_patch()
