"""Unit tests for MLX runtime patches."""

from __future__ import annotations

import sys
import types


class TestGdnPatchModuleSweep:
    def test_sweep_skips_foreign_lazy_modules(self) -> None:
        """patch_gdn_softplus must not probe non-mlx modules for compute_g.

        Regression test: transformers 5.10's lazy top-level namespace
        resolves a "compute_g" attribute probe by importing an unrelated
        aria image-processing module that requires torchvision — crashing
        the runner at startup. The sweep must only touch mlx_lm/mlx_vlm
        modules.
        """

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
