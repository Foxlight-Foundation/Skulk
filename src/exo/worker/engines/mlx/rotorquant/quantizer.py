"""3-bit IsoQuant quantizer with norm correction.

Mirrors the quantize/dequantize logic in
``johndpope/llama-cpp-turboquant/ggml/src/ggml-iso-quant.c`` but in pure
MLX. Storage is one fp16 norm and one uint8 index per element per
``(B, H, T)`` block — bit-packing into the 50-byte ``block_iso3_0``
layout is a future-work optimization that does not affect correctness.

The norm-correction trick (stored norm = ``||x|| / sqrt(Σ centroid²)``)
is critical and not mentioned in the README of either upstream project.
It absorbs the L2 shrinkage induced by centroid quantization so that
attention's inner products remain unbiased.
"""

import mlx.core as mx

from exo.worker.engines.mlx.rotorquant.rotation import (
    iso3_rotate_forward,
    iso3_rotate_inverse,
)
from exo.worker.engines.mlx.rotorquant.tables import (
    ISO3_BLOCK_SIZE,
    centroid_table,
)

_NORM_EPS = 1e-10


class IsoQuantizer:
    """Stateless 3-bit quantizer for ``head_dim``-multiple-of-128 vectors.

    Holds the precomputed centroid table and centroid-boundary midpoints
    on the active MLX device. The quantizer itself carries no per-token
    state; the cache class owns the actual storage tensors.
    """

    def __init__(self) -> None:
        self._centroids = centroid_table()
        # Boundary midpoints between adjacent centroids: a value falls
        # into bin ``i`` iff it lies between boundary i-1 and i. With 8
        # centroids there are 7 internal boundaries.
        boundaries = (self._centroids[:-1] + self._centroids[1:]) / 2.0
        self._boundary_values: list[float] = boundaries.tolist()  # type: ignore[reportAny]

    @property
    def centroids(self) -> mx.array:
        """Return the centroid table as an fp32 array of shape ``(8,)``."""
        return self._centroids

    def quantize(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Quantize a tensor whose last dim is a multiple of the block size.

        Args:
            x: ``(..., d)`` tensor where ``d`` is a multiple of 128.

        Returns:
            A pair ``(indices, norms)``:
              - ``indices``: ``(..., d)`` uint8 in ``[0, 7]``
              - ``norms``: ``(..., d // 128)`` fp16 corrected norms,
                one per iso3 block.
        """
        if x.shape[-1] % ISO3_BLOCK_SIZE != 0:
            raise ValueError(
                f"IsoQuantizer requires last dim divisible by {ISO3_BLOCK_SIZE}, "
                f"got {x.shape[-1]}"
            )

        x32 = x.astype(mx.float32)
        leading = x32.shape[:-1]
        n_blocks = x32.shape[-1] // ISO3_BLOCK_SIZE

        # Reshape so each iso3 block of 128 elements becomes its own row
        # along a new axis. The norm and rotation are then per-block.
        blocks = x32.reshape(*leading, n_blocks, ISO3_BLOCK_SIZE)

        # Per-block L2 norm — matches grp_norm in the C reference.
        norm_sq = mx.sum(blocks * blocks, axis=-1, keepdims=True)
        grp_norm = mx.sqrt(norm_sq)
        safe_norm = mx.maximum(grp_norm, mx.array(_NORM_EPS, dtype=mx.float32))
        unit = blocks / safe_norm

        # Forward quaternion rotation in the unit sphere.
        rotated = iso3_rotate_forward(unit)

        # Find nearest centroid index per coordinate. Walks the boundary
        # list once with a running ``where`` accumulator. Cheap because
        # there are only 7 boundaries for 8 centroids.
        indices = mx.zeros(rotated.shape, dtype=mx.uint8)
        for i, boundary in enumerate(self._boundary_values):
            indices = indices + (rotated > boundary).astype(mx.uint8)
            del i  # boundary index unused; the cumulative count is what matters

        # Norm correction: absorb the L2 shrinkage of the quantized unit
        # vector into the stored norm so dequantization yields an
        # unbiased magnitude. ``recon_sq = Σ centroid[idx]²`` exactly
        # matches the C reference path.
        centroid_values = self._centroids[indices.astype(mx.int32)]
        recon_sq = mx.sum(centroid_values * centroid_values, axis=-1, keepdims=True)
        recon_norm = mx.sqrt(recon_sq)
        safe_recon = mx.maximum(recon_norm, mx.array(_NORM_EPS, dtype=mx.float32))
        corrected_norm = grp_norm / safe_recon

        # Pack indices back to (..., d) and norms to (..., n_blocks).
        indices_flat = indices.reshape(*leading, n_blocks * ISO3_BLOCK_SIZE)
        norms_flat = corrected_norm.reshape(*leading, n_blocks).astype(mx.float16)
        return indices_flat, norms_flat

    def dequantize(
        self,
        indices: mx.array,
        norms: mx.array,
        output_dtype: mx.Dtype,
    ) -> mx.array:
        """Inverse of :meth:`quantize`.

        Args:
            indices: ``(..., d)`` uint8 indices, ``d`` a multiple of 128.
            norms: ``(..., d // 128)`` corrected fp16 norms.
            output_dtype: target dtype of the reconstructed tensor.

        Returns:
            ``(..., d)`` reconstructed tensor in ``output_dtype``.
        """
        if indices.shape[-1] % ISO3_BLOCK_SIZE != 0:
            raise ValueError(
                f"IsoQuantizer.dequantize requires indices last dim divisible by "
                f"{ISO3_BLOCK_SIZE}, got {indices.shape[-1]}"
            )

        leading = indices.shape[:-1]
        n_blocks = indices.shape[-1] // ISO3_BLOCK_SIZE

        # Look up centroid values for every index, then reshape to
        # per-block layout for the inverse rotation.
        centroid_values = self._centroids[indices.astype(mx.int32)]
        blocks = centroid_values.reshape(*leading, n_blocks, ISO3_BLOCK_SIZE)

        rotated_back = iso3_rotate_inverse(blocks)

        # Multiply each block by its stored corrected norm. The expand
        # broadcasts ``(..., n_blocks)`` against the trailing block dim.
        norms_f32 = norms.astype(mx.float32)[..., None]
        scaled = rotated_back * norms_f32

        flat = scaled.reshape(*leading, n_blocks * ISO3_BLOCK_SIZE)
        return flat.astype(output_dtype)
