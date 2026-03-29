import math

import mlx.core as mx

from exo.worker.engines.mlx.turboquant.rotation import (
    inverse_randomized_hadamard_transform,
    is_power_of_two,
    randomized_hadamard_transform,
    randomized_signs,
)

_GAUSSIAN_CODEBOOKS: dict[int, list[float]] = {
    1: [-0.7979, 0.7979],
    2: [-1.5104, -0.4528, 0.4528, 1.5104],
    3: [-2.1520, -1.3440, -0.7560, -0.2451, 0.2451, 0.7560, 1.3440, 2.1520],
    4: [
        -2.7326,
        -2.0690,
        -1.6180,
        -1.2562,
        -0.9423,
        -0.6568,
        -0.3881,
        -0.1284,
        0.1284,
        0.3881,
        0.6568,
        0.9423,
        1.2562,
        1.6180,
        2.0690,
        2.7326,
    ],
}


class TurboQuantizer:
    def __init__(self, dim: int, bits: int, seed: int) -> None:
        if bits not in _GAUSSIAN_CODEBOOKS:
            raise ValueError(f"Unsupported TurboQuant bit width: {bits}")
        if not is_power_of_two(dim):
            raise ValueError(
                f"TurboQuant currently requires power-of-two head dimensions, got {dim}"
            )

        self.dim = dim
        self.bits = bits
        self.signs = randomized_signs(dim, seed)
        self.centroids = mx.array(_GAUSSIAN_CODEBOOKS[bits], dtype=mx.float32)
        self.boundary_values = [
            (left + right) / 2.0
            for left, right in zip(
                _GAUSSIAN_CODEBOOKS[bits][:-1],
                _GAUSSIAN_CODEBOOKS[bits][1:],
                strict=True,
            )
        ]
        self.scale = 1.0 / math.sqrt(dim)

    def quantize(self, x: mx.array) -> tuple[mx.array, mx.array]:
        x32 = x.astype(mx.float32)
        norms = mx.sqrt(mx.sum(x32 * x32, axis=-1, keepdims=True))
        safe_norms = mx.maximum(norms, 1e-8)
        unit = x32 / safe_norms
        rotated = randomized_hadamard_transform(unit, self.signs)
        scaled = rotated / self.scale

        indices = mx.zeros(scaled.shape, dtype=mx.uint8)
        for boundary in self.boundary_values:
            indices = indices + (scaled > boundary).astype(mx.uint8)
        return indices, mx.squeeze(norms, axis=-1)

    def dequantize(
        self,
        indices: mx.array,
        norms: mx.array,
        output_dtype: mx.Dtype,
    ) -> mx.array:
        flat = indices.reshape(-1).astype(mx.int32)
        values = self.centroids[flat].reshape(*indices.shape)
        values = values * self.scale
        restored = inverse_randomized_hadamard_transform(values, self.signs)
        return (restored * norms[..., None]).astype(output_dtype)
