from typing import cast

import mlx.core as mx


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def randomized_signs(dim: int, seed: int) -> mx.array:
    indices = mx.arange(dim, dtype=mx.int32)
    return (((indices * 1103515245 + seed) % 2).astype(mx.float32) * 2.0) - 1.0


def hadamard_transform(x: mx.array) -> mx.array:
    dim = int(x.shape[-1])
    if not is_power_of_two(dim):
        raise ValueError(f"Hadamard transform requires power-of-two dimension, got {dim}")

    output = x
    step = 1
    while step < dim:
        reshaped = output.reshape(*output.shape[:-1], dim // (2 * step), 2, step)
        left = reshaped[..., 0, :]
        right = reshaped[..., 1, :]
        output = mx.concatenate([left + right, left - right], axis=-1).reshape(
            *x.shape[:-1],
            dim,
        )
        step *= 2
    return cast(mx.array, output / (dim**0.5))


def randomized_hadamard_transform(x: mx.array, signs: mx.array) -> mx.array:
    return hadamard_transform(x * signs)


def inverse_randomized_hadamard_transform(x: mx.array, signs: mx.array) -> mx.array:
    return hadamard_transform(x) * signs
