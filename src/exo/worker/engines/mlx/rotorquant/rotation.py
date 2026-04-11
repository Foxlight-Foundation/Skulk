"""IsoQuant block-diagonal quaternion rotation in pure MLX.

Each 128-element vector is split into 32 groups of 4. Each group is
rotated by a fixed unit quaternion via the Hamilton product
``q_L * v`` (treating ``v`` as a pure quaternion). The inverse uses
``conj(q_L) * v`` because the rotation quaternion is unit.

This is mathematically equivalent to a block-diagonal SO(4) rotation
with 3 degrees of freedom per block, costing 16 multiplies + 12 adds
per group versus the ``O(d log d)`` randomized Hadamard used by the
older native TurboQuant backend.
"""

import mlx.core as mx

from exo.worker.engines.mlx.rotorquant.tables import (
    ISO3_BLOCK_SIZE,
    ISO3_GROUPS_PER_BLOCK,
    quaternion_table,
)

_qw, _qx, _qy, _qz = quaternion_table()


def _quat_mul(
    aw: mx.array,
    ax: mx.array,
    ay: mx.array,
    az: mx.array,
    bw: mx.array,
    bx: mx.array,
    by: mx.array,
    bz: mx.array,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Hamilton product ``a * b`` of two quaternions, broadcast over groups.

    Mirrors the ``quat_mul`` helper in ``ggml-iso-quant.c`` exactly.
    All inputs broadcast against each other along the trailing group axis.
    """
    rw = aw * bw - ax * bx - ay * by - az * bz
    rx = aw * bx + ax * bw + ay * bz - az * by
    ry = aw * by - ax * bz + ay * bw + az * bx
    rz = aw * bz + ax * by - ay * bx + az * bw
    return rw, rx, ry, rz


def iso3_rotate_forward(unit: mx.array) -> mx.array:
    """Apply the per-group quaternion rotation to a unit-normalized tensor.

    Args:
        unit: ``(..., 128)`` fp32 tensor with unit-norm leading slices.

    Returns:
        ``(..., 128)`` fp32 tensor in the rotated coordinate frame.
    """
    if unit.shape[-1] != ISO3_BLOCK_SIZE:
        raise ValueError(
            f"iso3_rotate_forward expects last dim {ISO3_BLOCK_SIZE}, "
            f"got {unit.shape[-1]}"
        )

    # Reshape into (..., groups, 4) so the four quaternion components map
    # cleanly onto the trailing axis. The quaternion tables broadcast as
    # (groups,) → (1, ..., groups) against the leading dims.
    grouped = unit.reshape(*unit.shape[:-1], ISO3_GROUPS_PER_BLOCK, 4)
    v0 = grouped[..., 0]
    v1 = grouped[..., 1]
    v2 = grouped[..., 2]
    v3 = grouped[..., 3]

    rw, rx, ry, rz = _quat_mul(_qw, _qx, _qy, _qz, v0, v1, v2, v3)

    rotated = mx.stack([rw, rx, ry, rz], axis=-1)
    return rotated.reshape(*unit.shape)


def iso3_rotate_inverse(rotated: mx.array) -> mx.array:
    """Inverse of :func:`iso3_rotate_forward`.

    Uses the conjugate quaternion ``(qw, -qx, -qy, -qz)`` because the
    rotation quaternion is unit.

    Args:
        rotated: ``(..., 128)`` fp32 tensor in the rotated frame.

    Returns:
        ``(..., 128)`` fp32 tensor back in the original frame.
    """
    if rotated.shape[-1] != ISO3_BLOCK_SIZE:
        raise ValueError(
            f"iso3_rotate_inverse expects last dim {ISO3_BLOCK_SIZE}, "
            f"got {rotated.shape[-1]}"
        )

    grouped = rotated.reshape(*rotated.shape[:-1], ISO3_GROUPS_PER_BLOCK, 4)
    q0 = grouped[..., 0]
    q1 = grouped[..., 1]
    q2 = grouped[..., 2]
    q3 = grouped[..., 3]

    rw, rx, ry, rz = _quat_mul(_qw, -_qx, -_qy, -_qz, q0, q1, q2, q3)

    restored = mx.stack([rw, rx, ry, rz], axis=-1)
    return restored.reshape(*rotated.shape)
