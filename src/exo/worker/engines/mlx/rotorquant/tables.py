"""Vendored constants for IsoQuant 3-bit KV cache compression.

Tables are lifted verbatim from johndpope/llama-cpp-turboquant
(``ggml/src/ggml-iso-quant.c``, MIT-licensed). Keeping them bit-exact
against the upstream C source means our quantizer agrees with the
llama.cpp reference modulo floating-point ordering.

The 32 unit quaternions parameterize one Hamilton-product rotation per
4D group; for a 128-dimensional head this is 32 groups × 4 dims. The
8-entry centroid table is the Lloyd-Max optimum for the post-rotation
N(0, 1/d)-like coordinate distribution.
"""

from typing import Final

import mlx.core as mx

#: Native block size used by the iso3 layout. Fits exactly one Llama/Qwen
#: head dimension. Heads with other sizes must be a multiple of this and
#: are processed as multiple iso3 blocks.
ISO3_BLOCK_SIZE: Final[int] = 128

#: Number of quaternion groups per ISO3 block (block_size / 4).
ISO3_GROUPS_PER_BLOCK: Final[int] = ISO3_BLOCK_SIZE // 4

#: Bits per element after compression (3-bit indices via 8-entry codebook).
ISO3_BITS: Final[int] = 3

# Hardcoded unit quaternion components, one per 4D group (32 groups).
# Sourced verbatim from ggml-iso-quant.c lines 51-54 in
# johndpope/llama-cpp-turboquant feature/planarquant-kv-cache.
# fmt: off
_ISO3_QW: Final[tuple[float, ...]] = (
    0.5765609741, 0.3176580369, -0.3234235942, -0.5127438903,
    0.9233905673, -0.3323571086, 0.5468608141, -0.2500519454,
    -0.5812215805, 0.3228830695, -0.7299832702, -0.4535493255,
    -0.7338157296, -0.2884652913, -0.9000198841, -0.0377033800,
    0.5104404092, 0.2033989877, -0.2462528497, 0.2314069420,
    0.0072374810, 0.3923372924, 0.4958070219, -0.7235037088,
    -0.9383618832, 0.4430379272, -0.2075705230, 0.1983736306,
    -0.8834578991, 0.7389573455, -0.0156172011, 0.7738668919,
)

_ISO3_QX: Final[tuple[float, ...]] = (
    0.4450169504, -0.5780548453, 0.7089627385, -0.3940812945,
    -0.0897334740, 0.4727236331, 0.5542563796, 0.0450818054,
    -0.3657043576, -0.4298477769, 0.4666220546, 0.7556306720,
    -0.5284956098, 0.7042509317, 0.0230921544, 0.7110687494,
    0.3024962246, -0.1157865301, 0.7490812540, -0.2582575679,
    -0.2255804837, 0.3838746250, -0.3209520578, -0.3477301002,
    0.1824720055, 0.4032751918, 0.8433781862, 0.9533935785,
    -0.0620501526, 0.0927560627, 0.2964956462, 0.2402082384,
)

_ISO3_QY: Final[tuple[float, ...]] = (
    0.2695076466, -0.0201656222, -0.1687686443, -0.5415957570,
    -0.2796611190, 0.3510629535, 0.2609911859, -0.2715902030,
    -0.0937586129, 0.3095585108, -0.4123268127, -0.4394895136,
    0.0626545250, -0.4811822474, -0.0407132693, -0.4566248953,
    0.7834537029, -0.6187923551, 0.0809760988, -0.8879503012,
    -0.8928058147, 0.8350352049, -0.6994170547, 0.5606835485,
    0.2933705449, 0.7377059460, 0.4534837306, -0.0009816211,
    -0.3632916510, -0.3959124386, 0.1631654203, 0.5088164806,
)

_ISO3_QZ: Final[tuple[float, ...]] = (
    -0.6300023794, -0.7513582706, -0.6035611629, 0.5370919704,
    0.2471584976, 0.7367672324, 0.5706370473, 0.9282674193,
    0.7208684087, -0.7843156457, -0.2817355990, -0.1736787707,
    0.4222335219, -0.4350655377, 0.4333281815, 0.5333415866,
    0.1847889870, 0.7498788238, 0.6096553802, -0.3021556735,
    -0.3898189068, 0.0377884321, 0.4024685621, 0.2031257302,
    0.0107116764, -0.3112498820, 0.1999502629, -0.2273492515,
    0.2892593443, 0.5372074246, 0.9408631325, 0.2907505929,
)

#: 3-bit Lloyd-Max centroids on the post-rotation coordinate distribution.
#: Sourced verbatim from ISO_CENTROIDS_3BIT in ggml-iso-quant.c.
_ISO3_CENTROIDS: Final[tuple[float, ...]] = (
    -0.1906850000, -0.1178320000, -0.0657170000, -0.0214600000,
    0.0214600000, 0.0657170000, 0.1178320000, 0.1906850000,
)
# fmt: on


def quaternion_table() -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Return the (qw, qx, qy, qz) quaternion-component arrays as fp32.

    Each array has shape ``(ISO3_GROUPS_PER_BLOCK,)`` and the four arrays
    together describe one unit quaternion per 4D rotation group.
    """
    return (
        mx.array(_ISO3_QW, dtype=mx.float32),
        mx.array(_ISO3_QX, dtype=mx.float32),
        mx.array(_ISO3_QY, dtype=mx.float32),
        mx.array(_ISO3_QZ, dtype=mx.float32),
    )


def centroid_table() -> mx.array:
    """Return the 8-entry 3-bit centroid array as fp32."""
    return mx.array(_ISO3_CENTROIDS, dtype=mx.float32)
