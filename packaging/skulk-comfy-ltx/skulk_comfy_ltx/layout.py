"""Translate channel-first Comfy latents to the reference's token-major states."""

from collections.abc import Sequence

from skulk_comfy_ltx.res2s import Latents


def patch_modalities(values: Latents) -> Latents:
    """Patch joint AV states using the reference's unit video/audio patches.

    Video tokens contain channels, while audio tokens contain channels and mel
    bins. This order determines both seeded noise coordinates and reductions.
    """
    if len(values) != 2 or values[0].ndim != 5 or values[1].ndim != 4:
        raise ValueError("LTX HQ requires channel-first joint video/audio latents")
    video, audio = values
    if video.shape[0] != audio.shape[0]:
        raise ValueError("LTX modalities must share their batch size")
    return (
        video.permute(0, 2, 3, 4, 1).reshape(video.shape[0], -1, video.shape[1]),
        audio.permute(0, 2, 1, 3).reshape(audio.shape[0], audio.shape[2], -1),
    )


def unpatch_modalities(values: Latents, shapes: Sequence[Sequence[int]]) -> Latents:
    """Restore exact Comfy modality geometry, including appended video guides."""
    if len(values) != 2 or len(shapes) != 2 or len(shapes[0]) != 5 or len(shapes[1]) != 4:
        raise ValueError("LTX HQ requires two token states and their AV geometry")
    video, audio = values
    vb, vc, vt, vh, vw = shapes[0]
    ab, ac, at, af = shapes[1]
    if tuple(video.shape) != (vb, vt * vh * vw, vc) or tuple(audio.shape) != (ab, at, ac * af):
        raise ValueError("LTX token states differ from their declared modality geometry")
    return (
        video.reshape(vb, vt, vh, vw, vc).permute(0, 4, 1, 2, 3),
        audio.reshape(ab, at, ac, af).permute(0, 2, 1, 3),
    )
