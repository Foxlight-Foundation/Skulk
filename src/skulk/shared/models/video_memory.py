"""Artifact-derived working weights for the pinned single-host video pipeline."""

from skulk.shared.models.model_cards import ModelCard
from skulk.shared.types.memory import Memory


def staged_video_weights(
    model_card: ModelCard, *, resolved_backend: str | None
) -> Memory | None:
    """Return a conservative staged weight envelope, or retain ordinary accounting.

    The owned MLX engine pins low-memory execution and disables its allocator
    cache before loading modules. Its text encoder and connector are released
    before loading the remaining components. Charge the larger group, including
    every other selected byte, rather than treating the download as one resident
    allocation. Request workspaces still need separate admission in the runner.
    This follows the pack layout and engine behavior, independent of GPU names
    or benchmark coverage. Unknown layouts keep the full-artifact estimate.
    """
    bundle = model_card.artifact_bundle
    video = model_card.video
    if (
        resolved_backend != "mlx_video-metal"
        or model_card.family != "ltx-2.5"
        or bundle is None
        or video is None
        or video.transformer_variant is None
    ):
        return None
    files = {entry.path: entry.size_bytes for entry in bundle.files}
    required = {
        "text_encoder.safetensors",
        "connector.safetensors",
        f"transformer-{video.transformer_variant}.safetensors",
        "vae_encoder_conv.safetensors",
        "vae_decoder_conv.safetensors",
        "audio_vae.safetensors",
        "vocoder.safetensors",
    }
    if not required <= files.keys():
        return None
    encoder_bytes = files["text_encoder.safetensors"] + files["connector.safetensors"]
    selected_bytes = max(model_card.storage_size.in_bytes, sum(files.values()))
    return Memory.from_bytes(max(encoder_bytes, selected_bytes - encoder_bytes))
