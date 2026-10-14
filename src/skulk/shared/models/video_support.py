"""Request-level engine compatibility, independent of hardware qualification."""

from skulk.provisioning.mlx_video import NativeVideoFeature, video_features_for_build
from skulk.shared.backends import engine_of
from skulk.shared.models.model_cards import ModelCard
from skulk.shared.types.video import VideoGenerationTaskParams


def video_request_supported(card: ModelCard, params: VideoGenerationTaskParams, backend: str | None, build: str | None) -> bool:
    """Check exact runner controls without requiring an empirical hardware claim.

    Intrinsic card validation remains separate from installed-build capability.
    Missing advanced controls exclude that placement, allowing another capable
    instance to serve the request instead of failing on a preferred old engine.
    """
    if card.family != "ltx-2.5":
        return True
    if card.video is None or backend is None:
        return False
    try:
        params.validate_family(card.family, card.video)
    except ValueError:
        return False
    pipeline = params.pipeline or (params.editing.operation if params.editing is not None else "distilled_two_stage" if card.video.transformer_variant == "distilled" else "dev_two_stage")
    engine = engine_of(backend)
    if engine == "comfy":
        return pipeline in ("distilled_two_stage", "dev_two_stage") and params.editing is None and params.prompt_relay is None
    if engine != "mlx_video" or pipeline == "dfr":
        return False
    needed: set[NativeVideoFeature] = set()
    if params.audio_conditioning is not None:
        needed.add("audio_conditioning")
    if params.editing is not None:
        needed.add(params.editing.operation)
    if params.prompt_relay is not None:
        needed.add("prompt_relay")
    return needed <= video_features_for_build(build)
