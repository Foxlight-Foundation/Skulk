"""Request-level engine compatibility, independent of hardware qualification."""

import re
from collections.abc import Set

from skulk.provisioning.comfy_extension import published_extensions
from skulk.provisioning.mlx_video import (
    PUBLISHED_WHEELS,
    NativeVideoFeature,
    video_features_for_build,
)
from skulk.shared.backends import engine_of
from skulk.shared.models.model_cards import ModelCard
from skulk.shared.types.video import VideoExecutionFeature, VideoGenerationTaskParams


def video_feature_build_key(build: str | None) -> str | None:
    """Normalize exact feature code identity, retaining Comfy source and payload.

    Torch remains part of the full placement identity. These source-feature
    selectors omit only its lane-specific version, which the installer probes.
    Unknown or malformed identities never acquire advanced features.
    """
    if build is None:
        return None
    if re.fullmatch(r"mlx_video@sha256:[0-9a-f]{64}", build):
        return build
    match = re.fullmatch(r"comfy@([0-9a-f]{40})/torch@[A-Za-z0-9.+_-]+/extension@sha256:([0-9a-f]{64})", build)
    return f"comfy@{match[1]}/extension@sha256:{match[2]}" if match else None


def video_engine_feature_builds(engines: Set[str | None]) -> dict[str, list[VideoExecutionFeature]]:
    """Describe only issued engine code; card controls and hardware remain separate."""
    builds: dict[str, list[VideoExecutionFeature]] = {}
    if "mlx_video" in engines:
        for wheel in PUBLISHED_WHEELS:
            builds[wheel.build_identity] = ["hq_two_stage", *wheel.features]
    if "comfy" in engines:
        for wheel in published_extensions():
            if "res2s" in wheel.features:
                builds[f"comfy@{wheel.comfy_revision}/extension@sha256:{wheel.payload_sha256}"] = ["hq_two_stage"]
    return builds


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
        from skulk.provisioning.comfy_extension import extension_features_for_build

        baseline = pipeline in ("distilled_two_stage", "dev_two_stage", "dfr")
        hq = pipeline == "hq_two_stage" and "res2s" in extension_features_for_build(build)
        return (baseline or hq) and params.editing is None and params.prompt_relay is None
    # The native wheel's muxer implements H.264 only. Exclude an incompatible
    # request before dispatch instead of letting it fail in the runner planner.
    if engine != "mlx_video" or pipeline == "dfr" or params.codec not in (None, "h264"):
        return False
    if pipeline == "hq_two_stage" and video_feature_build_key(build) not in video_engine_feature_builds({"mlx_video"}):
        return False
    needed: set[NativeVideoFeature] = set()
    if params.audio_conditioning is not None:
        needed.add("audio_conditioning")
    if params.editing is not None:
        needed.add(params.editing.operation)
    if params.prompt_relay is not None:
        needed.add("prompt_relay")
    return needed <= video_features_for_build(build)
