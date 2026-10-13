"""Map immutable image contracts to native arguments and bounded requests."""

from __future__ import annotations

import base64
import hashlib
import io
from pathlib import Path

from PIL import Image

from skulk.api.types.api import ImageEditsTaskParams, ImageGenerationTaskParams
from skulk.download.download_utils import (
    build_sidecar_path,
    companion_artifact_location,
)
from skulk.shared.models.image import ImageCardConfig, validate_reference_images
from skulk.shared.models.model_cards import ModelCard

_ROLE_FLAGS = {
    "model": "--model",
    "diffusion": "--diffusion-model",
    "vae": "--vae",
    "text_encoder": "--llm",
    "text_encoder_vision": "--llm_vision",
    "clip_l": "--clip_l",
    "clip_g": "--clip_g",
    "t5xxl": "--t5xxl",
}


def native_component_arguments(card: ModelCard, *, backend: str) -> list[str]:
    """Verify component identities and select loader flags for the placed backend.

    Qwen Image 2.1 CUDA/Vulkan instances enable Flash Attention to reduce attention
    memory at the default 2048-square canvas. ROCm uses the same path to avoid
    observed native kernel-launch failures at that canvas.
    SDXL Metal loads disable batched guidance to avoid blank outputs and enable
    Flash Attention to bound workspace at the default 1024-square canvas.
    Qwen Vulkan VAE decoding uses bounded tiles to avoid the device's individual
    buffer limit and the resulting expensive CPU retry at the default canvas.
    """
    profile = card.image
    if profile is None:
        raise ValueError("native image placement has no image contract")
    arguments: list[str] = []
    verified: set[tuple[str, str, str]] = set()
    for component in profile.components:
        identifier, revision = companion_artifact_location(
            card, component.repository, component.revision
        )
        entry = build_sidecar_path(identifier, component.entry_file, revision)
        if entry is None:
            raise ValueError(
                "image component entry is not staged at its pinned revision"
            )
        root = entry.parents[len(Path(component.entry_file).parts) - 1]
        for item in component.files:
            identity = (component.repository, component.revision, item.path)
            if identity in verified:
                continue
            path = root / item.path
            if not path.is_file() or path.stat().st_size != item.size_bytes:
                raise ValueError(
                    "image component is absent or has the wrong immutable size"
                )
            algorithm, expected = item.object_id.split(":", 1)
            digest = hashlib.sha256() if algorithm == "sha256" else hashlib.sha1()
            if algorithm == "git-sha1":
                digest.update(f"blob {item.size_bytes}\0".encode())
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise ValueError(
                    "image component differs from its immutable file identity"
                )
            verified.add(identity)
        arguments.extend(
            (_ROLE_FLAGS[component.role], str(root / component.entry_file))
        )
    # Full attention exceeded tested Vulkan/CUDA memory capacity and failed
    # native ROCm kernel launches at the default canvas. The same pinned builds
    # completed with Flash Attention; the ROCm launch failure is not an OOM claim.
    if backend in (
        "stable_diffusion_cpp-vulkan",
        "stable_diffusion_cpp-cuda",
        "stable_diffusion_cpp-rocm",
    ) and profile.family in (
        "qwen_image_21",
        "qwen_image_21_turbo",
    ):
        arguments.append("--fa")
    if backend == "stable_diffusion_cpp-vulkan" and profile.family in (
        "qwen_image_21", "qwen_image_21_turbo",
    ):
        # Untiled 2K decoding exceeded the tested device's single-buffer limit;
        # 512-pixel tiles completed on GPU without the unsafe host-memory retry.
        arguments.extend(("--vae-tiling", "--vae-tile-size", "512", "--vae-tile-overlap", "0.25"))
    if backend == "stable_diffusion_cpp-metal" and profile.family == "sdxl":
        # The pinned Metal build produced white output with batched guidance;
        # separate conditional/unconditional forwards restored the reference.
        arguments.extend(("--batched-cfg", "off", "--fa"))
    return arguments


def native_image_request(
    card: ModelCard,
    params: ImageGenerationTaskParams | ImageEditsTaskParams,
    seed: int,
    references: tuple[bytes, ...] = (),
    *,
    preview: bool = False,
) -> dict[str, object]:
    """Build a literal-prompt native request with enforced geometry and checkpoint sampling."""
    profile = card.image
    if profile is None:
        raise ValueError("native image request has no image contract")
    if params.size == "auto":
        width, height = profile.default_width, profile.default_height
    else:
        width_text, height_text = str(params.size).split("x", 1)
        width, height = int(width_text), int(height_text)
    profile.validate_canvas(width, height)
    if not 0 <= seed < 2**63 or len(params.prompt.encode("utf-8")) > 32_768:
        raise ValueError("native image prompt or seed exceeds its bound")
    if (
        len(references) > profile.max_reference_images
        or sum(map(len, references)) > 24 * 1024 * 1024
    ):
        raise ValueError("image reference input exceeds the model or media limit")
    validate_reference_images(references)
    advanced = params.advanced_params
    steps = (
        advanced.num_inference_steps
        if advanced and advanced.num_inference_steps is not None
        else profile.default_steps
    )
    guidance = (
        advanced.guidance
        if advanced and advanced.guidance is not None
        else profile.default_guidance
    )
    negative = (
        advanced.negative_prompt
        if advanced and advanced.negative_prompt is not None
        else ""
    )
    if len(negative.encode("utf-8")) > 32_768:
        raise ValueError("native negative prompt exceeds its bound")
    if profile.family == "qwen_image_21_turbo" and (
        steps != 8 or guidance != 1.0 or negative
    ):
        raise ValueError("Turbo requires eight steps, CFG=1, and no negative prompt")
    sampling: dict[str, object] = {
        "sample_method": "euler",
        "sample_steps": steps,
        "guidance": {"txt_cfg": guidance},
    }
    if profile.fixed_sigmas:
        sampling["custom_sigmas"] = list(profile.fixed_sigmas)
    body: dict[str, object] = {
        "prompt": params.prompt,
        "negative_prompt": negative,
        "width": width,
        "height": height,
        "seed": seed,
        "batch_count": 1,
        "sample_params": sampling,
        "output_format": params.output_format,
        "preview": "vae" if preview else "none",
        "preview_interval": 1,
    }
    encoded = [base64.b64encode(data).decode() for data in references]
    if profile.family == "sdxl" and encoded:
        body["init_image"] = encoded[0]
        if isinstance(params, ImageEditsTaskParams):
            body["strength"] = (
                params.image_strength if params.image_strength is not None else 0.7
            )
    elif encoded:
        body["ref_images"] = encoded
    return body


def validate_native_output(
    encoded: str,
    image_format: str,
    profile: ImageCardConfig,
    expected_canvas: tuple[int, int],
    *,
    partial: bool,
) -> None:
    """Verify encoded output geometry and reject blank terminal render failures.

    Exact-flat and fully transparent final renders are failed output under the
    native image contract. Sampling previews may legitimately still be blank.
    """
    data = base64.b64decode(encoded, validate=True)
    with Image.open(io.BytesIO(data)) as image:
        profile.validate_canvas(*image.size)
        if image.format != {"png": "PNG", "jpeg": "JPEG", "webp": "WEBP"}[image_format]:
            raise ValueError("native output format differs from its declaration")
        if not partial:
            if image.size != expected_canvas:
                raise ValueError(
                    "native output canvas differs from the admitted request"
                )
            if image.getcolors(maxcolors=1) is not None or (
                image.mode == "RGBA" and image.getchannel("A").getextrema() == (0, 0)
            ):
                raise ValueError(
                    "native image generation returned a blank terminal render"
                )
        else:
            image.verify()
