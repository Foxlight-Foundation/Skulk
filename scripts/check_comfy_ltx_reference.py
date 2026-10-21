"""Compare an owned wheel to pinned reference functions on bounded CPU tensors.

This checks numerical contracts without model weights or CPU model rendering.
Only named functions from immutable, hash-checked upstream files are evaluated.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import subprocess
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
from importlib.resources import files
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Protocol, cast, final

import einops
import torch
from torch import Tensor

from scripts.check_comfy_ltx_inpaint import check_inpaint_reference
from scripts.check_comfy_ltx_relay import check_relay_reference

if TYPE_CHECKING:
    from skulk_comfy_ltx.comfy_sampler import Guider, LatentModel, SamplerProgress
    from skulk_comfy_ltx.editing import AudioClock, Modality
    from skulk_comfy_ltx.res2s import Latents

REFERENCE = "e98f6d927d83c92ef2be6aec76c8440b19306e39"
COMFY = "40c4fcdf513a4523e39d54a9d391908af8df8171"
HASHES = {
    "packages/ltx-core/src/ltx_core/components/diffusion_steps.py": "9e0f97c20639bee0c54524ccd2dc2218745767e4c440dc123c7b1d652d2628f2",
    "packages/ltx-pipelines/src/ltx_pipelines/utils/res2s.py": "c4c17d9a3fb9c656977c8610447eddf2a456692c8a1cac345c2547de1e5dd419",
    "packages/ltx-pipelines/src/ltx_pipelines/utils/helpers.py": "f4e9969b8614d66d74e29ea8bfafedb10e0a6cbfb1f8a1ecfdf414a53265aed4",
    "packages/ltx-pipelines/src/ltx_pipelines/utils/samplers.py": "dd2fef08a3fbf278d3dc4dbd261ccd75e257f21a3a025c90aec13eefdf6f3b39",
}


@dataclass
class _Mask:
    tensor: Tensor


@dataclass
class _State:
    latent: Tensor
    denoise_mask: _Mask
    clean_latent: Tensor


@dataclass
class _Pair:
    host_tensor: Tensor
    device_tensor: Tensor


@dataclass
class _AudioVideo:
    video: _State | None
    audio: _State | None


@dataclass
class _Prediction:
    denoised: Tensor


def _predict(values: Latents, sigma: Tensor, *, dtype: torch.dtype = torch.bfloat16) -> Latents:
    return tuple(torch.tanh(value.float() * 0.3 + sigma.float() * 0.02).to(dtype) for value in values)


def _denoise(_transformer: object, video_state: _State | None = None, audio_state: _State | None = None, sigmas: Tensor | None = None, step_index: int = 0, *, prediction_dtype: torch.dtype = torch.bfloat16) -> tuple[_Prediction | None, ...]:
    assert sigmas is not None
    sigma = sigmas[step_index]
    return tuple(_Prediction(_predict((state.latent,), sigma, dtype=prediction_dtype)[0]) if state is not None else None for state in (video_state, audio_state))


def _reference_source(source: Path, revision: str, relative: str) -> bytes:
    return subprocess.run(["git", "-C", str(source), "show", f"{revision}:{relative}"], check=True, capture_output=True).stdout


def _load(namespace: dict[str, object], data: bytes, filename: str, names: tuple[str, ...]) -> None:
    tree = ast.parse(data)
    selected = [item for item in tree.body if isinstance(item, (ast.FunctionDef, ast.ClassDef)) and item.name in names]
    assert len(selected) == len(names)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), filename, "exec"), namespace)


@dataclass(frozen=True)
class _PatchGeometry:
    _patch_size: tuple[int, int, int] = (1, 1, 1)


@dataclass(frozen=True)
class _OutputShape:
    frames: int
    height: int = 1
    width: int = 1
    channels: int = 1
    mel_bins: int = 1


@dataclass(frozen=True)
class _ReferenceLayout:
    video_patchify: Callable[..., Tensor]
    video_unpatchify: Callable[..., Tensor]
    audio_patchify: Callable[..., Tensor]
    audio_unpatchify: Callable[..., Tensor]
    source_sha256: str

    def patch(self, values: Latents) -> Latents:
        return (self.video_patchify(_PatchGeometry(), values[0]), self.audio_patchify(_PatchGeometry(), values[1]))

    def unpatch(self, values: Latents, shapes: Sequence[Sequence[int]]) -> Latents:
        video, audio = shapes
        return (
            self.video_unpatchify(_PatchGeometry(), values[0], _OutputShape(video[2], video[3], video[4])),
            self.audio_unpatchify(_PatchGeometry(), values[1], _OutputShape(audio[2], channels=audio[1], mel_bins=audio[3])),
        )


def _reference_layout(source: Path) -> _ReferenceLayout:
    relative = "packages/ltx-core/src/ltx_core/components/patchifiers.py"
    expected_hash = "9618d6adac114b8b21a91c53a582ae19ab944813b25c88ffc11a1aa7e0239e0a"
    data = _reference_source(source, REFERENCE, relative)
    if hashlib.sha256(data).hexdigest() != expected_hash:
        raise ValueError("patchifier reference differs from its reviewed immutable source")
    namespace: dict[str, object] = {"torch": torch, "einops": einops}
    selected: list[ast.FunctionDef] = []
    tree = ast.parse(data)
    for class_name, prefix in (("VideoLatentPatchifier", "video"), ("AudioPatchifier", "audio")):
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
        for method_name in ("patchify", "unpatchify"):
            method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
            method.name = f"{prefix}_{method_name}"
            selected.append(method)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), relative, "exec"), namespace)
    return _ReferenceLayout(
        cast("Callable[..., Tensor]", namespace["video_patchify"]),
        cast("Callable[..., Tensor]", namespace["video_unpatchify"]),
        cast("Callable[..., Tensor]", namespace["audio_patchify"]),
        cast("Callable[..., Tensor]", namespace["audio_unpatchify"]),
        expected_hash,
    )


def _inputs(modalities: int, terminal: bool, masked: bool, *, dtype: torch.dtype = torch.bfloat16) -> tuple[Latents, Latents, Latents, Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(17)
    shapes = [(1, 12, 4), (1, 5, 16)][:modalities]
    latents = tuple(torch.randn(shape, dtype=dtype, generator=generator) for shape in shapes)
    clean = tuple(torch.randn(value.shape, dtype=value.dtype, generator=generator) for value in latents)
    masks = tuple(torch.ones_like(value, dtype=torch.float32) for value in latents)
    if masked:
        for mask in masks:
            mask[:, 0, :] = 0
            mask[:, 1, :] = 0.5
    sigmas = torch.tensor([0.98, 0.9, 0.72, 0.43, 0.0 if terminal else 0.1], dtype=torch.float32)
    return latents, clean, masks, sigmas


@dataclass
class _Arguments(argparse.Namespace):
    """Typed required qualification inputs, never a model-render request."""

    ltx_source: Path = Path(".")
    comfy_source: Path = Path(".")
    ltx_comfy_source: Path = Path(".")
    wheel: Path = Path(".")
    output: Path = Path(".")


def _highest_precision_float(_device: object) -> torch.dtype:
    return torch.float64


def _progress[T](values: T) -> T:
    return values


@dataclass(frozen=True)
class _GuidanceParameters:
    cfg_scale: float
    stg_scale: float
    modality_scale: float
    rescale_scale: float


@dataclass(frozen=True)
class _GuidanceState:
    params: _GuidanceParameters


def _check_guidance_rescale(source: Path, layout: _ReferenceLayout) -> dict[str, object]:
    from skulk_comfy_ltx.comfy_sampler import pack_latents
    from skulk_comfy_ltx.guidance import rescale_modalities

    relative = "packages/ltx-core/src/ltx_core/components/guiders.py"
    expected_hash = "fca85e9fa3e45bc2755f1d3d66d91bdb8db04bb4c38a17bf1b0cbaa70fe99e6e"
    data = _reference_source(source, REFERENCE, relative)
    if hashlib.sha256(data).hexdigest() != expected_hash:
        raise ValueError("guidance reference differs from its reviewed immutable source")
    tree = ast.parse(data)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MultiModalGuider")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "calculate")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method], type_ignores=[])
    namespace: dict[str, object] = {"torch": torch}
    exec(compile(ast.fix_missing_locations(module), relative, "exec"), namespace)
    calculate = cast("Callable[..., Tensor]", namespace["calculate"])
    generator = torch.Generator(device="cpu").manual_seed(39)
    shapes = [(1, 4, 3, 2, 2), (1, 4, 5, 4)]
    conditioned = layout.patch(tuple(torch.randn(shape, generator=generator) for shape in shapes))
    text = layout.patch(tuple(torch.randn(shape, generator=generator) for shape in shapes))
    perturbed = layout.patch(tuple(torch.randn(shape, generator=generator) for shape in shapes))
    modality = layout.patch(tuple(torch.randn(shape, generator=generator) for shape in shapes))
    results: list[dict[str, object]] = []
    for video_scale, audio_scale in ((0.0, 0.0), (0.45, 1.0), (1.0, 0.45)):
        raw: list[Tensor] = []
        expected: list[Tensor] = []
        for index, (cond, neg, stg, mod, scale) in enumerate(zip(conditioned, text, perturbed, modality, (video_scale, audio_scale), strict=True)):
            cfg = 3.0 if index == 0 else 7.0
            raw.append(calculate(_GuidanceState(_GuidanceParameters(cfg, 0.75, 3.0, 0.0)), cond, neg, stg, mod))
            expected.append(calculate(_GuidanceState(_GuidanceParameters(cfg, 0.75, 3.0, scale)), cond, neg, stg, mod))
        actual = rescale_modalities(pack_latents(layout.unpatch(conditioned, shapes)), pack_latents(layout.unpatch(tuple(raw), shapes)), shapes, video_scale, audio_scale)
        target = pack_latents(layout.unpatch(tuple(expected), shapes))
        torch.testing.assert_close(actual, target, rtol=0, atol=0)
        results.append(dict(video_scale=video_scale, audio_scale=audio_scale, max_absolute_error=float((actual - target).abs().max())))
    return dict(reference_sha256=expected_hash, cases=results, pass_all=True)


@dataclass(frozen=True)
class _VideoGrid(_OutputShape):
    batch: int = 1


@dataclass(frozen=True)
class _AudioGrid(_OutputShape):
    batch: int = 1


@dataclass(frozen=True)
class _Scales:
    time: int = 8
    height: int = 32
    width: int = 32

    @classmethod
    def default(cls) -> _Scales:
        return cls()


class _MaskPatchifier(Protocol):
    def get_patch_grid_bounds(self, output_shape: _VideoGrid | _AudioGrid, device: torch.device | None = None) -> Tensor: ...


@dataclass(frozen=True)
class _RegionTools:
    patchifier: _MaskPatchifier
    target_shape: _VideoGrid | _AudioGrid
    scale_factors: _Scales = _Scales()
    causal_fix: bool = True


@dataclass(frozen=True)
class _RegionState:
    denoise_mask: _Mask

    def clone(self) -> _RegionState:
        return _RegionState(_Mask(self.denoise_mask.tensor.clone()))


class _ReferenceRegion(Protocol):
    def apply_to(self, latent_state: _RegionState, latent_tools: _RegionTools) -> _RegionState: ...


def _check_temporal_masks(source: Path) -> dict[str, object]:
    from skulk_comfy_ltx.editing import temporal_mask

    patch_file = "packages/ltx-core/src/ltx_core/components/patchifiers.py"
    patch_hash = "9618d6adac114b8b21a91c53a582ae19ab944813b25c88ffc11a1aa7e0239e0a"
    region_file = "packages/ltx-core/src/ltx_core/conditioning/types/noise_mask_cond.py"
    region_hash = "4f556155f28032a486672adeafc730cb6a461d9dfce92577b4857e7b7af5c4d2"
    namespace: dict[str, object] = dict(torch=torch, math=math, einops=einops, Patchifier=object, ConditioningItem=object, dataclass=dataclass, AudioLatentShape=_AudioGrid, VideoLatentShape=_VideoGrid, SpatioTemporalScaleFactors=_Scales)
    for relative, digest, names in ((patch_file, patch_hash, ("VideoLatentPatchifier", "AudioPatchifier", "get_pixel_coords")), (region_file, region_hash, ("TemporalRegionMask",))):
        data = _reference_source(source, REFERENCE, relative)
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("temporal mask reference differs from its reviewed immutable source")
        _load(namespace, data, relative, names)
    video_patch = cast("Callable[..., _MaskPatchifier]", namespace["VideoLatentPatchifier"])(patch_size=1)
    audio_patch = cast("Callable[..., _MaskPatchifier]", namespace["AudioPatchifier"])(patch_size=1, sample_rate=16000, hop_length=160, audio_latent_downsample_factor=4)
    region = cast("Callable[..., _ReferenceRegion]", namespace["TemporalRegionMask"])
    cases: list[dict[str, object]] = []
    for batch in (1, 2):
        for fps in (24, 25):
            modalities: tuple[tuple[Modality, _MaskPatchifier, _VideoGrid | _AudioGrid], ...] = (("video", video_patch, _VideoGrid(7, height=2, width=3, batch=batch)), ("audio", audio_patch, _AudioGrid(52, mel_bins=4, batch=batch)))
            for modality, patchifier, grid in modalities:
                for beginning, end in ((0.0, 0.01), (1 / fps, 9 / fps), (0.01, 0.05), (0.5, 1.0), (2.02, 2.04)):
                    shape = (batch, 1, 7, 2, 3) if modality == "video" else (batch, 1, 52, 4)
                    values = torch.zeros(shape)
                    original_mask = torch.full((batch, 42 if modality == "video" else 52, 1), -1.0)
                    expected = region(beginning, end, fps).apply_to(_RegionState(_Mask(original_mask)), _RegionTools(patchifier, grid)).denoise_mask.tensor
                    observed = temporal_mask(values, modality=modality, start_seconds=beginning, end_seconds=end, fps=fps)
                    if modality == "video":
                        observed = einops.rearrange(observed.expand(batch, 1, 7, 2, 3), "b c t h w -> b (t h w) c")
                    else:
                        observed = observed[:, :, :, 0].expand(batch, 1, 52).movedim(1, 2)
                    torch.testing.assert_close(observed, expected, rtol=0, atol=0)
                    assert bool((original_mask == -1).all())
                    cases.append(dict(batch=batch, fps=fps, modality=modality, start_seconds=beginning, end_seconds=end, exact=True))
    return dict(patchifier_sha256=patch_hash, region_sha256=region_hash, cases=cases, pass_all=True)


@dataclass
class _SourceAudioClock:
    sample_rate: int = 16000
    mel_hop_length: int = 160
    latents_per_second: float = 25.0


class _SourceAudioVae:
    first_stage_model: AudioClock
    crop_input: bool
    output_channels: int = 2
    pad_channel_value: None = None

    def __init__(self) -> None:
        self.first_stage_model = _SourceAudioClock()
        self.crop_input = True

    def spacial_compression_encode(self) -> int:
        # Keep the pinned upstream spelling: the real crop calls this method.
        return 4096


class _SourceAudioCrop(Protocol):
    first_stage_model: AudioClock
    crop_input: bool
    output_channels: int
    pad_channel_value: None

    def spacial_compression_encode(self) -> int: ...


def _check_source_audio_clock(source: Path) -> dict[str, object]:
    from skulk_comfy_ltx.editing import uncropped_audio_vae

    relative = "comfy/sd.py"
    expected_hash = "ef373b88ced93955a7147dcd3323ecac9cd5316f0ae4623adfb9b6b128c37ba7"
    data = _reference_source(source, COMFY, relative)
    if hashlib.sha256(data).hexdigest() != expected_hash:
        raise ValueError("Comfy VAE source differs from the reviewed immutable crop boundary")
    tree = ast.parse(data)
    vae = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "VAE")
    crop = next(node for node in vae.body if isinstance(node, ast.FunctionDef) and node.name == "vae_encode_crop_pixels")
    namespace: dict[str, object] = {"torch": torch}
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), crop], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), relative, "exec"), namespace)
    crop_pixels = cast("Callable[[_SourceAudioCrop, Tensor], Tensor]", namespace["vae_encode_crop_pixels"])
    cases: list[dict[str, object]] = []
    for frames in (25, 49, 73, 97, 121, 241, 481):
        # The pinned public audio node resamples to VAE.audio_sample_rate,
        # which remains 44100 for LTX before its internal model resampling.
        length = math.ceil(frames * 44100 / 24)
        waveform = torch.arange(length, dtype=torch.float32).reshape(1, length, 1).expand(1, length, 2)
        original = _SourceAudioVae()
        legacy = crop_pixels(original, waveform)
        encoder = uncropped_audio_vae(original)
        observed = crop_pixels(encoder, waveform)
        assert original.crop_input and encoder.first_stage_model is original.first_stage_model
        torch.testing.assert_close(observed, waveform, rtol=0, atol=0)
        assert legacy.shape[1] < length
        cases.append(dict(frames=frames, fps=24, samples=length, legacy_samples=legacy.shape[1], legacy_start_sample=float(legacy[0, 0, 0]), full_waveform_exact=True))
    return dict(comfy_vae_sha256=expected_hash, cases=cases, pass_all=True)


def _check_dfr_masks(source: Path) -> dict[str, object]:
    from skulk_comfy_ltx.dfr import detailing_frame_override, prepare_dfr_latent

    hashes = {
        "comfy_extras/nodes_lt.py": "01575b886e9abacd08c7ce39e7bde9ef7c7cc8f17063d726e2bf4a320a512d95",
        "comfy_extras/nodes_lt_keyframes.py": "803ff1004698b6090d10026aa8e784151af4dee51c032f63255c5dae47229417",
    }
    namespace: dict[str, object] = {"torch": torch, "_OCCUPIED_MASK_MAX": 1.0 - 1e-4}
    for relative, expected in hashes.items():
        data = _reference_source(source, COMFY, relative)
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("Comfy DFR source differs from the reviewed immutable mask boundary")
        names = ("get_noise_mask",) if relative.endswith("nodes_lt.py") else ("_as_int_set", "occupied_pixel_frames", "_spaced_positions_keep_last", "detailing_positions", "free_detailing_slots")
        _load(namespace, data, relative, names)
    native_mask = cast("Callable[[Mapping[str, object]], Tensor]", namespace["get_noise_mask"])
    occupied_frames = cast("Callable[[Mapping[str, object], int, int, int], set[int]]", namespace["occupied_pixel_frames"])
    positions = cast("Callable[[int, float], list[int]]", namespace["detailing_positions"])
    free_slots = cast("Callable[[int, float, set[int]], list[int]]", namespace["free_detailing_slots"])
    cases: list[dict[str, object]] = []
    for frames in (25, 49, 97):
        latent_frames = (frames - 1) // 8 + 1
        for dtype in (torch.bfloat16, torch.float32):
            dense: dict[str, object] = {"samples": torch.full((1, 4, latent_frames, 2, 3), 0.75, dtype=dtype)}
            legacy_occupied = occupied_frames(dense, 8, frames, latent_frames)
            try:
                free_slots(frames, 24.0, legacy_occupied)
            except ValueError:
                pass
            else:
                raise AssertionError("the retained dense-video DFR failure was not reproduced")
            prepared = prepare_dfr_latent(dense)
            observed = prepared["noise_mask"]
            assert isinstance(observed, Tensor) and prepared["samples"] is dense["samples"] and "noise_mask" not in dense
            torch.testing.assert_close(observed, native_mask(dense), rtol=0, atol=0)
            occupied = occupied_frames(prepared, 8, frames, latent_frames)
            assert occupied == set() and free_slots(frames, 24.0, occupied)
            assert detailing_frame_override(frames, occupied, positions(frames, 24.0)) == ""
            # The real occupancy helper must retain first/last anchors; when
            # both consume the short stock grid, choose a free interior pixel.
            anchored_mask = observed.clone()
            anchored_mask[:, :, 0] = 0
            anchored_mask[:, :, -1] = 0.05
            anchored = prepare_dfr_latent({**dense, "noise_mask": anchored_mask})
            occupied = occupied_frames(anchored, 8, frames, latent_frames)
            assert occupied == {0, frames - 1}
            override = detailing_frame_override(frames, occupied, positions(frames, 24.0))
            assert override == ("12" if frames == 25 else "")
            torch.testing.assert_close(anchored_mask, anchored["noise_mask"], rtol=0, atol=0)
            cases.append(dict(frames=frames, dtype=str(dtype), legacy_dense_failure_reproduced=True, sampling_mask_matches_pinned_helper=True, anchors_preserved=True, short_frame_override=override))
    return dict(source_sha256=hashes, cases=cases, pass_all=True)


class _ReferenceNodeOutput:
    """Capture native node results without importing GPU process startup."""

    def __init__(self, *values: object) -> None:
        self.result = values


class _ReferenceNode(Protocol):
    """Explicit call boundary for dynamically evaluated upstream node classes."""

    def execute(self, *arguments: object, **keywords: object) -> _ReferenceNodeOutput:
        """Execute the selected native or packed bridge method."""
        ...


class _UnionReferenceVae:
    """Bounded encoder stand-in; model weights are tested separately on GPU."""

    downscale_index_formula: tuple[int, int, int] = (8, 32, 32)

    def encode(self, pixels: Tensor) -> Tensor:
        """Check the native half-grid resize and return deterministic latents."""
        assert pixels.shape[1:3] == (32, 64)
        return torch.full((1, 128, (pixels.shape[0] - 1) // 8 + 1, 1, 2), 3.0, dtype=pixels.dtype)


class _UnionReferenceAdapter:
    """Represent the native loader's metadata attachment protocol."""

    def get_attachment(self, name: str) -> object:
        """Return the actual reference scale from the selected adapter."""
        assert name == "lora_metadata"
        return {"reference_downscale_factor": "2"}


class _InpaintReferenceVae:
    """Bounded full-grid encoder stand-in for the actual native guide contract."""

    downscale_index_formula: tuple[int, int, int] = (8, 32, 32)

    def encode(self, pixels: Tensor) -> Tensor:
        """Require the complete current-stage canvas, with no half-grid resize."""
        assert pixels.shape[1:3] == (64, 128)
        return torch.full((1, 128, (pixels.shape[0] - 1) // 8 + 1, 2, 4), 3.0, dtype=pixels.dtype)


class _InpaintReferenceAdapter:
    """Expose the actual factor-one metadata attachment of the inpainting recipe."""

    def get_attachment(self, name: str) -> object:
        """Return the loaded adapter's full-grid scale, rather than a fallback."""
        assert name == "lora_metadata"
        return {"reference_downscale_factor": "1"}


def _check_union_reference(source: Path) -> dict[str, object]:
    """Run pinned guide, dilation and crop methods on bounded reference tensors."""
    from skulk_comfy_ltx.inpaint_control import validate_inpaint_guide
    from skulk_comfy_ltx.union_control import validate_union_inputs

    hashes = {
        "comfy_extras/nodes_lt.py": "01575b886e9abacd08c7ce39e7bde9ef7c7cc8f17063d726e2bf4a320a512d95",
        "node_helpers.py": "22317a033cce556331bfedc5f4ab6c2833184fd2bbdf9496c9182a4973a7098a",
        "comfy/utils.py": "f3ee59baf72f6910b7662224854e372c713ba3ad01c89739ba092e5ee22460e6",
        "comfy/ldm/lightricks/symmetric_patchifier.py": "e2d9b8de7a5cc003e74eaba6d0b8303b596df80e59a431eda2d3593ec06c2633",
    }
    sources = {name: _reference_source(source, COMFY, name) for name in hashes}
    if any(hashlib.sha256(data).hexdigest() != hashes[name] for name, data in sources.items()):
        raise ValueError("Union source differs from its reviewed immutable guide contract")
    namespace: dict[str, object] = {"torch": torch, "ABC": ABC, "abstractmethod": abstractmethod, "rearrange": einops.rearrange, "io": SimpleNamespace(ComfyNode=object, NodeOutput=_ReferenceNodeOutput)}
    _load(namespace, sources["node_helpers.py"], "node_helpers.py", ("conditioning_set_values",))
    namespace["node_helpers"] = SimpleNamespace(conditioning_set_values=namespace["conditioning_set_values"])
    _load(namespace, sources["comfy/utils.py"], "comfy/utils.py", ("common_upscale",))
    namespace["comfy"] = SimpleNamespace(utils=SimpleNamespace(common_upscale=namespace["common_upscale"]))
    _load(namespace, sources["comfy/ldm/lightricks/symmetric_patchifier.py"], "symmetric_patchifier.py", ("latent_to_pixel_coords", "Patchifier", "SymmetricPatchifier"))
    _load(namespace, sources["comfy_extras/nodes_lt.py"], "nodes_lt.py", ("conditioning_get_any_value", "get_noise_mask", "get_keyframe_idxs", "_append_guide_attention_entry", "LTXVAddGuide", "LTXVCropGuides", "GetICLoRAParameters"))
    add = cast(_ReferenceNode, namespace["LTXVAddGuide"]).execute
    crop = cast(_ReferenceNode, namespace["LTXVCropGuides"]).execute
    parameters = cast(_ReferenceNode, namespace["GetICLoRAParameters"]).execute
    # The selected wheel's own bridge delegates to the actual pinned native
    # class here; no import initializes a GPU or loads model weights.
    namespace.update(final=final, validate_union_inputs=validate_union_inputs)
    _load(namespace, files("skulk_comfy_ltx").joinpath("union_nodes.py").read_bytes(), "candidate_union_nodes.py", ("SkulkLTXUnionGuide",))
    bridge = cast(_ReferenceNode, namespace["SkulkLTXUnionGuide"]).execute
    adapter = _UnionReferenceAdapter()
    actual_parameters = parameters(adapter).result[0]
    assert actual_parameters == {"reference_downscale_factor": 2}
    cases: list[dict[str, object]] = []
    for frames in (9, 25):
        for dtype in (torch.float32, torch.bfloat16):
            for strength in (0.0, 0.7, 1.0):
                latent: dict[str, object] = {"samples": torch.full((1, 128, (frames - 1) // 8 + 1, 2, 4), 7.0, dtype=dtype)}
                images = torch.zeros(frames, 64, 128, 3, dtype=dtype)
                validate_union_inputs(adapter.get_attachment("lora_metadata"), latent, images, frames)
                positive: list[list[object]] = [[torch.zeros(1, 2, 8), {}]]
                negative: list[list[object]] = [[torch.ones(1, 2, 8), {}]]
                result = add(positive, negative, _UnionReferenceVae(), latent, images, frame_idx=0, strength=strength, iclora_parameters=actual_parameters)
                bridged = bridge(positive, negative, _UnionReferenceVae(), latent, images, adapter, strength, frames)
                appended = cast("Mapping[str, Tensor]", result.result[2])
                bridged_latent = cast("Mapping[str, Tensor]", bridged.result[2])
                for key in ("samples", "noise_mask"):
                    torch.testing.assert_close(appended[key], bridged_latent[key], rtol=0, atol=0)
                samples = appended["samples"]
                assert samples.shape[2] == 2 * ((frames - 1) // 8 + 1)
                guide = samples[:, :, samples.shape[2] // 2:]
                assert bool((guide[..., ::2, ::2] == 3).all())
                assert bool((guide[..., 1::2, :] == 0).all())
                restored = crop(result.result[0], result.result[1], appended)
                restored_latent = cast("Mapping[str, Tensor]", restored.result[2])
                torch.testing.assert_close(restored_latent["samples"], cast(Tensor, latent["samples"]), rtol=0, atol=0)
                assert bool((restored_latent["noise_mask"] == 1).all())
                assert "noise_mask" not in latent and positive[0][1] == {} and negative[0][1] == {}
                cases.append(dict(frames=frames, dtype=str(dtype), guide_strength=strength, exact_crop=True, original_inputs_preserved=True))
    namespace.update(validate_inpaint_guide=validate_inpaint_guide, comfy=SimpleNamespace(utils=SimpleNamespace(common_upscale=namespace["common_upscale"]), model_management=SimpleNamespace(throw_exception_if_processing_interrupted=lambda: None)))
    _load(namespace, files("skulk_comfy_ltx").joinpath("inpaint_nodes.py").read_bytes(), "candidate_inpaint_nodes.py", ("SkulkLTXInpaintGuide",))
    inpaint_bridge = cast(_ReferenceNode, namespace["SkulkLTXInpaintGuide"]).execute
    inpaint_cases: list[dict[str, object]] = []
    for frames in (9, 25):
        for dtype in (torch.float32, torch.bfloat16):
            latent = {"samples": torch.full((1, 128, (frames - 1) // 8 + 1, 2, 4), 7.0, dtype=dtype)}
            images = torch.zeros(frames, 64, 128, 3)
            positive = [[torch.zeros(1, 2, 8), {}]]
            negative = [[torch.ones(1, 2, 8), {}]]
            reference = add(positive, negative, _InpaintReferenceVae(), latent, images, frame_idx=0, strength=1.0, iclora_parameters={"reference_downscale_factor": 1})
            actual = inpaint_bridge(positive, negative, _InpaintReferenceVae(), latent, images, _InpaintReferenceAdapter(), frames)
            expected_latent = cast("Mapping[str, Tensor]", reference.result[2])
            actual_latent = cast("Mapping[str, Tensor]", actual.result[2])
            for key in ("samples", "noise_mask"):
                torch.testing.assert_close(actual_latent[key], expected_latent[key], rtol=0, atol=0)
            samples = actual_latent["samples"]
            assert bool((samples[:, :, samples.shape[2] // 2:] == 3).all())
            restored = crop(actual.result[0], actual.result[1], actual_latent)
            expected_crop = crop(reference.result[0], reference.result[1], expected_latent)
            restored_samples = cast("Mapping[str, Tensor]", restored.result[2])["samples"]
            torch.testing.assert_close(restored_samples, cast("Mapping[str, Tensor]", expected_crop.result[2])["samples"], rtol=0, atol=0)
            # Native concatenation promotes BF16 targets when the VAE guide is
            # FP32. Preserve that reference behavior instead of forcing a dtype.
            torch.testing.assert_close(restored_samples, cast(Tensor, latent["samples"]).to(restored_samples.dtype), rtol=0, atol=0)
            inpaint_cases.append(dict(frames=frames, input_latent_dtype=str(dtype), native_guided_dtype=str(samples.dtype), reference_factor=1, native_guide_and_crop_exact=True))
    return dict(source_sha256=hashes, cases=cases, inpaint_cases=inpaint_cases, actual_packed_bridge=True, encoder_standin_only=True, model_weights_used=False, native_reference_factor=2, pass_all=True)


def main() -> None:
    """Verify the wheel math, actual packed boundary and cancellation; retain a receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ltx-source", type=Path, required=True)
    parser.add_argument("--comfy-source", type=Path, required=True)
    parser.add_argument("--ltx-comfy-source", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(namespace=_Arguments())
    sys.path.insert(0, str(args.wheel.resolve(strict=True)))
    from skulk_comfy_ltx import res2s as candidate
    from skulk_comfy_ltx.comfy_sampler import sample_ltx_res2s

    namespace: dict[str, object] = dict(torch=torch, math=math, replace=replace, partial=partial, DiffusionStepProtocol=object, highest_precision_float=_highest_precision_float, HostDeviceTensorPair=_Pair, VideoAudio=_AudioVideo, tqdm=_progress)
    names = (
        ("Res2sDiffusionStep",),
        ("phi", "get_res2s_coefficients"),
        ("post_process_latent",),
        ("_channelwise_normalize", "_get_new_noise", "_inject_sde_noise", "res2s_audio_video_denoising_loop"),
    )
    for (relative, expected_hash), selected in zip(HASHES.items(), names, strict=True):
        data = _reference_source(args.ltx_source, REFERENCE, relative)
        if hashlib.sha256(data).hexdigest() != expected_hash:
            raise ValueError("reference source differs from the reviewed immutable function set")
        _load(namespace, data, relative, selected)
    # The evaluated function is an opaque upstream boundary. Its input and
    # output state contracts are represented above, separately from the wheel.
    reference_loop = cast("Callable[..., _AudioVideo]", namespace["res2s_audio_video_denoising_loop"])
    results: list[dict[str, object]] = []
    for initial_dtype in (torch.bfloat16, torch.float32):
        for prediction_dtype in (torch.bfloat16, torch.float32):
            for modalities in (1, 2):
                for terminal in (False, True):
                    for masked in (False, True):
                        latents, clean, masks, sigmas = _inputs(modalities, terminal, masked, dtype=initial_dtype)
                        states = tuple(_State(value, _Mask(mask), source) for value, mask, source in zip(latents, masks, clean, strict=True))
                        reference = reference_loop(sigmas, states[0], states[1] if modalities == 2 else None, None, partial(_denoise, prediction_dtype=prediction_dtype), noise_seed=7)
                        observed = candidate.sample_res2s(latents, sigmas, partial(_predict, dtype=prediction_dtype), seed=7, masks=masks, clean=clean)
                        expected = tuple(state.latent for state in (reference.video, reference.audio) if state is not None)
                        diffs = [float((left.float() - right.float()).abs().max()) for left, right in zip(observed, expected, strict=True)]
                        for left, right in zip(observed, expected, strict=True):
                            torch.testing.assert_close(left, right, rtol=0, atol=0)
                        if masked:
                            for value, source, mask in zip(observed, clean, masks, strict=True):
                                torch.testing.assert_close(value[mask == 0], source.to(value.dtype)[mask == 0], rtol=0, atol=0)
                        results.append(dict(initial_dtype=str(initial_dtype), prediction_dtype=str(prediction_dtype), modalities=modalities, terminal=terminal, masked=masked, max_absolute_errors=diffs))
    utils: dict[str, object] = {"torch": torch, "math": math}
    _load(utils, _reference_source(args.comfy_source, COMFY, "comfy/utils.py"), "comfy/utils.py", ("pack_latents", "unpack_latents"))
    pack = cast("Callable[[Sequence[Tensor]], tuple[Tensor, Sequence[Sequence[int]]]]", utils["pack_latents"])
    unpack = cast("Callable[[Tensor, Sequence[Sequence[int]]], list[Tensor]]", utils["unpack_latents"])
    layout = _reference_layout(args.ltx_source)
    from skulk_comfy_ltx.layout import patch_modalities, unpatch_modalities

    layout_cases: list[dict[str, int]] = []
    for batch, video_frames, audio_frames in ((1, 3, 5), (2, 4, 7)):
        raw = (torch.arange(batch * 4 * video_frames * 2 * 2).reshape(batch, 4, video_frames, 2, 2), torch.arange(batch * 4 * audio_frames * 4).reshape(batch, 4, audio_frames, 4))
        geometry = tuple(tuple(value.shape) for value in raw)
        for observed, expected in zip(patch_modalities(raw), layout.patch(raw), strict=True):
            torch.testing.assert_close(observed, expected, rtol=0, atol=0)
        for observed, expected in zip(unpatch_modalities(layout.patch(raw), geometry), raw, strict=True):
            torch.testing.assert_close(observed, expected, rtol=0, atol=0)
        layout_cases.append(dict(batch=batch, video_frames=video_frames, audio_frames=audio_frames))
    packed_cases: list[dict[str, object]] = []
    for initial_dtype in (torch.bfloat16, torch.float32):
        token_latents, token_clean, token_masks, sigmas = _inputs(2, True, True, dtype=initial_dtype)
        raw_shapes = [(1, 4, 3, 2, 2), (1, 4, 5, 4)]
        latents = layout.unpatch(token_latents, raw_shapes)
        clean = layout.unpatch(token_clean, raw_shapes)
        masks = layout.unpatch(token_masks, raw_shapes)
        packed, shapes = pack(latents)
        packed_clean, _ = pack(clean)
        packed_masks, _ = pack(masks)

        @dataclass
        class Geometry:
            """The pinned Comfy model's packed modality dimensions."""

            latent_shapes: Sequence[Sequence[int]]

        class ControlledGuider:
            """Use a fixed denoiser at the real Comfy sampler boundary."""

            inner_model: LatentModel = Geometry(shapes)

            def __call__(self, values: Tensor, sigma: Tensor, *, model_options: dict[str, object], seed: int) -> Tensor:
                del model_options, seed
                return pack(_predict(tuple(unpack(values, self.inner_model.latent_shapes)), sigma[0], dtype=torch.float32))[0]

        @dataclass
        class PackedModel:
            """Represent the exact x0 inpaint wrapper contract without its extra blend."""

            inner_model: Guider
            latent_image: Tensor

        callbacks: list[SamplerProgress] = []
        actual = sample_ltx_res2s(PackedModel(ControlledGuider(), packed_clean), packed, sigmas, extra_args={"seed": 7, "denoise_mask": packed_masks}, callback=callbacks.append)
        initial = tuple((value * mask + source * (1 - mask)).to(value.dtype) for value, mask, source in zip(latents, masks, clean, strict=True))
        states = tuple(_State(value, _Mask(mask), source) for value, mask, source in zip(layout.patch(initial), layout.patch(masks), layout.patch(clean), strict=True))
        reference = reference_loop(sigmas, states[0], states[1], None, partial(_denoise, prediction_dtype=torch.float32))
        expected = tuple(state.latent for state in (reference.video, reference.audio) if state is not None)
        torch.testing.assert_close(actual, pack(layout.unpatch(expected, shapes))[0], rtol=0, atol=0)
        assert len(callbacks) == len(sigmas) - 1
        packed_cases.append(dict(initial_dtype=str(initial_dtype), prediction_dtype=str(torch.float32), callback_intervals=len(callbacks), exact=True))

    latents, clean, masks, sigmas = _inputs(2, True, True, dtype=torch.float32)

    class CancelledError(Exception):
        """A controlled owner cancellation during sampler progress."""

    predict_calls: list[int] = []

    def cancelled_progress(_index: int, _denoised: Latents, _values: Latents) -> None:
        raise CancelledError()

    def counted_predict(values: Latents, sigma: Tensor) -> Latents:
        predict_calls.append(1)
        return _predict(values, sigma)

    try:
        candidate.sample_res2s(latents, sigmas, counted_predict, seed=7, masks=masks, clean=clean, callback=cancelled_progress)
    except CancelledError:
        assert len(predict_calls) == 2
    else:
        raise AssertionError("cancellation did not escape the sampler")
    report = dict(inpaint_pixels=check_inpaint_reference(args.ltx_comfy_source), union_control=_check_union_reference(args.comfy_source), prompt_relay=check_relay_reference(args.comfy_source), dfr_masks=_check_dfr_masks(args.comfy_source), source_audio_clock=_check_source_audio_clock(args.comfy_source), temporal_masks=_check_temporal_masks(args.ltx_source), guidance_rescale=_check_guidance_rescale(args.ltx_source, layout), patchifier_sha256=layout.source_sha256, layout_cases=layout_cases, reference_state_layout="batch_tokens_channels", reference_step_noise_seed=-1, reference_commit=REFERENCE, reference_sha256=HASHES, results=results, packed_adapter_exact=True, packed_adapter_cases=packed_cases, comfy_commit=COMFY, callback_intervals=len(sigmas) - 1, cancellation_stops_after_first_interval=True, pass_all=True)
    _ = args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
