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
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, cast

import einops
import torch
from torch import Tensor

if TYPE_CHECKING:
    from skulk_comfy_ltx.comfy_sampler import Guider, LatentModel, SamplerProgress
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


def main() -> None:
    """Verify the wheel math, actual packed boundary and cancellation; retain a receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ltx-source", type=Path, required=True)
    parser.add_argument("--comfy-source", type=Path, required=True)
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
    report = dict(guidance_rescale=_check_guidance_rescale(args.ltx_source, layout), patchifier_sha256=layout.source_sha256, layout_cases=layout_cases, reference_state_layout="batch_tokens_channels", reference_step_noise_seed=-1, reference_commit=REFERENCE, reference_sha256=HASHES, results=results, packed_adapter_exact=True, packed_adapter_cases=packed_cases, comfy_commit=COMFY, callback_intervals=len(sigmas) - 1, cancellation_stops_after_first_interval=True, pass_all=True)
    _ = args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
