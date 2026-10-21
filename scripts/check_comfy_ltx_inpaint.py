"""Compare inpaint math with hash-bound native source on bounded CPU tensors.

Only the selected immutable reference methods are evaluated. No plugin is
installed or initialized, and this check never loads or renders a model.
"""

from __future__ import annotations

import ast
import hashlib
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol, cast

import torch
import torch.nn.functional as functional
from torch import Tensor

LTX_COMFY_REVISION = "44c8a36b7c31ee00b4271daed11dc2a4aa481140"
_SOURCE_SHA256 = "0d7587fabd899661c5cd7d0bbd010eb6c02af78524c0c8ab073a415d66b49736"
_BLEND_SHA256 = "7fbe5e87c8c0a2c4400fbcc9b25009ca9c9d3ef134bad80874331b7e18eab44c"
_AUDIO_SHA256 = "5737824a18438b6b5782836f1c606e47caa175d9efa93f56c4cbf9f91b5c72e5"


@dataclass(frozen=True)
class _Output:
    values: tuple[object, ...]


def _output(*values: object) -> _Output:
    return _Output(values)


class _GreenReference(Protocol):
    def execute(self, images: Tensor, mask: Tensor) -> _Output: ...


class _DilationReference(Protocol):
    def execute(self, spatial_radius: int, temporal_radius: int, *, mask: Tensor) -> _Output: ...


class _BlendReference(Protocol):
    def __call__(self, image1: Tensor, image2: Tensor, mask: Tensor, *, max_level: int, device: torch.device, output_device: torch.device) -> Tensor: ...


class _BlendMaskReference(Protocol):
    def __call__(self, mask: Tensor, spatial_radius: int) -> Tensor: ...


class _AudioReference(Protocol):
    def __call__(self, latent: Tensor) -> dict[str, Tensor]: ...


def _check_audio_reference(source: Path) -> list[dict[str, object]]:
    from skulk_comfy_ltx.inpaint_audio import inpaint_audio_context

    data = subprocess.run(["git", "-C", str(source), "show", f"{LTX_COMFY_REVISION}:iclora.py"], check=True, capture_output=True).stdout
    if hashlib.sha256(data).hexdigest() != _AUDIO_SHA256:
        raise ValueError("inpainting audio differs from reviewed immutable source")
    selected: list[ast.stmt] = [item for item in ast.parse(data).body if isinstance(item, ast.FunctionDef) and item.name == "_patchify_audio_latent"]
    if len(selected) != 1:
        raise ValueError("inpainting audio reference is incomplete")
    namespace: dict[str, object] = {}
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "<pinned-native-inpainting-audio>", "exec"), namespace)
    native = cast(_AudioReference, namespace["_patchify_audio_latent"])
    cases: list[dict[str, object]] = []
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        for transposed in (False, True):
            samples = torch.arange(8 * 11 * 4, dtype=dtype).reshape(1, 8, 11, 4)
            if transposed:
                samples = samples.transpose(1, 3)
            tokens, frozen = inpaint_audio_context(samples, maximum_bytes=65536, check_cancelled=lambda: None)
            torch.testing.assert_close(tokens, native(samples)["tokens"], rtol=0, atol=0)
            assert frozen.shape == (1, 1, samples.shape[2], 1) and frozen.dtype == torch.float32 and frozen.device.type == "cpu" and not bool(frozen.any())
            cases.append(dict(dtype=str(dtype), transposed=transposed, exact_tokens=True, frozen_all_time_positions=True))
    return cases


def _blend_reference(source: Path) -> tuple[_BlendReference, _BlendMaskReference]:
    from kornia.geometry.transform.pyramid import (
        PyrUp,
        build_laplacian_pyramid,
        build_pyramid,
        find_next_powerof_two,
        is_powerof_two,
    )

    data = subprocess.run(["git", "-C", str(source), "show", f"{LTX_COMFY_REVISION}:pyramid_blending.py"], check=True, capture_output=True).stdout
    if hashlib.sha256(data).hexdigest() != _BLEND_SHA256:
        raise ValueError("inpainting blend differs from reviewed immutable source")
    functions = {"_pad_for_laplacian", "_gaussian_pyramid", "_resize_preserving_aspect_ratio", "_apply_low_res_mask_dilation", "_pyramid_blend_chunk", "_pyramid_blend"}
    constants = {"_CHUNK_SIZE", "_MASK_LOW_RES_LONG_SIDE"}
    selected: list[ast.stmt] = []
    for item in ast.parse(data).body:
        if isinstance(item, ast.FunctionDef) and item.name in functions or isinstance(item, ast.Assign) and any(isinstance(target, ast.Name) and target.id in constants for target in item.targets):
            selected.append(item)
    if len(selected) != len(functions) + len(constants):
        raise ValueError("inpainting blend reference is incomplete")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected], type_ignores=[])
    namespace: dict[str, object] = dict(torch=torch, math=math, F=functional, PyrUp=PyrUp, build_laplacian_pyramid=build_laplacian_pyramid, build_pyramid=build_pyramid, find_next_powerof_two=find_next_powerof_two, is_powerof_two=is_powerof_two)
    # Evaluate the exact pinned numerical functions, without plugin registration
    # or Comfy device initialization. The original chunk size remains unchanged.
    exec(compile(ast.fix_missing_locations(module), "<pinned-native-inpainting-blend>", "exec"), namespace)
    return cast(_BlendReference, namespace["_pyramid_blend"]), cast(_BlendMaskReference, namespace["_apply_low_res_mask_dilation"])


def _check_blend_reference(source: Path) -> list[dict[str, object]]:
    from skulk_comfy_ltx.inpaint_blend import blend_inpaint_images

    native_blend, native_mask = _blend_reference(source)
    generator = torch.Generator(device="cpu").manual_seed(24681358)
    # Nine frames cross the native eight-frame chunk boundary. Rectangular,
    # non-power-of-two geometry also exercises padding and final exact cropping.
    generated, context = torch.rand((9, 65, 97, 3), generator=generator), torch.rand((9, 65, 97, 3), generator=generator)
    masks: dict[str, Tensor] = {"black": torch.zeros(9, 65, 97), "white": torch.ones(9, 65, 97), "soft": torch.rand((9, 65, 97), generator=generator)}
    moving, edge = torch.zeros(9, 65, 97), torch.zeros(9, 65, 97)
    for index in range(9):
        moving[index, 16:32, index * 7:index * 7 + 16] = 1
    edge[:, 0, 0] = 1
    edge[:, -1, -1] = 1
    masks.update(moving=moving, edge=edge)
    device = torch.device("cpu")
    cases: list[dict[str, object]] = []
    for name, mask in masks.items():
        for radius in (5, 6):
            expanded = native_mask(mask.unsqueeze(1), radius)
            expected = native_blend(generated.permute(0, 3, 1, 2), context.permute(0, 3, 1, 2), expanded, max_level=7, device=device, output_device=device).permute(0, 2, 3, 1)
            actual = blend_inpaint_images(generated, context, mask, radius, maximum_bytes=256 * 1024 * 1024, device=device, output_device=device, check_cancelled=lambda: None)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0.000002)
            cases.append(dict(mask=name, radius=radius, frames=9, width=97, height=65, absolute_tolerance=0.000002, max_absolute_error=float((actual - expected).abs().max()), exact=torch.equal(actual, expected)))
    return cases


def _reference(source: Path) -> tuple[_GreenReference, _DilationReference]:
    data = subprocess.run(["git", "-C", str(source), "show", f"{LTX_COMFY_REVISION}:vanish_nodes.py"], check=True, capture_output=True).stdout
    if hashlib.sha256(data).hexdigest() != _SOURCE_SHA256:
        raise ValueError("inpainting reference differs from reviewed immutable source")
    tree = ast.parse(data)
    names = {"LTXVInpaintPreprocess", "LTXVDilateVideoMask"}
    selected: list[ast.stmt] = []
    for item in tree.body:
        if isinstance(item, ast.ClassDef) and item.name in names:
            # Registration has no numerical role; importing the upstream plugin
            # would initialize code outside this reviewed reference boundary.
            item.decorator_list = []
            selected.append(item)
        elif isinstance(item, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "_BG_COLOR_RGB" for target in item.targets):
            selected.append(item)
    if len(selected) != 3:
        raise ValueError("inpainting reference class or trained color is absent")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected], type_ignores=[])
    namespace: dict[str, object] = dict(torch=torch, F=functional, io=SimpleNamespace(ComfyNode=object, NodeOutput=_output))
    exec(compile(ast.fix_missing_locations(module), "<pinned-native-inpainting>", "exec"), namespace)
    return cast(_GreenReference, namespace["LTXVInpaintPreprocess"]), cast(_DilationReference, namespace["LTXVDilateVideoMask"])


def _tensor(output: _Output) -> Tensor:
    value = output.values[0]
    if not isinstance(value, Tensor):
        raise ValueError("native inpainting reference returned a non-tensor")
    return value


def check_inpaint_reference(source: Path) -> dict[str, object]:
    """Compare complete source/guide pixels and masks to the pinned native methods.

    Args:
        source: Git checkout containing the immutable LTXVideo reference commit.

    Returns:
        JSON-safe provenance and exact bounded-tensor case results.

    Raises:
        ValueError: The reference bytes or output contract differ from their pin.
        AssertionError: The wheel differs numerically from the native workflow.
    """
    from skulk_comfy_ltx.inpaint_pixels import (
        dilate_inpaint_mask,
        green_inpaint_context,
    )

    native_green, native_dilation = _reference(source)
    generator = torch.Generator(device="cpu").manual_seed(24681357)
    images = torch.rand((9, 32, 48, 3), generator=generator)
    masks: dict[str, Tensor] = {"black": torch.zeros(1, 32, 48), "white": torch.ones(1, 32, 48), "soft": torch.rand((1, 32, 48), generator=generator), "moving": torch.rand((9, 32, 48), generator=generator)}
    edge = torch.zeros(1, 32, 48)
    edge[0, 0, 0] = 1
    edge[0, -1, -1] = 1
    masks["edge"] = edge
    cases: list[dict[str, object]] = []
    for name, mask in masks.items():
        reference = _tensor(native_green.execute(images.clone(), mask.clone()))
        actual = green_inpaint_context(images, mask, maximum_bytes=16 * 1024 * 1024, output_device=torch.device("cpu"), check_cancelled=lambda: None)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        radii: list[int] = []
        for radius in (0, 1, 15, 30):
            expected = _tensor(native_dilation.execute(radius, 0, mask=mask.clone()))
            dilated = dilate_inpaint_mask(mask, radius, maximum_bytes=16 * 1024 * 1024, check_cancelled=lambda: None)
            torch.testing.assert_close(dilated, expected, rtol=0, atol=0)
            radii.append(radius)
        cases.append(dict(mask=name, source_frames=images.shape[0], mask_frames=mask.shape[0], width=images.shape[2], height=images.shape[1], exact_green_pixels=True, exact_dilation_radii=radii))
    return dict(reference_commit=LTX_COMFY_REVISION, source_sha256=_SOURCE_SHA256, blend_source_sha256=_BLEND_SHA256, audio_source_sha256=_AUDIO_SHA256, cases=cases, blend_cases=_check_blend_reference(source), audio_cases=_check_audio_reference(source), numerical_scope="native SDR green context, spatial dilation, Laplacian blending and encoded source-audio token layout", model_weights_used=False, gpu_render_qualification=False, pass_all=True)
