"""Clone-scoped relay patches using the pinned Comfy attention override hook."""

from __future__ import annotations

from collections.abc import Callable, Sized
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Protocol, cast

from torch import Tensor

from skulk_comfy_ltx.relay import relay_attention
from skulk_comfy_ltx.relay_encoding import RELAY_CONDITION_KEY, RelayLayout
from skulk_comfy_ltx.relay_geometry import RelayGeometry, prepare_relay_geometry


class RelayModelPatcher(Protocol):
    """The pinned clone and object-patch lifecycle; weights remain shared."""

    def clone(self) -> RelayModelPatcher:
        """Create independent graph patches without copying model weights."""
        ...

    def get_model_object(self, name: str) -> object:
        """Read the current patch or original object by its dotted path."""
        ...

    def add_object_patch(self, name: str, value: object) -> None:
        """Let Comfy install and restore this clone's method through its lifecycle."""
        ...


class AttentionKernel(Protocol):
    """The selected optimized kernel behind Comfy's attention wrapper."""

    def __call__(self, query: Tensor, key: Tensor, value: Tensor, heads: int, *, mask: Tensor | None = None, **kwargs: object) -> Tensor:
        """Apply the existing backend with its unchanged precision and options."""
        ...


class AttentionOverride(Protocol):
    """The pinned composable optimized-attention override signature."""

    def __call__(self, kernel: AttentionKernel, query: Tensor, key: Tensor, value: Tensor, heads: int, *, mask: Tensor | None = None, **kwargs: object) -> Tensor:
        """Delegate to the supplied kernel with any prior compatible override."""
        ...


@dataclass(slots=True)
class _RelayCall:
    layout: RelayLayout
    geometry: RelayGeometry | None = None


_ACTIVE_RELAY: ContextVar[_RelayCall | None] = ContextVar("skulk_ltx_relay_call", default=None)


@dataclass(frozen=True, slots=True)
class _RelayKernel:
    geometry: RelayGeometry
    previous: AttentionOverride | None
    interrupt: Callable[[], None]

    def __call__(self, kernel: AttentionKernel, query: Tensor, key: Tensor, value: Tensor, heads: int, *, mask: object = None, **kwargs: object) -> Tensor:
        """Chunk only the bias and queries while preserving the actual GPU kernel."""
        if kwargs.get("skip_reshape") not in (None, False) or kwargs.get("skip_output_reshape") not in (None, False):
            raise ValueError("relay requires the pinned three-dimensional attention hook")
        if mask is not None and not isinstance(mask, Tensor):
            raise ValueError("relay requires the pinned floating text-attention mask")
        previous = self.previous

        def attend(queries: Tensor, keys: Tensor, values: Tensor, bias: Tensor) -> Tensor:
            # A large video has many bounded query chunks. Honor Comfy's
            # normal interrupt between kernels instead of waiting a full pass.
            self.interrupt()
            # Comfy supplies its recursion guard in kwargs. Keep it when a
            # selected kernel falls back through another decorated backend.
            if previous is not None:
                return previous(kernel, queries, keys, values, heads, mask=bias, **kwargs)
            return kernel(queries, keys, values, heads, mask=bias, **kwargs)

        return relay_attention(query, key, value, query_frames=self.geometry.query_frames, real_queries=self.geometry.real_queries, segments=self.geometry.segments, attention=attend, mask=mask)


@dataclass(frozen=True, slots=True)
class _RelayVideoAttention:
    original: Callable[..., Tensor]
    interrupt: Callable[[], None]

    def __call__(self, x: Tensor, context: Tensor | None = None, mask: object = None, pe: object = None, k_pe: object = None, transformer_options: dict[str, object] | None = None) -> Tensor:
        """Delegate all projection, normalization, RoPE, gating and output code."""
        options = {} if transformer_options is None else transformer_options
        scope = _ACTIVE_RELAY.get()
        if scope is not None and context is not None:
            if scope.geometry is None:
                raise ValueError("relay attention ran before its actual video grid was prepared")
            prior = options.get("optimized_attention_override")
            if prior is not None and not callable(prior):
                raise ValueError("relay cannot compose an invalid attention override")
            options = {**options, "optimized_attention_override": _RelayKernel(scope.geometry, None if prior is None else cast(AttentionOverride, prior), self.interrupt)}
        return self.original(x, context=context, mask=mask, pe=pe, k_pe=k_pe, transformer_options=options)


@dataclass(frozen=True, slots=True)
class _RelayForward:
    original: Callable[..., object]

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Isolate positive, negative and nested calls and clear tensors in finally."""
        layout = kwargs.get(RELAY_CONDITION_KEY)
        if layout is not None and not isinstance(layout, RelayLayout):
            raise ValueError("relay conditioning lost its measured text layout")
        scope = None if layout is None else _RelayCall(layout)
        token = _ACTIVE_RELAY.set(scope)
        try:
            return self.original(*args, **kwargs)
        finally:
            if scope is not None:
                scope.geometry = None
            _ACTIVE_RELAY.reset(token)


@dataclass(frozen=True, slots=True)
class _RelayConditions:
    original: Callable[..., object]
    constant: Callable[[object], object]

    def __call__(self, **kwargs: object) -> object:
        """Retain the original single connector pass and prevent mixed CFG batches."""
        result = self.original(**kwargs)
        layout = kwargs.get(RELAY_CONDITION_KEY)
        if layout is None:
            return result
        if not isinstance(layout, RelayLayout) or not isinstance(result, dict):
            raise ValueError("relay requires the pinned conditioning dictionary and layout")
        entries = cast("dict[object, object]", result)
        if any(not isinstance(key, str) for key in entries):
            raise ValueError("relay conditioning keys must remain strings")
        output = {cast(str, key): value for key, value in entries.items()}
        output[RELAY_CONDITION_KEY] = self.constant(layout)
        return output


def _sequence(value: object, length: int, label: str) -> list[object] | tuple[object, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("relay requires the pinned " + label)
    items = cast("list[object] | tuple[object, ...]", value)
    if len(items) != length:
        raise ValueError("relay requires the pinned " + label)
    return items


def _triplet(value: object, label: str) -> tuple[int, int, int]:
    items = _sequence(value, 3, label)
    if any(type(item) is not int or item < 1 for item in items):
        raise ValueError("relay requires positive integer " + label)
    return cast("tuple[int, int, int]", tuple(items))


@dataclass(frozen=True, slots=True)
class _RelayInput:
    original: Callable[..., object]
    scale_factors: tuple[int, int, int]
    patch_size: tuple[int, int, int]
    causal: bool

    def __call__(self, *args: object, **kwargs: object) -> object:
        """Read only the original model's actual filtered coordinates and guides."""
        result = self.original(*args, **kwargs)
        scope = _ACTIVE_RELAY.get()
        if scope is None:
            return result
        raw = args[0] if args else kwargs.get("x")
        if isinstance(raw, (list, tuple)):
            modalities = cast("list[object] | tuple[object, ...]", raw)
            if not modalities:
                raise ValueError("relay requires the original video modality")
            raw = modalities[0]
        if not isinstance(raw, Tensor):
            raise ValueError("relay requires the original video latent tensor")
        indices = args[1] if len(args) > 1 else kwargs.get("keyframe_idxs")
        if indices is not None and not isinstance(indices, Tensor):
            raise ValueError("relay requires pinned tensor keyframe indices")
        processed = _sequence(result, 3, "processed AV input")
        coordinates = _sequence(processed[1], 2, "video/audio coordinate pair")
        info = processed[2]
        if not isinstance(coordinates[0], Tensor) or not isinstance(info, dict):
            raise ValueError("relay requires the actual video grid and surviving-guide metadata")
        guide_tokens = cast("dict[object, object]", info).get("num_guide_tokens", 0)
        if type(guide_tokens) is not int:
            raise ValueError("relay surviving guide count must remain an integer")
        scope.geometry = prepare_relay_geometry(raw, coordinates[0], indices, guide_tokens, scale_factors=self.scale_factors, patch_size=self.patch_size, causal=self.causal, layout=scope.layout)
        return result


def _original_callable(value: object) -> Callable[..., object]:
    # A later graph stage may clone an already patched model. Unwrap only our
    # immutable adapters so the new clone never installs nested relay hooks.
    if isinstance(value, (_RelayForward, _RelayInput, _RelayConditions, _RelayVideoAttention)):
        return value.original
    if not callable(value):
        raise ValueError("relay requires the pinned callable model methods")
    return value


def _constant(value: object) -> object:
    from comfy.conds import CONDConstant

    return CONDConstant(value)


def _check_interrupt() -> None:
    from comfy.model_management import throw_exception_if_processing_interrupted

    throw_exception_if_processing_interrupted()


def patch_prompt_relay(model: RelayModelPatcher, *, constant_factory: Callable[[object], object] = _constant, interrupt: Callable[[], None] = _check_interrupt) -> RelayModelPatcher:
    """Attach restorable, call-scoped video routing to one cloned LTX graph model.

    Only video text-attention methods receive the pinned override hook. Audio,
    self-attention, cross-modal attention and the original model calculations
    remain unchanged. Comfy owns installation/restoration of all object patches;
    no method is changed globally or directly on a live shared model here.
    """
    patched = model.clone()
    scale = _triplet(patched.get_model_object("diffusion_model.vae_scale_factors"), "VAE scale factors")
    patches = _triplet(patched.get_model_object("diffusion_model.patchifier.patch_size"), "patch dimensions")
    causal = patched.get_model_object("diffusion_model.causal_temporal_positioning")
    blocks = patched.get_model_object("diffusion_model.transformer_blocks")
    if type(causal) is not bool or not isinstance(blocks, Sized) or not 1 <= len(blocks) <= 128:
        raise ValueError("relay requires the pinned LTX video transformer geometry")
    patched.add_object_patch("extra_conds", _RelayConditions(_original_callable(patched.get_model_object("extra_conds")), constant_factory))
    patched.add_object_patch("diffusion_model._forward", _RelayForward(_original_callable(patched.get_model_object("diffusion_model._forward"))))
    patched.add_object_patch("diffusion_model._process_input", _RelayInput(_original_callable(patched.get_model_object("diffusion_model._process_input")), scale, patches, causal))
    for index in range(len(blocks)):
        name = f"diffusion_model.transformer_blocks.{index}.attn2.forward"
        patched.add_object_patch(name, _RelayVideoAttention(cast("Callable[..., Tensor]", _original_callable(patched.get_model_object(name))), interrupt))
    return patched
