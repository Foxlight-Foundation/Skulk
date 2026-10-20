"""Check relay against hash-bound Comfy attention and object-patch lifecycles.

Only selected reference methods are evaluated on bounded CPU tensors. This is
source-contract evidence; actual GPU rendering remains a separate acceptance.
"""

from __future__ import annotations

import ast
import contextlib
import functools
import hashlib
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol, cast

import torch
from torch import Tensor

HASHES = {
    "comfy/ldm/lightricks/model.py": "f0292be2a39491d411ad3cf4b58cebd87e62bf2568aafa35814b954828733718",
    "comfy/ldm/modules/attention.py": "9cafaafaf93ff53e8cbefb5e4a204014019985df2da8c1996bf40f1235fc2960",
    "comfy/model_patcher.py": "26da6710a7d84e8e98ea3e0f0bc07b88cfef3f99a7f7d59daa71e650512c6ed4",
    "comfy/conds.py": "72058e9a22c972a9c875819e59d432d30d367fd2f7092ee6c6c45e5a60c959b0",
    "comfy/samplers.py": "f2c264ca9d394612f828e3ffe167c856a278a10e1711b269f0ba65dccb66393f",
    "comfy/utils.py": "f3ee59baf72f6910b7662224854e372c713ba3ad01c89739ba092e5ee22460e6",
    "comfy/ldm/lightricks/symmetric_patchifier.py": "e2d9b8de7a5cc003e74eaba6d0b8303b596df80e59a431eda2d3593ec06c2633",
    "comfy/model_management.py": "ef3f3af0657c1b022ad69b25928fa52dd429db9aed9cb688e89c7fbe68ab50cd",
}
COMFY_REVISION = "40c4fcdf513a4523e39d54a9d391908af8df8171"


def _source(source: Path, relative: str) -> ast.Module:
    # Repositories may currently be on a newer branch. Exported immutable test
    # fixtures have no Git metadata; both paths require the same reviewed hash.
    data = subprocess.run(["git", "-C", str(source), "show", f"{COMFY_REVISION}:{relative}"], check=True, capture_output=True).stdout if (source / ".git").exists() else (source / relative).read_bytes()
    if hashlib.sha256(data).hexdigest() != HASHES[relative]:
        raise ValueError("relay reference differs from pinned Comfy source: " + relative)
    return ast.parse(data)


def _evaluate(namespace: dict[str, object], nodes: list[ast.stmt]) -> None:
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "<pinned-comfy-relay>", "exec"), namespace)


class _Counted:
    def __init__(self, operation: Callable[[Tensor], Tensor]) -> None:
        self.operation = operation
        self.calls = 0

    def __call__(self, values: Tensor) -> Tensor:
        self.calls += 1
        return self.operation(values)


class _AttentionState:
    def __init__(self) -> None:
        self.heads = 2
        self.dim_head = 4
        self.attn_precision = torch.float32
        self.to_q = _Counted(lambda values: values * 0.9 + 0.02)
        self.to_k = _Counted(lambda values: values * 1.1 - 0.04)
        self.to_v = _Counted(lambda values: values * 0.8 + 0.06)
        self.q_norm = _Counted(lambda values: torch.nn.functional.rms_norm(values, (8,), eps=1e-5))
        self.k_norm = _Counted(lambda values: torch.nn.functional.rms_norm(values, (8,), eps=1e-5))
        self.to_gate_logits = _Counted(lambda values: values[..., :2] * 0.15)
        self.to_out = _Counted(lambda values: values * 0.7 - 0.03)

    def assert_once(self) -> None:
        for operation in (self.to_q, self.to_k, self.to_v, self.q_norm, self.k_norm, self.to_gate_logits, self.to_out):
            assert operation.calls == 1


class _Forward(Protocol):
    def __call__(self, owner: _AttentionState, values: Tensor, **kwargs: object) -> Tensor: ...


class _IndexList(list[object]):
    def __getattr__(self, name: str) -> object:
        # Mirrors ModuleList's numbered attribute lookup without model weights.
        if name.isdecimal():
            return self[int(name)]
        raise AttributeError(name)


@dataclass
class _AttentionLayer:
    forward: Callable[..., Tensor]


@dataclass
class _Block:
    attn2: _AttentionLayer


class _Diffusion:
    def __init__(self, block: _Block, process: Callable[..., object]) -> None:
        self.vae_scale_factors = (8, 32, 32)
        self.patchifier = SimpleNamespace(patch_size=(1, 1, 1))
        self.causal_temporal_positioning = True
        self.transformer_blocks = _IndexList([block])
        self._process_input = process
        self._forward = process

    def process(self, *args: object, **kwargs: object) -> object:
        return self._process_input(*args, **kwargs)

    def run(self, **kwargs: object) -> object:
        return self._forward(**kwargs)

    def set_forward(self, forward: Callable[..., object]) -> None:
        self._forward = forward

    def assert_original(self, forward: Callable[..., object], process: Callable[..., object]) -> None:
        assert self._forward is forward and self._process_input is process


class _PatcherState(SimpleNamespace):
    """Empty weight bookkeeping around the original lifecycle methods under test."""

    def __init__(self, model: object, load_device: object, offload_device: object, size: int = 1, weight_inplace_update: bool = False) -> None:
        super().__init__(model=model, load_device=load_device, offload_device=offload_device, size=size, weight_inplace_update=weight_inplace_update, patches={}, patches_uuid=None, object_patches={}, weight_wrapper_patches={}, model_options={"transformer_options": {}}, parent=None, force_cast_weights=False, backup={}, backup_buffers={}, object_patches_backup={}, pinned=set(), attachments={}, additional_models={}, callbacks={}, wrappers={}, is_injected=False, skip_injection=False, injections={}, hook_patches={}, hook_patches_backup=None, cached_hook_patches={}, hook_backup={}, current_hooks=None, forced_hooks=None, is_clip=False, hook_mode=None, cached_patcher_init=None, is_multigpu_base_clone=False, clone_base_uuid=None)

    def get_all_callbacks(self, _kind: object) -> tuple[Callable[..., object], ...]:
        return ()

    def use_ejected(self) -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()

    def inject_model(self) -> None:
        pass

    def eject_model(self) -> None:
        pass


class _Lifecycle(Protocol):
    object_patches: dict[str, object]
    object_patches_backup: dict[str, object]

    def clone(self) -> _Lifecycle: ...
    def get_model_object(self, name: str) -> object: ...
    def add_object_patch(self, name: str, value: object) -> None: ...
    def patch_model(self, *, load_weights: bool) -> object: ...
    def unpatch_model(self, *, unpatch_weights: bool) -> None: ...


def _lifecycle_factory(trees: dict[str, ast.Module], namespace: dict[str, object], comfy: SimpleNamespace) -> Callable[[object, object, object], _Lifecycle]:
    utils = {node.name: node for node in trees["comfy/utils.py"].body if isinstance(node, ast.FunctionDef) and node.name in ("resolve_attr", "get_attr", "set_attr", "deepcopy_list_dict")}
    assert len(utils) == 4
    unset = [node for node in trees["comfy/utils.py"].body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "ATTR_UNSET" for target in node.targets)]
    assert len(unset) == 1
    _evaluate(namespace, list(unset))
    _evaluate(namespace, list(utils.values()))
    comfy.utils = SimpleNamespace(**{name: namespace[name] for name in utils})
    namespace["_PatcherState"] = _PatcherState
    namespace["CallbacksMP"] = SimpleNamespace(ON_CLONE="clone")
    patcher_tree = trees["comfy/model_patcher.py"]
    hooks = next(node for node in patcher_tree.body if isinstance(node, ast.FunctionDef) and node.name == "create_hook_patches_clone")
    patcher = next(node for node in patcher_tree.body if isinstance(node, ast.ClassDef) and node.name == "ModelPatcher")
    names = {"is_dynamic", "model_size", "get_clone_model_override", "clone", "add_object_patch", "get_model_object", "patch_model", "unpatch_model"}
    methods = [node for node in patcher.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(methods) == len(names)
    selected = ast.ClassDef(name="ModelPatcher", bases=[ast.Name(id="_PatcherState", ctx=ast.Load())], keywords=[], body=list(methods), decorator_list=[], type_params=[])
    _evaluate(namespace, [hooks, selected])
    return cast("Callable[[object, object, object], _Lifecycle]", namespace["ModelPatcher"])


def check_relay_reference(source: Path) -> dict[str, object]:
    """Exercise actual pinned hooks with RoPE, all-head norms, gating and CFG.

    The original full CrossAttention forward is compared with a dense reference
    override. A decorated backend fallback must retain Comfy's recursion guard,
    and projections must execute once despite multiple query chunks.
    """
    from skulk_comfy_ltx.relay import relay_bias, relay_segments
    from skulk_comfy_ltx.relay_binding import AttentionKernel, patch_prompt_relay
    from skulk_comfy_ltx.relay_encoding import RELAY_CONDITION_KEY, RelayLayout
    from skulk_comfy_ltx.relay_geometry import prepare_relay_geometry

    trees = {relative: _source(source, relative) for relative in HASHES}
    namespace: dict[str, object] = {"torch": torch, "functools": functools, "threading": threading, "AttentionTensorContainer": type("UnusedContainer", (), {})}
    interrupt_nodes: list[ast.stmt] = []
    for node in trees["comfy/model_management.py"].body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in ("InterruptProcessingException", "interrupt_current_processing", "processing_interrupted", "throw_exception_if_processing_interrupted") or isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in ("interrupt_processing_mutex", "interrupt_processing") for target in node.targets):
            interrupt_nodes.append(node)
    assert len(interrupt_nodes) == 6
    _evaluate(namespace, interrupt_nodes)
    set_interrupt = cast("Callable[[bool], None]", namespace["interrupt_current_processing"])
    check_interrupt = cast("Callable[[], None]", namespace["throw_exception_if_processing_interrupted"])
    interrupted = cast("Callable[[], bool]", namespace["processing_interrupted"])
    interrupt_exception = cast("type[BaseException]", namespace["InterruptProcessingException"])
    interrupt_checks: list[bool] = []
    interrupt_after_chunk = False

    def interrupt() -> None:
        interrupt_checks.append(True)
        check_interrupt()

    coordinate_functions = [node for node in trees["comfy/ldm/lightricks/symmetric_patchifier.py"].body if isinstance(node, ast.FunctionDef) and node.name == "latent_to_pixel_coords"]
    assert len(coordinate_functions) == 1
    _evaluate(namespace, list(coordinate_functions))
    coordinate_map = cast("Callable[[Tensor, tuple[int, int, int], bool], Tensor]", namespace["latent_to_pixel_coords"])
    coordinate_cases = 0
    for causal in (False, True):
        for scale in (4, 8, 16):
            latent_coords = torch.zeros(1, 3, 5, 2)
            latent_coords[:, 0, :, 0] = torch.tensor([0, 2, 3, 6, 9])
            latent_coords[:, 0, :, 1] = latent_coords[:, 0, :, 0] + 1
            actual_coords = coordinate_map(latent_coords, (scale, 32, 32), causal)
            grid = prepare_relay_geometry(torch.zeros(1, 1, 10, 1, 1), actual_coords, None, 0, scale_factors=(scale, 32, 32), patch_size=(1, 1, 1), causal=causal, layout=RelayLayout(((1, 3), (3, 5)), (1, 1)))
            torch.testing.assert_close(grid.query_frames, torch.tensor([[0.0, 2.0, 3.0, 6.0, 9.0]]))
            coordinate_cases += 1
    wrapper_nodes = [node for node in trees["comfy/ldm/modules/attention.py"].body if isinstance(node, ast.FunctionDef) and node.name == "wrap_attn"]
    assert len(wrapper_nodes) == 1
    _evaluate(namespace, list(wrapper_nodes))
    wrap = cast("Callable[[AttentionKernel], AttentionKernel]", namespace["wrap_attn"])
    kernel_rows: list[int] = []
    prior_calls: list[int] = []

    def sdpa(query: Tensor, key: Tensor, value: Tensor, heads: int, *, mask: Tensor | None = None, **kwargs: object) -> Tensor:
        assert kwargs.get("_inside_attn_wrapper") is True
        batch, queries, channels = query.shape
        result = torch.nn.functional.scaled_dot_product_attention(query.reshape(batch, queries, heads, channels // heads).transpose(1, 2), key.reshape(batch, -1, heads, channels // heads).transpose(1, 2), value.reshape(batch, -1, heads, channels // heads).transpose(1, 2), attn_mask=mask)
        return result.transpose(1, 2).reshape(batch, queries, channels)

    fallback = wrap(sdpa)

    def selected(query: Tensor, key: Tensor, value: Tensor, heads: int, *, mask: Tensor | None = None, **kwargs: object) -> Tensor:
        kernel_rows.append(query.shape[1])
        assert kwargs.get("attn_precision") == torch.float32
        if interrupt_after_chunk:
            set_interrupt(True)
        return fallback(query, key, value, heads, mask=mask, **kwargs)

    def prior(kernel: AttentionKernel, query: Tensor, key: Tensor, value: Tensor, heads: int, *, mask: Tensor | None = None, **kwargs: object) -> Tensor:
        prior_calls.append(query.shape[1])
        assert mask is not None
        adjusted = mask.clone()
        adjusted[..., 0] += 0.07
        return kernel(query, key, value, heads, mask=adjusted, **kwargs)

    comfy = SimpleNamespace(model_management=SimpleNamespace(in_training=True), ldm=SimpleNamespace(modules=SimpleNamespace(attention=SimpleNamespace(optimized_attention=wrap(selected)))))
    namespace["comfy"] = comfy
    namespace["GuideAttentionMask"] = type("UnusedGuideMask", (), {})
    model_tree = trees["comfy/ldm/lightricks/model.py"]
    nodes = [node for node in model_tree.body if isinstance(node, ast.FunctionDef) and node.name in ("apply_rotary_emb", "apply_rotary_emb_qk")]
    attention_class = next(node for node in model_tree.body if isinstance(node, ast.ClassDef) and node.name == "CrossAttention")
    forward = next(node for node in attention_class.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
    _evaluate(namespace, [*nodes, forward])
    original = cast(_Forward, namespace["forward"])
    generator = torch.Generator().manual_seed(5729)
    query = torch.randn(1, 2337, 8, generator=generator)
    context = torch.randn(1, 13, 8, generator=generator)
    frames = torch.arange(query.shape[1], dtype=torch.float32).remainder(7).unsqueeze(0)
    real = torch.ones_like(frames, dtype=torch.bool)
    real[:, -3:] = False
    segments = relay_segments(((2, 5), (6, 9)), (2, 1), 7)
    dense_bias = relay_bias(frames, segments, context.shape[1], real)
    existing_mask = torch.randn(1, 1, query.shape[1], context.shape[1], generator=generator) * 0.01
    # Nontrivial independent rotations exercise the unchanged original RoPE path.
    angle = torch.tensor(0.41)
    rotation = torch.stack([torch.stack([angle.cos(), -angle.sin()]), torch.stack([angle.sin(), angle.cos()])]).reshape(1, 1, 1, 1, 2, 2).expand(1, 1, 2, 2, 2, 2)
    pe = (rotation, False)

    def dense(kernel: AttentionKernel, queries: Tensor, keys: Tensor, values: Tensor, heads: int, *, mask: Tensor | None = None, **kwargs: object) -> Tensor:
        assert mask is not None
        return prior(kernel, queries, keys, values, heads, mask=mask + dense_bias, **kwargs)

    expected_state = _AttentionState()
    expected = original(expected_state, query, context=context, mask=existing_mask, pe=pe, k_pe=pe, transformer_options={"optimized_attention_override": dense})
    expected_state.assert_once()
    kernel_rows.clear()
    prior_calls.clear()
    actual_state = _AttentionState()
    original_attention = functools.partial(original, actual_state)
    options: dict[str, object] = {"optimized_attention_override": prior, "sentinel": object()}

    pixel_coords = torch.zeros(1, 3, query.shape[1], 2)
    pixel_coords[:, 0, :, 0] = torch.where(frames == 0, 0, frames * 8 - 7)

    def process(*_args: object, **_kwargs: object) -> object:
        return ([query, context], [pixel_coords, torch.zeros(1, 1)], {"num_guide_tokens": 3})

    # Seven original frames, one appended guide frame, and some filtered rows.
    video = torch.zeros(1, 1, 8, 1, 334)
    guide_indices = torch.zeros(1, 3, 334, 2)
    block = _Block(_AttentionLayer(original_attention))
    diffusion = _Diffusion(block, process)

    def run(**_kwargs: object) -> object:
        diffusion.process([video, context], guide_indices, None)
        return block.attn2.forward(query, context=context, mask=existing_mask, pe=pe, k_pe=pe, transformer_options=options)

    diffusion.set_forward(run)

    def conditions(**_kwargs: object) -> dict[str, object]:
        return {}

    def identity(value: object) -> object:
        return value

    root = SimpleNamespace(diffusion_model=diffusion, extra_conds=conditions)
    factory = _lifecycle_factory(trees, namespace, comfy)
    unpatched = factory(root, torch.device("cpu"), torch.device("cpu"))
    # Weight loading is absent; these are complete original clone, patch-lookup,
    # method-installation and restoration methods around empty weight bookkeeping.
    model = cast(_Lifecycle, patch_prompt_relay(unpatched, constant_factory=identity, interrupt=interrupt))
    second = cast(_Lifecycle, patch_prompt_relay(model, constant_factory=identity, interrupt=interrupt))
    assert not unpatched.object_patches
    assert model.object_patches == second.object_patches
    diffusion.assert_original(run, process)
    assert block.attn2.forward is original_attention
    assert model.patch_model(load_weights=False) is root
    actual = diffusion.run(**{RELAY_CONDITION_KEY: RelayLayout(((2, 5), (6, 9)), (2, 1))})
    assert isinstance(actual, Tensor)
    torch.testing.assert_close(actual, expected, atol=5e-7, rtol=1e-5)
    actual_state.assert_once()
    assert kernel_rows == prior_calls == [1024, 1024, 289]
    assert len(interrupt_checks) == 3
    assert options["optimized_attention_override"] is prior and set(options) == {"optimized_attention_override", "sentinel"}
    attention_error = float((actual - expected).abs().max())
    chunk_sizes = kernel_rows.copy()
    kernel_rows.clear()
    interrupt_checks.clear()
    interrupt_after_chunk = True
    try:
        diffusion.run(**{RELAY_CONDITION_KEY: RelayLayout(((2, 5), (6, 9)), (2, 1))})
    except interrupt_exception:
        assert kernel_rows == [1024] and len(interrupt_checks) == 2
        assert not interrupted()  # The real Comfy exception consumes its flag.
    else:
        raise AssertionError("relay ignored the real Comfy interrupt between query chunks")
    interrupt_after_chunk = False
    kernel_rows.clear()
    interrupt_checks.clear()
    # A direct unmarked call after the exception must bypass the hook entirely.
    block.attn2.forward(query, context=context, mask=existing_mask, pe=pe, k_pe=pe, transformer_options=options)
    assert not interrupt_checks and kernel_rows == [query.shape[1]]
    kernel_rows.clear()
    recovered = diffusion.run(**{RELAY_CONDITION_KEY: RelayLayout(((2, 5), (6, 9)), (2, 1))})
    assert isinstance(recovered, Tensor)
    torch.testing.assert_close(recovered, expected, atol=5e-7, rtol=1e-5)
    assert kernel_rows == chunk_sizes and len(interrupt_checks) == 3
    model.unpatch_model(unpatch_weights=False)
    assert not model.object_patches_backup
    diffusion.assert_original(run, process)
    assert block.attn2.forward is original_attention
    assert second.patch_model(load_weights=False) is root
    second.unpatch_model(unpatch_weights=False)
    assert not second.object_patches_backup
    diffusion.assert_original(run, process)
    assert block.attn2.forward is original_attention

    # Actual CONDConstant equality and the sampler's exact dictionary-key gate
    # prevent a positive layout from ever joining its unmarked negative batch.
    cond_nodes = [node for node in trees["comfy/conds.py"].body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in ("is_equal", "CONDRegular", "CONDConstant")]
    _evaluate(namespace, list(cond_nodes))
    constant = cast("Callable[[object], _ConstantContract]", namespace["CONDConstant"])
    layout = RelayLayout(((2, 5), (6, 9)), (2, 1))
    positive = constant(layout)
    assert positive.can_concat(constant(layout))
    assert not positive.can_concat(constant(RelayLayout(((2, 5), (6, 9)), (1, 2))))
    assert positive.concat([constant(layout)]) == layout
    assert positive.process_cond(2).concat([]) == layout
    cond_functions = [node for node in trees["comfy/samplers.py"].body if isinstance(node, ast.FunctionDef) and node.name in ("cond_equal_size", "can_concat_cond")]
    assert len(cond_functions) == 2
    _evaluate(namespace, list(cond_functions))
    concat = cast("Callable[[object, object], bool]", namespace["can_concat_cond"])
    pos = SimpleNamespace(input_x=query, control=None, patches=None, conditioning={RELAY_CONDITION_KEY: positive})
    neg = SimpleNamespace(input_x=query, control=None, patches=None, conditioning={})
    assert not concat(pos, neg) and not concat(neg, pos)
    assert concat(pos, pos)
    return {"pass_all": True, "source_sha256": HASHES, "coordinate_cases": coordinate_cases, "attention_max_absolute_error": attention_error, "query_chunks": chunk_sizes, "projections_norms_gates_once": True, "prior_override_and_recursion_guard": True, "actual_constant_cfg_contract": True, "actual_clone_install_restore": True, "actual_comfy_chunk_interrupt_and_recovery": True}


class _ConstantContract(Protocol):
    def can_concat(self, other: _ConstantContract) -> bool: ...
    def concat(self, others: list[_ConstantContract]) -> object: ...
    def process_cond(self, batch_size: int) -> _ConstantContract: ...
