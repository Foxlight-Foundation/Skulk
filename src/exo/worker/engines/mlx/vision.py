"""Vision encoding pipeline for multimodal (VLM) inference.

Converts images from base64 into vision embeddings that replace image-token
placeholders in the prompt. The pipeline is:

1. **Decode** base64 images to PIL
2. **Preprocess** via the model's HuggingFace image processor
3. **Encode** through the vision tower (+ optional projector) to get features
4. **Expand** image-token placeholders in the prompt to match feature count
5. **Embed** by replacing placeholder token embeddings with vision features

Supports models with bundled vision weights (e.g. Qwen3-VL) and models
with separate vision repos. Results are cached by image content hash to
avoid re-encoding identical images across turns.
"""

import base64
import contextlib
import hashlib
import importlib
import inspect
import io
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mlx_vlm.utils import ImageProcessor
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_vlm.prompt_utils import get_message_json
from mlx_vlm.utils import load_image_processor
from PIL import Image
from safetensors import safe_open
from transformers import AutoImageProcessor

from exo.download.download_utils import build_model_path
from exo.shared.models.model_cards import VisionCardConfig
from exo.shared.types.common import ModelId
from exo.shared.types.mlx import Model
from exo.worker.engines.mlx.cache import encode_prompt
from exo.worker.engines.mlx.utils_mlx import fix_unmatched_think_end_tokens
from exo.worker.runner.bootstrap import logger


def _filter_config(cls: type, d: dict[str, Any]) -> dict[str, Any]:
    valid = set(inspect.signature(cls.__init__).parameters.keys()) - {"self"}
    return {k: v for k, v in d.items() if k in valid}  # type: ignore


_video_processor_patched = False


def _patch_video_processor() -> None:
    """Patch so we don't crash horribly when torch vision isn't installed"""
    # TODO: Update if we add torch vision.
    global _video_processor_patched
    if _video_processor_patched:
        return
    try:
        from transformers.processing_utils import MODALITY_TO_AUTOPROCESSOR_MAPPING

        mapping = MODALITY_TO_AUTOPROCESSOR_MAPPING._MAPPING_NAMES  # type: ignore
        mapping.pop("video_processor", None)
    except (ImportError, AttributeError):
        pass
    _video_processor_patched = True


def decode_base64_image(b64_data: str) -> Image.Image:
    """Decode a raw base64 string into an RGB PIL Image."""
    raw = base64.b64decode(b64_data)
    img = Image.open(io.BytesIO(raw))
    return img.convert("RGB")


def _format_vlm_messages(
    messages: list[dict[str, Any]],
    model_type: str,
) -> list[dict[str, Any]]:
    formatted: list[dict[str, Any]] = []
    for msg in messages:
        role: str = str(msg.get("role", "user"))  # type: ignore
        content: Any = msg.get("content")
        if not isinstance(content, list):
            formatted.append(msg)
            continue
        parts: list[dict[str, Any]] = content  # type: ignore
        text_parts = [str(p["text"]) for p in parts if p.get("type") == "text"]  # type: ignore
        n_images = sum(1 for p in parts if p.get("type") in ("image", "image_url"))
        result: dict[str, Any] = get_message_json(
            model_type, " ".join(text_parts), role, num_images=n_images
        )
        formatted.append(result)
    return formatted


def build_vision_prompt(
    tokenizer: TokenizerWrapper,
    chat_template_messages: list[dict[str, Any]],
    n_tokens_per_image: list[int],
    image_token: str,
) -> str:
    """Build the full prompt string, expanding each image placeholder to the
    correct number of image tokens based on the encoder's output size."""
    logger.info(
        f"Vision prompt messages: {[{k: (v[:50] if isinstance(v, str) else v) for k, v in m.items()} for m in chat_template_messages]}"  # type: ignore
    )
    prompt: str = tokenizer.apply_chat_template(
        chat_template_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Walk the prompt and expand each single image_token placeholder into
    # N copies where N = the number of vision features for that image.
    image_idx = 0
    result: list[str] = []
    i = 0
    pad_len = len(image_token)
    while i < len(prompt):
        if prompt[i : i + pad_len] == image_token:
            n = (
                n_tokens_per_image[image_idx]
                if image_idx < len(n_tokens_per_image)
                else 1
            )
            result.append(image_token * n)
            image_idx += 1
            i += pad_len
        else:
            result.append(prompt[i])
            i += 1

    return "".join(result)


@dataclass
class MediaRegion:
    """A contiguous span of image-token positions in the prompt, tagged with
    a content hash so the KV prefix cache can detect when images change."""

    content_hash: str
    start_pos: int
    end_pos: int


@dataclass
class VisionResult:
    """Output of the vision pipeline: the expanded prompt, its token IDs,
    the vision embeddings to splice in, and the media regions for caching."""

    prompt: str
    prompt_tokens: mx.array
    embeddings: mx.array
    media_regions: list[MediaRegion]


_QUANTIZATION_SUFFIXES = (".biases", ".scales")


def _load_projector_weights(
    projector: nn.Module, weights: dict[str, mx.array], config: dict[str, Any]
) -> None:
    """Load projector weights, quantizing the module first if needed.

    Quantized models store Linear weights in a compressed format with
    accompanying ``.scales`` and ``.biases`` tensors. We detect this by
    checking for those keys and apply ``nn.quantize`` to the projector
    before loading so the shapes match."""
    has_quantized = any(k.endswith(_QUANTIZATION_SUFFIXES) for k in weights)
    if has_quantized:
        quant_cfg = config.get("quantization", {})  # pyright: ignore[reportAny]
        bits = int(quant_cfg.get("bits", 4))  # pyright: ignore[reportAny]
        group_size = int(quant_cfg.get("group_size", 64))  # pyright: ignore[reportAny]
        logger.info(
            f"Quantizing projector module ({bits}-bit, group_size={group_size}) "
            "to match checkpoint format"
        )
        nn.quantize(projector, bits=bits, group_size=group_size)
    projector.load_weights(list(weights.items()))


def _instantiate_projector(
    cls: type,
    model_config: Any,  # pyright: ignore[reportAny]
    vision_config: Any,  # pyright: ignore[reportAny]
    text_config: Any,  # pyright: ignore[reportAny]
) -> nn.Module:
    """Try multiple calling conventions for projector/embedder classes.

    Different model families use different constructor signatures:
    - Qwen/Kimi: ``Projector(model_config)``
    - Gemma 3n: ``Embedder(vision_config, text_config=text_config)``
    - Gemma 4: ``Embedder(embedding_dim, text_hidden_size, eps)``
    """
    params = set(inspect.signature(cls.__init__).parameters.keys()) - {"self"}
    # Try single ModelConfig arg first (Qwen/Kimi pattern)
    if "config" in params or len(params) == 1:
        try:
            return cls(model_config)  # type: ignore
        except TypeError:
            pass
    # Try (multimodal_config, text_config) pattern (Gemma 3n)
    if "multimodal_config" in params or "text_config" in params:
        try:
            return cls(vision_config, text_config=text_config)  # type: ignore
        except TypeError:
            pass
    # Try raw dims pattern (Gemma 4: embedding_dim, text_hidden_size, eps)
    if "embedding_dim" in params:
        kwargs: dict[str, Any] = {
            "embedding_dim": getattr(vision_config, "hidden_size", 768),  # pyright: ignore[reportAny]
            "text_hidden_size": getattr(text_config, "hidden_size", 2048),  # pyright: ignore[reportAny]
        }
        if "eps" in params:
            kwargs["eps"] = getattr(vision_config, "rms_norm_eps", 1e-6)  # pyright: ignore[reportAny]
        return cls(**kwargs)  # type: ignore
    # Fallback: pass through _filter_config
    return cls(**_filter_config(cls, vars(model_config)))  # type: ignore


class VisionEncoder:
    """Lazy-loaded vision tower + projector that encodes PIL images into
    feature tensors. Supports both bundled weights (loaded from the main
    model repo) and separate vision weight repos."""

    def __init__(self, config: VisionCardConfig, model_id: ModelId):
        self._config = config
        self._main_model_path = build_model_path(model_id)
        self._model_path = build_model_path(ModelId(config.weights_repo))
        self._vision_tower: nn.Module | None = None
        self._projector: nn.Module | None = None
        self._processor: "ImageProcessor | None" = None
        self._spatial_merge_size: int = 2
        self._merge_kernel_size: list[int] | None = None
        self._needs_nhwc: bool = False
        self._loaded = False

    def _load_config_json(self) -> dict[str, Any]:
        for candidate in (self._main_model_path, self._model_path):
            path = candidate / "config.json"
            if path.exists():
                with open(path) as f:
                    return json.load(f)  # type: ignore
        return {}

    def _import_mlx_vlm(self, *submodules: str) -> Any:  # type: ignore
        mt = self._config.model_type
        results: list[Any] = []
        for sub in submodules:
            name = f"mlx_vlm.models.{mt}.{sub}"
            results.append(importlib.import_module(name))
        return results[0] if len(results) == 1 else tuple(results)

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._load_weights()
        self._loaded = True

    def _load_weights(self) -> None:
        _patch_video_processor()
        logger.info(f"Loading vision weights from {self._model_path}")
        config = self._load_config_json()
        if not config:
            raise FileNotFoundError(f"config.json not found in {self._model_path}")

        vision_cfg = config.get("vision_config", {})  # type: ignore

        config_mod, vision_mod = self._import_mlx_vlm("config", "vision")  # type: ignore
        vision_config_cls = config_mod.VisionConfig  # type: ignore
        vision_model_cls = vision_mod.VisionModel  # type: ignore

        vision_config = vision_config_cls(  # type: ignore
            **_filter_config(vision_config_cls, vision_cfg)  # type: ignore
        )
        self._spatial_merge_size = getattr(vision_config, "spatial_merge_size", 2)  # type: ignore
        self._vision_tower = vision_model_cls(vision_config)
        model_mod: Any = None
        with contextlib.suppress(ImportError):
            model_mod = self._import_mlx_vlm(self._config.model_type)  # type: ignore

        projector_cls = None
        # Match common projector class naming conventions across model
        # families: *Projector (Qwen, Kimi), *Embedder (Gemma 3n/4).
        _projector_patterns = ("Projector", "Embedder")
        if model_mod is not None:
            for attr_name in dir(model_mod):  # type: ignore
                obj = getattr(model_mod, attr_name)  # type: ignore
                if (
                    isinstance(obj, type)
                    and issubclass(obj, nn.Module)
                    and any(p in attr_name for p in _projector_patterns)
                ):
                    projector_cls = obj
                    break

        if projector_cls is not None:
            text_config = config_mod.TextConfig(  # type: ignore
                **_filter_config(config_mod.TextConfig, config.get("text_config", {}))  # type: ignore
            )
            extra = {
                k: v
                for k, v in config.items()  # type: ignore
                if k not in ("text_config", "vision_config")
            }
            extra.setdefault("model_type", self._config.model_type)
            model_config = config_mod.ModelConfig(  # type: ignore
                text_config=text_config,
                vision_config=vision_config,
                **_filter_config(config_mod.ModelConfig, extra),  # type: ignore
            )
            self._projector = _instantiate_projector(
                projector_cls, model_config, vision_config, text_config
            )

        processor_repo = self._config.processor_repo
        if processor_repo:
            self._load_weights_from_separate_repo(config)
        else:
            self._load_weights_from_model_repo(config)

        repo = processor_repo or str(self._model_path)
        image_proc = load_image_processor(repo)
        if image_proc is not None:
            self._processor = image_proc
        else:
            try:
                self._processor = AutoImageProcessor.from_pretrained(  # type: ignore
                    repo, trust_remote_code=True
                )
            except (ValueError, OSError):
                # transformers may not recognize newer model types (e.g.
                # gemma4). Fall back to the mlx_vlm processor class.
                proc_mod = self._import_mlx_vlm(  # type: ignore
                    f"processing_{self._config.model_type}"
                )
                # Find the *ImageProcessor class in the module.
                proc_cls = None
                for attr in dir(proc_mod):  # type: ignore
                    obj = getattr(proc_mod, attr)  # type: ignore
                    if isinstance(obj, type) and "ImageProcessor" in attr:
                        proc_cls = obj
                        break
                if proc_cls is None:
                    raise ValueError(
                        f"No ImageProcessor found in mlx_vlm.models."
                        f"{self._config.model_type}.processing_{self._config.model_type}"
                    ) from None
                self._processor = proc_cls()  # type: ignore
                logger.info(f"Using mlx_vlm {proc_cls.__name__} as image processor")
        if processor_repo:
            self._merge_kernel_size = vision_cfg.get("merge_kernel_size", [2, 2])  # type: ignore
            self._needs_nhwc = True
        logger.info(f"HF image processor loaded from {repo}")

    def _load_weights_from_separate_repo(self, config: dict[str, Any]) -> None:
        safetensors_files = list(self._model_path.glob("*.safetensors"))
        if not safetensors_files:
            raise FileNotFoundError(f"No safetensors files found in {self._model_path}")

        weights: dict[str, mx.array] = {}
        for sf_path in safetensors_files:
            with safe_open(str(sf_path), framework="pt") as f:
                keys = f.keys()
                for key in keys:
                    tensor = f.get_tensor(key)  # type: ignore
                    np_tensor = tensor.float().numpy()  # type: ignore
                    weights[key] = mx.array(np_tensor, dtype=mx.bfloat16)  # type: ignore

        # Partition weights into vision tower vs projector, stripping prefixes
        # and remapping key names to match mlx_vlm's expected parameter layout.
        vision_weights: dict[str, mx.array] = {}
        projector_weights: dict[str, mx.array] = {}
        for key, val in weights.items():
            if key.startswith("vision_tower."):
                short_key = key[len("vision_tower.") :]
                if short_key.startswith("encoder."):
                    short_key = short_key[len("encoder.") :]
                m = re.match(r"^(blocks\.\d+)\.(wqkv|wo)\.(weight|bias)$", short_key)
                if m:
                    short_key = f"{m.group(1)}.attn.{m.group(2)}.{m.group(3)}"
                if short_key == "patch_embed.proj.weight" and val.ndim == 4:
                    val = val.transpose(0, 2, 3, 1)
                vision_weights[short_key] = val
            elif key.startswith(("mm_projector.", "multi_modal_projector.")):
                if key.startswith("multi_modal_projector."):
                    short_key = key[len("multi_modal_projector.") :]
                    if short_key.startswith("mm_projector."):
                        short_key = short_key[len("mm_projector.") :]
                else:
                    short_key = key[len("mm_projector.") :]
                short_key = short_key.replace("proj.0.", "linear_1.").replace(
                    "proj.2.", "linear_2."
                )
                projector_weights[short_key] = val

        assert self._vision_tower is not None
        self._vision_tower.load_weights(list(vision_weights.items()))
        mx.eval(self._vision_tower.parameters())

        if self._projector is not None and projector_weights:
            _load_projector_weights(self._projector, projector_weights, config)
            mx.eval(self._projector.parameters())

        n_vision = sum(v.size for _, v in vision_weights.items())
        n_proj = sum(v.size for _, v in projector_weights.items())
        logger.info(
            f"Vision encoder loaded: {n_vision / 1e6:.1f}M params"
            + (f", projector: {n_proj / 1e6:.1f}M params" if n_proj else "")
        )

    def _load_weights_from_model_repo(self, config: dict[str, Any]) -> None:
        safetensors_files = sorted(self._model_path.glob("*.safetensors"))
        if not safetensors_files:
            raise FileNotFoundError(f"No safetensors files found in {self._model_path}")

        vision_prefixes = ["vision_tower.", "model.visual."]
        # Gemma 3n/4 store their projector (MultimodalEmbedder) under
        # "embed_vision." in the same safetensors files as vision weights.
        projector_prefixes = ["embed_vision."]
        vision_weights: dict[str, mx.array] = {}
        projector_weights: dict[str, mx.array] = {}
        found_raw_prefix = False
        for sf_path in safetensors_files:
            file_weights: dict[str, mx.array] = mx.load(str(sf_path))  # type: ignore
            for key, val in file_weights.items():
                for prefix in vision_prefixes:
                    if key.startswith(prefix):
                        short_key = key[len(prefix) :]
                        vision_weights[short_key] = val
                        if prefix == "model.visual.":
                            found_raw_prefix = True
                        break
                else:
                    for prefix in projector_prefixes:
                        if key.startswith(prefix):
                            short_key = key[len(prefix) :]
                            projector_weights[short_key] = val
                            break

        if not vision_weights:
            raise ValueError(
                f"No vision weights found with prefixes {vision_prefixes} in {self._model_path}. "
                "Ensure the model repo contains bundled vision weights."
            )

        assert self._vision_tower is not None
        if found_raw_prefix and hasattr(self._vision_tower, "sanitize"):
            vision_weights = self._vision_tower.sanitize(vision_weights)  # type: ignore

        self._vision_tower.load_weights(list(vision_weights.items()))  # type: ignore
        mx.eval(self._vision_tower.parameters())

        if self._projector is not None and projector_weights:
            _load_projector_weights(self._projector, projector_weights, config)
            mx.eval(self._projector.parameters())

        n_vision = sum(v.size for _, v in vision_weights.items())  # type: ignore
        n_proj = sum(v.size for _, v in projector_weights.items())
        logger.info(
            f"Vision encoder loaded: {n_vision / 1e6:.1f}M params"
            + (f", projector: {n_proj / 1e6:.1f}M params" if n_proj else "")
        )

    def encode_images(self, images: list[str]) -> tuple[mx.array, list[int]]:
        """Encode base64 images into feature tensors and per-image token counts."""
        self.ensure_loaded()
        assert self._vision_tower is not None
        assert self._processor is not None

        pil_images = [decode_base64_image(b64) for b64 in images]
        for idx, img in enumerate(pil_images):
            logger.info(f"Image {idx}: {img.width}x{img.height} mode={img.mode}")

        pixel_values, grid_thw, n_tokens_per_image = self._preprocess_images(
            pil_images
        )
        hidden_states = self._run_vision_tower(pixel_values, grid_thw)

        if self._projector is not None:
            image_features: mx.array = self._projector(hidden_states)
        else:
            image_features = hidden_states

        return image_features, n_tokens_per_image

    def _preprocess_images(
        self, pil_images: list[Image.Image]
    ) -> tuple[mx.array | list[mx.array], mx.array | None, list[int]]:
        """Run the image processor and return pixel values, optional grid, and token counts."""
        assert self._processor is not None

        if self._config.processor_repo:
            processed = self._processor.preprocess(
                [{"type": "image", "image": img} for img in pil_images],
                return_tensors="np",
            )
            pixel_values = mx.array(processed["pixel_values"])  # type: ignore
            grid_thw = mx.array(processed["grid_thws"])  # type: ignore
            assert self._merge_kernel_size is not None
            merge_length = int(np.prod(self._merge_kernel_size))
            n_tokens_per_image = [
                int(mx.prod(grid_thw[i]).item()) // merge_length
                for i in range(grid_thw.shape[0])
            ]
            return pixel_values, grid_thw, n_tokens_per_image

        raw: Any = self._processor(images=pil_images, return_tensors="np")

        # Gemma 4's processor returns (data_dict, num_soft_tokens_per_image).
        if isinstance(raw, tuple):
            data_dict = dict(raw[0])  # type: ignore
            soft_tokens: list[int] = list(raw[1])  # type: ignore
            pv_raw = data_dict["pixel_values"]  # pyright: ignore[reportUnknownVariableType]
            if isinstance(pv_raw, list):
                # Variable-sized images — keep as list for per-image encoding.
                pixel_values_list: list[mx.array] = [mx.array(v) for v in pv_raw]  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
                return pixel_values_list, None, soft_tokens
            return mx.array(pv_raw), None, soft_tokens  # pyright: ignore[reportUnknownArgumentType]

        # Standard HuggingFace processor (Qwen, SiglipImageProcessor, etc.)
        pv_key = "pixel_values"
        pixel_values = mx.array(raw[pv_key])  # type: ignore

        # Processors that return grid info (Qwen-VL family).
        grid_key = "image_grid_thw"
        if grid_key in raw:
            grid_thw = mx.array(raw[grid_key])  # type: ignore
            merge_unit = self._spatial_merge_size**2
            n_tokens_per_image = [
                int(
                    grid_thw[i, 0].item()
                    * grid_thw[i, 1].item()
                    * grid_thw[i, 2].item()
                )
                // merge_unit
                for i in range(grid_thw.shape[0])
            ]
            return pixel_values, grid_thw, n_tokens_per_image

        # Gemma 3n / simple processors: no grid info, use config's
        # vision_soft_tokens_per_image or the vision tower's default_output_length.
        n_per_image = self._get_soft_tokens_per_image()
        n_tokens_per_image = [n_per_image] * len(pil_images)
        return pixel_values, None, n_tokens_per_image

    def _get_soft_tokens_per_image(self) -> int:
        """Determine the number of soft tokens per image from model config.

        Caches the result to avoid re-reading config.json on every call."""
        cached = getattr(self, "_soft_tokens_cache", None)
        if cached is not None:
            return int(cached)

        result = 256  # sensible default for SiglipImageProcessor-based models
        config = self._load_config_json()
        if config:
            # Top-level vision_soft_tokens_per_image (gemma3n, gemma4).
            vst = config.get("vision_soft_tokens_per_image")
            if vst is not None:
                result = int(vst)  # pyright: ignore[reportAny]
            else:
                # VisionConfig.default_output_length (gemma4).
                vc: dict[str, Any] = config.get("vision_config", {})  # pyright: ignore[reportAny]
                dol = vc.get("default_output_length")
                if dol is not None:
                    result = int(dol)  # pyright: ignore[reportAny]

        self._soft_tokens_cache = result
        return result

    def _run_vision_tower(
        self, pixel_values: mx.array | list[mx.array], grid_thw: mx.array | None
    ) -> mx.array:
        """Run the vision tower, dispatching by input format."""
        assert self._vision_tower is not None

        if self._needs_nhwc:
            assert grid_thw is not None
            grid_hw = grid_thw[:, 1:] if grid_thw.shape[-1] == 3 else grid_thw
            return self._vision_tower(
                pixel_values.transpose(0, 2, 3, 1),  # type: ignore
                output_hidden_states=True,
                grid_thw=grid_hw,
            )

        if isinstance(pixel_values, list):
            # Variable-sized images (Gemma 4): encode each independently and
            # concatenate features.
            features: list[mx.array] = []
            for pv in pixel_values:
                if pv.ndim == 3:
                    pv = pv[None]  # add batch dim
                out: mx.array = self._vision_tower(pv)
                out = out[0] if isinstance(out, tuple) else out
                if out.ndim == 3:
                    out = out.reshape(-1, out.shape[-1])  # flatten batch
                features.append(out)
            return mx.concatenate(features, axis=0)

        if grid_thw is not None:
            result = self._vision_tower(pixel_values, grid_thw)
        else:
            # No grid info (Gemma 3n/4, SiglipImageProcessor) — vision tower
            # takes just pixel_values.
            result = self._vision_tower(pixel_values)
        out = result[0] if isinstance(result, tuple) else result
        # Some vision towers (Gemma 4) preserve the batch dimension.
        # Flatten to (total_tokens, hidden_dim) for create_vision_embeddings.
        if out.ndim == 3:
            out = out.reshape(-1, out.shape[-1])
        return out


def get_inner_model(model: nn.Module) -> Any:  # type: ignore
    """Traverse the model tree to find the inner transformer with ``embed_tokens``."""
    for candidate in (
        getattr(model, "model", None),
        getattr(getattr(model, "language_model", None), "model", None),
    ):
        if candidate is not None and hasattr(candidate, "embed_tokens"):  # type: ignore
            return candidate  # type: ignore

    raise ValueError(
        f"Could not find inner transformer (embed_tokens) in {type(model).__name__}. "
        "Add a new pattern to _get_inner_model() for this architecture."
    )


def create_vision_embeddings(
    model: Model,
    prompt_tokens: mx.array,
    image_features: mx.array,
    image_token_id: int,
) -> mx.array:
    """Replace image-token placeholder embeddings with vision features.

    Uses cumsum indexing to map each image-token position to the corresponding
    row in ``image_features``, then splices them into the text embeddings."""
    inner = get_inner_model(model)  # type: ignore
    embed_tokens = inner.embed_tokens  # type: ignore

    input_embeddings: mx.array = embed_tokens(prompt_tokens[None])  # type: ignore

    is_image: mx.array = mx.equal(prompt_tokens, image_token_id)
    n_placeholders = int(mx.sum(is_image).item())

    if n_placeholders > 0:
        if n_placeholders != image_features.shape[0]:
            logger.warning(
                f"Placeholder count ({n_placeholders}) != image features "
                f"({image_features.shape[0]}). Using min of both."
            )
            n = min(n_placeholders, image_features.shape[0])
            image_features = image_features[:n]

        # Map each image-token position to its feature row via cumulative sum:
        # cumsum over the boolean mask gives 1-based indices; subtract 1 for 0-based.
        # Clip so non-image positions (which get -1) don't go out of bounds.
        image_indices = mx.cumsum(is_image.astype(mx.int32)) - 1
        image_indices = mx.clip(image_indices, 0, image_features.shape[0] - 1)

        # Gather vision features at image positions, keep text embeddings elsewhere
        gathered = image_features[image_indices].astype(input_embeddings.dtype)
        result = mx.where(is_image[:, None], gathered, input_embeddings[0])
        input_embeddings = result[None]

    return input_embeddings


def _find_media_regions(
    prompt_tokens: mx.array,
    images: list[str],
    image_token_id: int,
    boi_token_id: int | None = None,
    eoi_token_id: int | None = None,
) -> list[MediaRegion]:
    """Find contiguous image-token runs and expand to include BOI/EOI markers."""
    tokens_np = np.array(prompt_tokens)
    is_pad = tokens_np == image_token_id  # type: ignore

    regions: list[MediaRegion] = []
    in_run = False
    run_start = 0
    for pos, pad in enumerate(is_pad):  # type: ignore
        if pad and not in_run:
            run_start = pos
            in_run = True
        elif not pad and in_run:
            regions.append(
                MediaRegion(content_hash="", start_pos=run_start, end_pos=pos)
            )
            in_run = False
    if in_run:
        regions.append(
            MediaRegion(content_hash="", start_pos=run_start, end_pos=len(tokens_np))
        )

    # Expand region boundaries to include surrounding BOI/EOI tokens so
    # the KV prefix cache invalidates the full image span.
    for region in regions:
        if (
            boi_token_id is not None
            and region.start_pos > 0
            and tokens_np[region.start_pos - 1] == boi_token_id
        ):
            region.start_pos -= 1
        if (
            eoi_token_id is not None
            and region.end_pos < len(tokens_np)
            and tokens_np[region.end_pos] == eoi_token_id
        ):
            region.end_pos += 1

    for i, region in enumerate(regions):
        if i < len(images):
            img = decode_base64_image(images[i])
            region.content_hash = hashlib.sha256(img.tobytes()).hexdigest()
        else:
            logger.warning(f"Media region {i} has no corresponding image")

    return regions


class VisionProcessor:
    """
    Pipeline for vision models:
    1. Encode images into features (or grab from cache)
    2. Replace image placeholders with the features
    3. Build vision prompt
    4. Provide media regions for prefix caching
    """

    def __init__(self, config: VisionCardConfig, model_id: ModelId):
        self.vision_config = config
        self._encoder = VisionEncoder(config, model_id)
        self._feature_cache: dict[str, tuple[mx.array, list[int]]] = {}
        self._feature_cache_max = 32

    def load(self) -> None:
        self._encoder.ensure_loaded()

    def _image_cache_key(self, images: list[str]) -> str:
        h = hashlib.sha256()
        for img in images:
            pil = decode_base64_image(img)
            h.update(pil.tobytes())
        return h.hexdigest()

    def process(
        self,
        images: list[str],
        chat_template_messages: list[dict[str, Any]],
        tokenizer: TokenizerWrapper,
        model: Model,
    ) -> VisionResult:
        logger.info(f"Vision pipeline: {len(images)} image(s)")

        cache_key = self._image_cache_key(images)
        cached = self._feature_cache.pop(cache_key, None)
        if cached is not None:
            self._feature_cache[cache_key] = cached
            image_features, n_tokens_per_image = cached
        else:
            image_features, n_tokens_per_image = self._encoder.encode_images(images)
            self._feature_cache[cache_key] = (image_features, n_tokens_per_image)
            while len(self._feature_cache) > self._feature_cache_max:
                del self._feature_cache[next(iter(self._feature_cache))]
        logger.info(
            f"Vision features: {image_features.shape} "
            f"({image_features.shape[0]} tokens, per-image: {n_tokens_per_image})"
        )

        image_token = self.vision_config.image_token
        if image_token is None:
            image_token = tokenizer.decode([self.vision_config.image_token_id])

        formatted_messages = _format_vlm_messages(
            chat_template_messages, self.vision_config.model_type
        )

        prompt = build_vision_prompt(
            tokenizer,
            formatted_messages,
            n_tokens_per_image,
            image_token,
        )

        logger.info(
            f"Expanded prompt has {prompt.count(image_token)} image_token occurrences, total len={len(prompt)}"
        )

        prompt_tokens: mx.array = encode_prompt(tokenizer, prompt)
        prompt_tokens = fix_unmatched_think_end_tokens(prompt_tokens, tokenizer)
        n_image_tokens = int(
            mx.sum(mx.equal(prompt_tokens, self.vision_config.image_token_id)).item()
        )
        logger.info(
            f"Encoded prompt: {len(prompt_tokens)} tokens, {n_image_tokens} image pad tokens"
        )

        embeddings = create_vision_embeddings(
            model,
            prompt_tokens,
            image_features,
            self.vision_config.image_token_id,
        )
        mx.eval(embeddings)

        media_regions = _find_media_regions(
            prompt_tokens,
            images,
            self.vision_config.image_token_id,
            boi_token_id=self.vision_config.boi_token_id,
            eoi_token_id=self.vision_config.eoi_token_id,
        )

        return VisionResult(
            prompt=prompt,
            prompt_tokens=prompt_tokens,
            embeddings=embeddings,
            media_regions=media_regions,
        )


def prepare_vision(
    images: list[str] | None,
    chat_template_messages: list[dict[str, Any]] | None,
    vision_processor: VisionProcessor,
    tokenizer: TokenizerWrapper,
    model: Model,
) -> VisionResult | None:
    """Top-level entry point: encode images and build the vision-augmented prompt.

    Returns ``None`` if no images are provided or chat template messages are missing."""
    if not images:
        return None
    if chat_template_messages is None:
        logger.warning(
            "Vision request missing chat_template_messages — ignoring images"
        )
        return None

    return vision_processor.process(
        images=images,
        chat_template_messages=chat_template_messages,
        tokenizer=tokenizer,
        model=model,
    )
