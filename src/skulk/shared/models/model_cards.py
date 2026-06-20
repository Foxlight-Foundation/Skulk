import json
import struct
from collections.abc import Awaitable, Callable, Iterable
from enum import Enum
from typing import Annotated, Any, Final, Literal, NamedTuple, cast

import aiofiles
import aiofiles.os as aios
import tomlkit
from anyio import Path, open_file
from huggingface_hub import model_info
from loguru import logger
from pydantic import (
    AliasChoices,
    BaseModel,
    Field,
    PositiveInt,
    ValidationError,
    ValidationInfo,
    field_serializer,
    field_validator,
    model_validator,
)
from tomlkit.exceptions import TOMLKitError

from skulk.shared.constants import (
    RESOURCES_DIR,
    SKULK_CUSTOM_MODEL_CARDS_DIR,
    SKULK_ENABLE_IMAGE_MODELS,
    SKULK_MODELS_DIRS,
)
from skulk.shared.types.common import ModelId
from skulk.shared.types.memory import Memory
from skulk.shared.types.text_generation import ReasoningEffort
from skulk.utils.pydantic_ext import CamelCaseModel

# kinda ugly...
# TODO: load search path from config.toml
_custom_cards_dir = Path(str(SKULK_CUSTOM_MODEL_CARDS_DIR))
_BUILTIN_CARD_DIRS = [
    Path(RESOURCES_DIR) / "inference_model_cards",
    Path(RESOURCES_DIR) / "image_model_cards",
    Path(RESOURCES_DIR) / "embedding_model_cards",
]

_card_cache: dict[ModelId, "ModelCard"] = {}


def _detect_vision_from_config(model_id: ModelId) -> "VisionCardConfig | None":
    normalized = model_id.normalize()
    for model_dir in [d / normalized for d in SKULK_MODELS_DIRS]:
        config_path = model_dir / "config.json"
        if not config_path.exists():
            continue
        try:
            with open(config_path) as f:
                raw = json.load(f)  # type: ignore
            return ConfigData.model_validate(
                raw, context={"model_id": str(model_id)}
            ).vision
        except Exception:
            continue
    return None


async def _load_cards_from_dir(directory: Path, *, is_custom: bool) -> None:
    """Load all TOML model cards from a directory into the cache."""
    async for toml_file in directory.rglob("*.toml"):
        try:
            card = await ModelCard.load_from_path(toml_file)
            if is_custom:
                card = card.model_copy(update={"is_custom": True})
            if card.vision is None:
                vision = _detect_vision_from_config(card.model_id)
                if vision is not None:
                    card = card.model_copy(update={"vision": vision})
            if card.model_id not in _card_cache:
                _card_cache[card.model_id] = card
        except (ValidationError, TOMLKitError):
            pass


async def _refresh_card_cache() -> None:
    for path in _BUILTIN_CARD_DIRS:
        await _load_cards_from_dir(path, is_custom=False)
    await _load_cards_from_dir(_custom_cards_dir, is_custom=True)


def _is_image_card(card: "ModelCard") -> bool:
    return any(t in (ModelTask.TextToImage, ModelTask.ImageToImage) for t in card.tasks)


def get_card(model_id: ModelId) -> "ModelCard | None":
    """Look up a single model card from the cache by ID."""
    return _card_cache.get(model_id)


async def get_model_cards() -> list["ModelCard"]:
    if len(_card_cache) == 0:
        await _refresh_card_cache()
    if SKULK_ENABLE_IMAGE_MODELS:
        return list(_card_cache.values())
    return [c for c in _card_cache.values() if not _is_image_card(c)]


class ModelTask(str, Enum):
    TextGeneration = "TextGeneration"
    TextToImage = "TextToImage"
    ImageToImage = "ImageToImage"
    TextEmbedding = "TextEmbedding"


class ComponentInfo(CamelCaseModel):
    """One weight component of a multi-component model (e.g. a diffusion stack)."""

    component_name: str
    """Logical name of this component (e.g. ``text_encoder``, ``transformer``)."""
    component_path: str
    """Repo-relative subdirectory holding this component's weights."""
    storage_size: Memory
    """On-disk size of this component's weights."""
    n_layers: PositiveInt | None = None
    """Layer count for this component when it is shardable; ``None`` otherwise."""
    can_shard: bool
    """Whether this component may be split across nodes (vs. loaded whole)."""
    safetensors_index_filename: str | None = None
    """The component's ``*.safetensors.index.json`` filename when sharded across
    files; ``None`` for a single-file component."""


class VisionCardConfig(CamelCaseModel):
    """Vision configuration attached to a model card for VLM support.

    Populated from the ``[vision]`` section of a TOML model card or
    auto-detected from ``config.json`` during card creation."""

    image_token_id: int
    """Token id the model uses as the image placeholder in the prompt."""
    model_type: str
    """Vision model-type tag from ``config.json``, selecting the image processor."""
    weights_repo: str = ""
    """Repo holding the vision-tower weights when separate from the LM; empty if
    bundled with the main weights."""
    image_token: str | None = None
    """The literal image placeholder string, when distinct from ``image_token_id``."""
    processor_repo: str | None = None
    """Repo providing the image processor/preprocessor config, if not the main repo."""
    boi_token_id: int | None = None
    """Begin-of-image token id, for families that bracket image spans."""
    eoi_token_id: int | None = None
    """End-of-image token id, for families that bracket image spans."""


def multi_node_speculation_disabled(
    runtime: "RuntimeCapabilityCardConfig | None", world_size: int
) -> bool:
    """True when the card forbids speculation for this placement size.

    Shared by the runner's drafter-load gate and the generation loop's
    distributed-agreement gate so both make the identical, rank-symmetric
    decision from the card alone.
    """
    return (
        world_size > 1
        and runtime is not None
        and runtime.speculative_multi_node is False
    )


class ReasoningFormat(str, Enum):
    """Reasoning marker formats used by model families."""

    None_ = "none"
    TokenDelimited = "token_delimited"
    ChannelDelimited = "channel_delimited"


class PromptRendererType(str, Enum):
    """Prompt renderer strategies supported by the runtime."""

    Tokenizer = "tokenizer"
    Gemma4 = "gemma4"
    Dsml = "dsml"


class OutputParserType(str, Enum):
    """Output parser strategies supported by the runtime."""

    Generic = "generic"
    Gemma4 = "gemma4"
    GptOss = "gpt_oss"
    DeepseekV32 = "deepseek_v32"


class ToolCallFormat(str, Enum):
    """Tool-call output formats emitted by model families."""

    Generic = "generic"
    Gemma4 = "gemma4"
    GptOss = "gpt_oss"
    Dsml = "dsml"


class BuiltinToolType(str, Enum):
    """Builtin tool contracts that Skulk can advertise to model families."""

    WebSearch = "web_search"
    OpenUrl = "open_url"
    ExtractPage = "extract_page"


class ReasoningCardConfig(CamelCaseModel):
    """Optional advanced reasoning capability declarations for a model card."""

    supports_toggle: bool | None = None
    """Whether the model can have reasoning turned on/off per request."""
    supports_budget: bool | None = None
    """Whether the model accepts a reasoning-effort/budget control."""
    format: ReasoningFormat | None = None
    """How reasoning is marked in the output stream: ``none``, ``token_delimited``
    (special tokens), or ``channel_delimited`` (a separate reasoning channel)."""
    default_effort: ReasoningEffort | None = None
    """Reasoning effort applied when the request does not specify one."""
    disabled_effort: ReasoningEffort | None = None
    """The effort value that means "reasoning off" for this model."""

    @field_validator("format", mode="before")
    @classmethod
    def _validate_format(
        cls, value: str | ReasoningFormat | None
    ) -> ReasoningFormat | None:
        if value is None or isinstance(value, ReasoningFormat):
            return value
        return ReasoningFormat(value)


class ModalitiesCardConfig(CamelCaseModel):
    """Optional advanced modality declarations for a model card."""

    supports_audio_input: bool | None = None
    """Whether the model accepts audio input."""
    supports_native_multimodal: bool | None = None
    """Whether the model natively interleaves modalities (vs. a bolt-on adapter)."""


class ToolingCardConfig(CamelCaseModel):
    """Optional tool-calling behavior declarations for a model card."""

    supports_tool_calling: bool | None = None
    """Whether the model supports function/tool calling."""
    tool_call_format: ToolCallFormat | None = None
    """The wire format the model emits tool calls in (``generic``, ``gemma4``,
    ``gpt_oss``, ``dsml``), selecting the output parser."""
    builtin_tools: list[BuiltinToolType] | None = None
    """Builtin tools Skulk advertises to this model (e.g. ``web_search``,
    ``open_url``, ``extract_page``)."""

    @field_validator("tool_call_format", mode="before")
    @classmethod
    def _validate_tool_call_format(
        cls, value: str | ToolCallFormat | None
    ) -> ToolCallFormat | None:
        if value is None or isinstance(value, ToolCallFormat):
            return value
        return ToolCallFormat(value)

    @field_validator("builtin_tools", mode="before")
    @classmethod
    def _validate_builtin_tools(
        cls,
        value: list[str | BuiltinToolType] | None,
    ) -> list[BuiltinToolType] | None:
        if value is None:
            return value
        return [
            item if isinstance(item, BuiltinToolType) else BuiltinToolType(item)
            for item in value
        ]


class PlacementCardConfig(CamelCaseModel):
    """Hardware/routing constraints the planner reads from a model card (#149).

    The only card section the planner consults directly. Defaults describe the
    current implicit assumption (an MLX model with no extra memory floor), so
    cards without a ``[placement]`` section behave exactly as before.
    """

    compatible_backends: frozenset[str] = frozenset({"mlx"})
    """Hard constraint: only route to nodes whose advertised backends intersect
    this set. Making the implicit ``{"mlx"}`` explicit is what enables future
    heterogeneous (llama_cpp / rocm / cuda) routing."""

    min_vram_gib: float | None = None
    """Hard constraint: planner gates on node available memory when set."""

    max_context_tokens: int | None = None
    """Soft: caps the placement-time KV budget check (see #145) when set."""

    backend_preference: tuple[str, ...] = ()
    """Soft, ordered preference among the node's backend tags (e.g.
    ``("llama_cpp-vulkan", "llama_cpp-rocm")``).

    Unlike ``compatible_backends`` (a hard filter on which nodes are eligible),
    this only *ranks* eligible nodes/devices: the planner prefers a node that
    advertises an earlier-listed tag, and the runner picks the earliest-listed
    backend the chosen node actually has. The same model runs on any compatible
    backend, but their performance differs per model, so this captures "fastest
    on Vulkan, ROCm is an acceptable fallback" while still degrading gracefully
    to a node that only offers the fallback. Order is significant and preserved;
    an empty tuple means no preference (use the node's default)."""

    @field_validator("compatible_backends", mode="before")
    @classmethod
    def _coerce_compatible_backends(cls, v: object) -> object:
        # TOML provides a list (compatible_backends = ["mlx"]); strict mode
        # would reject it for a frozenset field and make any card with an
        # explicit [placement] section unloadable. Coerce before validation.
        if isinstance(v, (list, tuple, set, frozenset)):
            return frozenset(cast("Iterable[str]", v))
        return v

    @field_serializer("compatible_backends")
    def _serialize_compatible_backends(self, value: frozenset[str]) -> list[str]:
        # tomlkit cannot encode a frozenset, and ModelCard.save() now always
        # includes this section. Emit a sorted list for TOML and JSON alike.
        return sorted(value)

    @field_validator("backend_preference", mode="before")
    @classmethod
    def _coerce_backend_preference(cls, v: object) -> object:
        # TOML/JSON deliver a list; strict mode rejects a list for a tuple
        # field. Coerce while PRESERVING ORDER (unlike compatible_backends, the
        # preference is ranked, so it must not be turned into a set).
        if isinstance(v, (list, tuple)):
            return tuple(cast("Iterable[str]", v))
        return v

    @field_serializer("backend_preference")
    def _serialize_backend_preference(self, value: tuple[str, ...]) -> list[str]:
        # tomlkit cannot encode a tuple; emit an order-preserving list.
        return list(value)


class RuntimeCapabilityCardConfig(CamelCaseModel):
    """Optional runtime behavior hints for a model card."""

    prompt_renderer: PromptRendererType | None = None
    """How prompts are rendered for this model (``tokenizer`` chat template,
    ``gemma4``, ``dsml``); ``None`` uses the family default."""
    output_parser: OutputParserType | None = None
    """How model output is parsed (``generic``, ``gemma4``, ``gpt_oss``,
    ``deepseek_v32``), e.g. for reasoning/tool-call extraction; ``None`` uses the
    family default."""
    metal_fast_synch: bool | None = None
    """Per-model override for the MLX ``MLX_METAL_FAST_SYNCH`` flag.

    ``None`` means "no opinion" — fall through to the cluster default
    selected by the runner. Set explicitly to ``False`` for models that
    deadlock under FAST_SYNCH on the ring backend (e.g. gemma-4 with
    multimodal load: the Metal command queue wedges in
    ``pipeline_last_eval_output``, transitively starves WindowServer,
    and trips the macOS kernel watchdog into a panic). Set explicitly
    to ``True`` for models that have been measured to benefit and are
    known to be safe under the deployment's collective backend.
    """
    mtp_heads: bool | None = None
    """True when native MTP prediction heads are available via sidecar.

    Set alongside ``mtp_sidecar_repo``. When false or absent, the runner
    skips sidecar loading and uses standard autoregressive generation.
    """
    mtp_max_depth: int | None = None
    """Maximum draft depth the MTP heads support.

    Start at 1 for Apple Silicon. Deeper values can be evaluated via
    profiling but are unlikely to amortize on Metal due to near-linear
    verify-pass scaling.
    """
    mtp_sidecar_repo: str | None = None
    """Hugging Face repo ID containing the published ``mtp.safetensors`` sidecar.

    Example: ``"FoxlightAI/qwen3-5-7b-instruct-mtp-q4k"``
    The sidecar is downloaded alongside the base model weights and loaded
    into the runner for speculative decoding. Produced by SWP.
    """
    mtp_norm_convention: Literal["zero_centered", "actual_scale"] | None = None
    """How the sidecar stores its RMSNorm weights.

    ``"zero_centered"`` means deviation-from-1 (the raw Qwen3.5 checkpoint
    convention — the runner applies a +1.0 shift at load, mirroring what
    mlx-lm's ``sanitize()`` does for trunk weights). ``"actual_scale"`` means
    the stored value is the scale itself (DeepSeek convention). ``None``
    falls through to the family default keyed off the detected sidecar
    layout. Override per card when a publisher changes conventions — getting
    this wrong measured 0% draft acceptance on Qwen3.5-2B (issue #192).
    """
    mtp_concat_order: Literal["embed_first", "hidden_first"] | None = None
    """Concatenation order of the MTP fc projection input.

    ``"embed_first"`` = ``fc(concat([enorm(embed(t_next)), hnorm(h)]))`` —
    verified for Qwen3.5 (72.4% offline agreement, issue #192).
    ``"hidden_first"`` is the inherited DeepSeek assumption (unverified).
    ``None`` falls through to the family default keyed off the detected
    sidecar layout.
    """
    speculative_multi_node: bool | None = None
    """Whether speculation may run on multi-node placements of this model.

    ``None`` (default) places no restriction. Set ``False`` for models
    where multi-node speculation is measured SLOWER than plain distributed
    decode: the 2026-06-06 benchmark matrix found gemma-4-26B-A4B (MoE)
    at 30.2 tok/s plain vs 28.2 with MTP on a 2-node pipeline (-7%), while
    single-node MTP on the same model measures 2.2x — fast sharded MoE
    decode plus modest acceptance makes the per-round draft+verify
    overhead net negative. Single-node speculation is unaffected by this
    knob. The decision is card-driven so every rank makes the same
    speculate-or-not choice (the distributed agreement collective requires
    rank symmetry).
    """

    assistant_model_repo: str | None = None
    """Hugging Face repo ID of a companion *assistant* (drafter) model.

    Gemma 4 does speculative decoding differently from the Qwen3/DeepSeek
    ``mtp.*`` heads: instead of embedded prediction heads, it pairs the target
    with a separate small ``gemma4_assistant`` model (e.g.
    ``"mlx-community/gemma-4-26B-A4B-it-assistant-bf16"``) that cross-attends
    over the target's KV cache. When set, the assistant repo is downloaded
    alongside the base model. Mutually exclusive with the ``mtp_*`` fields.

    NOTE: consuming the assistant for speculative generation requires the
    ``gemma4_assistant`` drafter from mlx-vlm >= 0.5.0 and is not yet wired into
    the runner — declaring it here only pre-downloads it. See the Gemma 4 MTP
    initiative in the foxlight-docs hub (Phase C).
    """

    @field_validator("prompt_renderer", mode="before")
    @classmethod
    def _validate_prompt_renderer(
        cls, value: str | PromptRendererType | None
    ) -> PromptRendererType | None:
        if value is None or isinstance(value, PromptRendererType):
            return value
        return PromptRendererType(value)

    @field_validator("output_parser", mode="before")
    @classmethod
    def _validate_output_parser(
        cls, value: str | OutputParserType | None
    ) -> OutputParserType | None:
        if value is None or isinstance(value, OutputParserType):
            return value
        return OutputParserType(value)


class ModelCard(CamelCaseModel):
    """The persisted, declarative metadata Skulk holds for one model.

    This is the **model-card interface**: the single source of truth for how a
    model is sized, sharded, placed, and run. It is created once (from a
    HuggingFace repo or hand-authored), broadcast cluster-wide, and read by the
    planner (placement), the downloader (sizing + which files to fetch), and the
    worker runner (engine + behavior). As a ``CamelCaseModel`` it is camelCase on
    the wire and strict (``extra="forbid"``), so every node in a cluster must run
    the same Skulk version (a stale node rejects newer fields).

    Two layers live here: the *card* (this declarative metadata) and the
    normalized *resolved capability profile* derived from it plus family defaults
    (see ``capabilities.py`` and ``website/docs/model-capabilities.md``). The
    optional ``reasoning`` / ``modalities`` / ``tooling`` / ``runtime`` /
    ``vision`` / ``placement`` sub-configs refine that resolution; when absent,
    conservative family defaults apply.
    """

    model_id: ModelId
    """The model's identifier (a HuggingFace repo id, e.g.
    ``mlx-community/Qwen3.5-9B-4bit``, or a custom id). Slashes become ``--`` in
    the on-disk store directory name."""
    storage_size: Memory
    """On-disk size of the weights this card loads (for a GGUF card, just the
    selected quant's shard group, not every quant the repo hosts). The planner
    uses this for memory-fit and placement-width decisions."""
    n_layers: PositiveInt
    """Number of transformer layers. Drives pipeline sharding (how layers split
    across nodes) and KV-cache sizing."""
    hidden_size: PositiveInt
    """Model hidden dimension, used in memory/KV-cache estimates."""
    supports_tensor: bool
    """Whether the model may be served with tensor parallelism (``Sharding.Tensor``).
    GGUF/llama.cpp cards set this ``False`` (single-node engine)."""
    num_key_value_heads: PositiveInt | None = None
    """KV-head count for grouped-query attention, used in KV-cache sizing. ``None``
    when unknown/not applicable."""
    tasks: list[ModelTask]
    """The task types this model serves (e.g. ``TextGeneration``, ``TextEmbedding``,
    ``ImageGeneration``); selects which runner handles it."""
    components: list[ComponentInfo] | None = None
    """For multi-component models (e.g. a diffusion stack), the per-component
    weight layout. ``None`` for a single-weights model."""
    family: str = ""
    """Model family token (e.g. ``qwen3``, ``gemma4``) used to pick family-specific
    defaults during capability resolution. Empty when not classified."""
    quantization: str = ""
    """Human quantization label (e.g. ``4bit``, ``Q4_K_M``); informational."""
    base_model: str = ""
    """The upstream base model id when this is a quant/finetune of another; empty
    if not applicable."""
    gguf_file: str | None = None
    """For GGUF (llama.cpp) models: the repo-relative path of the weights file the
    runner loads (the selected quant's first shard). Resolved once at card creation
    (preferring a quant over BF16) so the download fetches only that quant and the
    runner loads deterministically, instead of each layer re-globbing/guessing.
    ``None`` for non-GGUF (safetensors/MLX) cards."""
    capabilities: list[str] = []
    """Free-form capability tags carried for compatibility/auxiliary use; the
    structured ``reasoning``/``modalities``/``tooling`` configs are authoritative
    for capability resolution."""
    context_length: int = 0
    """The model's advertised maximum context length in tokens (``0`` if unknown).
    The admission ceiling is the smaller of this and what fits in memory."""
    uses_cfg: bool = False
    """Whether the model uses classifier-free guidance (relevant to some image /
    diffusion models)."""
    trust_remote_code: bool = True
    """Passed to the model loader: whether to execute the repo's custom Python.
    Defaults ``True`` to match upstream loaders; set ``False`` to refuse it."""
    is_custom: bool = False
    """Marks an operator-added custom card (not from the curated catalog). Excluded
    from the persisted card file so it is recomputed per environment."""
    vision: VisionCardConfig | None = None
    """Optional vision (image-input) configuration; ``None`` for text-only models."""
    reasoning: ReasoningCardConfig | None = None
    """Optional reasoning/thinking configuration (toggle, budget, format, default
    effort); ``None`` falls back to family defaults."""
    modalities: ModalitiesCardConfig | None = None
    """Optional extra-modality flags (audio input, native multimodal); ``None``
    falls back to family defaults."""
    tooling: ToolingCardConfig | None = None
    """Optional tool-calling configuration (support, call format, builtin tools);
    ``None`` falls back to family defaults."""
    runtime: RuntimeCapabilityCardConfig | None = None
    """Optional runtime-behavior configuration (prompt renderer, output parser,
    MTP/speculative-decoding sidecar, MLX knobs); ``None`` falls back to defaults."""
    placement: PlacementCardConfig = PlacementCardConfig()
    """Where the model is allowed to run and which backend is preferred: the
    ``compatible_backends`` hard filter and ``backend_preference`` soft score the
    planner uses to route the model to suitable nodes."""

    @model_validator(mode="after")
    def _fill_vision_weights_repo(self) -> "ModelCard":
        if self.vision is not None and not self.vision.weights_repo:
            object.__setattr__(
                self,
                "vision",
                self.vision.model_copy(update={"weights_repo": str(self.model_id)}),
            )
        return self

    @field_validator("tasks", mode="before")
    @classmethod
    def _validate_tasks(cls, v: list[str | ModelTask]) -> list[ModelTask]:
        return [item if isinstance(item, ModelTask) else ModelTask(item) for item in v]

    async def save(self, path: Path) -> None:
        async with await open_file(path, "w") as f:
            py = self.model_dump(exclude_none=True, exclude={"is_custom"})
            data = tomlkit.dumps(py)  # pyright: ignore[reportUnknownMemberType]
            await f.write(data)

    async def save_to_custom_dir(self) -> None:
        await aios.makedirs(str(_custom_cards_dir), exist_ok=True)
        await self.save(_custom_cards_dir / (self.model_id.normalize() + ".toml"))

    @staticmethod
    async def load_from_path(path: Path) -> "ModelCard":
        async with await open_file(path, "r") as f:
            py = tomlkit.loads(await f.read())
            return ModelCard.model_validate(py)

    # Is it okay that model card.load defaults to network access if the card doesn't exist? do we want to be more explicit here?
    @staticmethod
    async def load(model_id: ModelId) -> "ModelCard":
        if model_id not in _card_cache:
            await _refresh_card_cache()
        if (mc := _card_cache.get(model_id)) is not None:
            return mc

        mc = await ModelCard.fetch_from_hf(model_id)
        await mc.save_to_custom_dir()
        _card_cache[model_id] = mc
        return mc

    @staticmethod
    async def fetch_from_hf(model_id: ModelId) -> "ModelCard":
        """Fetches storage size and number of layers for a Hugging Face model, returns Pydantic ModelMeta.

        This is a pure fetch — it does NOT save to disk or update the cache.
        Persistence is handled by the event-sourcing layer (worker event handler).

        Detects GGUF repos (which `mlx-lm` cannot load) and builds a llama.cpp
        card for them instead of the default MLX/safetensors card.
        """
        # GGUF detection is a best-effort probe: it hits the HF model-info API,
        # which the safetensors path below does NOT require (it has its own retry
        # and local-file fallback). A transient/offline/rate-limited probe must
        # therefore NOT block a safetensors card that could otherwise load from
        # cache: treat any probe failure as "not proven GGUF" and fall through.
        try:
            gguf_files = gguf_weight_siblings(model_id)
        except Exception as exc:  # noqa: BLE001  (best-effort probe, see above)
            logger.debug(f"GGUF probe for {model_id} failed ({exc}); assuming non-GGUF")
            gguf_files = []
        if gguf_files:
            return await ModelCard._fetch_gguf_from_hf(model_id, gguf_files)

        # TODO: failure if files do not exist
        config_data = await fetch_config_data(model_id)
        num_layers = config_data.layer_count
        mem_size_bytes = await fetch_safetensors_size(model_id)

        return ModelCard(
            model_id=ModelId(model_id),
            storage_size=mem_size_bytes,
            n_layers=num_layers,
            hidden_size=config_data.hidden_size or 0,
            supports_tensor=config_data.supports_tensor,
            num_key_value_heads=config_data.num_key_value_heads,
            context_length=config_data.max_position_embeddings or 0,
            tasks=[ModelTask.TextGeneration],
            trust_remote_code=True,
            is_custom=True,
            vision=config_data.vision,
        )

    @staticmethod
    async def _fetch_gguf_from_hf(
        model_id: ModelId, gguf_files: "list[tuple[str, int]]"
    ) -> "ModelCard":
        """Build a llama.cpp model card for a GGUF repo.

        Sizes the weights from the GGUF file the runner would actually load
        (`select_gguf_file` picks the first sorted file + its shard group).
        Structural fields come from `config.json` when the repo ships one (most
        community GGUF repos do); a bare repo with no usable config.json has its
        metadata read straight from the selected GGUF file's binary header via a
        ranged read of the file start (#327), so neither path fabricates the
        layer/hidden sizes placement's memory and KV-budget math depend on.
        Stamps the llama.cpp backend tags so placement routes the model only to
        nodes with a llama.cpp engine and prefers a GPU backend.
        """
        selected = select_preferred_gguf(gguf_files)
        selected_size = gguf_shard_group_size(selected, gguf_files)

        # Structural metadata comes from config.json, which most community GGUF
        # repos (bartowski, unsloth, mlx-community) ship alongside the weights.
        # Treat a missing config.json (FileNotFoundError) OR a present-but-unusable
        # one (ValidationError, e.g. no num_hidden_layers, which ConfigData
        # requires) as "no usable config" and fall back to the GGUF header. HF
        # auth / rate-limit / transient network errors are neither and propagate
        # unchanged instead of being mislabeled as "no config.json".
        try:
            config_data = await fetch_config_data(model_id)
        except (FileNotFoundError, ValidationError):
            config_data = None

        if config_data is not None and config_data.layer_count and (
            config_data.hidden_size
        ):
            n_layers = config_data.layer_count
            hidden_size = config_data.hidden_size
            num_key_value_heads = config_data.num_key_value_heads
            context_length = config_data.max_position_embeddings or 0
        else:
            # No usable config.json: read the structural fields from the GGUF
            # binary header. ``selected`` is the first shard, which carries the
            # metadata block, so a ranged read of its start is enough.
            reason = "absent" if config_data is None else "missing layer/hidden sizes"
            logger.info(
                f"GGUF repo {model_id} config.json {reason}; reading model "
                f"metadata from the GGUF header of {selected}"
            )
            from skulk.download.download_utils import range_read

            async def _fetch(offset: int, length: int) -> bytes:
                return await range_read(model_id, "main", selected, offset, length)

            fields = await read_gguf_structural_fields(_fetch)
            n_layers = fields.n_layers
            hidden_size = fields.hidden_size
            num_key_value_heads = fields.num_key_value_heads
            context_length = fields.context_length

        return ModelCard(
            model_id=ModelId(model_id),
            storage_size=selected_size,
            n_layers=n_layers,
            hidden_size=hidden_size,
            # llama.cpp runs single-node in Skulk (no tensor parallelism).
            supports_tensor=False,
            num_key_value_heads=num_key_value_heads,
            context_length=context_length,
            tasks=[ModelTask.TextGeneration],
            quantization=_gguf_quant_label(selected),
            gguf_file=selected,
            trust_remote_code=False,
            is_custom=True,
            placement=PlacementCardConfig(
                compatible_backends=frozenset(
                    {
                        "llama_cpp-vulkan",
                        "llama_cpp-rocm",
                        "llama_cpp-cuda",
                        "llama_cpp-cpu",
                    }
                ),
                # Prefer a GPU backend over CPU; the GPU ordering is a sensible
                # default a card author can tune per model (Vulkan vs ROCm
                # performance is model-dependent).
                backend_preference=(
                    "llama_cpp-vulkan",
                    "llama_cpp-rocm",
                    "llama_cpp-cuda",
                    "llama_cpp-cpu",
                ),
            ),
        )


def add_to_card_cache(card: "ModelCard") -> None:
    """Add or update a model card in the in-memory cache."""
    _card_cache[card.model_id] = card


async def delete_custom_card(model_id: ModelId) -> bool:
    """Delete a user-added custom model card. Returns True if deleted."""
    card_path = _custom_cards_dir / (ModelId(model_id).normalize() + ".toml")
    if await card_path.exists():
        await card_path.unlink()
        _card_cache.pop(model_id, None)
        return True
    return False


class ConfigData(BaseModel):
    model_config = {"extra": "ignore"}  # Allow unknown fields

    architectures: list[str] | None = None
    hidden_size: Annotated[int, Field(ge=0)] | None = None
    num_key_value_heads: PositiveInt | None = None
    layer_count: int = Field(
        validation_alias=AliasChoices(
            "num_hidden_layers",
            "num_layers",
            "n_layer",
            "n_layers",
            "num_decoder_layers",
            "decoder_layers",
        )
    )
    max_position_embeddings: int | None = None
    vision: VisionCardConfig | None = None

    @property
    def supports_tensor(self) -> bool:
        return self.architectures in [
            ["Glm4MoeLiteForCausalLM"],
            ["GlmMoeDsaForCausalLM"],
            ["DeepseekV32ForCausalLM"],
            ["DeepseekV3ForCausalLM"],
            ["Qwen3NextForCausalLM"],
            ["Qwen3MoeForCausalLM"],
            ["Qwen3_5MoeForConditionalGeneration"],
            ["Qwen3_5ForConditionalGeneration"],
            ["MiniMaxM2ForCausalLM"],
            ["LlamaForCausalLM"],
            ["GptOssForCausalLM"],
            ["Step3p5ForCausalLM"],
            ["NemotronHForCausalLM"],
        ]

    @model_validator(mode="before")
    @classmethod
    def defer_to_text_config(cls, data: dict[str, Any], info: ValidationInfo):
        text_config = data.get("text_config")
        if text_config is not None:
            for field in [
                "architectures",
                "hidden_size",
                "num_key_value_heads",
                "num_hidden_layers",
                "num_layers",
                "n_layer",
                "n_layers",
                "num_decoder_layers",
                "decoder_layers",
                "max_position_embeddings",
            ]:
                if (val := text_config.get(field)) is not None:  # pyright: ignore[reportAny]
                    data[field] = val

        vision_config = data.get("vision_config")
        image_token_id = data.get("image_token_id")
        if vision_config is not None and image_token_id is not None:
            # Prefer top-level model_type (e.g. "gemma4") over
            # vision_config.model_type (e.g. "gemma4_vision") — the top-level
            # value matches the mlx_vlm.models.{model_type} import path.
            model_type = str(
                data.get("model_type", vision_config.get("model_type", ""))  # pyright: ignore[reportAny]
            )
            assert info.context is not None

            boi = data.get("boi_token_id")
            eoi = data.get("eoi_token_id")
            data["vision"] = VisionCardConfig(
                image_token_id=int(image_token_id),  # pyright: ignore[reportAny]
                model_type=model_type,
                weights_repo=info.context["model_id"],  # type: ignore
                boi_token_id=int(boi) if boi is not None else None,  # pyright: ignore[reportAny]
                eoi_token_id=int(eoi) if eoi is not None else None,  # pyright: ignore[reportAny]
            )

        return data


async def fetch_config_data(model_id: ModelId) -> ConfigData:
    """Downloads and parses config.json for a model."""
    from skulk.download.download_utils import (
        download_file_with_retry,
        ensure_models_dir,
    )

    target_dir = (await ensure_models_dir()) / model_id.normalize()
    await aios.makedirs(target_dir, exist_ok=True)
    config_path = await download_file_with_retry(
        model_id,
        "main",
        "config.json",
        target_dir,
        lambda curr_bytes, total_bytes, is_renamed: logger.debug(
            f"Downloading config.json for {model_id}: {curr_bytes}/{total_bytes} ({is_renamed=})"
        ),
    )
    async with aiofiles.open(config_path, "r") as f:
        return ConfigData.model_validate_json(
            await f.read(), context={"model_id": str(model_id)}
        )


async def fetch_safetensors_size(model_id: ModelId) -> Memory:
    """Gets model size from safetensors index or falls back to HF API."""
    from skulk.download.download_utils import (
        download_file_with_retry,
        ensure_models_dir,
    )
    from skulk.shared.types.worker.downloads import ModelSafetensorsIndex

    target_dir = (await ensure_models_dir()) / model_id.normalize()
    await aios.makedirs(target_dir, exist_ok=True)
    index_path = await download_file_with_retry(
        model_id,
        "main",
        "model.safetensors.index.json",
        target_dir,
        lambda curr_bytes, total_bytes, is_renamed: logger.debug(
            f"Downloading model.safetensors.index.json for {model_id}: {curr_bytes}/{total_bytes} ({is_renamed=})"
        ),
    )
    async with aiofiles.open(index_path, "r") as f:
        index_data = ModelSafetensorsIndex.model_validate_json(await f.read())

    metadata = index_data.metadata
    if metadata is not None:
        return Memory.from_bytes(metadata.total_size)

    info = model_info(model_id)
    if info.safetensors is None:
        raise ValueError(f"No safetensors info found for {model_id}")
    return Memory.from_bytes(info.safetensors.total)


def gguf_weight_siblings(model_id: ModelId) -> "list[tuple[str, int]]":
    """List a repo's GGUF weight files (filename, size) via the HF API.

    Excludes multimodal projector files (``mmproj*``), which are not the LM
    weights. Returns an empty list for a non-GGUF repo, so callers use this as a
    "is this a GGUF model?" probe. Sizes are bytes (0 when HF omits the size).
    """
    info = model_info(model_id, files_metadata=True)
    siblings = info.siblings or []
    return [
        (sibling.rfilename, sibling.size or 0)
        for sibling in siblings
        if sibling.rfilename.endswith(".gguf")
        and "mmproj" not in sibling.rfilename.lower()
    ]


# Preferred GGUF quantizations, best first. Q4_K_M is the de-facto default
# (good quality/size); unquantized (BF16/F16/F32) is deprioritized hard so a
# multi-quant repo never defaults to the full-precision weights (#334).
_GGUF_QUANT_PREFERENCE: Final[tuple[str, ...]] = (
    "q4_k_m",
    "q4_k_s",
    "q5_k_m",
    "q5_k_s",
    "q6_k",
    "q4_0",
    "q5_0",
    "q8_0",
    "q3_k_m",
    "q3_k_l",
    "q3_k_s",
    "q2_k",
    "iq4_xs",
    "iq4_nl",
    "iq4",
    "iq3",
    "iq2",
    "iq1",
)
_GGUF_UNQUANTIZED: Final[tuple[str, ...]] = ("bf16", "f16", "fp16", "f32", "fp32")
_GGUF_RANK_UNRECOGNIZED: Final = 1_000  # unknown quant: after known quants
_GGUF_RANK_UNQUANTIZED: Final = 10_000  # full precision: dead last


def gguf_quant_rank(name: str) -> int:
    """Preference rank for a GGUF filename (lower is better); see #334."""
    low = name.rsplit("/", 1)[-1].lower()
    for index, marker in enumerate(_GGUF_QUANT_PREFERENCE):
        if marker in low:
            return index
    if any(token in low for token in _GGUF_UNQUANTIZED):
        return _GGUF_RANK_UNQUANTIZED
    return _GGUF_RANK_UNRECOGNIZED


def _gguf_quant_label(name: str) -> str:
    """Human quant label parsed from a GGUF filename (e.g. ``Q4_K_M``), or ``""``."""
    low = name.rsplit("/", 1)[-1].lower()
    for marker in _GGUF_QUANT_PREFERENCE:
        if marker in low:
            return marker.upper()
    for token in _GGUF_UNQUANTIZED:
        if token in low:
            return token.upper()
    return ""


def select_preferred_gguf(gguf_files: "list[tuple[str, int]]") -> str:
    """Pick the GGUF weights file to load: best quant, then basename order.

    Prefers a real quantization (Q4_K_M first) over the unquantized BF16/F16 a
    multi-quant repo also ships (#334). The basename tie-break makes the choice
    deterministic and, for a shard group, picks the first shard. The runner's
    ``select_gguf_file`` falls back to this same ranking when the card does not
    pin a file, so download, sizing, and loading all agree.
    """
    return min(
        (name for name, _ in gguf_files),
        key=lambda name: (gguf_quant_rank(name), name.rsplit("/", 1)[-1]),
    )


def gguf_allow_patterns(gguf_file: str) -> list[str]:
    """Download allow-patterns for a card's selected GGUF (its shard group only).

    A single-file quant downloads just itself; a sharded quant downloads its
    whole ``<base>-NNNNN-of-NNNNN`` group via a glob. This is what lets the
    downloader fetch only the chosen quant instead of every quant a multi-quant
    repo hosts (#332). Caller adds non-weight files (e.g. ``config.json``).
    """
    base = _gguf_shard_base(gguf_file)
    if base is None:
        return [gguf_file]
    return [f"{base}-*-of-*.gguf"]


def gguf_shard_group_size(selected: str, gguf_files: "list[tuple[str, int]]") -> Memory:
    """Total bytes of ``selected`` plus its sibling shards (its shard group).

    A single-file quant sums to itself; a sharded quant sums every file sharing
    its ``<base>-NNNNN-of-NNNNN`` group. Keeps ``storage_size`` consistent with
    what actually loads, not the sum of every quant the repo hosts.
    """
    base = _gguf_shard_base(selected)
    if base is None:
        return Memory.from_bytes(dict(gguf_files)[selected])
    return Memory.from_bytes(
        sum(size for name, size in gguf_files if _gguf_shard_base(name) == base)
    )


def _gguf_shard_base(name: str) -> str | None:
    """Return the shard-group base of a GGUF filename, or ``None`` if unsharded.

    ``foo-00001-of-00003.gguf`` -> ``foo``; ``foo.gguf`` -> ``None``. Detects the
    ``-NNNNN-of-NNNNN.gguf`` suffix without a regex (re is not imported here).
    The base itself may contain dashes (e.g. ``Qwen2.5-7B-Instruct-Q4_K_M``), so
    we split only the trailing three ``-`` tokens.
    """
    stem = name[: -len(".gguf")] if name.endswith(".gguf") else name
    parts = stem.rsplit("-", 3)
    if (
        len(parts) == 4
        and parts[1].isdigit()
        and parts[2] == "of"
        and parts[3].isdigit()
    ):
        return parts[0]
    return None


# --- GGUF binary header parsing (#327) -------------------------------------
#
# A GGUF file begins with a metadata block (magic, version, tensor count, then
# a list of typed key/value pairs) followed by the tensor data. The structural
# fields a model card needs (layer count, hidden size, KV-head count, context
# length) live in that block under arch-prefixed keys, so they can be read from
# the start of the file via a ranged HTTP read instead of pulling the multi-GB
# weights. This is the fallback for GGUF repos that ship no ``config.json``.
#
# Spec: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

_GGUF_MAGIC: Final = b"GGUF"
# Lengths/counts are 64-bit in GGUF v2/v3; v1 used 32-bit and is obsolete.
_GGUF_SUPPORTED_VERSIONS: Final = frozenset({2, 3})

# GGUF metadata value type tags.
_GGUF_TYPE_STRING: Final = 8
_GGUF_TYPE_ARRAY: Final = 9
# Scalar value type tag -> struct format. ``bool`` (tag 7) is read as a byte;
# we deliberately do not store it as an int field (see the parse loop).
_GGUF_SCALAR_FMT: Final[dict[int, str]] = {
    0: "<B",  # uint8
    1: "<b",  # int8
    2: "<H",  # uint16
    3: "<h",  # int16
    4: "<I",  # uint32
    5: "<i",  # int32
    6: "<f",  # float32
    7: "<B",  # bool (stored as one byte)
    10: "<Q",  # uint64
    11: "<q",  # int64
    12: "<d",  # float64
}
_GGUF_SCALAR_SIZE: Final[dict[int, int]] = {
    tag: struct.calcsize(fmt) for tag, fmt in _GGUF_SCALAR_FMT.items()
}

# Structural keys we read, relative to ``general.architecture`` (e.g. for arch
# ``llama`` we read ``llama.block_count``). These mirror config.json's
# layer_count / hidden_size / num_key_value_heads / max_position_embeddings.
_GGUF_KEY_BLOCK_COUNT: Final = "block_count"
_GGUF_KEY_EMBEDDING_LENGTH: Final = "embedding_length"
_GGUF_KEY_HEAD_COUNT_KV: Final = "attention.head_count_kv"
_GGUF_KEY_CONTEXT_LENGTH: Final = "context_length"
_GGUF_WANTED_SUFFIXES: Final = (
    _GGUF_KEY_BLOCK_COUNT,
    _GGUF_KEY_EMBEDDING_LENGTH,
    _GGUF_KEY_HEAD_COUNT_KV,
    _GGUF_KEY_CONTEXT_LENGTH,
)
# The keys a card cannot be built without; head_count_kv and context_length are
# best-effort (default to None / 0). Used to decide the early stop at the start
# of the tokenizer section when a best-effort key is simply absent.
_GGUF_REQUIRED_SUFFIXES: Final = (
    _GGUF_KEY_BLOCK_COUNT,
    _GGUF_KEY_EMBEDDING_LENGTH,
)


class GgufStructuralFields(NamedTuple):
    """Structural model dimensions read from a GGUF metadata header (#327)."""

    n_layers: int
    hidden_size: int
    num_key_value_heads: int | None
    context_length: int


class _GgufHeaderReader:
    """Sequential cursor over a GGUF metadata header fetched on demand.

    Bytes are pulled through an injected ``fetch(offset, length)`` coroutine
    (HTTP range in production, an in-memory blob in tests) into a growing buffer
    and cached, so the header is read in a few windows rather than one giant
    request. The cursor advances as typed values are read; large arrays (e.g. a
    tokenizer vocabulary) are skipped by advancing the cursor, fetching more
    only if a wanted key happens to sit past them.
    """

    _WINDOW: Final = 1 << 20  # 1 MiB per fetch; structural keys sit well inside.

    def __init__(self, fetch: "Callable[[int, int], Awaitable[bytes]]") -> None:
        self._fetch = fetch
        self._buf = bytearray()
        self._eof = False
        self.pos = 0

    async def _ensure(self, end: int) -> None:
        """Fetch until the buffer holds at least ``end`` bytes (or EOF)."""
        while len(self._buf) < end and not self._eof:
            start = len(self._buf)
            # Cap each fetch at one window and loop, so a large forward jump (a
            # bulk array skip past ``end``) never balloons into a single huge
            # range request; the loop keeps memory/request size window-bounded.
            chunk = await self._fetch(start, self._WINDOW)
            # Only an empty read is EOF; a short read is a partial transport
            # chunk, so keep fetching from the new offset rather than stopping.
            if not chunk:
                self._eof = True
                return
            self._buf.extend(chunk)

    async def _take(self, n: int) -> bytes:
        await self._ensure(self.pos + n)
        if self.pos + n > len(self._buf):
            raise ValueError("GGUF header truncated before the wanted metadata")
        data = bytes(self._buf[self.pos : self.pos + n])
        self.pos += n
        return data

    async def _u32(self) -> int:
        return cast(int, struct.unpack("<I", await self._take(4))[0])

    async def _u64(self) -> int:
        return cast(int, struct.unpack("<Q", await self._take(8))[0])

    async def read_string(self) -> str:
        length = await self._u64()
        return (await self._take(length)).decode("utf-8", errors="replace")

    async def read_value(self, value_type: int) -> "int | str | None":
        """Read one metadata value; return ints/strings, skip everything else.

        Returns the value for integer and string scalars (the only kinds the
        wanted keys use), and ``None`` for floats, bools, and arrays after
        advancing past them.
        """
        if value_type == _GGUF_TYPE_STRING:
            return await self.read_string()
        if value_type == _GGUF_TYPE_ARRAY:
            await self._skip_array()
            return None
        fmt = _GGUF_SCALAR_FMT.get(value_type)
        if fmt is None:
            raise ValueError(f"unknown GGUF metadata value type {value_type}")
        raw = await self._take(_GGUF_SCALAR_SIZE[value_type])
        # Floats and bools are not structural fields we read; only surface ints.
        if value_type in (6, 12, 7):  # float32, float64, bool
            return None
        return cast(int, struct.unpack(fmt, raw)[0])

    async def _skip_array(self) -> None:
        element_type = await self._u32()
        count = await self._u64()
        if element_type in _GGUF_SCALAR_SIZE:
            # Bulk-skip fixed-width elements (e.g. a token-score float array).
            self.pos += _GGUF_SCALAR_SIZE[element_type] * count
        elif element_type == _GGUF_TYPE_STRING:
            for _ in range(count):
                _ = await self.read_string()
        elif element_type == _GGUF_TYPE_ARRAY:
            for _ in range(count):
                await self._skip_array()
        else:
            raise ValueError(f"unknown GGUF array element type {element_type}")

    async def parse_structural_fields(self) -> GgufStructuralFields:
        """Read the structural model dimensions from this GGUF header.

        Reads ``general.architecture`` and the arch-prefixed ``block_count`` /
        ``embedding_length`` / ``attention.head_count_kv`` / ``context_length``
        keys. Stops as soon as all are seen, or, if a best-effort key is absent,
        at the start of the tokenizer section (the arch scalars always precede
        it), so a trailing multi-megabyte tokenizer array is never fetched.

        Raises:
            ValueError: The bytes are not a supported GGUF header, or the header
                lacks the architecture / layer / hidden-size keys needed to
                build a card.
        """
        if (await self._take(4)) != _GGUF_MAGIC:
            raise ValueError("not a GGUF file (bad magic)")
        version = await self._u32()
        if version not in _GGUF_SUPPORTED_VERSIONS:
            raise ValueError(f"unsupported GGUF version {version}")
        _ = await self._u64()  # tensor_count
        kv_count = await self._u64()

        architecture: str | None = None
        collected: dict[str, int] = {}

        def _has(suffixes: "tuple[str, ...]") -> bool:
            return architecture is not None and all(
                f"{architecture}.{suffix}" in collected for suffix in suffixes
            )

        for _ in range(kv_count):
            key = await self.read_string()
            # The arch-prefixed scalar keys precede the tokenizer section; once a
            # tokenizer key appears no more arch scalars follow, so if the
            # required fields are known, stop before reading a potentially huge
            # tokenizer array for best-effort fields that are simply absent.
            if key.startswith("tokenizer.") and _has(_GGUF_REQUIRED_SUFFIXES):
                break
            value = await self.read_value(await self._u32())
            if key == "general.architecture" and isinstance(value, str):
                architecture = value
            elif isinstance(value, int):
                collected[key] = value
            # Fast path: every wanted key (including best-effort ones) is in,
            # before any tokenizer array.
            if _has(_GGUF_WANTED_SUFFIXES):
                break

        if architecture is None:
            raise ValueError("GGUF header missing general.architecture")

        def field(suffix: str) -> int | None:
            return collected.get(f"{architecture}.{suffix}")

        n_layers = field(_GGUF_KEY_BLOCK_COUNT)
        hidden_size = field(_GGUF_KEY_EMBEDDING_LENGTH)
        if not n_layers or not hidden_size:
            raise ValueError(
                f"GGUF header for arch {architecture!r} is missing block_count / "
                "embedding_length"
            )
        return GgufStructuralFields(
            n_layers=n_layers,
            hidden_size=hidden_size,
            num_key_value_heads=field(_GGUF_KEY_HEAD_COUNT_KV) or None,
            context_length=field(_GGUF_KEY_CONTEXT_LENGTH) or 0,
        )


async def read_gguf_structural_fields(
    fetch: "Callable[[int, int], Awaitable[bytes]]",
) -> GgufStructuralFields:
    """Parse structural model dimensions from a GGUF metadata header.

    ``fetch(offset, length)`` supplies bytes from the start of the GGUF file
    (an HTTP range read in production, an in-memory blob in tests). See
    :meth:`_GgufHeaderReader.parse_structural_fields` for what is read.

    Raises:
        ValueError: The bytes are not a supported GGUF header, or the header
            lacks the architecture / layer / hidden-size keys needed to build a
            card.
    """
    return await _GgufHeaderReader(fetch).parse_structural_fields()
