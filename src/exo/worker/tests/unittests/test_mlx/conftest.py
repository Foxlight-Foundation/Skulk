# type: ignore
import json
import os
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn

from exo.shared.constants import EXO_MODELS_DIR
from exo.shared.models.model_cards import ModelCard, ModelTask
from exo.shared.types.common import ModelId
from exo.shared.types.memory import Memory
from exo.shared.types.mlx import Model
from exo.shared.types.text_generation import InputMessage, TextGenerationTaskParams
from exo.shared.types.worker.shards import PipelineShardMetadata, TensorShardMetadata
from exo.worker.engines.mlx.generator.generate import mlx_generate
from exo.worker.engines.mlx.utils_mlx import apply_chat_template, shard_and_load


class MockLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.custom_attr = "test_value"
        self.use_sliding = True

    def __call__(self, x: mx.array, *args: object, **kwargs: object) -> mx.array:
        return x * 2


@dataclass(frozen=True)
class PipelineTestConfig:
    model_path: Path
    total_layers: int
    base_port: int
    max_tokens: int


def create_hostfile(world_size: int, base_port: int) -> tuple[str, list[str]]:
    hosts = [f"127.0.0.1:{base_port + i}" for i in range(world_size)]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(hosts, f)
        hostfile_path = f.name

    return hostfile_path, hosts


@dataclass(frozen=True)
class DistributedTestModel:
    """A model usable by the distributed/prefix-cache tests.

    Pipeline layer splits in test_distributed_fix.py are derived from
    ``total_layers``, so entries may have different layer counts. Any
    substitute must use an architecture supported by
    ``tensor_auto_parallel`` (see auto_parallel.py's model allowlist).
    """

    model_id: str
    directory_name: str
    total_layers: int
    hidden_size: int
    storage: Memory


# GPT-OSS-20B is the preferred test model because it has a lot of strange
# behaviour (MoE routing, MXFP4 quantization, alternating sliding-window
# attention) that historically shook out distributed bugs dense models missed.
# But it needs ~12 GB of wired GPU memory across multi-rank subprocesses, which
# memory-exhausts (and has hard-rebooted) 16 GB machines. Llama-3.2-1B
# exercises the same pipeline/tensor-parallel/prefix-cache infrastructure at
# ~0.7 GB (llama is on the tensor_auto_parallel allowlist; most tiny models,
# e.g. qwen2, are not).
_GPT_OSS_20B = DistributedTestModel(
    model_id="mlx-community/gpt-oss-20b-MXFP4-Q8",
    directory_name="mlx-community--gpt-oss-20b-MXFP4-Q8",
    total_layers=24,
    hidden_size=2880,
    storage=Memory.from_gb(12),
)
_LLAMA_3_2_1B = DistributedTestModel(
    model_id="mlx-community/Llama-3.2-1B-Instruct-4bit",
    directory_name="mlx-community--Llama-3.2-1B-Instruct-4bit",
    total_layers=16,
    hidden_size=2048,
    storage=Memory.from_gb(1),
)

_TEST_MODELS_BY_NAME: dict[str, DistributedTestModel] = {
    "gpt-oss-20b": _GPT_OSS_20B,
    "llama-3.2-1b": _LLAMA_3_2_1B,
}

# The 20B needs its 12 GB plus activations and multi-rank overhead resident at
# once; below a 24 GB Metal working set that ends in SIGABRT (or an OS reboot),
# not a test failure.
_GPT_OSS_MIN_WORKING_SET_BYTES = 24 * 1024**3


def _select_distributed_test_model() -> DistributedTestModel:
    """Pick the distributed-test model for this machine.

    ``SKULK_TEST_DISTRIBUTED_MODEL`` (a key of ``_TEST_MODELS_BY_NAME``)
    forces a specific model; otherwise GPT-OSS-20B is used only when the
    Metal working-set budget can actually hold it.
    """
    override = os.environ.get("SKULK_TEST_DISTRIBUTED_MODEL")
    if override is not None:
        if override not in _TEST_MODELS_BY_NAME:
            raise ValueError(
                f"SKULK_TEST_DISTRIBUTED_MODEL={override!r} is not one of "
                f"{sorted(_TEST_MODELS_BY_NAME)}"
            )
        return _TEST_MODELS_BY_NAME[override]
    device_info = mx.device_info()
    working_set = int(device_info["max_recommended_working_set_size"])
    if working_set >= _GPT_OSS_MIN_WORKING_SET_BYTES:
        return _GPT_OSS_20B
    return _LLAMA_3_2_1B


DISTRIBUTED_TEST_MODEL = _select_distributed_test_model()

DISTRIBUTED_TEST_CONFIG = PipelineTestConfig(
    model_path=EXO_MODELS_DIR / DISTRIBUTED_TEST_MODEL.directory_name,
    total_layers=DISTRIBUTED_TEST_MODEL.total_layers,
    base_port=29600,
    max_tokens=200,
)

DISTRIBUTED_TEST_MODEL_ID = DISTRIBUTED_TEST_MODEL.model_id


def run_distributed_pipeline_device(
    rank: int,
    world_size: int,
    hostfile_path: str,
    layer_splits: list[tuple[int, int]],
    prompt_tokens: int,
    prefill_step_size: int,
    result_queue: Any,  # pyright: ignore[reportAny]
    max_tokens: int = 200,
) -> None:
    os.environ["MLX_HOSTFILE"] = hostfile_path
    os.environ["MLX_RANK"] = str(rank)

    try:
        group = mx.distributed.init(backend="ring", strict=True)

        start_layer, end_layer = layer_splits[rank]

        shard_meta = PipelineShardMetadata(
            model_card=ModelCard(
                model_id=ModelId(DISTRIBUTED_TEST_MODEL_ID),
                storage_size=DISTRIBUTED_TEST_MODEL.storage,
                n_layers=DISTRIBUTED_TEST_MODEL.total_layers,
                hidden_size=DISTRIBUTED_TEST_MODEL.hidden_size,
                supports_tensor=False,
                tasks=[ModelTask.TextGeneration],
            ),
            device_rank=rank,
            world_size=world_size,
            start_layer=start_layer,
            end_layer=end_layer,
            n_layers=DISTRIBUTED_TEST_MODEL.total_layers,
        )

        model, tokenizer = shard_and_load(
            shard_meta, group, on_timeout=None, on_layer_loaded=None
        )
        model = cast(Model, model)

        # Generate a prompt of exact token length
        base_text = "The quick brown fox jumps over the lazy dog. "
        base_tokens = tokenizer.encode(base_text)
        base_len = len(base_tokens)

        # Build prompt with approximate target length
        repeats = (prompt_tokens // base_len) + 2
        long_text = base_text * repeats
        tokens = tokenizer.encode(long_text)
        # Truncate to exact target length
        tokens = tokens[:prompt_tokens]
        prompt_text = tokenizer.decode(tokens)

        task = TextGenerationTaskParams(
            model=DISTRIBUTED_TEST_MODEL_ID,
            input=[InputMessage(role="user", content=prompt_text)],
            max_output_tokens=max_tokens,
        )

        prompt = apply_chat_template(tokenizer, task)

        generated_text = ""

        for response in mlx_generate(
            model=model,
            tokenizer=tokenizer,
            task=task,
            prompt=prompt,
            kv_prefix_cache=None,
            group=group,
        ):
            generated_text += response.text
            if response.finish_reason is not None:
                break

        result_queue.put((rank, True, generated_text))  # pyright: ignore[reportAny]

    except Exception as e:
        result_queue.put((rank, False, f"{e}\n{traceback.format_exc()}"))  # pyright: ignore[reportAny]


def run_distributed_tensor_parallel_device(
    rank: int,
    world_size: int,
    hostfile_path: str,
    prompt_tokens: int,
    prefill_step_size: int,
    result_queue: Any,  # pyright: ignore[reportAny]
    max_tokens: int = 10,
) -> None:
    os.environ["MLX_HOSTFILE"] = hostfile_path
    os.environ["MLX_RANK"] = str(rank)

    try:
        group = mx.distributed.init(backend="ring", strict=True)

        # For tensor parallelism, all devices run all layers
        shard_meta = TensorShardMetadata(
            model_card=ModelCard(
                model_id=ModelId(DISTRIBUTED_TEST_MODEL_ID),
                storage_size=DISTRIBUTED_TEST_MODEL.storage,
                n_layers=DISTRIBUTED_TEST_MODEL.total_layers,
                hidden_size=DISTRIBUTED_TEST_MODEL.hidden_size,
                supports_tensor=True,
                tasks=[ModelTask.TextGeneration],
            ),
            device_rank=rank,
            world_size=world_size,
            start_layer=0,
            end_layer=DISTRIBUTED_TEST_MODEL.total_layers,
            n_layers=DISTRIBUTED_TEST_MODEL.total_layers,
        )

        model, tokenizer = shard_and_load(
            shard_meta, group, on_timeout=None, on_layer_loaded=None
        )
        model = cast(Model, model)

        base_text = "The quick brown fox jumps over the lazy dog. "
        base_tokens = tokenizer.encode(base_text)
        base_len = len(base_tokens)

        repeats = (prompt_tokens // base_len) + 2
        long_text = base_text * repeats
        tokens = tokenizer.encode(long_text)
        tokens = tokens[:prompt_tokens]
        prompt_text = tokenizer.decode(tokens)

        task = TextGenerationTaskParams(
            model=DISTRIBUTED_TEST_MODEL_ID,
            input=[InputMessage(role="user", content=prompt_text)],
            max_output_tokens=max_tokens,
        )

        prompt = apply_chat_template(tokenizer, task)

        generated_text = ""
        for response in mlx_generate(
            model=model,
            tokenizer=tokenizer,
            task=task,
            prompt=prompt,
            kv_prefix_cache=None,
            group=group,
        ):
            generated_text += response.text
            if response.finish_reason is not None:
                break

        result_queue.put((rank, True, generated_text))  # pyright: ignore[reportAny]

    except Exception as e:
        result_queue.put((rank, False, f"{e}\n{traceback.format_exc()}"))  # pyright: ignore[reportAny]
