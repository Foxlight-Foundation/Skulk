"""The exact load an engine package is asked for, for one placement.

The runner states these settings outright when it loads, and the worker asks
the adapter what the same load needs before it dispatches it. Both build the
request here, so the figure the engine reports is for the load it then
receives.
"""

from typing import Final

from skulk.download.download_utils import build_model_path
from skulk.engines.protocol import LoadSettings, ModelIdentity
from skulk.shared.types.common import ModelId
from skulk.shared.types.worker.shards import ShardMetadata
from skulk.worker.runner.llama_cpp.runner import serving_n_ctx

PARALLEL_STREAMS: Final = 1
"""Streams a packaged engine is loaded for. One until a placement carries the
admitted stream count; the engine is told this number and may not exceed it."""

DEFAULT_CACHE_PRECISION: Final = "default"
"""Prompt-cache precision stated at load until a card carries one."""


def package_load_request(
    shard: ShardMetadata, context_token_limit: int | None
) -> tuple[ModelIdentity, LoadSettings]:
    """Return the model and the admitted settings for a placement's load.

    Args:
        shard: The placement's shard on this node; its card names the model.
        context_token_limit: The instance's stamped serving window, or ``None``
            when placement stamped none.

    Returns:
        The model's identity with its directory on this node, and the
        settings the engine must be launched with.
    """
    card = shard.model_card
    directory = build_model_path(
        ModelId(card.model_id),
        card.source_revision,
        card.artifact_bundle.root if card.artifact_bundle is not None else None,
    )
    return (
        ModelIdentity(
            model_id=str(card.model_id),
            source_revision=card.source_revision,
            directory=str(directory.resolve()),
        ),
        LoadSettings(
            context_tokens=serving_n_ctx(context_token_limit, logits_all=False),
            parallel_streams=PARALLEL_STREAMS,
            cache_precision=DEFAULT_CACHE_PRECISION,
            drafting=False,
        ),
    )
