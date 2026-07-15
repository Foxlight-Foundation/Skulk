from skulk.shared.types.worker.shards import (
    CfgShardMetadata,
    PipelineShardMetadata,
    ShardMetadata,
)


def is_primary_image_output_node(shard_metadata: ShardMetadata) -> bool:
    """Return whether an image shard owns the generated image output.

    CFG placements produce output from the last positive-prompt pipeline stage;
    ordinary pipeline placements produce output from their last stage.
    """

    if isinstance(shard_metadata, CfgShardMetadata):
        is_pipeline_last = (
            shard_metadata.pipeline_rank == shard_metadata.pipeline_world_size - 1
        )
        return is_pipeline_last and shard_metadata.cfg_rank == 0
    if isinstance(shard_metadata, PipelineShardMetadata):
        return shard_metadata.device_rank == shard_metadata.world_size - 1
    return False
