# pyright: reportPrivateUsage=false
"""Skulk's readers of mlx-lm's private BatchGenerator counters, on the real class.

The other generator tests drive fakes, and the end-to-end batch test is marked
slow. When mlx-lm 0.32.2 moved these counters into ``_counters``, every
default test passed while every MLX text generation on hardware failed at its
first request. These run the readers against a genuine upstream instance in
the default suite, so the next rename fails here first.
"""

import mlx.nn as nn
from mlx_lm.generate import BatchGenerator

from skulk.worker.engines.mlx.generator.batch_generate import (
    upstream_decode_steps,
    upstream_prompt_seconds,
)


def test_counters_read_from_a_real_upstream_generator() -> None:
    # Construction only records the model, so an empty module is enough.
    generator = BatchGenerator(model=nn.Module(), stop_tokens=[[0]])
    try:
        assert upstream_prompt_seconds(generator) == 0.0
        assert upstream_decode_steps(generator) == 0
    finally:
        generator.close()


def test_readers_track_the_record_upstream_stats_reports() -> None:
    """The readers follow the same record upstream's own ``stats()`` reads."""
    generator = BatchGenerator(model=nn.Module(), stop_tokens=[[0]])
    try:
        with generator.stats() as stats:
            generator._counters.prompt_time += 1.5
            generator._counters.generation_steps += 3
        assert stats.prompt_time == 1.5
        assert upstream_prompt_seconds(generator) == 1.5
        assert upstream_decode_steps(generator) == 3
    finally:
        generator.close()
