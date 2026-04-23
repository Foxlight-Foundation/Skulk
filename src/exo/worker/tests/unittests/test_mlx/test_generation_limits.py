import os

from exo.worker.engines.mlx.constants import DEFAULT_MAX_OUTPUT_TOKENS, MAX_TOKENS


def test_default_max_output_tokens_is_a_conservative_generation_budget() -> None:
    """Default generation budget should not mirror long model context windows."""
    assert DEFAULT_MAX_OUTPUT_TOKENS == 2048

    if (
        "SKULK_MAX_OUTPUT_TOKENS" not in os.environ
        and "EXO_MAX_TOKENS" not in os.environ
    ):
        assert MAX_TOKENS == DEFAULT_MAX_OUTPUT_TOKENS
