"""The cap on MLX's freed-buffer cache a runner applies after loading a model."""

import pytest

from skulk.shared.types.memory import Memory
from skulk.worker.engines.mlx.utils_mlx import mlx_buffer_cache_limit

# max_recommended_working_set_size on a 24 GB M4 Mac mini.
DESKTOP_24_GB = Memory.from_bytes(19_069_665_280)


def test_the_default_keeps_a_tenth_of_the_recommended_working_set() -> None:
    limit = mlx_buffer_cache_limit(DESKTOP_24_GB, None)

    assert limit == DESKTOP_24_GB * 0.10
    # The measured problem: 9.5 GB of cache beside 7.7 GB active on this Mac.
    assert limit < Memory.from_gb(2)


def test_the_default_never_drops_below_512_mib() -> None:
    assert mlx_buffer_cache_limit(Memory.from_gb(4), None) == Memory.from_mb(512)


@pytest.mark.parametrize("blank", ["", "   "])
def test_a_blank_override_means_the_default(blank: str) -> None:
    assert mlx_buffer_cache_limit(DESKTOP_24_GB, blank) == mlx_buffer_cache_limit(
        DESKTOP_24_GB, None
    )


@pytest.mark.parametrize(
    ("override", "expected"),
    [("0", Memory.from_bytes(0)), ("2048", Memory.from_gb(2)), (" 768 ", Memory.from_mb(768))],
)
def test_an_override_sets_the_limit_in_mib(override: str, expected: Memory) -> None:
    assert mlx_buffer_cache_limit(DESKTOP_24_GB, override) == expected


@pytest.mark.parametrize("invalid", ["-1", "1.5", "2GB", "lots"])
def test_an_invalid_override_is_refused(invalid: str) -> None:
    with pytest.raises(ValueError):
        mlx_buffer_cache_limit(DESKTOP_24_GB, invalid)


@pytest.mark.parametrize(
    ("memory", "shown"),
    [
        (Memory.from_mb(1819), "1819 MiB"),
        (DESKTOP_24_GB, "17.76 GiB"),
        (Memory.from_mb(512), "512 MiB"),
        (Memory.from_bytes(1), "1 B"),
    ],
)
def test_logged_sizes_name_their_unit_once(memory: Memory, shown: str) -> None:
    # The limit is logged as str(Memory), which used to read "1819.00 MiB MiB".
    assert str(memory) == shown
