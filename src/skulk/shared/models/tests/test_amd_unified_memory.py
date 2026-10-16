"""The shared AMD unified-memory classifier: kernel verdict first, sizes last.

Invariants:

1. A collector's kernel-derived verdict decides. A unified GPU's limit is the
   TTM page limit it reports, whatever GTT, VRAM and RAM sizes say. A GPU the
   kernel reports as discrete is never unified, even when the old size rule
   would pass.
2. Only when the verdict is unknown does the old rule apply: GTT above the
   carve and covering all of RAM.
3. Non-AMD devices are never classified here (GB10 has its own pool).
"""

import pytest

from skulk.shared.models.memory_estimate import amd_unified_memory_limit
from skulk.shared.types.profiling import AcceleratorMetrics, MemoryUsage

# kite5 as read on 2026-10-16: AMD's recommended 512 MB carve, TTM 60 GiB, and
# RAM 60.9 GiB, so GTT/TTM falls just short of RAM.
_KITE5_RAM_TOTAL = 65_430_781_952
_KITE5_TTM = 64_424_509_440
_KITE5_VRAM = 536_870_912


def _memory(ram_total: int = _KITE5_RAM_TOTAL) -> MemoryUsage:
    return MemoryUsage.from_bytes(
        ram_total=ram_total,
        ram_available=62_244_913_152,
        swap_total=0,
        swap_available=0,
    )


def _amd(**fields: object) -> AcceleratorMetrics:
    values: dict[str, object] = {
        "vendor": "amd",
        "vram_total_bytes": _KITE5_VRAM,
        "gtt_total_bytes": _KITE5_TTM,
    }
    values.update(fields)
    return AcceleratorMetrics.model_validate(values)


def test_kernel_verdict_classifies_the_recommended_carve_apu() -> None:
    """kite5 is unified even though its TTM/GTT sits below RAM (invariant 1)."""
    accelerator = _amd(unified_memory=True, addressable_memory_bytes=_KITE5_TTM)
    assert amd_unified_memory_limit(accelerator, _memory()) == _KITE5_TTM


def test_unified_limit_falls_back_to_gtt_when_ttm_unread() -> None:
    accelerator = _amd(unified_memory=True, addressable_memory_bytes=None)
    assert amd_unified_memory_limit(accelerator, _memory()) == _KITE5_TTM


def test_unified_verdict_needs_no_memory_reading() -> None:
    """The kernel verdict stands alone; callers decide what to do without RAM."""
    accelerator = _amd(unified_memory=True, addressable_memory_bytes=_KITE5_TTM)
    assert amd_unified_memory_limit(accelerator, None) == _KITE5_TTM


def test_discrete_verdict_wins_over_the_size_rule() -> None:
    """A kernel-reported discrete card stays discrete (invariant 1)."""
    accelerator = _amd(
        vram_total_bytes=25_769_803_776,
        gtt_total_bytes=200_000_000_000,
        unified_memory=False,
    )
    assert amd_unified_memory_limit(accelerator, _memory()) is None


@pytest.mark.parametrize(
    ("gtt", "ram_total", "expected"),
    [
        # GTT covers RAM: the old rule admits it.
        (133_143_986_176, 131_732_426_752, 133_143_986_176),
        # kite5's numbers: the old rule misses it, which is why the kernel
        # verdict must come first.
        (_KITE5_TTM, _KITE5_RAM_TOTAL, None),
        # GTT no larger than the carve: discrete.
        (_KITE5_VRAM, _KITE5_RAM_TOTAL, None),
    ],
)
def test_unknown_verdict_keeps_the_size_rule(
    gtt: int, ram_total: int, expected: int | None
) -> None:
    """Invariant 2: only an unknown verdict falls back to sizes."""
    accelerator = _amd(gtt_total_bytes=gtt, unified_memory=None)
    assert amd_unified_memory_limit(accelerator, _memory(ram_total)) == expected


def test_unknown_verdict_without_memory_is_not_unified() -> None:
    accelerator = _amd(gtt_total_bytes=133_143_986_176, unified_memory=None)
    assert amd_unified_memory_limit(accelerator, None) is None


def test_non_amd_devices_are_never_classified_here() -> None:
    """Invariant 3."""
    nvidia = AcceleratorMetrics(
        vendor="nvidia",
        name="NVIDIA GB10",
        compute_capability="12.1",
        vram_total_bytes=128 * 2**30,
        unified_memory=True,
        addressable_memory_bytes=_KITE5_TTM,
    )
    assert amd_unified_memory_limit(nvidia, _memory()) is None
