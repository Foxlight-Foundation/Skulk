"""Tests for the Mach page-category snapshot powering placement availability.

The GPU-wireable availability fix: mactop's raw ``available`` counts
reclaimable file cache as used, so placement refused fits that run comfortably
(a just-downloaded model's weights sitting in file cache deflate availability
by the model's full size). These tests pin the ``vm_stat`` parser and the
``total − wired − anonymous − compressor`` formula against a real captured
sample.
"""

from skulk.shared.types.profiling import (
    MachMemoryCategories,
    gpu_wireable_memory_bytes,
    parse_vm_stat_output,
)

# Captured verbatim from a 24 GB M4 (kite3) on 2026-06-10, minutes after the
# placement over-refusal this fix addresses: mactop reported ~12 GB available
# while 11.6 GB of just-downloaded gpt-oss weights sat in reclaimable file
# cache. Page size 16384 bytes.
_VM_STAT_SAMPLE = """\
Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                              161895.
Pages active:                            624486.
Pages inactive:                          379346.
Pages speculative:                       244416.
Pages throttled:                              0.
Pages wired down:                        127991.
Pages purgeable:                          23214.
"Translation faults":                   8237502.
Pages copy-on-write:                     497486.
Pages zero filled:                      2892268.
Pages reactivated:                         1268.
Pages purged:                              4072.
File-backed pages:                       758109.
Anonymous pages:                         490139.
Pages stored in compressor:                   0.
Pages occupied by compressor:                 0.
Decompressions:                               0.
Compressions:                                 0.
Pageins:                                 631600.
Pageouts:                                     0.
Swapins:                                      0.
Swapouts:                                     0.
"""

_PAGE = 16384
_KITE3_TOTAL = 25_769_803_776  # 24 GB


def test_parses_real_vm_stat_sample():
    categories = parse_vm_stat_output(_VM_STAT_SAMPLE)
    assert categories is not None
    assert categories.wired_bytes == 127991 * _PAGE
    assert categories.anonymous_bytes == 490139 * _PAGE
    assert categories.compressor_bytes == 0


def test_gpu_wireable_matches_incident_arithmetic():
    # 24 − 1.95 (wired) − 7.48 (anonymous) ≈ 14.6 GB genuinely wireable, vs
    # the ~12 GB cache-deflated figure mactop gossiped during the incident.
    categories = parse_vm_stat_output(_VM_STAT_SAMPLE)
    assert categories is not None
    wireable = gpu_wireable_memory_bytes(_KITE3_TOTAL, categories)
    assert wireable == _KITE3_TOTAL - (127991 + 490139) * _PAGE
    assert 14.0 < wireable / 2**30 < 15.0


def test_compressor_occupancy_reduces_wireable():
    # A box under memory pressure holds gigabytes in the compressor; those
    # pages are resident and must not be credited as wireable.
    categories = MachMemoryCategories(
        wired_bytes=2 * 2**30,
        anonymous_bytes=6 * 2**30,
        compressor_bytes=4 * 2**30,
    )
    assert gpu_wireable_memory_bytes(16 * 2**30, categories) == 4 * 2**30


def test_wireable_clamps_at_zero():
    categories = MachMemoryCategories(
        wired_bytes=10 * 2**30,
        anonymous_bytes=10 * 2**30,
        compressor_bytes=0,
    )
    assert gpu_wireable_memory_bytes(16 * 2**30, categories) == 0


def test_missing_page_size_header_returns_none():
    assert parse_vm_stat_output("Pages wired down: 1.\n") is None


def test_missing_counter_returns_none():
    # Format drift (a renamed counter) must degrade to "no snapshot", which
    # callers turn into the raw-mactop fallback — never a wrong number.
    truncated = _VM_STAT_SAMPLE.replace("Anonymous pages", "Renamed pages")
    assert parse_vm_stat_output(truncated) is None


def test_empty_output_returns_none():
    assert parse_vm_stat_output("") is None
