"""Tests for the shared llama-engine generation statistics helpers (#532).

The llama.cpp engines previously emitted terminal chunks with ``stats=None``,
leaving dashboards, field telemetry, and the results ledger blind to token
throughput on GPU/Linux nodes. These cover the pure measurement helpers; the
runner wiring is exercised live on a GPU node.
"""

from skulk.worker.runner.generation_stats import (
    StreamStatsClock,
    blocking_call_stats,
    parse_vm_hwm,
    process_peak_memory,
    stats_from_llama_server_timings,
    subprocess_peak_memory,
)


def _fake_clock(moments: list[float]) -> StreamStatsClock:
    """Clock fed by a scripted sequence of perf-counter readings."""
    iterator = iter(moments)
    return StreamStatsClock(now=lambda: next(iterator))


def test_stream_clock_splits_prefill_and_decode_phases() -> None:
    # start=0.0; first piece at 2.0 (prefill 2s); last piece at 4.0 (decode 2s).
    clock = _fake_clock([0.0, 2.0, 3.0, 4.0])
    clock.mark_piece()
    clock.mark_piece()
    clock.mark_piece()

    stats = clock.stats(prompt_tokens=100, generation_tokens=clock.pieces)

    assert stats.prompt_tokens == 100
    assert stats.generation_tokens == 3
    assert stats.prompt_tps == 50.0  # 100 tokens / 2s prefill
    assert stats.generation_tps == 1.5  # 3 tokens / 2s decode span


def test_stream_clock_zero_spans_report_zero_not_fabricated_rates() -> None:
    # No pieces at all: both spans are empty; rates must be 0.0, not inf/nan.
    clock = _fake_clock([0.0, 5.0])
    stats = clock.stats(prompt_tokens=0, generation_tokens=0)
    assert stats.prompt_tps == 0.0
    assert stats.generation_tps == 0.0
    assert stats.generation_tokens == 0


def test_stream_clock_single_piece_has_no_decode_span() -> None:
    clock = _fake_clock([0.0, 1.0])
    clock.mark_piece()
    stats = clock.stats(prompt_tokens=10, generation_tokens=1)
    assert stats.prompt_tps == 10.0
    assert stats.generation_tps == 0.0  # one piece = zero decode span


def test_llama_server_timings_map_to_engine_exact_stats() -> None:
    stats = stats_from_llama_server_timings(
        {
            "prompt_n": 50,
            "prompt_ms": 250.0,
            "predicted_n": 128,
            "predicted_ms": 4000.0,
        }
    )
    assert stats is not None
    assert stats.prompt_tokens == 50
    assert stats.generation_tokens == 128
    assert stats.prompt_tps == 200.0  # 50 / 0.25s
    assert stats.generation_tps == 32.0  # 128 / 4s


def test_llama_server_timings_reject_malformed_shapes() -> None:
    # Missing counts, or boolean/string values, must fall back (None), not
    # produce a half-real stats object.
    assert stats_from_llama_server_timings({}) is None
    assert stats_from_llama_server_timings({"prompt_n": 50}) is None
    assert (
        stats_from_llama_server_timings({"prompt_n": True, "predicted_n": 5}) is None
    )
    assert (
        stats_from_llama_server_timings({"prompt_n": "50", "predicted_n": 5}) is None
    )


def test_llama_server_timings_zero_ms_reports_zero_rate() -> None:
    stats = stats_from_llama_server_timings(
        {"prompt_n": 50, "prompt_ms": 0, "predicted_n": 5, "predicted_ms": 0}
    )
    assert stats is not None
    assert stats.prompt_tps == 0.0
    assert stats.generation_tps == 0.0


def test_blocking_call_stats_uses_exact_counts_and_wall_rates() -> None:
    stats = blocking_call_stats(
        {"prompt_tokens": 80, "completion_tokens": 40}, wall_seconds=4.0
    )
    assert stats is not None
    assert stats.prompt_tokens == 80
    assert stats.generation_tokens == 40
    assert stats.prompt_tps == 20.0
    assert stats.generation_tps == 10.0


def test_blocking_call_stats_rejects_missing_or_non_dict_usage() -> None:
    assert blocking_call_stats(None, 1.0) is None
    assert blocking_call_stats({"prompt_tokens": 80}, 1.0) is None
    assert blocking_call_stats("usage", 1.0) is None


def test_process_peak_memory_reports_a_positive_reading() -> None:
    # The exact value is process-dependent; it just must be a real measurement.
    assert process_peak_memory().in_bytes > 0


def test_parse_vm_hwm_reads_peak_rss_in_kilobytes() -> None:
    status = "Name:\tllama-server\nVmPeak:\t  999 kB\nVmHWM:\t  204800 kB\n"
    peak = parse_vm_hwm(status)
    assert peak is not None
    assert peak.in_bytes == 204800 * 1024


def test_parse_vm_hwm_absent_or_malformed_is_none() -> None:
    assert parse_vm_hwm("Name:\tx\n") is None
    assert parse_vm_hwm("VmHWM:\tnot-a-number kB\n") is None


def test_subprocess_peak_memory_unreadable_proc_is_none() -> None:
    # PID -1 never has a /proc entry (and macOS has no /proc at all); the
    # served runner then reports zero rather than its own misleading RSS.
    assert subprocess_peak_memory(-1) is None


def test_llama_server_timings_count_cached_prompt_prefix() -> None:
    # Slot-cache hit: prompt_n is only the newly processed suffix, cache_n the
    # reused prefix; the request's true prompt size is their sum, while rates
    # stay over processed tokens (prompt_ms measures exactly that work).
    stats = stats_from_llama_server_timings(
        {
            "prompt_n": 10,
            "prompt_ms": 100.0,
            "cache_n": 490,
            "predicted_n": 20,
            "predicted_ms": 1000.0,
        }
    )
    assert stats is not None
    assert stats.prompt_tokens == 500
    assert stats.prompt_tps == 100.0  # 10 processed / 0.1s, not 500
    assert stats.generation_tokens == 20
