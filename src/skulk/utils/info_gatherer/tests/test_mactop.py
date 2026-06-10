from pydantic import TypeAdapter

from skulk.shared.types.profiling import MachMemoryCategories
from skulk.utils.info_gatherer.info_gatherer import GatheredInfo
from skulk.utils.info_gatherer.mactop import MacmonMetrics, MactopMetrics

# A representative `mactop --headless --format json` line (trimmed; real output
# also carries net_disk, gpu_metrics, core_usages, tflops_* — all ignored).
_SAMPLE = (
    '{"timestamp":"2026-06-09T07:41:53-05:00",'
    '"soc_metrics":{"cpu_power":0.06,"gpu_power":0.09,"ane_power":0,'
    '"system_power":11.36,"total_power":11.65,"gpu_freq_mhz":925,'
    '"gpu_active":8.66,"soc_temp":44.58,"cpu_temp":44.58,"gpu_temp":39.46,'
    '"dram_bw_combined_gbs":1.78},'
    '"memory":{"total":17179869184,"used":7687569408,"available":9492299776,'
    '"swap_total":2147483648,"swap_used":536870912},'
    '"net_disk":{"out_packets_per_sec":1.0},'
    '"cpu_usage":6.4,"ecpu_usage":[1196,39.69],"pcpu_usage":[1491,0.55],'
    '"gpu_usage":8.66,"gpu_metrics":{"freq_mhz":925,"active_percent":8.66},'
    '"core_usages":[18.9,15.8]}'
)


def test_parses_system_profile():
    m = MactopMetrics.from_raw_json(_SAMPLE)
    assert m.system_profile.gpu_usage == 8.66
    assert m.system_profile.temp == 39.46
    assert m.system_profile.sys_power == 11.36
    # ecpu/pcpu come from the [freq, usage%] tuples — we keep the usage %.
    assert m.system_profile.ecpu_usage == 39.69
    assert m.system_profile.pcpu_usage == 0.55


def test_parses_memory():
    m = MactopMetrics.from_raw_json(_SAMPLE)
    assert m.memory.ram_total.in_bytes == 17179869184
    # Without a Mach page-category snapshot, mactop's raw `available` (which
    # counts reclaimable file cache as used) is kept as the fallback.
    assert m.memory.ram_available.in_bytes == 9492299776
    assert m.memory.swap_total.in_bytes == 2147483648
    assert m.memory.swap_available.in_bytes == (2147483648 - 536870912)


def test_mach_snapshot_overrides_cache_deflated_available():
    # The placement-facing fix: with a vm_stat snapshot, ram_available is the
    # GPU-wireable figure (total − wired − anonymous − compressor), which does
    # not count reclaimable file cache as used the way mactop's raw figure
    # does. Numbers shaped like the observed kite incident: a just-downloaded
    # model's file cache deflated mactop's `available` to ~9.5 GB while far
    # more was genuinely wireable.
    categories = MachMemoryCategories(
        wired_bytes=2_000_000_000,
        anonymous_bytes=3_000_000_000,
        compressor_bytes=500_000_000,
    )
    m = MactopMetrics.from_raw_json(_SAMPLE, categories)
    assert m.memory.ram_available.in_bytes == (
        17179869184 - 2_000_000_000 - 3_000_000_000 - 500_000_000
    )
    # The rest of the memory block still comes from mactop.
    assert m.memory.ram_total.in_bytes == 17179869184
    assert m.memory.swap_available.in_bytes == (2147483648 - 536870912)


def test_mach_snapshot_never_yields_negative_available():
    # Pathological counters (e.g. anonymous + wired exceeding total mid-churn)
    # clamp to zero rather than gossiping a negative availability.
    categories = MachMemoryCategories(
        wired_bytes=17_179_869_184,
        anonymous_bytes=17_179_869_184,
        compressor_bytes=0,
    )
    m = MactopMetrics.from_raw_json(_SAMPLE, categories)
    assert m.memory.ram_available.in_bytes == 0


def test_ignores_unknown_fields():
    # Forward-compatibility: extra/new mactop fields must not raise.
    noisy = _SAMPLE[:-1] + ',"some_future_field":{"nested":1}}'
    m = MactopMetrics.from_raw_json(noisy)
    assert m.system_profile.gpu_usage == 8.66


def test_old_macmon_event_still_decodes():
    # Rolling-upgrade back-compat: NodeGatheredInfo.info is gossiped/replayed, so
    # a node on the new build must still decode the `{"MacmonMetrics": ...}` tag
    # emitted by macOS workers still on the pre-mactop build, onto the same
    # normalized system_profile/memory shape (no telemetry gap mid-upgrade).
    mactop = MactopMetrics.from_raw_json(_SAMPLE)
    macmon = MacmonMetrics(
        system_profile=mactop.system_profile, memory=mactop.memory
    )
    wire = macmon.model_dump_json()
    assert '"MacmonMetrics"' in wire  # TaggedModel keys by class name

    adapter: TypeAdapter[GatheredInfo] = TypeAdapter(GatheredInfo)
    decoded = adapter.validate_json(wire)
    assert isinstance(decoded, MacmonMetrics)
    assert decoded.system_profile.gpu_usage == 8.66
    assert decoded.memory.ram_total.in_bytes == 17179869184
