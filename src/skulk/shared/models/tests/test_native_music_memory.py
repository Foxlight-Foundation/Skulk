"""Whole-workload and separate-domain native ACE reservation regressions."""

import pytest
from pydantic import ValidationError

from skulk.shared.models.model_cards import ModelCard
from skulk.shared.models.native_music_layout import NativeMusicTensorLayouts
from skulk.shared.models.native_music_memory import (
    NativeMusicMemoryEstimate,
    estimate_native_music_peak,
)
from skulk.shared.models.native_music_runtime import resolve_native_music_policy
from skulk.shared.models.tests.test_native_music_layout import body, section
from skulk.shared.models.tests.test_native_music_runtime import card_with_layouts, probe
from skulk.shared.types.native_music import NativeMusicDevice


@pytest.mark.parametrize(
    ("backend", "device", "state"),
    [
        ("ace_step-cpu", "cpu", 12633377824),
        ("ace_step-metal", "mps", 12295986972),
        ("ace_step-cuda", "cuda", 6316688912),
        ("ace_step-rocm", "rocm", 12633377824),
    ],
)
def test_all_three_actual_components_and_full_workload_are_reserved(
    backend: str,
    device: NativeMusicDevice,
    state: int,
) -> None:
    """BF16 object bytes cannot stand in for FP32 model/loader/forward costs."""
    card = card_with_layouts()
    policy = resolve_native_music_policy(card, backend, probe(device))
    peak = estimate_native_music_peak(card, policy)
    assert peak.projected_tensor_bytes == state
    assert peak.assumptions == ("workspace_allowance_unqualified",)
    assert peak.workspace_bytes >= 8 * 1024**3
    assert peak.host_peak_bytes > 2 * 1024**3
    if device == "cpu":
        assert peak.device_peak_bytes == 0
        assert peak.host_forward_bytes > state + peak.workspace_bytes
        assert peak.host_load_bytes > state
        assert peak.unified_peak_bytes == peak.host_peak_bytes
    else:
        assert peak.device_forward_bytes > state + peak.workspace_bytes
        assert peak.host_load_bytes > 4 * 1024**3
        assert peak.unified_peak_bytes >= peak.host_peak_bytes
        assert peak.unified_peak_bytes >= peak.device_peak_bytes
    assert NativeMusicMemoryEstimate.model_validate_json(peak.model_dump_json()) == peak


def test_unified_reservation_combines_simultaneous_phases_once() -> None:
    """Independent host and device maxima must not be added as if simultaneous."""
    projection = NativeMusicMemoryEstimate(
        projected_tensor_bytes=10,
        host_load_bytes=30,
        device_load_bytes=10,
        host_forward_bytes=5,
        device_forward_bytes=50,
        workspace_bytes=1,
        assumptions=("workspace_allowance_unqualified",),
    )
    assert projection.host_peak_bytes == 30
    assert projection.device_peak_bytes == 50
    assert projection.unified_peak_bytes == 55
    assert (
        projection.unified_peak_bytes
        < projection.host_peak_bytes + projection.device_peak_bytes
    )


def test_gpu_host_forward_reserves_upstream_temporary_cpu_vae_decode() -> None:
    """A final restored GPU device does not erase CPU fallback's physical cost."""
    card = card_with_layouts()
    cuda = estimate_native_music_peak(
        card, resolve_native_music_policy(card, "ace_step-cuda", probe("cuda"))
    )
    metal = estimate_native_music_peak(
        card, resolve_native_music_policy(card, "ace_step-metal", probe("mps"))
    )
    assert cuda.host_forward_bytes >= metal.host_forward_bytes + cuda.workspace_bytes
    assert cuda.host_forward_bytes > cuda.workspace_bytes


def test_loading_and_forward_are_both_charged_on_cpu() -> None:
    """Loading casts can peak above inference; reserving only steady weights is unsafe."""
    card = card_with_layouts()
    cpu = estimate_native_music_peak(
        card, resolve_native_music_policy(card, "ace_step-cpu", probe())
    )
    assert cpu.host_load_bytes > cpu.host_forward_bytes
    assert cpu.host_peak_bytes == cpu.host_load_bytes


def test_shorter_card_workload_reduces_workspace_without_losing_model_state() -> None:
    """An admitted duration ceiling affects activation/media policy, not weight bytes."""
    full = card_with_layouts()
    document = full.model_dump(mode="json")
    section(document, "music")["max_seconds"] = 30
    short = ModelCard.model_validate(document)
    full_peak = estimate_native_music_peak(
        full, resolve_native_music_policy(full, "ace_step-cuda", probe("cuda"))
    )
    short_peak = estimate_native_music_peak(
        short, resolve_native_music_policy(short, "ace_step-cuda", probe("cuda"))
    )
    assert short_peak.projected_tensor_bytes == full_peak.projected_tensor_bytes
    assert short_peak.workspace_bytes < full_peak.workspace_bytes
    assert short_peak.device_forward_bytes < full_peak.device_forward_bytes
    assert short_peak.host_load_bytes <= full_peak.host_load_bytes


def test_missing_header_is_explicit_pessimism_and_can_be_reassessed() -> None:
    """Unknown geometry remains runnable if resources fit; observation lowers its bound."""
    card = ModelCard.model_validate(body())
    unknown = resolve_native_music_policy(card, "ace_step-cpu", probe())
    known_card = card_with_layouts()
    native = known_card.music.native if known_card.music is not None else None
    assert native is not None and native.tensor_layouts is not None
    observed = resolve_native_music_policy(
        card, "ace_step-cpu", probe(), observed_layouts=native.tensor_layouts
    )
    upper = estimate_native_music_peak(card, unknown)
    reassessed = estimate_native_music_peak(card, observed)
    assert unknown.model_identity == observed.model_identity
    assert "tensor_layout_unknown" in upper.assumptions
    assert "tensor_layout_unknown" not in reassessed.assumptions
    assert upper.host_peak_bytes > reassessed.host_peak_bytes
    assert reassessed.projected_tensor_bytes == 12633377824


@pytest.mark.parametrize(
    "fault",
    ["identity", "duration", "threads", "precision", "layout", "memory_version"],
)
def test_changed_saved_policy_cannot_understate_an_admitted_workload(
    fault: str,
) -> None:
    """Copied/malformed state cannot bypass exact data, dtype, header and workload binding."""
    card = card_with_layouts()
    policy = resolve_native_music_policy(card, "ace_step-cpu", probe())
    if fault == "identity":
        changed = policy.model_copy(update={"model_identity": "0" * 64})
    elif fault == "duration":
        changed = policy.model_copy(update={"max_target_frames": 48000})
    elif fault == "threads":
        changed = policy.model_copy(update={"threads": 1})
    elif fault == "layout":
        changed = policy.model_copy(update={"tensor_layouts": None})
    elif fault == "memory_version":
        changed = policy.model_copy(update={"memory_policy_version": 2})
    else:
        changed = policy.model_copy(
            update={
                "execution": policy.execution.model_copy(
                    update={"precision": "float16"}
                )
            }
        )
    with pytest.raises((ValueError, ValidationError)):
        estimate_native_music_peak(card, changed)


def test_observed_layout_cannot_understate_a_complete_selected_object() -> None:
    """A card without declared facts still binds observations to exact file lengths."""
    card = ModelCard.model_validate(body())
    declared = card_with_layouts()
    native = declared.music.native if declared.music is not None else None
    assert native is not None and native.tensor_layouts is not None
    layouts = native.tensor_layouts.model_copy(
        update={
            "base": native.tensor_layouts.base.model_copy(update={"header_bytes": 16})
        }
    )
    policy = resolve_native_music_policy(
        card, "ace_step-cpu", probe(), observed_layouts=layouts
    )
    with pytest.raises(ValueError, match="complete objects"):
        estimate_native_music_peak(card, policy)


def test_nonfloating_layout_has_visible_upper_bound_instead_of_cast_savings() -> None:
    """Grouped integer/packed semantics cannot pretend to have known FP32 savings."""
    card = ModelCard.model_validate(body())
    declared = card_with_layouts()
    native = declared.music.native if declared.music is not None else None
    assert native is not None and native.tensor_layouts is not None
    document = native.tensor_layouts.model_dump(mode="json")
    base = section(document, "base")
    base["dtypes"] = [
        {"dtype": "U16", "tensor_count": 677, "element_count": 2393872518}
    ]
    layouts = NativeMusicTensorLayouts.model_validate(document)
    policy = resolve_native_music_policy(
        card, "ace_step-cpu", probe(), observed_layouts=layouts
    )
    peak = estimate_native_music_peak(card, policy)
    assert "nonfloating_layout_upper_bound" in peak.assumptions
    assert peak.projected_tensor_bytes > 12633377824
