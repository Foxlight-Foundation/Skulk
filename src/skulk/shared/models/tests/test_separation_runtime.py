"""Native artifact/compute policy rejects real loader mismatches, not test gaps."""

import tomllib
from pathlib import Path
from typing import Literal

import pytest
from pydantic import ValidationError

from skulk.shared.backends import platform_compatible_backends
from skulk.shared.models.model_cards import ModelCard, model_engine_requirements
from skulk.shared.models.separation_runtime import (
    DemucsModelPolicy,
    estimate_demucs_peak,
    resolve_demucs_policy,
)
from skulk.shared.types.audio_separation import SeparationDevice


def card() -> ModelCard:
    """Read an exact artifact fixture with no support/qualification claims."""
    return ModelCard.model_validate(
        tomllib.loads(Path(__file__).with_name("htdemucs-card.toml").read_text())
    )


@pytest.mark.parametrize(
    ("backend", "device"),
    [
        ("demucs-cpu", "cpu"),
        ("demucs-metal", "mps"),
        ("demucs-cuda", "cuda"),
        ("demucs-rocm", "rocm"),
    ],
)
def test_unqualified_artifact_resolves_explicit_native_policy(
    backend: str, device: SeparationDevice
) -> None:
    """Compute spelling, FP32 and model geometry do not depend on tested hardware."""
    policy = resolve_demucs_policy(card(), backend)
    assert policy.device == device and policy.backend == backend
    assert policy.threads == 2 and policy.precision == "float32"
    assert policy.segment_seconds == 7.8
    assert policy.weights_path == "955717e8.safetensors"
    assert DemucsModelPolicy.model_validate_json(policy.model_dump_json()) == policy


@pytest.mark.parametrize(
    "backend", ["demucs", "demucs-mps", "demucs-vulkan", "audio_cpp-cpu"]
)
def test_unknown_or_implicit_lane_cannot_select_native_device(backend: str) -> None:
    """A real missing implementation cannot silently select an accelerator or CPU."""
    with pytest.raises(ValidationError):
        resolve_demucs_policy(card(), backend)


def test_native_selection_uses_bytes_not_repository_or_test_inventory() -> None:
    """An exact mirror retains loader compatibility, with a selected nested root."""
    original = card()
    profile, bundle = original.separation, original.artifact_bundle
    assert profile is not None and bundle is not None
    mirrored = original.model_copy(
        update={
            "model_id": "a-mirror/the-same-native-model",
            "artifact_bundle": bundle.model_copy(
                update={
                    "root": "native",
                    "files": (
                        bundle.files[0].model_copy(
                            update={"path": "native/955717e8.safetensors"}
                        ),
                    ),
                }
            ),
        }
    )
    policy = resolve_demucs_policy(mirrored, "demucs-cpu")
    assert policy.weights_path == "native/955717e8.safetensors"


@pytest.mark.parametrize(
    "change", ["identity", "size", "precision", "constructor", "window"]
)
def test_loader_refuses_a_different_artifact_or_constructor(
    change: Literal["identity", "size", "precision", "constructor", "window"],
) -> None:
    """Runtime restrictions describe its implemented loader rather than claims."""
    original = card()
    profile, bundle = original.separation, original.artifact_bundle
    assert profile is not None and bundle is not None
    changed = original
    if change in ("identity", "size"):
        file = bundle.files[0].model_copy(
            update={
                "object_id": "sha256:" + "f" * 64,
            }
            if change == "identity"
            else {"size_bytes": 1234}
        )
        changed = original.model_copy(
            update={"artifact_bundle": bundle.model_copy(update={"files": (file,)})}
        )
    elif change == "precision":
        changed = original.model_copy(update={"quantization": "F16"})
    else:
        changed = original.model_copy(
            update={
                "separation": profile.model_copy(
                    update={
                        "transformer_channels": 1024,
                    }
                    if change == "constructor"
                    else {"trained_segment_seconds": 15.0}
                )
            }
        )
    with pytest.raises(ValueError, match="does not implement"):
        resolve_demucs_policy(changed, "demucs-cpu")


def test_separation_model_truth_and_platform_routing_are_distinct() -> None:
    """An empty legacy declaration still resolves model truth without a text runner."""
    assert model_engine_requirements(card()) == frozenset({"demucs"})
    declared = frozenset({"demucs", "mlx", "llama_cpp", "audio_cpp"})
    assert platform_compatible_backends(
        declared, card_serves_vision=False, card_serves_separation=True
    ) == frozenset({"demucs-cpu", "demucs-metal", "demucs-cuda", "demucs-rocm"})
    assert "demucs" not in platform_compatible_backends(
        declared, card_serves_vision=False
    )


@pytest.mark.parametrize("change", ["device", "path", "threads", "window", "precision"])
def test_deserialized_policy_cannot_change_its_native_binding(change: str) -> None:
    """Rehydration checks the same device/path/FP32/budget constraints as resolution."""
    values = resolve_demucs_policy(card(), "demucs-cpu").model_dump(mode="json")
    values.update(
        {
            "device": "cuda",
        }
        if change == "device"
        else {
            "weights_path": "../other.safetensors",
        }
        if change == "path"
        else {
            "threads": 32,
        }
        if change == "threads"
        else {
            "segment_seconds": 15.0,
        }
        if change == "window"
        else {"precision": "float16"}
    )
    with pytest.raises(ValidationError):
        DemucsModelPolicy.model_validate(values)


def test_native_reservation_covers_runtime_not_just_downloaded_weights() -> None:
    """The actual FP32 forward and source/four-output buffers dominate the weights."""
    from skulk.shared.models.memory_estimate import (
        estimate_shard_footprint,
        memory_estimate_notes,
    )

    model = card()
    estimate = estimate_demucs_peak(model, backend="demucs-cpu")
    assert estimate.in_bytes >= 3 * 1024**3 > model.storage_size.in_bytes
    assert estimate_shard_footprint(model, 1.0, context_budget=1, resolved_backend="demucs-cpu") == estimate
    assert estimate_shard_footprint(model, 1.0, context_budget=1000000, resolved_backend="demucs-cpu") == estimate
    assert any("two-thread" in note for note in memory_estimate_notes(model, resolved_backend="demucs-cpu", memory_settings=None))
    with pytest.raises(ValueError, match="complete model shard"):
        estimate_shard_footprint(model, 0.5, resolved_backend="demucs-cpu")
    with pytest.raises(ValueError, match="not packaged"):
        estimate_demucs_peak(model, backend="demucs-cuda")
