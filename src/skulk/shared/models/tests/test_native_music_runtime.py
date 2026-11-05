"""Actual execution, artifact identity and scalar state never imply readiness."""

import copy
from typing import cast

import pytest
from pydantic import ValidationError

from skulk.shared.models.model_cards import ModelCard, model_engine_requirements
from skulk.shared.models.native_music_layout import NativeMusicTensorLayouts
from skulk.shared.models.native_music_runtime import (
    NativeMusicModelPolicy,
    native_music_model_identity,
    native_music_tensor_state,
    resolve_native_music_policy,
)
from skulk.shared.models.tests.test_native_music_layout import body, layouts, section
from skulk.shared.types.native_music import NativeMusicDevice, NativeMusicLibraryProbe


def probe(
    device: NativeMusicDevice = "cpu", *, cuda_major: int = 12
) -> NativeMusicLibraryProbe:
    """Represent the actual library probe contract, not a physical device run."""
    precision = "bfloat16" if cuda_major >= 8 else "float16"
    return NativeMusicLibraryProbe(
        payload_sha256="a" * 64,
        device=device,
        precision=precision if device == "cuda" else "float32",
        autoencoder_precision="float16"
        if device == "mps"
        else precision
        if device == "cuda"
        else "float32",
        cuda_compute_capability=(cuda_major, 0) if device == "cuda" else None,
        torch_version="controlled-library-probe",
    )


def card_with_layouts() -> ModelCard:
    """Bind the original Registry header facts to the complete selected objects."""
    document = body()
    section(document, "music", "native")["tensor_layouts"] = layouts().model_dump(
        mode="json"
    )
    return ModelCard.model_validate(document)


@pytest.mark.parametrize(
    ("backend", "device", "major", "expected"),
    [
        ("ace_step-cpu", "cpu", 12, 12633377824),
        ("ace_step-metal", "mps", 12, 12295986972),
        ("ace_step-rocm", "rocm", 12, 12633377824),
        ("ace_step-cuda", "cuda", 12, 6316688912),
        ("ace_step-cuda", "cuda", 7, 6316688912),
    ],
)
def test_actual_component_math_projects_checkpoint_state_without_a_memory_floor(
    backend: str,
    device: NativeMusicDevice,
    major: int,
    expected: int,
) -> None:
    """The same BF16 files expand differently; no hardware test allowlist applies."""
    card = card_with_layouts()
    policy = resolve_native_music_policy(card, backend, probe(device, cuda_major=major))
    assert policy.max_target_frames == 120 * 48000
    assert policy.payload_sha256 == "a" * 64
    assert policy.model_identity == native_music_model_identity(card)
    assert (
        policy.execution.autoencoder_precision
        == probe(device, cuda_major=major).autoencoder_precision
    )
    state = native_music_tensor_state(policy)
    assert state is not None and state.total_bytes == expected
    assert (
        NativeMusicModelPolicy.model_validate_json(policy.model_dump_json()) == policy
    )
    assert not model_engine_requirements(card)
    assert not card.placement.compatible_backends
    assert "required_memory" not in policy.model_dump()


def test_legacy_native_card_keeps_unknown_layout_until_actual_observation() -> None:
    """A missing projection is not a hardware or model compatibility ban."""
    card = ModelCard.model_validate(body())
    before = resolve_native_music_policy(card, "ace_step-cpu", probe())
    assert native_music_tensor_state(before) is None
    after = resolve_native_music_policy(
        card, "ace_step-cpu", probe(), observed_layouts=layouts()
    )
    assert before.model_identity == after.model_identity
    assert after.tensor_layouts == layouts()
    assert native_music_tensor_state(after) is not None


@pytest.mark.parametrize(
    "backend", ["ace_step", "ace_step-vulkan", "audio_cpp-cpu", "", "ace_step-CUDA"]
)
def test_invalid_or_unimplemented_explicit_lane_is_not_auto_selected(
    backend: str,
) -> None:
    """A prepared CPU probe cannot silently fulfill a different requested backend."""
    with pytest.raises(ValidationError):
        resolve_native_music_policy(card_with_layouts(), backend, probe())


@pytest.mark.parametrize("device", ["mps", "cuda", "rocm"])
def test_another_actual_lane_cannot_satisfy_cpu_placement(
    device: NativeMusicDevice,
) -> None:
    """Actual dependency execution remains part of the binding, including untested GPUs."""
    with pytest.raises(ValueError, match="another compute lane"):
        resolve_native_music_policy(card_with_layouts(), "ace_step-cpu", probe(device))


@pytest.mark.parametrize(
    "fault",
    ["device", "encoder_device", "encoder_dtype", "vae_dtype", "diffusion_dtype"],
)
def test_restored_policy_refuses_component_precision_or_device_drift(
    fault: str,
) -> None:
    """Persisted execution cannot bypass the same pure resolver's invariants."""
    policy = resolve_native_music_policy(
        card_with_layouts(), "ace_step-metal", probe("mps")
    )
    document = policy.model_dump(mode="json")
    execution = section(document, "execution")
    if fault == "device":
        execution["device"] = "cpu"
    elif fault == "encoder_device":
        execution["text_encoder_device"] = "cpu"
    elif fault == "encoder_dtype":
        execution["text_encoder_precision"] = "float16"
    elif fault == "vae_dtype":
        execution["autoencoder_precision"] = "float32"
    else:
        execution["precision"] = "float16"
    with pytest.raises(ValidationError, match="admitted lane"):
        NativeMusicModelPolicy.model_validate(document)


@pytest.mark.parametrize("role", ["base", "text_encoder", "autoencoder"])
def test_observed_header_cannot_contradict_card_facts(role: str) -> None:
    """A correct-size but different actual prefix is not the admitted tensor layout."""
    document = layouts().model_dump(mode="json")
    section(document, role)["header_sha256"] = "b" * 64
    observed = NativeMusicTensorLayouts.model_validate(document)
    with pytest.raises(ValueError, match="contradict"):
        resolve_native_music_policy(
            card_with_layouts(), "ace_step-cpu", probe(), observed_layouts=observed
        )


def test_data_identity_ignores_aliases_metadata_and_file_order_but_not_selection() -> (
    None
):
    """Correcting attribution or adding header facts does not invent new weight bytes."""
    original = ModelCard.model_validate(body())
    changed = body()
    changed["model_id"] = "custom/native-alias"
    section(changed, "license")["name"] = "MIT attribution correction"
    section(changed, "music", "native")["tensor_layouts"] = layouts().model_dump(
        mode="json"
    )
    for bundle in (
        section(changed, "artifact_bundle"),
        section(changed, "music", "native", "text_encoder", "artifact_bundle"),
        section(changed, "music", "native", "autoencoder", "artifact_bundle"),
    ):
        cast("list[object]", bundle["files"]).reverse()
    assert native_music_model_identity(original) == native_music_model_identity(
        ModelCard.model_validate(changed)
    )
    assert original.model_id != ModelCard.model_validate(changed).model_id


@pytest.mark.parametrize("fault", ["revision", "repository", "object", "root"])
def test_data_identity_retains_every_component_and_loader_root(fault: str) -> None:
    """Equal tensor sizes or one primary alias cannot hide a different companion."""
    original = body()
    changed = copy.deepcopy(original)
    component = section(changed, "music", "native", "text_encoder")
    if fault == "revision":
        # Shared-source roles must keep one consistent revision together.
        component["revision"] = "b" * 40
        section(changed, "music", "native", "autoencoder")["revision"] = "b" * 40
    elif fault == "repository":
        component["repo"] = "custom/encoder-mirror"
    elif fault == "object":
        files = cast(
            "list[dict[str, object]]", section(component, "artifact_bundle")["files"]
        )
        next(item for item in files if str(item["path"]).endswith(".safetensors"))[
            "object_id"
        ] = "sha256:" + "b" * 64
    else:
        bundle = section(component, "artifact_bundle")
        old = str(bundle["root"])
        bundle["root"] = "another-encoder-root"
        for item in cast("list[dict[str, object]]", bundle["files"]):
            item["path"] = str(item["path"]).replace(
                old + "/", "another-encoder-root/", 1
            )
    assert native_music_model_identity(
        ModelCard.model_validate(original)
    ) != native_music_model_identity(ModelCard.model_validate(changed))


def test_unknown_nonfloating_buffer_semantics_do_not_invent_cast_savings() -> None:
    """Header observation can be complete while loader memory semantics stay unknown."""
    document = layouts().model_dump(mode="json")
    groups = cast("list[dict[str, object]]", section(document, "base")["dtypes"])
    groups[0]["dtype"] = "I16"
    observed = NativeMusicTensorLayouts.model_validate(document)
    policy = resolve_native_music_policy(
        ModelCard.model_validate(body()),
        "ace_step-cpu",
        probe(),
        observed_layouts=observed,
    )
    assert policy.tensor_layouts is not None
    assert native_music_tensor_state(policy) is None


def test_equivalent_group_order_preserves_admitted_metadata_policy() -> None:
    """Group order is not tensor truth and cannot cause false placement/worker drift."""
    document = body()
    declared = layouts().model_dump(mode="json")
    section(declared, "base")["dtypes"] = [
        {"dtype": "BF16", "tensor_count": 676, "element_count": 2393872516},
        {"dtype": "F16", "tensor_count": 1, "element_count": 2},
    ]
    section(document, "music", "native")["tensor_layouts"] = declared
    card = ModelCard.model_validate(document)
    observed = copy.deepcopy(declared)
    cast("list[object]", section(observed, "base")["dtypes"]).reverse()
    expected = resolve_native_music_policy(card, "ace_step-cpu", probe())
    actual = resolve_native_music_policy(
        card,
        "ace_step-cpu",
        probe(),
        observed_layouts=NativeMusicTensorLayouts.model_validate(observed),
    )
    assert actual == expected
