"""Wire-safe native math observations remain separate from installable packages."""

import pytest
from pydantic import ValidationError

from skulk.shared.models.tests.test_native_music_runtime import probe
from skulk.shared.types.native_music import NativeMusicDevice
from skulk.shared.types.profiling import NodeResources


@pytest.mark.parametrize(
    ("backend", "device"),
    [
        ("ace_step-cpu", "cpu"),
        ("ace_step-metal", "mps"),
        ("ace_step-cuda", "cuda"),
        ("ace_step-rocm", "rocm"),
    ],
)
def test_verified_native_library_facts_round_trip_without_paths(
    backend: str,
    device: NativeMusicDevice,
) -> None:
    """Unknown hardware may report real library math without claiming a model load."""
    observed = probe(device)
    resources = NodeResources(
        backends=frozenset({backend}),
        engine_builds={backend: "ace_step@sha256:" + observed.payload_sha256},
        native_music_probes={backend: observed},
    )
    assert NodeResources.model_validate_json(resources.model_dump_json()) == resources
    assert NodeResources.model_validate(resources.model_dump(mode="json")) == resources
    assert "model_identity" not in resources.model_dump_json()
    assert "generation" not in resources.model_dump_json()
    assert "interpreter" not in resources.model_dump_json()


@pytest.mark.parametrize("fault", ["installable", "build", "device", "unknown"])
def test_inconsistent_library_observation_cannot_supply_placement_facts(
    fault: str,
) -> None:
    """Availability, another build/lane, or an arbitrary label cannot prove native math."""
    backend = "unimplemented" if fault == "unknown" else "ace_step-cpu"
    observed = probe("cuda" if fault == "device" else "cpu")
    build = "ace_step@sha256:" + ("b" if fault == "build" else "a") * 64
    with pytest.raises(ValidationError, match="ready backend/build"):
        NodeResources(
            backends=frozenset() if fault == "installable" else frozenset({backend}),
            installable_engine_builds={backend: build},
            engine_builds={} if fault == "installable" else {backend: build},
            native_music_probes={backend: observed},
        )


def test_older_telemetry_without_native_probe_remains_readable() -> None:
    """The additive field does not turn a mixed-version peer into native capacity."""
    resources = NodeResources.model_validate({"backends": ["mlx"]})
    assert resources.native_music_probes == {}
    assert "native_music_probes" not in resources.model_dump(mode="json")
    assert "nativeMusicProbes" not in resources.model_dump(mode="json", by_alias=True)


@pytest.mark.parametrize("capability", [[12.0, 0], ["12", 0], [12], [12, 0, 1]])
def test_wire_shape_recovery_does_not_relax_native_precision_contract(
    capability: list[object],
) -> None:
    """Only the container changes; CUDA capability items and pair length stay strict."""
    observed = probe("cuda").model_dump(mode="json")
    observed["cuda_compute_capability"] = capability
    with pytest.raises(ValidationError):
        NodeResources.model_validate(
            {
                "backends": ["ace_step-cuda"],
                "engine_builds": {"ace_step-cuda": "ace_step@sha256:" + "a" * 64},
                "native_music_probes": {"ace_step-cuda": observed},
            }
        )
