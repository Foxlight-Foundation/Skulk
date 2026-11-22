"""Resolve implemented native separator policy independently of qualification."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Final, Literal, Self, final

from pydantic import Field, TypeAdapter, model_validator

from skulk.shared.models.model_cards import ModelCard
from skulk.shared.types.audio_separation import (
    MAX_SEPARATION_DECODED_BYTES,
    MAX_SEPARATION_SOURCE_BYTES,
    MAX_SEPARATION_STEM_BYTES,
    SeparationDevice,
)
from skulk.shared.types.memory import Memory
from skulk.utils.pydantic_ext import FrozenModel

HTDEMUCS_WEIGHTS_SHA256: Final = (
    "d9fa14133cfcc034a6758923bb3a8ca9f8dfd0b582134643bbf83f72c17576dd"
)
HTDEMUCS_WEIGHTS_BYTES: Final = 84_025_440
DemucsBackend = Literal["demucs-cpu", "demucs-metal", "demucs-cuda", "demucs-rocm"]
_DEVICES: Final[dict[DemucsBackend, SeparationDevice]] = {
    "demucs-cpu": "cpu",
    "demucs-metal": "mps",
    "demucs-cuda": "cuda",
    "demucs-rocm": "rocm",
}
_BACKEND = TypeAdapter[DemucsBackend](DemucsBackend)


@final
class DemucsModelPolicy(FrozenModel):
    """Exact mounted native policy; no weight/package download or load is implied.

    Backend vocabulary uses ``metal`` consistently with other Skulk engines;
    only the private native adapter translates it to PyTorch's ``mps`` device.
    The two-thread policy and FP32 training window bind resource assessment and
    native execution. Callers cannot substitute native options or precision.
    """

    backend: DemucsBackend = Field(description="Resolved explicit engine/compute lane.")
    device: SeparationDevice = Field(description="Private native device spelling.")
    weights_path: str = Field(
        min_length=1,
        max_length=4096,
        description="Selected path relative to the model download.",
    )
    weights_sha256: Literal[
        "d9fa14133cfcc034a6758923bb3a8ca9f8dfd0b582134643bbf83f72c17576dd"
    ] = Field(
        default=HTDEMUCS_WEIGHTS_SHA256, description="Implemented tensor identity."
    )
    precision: Literal["float32"] = Field(
        default="float32", description="Actual inference precision."
    )
    threads: Literal[2] = Field(
        default=2, description="Bounded native CPU thread policy."
    )
    segment_seconds: float = Field(
        default=7.8,
        ge=7.8,
        le=7.8,
        allow_inf_nan=False,
        description="Model's trained forward window.",
    )

    @model_validator(mode="after")
    def validate_native_binding(self) -> Self:
        """Keep a rehydrated policy's private device and path bound to its lane."""
        path = PurePosixPath(self.weights_path)
        if self.device != _DEVICES[self.backend]:
            raise ValueError("native separation device differs from its engine lane")
        if (
            path.is_absolute()
            or self.weights_path != path.as_posix()
            or ".." in path.parts
            or "\\" in self.weights_path
            or path.suffix != ".safetensors"
        ):
            raise ValueError("native separation requires an owned relative tensor path")
        return self


def resolve_demucs_policy(card: ModelCard, backend: str) -> DemucsModelPolicy:
    """Bind one exact implemented weight/constructor selection to a compute lane.

    Artifact identity, rather than repository name or tested hardware, selects
    this closed safetensors loader. Other separator artifacts require an actual
    loader implementation; a support/qualification record cannot make one work.
    The caller still verifies its package, local files and device before load.
    """
    selected = _BACKEND.validate_python(backend, strict=True)
    profile, bundle = card.separation, card.artifact_bundle
    if profile is None or bundle is None:
        raise ValueError("native separation requires typed exact model components")
    if (
        card.quantization != "F32"
        or profile.encoder_depth != 4
        or profile.transformer_layers_per_branch != 5
        or profile.transformer_channels != 512
        or profile.trained_segment_seconds != 7.8
    ):
        raise ValueError(
            "native separator does not implement this model constructor/precision"
        )
    relative = PurePosixPath(bundle.root or ".") / profile.weights_file
    if len(bundle.files) != 1:
        raise ValueError("native separator requires one exact safetensors component")
    tensor = bundle.files[0]
    if (
        tensor.path != relative.as_posix()
        or tensor.size_bytes != HTDEMUCS_WEIGHTS_BYTES
        or tensor.object_id != "sha256:" + HTDEMUCS_WEIGHTS_SHA256
    ):
        raise ValueError(
            "native separator does not implement the selected tensor identity"
        )
    return DemucsModelPolicy(
        backend=selected,
        device=_DEVICES[selected],
        weights_path=relative.as_posix(),
    )


def estimate_demucs_peak(card: ModelCard, *, backend: str | None) -> Memory:
    """Reserve the implemented FP32/two-thread/120-second CPU or MPS workload.

    This engine/workload policy is not a model-card required-memory number.
    It includes the complete trained forward window even for short sources,
    native libraries, actual F32 weights, two decoded-source workspaces, source
    bytes, three canonical stereo workspaces, all four float32 estimates and
    one bounded encoded verification buffer. Round up with a 3 GiB reference
    floor above the measured sealed CPU peak; enforce the same bound at load
    and inference. Runtime/workload changes need a corresponding policy update.
    GPU dependency profiles need separate host/device accounting before use.
    """
    selected = backend or "demucs-cpu"
    policy = resolve_demucs_policy(card, selected)
    if policy.device not in {"cpu", "mps"}:
        raise ValueError("native separation accelerator memory policy is not packaged")
    stereo_frames = 44100 * 120 * 2
    stereo_float_bytes = stereo_frames * 4
    native_bytes = (
        HTDEMUCS_WEIGHTS_BYTES
        + 2 * 1024**3
        + MAX_SEPARATION_SOURCE_BYTES
        + 2 * MAX_SEPARATION_DECODED_BYTES
        + 3 * stereo_float_bytes
        + 4 * stereo_float_bytes
        + MAX_SEPARATION_STEM_BYTES
    )
    block = 64 * 1024**2
    return Memory.from_bytes(
        max(3 * 1024**3, ((native_bytes + block - 1) // block) * block)
    )
