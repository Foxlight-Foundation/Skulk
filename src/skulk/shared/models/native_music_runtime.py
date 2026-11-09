"""Bind native model data and actually probed execution without qualification gates.

Serialized tensor counts can project the stored floating state at a loader's
effective precision. They are never the complete loading or inference footprint;
buffers, native libraries, transient copies and workload remain additional costs.
"""

from __future__ import annotations

import hashlib
import json
from typing import Final, Literal, Self, final

from pydantic import Field, TypeAdapter, model_validator

from skulk.shared.models.model_cards import ModelCard, ModelTask
from skulk.shared.models.native_music_layout import (
    NativeMusicTensorLayout,
    NativeMusicTensorLayouts,
)
from skulk.shared.types.native_music import (
    NativeMusicDevice,
    NativeMusicExecution,
    NativeMusicLibraryProbe,
    NativeMusicTensorPrecision,
)
from skulk.utils.pydantic_ext import FrozenModel

NativeMusicBackend = Literal[
    "ace_step-cpu", "ace_step-metal", "ace_step-cuda", "ace_step-rocm"
]
_DEVICES: Final[dict[NativeMusicBackend, NativeMusicDevice]] = {
    "ace_step-cpu": "cpu",
    "ace_step-metal": "mps",
    "ace_step-cuda": "cuda",
    "ace_step-rocm": "rocm",
}
_BACKEND = TypeAdapter[NativeMusicBackend](NativeMusicBackend)
_PRECISION_BYTES: Final[dict[NativeMusicTensorPrecision, int]] = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
}


def native_music_model_identity(card: ModelCard) -> str:
    """Hash complete role data/loader selections, excluding aliases and metadata.

    No machine, local path, license correction, optional header projection,
    runtime build or tested-device identity contributes to this artifact identity.
    Those facts retain their own separate policy and provenance boundaries.
    """
    profile, base = card.music, card.artifact_bundle
    native = profile.native if profile is not None else None
    if native is None or base is None or card.source_revision is None:
        raise ValueError("native music requires three complete immutable data roles")
    roles = (
        ("base", str(card.artifact_repository), card.source_revision, base),
        (
            "text_encoder",
            str(native.text_encoder.repo),
            native.text_encoder.revision,
            native.text_encoder.artifact_bundle,
        ),
        (
            "autoencoder",
            str(native.autoencoder.repo),
            native.autoencoder.revision,
            native.autoencoder.artifact_bundle,
        ),
    )
    selection = [
        {
            "role": role,
            "repository": repository,
            "revision": revision,
            "root": bundle.root,
            "files": [
                {
                    "path": item.path,
                    "size_bytes": item.size_bytes,
                    "object_id": item.object_id,
                }
                for item in sorted(bundle.files, key=lambda item: item.path)
            ],
        }
        for role, repository, revision, bundle in roles
    ]
    return hashlib.sha256(
        json.dumps(
            selection, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


@final
class NativeMusicModelPolicy(FrozenModel):
    """Pure admitted data/execution binding; no package install or model is ready."""

    backend: NativeMusicBackend = Field(description="Resolved explicit compute lane.")
    memory_policy_version: Literal[1] = Field(
        default=1,
        description="Shared native host/device loading and forward reservation policy assessed at placement; a changed policy requires re-admission.",
    )
    model_identity: str = Field(
        pattern=r"^[a-f0-9]{64}$", description="Exact data/loader selection identity."
    )
    payload_sha256: str = Field(
        pattern=r"^[a-f0-9]{64}$", description="Independently verified runtime payload."
    )
    execution: NativeMusicExecution = Field(
        description="Actual library-probed component dtype/device policy, still to be verified after model load."
    )
    threads: Literal[2] = Field(
        default=2, description="Pinned native intra-op CPU policy."
    )
    max_target_frames: int = Field(
        ge=48000,
        le=120 * 48000,
        description="Admitted native frame ceiling from card and global duration bounds.",
    )
    tensor_layouts: NativeMusicTensorLayouts | None = Field(
        description="Available intrinsic header facts; absence is an explicit estimation unknown."
    )

    @model_validator(mode="after")
    def validate_execution(self) -> Self:
        """Keep a restored policy's lane and complete component placement together."""
        actual = self.execution
        device = _DEVICES[self.backend]
        if (
            actual.device != device
            or actual.text_encoder_device != device
            or actual.autoencoder_device != device
            or actual.text_encoder_precision != actual.precision
            or device != "cuda"
            and actual.precision != "float32"
            or actual.autoencoder_precision
            != ("float16" if device == "mps" else actual.precision)
        ):
            raise ValueError("native music execution differs from its admitted lane")
        return self


def resolve_native_music_policy(
    card: ModelCard,
    backend: str,
    probe: NativeMusicLibraryProbe,
    *,
    observed_layouts: NativeMusicTensorLayouts | None = None,
) -> NativeMusicModelPolicy:
    """Resolve actual library math and exact model data without a hardware allowlist.

    Qualification is separate evidence. Genuine lane/probe disagreement and
    contradictory declared/observed file facts are refused. The worker must
    verify full selected object identities before supplying observed layouts.
    This pure binding grants neither model readiness nor memory admission.
    """
    selected = _BACKEND.validate_python(backend, strict=True)
    profile = card.music
    native = profile.native if profile is not None else None
    if (
        native is None
        or profile is None
        or set(card.tasks) != {ModelTask.TextToMusic, ModelTask.MusicEditing}
    ):
        raise ValueError("native music needs its complete generation/editing contract")
    if probe.device != _DEVICES[selected]:
        raise ValueError("native music library probe belongs to another compute lane")
    declared = native.tensor_layouts
    if (
        declared is not None
        and observed_layouts is not None
        and any(
            (
                expected.header_bytes,
                expected.header_sha256,
                sorted(
                    (group.dtype, group.tensor_count, group.element_count)
                    for group in expected.dtypes
                ),
            )
            != (
                observed.header_bytes,
                observed.header_sha256,
                sorted(
                    (group.dtype, group.tensor_count, group.element_count)
                    for group in observed.dtypes
                ),
            )
            for expected, observed in (
                (declared.base, observed_layouts.base),
                (declared.text_encoder, observed_layouts.text_encoder),
                (declared.autoencoder, observed_layouts.autoencoder),
            )
        )
    ):
        raise ValueError(
            "native music observed tensor facts contradict the selected card"
        )
    return NativeMusicModelPolicy(
        backend=selected,
        model_identity=native_music_model_identity(card),
        payload_sha256=probe.payload_sha256,
        execution=NativeMusicExecution(
            device=probe.device,
            precision=probe.precision,
            text_encoder_device=probe.device,
            text_encoder_precision=probe.precision,
            autoencoder_device=probe.device,
            autoencoder_precision=probe.autoencoder_precision,
        ),
        max_target_frames=min(120, profile.max_seconds) * 48000,
        tensor_layouts=declared or observed_layouts,
    )


@final
class NativeMusicTensorState(FrozenModel):
    """Projected stored floating tensor state only, excluding all peak overheads."""

    base: int = Field(
        ge=0, description="Selected base tensor state after actual loader casting."
    )
    text_encoder: int = Field(
        ge=0, description="Selected encoder tensor state after casting."
    )
    autoencoder: int = Field(
        ge=0, description="Selected VAE tensor state after casting."
    )

    @property
    def total_bytes(self) -> int:
        """Return selected tensor state, never a complete memory-fit threshold."""
        return self.base + self.text_encoder + self.autoencoder


def _floating_state(
    layout: NativeMusicTensorLayout, precision: NativeMusicTensorPrecision
) -> int | None:
    # Grouped serialization cannot identify integer buffers versus parameters,
    # or packed/complex loader semantics. Preserve an explicit unknown instead
    # of inventing cast savings for representations this loader has not resolved.
    if any(group.dtype not in {"BF16", "F16", "F32", "F64"} for group in layout.dtypes):
        return None
    return (
        sum(group.element_count for group in layout.dtypes)
        * _PRECISION_BYTES[precision]
    )


def native_music_tensor_state(
    policy: NativeMusicModelPolicy,
) -> NativeMusicTensorState | None:
    """Project known floating state or retain unknowns without fabricating RAM fit.

    This is a lower-level input to the still-required shared peak policy, not
    that policy. New buffers, libraries, loading copies, activations, native
    workspaces and workload/concurrency must be accounted independently.
    """
    layouts, execution = policy.tensor_layouts, policy.execution
    if layouts is None:
        return None
    base = _floating_state(layouts.base, execution.precision)
    encoder = _floating_state(layouts.text_encoder, execution.text_encoder_precision)
    decoder = _floating_state(layouts.autoencoder, execution.autoencoder_precision)
    if base is None or encoder is None or decoder is None:
        return None
    return NativeMusicTensorState(base=base, text_encoder=encoder, autoencoder=decoder)
