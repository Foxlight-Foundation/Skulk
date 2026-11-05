"""One explicit native ACE loading/forward memory projection for every owner.

This versioned planning policy includes complete component casts, loader copies,
bounded input/output custody and the pinned decoder's temporary CPU fallback.
The workspace allowance is conservative policy, not measured qualification or
a promise that an arbitrary custom configuration fits. Peak qualification and
the live process guard remain required before advertising native capacity.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final, Literal, final

from pydantic import Field

from skulk.shared.models.model_cards import ArtifactBundleConfig, ModelCard
from skulk.shared.models.native_music_layout import NativeMusicTensorLayout
from skulk.shared.models.native_music_runtime import (
    NativeMusicModelPolicy,
    NativeMusicTensorState,
    native_music_model_identity,
    native_music_tensor_state,
)
from skulk.shared.types.audio_separation import (
    MAX_SEPARATION_DECODED_BYTES,
    MAX_SEPARATION_SOURCE_BYTES,
)
from skulk.shared.types.native_music import (
    NATIVE_MUSIC_MAX_FRAMES,
    NATIVE_MUSIC_MIN_DECODE_FRAMES,
    NativeMusicDevice,
    NativeMusicMemoryCapacity,
    native_music_conditioning_frames,
)
from skulk.utils.pydantic_ext import FrozenModel

_GIB: Final = 1024**3
_NATIVE_LIBRARIES: Final = 2 * _GIB
_FULL_WORKSPACE: Final = 8 * _GIB
_MIN_WORKSPACE: Final = 2 * _GIB
_ROUNDING: Final = 64 * 1024**2
_CAPACITY_MAX_AGE_MS: Final = 120_000
_CAPACITY_CLOCK_SKEW_MS: Final = 30_000
NativeMusicMemoryAssumption = Literal[
    "workspace_allowance_unqualified",
    "tensor_layout_unknown",
    "nonfloating_layout_upper_bound",
]


@final
class NativeMusicMemoryEstimate(FrozenModel):
    """Separate physical allocations at load and forward, never a hardware claim.

    CPU retained tensor state is charged only to host memory; GPU retained state
    is charged only to its device. Unified memory sums distinct host/device
    storage within the same phase, then takes the phase maximum. It never adds
    independently timed host and device peaks or counts a shared weight twice.
    """

    policy_version: Literal[1] = Field(
        default=1, description="Native peak-policy version."
    )
    projected_tensor_bytes: int = Field(
        ge=0,
        description="All three cast component states, or an explicit conservative upper projection.",
    )
    host_load_bytes: int = Field(
        ge=0,
        description="Host library, source/cast-copy and media reservation while loading.",
    )
    device_load_bytes: int = Field(
        ge=0,
        description="One actual device's retained state and temporary cast reservation while loading.",
    )
    host_forward_bytes: int = Field(
        ge=0,
        description="Host forward/media reservation, including permitted temporary CPU VAE decode.",
    )
    device_forward_bytes: int = Field(
        ge=0,
        description="One actual device's retained state and bounded forward workspace.",
    )
    workspace_bytes: int = Field(
        ge=0,
        description="Explicit unqualified conservative forward allowance for the mounted duration ceiling.",
    )
    assumptions: tuple[NativeMusicMemoryAssumption, ...] = Field(
        min_length=1,
        max_length=3,
        description="Visible unknowns that qualification or exact header observation must reassess.",
    )

    @property
    def host_peak_bytes(self) -> int:
        """Return the largest host reservation across the two execution phases."""
        return max(self.host_load_bytes, self.host_forward_bytes)

    @property
    def device_peak_bytes(self) -> int:
        """Return the largest reservation for the selected device, never a GPU sum."""
        return max(self.device_load_bytes, self.device_forward_bytes)

    @property
    def unified_peak_bytes(self) -> int:
        """Combine distinct simultaneous storage once, without adding separate peaks."""
        return max(
            self.host_load_bytes + self.device_load_bytes,
            self.host_forward_bytes + self.device_forward_bytes,
        )


@final
class NativeMusicMemoryReservation(FrozenModel):
    """Hold the entire execution peak, crediting only verified current allocations.

    A loaded model's weights do not stand for its future forward workspace.
    Default zero credits are safe for pending work and missing observations.
    A caller may credit only exclusive allocations already included in the same
    capacity sample, never estimated tensor sizes or another instance's pages.
    Shared credit measures unique physical storage, not a sum of aliases.
    """

    estimate: NativeMusicMemoryEstimate = Field(
        description="The admitted complete loading/forward peak held by this instance."
    )
    reflected_at_unix_ms: int = Field(
        default=0,
        ge=0,
        description="Exact capacity snapshot containing the verified allocations; zero grants no credit.",
    )
    reflected_host_bytes: int = Field(
        default=0,
        ge=0,
        description="Verified exclusive host allocation already reflected by that snapshot.",
    )
    reflected_device_bytes: int = Field(
        default=0,
        ge=0,
        description="Verified allocation on the selected logical device zero reflected by that snapshot.",
    )
    reflected_shared_bytes: int = Field(
        default=0,
        ge=0,
        description="Verified unique physical allocation in the shared pool reflected by that snapshot.",
    )


def reserve_native_music_memory(
    capacity: NativeMusicMemoryCapacity,
    committed: Iterable[NativeMusicMemoryReservation],
) -> NativeMusicMemoryCapacity:
    """Reserve every still-unallocated peak against the current observed free pool.

    ``committed`` includes pending and loaded instances exactly once. A loaded
    state alone earns no credit: only exclusive allocations verified in this
    exact capacity snapshot reduce its outstanding reservation. Stale or unknown
    allocation readings conservatively reserve the full peak. Each domain also
    obeys total capacity minus all commitments, and no credit exceeds its peak.
    The caller owns instance identity/deduplication and allocation attribution.
    """
    held = tuple(committed)

    def credit(item: NativeMusicMemoryReservation, reflected: int, peak: int) -> int:
        return (
            min(reflected, peak)
            if capacity.sampled_at_unix_ms > 0
            and item.reflected_at_unix_ms == capacity.sampled_at_unix_ms
            else 0
        )

    host = max(
        0,
        min(
            capacity.host_available_bytes
            - sum(
                item.estimate.host_peak_bytes
                - credit(item, item.reflected_host_bytes, item.estimate.host_peak_bytes)
                for item in held
            ),
            capacity.host_total_bytes
            - sum(item.estimate.host_peak_bytes for item in held),
        ),
    )
    device = capacity.device_available_bytes
    if device is not None and capacity.device_total_bytes is not None:
        device = max(
            0,
            min(
                device
                - sum(
                    item.estimate.device_peak_bytes
                    - credit(
                        item,
                        item.reflected_device_bytes,
                        item.estimate.device_peak_bytes,
                    )
                    for item in held
                ),
                capacity.device_total_bytes
                - sum(item.estimate.device_peak_bytes for item in held),
            ),
        )
    shared = capacity.shared_available_bytes
    if shared is not None and capacity.shared_total_bytes is not None:
        shared = max(
            0,
            min(
                shared
                - sum(
                    item.estimate.unified_peak_bytes
                    - credit(
                        item,
                        item.reflected_shared_bytes,
                        item.estimate.unified_peak_bytes,
                    )
                    for item in held
                ),
                capacity.shared_total_bytes
                - sum(item.estimate.unified_peak_bytes for item in held),
            ),
        )
    return capacity.model_copy(
        update={
            "host_available_bytes": host,
            "device_available_bytes": device,
            "shared_available_bytes": shared,
        }
    )


def native_music_memory_error(
    estimate: NativeMusicMemoryEstimate,
    capacity: NativeMusicMemoryCapacity,
    *,
    device: NativeMusicDevice,
    now_unix_ms: int,
) -> str | None:
    """Return a shared actionable refusal, requiring every actual memory domain.

    An explicit CPU workload never borrows a GPU pool. GPU workloads satisfy
    both host staging/fallback and selected-device requirements. Established
    unified memory additionally satisfies the physical combined phase peak.
    Missing device observations are unknown capacity, never an invented sum.
    """
    sampled = capacity.sampled_at_unix_ms
    if now_unix_ms <= 0 or sampled <= 0:
        return "Native music needs a timestamped current memory observation"
    if sampled - now_unix_ms > _CAPACITY_CLOCK_SKEW_MS:
        return "Native music memory observation is ahead of the local clock; check node clocks and refresh resources"
    if now_unix_ms - sampled > _CAPACITY_MAX_AGE_MS:
        return "Native music memory observation is stale; refresh native resources before placement"
    if estimate.host_peak_bytes > capacity.host_available_bytes:
        return "Native music needs more host memory for loading, media and forward execution"
    if (
        capacity.shared_available_bytes is not None
        and estimate.unified_peak_bytes > capacity.shared_available_bytes
    ):
        return "Native music needs more unified physical memory for simultaneous host/device work"
    if device == "cpu":
        if estimate.device_peak_bytes != 0:
            return "Native music CPU admission received a GPU execution estimate"
        return None
    if estimate.device_peak_bytes == 0:
        return "Native music GPU admission received a CPU execution estimate"
    if capacity.device_index != 0 or capacity.device_available_bytes is None:
        return "Native music needs current memory for its actual selected device zero"
    if capacity.device_memory_kind is None:
        return (
            "Native music needs the actual device's discrete or unified memory topology"
        )
    if device == "mps" and capacity.device_memory_kind != "unified":
        return "Native music Metal requires its unified physical-memory observation"
    if estimate.device_peak_bytes > capacity.device_available_bytes:
        return "Native music needs more memory on its actual selected device zero"
    return None


def _round_up(value: int) -> int:
    return -(-value // _ROUNDING) * _ROUNDING


def _tensor_file_bytes(bundle: ArtifactBundleConfig) -> int:
    selected = [
        item.size_bytes for item in bundle.files if item.path.endswith(".safetensors")
    ]
    if len(selected) != 1:
        raise ValueError(
            "native music memory needs one complete tensor object per role"
        )
    return selected[0]


def _upper_state(layout: NativeMusicTensorLayout | None, file_bytes: int) -> int:
    # Without a header, the smallest supported serialized scalar is four bits.
    # Eight bytes per scalar also covers integer/complex buffers that cannot be
    # projected by the floating-state helper. Exact observed headers replace
    # this deliberately pessimistic assumption; it is not a compatibility ban.
    return (
        sum(group.element_count for group in layout.dtypes) * 8
        if layout is not None
        else file_bytes * 16
    )


def estimate_native_music_peak(
    card: ModelCard,
    policy: NativeMusicModelPolicy,
) -> NativeMusicMemoryEstimate:
    """Project complete one-model/batch-one native load and forward reservations.

    Args:
        card: Complete immutable native base, text-encoder and autoencoder data.
        policy: Actual library-probed component precision and mounted duration.

    Returns:
        Phase-specific host/device costs and explicit estimation assumptions.
        CPU reservations do not borrow GPU capacity. Discrete GPU admission must
        satisfy both domains; unified admission uses ``unified_peak_bytes``.

    The pinned base loader receives dtype explicitly. Encoder and VAE construct
    at FP32, move to device and only then cast; reserve the old device copy when
    a narrower cast is required. Source/mapped bytes and a second component copy
    cover loading transients. GPU host forward includes VAE/forward allowance
    because upstream may temporarily fall back to CPU below its free-VRAM floor.
    These estimates do not activate placement or replace actual resource guards.
    """
    profile, base = card.music, card.artifact_bundle
    native = profile.native if profile is not None else None
    if profile is None or native is None or base is None:
        raise ValueError("native music memory requires all three selected data roles")
    if (
        policy.model_identity != native_music_model_identity(card)
        or policy.max_target_frames != min(120, profile.max_seconds) * 48000
        or policy.threads != 2
    ):
        raise ValueError(
            "native music memory policy differs from its selected card/workload"
        )
    # Revalidate saved/copy-updated contracts too; model_copy does not validate.
    validated = NativeMusicModelPolicy.model_validate_json(policy.model_dump_json())
    if validated != policy:
        raise ValueError("native music memory policy is not canonical")
    if (
        native.tensor_layouts is not None
        and native.tensor_layouts != policy.tensor_layouts
    ):
        raise ValueError("native music memory policy contradicts declared tensor facts")
    bundles = (
        base,
        native.text_encoder.artifact_bundle,
        native.autoencoder.artifact_bundle,
    )
    stored = tuple(_tensor_file_bytes(bundle) for bundle in bundles)
    layouts = policy.tensor_layouts
    if layouts is not None and any(
        layout.header_bytes + layout.stored_tensor_bytes != file_bytes
        for layout, file_bytes in zip(
            (layouts.base, layouts.text_encoder, layouts.autoencoder),
            stored,
            strict=True,
        )
    ):
        raise ValueError(
            "native music tensor facts do not cover the selected complete objects"
        )
    known = native_music_tensor_state(policy)
    assumptions: tuple[NativeMusicMemoryAssumption, ...] = (
        "workspace_allowance_unqualified",
    )
    if known is None:
        assumptions += (
            "tensor_layout_unknown"
            if layouts is None
            else "nonfloating_layout_upper_bound",
        )
        state = NativeMusicTensorState(
            base=_upper_state(layouts.base if layouts else None, stored[0]),
            text_encoder=_upper_state(
                layouts.text_encoder if layouts else None, stored[1]
            ),
            autoencoder=_upper_state(
                layouts.autoencoder if layouts else None, stored[2]
            ),
        )
    else:
        state = known
    components = (state.base, state.text_encoder, state.autoencoder)
    if layouts is not None:
        initial_encoder = (
            sum(group.element_count for group in layouts.text_encoder.dtypes) * 4
        )
        initial_vae = (
            sum(group.element_count for group in layouts.autoencoder.dtypes) * 4
        )
        if known is None:
            initial_encoder = max(initial_encoder, state.text_encoder)
            initial_vae = max(initial_vae, state.autoencoder)
    else:
        initial_encoder, initial_vae = state.text_encoder, state.autoencoder
    loader_states = (state.base, initial_encoder, initial_vae)
    frames = max(
        NATIVE_MUSIC_MIN_DECODE_FRAMES,
        native_music_conditioning_frames(policy.max_target_frames),
    )
    workspace = _round_up(
        max(_MIN_WORKSPACE, -(-_FULL_WORKSPACE * frames // NATIVE_MUSIC_MAX_FRAMES))
    )
    # All encoded/decoded originals, canonical copies, result/verification and
    # transfer buffers coexist with a model. Use the actual protocol ceilings,
    # not only the shorter output duration, for the admitted uploaded original.
    stereo_float = frames * 2 * 4
    host_media = (
        3 * MAX_SEPARATION_SOURCE_BYTES
        + 2 * MAX_SEPARATION_DECODED_BYTES
        + 8 * stereo_float
    )
    device_media = 2 * stereo_float
    auxiliary = (
        sum(
            item.size_bytes
            for bundle in bundles
            for item in bundle.files
            if not item.path.endswith(".safetensors")
        )
        * 4
    )
    common_host = _NATIVE_LIBRARIES + host_media + auxiliary
    loading_copy = max(
        source + max(cast, final)
        for source, cast, final in zip(stored, loader_states, components, strict=True)
    )
    if policy.execution.device == "cpu":
        return NativeMusicMemoryEstimate(
            projected_tensor_bytes=state.total_bytes,
            host_load_bytes=_round_up(common_host + state.total_bytes + loading_copy),
            device_load_bytes=0,
            host_forward_bytes=_round_up(common_host + state.total_bytes + workspace),
            device_forward_bytes=0,
            workspace_bytes=workspace,
            assumptions=assumptions,
        )
    cast_transient = max(
        initial_encoder if policy.execution.text_encoder_precision != "float32" else 0,
        initial_vae if policy.execution.autoencoder_precision != "float32" else 0,
    )
    # MPS explicitly skips upstream's low-VRAM CPU fallback. CUDA/HIP can move
    # the VAE to CPU and restore it before emitting completion; post-run device
    # equality alone cannot establish the peak of that temporary execution.
    cpu_decode = (initial_vae + workspace) if policy.execution.device != "mps" else 0
    return NativeMusicMemoryEstimate(
        projected_tensor_bytes=state.total_bytes,
        host_load_bytes=_round_up(common_host + loading_copy),
        device_load_bytes=_round_up(state.total_bytes + cast_transient + auxiliary),
        host_forward_bytes=_round_up(common_host + cpu_decode),
        device_forward_bytes=_round_up(
            state.total_bytes + workspace + device_media + auxiliary
        ),
        workspace_bytes=workspace,
        assumptions=assumptions,
    )
