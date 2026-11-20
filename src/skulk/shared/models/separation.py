"""Model-intrinsic source separation metadata, independent of serving policy."""

from pathlib import PurePosixPath
from typing import Literal, Self, cast, final

from pydantic import Field, PositiveInt, field_validator, model_validator

from skulk.shared.types.audio_separation import SEPARATION_STEMS, SeparationStemName
from skulk.utils.pydantic_ext import FrozenModel


def separation_is_absent(value: object) -> bool:
    """Return whether optional separator metadata must be omitted from serialization.

    Keeping absent metadata out of existing cards preserves the complete bytes
    bound by installed-artifact identities and authorization receipts.
    """
    return value is None


@final
class SeparationCardConfig(FrozenModel):
    """Native HTDemucs architecture, training geometry and ordered output roles."""

    family: Literal["htdemucs"] = Field(description="Explicit separator model family.")
    stems: tuple[SeparationStemName, ...] = Field(
        min_length=4,
        max_length=4,
        description="Ordered native roles, before any mix or export processing.",
    )
    sample_rate: Literal[44100] = Field(description="Native sampling rate in Hz.")
    audio_channels: Literal[2] = Field(description="Native channels in each stem.")
    weights_file: str = Field(
        min_length=1,
        description="Selected safetensors path relative to the artifact loader root.",
    )
    encoder_depth: PositiveInt = Field(description="Hybrid encoder's actual depth.")
    transformer_layers_per_branch: PositiveInt = Field(
        description="Transformer layers per time/frequency branch, not their sum."
    )
    transformer_channels: PositiveInt = Field(
        description="Actual transformer width, including bottom-channel projection."
    )
    trained_segment_seconds: float = Field(
        gt=0,
        allow_inf_nan=False,
        description="Training window, independent of API duration or split limits.",
    )

    @field_validator("stems", mode="before")
    @classmethod
    def coerce_stems(cls, value: object) -> object:
        """Accept persisted JSON/TOML arrays as the immutable output-role tuple."""
        return tuple(cast("list[object]", value)) if isinstance(value, list) else value

    @field_validator("weights_file")
    @classmethod
    def validate_weights_file(cls, value: str) -> str:
        """Keep the selected non-executable tensor file inside its loader root."""
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or value != path.as_posix()
            or "\\" in value
            or ".." in path.parts
            or not value.endswith(".safetensors")
        ):
            raise ValueError("separation weights_file must be a relative safetensors path")
        return value

    @model_validator(mode="after")
    def validate_stems(self) -> Self:
        """Bind the HTDemucs outputs to their native ordering."""
        if self.stems != SEPARATION_STEMS:
            raise ValueError("HTDemucs requires ordered drums, bass, other, vocals stems")
        return self
