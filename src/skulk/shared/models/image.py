"""Immutable image workload contracts independent of a particular runner."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import PurePosixPath
from typing import Annotated, Literal, cast, final

from pydantic import Field, field_validator, model_validator

from skulk.utils.pydantic_ext import FrozenModel

ImageFamily = Literal[
    "qwen_image_21", "qwen_image_21_turbo", "flux_2_klein_4b", "z_image_turbo", "sdxl"
]
ImageComponentRole = Literal[
    "model",
    "diffusion",
    "vae",
    "text_encoder",
    "text_encoder_vision",
    "clip_l",
    "clip_g",
    "t5xxl",
]
_TURBO_SIGMAS = (
    1.0,
    0.978453,
    0.95418,
    0.926626,
    0.89508,
    0.845148,
    0.704534,
    0.414568,
    0.0,
)


@final
class ImageComponentFile(FrozenModel):
    """One immutable file needed by a component, including all weight shards."""

    path: str = Field(description="Canonical repository-relative POSIX path.")
    size_bytes: int = Field(ge=0, description="Pinned upstream file size in bytes.")
    object_id: str = Field(
        pattern=r"^(?:sha256:[0-9a-f]{64}|git-sha1:[0-9a-f]{40})$",
        description="Immutable Hub object identity, checked during staging.",
    )

    @model_validator(mode="after")
    def canonical_path(self) -> ImageComponentFile:
        """Reject filesystem escapes and ambiguous component paths."""
        path = PurePosixPath(self.path)
        if (
            path.is_absolute()
            or path.as_posix() != self.path
            or "\\" in self.path
            or any(part in {"", ".", ".."} for part in self.path.split("/"))
        ):
            raise ValueError("image component paths must be canonical relative paths")
        return self


@final
class ImageComponent(FrozenModel):
    """Complete pinned component source and the entry file its engine loads."""

    role: ImageComponentRole = Field(
        description="Typed loader role, never an arbitrary CLI flag."
    )
    repository: str = Field(
        pattern=r"^[^/\s]+/[^/\s]+$",
        description="Hub repository hosting this component.",
    )
    revision: str = Field(
        pattern=r"^[0-9a-f]{40}$", description="Immutable Hub commit."
    )
    entry_file: str = Field(
        description="Path of the loader entry file among the selected files."
    )
    files: tuple[ImageComponentFile, ...] = Field(
        min_length=1, description="All files required by this component."
    )

    @field_validator("files", mode="before")
    @classmethod
    def wire_files(cls, value: object) -> object:
        """Accept the JSON/TOML array representation without weakening strict file validation."""
        return tuple(cast("list[object]", value)) if isinstance(value, list) else value

    @model_validator(mode="after")
    def complete_entry(self) -> ImageComponent:
        """Require an unambiguous entry in the complete immutable file set."""
        paths = [item.path for item in self.files]
        if len(paths) != len(set(paths)) or self.entry_file not in paths:
            raise ValueError("image component entry must belong to a unique file set")
        return self


@final
class ImageCardConfig(FrozenModel):
    """Image geometry, component completeness, sampling, and input/output truth."""

    family: ImageFamily = Field(
        description="Intrinsic image architecture/checkpoint family."
    )
    components: tuple[ImageComponent, ...] = Field(
        min_length=1, description="Complete immutable component sources."
    )
    default_width: int = Field(
        default=1024, ge=64, le=4096, description="Width used for size=auto."
    )
    default_height: int = Field(
        default=1024, ge=64, le=4096, description="Height used for size=auto."
    )
    dimension_multiple: int = Field(
        default=32,
        ge=1,
        le=256,
        description="Required alignment of each canvas dimension.",
    )
    max_pixels: int = Field(
        default=4_400_000,
        ge=4096,
        le=16_777_216,
        description="Maximum qualified canvas area.",
    )
    max_reference_images: int = Field(
        default=0,
        ge=0,
        le=10,
        description="Maximum supported ordered reference images.",
    )
    supports_alpha: bool = Field(
        default=False, description="Whether the model produces and consumes RGBA."
    )
    default_steps: int = Field(
        ge=1, le=100, description="Recommended sampling steps for this checkpoint."
    )
    default_guidance: float = Field(
        default=1.0,
        ge=1.0,
        le=20.0,
        description="Recommended classifier-free guidance.",
    )
    fixed_sigmas: tuple[Annotated[float, Field(ge=0.0, le=1.0)], ...] = Field(
        default=(),
        description="Fixed sampling grid including terminal zero; empty selects the normal family schedule.",
    )

    @field_validator("components", "fixed_sigmas", mode="before")
    @classmethod
    def wire_sequences(cls, value: object) -> object:
        """Restore immutable sequences from the JSON/TOML transport representation."""
        return tuple(cast("list[object]", value)) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_contract(self) -> ImageCardConfig:
        """Keep component roles, geometry, and distilled sampling mutually consistent."""
        roles = [component.role for component in self.components]
        if len(roles) != len(set(roles)):
            raise ValueError("image component roles must be unique")
        identities: dict[tuple[str, str, str], tuple[int, str]] = {}
        for component in self.components:
            for item in component.files:
                identity = (component.repository, component.revision, item.path)
                value = (item.size_bytes, item.object_id)
                if identity in identities and identities[identity] != value:
                    raise ValueError(
                        "shared image files have conflicting immutable identities"
                    )
                identities[identity] = value
        required = (
            {"model"} if self.family == "sdxl" else {"diffusion", "vae", "text_encoder"}
        )
        if not required.issubset(roles):
            raise ValueError("image card lacks required components")
        if (
            self.max_reference_images
            and self.family.startswith("qwen_image_21")
            and "text_encoder_vision" not in roles
        ):
            raise ValueError(
                "Qwen reference editing requires a pinned vision component"
            )
        self.validate_canvas(self.default_width, self.default_height)
        if self.fixed_sigmas and (
            len(self.fixed_sigmas) != self.default_steps + 1
            or self.fixed_sigmas[-1] != 0.0
            or any(
                left <= right
                for left, right in zip(
                    self.fixed_sigmas, self.fixed_sigmas[1:], strict=False
                )
            )
        ):
            raise ValueError(
                "fixed sampling sigmas must descend to zero and agree with steps"
            )
        if self.family == "qwen_image_21_turbo" and (
            self.default_steps != 8
            or self.default_guidance != 1.0
            or self.fixed_sigmas != _TURBO_SIGMAS
        ):
            raise ValueError(
                "Qwen Image 2.1 Turbo requires its exact eight-step CFG=1 schedule"
            )
        return self

    def validate_canvas(self, width: int, height: int) -> None:
        """Validate a canvas before admitting GPU work; raise ValueError on invalid geometry."""
        if not (64 <= width <= 4096 and 64 <= height <= 4096) or (
            width % self.dimension_multiple
            or height % self.dimension_multiple
            or width * height > self.max_pixels
        ):
            raise ValueError(
                "image canvas exceeds this model's area or alignment limits"
            )

    @property
    def storage_bytes(self) -> int:
        """Return complete component storage, deduplicating shared immutable files."""
        return sum(
            {
                (component.repository, component.revision, item.path): item.size_bytes
                for component in self.components
                for item in component.files
            }.values()
        )


def validate_reference_images(references: Sequence[bytes]) -> None:
    """Verify bounded encoded image headers before transport or native decoding.

    Pillow verifies containers without allocating full decoded pixels. Aggregate
    pixels bound later native decoding even for highly compressed input.
    """
    import io

    from PIL import Image, UnidentifiedImageError

    pixels = 0
    try:
        for reference in references:
            with Image.open(io.BytesIO(reference)) as image:
                pixels += image.width * image.height
                if pixels > 16_777_216:
                    raise ValueError("Reference images exceed the decoded pixel limit")
                if image.format not in {"PNG", "JPEG", "WEBP"}:
                    raise ValueError("Reference images must be PNG, JPEG, or WebP")
                image.verify()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError(
            "Reference image container is invalid or exceeds its pixel limit"
        ) from error
