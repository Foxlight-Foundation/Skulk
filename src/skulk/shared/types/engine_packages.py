"""What a node reports about engine packages: the ones it holds and the ones it may install."""

from typing import Final, Literal

from pydantic import ConfigDict, Field

from skulk.utils.pydantic_ext import CamelCaseModel

MAX_NODE_ENGINE_PACKAGES: Final = 64
"""Most package lanes one node reports. A node holds a handful; the bound keeps
a damaged packages folder from growing its telemetry."""

MAX_REPORTED_ENGINE_BUILDS: Final = 16
"""Most installed builds reported for one lane."""

MAX_REPORTED_UNSUPPORTED_MODELS: Final = 32
"""Most models one lane reports as refused by one of its builds."""

UNSUPPORTED_MODEL_CODE: Final = "unsupported_model"
"""The refusal code that says a build cannot load a model at all. It is the
one refusal a node remembers and placement acts on: it depends on the build
and the model only, so asking the same build again repeats it."""

_DIGEST: Final = r"^[0-9a-f]{64}$"

ENGINE_BUILD_PATTERN: Final = _DIGEST
"""Shape of a build identity: the SHA-256 of a package's signed manifest."""

ENGINE_ID_PATTERN: Final = r"^[a-z0-9]+(?:-[a-z0-9]+)*\.[a-z0-9]+(?:_[a-z0-9]+)*$"
"""Shape of ``publisher.engine``, the same one a signed manifest must have."""

EnginePackageLane = Literal["metal", "cuda", "rocm", "vulkan", "cpu"]
"""The accelerator classes a package build may execute on."""

EnginePackageAction = Literal["activate", "rollback", "remove"]
"""What a person may ask a node to do with a package it holds."""


class NodeEngineLoadResult(CamelCaseModel):
    """How the last model load on one engine package build ended on one node.

    Evidence about one build and one model, kept by the node since it
    started. As a lane's last load result it never makes a node more or less
    eligible for anything; the same record under ``unsupported_models`` does.
    """

    model_config = ConfigDict(frozen=True)

    build: str = Field(
        pattern=_DIGEST, description="Manifest digest of the build that was loaded."
    )
    outcome: Literal["loaded", "refused", "failed"] = Field(
        description=(
            "loaded: the model became ready. refused: the engine answered the "
            "load with a typed refusal; see code. failed: the runner ended "
            "while loading for another reason, which the node's log holds."
        )
    )
    code: str | None = Field(
        default=None,
        max_length=64,
        pattern=r"^[a-z_]+$",
        description="The engine's typed refusal code. Present only for refused.",
    )
    model_id: str = Field(
        min_length=1, max_length=256, description="The model that was loaded."
    )
    at_unix_ms: int = Field(
        ge=0, description="When the load ended, by the node's own clock."
    )


class NodeEnginePackage(CamelCaseModel):
    """One engine package lane on one node.

    A lane is one engine ID on one accelerator class, for example
    ``foxlight.tensorfold`` on ``rocm``. The node reports it when it holds a
    build for the lane, or when the package list it last verified offers one
    it could install. Nothing here is a path, and nothing here promises that
    a model will load: only a load answers that.
    """

    model_config = ConfigDict(frozen=True)

    engine_id: str = Field(
        min_length=1, max_length=64, description="The publisher.engine identity."
    )
    lane: str = Field(
        min_length=1,
        max_length=16,
        description="Accelerator class of the build: metal, cuda, rocm, vulkan or cpu.",
    )
    backend_tag: str = Field(
        min_length=1,
        max_length=96,
        description="The backend tag placements are stamped with, such as tensorfold-rocm.",
    )
    status: Literal["active", "unusable", "offered"] = Field(
        description=(
            "active: the node holds a usable build and new launches use it. "
            "unusable: the node holds a build it cannot offer; see problem. "
            "offered: the node holds none and may install this build when a "
            "model placed there needs it."
        )
    )
    build: str | None = Field(
        default=None,
        pattern=_DIGEST,
        description=(
            "Manifest digest of the active or offered build, which is the "
            "build identity stamped on a placement. Absent when the node "
            "cannot tell which build is active."
        ),
    )
    engine_version: str | None = Field(
        default=None,
        max_length=64,
        description="Version of the engine inside the active or offered build.",
    )
    generation: int | None = Field(
        default=None,
        ge=1,
        description="Package generation of the active or offered build.",
    )
    publisher: str | None = Field(
        default=None,
        max_length=64,
        description="Trust ID of the publisher that signed the active build.",
    )
    channel: Literal["stable", "candidate"] | None = Field(
        default=None,
        description="How far an offered build is promoted. Absent for a held build.",
    )
    installed_builds: tuple[str, ...] = Field(
        default=(),
        max_length=MAX_REPORTED_ENGINE_BUILDS,
        description="Manifest digests of every build of the lane on the node's disk.",
    )
    rollback_build: str | None = Field(
        default=None,
        pattern=_DIGEST,
        description="The build a rollback would return to, when one is kept.",
    )
    problem: str | None = Field(
        default=None,
        max_length=64,
        description="Why an unusable build is not offered: a short refusal code.",
    )
    download_bytes: int | None = Field(
        default=None,
        ge=0,
        description="Bytes an offered build downloads when it is installed.",
    )
    overhead_bytes: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Memory the active or offered build declares beyond a model's "
            "weight bytes, which admission counts for a launch."
        ),
    )
    load_results: tuple[NodeEngineLoadResult, ...] = Field(
        default=(),
        max_length=MAX_REPORTED_ENGINE_BUILDS,
        description=(
            "How the last model load on each build of the lane ended since "
            "the node started, newest first. Empty until a model is loaded."
        ),
    )
    unsupported_models: tuple[NodeEngineLoadResult, ...] = Field(
        default=(),
        max_length=MAX_REPORTED_UNSUPPORTED_MODELS,
        description=(
            "Models a build of the lane refused to load as unsupported since "
            "the node started, newest first. Placement reads this: a build "
            "that refused a model is not given that model again, on any "
            "node, so the next engine in the model's order serves it. An "
            "entry leaves when the node restarts, when the build leaves the "
            "node, or when the build later loads the model."
        ),
    )
