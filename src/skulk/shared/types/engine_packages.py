"""What a node reports about engine packages: the ones it holds and the ones it may install."""

from typing import Final, Literal

from pydantic import ConfigDict, Field

from skulk.utils.pydantic_ext import CamelCaseModel

MAX_NODE_ENGINE_PACKAGES: Final = 64
"""Most package lanes one node reports. A node holds a handful; the bound keeps
a damaged packages folder from growing its telemetry."""

MAX_REPORTED_ENGINE_BUILDS: Final = 16
"""Most installed builds reported for one lane."""

_DIGEST: Final = r"^[0-9a-f]{64}$"


class NodeEngineLoadResult(CamelCaseModel):
    """How the last model load on one engine package build ended on one node.

    Evidence about one build and one model, kept by the node since it
    started. It never makes a node more or less eligible for anything.
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
