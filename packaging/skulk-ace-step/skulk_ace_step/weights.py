"""Verify complete native weights and prepare runtime-owned local model code."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal, Self, final

from pydantic import ConfigDict, Field, model_validator

from .contracts import NativeMusicComponent, NativeMusicContract, NativeMusicWeights


class _BaseConfiguration(NativeMusicContract):
    """Known model geometry and local-only dynamic code mapping before any loader."""

    model_config = ConfigDict(frozen=True, strict=True, extra="ignore")
    model_type: Literal["acestep"]
    hidden_size: Literal[2048]
    num_hidden_layers: Literal[24]
    architectures: tuple[Literal["AceStepConditionGenerationModel"], ...] = Field(
        min_length=1, max_length=1
    )
    auto_map: dict[str, str] = Field(
        description="Exact local runtime-owned implementation classes."
    )

    @model_validator(mode="after")
    def validate_local_code(self) -> Self:
        """Refuse external repositories and alternative model Python entry points."""
        if self.auto_map != {
            "AutoConfig": "configuration_acestep_v15.AceStepConfig",
            "AutoModel": "modeling_acestep_v15_base.AceStepConditionGenerationModel",
        }:
            raise ValueError(
                "native music base configuration must select only reviewed local runtime code"
            )
        return self


@final
class ModelCodeFile(NativeMusicContract):
    """One reviewed implementation file from the sealed runtime, never the Hub."""

    source: Literal[
        "acestep/models/base/configuration_acestep_v15.py",
        "acestep/models/base/modeling_acestep_v15_base.py",
        "acestep/models/base/apg_guidance.py",
    ] = Field(
        description="Known runtime-library source path beneath its verified import root."
    )
    destination: Literal[
        "configuration_acestep_v15.py",
        "modeling_acestep_v15_base.py",
        "apg_guidance.py",
    ] = Field(description="Exact local module name read by the pinned model loader.")
    size_bytes: int = Field(
        gt=0, le=2 * 1024**2, description="Exact reviewed source-file length."
    )
    sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="Source digest retained inside the sealed package manifest.",
    )

    @model_validator(mode="after")
    def validate_mapping(self) -> Self:
        """Refuse cross-assignment of model configuration and implementation files."""
        if self.source != "acestep/models/base/" + self.destination:
            raise ValueError(
                "native music runtime code mapping differs from its reviewed role"
            )
        return self


def _copy_verified(
    source: Path,
    target: Path,
    size: int,
    object_id: str,
    checkpoint: Callable[[], None],
) -> None:
    """Copy verified bytes once into a fresh private inode, without a hash/copy race."""
    checkpoint()
    descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as incoming:
        facts = os.fstat(incoming.fileno())
        if not stat.S_ISREG(facts.st_mode) or facts.st_size != size:
            raise ValueError(
                "native music component is not its selected regular object"
            )
        algorithm, digest = object_id.split(":", 1)
        hasher = (
            hashlib.sha256()
            if algorithm == "sha256"
            else hashlib.sha1(usedforsecurity=False)
        )
        if algorithm == "git-sha1":
            hasher.update(f"blob {size}\0".encode())
        written = 0
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as outgoing:
            while block := incoming.read(1024 * 1024):
                checkpoint()
                written += len(block)
                if written > size:
                    raise ValueError(
                        "native music component changed during preparation"
                    )
                hasher.update(block)
                outgoing.write(block)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        if written != size or hasher.hexdigest() != digest:
            raise ValueError(
                "native music component bytes differ from the selected object"
            )
    checkpoint()


def _copy_component(
    component: NativeMusicComponent, target: Path, checkpoint: Callable[[], None]
) -> None:
    target.mkdir(mode=0o700)
    prefix = component.root + "/" if component.root is not None else ""
    for item in component.files:
        name = item.path.removeprefix(prefix)
        # The strict role contract has already reduced this to the exact flat
        # loader inventory. Do not recursively copy unrelated repository files.
        if "/" in name:
            raise ValueError(
                "native music selected component has an unexpected nested file"
            )
        _copy_verified(
            Path(component.directory) / item.path,
            target / name,
            item.size_bytes,
            item.object_id,
            checkpoint,
        )


def prepare_weights(
    target: Path,
    weights: NativeMusicWeights,
    runtime_root: Path,
    code: Sequence[ModelCodeFile],
    *,
    checkpoint: Callable[[], None] = lambda: None,
) -> Path:
    """Prepare one complete private loader view without modifying model-store bytes.

    The caller owns the fresh lifetime and retains partial preparation until its
    process has drained. Existing targets are never reused, replaced or treated
    as receipts. Return only after all fourteen data files and three reviewed
    implementation modules have been copied and verified. The model loader is
    not called by this function.
    """
    checkpoint()
    expected_code = {
        "configuration_acestep_v15.py",
        "modeling_acestep_v15_base.py",
        "apg_guidance.py",
    }
    if len(code) != 3 or {item.destination for item in code} != expected_code:
        raise ValueError(
            "native music requires exactly its three reviewed implementation modules"
        )
    target.mkdir(mode=0o700)
    _copy_component(weights.base, target / "acestep-v15-base", checkpoint)
    _copy_component(weights.text_encoder, target / "Qwen3-Embedding-0.6B", checkpoint)
    _copy_component(weights.autoencoder, target / "vae", checkpoint)
    with (target / "acestep-v15-base/config.json").open("rb") as stream:
        configuration = stream.read(64 * 1024 + 1)
    if len(configuration) > 64 * 1024:
        raise ValueError(
            "native music base configuration exceeds its bounded metadata size"
        )
    # Upstream uses trust_remote_code=True for a local dynamic class. The
    # mapping itself must therefore be checked before its loader can run.
    _ = _BaseConfiguration.model_validate_json(configuration)
    for item in code:
        _copy_verified(
            runtime_root / item.source,
            target / "acestep-v15-base" / item.destination,
            item.size_bytes,
            "sha256:" + item.sha256,
            checkpoint,
        )
    checkpoint()
    return target
