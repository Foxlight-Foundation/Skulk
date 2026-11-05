"""Observe complete pinned native data and its bounded tensor headers in Core.

The native process independently hashes its fresh loader copies again. This
earlier observation lets admission bind serialized facts to actual selected
objects without importing Torch, loading weights, or trusting file size alone.
"""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, final

from skulk.shared.models.model_cards import ArtifactBundleConfig, ModelCard
from skulk.shared.models.native_music_layout import (
    MAX_NATIVE_MUSIC_HEADER_BYTES,
    NativeMusicTensorLayout,
    NativeMusicTensorLayouts,
    project_native_music_tensor_header,
)
from skulk.shared.types.native_music import (
    NativeMusicComponent,
    NativeMusicFile,
    NativeMusicWeights,
)


@final
@dataclass(frozen=True)
class ObservedMusicArtifacts:
    """Complete verified data selection and intrinsic geometry, never model ready."""

    weights: NativeMusicWeights
    tensor_layouts: NativeMusicTensorLayouts


def _open_selected(directory: Path, path: str) -> int:
    # Opening every relative ancestor rejects a redirected nested loader root,
    # not just a symlink at the final tensor. Store hard links remain valid.
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        parts = Path(path).parts
        for part in parts[:-1]:
            nested = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor
            )
            os.close(descriptor)
            descriptor = nested
        return os.open(
            parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor
        )
    finally:
        os.close(descriptor)


def _verify_selected(
    directory: Path,
    selected: NativeMusicFile,
    checkpoint: Callable[[], None],
) -> NativeMusicTensorLayout | None:
    checkpoint()
    descriptor = _open_selected(directory, selected.path)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size != selected.size_bytes:
            raise ValueError(
                "native music data is not its complete selected regular object"
            )
        algorithm, expected = selected.object_id.split(":", 1)
        digest = (
            hashlib.sha256()
            if algorithm == "sha256"
            else hashlib.sha1(usedforsecurity=False)
        )
        if algorithm == "git-sha1":
            digest.update(f"blob {selected.size_bytes}\0".encode())
        header = b""
        if selected.path.endswith(".safetensors"):
            word = stream.read(8)
            length = int.from_bytes(word, "little")
            if len(word) != 8 or not 2 <= length <= min(
                MAX_NATIVE_MUSIC_HEADER_BYTES - 8, selected.size_bytes - 8
            ):
                raise ValueError(
                    "native music selected tensor has an unbounded or incomplete header"
                )
            header = word + stream.read(length)
            if len(header) != length + 8:
                raise ValueError(
                    "native music selected tensor has an incomplete header"
                )
            digest.update(header)
        count = len(header)
        while block := stream.read(min(1024**2, selected.size_bytes + 1 - count)):
            count += len(block)
            digest.update(block)
            checkpoint()
            if count > selected.size_bytes:
                raise ValueError("native music selected data grew during observation")
        after = os.fstat(stream.fileno())
        if count != selected.size_bytes or digest.hexdigest() != expected:
            raise ValueError(
                "native music data differs from its selected object identity"
            )
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError("native music selected data changed during observation")
    checkpoint()
    return (
        project_native_music_tensor_header(header, file_size=selected.size_bytes)
        if header
        else None
    )


def observe_native_music_artifacts(
    card: ModelCard,
    sources: Mapping[tuple[str, str], Path],
    *,
    checkpoint: Callable[[], None] = lambda: None,
) -> ObservedMusicArtifacts:
    """Verify all fourteen selected files, projecting only bounded tensor prefixes.

    Args:
        card: Complete immutable native base/encoder/VAE selection.
        sources: Worker-resolved installation roots keyed by exact repository/revision.
            Shared-source roles use one complete installation, never partial roots.
        checkpoint: Cancellation, deadline and resource checkpoint between bounded reads.

    Returns:
        Exact private component paths and complete observed tensor facts. No file
        is copied, no native library is imported and no readiness is advertised.

    Raises:
        ValueError: Selection, actual data, header or complete source mapping differs.
        OSError: A selected ordinary file cannot be opened without redirection.
    """
    native = card.music.native if card.music is not None else None
    base = card.artifact_bundle
    if native is None or base is None or card.source_revision is None:
        raise ValueError("native music requires three immutable complete data roles")
    expected_sources = {
        (source.repository, source.revision)
        for source in card.native_music_artifact_sources()
    }
    if set(sources) != expected_sources or any(
        not path.is_absolute() for path in sources.values()
    ):
        raise ValueError(
            "native music observations require exactly the selected source roots"
        )
    roles: tuple[
        tuple[
            Literal["base", "text_encoder", "autoencoder"],
            str,
            str,
            ArtifactBundleConfig,
        ],
        ...,
    ] = (
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
    components: list[NativeMusicComponent] = []
    layouts: list[NativeMusicTensorLayout] = []
    for role, repository, revision, bundle in roles:
        selected_files: list[NativeMusicFile] = []
        for item in bundle.files:
            if item.object_id is None:
                raise ValueError(
                    "native music selected objects require complete identity pins"
                )
            selected_files.append(
                NativeMusicFile(
                    path=item.path, size_bytes=item.size_bytes, object_id=item.object_id
                )
            )
        component = NativeMusicComponent(
            kind=role,
            directory=str(sources[(repository, revision)]),
            root=bundle.root,
            files=tuple(selected_files),
        )
        observed: list[NativeMusicTensorLayout] = []
        for selected in component.files:
            layout = _verify_selected(Path(component.directory), selected, checkpoint)
            if layout is not None:
                observed.append(layout)
        if len(observed) != 1:
            raise ValueError(
                "native music requires one complete tensor layout per role"
            )
        components.append(component)
        layouts.append(observed[0])
    return ObservedMusicArtifacts(
        weights=NativeMusicWeights(
            base=components[0], text_encoder=components[1], autoencoder=components[2]
        ),
        tensor_layouts=NativeMusicTensorLayouts(
            base=layouts[0], text_encoder=layouts[1], autoencoder=layouts[2]
        ),
    )
