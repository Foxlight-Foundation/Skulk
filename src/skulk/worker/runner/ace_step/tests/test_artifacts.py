"""Real selected-file hashing and bounded header observation before native loads."""

import hashlib
import json
import os
from pathlib import Path
from typing import cast

import pytest

from skulk.shared.models.model_cards import ModelCard
from skulk.shared.models.native_music_runtime import resolve_native_music_policy
from skulk.shared.models.tests.test_native_music_layout import body, prefix, section
from skulk.shared.models.tests.test_native_music_runtime import probe
from skulk.worker.runner.ace_step.artifacts import observe_native_music_artifacts


def selected_fixture(
    root: Path, *, elements: int = 4, malformed: bool = False
) -> tuple[ModelCard, dict[tuple[str, str], Path]]:
    """Build tiny complete real files, retaining exact role/selection schema semantics."""
    document = body()
    native = section(document, "music", "native")
    selections = (
        (
            str(document["source_repository"]),
            str(document["source_revision"]),
            section(document, "artifact_bundle"),
        ),
        (
            str(section(native, "text_encoder")["repo"]),
            str(section(native, "text_encoder")["revision"]),
            section(native, "text_encoder", "artifact_bundle"),
        ),
        (
            str(section(native, "autoencoder")["repo"]),
            str(section(native, "autoencoder")["revision"]),
            section(native, "autoencoder", "artifact_bundle"),
        ),
    )
    sources: dict[tuple[str, str], Path] = {}
    total = 0
    for repository, revision, bundle in selections:
        key = repository, revision
        if key not in sources:
            sources[key] = root / str(len(sources))
            sources[key].mkdir(parents=True)
        size = 0
        for selected in cast("list[dict[str, object]]", bundle["files"]):
            path = sources[key] / str(selected["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".safetensors":
                header = prefix(
                    {
                        "weight": {
                            "dtype": "BF16",
                            "shape": [elements + int(malformed)],
                            "data_offsets": [0, elements * 2],
                        }
                    }
                )
                data = header + b"\x00\x00" * elements
            else:
                data = json.dumps({"controlled": path.name}).encode()
            path.write_bytes(data)
            selected["size_bytes"] = len(data)
            # Include Git's blob framing; a raw SHA-1 is not a Git object ID.
            selected["object_id"] = (
                "git-sha1:"
                + hashlib.sha1(
                    f"blob {len(data)}\0".encode() + data, usedforsecurity=False
                ).hexdigest()
                if str(selected["object_id"]).startswith("git-sha1:")
                else "sha256:" + hashlib.sha256(data).hexdigest()
            )
            size += len(data)
        bundle["download_size"] = size
        total += size
    section(document, "storage_size")["in_bytes"] = total
    return ModelCard.model_validate(document), sources


def test_all_roles_observed_from_complete_sources_without_mutating_store(
    tmp_path: Path,
) -> None:
    """Shared encoder/VAE installation and independent roots retain all fourteen pins."""
    card, sources = selected_fixture(tmp_path)
    before = {
        path: path.read_bytes()
        for root in sources.values()
        for path in root.rglob("*")
        if path.is_file()
    }
    observed = observe_native_music_artifacts(card, sources)
    assert len(before) == 14
    assert {path: path.read_bytes() for path in before} == before
    assert (
        observed.weights.text_encoder.directory
        == observed.weights.autoencoder.directory
    )
    assert observed.weights.text_encoder.root != observed.weights.autoencoder.root
    assert len(observed.weights.base.files) == 3
    assert len(observed.weights.text_encoder.files) == 9
    assert len(observed.weights.autoencoder.files) == 2
    for layout in (
        observed.tensor_layouts.base,
        observed.tensor_layouts.text_encoder,
        observed.tensor_layouts.autoencoder,
    ):
        assert layout.stored_tensor_bytes == 8
        assert layout.dtypes[0].element_count == 4
    policy = resolve_native_music_policy(
        card, "ace_step-cpu", probe(), observed_layouts=observed.tensor_layouts
    )
    assert policy.tensor_layouts == observed.tensor_layouts


@pytest.mark.parametrize(
    "name", ["model.safetensors", "config.json", "silence_latent.pt"]
)
def test_equal_size_mutation_cannot_supply_verified_tensor_facts(
    tmp_path: Path, name: str
) -> None:
    """Both SHA-256 and Git object pins are checked over complete original data."""
    card, sources = selected_fixture(tmp_path)
    path = sources[(str(card.artifact_repository), str(card.source_revision))] / name
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match="object identity"):
        observe_native_music_artifacts(card, sources)


@pytest.mark.parametrize("fault", ["missing", "extra", "relative"])
def test_exact_source_mapping_cannot_be_assembled_from_another_selection(
    tmp_path: Path, fault: str
) -> None:
    """No first-match file search or arbitrary undeclared root fills missing roles."""
    card, sources = selected_fixture(tmp_path)
    key = next(iter(sources))
    if fault == "missing":
        del sources[key]
    elif fault == "extra":
        sources[("unselected/repo", "f" * 40)] = tmp_path
    else:
        sources[key] = Path("relative")
    with pytest.raises(ValueError, match="exactly the selected source"):
        observe_native_music_artifacts(card, sources)


@pytest.mark.parametrize("kind", ["file", "nested_root", "source_root"])
def test_symlinked_objects_or_component_ancestors_are_not_loaded(
    tmp_path: Path, kind: str
) -> None:
    """Even equal pinned content cannot redirect observation through a changed path."""
    card, sources = selected_fixture(tmp_path)
    if kind == "file":
        path = next(iter(sources.values())) / "model.safetensors"
    elif kind == "nested_root":
        path = (
            sources[
                ("ACE-Step/Ace-Step1.5", "19671f406d603126926c1b7e2adc169acbcade22")
            ]
            / "vae"
        )
    else:
        path = next(iter(sources.values()))
    moved = path.with_name(path.name + "-original")
    path.rename(moved)
    path.symlink_to(moved, target_is_directory=moved.is_dir())
    with pytest.raises(OSError):
        observe_native_music_artifacts(card, sources)


def test_real_store_hardlinks_remain_eligible(tmp_path: Path) -> None:
    """Source generations may retain the same ordinary inode without weakening pins."""
    card, sources = selected_fixture(tmp_path)
    path = next(iter(sources.values())) / "model.safetensors"
    os.link(path, tmp_path / "retained.safetensors")
    assert path.stat().st_nlink == 2
    assert (
        observe_native_music_artifacts(
            card, sources
        ).tensor_layouts.base.stored_tensor_bytes
        == 8
    )


def test_truncated_tensor_or_support_file_is_not_a_header_observation(
    tmp_path: Path,
) -> None:
    """A prefix alone cannot establish that the selected tensor's body exists."""
    card, sources = selected_fixture(tmp_path)
    path = next(iter(sources.values())) / "model.safetensors"
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="complete selected regular object"):
        observe_native_music_artifacts(card, sources)


def test_correctly_pinned_but_malformed_header_is_not_usable_geometry(
    tmp_path: Path,
) -> None:
    """Complete-file integrity does not make inconsistent tensor extents valid."""
    card, sources = selected_fixture(tmp_path, malformed=True)
    with pytest.raises(ValueError, match="offsets disagree"):
        observe_native_music_artifacts(card, sources)


def test_cancel_checkpoint_interrupts_large_body_observation(tmp_path: Path) -> None:
    """Hashing has bounded reads/checkpoints and performs no native or cleanup effects."""
    card, sources = selected_fixture(tmp_path, elements=1024**2)
    calls = 0

    def cancel() -> None:
        nonlocal calls
        calls += 1
        if calls == 4:
            raise InterruptedError("controlled cancellation")

    with pytest.raises(InterruptedError):
        observe_native_music_artifacts(card, sources, checkpoint=cancel)
    assert calls == 4
    assert (
        observe_native_music_artifacts(
            card, sources
        ).tensor_layouts.base.stored_tensor_bytes
        == 2 * 1024**2
    )


def test_inflight_file_mutation_cannot_supply_a_stale_policy(tmp_path: Path) -> None:
    """Actual stat changes are rejected even if a reader already saw original blocks."""
    card, sources = selected_fixture(tmp_path, elements=1024**2)
    path = next(iter(sources.values())) / "model.safetensors"
    calls = 0

    def mutate() -> None:
        nonlocal calls
        calls += 1
        # config has three checkpoints; mutate only after the tensor's first
        # bounded body read, while its descriptor and earlier stat are active.
        if calls == 5:
            original = path.read_bytes()
            path.write_bytes(original)

    with pytest.raises(ValueError, match="changed during observation"):
        observe_native_music_artifacts(card, sources, checkpoint=mutate)
