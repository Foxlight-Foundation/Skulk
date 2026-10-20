# pyright: reportPrivateUsage=false
"""External adapters retain their signed layout while resolving in Comfy loaders."""

import tomllib
from pathlib import Path

import pytest
import yaml
from pydantic import TypeAdapter

import skulk.shared.constants as constants
from skulk.download.download_utils import _SOURCE_REVISION_MARKER
from skulk.shared.models.model_cards import (
    ModelCard,
    VideoCompanionConfig,
    VideoCompanionKind,
)
from skulk.shared.types.video import VideoGenerationTaskParams, VideoReferenceSpec
from skulk.worker.runner.comfy.graph import (
    ReferenceBinding,
    build_prompt,
    companion_file,
    extra_model_paths_yaml,
    plan_comfy_render,
    resolve_model_files,
)
from skulk.worker.runner.comfy.runner import _companion_search_roots


def _published_card() -> ModelCard:
    """Load the exact DFR body published as card_5h5amol3q65a2pmleiqcveqcili5oy64hhxwtckncueghnqpjwpq."""
    path = Path(__file__).with_name("fixtures") / "ltx25-dfr-published.toml"
    return ModelCard.model_validate(tomllib.loads(path.read_text()))


@pytest.mark.parametrize("relative_path", [
    "ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors",
    "adapters/detail.safetensors",
    "loras/detail.safetensors",
    "loras/nested/detail.safetensors",
])
def test_external_adapter_graph_name_matches_its_real_staged_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative_path: str,
) -> None:
    """Resolve the published root layout and both conventional/nested repositories."""
    card = _published_card()
    assert card.video is not None
    spatial = next(item for item in card.video.companions if item.name == "dfr_spatial")
    spatial = VideoCompanionConfig.model_validate({**spatial.model_dump(), "path": relative_path})
    card = card.model_copy(update={"video": card.video.model_copy(update={"companions": tuple(
        spatial if item.name == "dfr_spatial" else item for item in card.video.companions
    )})})
    assert spatial.repo is not None and spatial.revision is not None
    staged = tmp_path / f"{spatial.repo.normalize()}--revision-{spatial.revision}"
    target = staged / spatial.path
    target.parent.mkdir(parents=True)
    target.write_bytes(b"downloaded-and-verified-weight-placeholder")
    (staged / _SOURCE_REVISION_MARKER).write_text(spatial.revision)
    monkeypatch.setattr(constants, "SKULK_MODELS_PATH", (tmp_path,))

    files = resolve_model_files(card)
    roots = _companion_search_roots(card)
    assert len(roots) == 1 and roots[0].folder == "loras"
    assert files.dfr_spatial_lora is not None
    assert (roots[0].directory / files.dfr_spatial_lora).samefile(target)
    image = VideoReferenceSpec(slot=0, kind="image", role="first_frame", media_type="image/png", size_bytes=1, sha256="0" * 64)
    params = VideoGenerationTaskParams(references=(image,), reference_bytes=1, total_input_chunks=1, model=str(card.model_id), prompt="A toy boat", seconds=1, size="512x384", pipeline="dfr")
    graph = build_prompt(plan_comfy_render(params, card), params, (ReferenceBinding(spec=image, input_name="image.png"),), "test")
    assert graph["ltx_dfr_lora"]["inputs"]["lora_name"] == files.dfr_spatial_lora
    text = extra_model_paths_yaml(tmp_path / "base", roots)
    search_paths = TypeAdapter(dict[str, dict[str, str]]).validate_python(yaml.safe_load(text))
    mapped = search_paths["skulk_companion_1"]
    assert set(mapped) == {"base_path", "loras"}
    assert (Path(mapped["base_path"]) / mapped["loras"] / files.dfr_spatial_lora).samefile(target)
    assert target.read_bytes() == b"downloaded-and-verified-weight-placeholder"
    target.unlink()
    with pytest.raises(RuntimeError, match="dfr_spatial .* is not staged"):
        _companion_search_roots(card)


def test_search_root_paths_with_yaml_punctuation_remain_literal(tmp_path: Path) -> None:
    """A path must not become YAML comments, fields or a different scalar type."""
    directory = tmp_path / "models: pinned #1"
    text = extra_model_paths_yaml(directory)
    parsed = TypeAdapter(dict[str, dict[str, str]]).validate_python(yaml.safe_load(text))
    assert parsed["skulk"]["base_path"] == directory.resolve().as_posix()


@pytest.mark.parametrize("explicit_repository", [False, True])
def test_primary_bundle_files_still_require_the_standard_loader_folder(explicit_repository: bool) -> None:
    """External layout flexibility must not change the primary bundle contract."""
    card = _published_card()
    item = VideoCompanionConfig(
        kind=VideoCompanionKind.Lora, name="local", path="root-adapter.safetensors",
        repo=card.artifact_repository if explicit_repository else None,
        revision=card.source_revision if explicit_repository else None,
    )
    with pytest.raises(ValueError, match="not under the folder"):
        companion_file(item, card.artifact_repository)


def test_external_companion_at_the_wrong_revision_is_not_exposed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finding a file with the right basename cannot bypass its immutable revision."""
    card = _published_card()
    assert card.video is not None
    item = next(item for item in card.video.companions if item.name == "dfr_spatial")
    assert item.repo is not None
    directory = tmp_path / item.repo.normalize()
    directory.mkdir()
    (directory / item.path).write_bytes(b"wrong revision")
    (directory / _SOURCE_REVISION_MARKER).write_text("0" * 40)
    monkeypatch.setattr(constants, "SKULK_MODELS_PATH", (tmp_path,))
    with pytest.raises(RuntimeError, match="dfr_spatial .* is not staged"):
        _companion_search_roots(card)
