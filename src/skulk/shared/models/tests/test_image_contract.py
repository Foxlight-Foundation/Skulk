"""Reject incomplete artifacts, altered distilled schedules and unsafe image geometry."""

import pytest
from pydantic import ValidationError

from skulk.shared.models.image import (
    ImageCardConfig,
    ImageComponent,
    ImageComponentFile,
)


def _component(role: str, path: str) -> dict[str, object]:
    return {
        "role": role,
        "repository": "owner/model",
        "revision": "a" * 40,
        "entry_file": path,
        "files": [{"path": path, "size_bytes": 7, "object_id": "sha256:" + "b" * 64}],
    }


def _document() -> dict[str, object]:
    return {
        "family": "qwen_image_21_turbo",
        "components": [
            _component("diffusion", "model.gguf"),
            _component("vae", "vae.safetensors"),
            _component("text_encoder", "encoder.gguf"),
            _component("text_encoder_vision", "vision.gguf"),
        ],
        "max_reference_images": 10,
        "supports_alpha": True,
        "default_steps": 8,
        "fixed_sigmas": [
            1.0,
            0.978453,
            0.95418,
            0.926626,
            0.89508,
            0.845148,
            0.704534,
            0.414568,
            0.0,
        ],
    }


def test_image_contract_restores_wire_sequences_and_complete_size() -> None:
    card = ImageCardConfig.model_validate(_document())
    assert card.storage_bytes == 28
    assert ImageCardConfig.model_validate_json(card.model_dump_json()) == card
    with pytest.raises(ValidationError):
        card.default_steps = 40


@pytest.mark.parametrize(
    "field,value",
    [
        ("default_steps", 6),
        ("default_guidance", 3.5),
        ("fixed_sigmas", [1.0, 0.5, 0.0]),
    ],
)
def test_turbo_refuses_a_different_sampling_contract(field: str, value: object) -> None:
    document = _document()
    document[field] = value
    with pytest.raises(ValidationError, match="sigmas|eight-step"):
        ImageCardConfig.model_validate(document)


@pytest.mark.parametrize(
    "path",
    [
        "../model.gguf",
        "/model.gguf",
        "a//model.gguf",
        "a/./model.gguf",
        "a\\model.gguf",
        "",
    ],
)
def test_component_refuses_noncanonical_paths(path: str) -> None:
    with pytest.raises(ValidationError):
        ImageComponentFile(path=path, size_bytes=1, object_id="sha256:" + "b" * 64)


def test_qwen_editing_requires_the_vision_weights() -> None:
    document = _document()
    document["components"] = [
        _component("diffusion", "model.gguf"),
        _component("vae", "vae.safetensors"),
        _component("text_encoder", "encoder.gguf"),
    ]
    with pytest.raises(ValidationError, match="vision"):
        ImageCardConfig.model_validate(document)


def test_component_entry_must_be_in_the_file_manifest() -> None:
    component = _component("diffusion", "model.gguf")
    component["entry_file"] = "other.gguf"
    with pytest.raises(ValidationError, match="entry"):
        ImageComponent.model_validate(component)


@pytest.mark.parametrize(
    "width,height", [(2048, 2048), (2400, 1792), (2752, 1536), (512, 512)]
)
def test_published_canvases_fit_the_contract(width: int, height: int) -> None:
    ImageCardConfig.model_validate(_document()).validate_canvas(width, height)


@pytest.mark.parametrize(
    "width,height", [(4096, 4096), (1025, 1024), (0, 512), (65536, 64)]
)
def test_oversized_or_unaligned_work_is_refused_before_admission(
    width: int, height: int
) -> None:
    with pytest.raises(ValueError, match="canvas"):
        ImageCardConfig.model_validate(_document()).validate_canvas(width, height)


def test_image_card_cannot_load_another_artifact_under_selected_gguf_identity() -> None:
    from skulk.shared.models.model_cards import ModelCard, ModelId, ModelTask
    from skulk.shared.types.memory import Memory

    profile = ImageCardConfig.model_validate(_document())
    with pytest.raises(ValidationError, match="selected GGUF"):
        ModelCard(
            model_id=ModelId("owner/model"),
            source_revision="a" * 40,
            gguf_file="different.gguf",
            storage_size=Memory.from_bytes(28),
            n_layers=1,
            hidden_size=1,
            supports_tensor=False,
            tasks=[ModelTask.TextToImage, ModelTask.ImageToImage],
            image=profile,
        )


def test_sdxl_unet_geometry_is_nonuniform_and_matches_card_dimensions() -> None:
    """Explicit SDXL modules bind generic dimensions without implying text KV."""
    from skulk.shared.models.model_cards import ModelCard, ModelId, ModelTask
    from skulk.shared.types.memory import Memory

    profile = ImageCardConfig.model_validate(
        {
            "family": "sdxl",
            "default_steps": 20,
            "components": [_component("model", "model.gguf")],
            "unet_structure": {
                "input_block_channels": [320, 640],
                "middle_block_channels": [1280],
                "output_block_channels": [640, 320],
                "input_channels": 4,
                "output_channels": 4,
            },
        }
    )
    structure = profile.unet_structure
    assert structure is not None
    assert structure.block_count == 5 and structure.maximum_channels == 1280
    assert ImageCardConfig.model_validate_json(profile.model_dump_json()) == profile

    def make_card(layers: int, width: int) -> ModelCard:
        """Construct a typed native card for one declared UNet structure."""
        return ModelCard(
            model_id=ModelId("owner/model"),
            source_revision="a" * 40,
            gguf_file="model.gguf",
            storage_size=Memory.from_bytes(7),
            n_layers=layers,
            hidden_size=width,
            supports_tensor=False,
            tasks=[ModelTask.TextToImage],
            image=profile,
        )

    for layers, width in [(5, 1280), (1, 1280), (5, 1)]:
        if layers == 5 and width == 1280:
            assert make_card(layers, width).image == profile
        else:
            with pytest.raises(ValidationError, match="UNet modules"):
                make_card(layers, width)
    wrong_family = profile.model_dump()
    wrong_family["family"] = "flux_2_klein_4b"
    with pytest.raises(ValidationError, match="only to an SDXL"):
        ImageCardConfig.model_validate(wrong_family)
