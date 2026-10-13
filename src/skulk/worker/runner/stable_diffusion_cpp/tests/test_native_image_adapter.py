"""Verify literal prompts, immutable component bytes, and ordered RGBA edits."""

import base64
import hashlib
import io
from pathlib import Path

import pytest
from PIL import Image

from skulk.api.types.api import (
    AdvancedImageParams,
    ImageEditsTaskParams,
    ImageGenerationTaskParams,
)
from skulk.shared.models.image import (
    ImageCardConfig,
    ImageComponent,
    ImageComponentFile,
)
from skulk.shared.models.model_cards import ModelCard, ModelId, ModelTask
from skulk.shared.types.memory import Memory
from skulk.worker.runner.stable_diffusion_cpp import adapter


def _card(*, turbo: bool = False) -> ModelCard:
    components = tuple(
        ImageComponent(
            role=role,
            repository="owner/model",
            revision="a" * 40,
            entry_file=f"{role}.gguf",
            files=(
                ImageComponentFile(
                    path=f"{role}.gguf",
                    size_bytes=3,
                    object_id="sha256:" + hashlib.sha256(b"abc").hexdigest(),
                ),
            ),
        )
        for role in ("diffusion", "vae", "text_encoder", "text_encoder_vision")
    )
    profile = ImageCardConfig(
        family="qwen_image_21_turbo" if turbo else "qwen_image_21",
        components=components,
        max_reference_images=2,
        supports_alpha=True,
        default_steps=8 if turbo else 20,
        default_guidance=1.0 if turbo else 6.0,
        fixed_sigmas=(
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
        if turbo
        else (),
    )
    return ModelCard(
        model_id=ModelId("owner/model"),
        source_revision="a" * 40,
        gguf_file="diffusion.gguf",
        storage_size=Memory.from_bytes(12),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextToImage, ModelTask.ImageToImage],
        image=profile,
    )


def _image(color: str) -> bytes:
    output = io.BytesIO()
    Image.new("RGBA", (64, 64), color).save(output, format="PNG")
    return output.getvalue()


def test_literal_prompt_and_exact_turbo_grid() -> None:
    prompt = "--steps 99 <script> literal prompt"
    body = adapter.native_image_request(
        _card(turbo=True),
        ImageGenerationTaskParams(model="owner/model", prompt=prompt, size="2048x2048"),
        42,
    )
    assert body["prompt"] == prompt
    assert body["sample_params"] == {
        "sample_method": "euler",
        "sample_steps": 8,
        "guidance": {"txt_cfg": 1.0},
        "custom_sigmas": [
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
    assert body["batch_count"] == 1


@pytest.mark.parametrize(
    "advanced",
    [
        AdvancedImageParams(num_inference_steps=9),
        AdvancedImageParams(guidance=2.0),
        AdvancedImageParams(negative_prompt="bad"),
    ],
)
def test_turbo_overrides_are_refused_before_native_work(
    advanced: AdvancedImageParams,
) -> None:
    with pytest.raises(ValueError, match="Turbo"):
        adapter.native_image_request(
            _card(turbo=True),
            ImageGenerationTaskParams(
                model="owner/model", prompt="a fox", advanced_params=advanced
            ),
            42,
        )


def test_reference_order_and_alpha_bytes_survive_translation() -> None:
    references = (_image("#ff000080"), _image("#0000ff00"))
    body = adapter.native_image_request(
        _card(),
        ImageEditsTaskParams(model="owner/model", prompt="edit", input_image_count=2),
        42,
        references,
    )
    assert body["ref_images"] == [
        base64.b64encode(reference).decode() for reference in references
    ]
    assert "init_image" not in body
    with pytest.raises(ValueError, match="reference input"):
        adapter.native_image_request(
            _card(),
            ImageEditsTaskParams(model="owner/model", prompt="edit"),
            42,
            (*references, references[0]),
        )


def test_invalid_reference_is_refused_without_native_decoding() -> None:
    with pytest.raises(ValueError, match="container"):
        adapter.native_image_request(
            _card(),
            ImageEditsTaskParams(model="owner/model", prompt="edit"),
            42,
            (b"invalid",),
        )


def test_component_verification_checks_every_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    card = _card()
    assert card.image is not None
    for component in card.image.components:
        (tmp_path / component.entry_file).write_bytes(b"abc")

    def locate(_identifier: ModelId, entry: str, _revision: str | None = None) -> Path:
        return tmp_path / entry

    monkeypatch.setattr(adapter, "build_sidecar_path", locate)
    arguments = adapter.native_component_arguments(card)
    assert arguments[::2] == ["--diffusion-model", "--vae", "--llm", "--llm_vision"]
    (tmp_path / "text_encoder_vision.gguf").write_bytes(b"bad")
    with pytest.raises(ValueError, match="immutable file identity"):
        adapter.native_component_arguments(card)


def test_ephemeral_reference_bytes_are_excluded_from_state_and_logs() -> None:
    params = ImageEditsTaskParams(
        model="owner/model",
        prompt="edit",
        reference_image_data=("PRIVATE_BYTES", "SECOND_PRIVATE_BYTES"),
        input_image_count=2,
    )
    assert "reference_image_data" not in params.model_dump()
    assert "PRIVATE_BYTES" not in repr(params)


@pytest.mark.parametrize("color", ["white", "black", "#ff000080"])
def test_blank_terminal_outputs_fail_but_previews_remain_valid(color: str) -> None:
    profile = _card().image
    assert profile is not None
    encoded = base64.b64encode(_image(color)).decode()
    with pytest.raises(ValueError, match="blank terminal"):
        adapter.validate_native_output(encoded, "png", profile, (64, 64), partial=False)
    adapter.validate_native_output(encoded, "png", profile, (64, 64), partial=True)


def test_final_output_must_match_admitted_canvas() -> None:
    profile = _card().image
    assert profile is not None
    encoded = base64.b64encode(_image("red")).decode()
    with pytest.raises(ValueError, match="canvas differs"):
        adapter.validate_native_output(
            encoded, "png", profile, (512, 512), partial=False
        )
