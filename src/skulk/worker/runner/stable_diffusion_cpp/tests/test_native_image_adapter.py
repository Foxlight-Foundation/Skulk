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
    ImageFamily,
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


@pytest.mark.parametrize(
    ("backend", "turbo", "runtime_flags"),
    [
        ("stable_diffusion_cpp-vulkan", False, ["--fa", "--vae-tiling", "--vae-tile-size", "512", "--vae-tile-overlap", "0.25"]),
        ("stable_diffusion_cpp-vulkan", True, ["--fa", "--vae-tiling", "--vae-tile-size", "512", "--vae-tile-overlap", "0.25"]),
        ("stable_diffusion_cpp-cuda", False, ["--fa"]),
        ("stable_diffusion_cpp-cuda", True, ["--fa"]),
        ("stable_diffusion_cpp-metal", True, []),
        ("stable_diffusion_cpp-rocm", False, ["--fa"]),
        ("stable_diffusion_cpp-rocm", True, ["--fa"]),
    ],
)
def test_component_verification_checks_every_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    turbo: bool,
    runtime_flags: list[str],
) -> None:
    card = _card(turbo=turbo)
    assert card.image is not None
    for component in card.image.components:
        (tmp_path / component.entry_file).write_bytes(b"abc")

    def locate(_identifier: ModelId, entry: str, _revision: str | None = None) -> Path:
        return tmp_path / entry

    monkeypatch.setattr(adapter, "build_sidecar_path", locate)
    arguments = adapter.native_component_arguments(card, backend=backend)
    assert arguments[:8:2] == ["--diffusion-model", "--vae", "--llm", "--llm_vision"]
    assert arguments[8:] == runtime_flags
    (tmp_path / "text_encoder_vision.gguf").write_bytes(b"bad")
    with pytest.raises(ValueError, match="immutable file identity"):
        adapter.native_component_arguments(card, backend=backend)


@pytest.mark.parametrize(
    ("backend", "runtime_flags"),
    [
        ("stable_diffusion_cpp-metal", ["--batched-cfg", "off", "--fa", "--vae-tiling", "--vae-tile-size", "512", "--vae-tile-overlap", "0.25"]),
        ("stable_diffusion_cpp-cuda", []),
        ("stable_diffusion_cpp-rocm", []),
        ("stable_diffusion_cpp-vulkan", []),
    ],
)
def test_sdxl_guidance_workaround_is_limited_to_metal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    runtime_flags: list[str],
) -> None:
    component = ImageComponent(
        role="model",
        repository="owner/model",
        revision="a" * 40,
        entry_file="checkpoint.safetensors",
        files=(
            ImageComponentFile(
                path="checkpoint.safetensors",
                size_bytes=3,
                object_id="sha256:" + hashlib.sha256(b"abc").hexdigest(),
            ),
        ),
    )
    card = ModelCard(
        model_id=ModelId("owner/model"),
        source_revision="a" * 40,
        storage_size=Memory.from_bytes(3),
        n_layers=1,
        hidden_size=1,
        supports_tensor=False,
        tasks=[ModelTask.TextToImage],
        image=ImageCardConfig(family="sdxl", components=(component,), default_steps=20),
    )
    (tmp_path / component.entry_file).write_bytes(b"abc")

    def locate(_identifier: ModelId, entry: str, _revision: str | None = None) -> Path:
        return tmp_path / entry

    monkeypatch.setattr(adapter, "build_sidecar_path", locate)
    arguments = adapter.native_component_arguments(card, backend=backend)
    assert arguments == ["--model", str(tmp_path / component.entry_file), *runtime_flags]


def test_z_image_metal_default_uses_bounded_attention_and_decoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = _card()
    assert card.image is not None
    card = card.model_copy(update={"image": ImageCardConfig(
        family="z_image_turbo", components=card.image.components, default_steps=9,
    )})
    assert card.image is not None
    for component in card.image.components:
        (tmp_path / component.entry_file).write_bytes(b"abc")

    def locate(_identifier: ModelId, entry: str, _revision: str | None = None) -> Path:
        return tmp_path / entry

    monkeypatch.setattr(adapter, "build_sidecar_path", locate)
    arguments = adapter.native_component_arguments(card, backend="stable_diffusion_cpp-metal")
    assert arguments[8:] == ["--fa", "--vae-tiling", "--vae-tile-size", "512", "--vae-tile-overlap", "0.25"]
    assert "--batched-cfg" not in arguments
    assert adapter.native_component_arguments(card, backend="stable_diffusion_cpp-cuda")[8:] == []


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


def _sdxl_card() -> ModelCard:
    """Return an intrinsic multi-reference fixture the current producer cannot honor."""
    card = _card()
    profile = card.image
    assert profile is not None
    selected = profile.components[0].model_copy(update={"role": "model"})
    modified = card.model_copy(update={
        "image": ImageCardConfig(family="sdxl", components=(selected,), max_reference_images=2, default_steps=20, default_guidance=7.5),
        "storage_size": Memory.from_bytes(3),
    })
    return ModelCard.model_validate_json(modified.model_dump_json())


def test_sdxl_does_not_silently_discard_extra_reference_input() -> None:
    """Reject the second valid image even under a profile declaring two references."""
    card = _sdxl_card()
    params = ImageEditsTaskParams(model="owner/model", prompt="fox", size="512x512")
    first = _image("red")
    body = adapter.native_image_request(card, params, 42, (first,))
    assert body["init_image"] == base64.b64encode(first).decode()
    with pytest.raises(ValueError, match="initialization image"):
        adapter.native_image_request(card, params, 42, (first, _image("blue")))


@pytest.mark.parametrize(
    ("family", "steps", "guidance"),
    [("flux_3_klein", 16, 1.0), ("flux_3_klein_base", 50, 3.5)],
)
def test_flux3_checkpoint_sampling_and_prepared_caption_survive_translation(
    family: ImageFamily, steps: int, guidance: float,
) -> None:
    """Distilled/Base captions and ten ordered references reach the native API unchanged."""
    original = _card()
    assert original.image is not None
    # FLUX.3 conditions references through its VAE; it needs the Qwen3-VL text
    # weights but does not require Qwen Image's separate vision projector.
    profile = ImageCardConfig(
        family=family, default_steps=steps, default_guidance=guidance,
        max_reference_images=10, supports_alpha=False,
        components=tuple(
            component for component in original.image.components
            if component.role != "text_encoder_vision"
        ),
    )
    card = ModelCard.model_validate_json(original.model_copy(
        update={"image": profile, "storage_size": Memory.from_bytes(9)}
    ).model_dump_json())
    prompt = 'Full-color studio photograph of a fox <fox> beside a blue mug. [{"id":"fox","bbox":[0,0,1000,500]}]'
    references = tuple(_image(f"#{index:06x}") for index in range(10))
    body = adapter.native_image_request(
        card,
        ImageEditsTaskParams(model="owner/model", prompt=prompt, input_image_count=10),
        42, references,
    )
    assert body["prompt"] == prompt
    assert body["sample_params"] == {
        "sample_method": "euler", "sample_steps": steps,
        "guidance": {"txt_cfg": guidance},
    }
    assert body["ref_images"] == [base64.b64encode(item).decode() for item in references]
    assert "init_image" not in body
    assert card.image is not None
    assert ImageCardConfig.model_validate_json(card.image.model_dump_json()) == card.image
    with pytest.raises(ValueError, match="reference input"):
        adapter.native_image_request(
            card, ImageEditsTaskParams(model="owner/model", prompt=prompt),
            42, (*references, references[0]),
        )


@pytest.mark.parametrize("prompt", [
    "An orange fox beside a blue mug.",
    '{"summary":{"caption":"A travel poster"},"items":[{"text":"EXPLORE","bbox":[-416,-384,-288,384]}],"bbox_coord_space":"abs_centralized","bbox_format":"yxyx"}',
    "A detailed scene. " * 500,
])
def test_moss_tokenizer_and_literal_prompt_contract(prompt: str) -> None:
    """MOSS preserves prose/layout prompts and its native shifted schedule."""
    original = _card()
    assert original.image is not None
    components = tuple(
        component for component in original.image.components
        if component.role != "text_encoder_vision"
    )
    with pytest.raises(ValueError, match="pinned Qwen3.5 tokenizer"):
        ImageCardConfig(
            family="moss_image", components=components,
            default_steps=50, default_guidance=3.0, dimension_multiple=16,
        )
    tokenizer = components[0].model_copy(update={"role": "tokenizer"})
    profile = ImageCardConfig(
        family="moss_image", components=(*components, tokenizer),
        default_steps=50, default_guidance=3.0, dimension_multiple=16,
    )
    card = ModelCard.model_validate_json(original.model_copy(update={
        "image": profile, "tasks": [ModelTask.TextToImage],
    }).model_dump_json())
    body = adapter.native_image_request(
        card, ImageGenerationTaskParams(
            model="owner/model", prompt=prompt, size="1008x1024",
        ), 42,
    )
    assert body["prompt"] == prompt
    assert body["sample_params"] == {
        "sample_method": "euler", "sample_steps": 50,
        "guidance": {"txt_cfg": 3.0}, "flow_shift": 5.0,
    }
    assert body["width"] == 1008
    assert "ref_images" not in body
    with pytest.raises(ValueError, match="reference input"):
        adapter.native_image_request(
            card, ImageEditsTaskParams(model="owner/model", prompt=prompt),
            42, (_image("blue"),),
        )
