"""Invariant gate over every bundled model card.

The 2026-07-05 card audit found bad cards that had shipped silently: a card
whose HF repo no longer existed, cards missing their context length (losing
the card-side context-admission bound), and instruct-only Qwen3 cards whose
resolved capability profile advertised a thinking contract the model cannot
honor. The repo-existence check needs the network and stays an operational
script, but every static invariant it applied lives here so a bad card fails
CI instead of shipping.

Every check runs over ALL bundled cards (inference, image, embedding, speech),
so a newly added card is held to the same bar automatically.
"""

import tomllib
from pathlib import Path

import pytest

from skulk.shared.backends import engine_of
from skulk.shared.constants import RESOURCES_DIR
from skulk.shared.models.capabilities import resolve_model_capability_profile
from skulk.shared.models.model_cards import ModelCard

_CARD_DIRS = {
    "inference": Path(RESOURCES_DIR) / "inference_model_cards",
    "image": Path(RESOURCES_DIR) / "image_model_cards",
    "embedding": Path(RESOURCES_DIR) / "embedding_model_cards",
    "speech": Path(RESOURCES_DIR) / "speech_model_cards",
}

# Mirrors _ENGINES x _COMPUTE_BACKENDS in skulk.shared.backends. If an engine
# or compute backend is added there, extend this and the assertions below.
_VALID_TAGS = frozenset(
    {"mlx", "mlx_audio", "llama_cpp", "llama_server", "vllm"}
) | frozenset(
    f"{engine}-{compute}"
    for engine in ("mlx", "mlx_audio", "llama_cpp", "llama_server", "vllm")
    for compute in ("metal", "vulkan", "rocm", "cuda", "cpu")
)


def _all_card_files() -> list[tuple[str, Path]]:
    files = [
        (kind, path)
        for kind, directory in _CARD_DIRS.items()
        for path in sorted(directory.glob("*.toml"))
    ]
    assert files, f"no bundled cards found under {RESOURCES_DIR}"
    return files


_CARD_FILES = _all_card_files()
_CARD_IDS = [path.stem for _, path in _CARD_FILES]


def _load(path: Path) -> ModelCard:
    return ModelCard.model_validate(tomllib.loads(path.read_text()))


def test_no_duplicate_model_ids() -> None:
    seen: dict[str, str] = {}
    for _, path in _CARD_FILES:
        model_id = str(_load(path).model_id)
        assert model_id not in seen, (
            f"duplicate model_id {model_id}: {seen[model_id]} and {path.name}"
        )
        seen[model_id] = path.name


def test_gemma4_12b_card_does_not_advertise_missing_vision_tower() -> None:
    """Keep the incomplete upstream 12B artifact out of vision placement."""
    card = _load(
        _CARD_DIRS["inference"]
        / "mlx-community--gemma-4-12B-it-4bit.toml"
    )

    assert "vision" not in card.capabilities
    assert card.vision is None
    assert card.modalities is None


@pytest.mark.parametrize(("kind", "path"), _CARD_FILES, ids=_CARD_IDS)
def test_bundled_card_invariants(kind: str, path: Path) -> None:
    card = _load(path)  # validation itself is the first gate

    # Filename convention: the loader saves custom cards as
    # model_id.normalize() + ".toml"; bundled cards follow the same rule so a
    # filename always names the model it configures.
    assert path.stem == card.model_id.normalize(), (
        f"filename {path.stem} != model_id.normalize() "
        f"({card.model_id.normalize()})"
    )

    assert card.storage_size.in_bytes > 0
    assert card.tasks, "card declares no tasks"
    assert card.family, "family must be set on bundled cards"

    compatible = set(card.placement.compatible_backends)
    unknown = compatible - _VALID_TAGS
    assert not unknown, f"unknown backend tags: {sorted(unknown)}"
    missing_preference = [
        tag for tag in card.placement.backend_preference if tag not in compatible
    ]
    assert not missing_preference, (
        f"backend_preference not in compatible_backends: {missing_preference}"
    )

    engines = {engine for tag in compatible if (engine := engine_of(tag))}
    is_gguf = card.gguf_file is not None or "gguf" in str(card.model_id).lower()
    if kind == "inference":
        # Model truth: a GGUF artifact can never run in-process MLX and a
        # safetensors artifact can never run the llama engines.
        if is_gguf:
            assert "mlx" not in engines, "GGUF card lists the mlx engine"
            assert {"llama_cpp", "llama_server"} & engines, (
                "GGUF card lists no llama engine"
            )
        else:
            assert not ({"llama_cpp", "llama_server"} & engines), (
                "safetensors card lists llama engines"
            )

        # The context-admission ceiling and the KV estimate both degrade to
        # fallbacks without these; the audit found shipped cards missing them.
        assert card.context_length > 0, "inference card must set context_length"
        assert card.num_key_value_heads is not None, (
            "inference card must set num_key_value_heads"
        )
        assert card.quantization, "inference card must set quantization"

    # Vision declarations must be coherent: the capability string is what the
    # resolver and placement gating read, the section is what the loader reads.
    if card.vision is not None:
        assert "vision" in card.capabilities, (
            "[vision] section without the 'vision' capability"
        )
    if card.audio is not None:
        assert {"tts", "stt"} & set(card.capabilities), (
            "[audio] section without a speech capability"
        )
        assert "mlx_audio" in engines, (
            "speech card must list the mlx_audio engine until another speech "
            "runner exists"
        )

    runtime = card.runtime
    if runtime is not None:
        mlx_spec_fields = {
            "mtp_heads": runtime.mtp_heads,
            "mtp_max_depth": runtime.mtp_max_depth,
            "mtp_sidecar_repo": runtime.mtp_sidecar_repo,
            "assistant_model_repo": runtime.assistant_model_repo,
        }
        dead_mlx_spec = {k for k, v in mlx_spec_fields.items() if v not in (None, False)}
        if dead_mlx_spec:
            assert "mlx" in engines, (
                f"MLX speculation fields {sorted(dead_mlx_spec)} on a card "
                "without the mlx engine (dead config)"
            )
        if runtime.served_spec_type is not None:
            assert "llama_server" in engines, (
                "served_spec_type without the llama_server engine (dead config)"
            )
        if runtime.served_spec_type in ("draft_simple", "draft_eagle3"):
            assert runtime.served_spec_draft_repo and runtime.served_spec_draft_file, (
                f"served_spec_type={runtime.served_spec_type} requires a draft "
                "repo and file"
            )
        if runtime.served_spec_draft_repo:
            assert runtime.served_spec_type is not None, (
                "served draft repo without served_spec_type"
            )
        if runtime.vllm_spec_method is not None:
            assert "vllm" in engines, (
                "vllm_spec_method without the vllm engine (dead config)"
            )
        if runtime.vllm_spec_num_tokens is not None:
            assert runtime.vllm_spec_method is not None, (
                "vllm_spec_num_tokens without vllm_spec_method (dead config)"
            )
        # The method/draft-repo pairing itself (dflash requires a draft repo,
        # mtp forbids one) is enforced by the card model validator, so an
        # inconsistent bundled card already fails at parse above.


@pytest.mark.parametrize(
    ("kind", "path"),
    [(kind, path) for kind, path in _CARD_FILES if kind == "inference"],
    ids=[path.stem for kind, path in _CARD_FILES if kind == "inference"],
)
def test_bundled_card_capability_resolution_is_coherent(kind: str, path: Path) -> None:
    """The resolved profile must agree with the card's thinking declaration.

    This is the gate the audit derived the qwen3 finding from: a family
    default forcing a thinking contract onto a card that explicitly declares
    no thinking (or the reverse) means either the resolver or the card is
    wrong, and both are ship blockers.
    """
    card = _load(path)
    profile = resolve_model_capability_profile(card.model_id, model_card=card)
    declares_thinking = "thinking" in card.capabilities or card.reasoning is not None
    assert profile.supports_thinking == declares_thinking, (
        f"resolver supports_thinking={profile.supports_thinking} but card "
        f"declaration says {declares_thinking} (family={profile.family})"
    )


def test_qwen_custom_voice_card_exposes_upstream_speaker_inventory() -> None:
    """The voice endpoint must expose the checkpoint's stable speaker IDs."""

    path = (
        Path(RESOURCES_DIR)
        / "speech_model_cards"
        / "mlx-community--Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit.toml"
    )
    card = _load(path)

    assert card.audio is not None
    assert card.audio.supports_voice_listing is True
    assert card.audio.default_voice == "ryan"
    assert card.audio.voices == (
        "serena",
        "vivian",
        "uncle_fu",
        "ryan",
        "aiden",
        "ono_anna",
        "sohee",
        "eric",
        "dylan",
    )
    assert tuple(voice.id for voice in card.audio.voice_catalog) == card.audio.voices
    assert card.audio.voice_catalog[3].name == "Ryan"
    assert card.audio.voice_catalog[3].preferred_languages == ("en",)
