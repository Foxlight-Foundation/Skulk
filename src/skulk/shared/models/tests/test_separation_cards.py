"""Native separator card truth is independent of speech, music and ready engines."""

import hashlib
import json
import tomllib
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from skulk.api.main import API
from skulk.shared.models.model_cards import ModelCard, ModelTask
from skulk.shared.tests.model_card_fixtures import fixture_card_path
from skulk.shared.types.audio_separation import SEPARATION_STEMS


def _body() -> dict[str, object]:
    return cast(
        "dict[str, object]",
        tomllib.loads(Path(__file__).with_name("htdemucs-card.toml").read_text()),
    )


def test_separation_card_round_trip_and_public_metadata() -> None:
    """Read the registry contract without granting a runner or loaded model."""
    card = ModelCard.model_validate(_body())
    assert card.tasks == [ModelTask.AudioSeparation]
    profile = card.separation
    assert profile is not None
    assert profile.stems == SEPARATION_STEMS
    assert ModelCard.model_validate_json(card.model_dump_json()).separation == profile
    entry = API._model_list_entry(card)  # pyright: ignore[reportPrivateUsage]
    assert entry.separation is not None
    assert entry.separation.stems == profile.stems
    assert entry.separation.sample_rate == 44100
    assert entry.separation.trained_segment_seconds == 7.8
    assert entry.music is None and entry.audio is None
    assert not card.placement.compatible_backends
    assert entry.engine_support == []


def test_absent_section_preserves_existing_card_authorization_bytes() -> None:
    """Existing installed identities keep their exact pre-extension card projection."""
    card = ModelCard.model_validate(
        tomllib.loads(fixture_card_path("audio-cpp/ACE-Step1.5-Turbo-BF16").read_text())
    )
    assert "separation" not in card.model_dump()
    assert "separation" not in card.model_dump(by_alias=True)
    # Captured from the unchanged initiative reader before adding this section.
    content = json.dumps(
        card.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode()
    assert hashlib.sha256(content).hexdigest() == (
        "642dfe6b85f55e19d67a18dcad600dec46693306ff0e167aa41884b53a1e7cee"
    )
    entry = API._model_list_entry(card)  # pyright: ignore[reportPrivateUsage]
    assert "separation" not in entry.model_dump()


@pytest.mark.parametrize("missing", ["task", "section"])
def test_separation_task_and_section_are_inseparable(missing: str) -> None:
    """A recognized task and its strict metadata must occur together."""
    body = _body()
    if missing == "task":
        body["tasks"] = ["SpeechToText"]
    else:
        del body["separation"]
    with pytest.raises(ValidationError, match=r"requires a \[separation\]"):
        ModelCard.model_validate(body)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n_layers", 4),
        ("hidden_size", 1024),
        ("tasks", ["AudioSeparation", "SpeechToText"]),
        ("audio", {"kind": "stt"}),
        ("gguf_file", "other.gguf"),
        ("artifact_bundle", None),
        ("storage_size", {"in_bytes": 1}),
    ],
)
def test_separation_rejects_contradictory_card_truth(field: str, value: object) -> None:
    """Do not turn fake text dimensions or another tensor selection into model truth."""
    body = _body()
    body[field] = value
    with pytest.raises(ValidationError):
        ModelCard.model_validate(body)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("weights_file", "../955717e8.safetensors"),
        ("weights_file", "/955717e8.safetensors"),
        ("weights_file", "a\\weights.safetensors"),
        ("weights_file", "other.safetensors"),
        ("weights_file", "weights.pkl"),
        ("stems", ["drums", "bass", "other", "other"]),
        ("stems", ["vocals", "drums", "bass", "other"]),
        ("sample_rate", 48000),
        ("trained_segment_seconds", float("inf")),
        ("trained_segment_seconds", 0.0),
        ("max_seconds", 120),
    ],
)
def test_separation_metadata_is_typed_and_excludes_runtime_policy(
    field: str, value: object
) -> None:
    """Reject unsafe paths, wrong native outputs and misplaced API policy."""
    body = _body()
    profile = cast("dict[str, object]", body["separation"])
    profile[field] = value
    with pytest.raises(ValidationError):
        ModelCard.model_validate(body)


def test_separation_bundle_requires_the_exact_weights_identity() -> None:
    """A pinned repository without its selected tensor digest is insufficient."""
    body = _body()
    bundle = cast("dict[str, object]", body["artifact_bundle"])
    files = cast("list[dict[str, object]]", bundle["files"])
    del files[0]["object_id"]
    with pytest.raises(ValidationError, match="SHA-256 object identity"):
        ModelCard.model_validate(body)
