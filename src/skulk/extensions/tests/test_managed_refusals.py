"""An owner's refusal may name its next step; anything else stays a bare refusal."""

import pytest
from pydantic import JsonValue

from skulk.extensions.managed import owner_response_payload
from skulk.extensions.refusals import (
    MAX_REFUSAL_MESSAGE_CHARACTERS,
    ManagedRefusalError,
    refusal_detail,
    refusal_message,
)


def _refusal(frame: dict[str, JsonValue]) -> ManagedRefusalError:
    with pytest.raises(ManagedRefusalError) as raised:
        owner_response_payload(frame)
    return raised.value


def test_a_successful_frame_returns_its_payload() -> None:
    assert owner_response_payload({"result": {"ready": True}}) == {"ready": True}


def test_a_refusal_carries_the_plugins_next_step() -> None:
    refusal = _refusal(
        {
            "error": "configuration operation refused",
            "message": "  Add your RunPod API key under Credentials.  ",
            "evidence_reference": "evidence-1",
        }
    )
    assert refusal.message == "Add your RunPod API key under Credentials."
    assert str(refusal) == "managed operation refused"


@pytest.mark.parametrize(
    "frame",
    [
        {"error": "configuration operation refused"},
        {"error": "refused", "evidence_reference": "evidence-1"},
    ],
)
def test_a_refusal_without_a_message_is_still_a_refusal(
    frame: dict[str, JsonValue],
) -> None:
    assert _refusal(frame).message is None


@pytest.mark.parametrize(
    "message",
    [
        "",
        "   ",
        "x" * (MAX_REFUSAL_MESSAGE_CHARACTERS + 1),
        "Turn it off\nthen on",
        "bell \x07",
        7,
        None,
        ["Add a key"],
    ],
)
def test_an_unusable_message_is_dropped_not_fatal(message: JsonValue) -> None:
    assert _refusal({"error": "refused", "message": message}).message is None


def test_the_longest_allowed_message_is_kept() -> None:
    text = "x" * MAX_REFUSAL_MESSAGE_CHARACTERS
    assert refusal_message(text) == text


@pytest.mark.parametrize(
    "frame",
    [
        {"error": "refused", "detail": "unexpected member"},
        {"message": "a message without an error"},
        {"result": {}, "error": "both"},
        {"result": []},
        {},
    ],
)
def test_any_other_frame_is_an_invalid_response_not_a_refusal(
    frame: dict[str, JsonValue],
) -> None:
    with pytest.raises(ValueError) as raised:
        owner_response_payload(frame)
    assert not isinstance(raised.value, ManagedRefusalError)
    assert str(raised.value) == "invalid managed response"


def test_an_in_process_refusal_gets_the_same_bounds() -> None:
    assert ManagedRefusalError("  Add a key.  ").message == "Add a key."
    assert ManagedRefusalError("Turn it off\nthen on").message is None
    too_long = "x" * (MAX_REFUSAL_MESSAGE_CHARACTERS + 1)
    assert ManagedRefusalError(too_long).message is None
    assert ManagedRefusalError(None).message is None


def test_the_detail_is_the_message_or_the_callers_own_sentence() -> None:
    fallback = "configuration refused; reload settings and validation"
    assert refusal_detail(ManagedRefusalError("Add a key."), fallback) == "Add a key."
    assert refusal_detail(ManagedRefusalError(None), fallback) == fallback
    assert refusal_detail(ValueError("sensitive"), fallback) == fallback
