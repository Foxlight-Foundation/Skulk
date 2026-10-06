from skulk.worker.main import (
    _INSTANCE_FAILURE_MESSAGE_LIMIT,  # pyright: ignore[reportPrivateUsage] - unit under test
    _instance_failure_message,  # pyright: ignore[reportPrivateUsage] - unit under test
)


def test_instance_failure_message_retains_only_classified_safe_reason() -> None:
    message = _instance_failure_message("runner crashed repeatedly")

    assert message == "runner crashed repeatedly"


def test_instance_failure_message_is_whitespace_normalized_and_bounded() -> None:
    message = _instance_failure_message("runner\n crashed " + "x" * 4096)

    assert "\n" not in message
    assert len(message) == _INSTANCE_FAILURE_MESSAGE_LIMIT
