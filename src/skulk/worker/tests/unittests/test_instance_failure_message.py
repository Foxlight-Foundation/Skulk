from skulk.worker.main import (
    _INSTANCE_FAILURE_MESSAGE_LIMIT,  # pyright: ignore[reportPrivateUsage] - unit under test
    _instance_failure_message,  # pyright: ignore[reportPrivateUsage] - unit under test
)


def test_instance_failure_message_retains_last_runner_error() -> None:
    message = _instance_failure_message(
        "runner crashed repeatedly", "allocation failed on rank 1"
    )

    assert message == (
        "runner crashed repeatedly Last runner error: allocation failed on rank 1"
    )


def test_instance_failure_message_is_whitespace_normalized_and_bounded() -> None:
    message = _instance_failure_message("runner\n crashed", "x" * 4096)

    assert "\n" not in message
    assert len(message) == _INSTANCE_FAILURE_MESSAGE_LIMIT
