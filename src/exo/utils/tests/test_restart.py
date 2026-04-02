# pyright: reportUnusedFunction=false, reportAny=false, reportPrivateUsage=false
"""Tests for exo.utils.restart — process restart scheduling."""

from unittest.mock import MagicMock, patch

from exo.utils import restart


def _reset_restart_state() -> None:
    """Reset the module-level restart guard between tests."""
    with restart._restart_lock:
        restart._restart_scheduled = False


def test_schedule_restart_calls_execv() -> None:
    """schedule_restart should call os.execv to replace the process."""
    _reset_restart_state()

    with (
        patch.object(restart.os, "execv") as mock_execv,
        patch.object(restart.threading, "Thread") as mock_thread_cls,
    ):
        mock_thread = MagicMock()
        mock_thread_cls.return_value = mock_thread

        result = restart.schedule_restart(delay=0)
        assert result is True
        mock_thread.start.assert_called_once()

        # Extract and run the restart function directly (skip the sleep)
        target_fn = mock_thread_cls.call_args[1]["target"]
        with patch("time.sleep"):
            target_fn()

        mock_execv.assert_called_once()


def test_schedule_restart_uses_python_m_exo() -> None:
    """Restart argv should use 'python -m exo' so console scripts work."""
    _reset_restart_state()

    with (
        patch.object(restart.os, "execv") as mock_execv,
        patch.object(restart.sys, "executable", "/usr/bin/python3"),
        patch.object(restart.sys, "argv", ["exo", "--port", "8080"]),
        patch.object(restart.threading, "Thread") as mock_thread_cls,
    ):
        mock_thread = MagicMock()
        mock_thread_cls.return_value = mock_thread

        restart.schedule_restart(delay=0)
        target_fn = mock_thread_cls.call_args[1]["target"]
        with patch("time.sleep"):
            target_fn()

        mock_execv.assert_called_once_with(
            "/usr/bin/python3",
            ["/usr/bin/python3", "-m", "exo", "--port", "8080"],
        )


def test_schedule_restart_idempotent() -> None:
    """Calling schedule_restart twice should return False the second time."""
    _reset_restart_state()

    with patch.object(restart.threading, "Thread") as mock_thread_cls:
        mock_thread = MagicMock()
        mock_thread_cls.return_value = mock_thread

        assert restart.schedule_restart(delay=0) is True
        assert restart.schedule_restart(delay=0) is False
        # Only one thread should have been created
        assert mock_thread_cls.call_count == 1


def test_schedule_restart_recovers_on_execv_failure() -> None:
    """If execv fails, the guard should reset so restart can be retried."""
    _reset_restart_state()

    with (
        patch.object(
            restart.os, "execv", side_effect=OSError("exec failed")
        ) as mock_execv,
        patch.object(restart.threading, "Thread") as mock_thread_cls,
    ):
        mock_thread = MagicMock()
        mock_thread_cls.return_value = mock_thread

        restart.schedule_restart(delay=0)

        # Run the target function
        target_fn = mock_thread_cls.call_args[1]["target"]
        with patch("time.sleep"):
            target_fn()

        mock_execv.assert_called_once()

        # Guard should be reset so we can schedule again
        assert not restart._restart_scheduled
