"""Tests for the orphaned vLLM engine-core startup sweep (#653)."""

import pathlib

from skulk.worker.runner.vllm.orphan_sweep import find_orphaned_vllm_engine_pids


def _fake_proc(
    root: pathlib.Path, pid: int, comm: str, ppid: int, cmdline: str
) -> None:
    entry = root / str(pid)
    entry.mkdir()
    (entry / "stat").write_text(f"{pid} ({comm}) S {ppid} {pid} {pid} 0 -1")
    (entry / "cmdline").write_text(cmdline.replace(" ", "\x00"))


def test_find_orphaned_vllm_engine_pids(tmp_path: pathlib.Path) -> None:
    """Only engine cores reparented to init match; everything else survives.

    A healthy EngineCore has a live ``vllm serve`` parent, and a manually
    launched ``vllm serve`` under nohup legitimately has ppid 1 but is not an
    engine core, so neither may match: the sweep must be impossible to aim at
    anything an operator still wants.
    """
    # The observed leak shape: engine core reparented to init. comm is
    # kernel-truncated to 15 chars.
    _fake_proc(
        tmp_path, 101, "VLLM::EngineCor", 1, "VLLM::EngineCore busy_loop"
    )
    # Healthy engine core: parent is a live vllm serve, not init.
    _fake_proc(tmp_path, 102, "VLLM::EngineCor", 500, "VLLM::EngineCore idle")
    # Manually launched vllm serve under nohup: ppid 1 but not an engine core.
    _fake_proc(tmp_path, 103, "vllm", 1, "/opt/venv/bin/vllm serve model")
    # Unrelated init-parented daemon.
    _fake_proc(tmp_path, 104, "sshd", 1, "/usr/sbin/sshd -D")
    # Comm containing parens and spaces must not break stat parsing.
    _fake_proc(tmp_path, 105, "weird (name) x", 1, "weird daemon")
    # Non-pid entries and unreadable processes are skipped.
    (tmp_path / "self").mkdir()
    (tmp_path / "201").mkdir()  # no stat/cmdline: vanished mid-scan

    assert find_orphaned_vllm_engine_pids(tmp_path) == [101]


def test_find_orphans_no_proc_tree(tmp_path: pathlib.Path) -> None:
    """Platforms without /proc (macOS) sweep nothing rather than erroring."""
    assert find_orphaned_vllm_engine_pids(tmp_path / "missing") == []
