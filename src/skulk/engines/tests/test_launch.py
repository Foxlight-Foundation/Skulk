"""The launch contract, exercised with a really installed package and a real adapter."""

import stat
import sys
import time
from dataclasses import replace
from pathlib import Path

import psutil
import pytest

from skulk.engines.adapter import AdapterFailedError
from skulk.engines.launch import (
    ADAPTER_PATH,
    adapter_session,
    engine_state_directory,
    plan_adapter_launch,
)
from skulk.engines.tests.package_fixtures import install_test_package
from skulk.provisioning.engine_install import verify_installed_engine
from skulk.provisioning.engine_packages import EnginePackageRefusedError


def _gone(pid: int, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                return True
        except psutil.NoSuchProcess:
            return True
        time.sleep(0.05)
    return False


def test_an_installed_adapter_runs_with_only_the_contract_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The adapter sees the package's own locations and nothing of the host's."""
    monkeypatch.setenv("SKULK_HOST_SECRET", "must not leak")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/host/lib")
    monkeypatch.setenv("DYLD_LIBRARY_PATH", "/host/lib")
    installed = install_test_package(tmp_path, tmp_path / "packages")
    state = engine_state_directory(installed.digest, root=tmp_path / "state")

    launch = plan_adapter_launch(installed, state_directory=state)
    with adapter_session(launch) as session:
        assert session.version == 1
        described = session.describe()

    seen = set(described.model_families)
    assert {
        "PATH",
        "HOME",
        "TMPDIR",
        "XDG_CACHE_HOME",
        "SKULK_ENGINE_ID",
        "SKULK_ENGINE_LANE",
        "SKULK_ENGINE_BUILD_DIGEST",
        "SKULK_ENGINE_GENERATION",
        "SKULK_ENGINE_EXECUTABLE",
        "SKULK_ENGINE_STATE",
    } <= seen
    assert not {"SKULK_HOST_SECRET", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"} & seen
    assert Path(described.engine_version).resolve() == state.resolve()
    assert launch.environment["PATH"] == ADAPTER_PATH
    assert launch.environment["SKULK_ENGINE_EXECUTABLE"] == str(installed.executable)
    assert launch.command[:3] == (sys.executable, "-I", "-B")


def test_the_adapter_is_told_where_the_pinned_requirements_are(tmp_path: Path) -> None:
    """Kept whole or unpacked, a requirement is found under one folder."""
    installed = install_test_package(tmp_path, tmp_path / "packages")
    state = engine_state_directory(installed.digest, root=tmp_path / "state")
    folder = installed.root / "requirements"

    without = plan_adapter_launch(installed, state_directory=state)
    unpacked = plan_adapter_launch(
        replace(installed, requirements={"vendor-1.0.whl": folder / "vendor"}),
        state_directory=state,
    )
    whole = plan_adapter_launch(
        replace(installed, requirements={"vendor-1.0.whl": folder / "vendor-1.0.whl"}),
        state_directory=state,
    )

    assert "SKULK_ENGINE_REQUIREMENTS" not in without.environment
    assert unpacked.environment["SKULK_ENGINE_REQUIREMENTS"] == str(folder)
    assert whole.environment["SKULK_ENGINE_REQUIREMENTS"] == str(folder)


def test_the_generation_is_never_written_and_the_log_is_private(tmp_path: Path) -> None:
    """Running an adapter leaves the installed files exactly as installed."""
    installed = install_test_package(tmp_path, tmp_path / "packages")
    state = tmp_path / "state" / installed.digest

    launch = plan_adapter_launch(installed, state_directory=state)
    with adapter_session(launch) as session:
        session.describe()

    assert verify_installed_engine(installed.root).digest == installed.digest
    assert b"adapter started" in launch.log_path.read_bytes()
    assert stat.S_IMODE(launch.log_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(state.stat().st_mode) == 0o700


def test_leaving_the_block_stops_the_adapter_even_after_an_error(
    tmp_path: Path,
) -> None:
    """A caller's own failure cannot leave an adapter running."""
    installed = install_test_package(tmp_path, tmp_path / "packages")
    launch = plan_adapter_launch(installed, state_directory=tmp_path / "state")
    process_id: int | None = None

    with (
        pytest.raises(RuntimeError, match="caller failed"),
        adapter_session(launch) as session,
    ):
        process_id = session.process_id
        raise RuntimeError("caller failed")

    assert process_id is not None and _gone(process_id)


def test_a_manifest_outside_the_protocol_window_is_refused(tmp_path: Path) -> None:
    """A package written only for newer protocols does not start on this build."""
    installed = install_test_package(tmp_path, tmp_path / "packages", protocol=(2, 3))
    with pytest.raises(EnginePackageRefusedError) as refused:
        plan_adapter_launch(installed, state_directory=tmp_path / "state")
    assert refused.value.code == "protocol_unsupported"


def test_the_interpreter_range_is_checked_again_at_launch(tmp_path: Path) -> None:
    """An interpreter that changed under an installed package is a clear refusal."""
    installed = install_test_package(tmp_path, tmp_path / "packages")
    with pytest.raises(EnginePackageRefusedError) as refused:
        plan_adapter_launch(
            installed, state_directory=tmp_path / "state", python_version="3.11.9"
        )
    assert refused.value.code == "adapter_python_unsupported"


def test_a_packaged_skulk_refuses_a_python_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A frozen application's executable is not an interpreter for adapter files."""
    installed = install_test_package(tmp_path, tmp_path / "packages")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    with pytest.raises(EnginePackageRefusedError) as refused:
        plan_adapter_launch(installed, state_directory=tmp_path / "state")
    assert refused.value.code == "adapter_python_unsupported"


def test_an_adapter_that_cannot_start_is_reported(tmp_path: Path) -> None:
    """A missing interpreter is a start failure, not a hang."""
    installed = install_test_package(tmp_path, tmp_path / "packages")
    launch = plan_adapter_launch(
        installed,
        state_directory=tmp_path / "state",
        interpreter=tmp_path / "no-such-python",
    )
    with pytest.raises(AdapterFailedError) as failed, adapter_session(launch):
        pass
    assert failed.value.reason == "exited"
