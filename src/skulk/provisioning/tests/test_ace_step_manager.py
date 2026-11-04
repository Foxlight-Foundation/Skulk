"""Real sealed-cache preparation reaches resources only after exact verification."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from skulk.provisioning.ace_step_manager import AceStepRuntimeManager
from skulk.provisioning.ace_step_runtime import (
    AceStepRuntimeCache,
    AceStepWheel,
    PreparedAceStepRuntime,
)
from skulk.provisioning.demucs_runtime import (
    DemucsBuildMismatchError,
    DemucsPreparationCancelledError,
    PreparationPhase,
)
from skulk.provisioning.tests import test_ace_step_runtime as cache_fixtures
from skulk.shared.types.native_music import NativeMusicDevice

interpreter_fixture = cache_fixtures.interpreter_fixture
installed = cache_fixtures.installed
pytestmark = pytest.mark.parametrize("variant", ("cpu", "cuda", "rocm"))


class LocalCache:
    """Supply retained synthetic original bytes to the actual sealed cache."""

    def __init__(self, root: Path, controlled: cache_fixtures.NativeInstall) -> None:
        self.cache = AceStepRuntimeCache(root)
        self.controlled = controlled

    def prepare(
        self,
        wheel: AceStepWheel,
        /,
        *,
        device: NativeMusicDevice,
        offline: bool,
        is_cancelled: Callable[[], bool],
        progress: Callable[[PreparationPhase], None],
        timeout_seconds: float = 30 * 60,
    ) -> PreparedAceStepRuntime:
        """Retain actual identity/tree/probe validation and every lifetime fence."""
        return self.cache.prepare(
            wheel, device=device, offline=offline, is_cancelled=is_cancelled,
            progress=progress, timeout_seconds=timeout_seconds,
            download=self.controlled.download,
        )

    def close(self) -> None:
        """Drain retained preparation before releasing ownership."""
        self.cache.close()


def manager(root: Path, installed: cache_fixtures.NativeInstall) -> AceStepRuntimeManager:
    """Construct the production manager over real verification and a local transport."""
    return AceStepRuntimeManager(
        root, (installed.wheel,), installed.wheel.required_platform_tags,
        create_cache=lambda path: LocalCache(path, installed),
    )


def prepare(
    manager: AceStepRuntimeManager, installed: cache_fixtures.NativeInstall,
    *, offline: bool = False,
) -> PreparedAceStepRuntime:
    """Prepare this exact pinned build and lane with its real cancellation owner."""
    return manager.prepare(
        installed.wheel, device=installed.device, offline=offline,
        is_cancelled=lambda: installed.cancelled, progress=lambda _: None,
    )


def test_installable_is_not_ready_and_verified_observation_is_exact(
    tmp_path: Path, installed: cache_fixtures.NativeInstall,
) -> None:
    """Available pins have no side effects/resources until actual preparation."""
    root = tmp_path / "cache"
    runtime = manager(root, installed)
    assert runtime.engine_builds() == {} and not root.exists()
    backend = "ace_step-" + installed.device
    with pytest.raises(DemucsBuildMismatchError, match="no longer prepared"):
        runtime.prepared_runtime(backend, installed.wheel.build_identity)
    prepared = prepare(runtime, installed)
    assert runtime.engine_builds() == {backend: installed.wheel.build_identity}
    assert runtime.prepared_runtime(backend, installed.wheel.build_identity) == prepared
    assert prepare(runtime, installed, offline=True) == prepared
    assert installed.installs == 1 and installed.probes == 2
    runtime.close()
    assert runtime.engine_builds() == {} and prepared.generation.is_dir()


def test_failed_revalidation_withdraws_previous_resources(
    tmp_path: Path, installed: cache_fixtures.NativeInstall,
) -> None:
    """A prior successful probe cannot conceal a new mismatched observation."""
    runtime = manager(tmp_path / "cache", installed)
    previous = prepare(runtime, installed)
    installed.foreign_payload = True
    with pytest.raises(DemucsBuildMismatchError, match="different payload"):
        prepare(runtime, installed, offline=True)
    assert runtime.engine_builds() == {} and previous.generation.is_dir()
    installed.foreign_payload = False
    recovered = prepare(runtime, installed, offline=True)
    assert recovered.generation == previous.generation
    assert runtime.engine_builds()
    runtime.close()


def test_foreign_build_and_incompatible_lane_never_enter_cache(
    tmp_path: Path, installed: cache_fixtures.NativeInstall,
) -> None:
    """A caller cannot promote a different package or turn a profile into another lane."""
    root = tmp_path / "cache"
    runtime = manager(root, installed)
    wrong = installed.wheel.model_copy(update={"payload_sha256": "f" * 64})
    with pytest.raises(DemucsBuildMismatchError, match="fixed inventory"):
        runtime.prepare(
            wrong, device=installed.device, offline=False,
            is_cancelled=lambda: False, progress=lambda _: None,
        )
    with pytest.raises(DemucsBuildMismatchError, match="fixed inventory"):
        runtime.prepare(
            installed.wheel,
            device="cpu" if installed.device != "cpu" else "cuda",
            offline=False, is_cancelled=lambda: False, progress=lambda _: None,
        )
    assert not root.exists() and runtime.engine_builds() == {}
    assert (installed.downloads, installed.installs, installed.probes) == (0, 0, 0)
    runtime.close()


def test_cancelled_probe_cannot_create_late_node_resources(
    tmp_path: Path, installed: cache_fixtures.NativeInstall,
) -> None:
    """A valid observation after cancellation cannot bypass owner settlement."""
    runtime = manager(tmp_path / "cache", installed)
    installed.cancel_after_probe = True
    with pytest.raises(DemucsPreparationCancelledError):
        prepare(runtime, installed)
    assert installed.probes == 1 and runtime.engine_builds() == {}
    runtime.close()


def test_shutdown_refuses_new_preparation_and_stale_build_lookup(
    tmp_path: Path, installed: cache_fixtures.NativeInstall,
) -> None:
    """Shutdown withdraws observations without deleting an existing installation."""
    runtime = manager(tmp_path / "cache", installed)
    prepared = prepare(runtime, installed)
    backend = "ace_step-" + installed.device
    with pytest.raises(DemucsBuildMismatchError, match="no longer prepared"):
        runtime.prepared_runtime(backend, "ace_step@sha256:" + "f" * 64)
    runtime.close()
    with pytest.raises(DemucsPreparationCancelledError):
        prepare(runtime, installed)
    with pytest.raises(DemucsBuildMismatchError, match="no longer prepared"):
        runtime.prepared_runtime(backend, installed.wheel.build_identity)
    assert prepared.generation.is_dir() and runtime.engine_builds() == {}
