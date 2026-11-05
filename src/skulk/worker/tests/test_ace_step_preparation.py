"""Actual ACE cache/controller assembly retains explicit offline and owner policy."""

from pathlib import Path

import pytest

from skulk.provisioning.demucs_runtime import DemucsPreparationCancelledError
from skulk.provisioning.tests import test_ace_step_runtime as cache_fixtures
from skulk.provisioning.tests.test_ace_step_manager import manager
from skulk.shared.types.common import CommandId
from skulk.worker.ace_step_preparation import AceStepPreparationController

interpreter_fixture = cache_fixtures.interpreter_fixture
installed = cache_fixtures.installed
pytestmark = pytest.mark.parametrize("variant", ("cpu", "cuda", "rocm"))


async def test_actual_ace_cache_preparation_preserves_probe_and_withdrawal(
    tmp_path: Path,
    installed: cache_fixtures.NativeInstall,
) -> None:
    """Native math facts follow a real sealed-cache check and disappear on close."""
    controller = AceStepPreparationController(
        manager(tmp_path / "cache", installed), offline=False
    )
    assert controller.library_probes() == {} and installed.installs == 0
    prepared = await controller.prepare(
        CommandId("prepare-ace"),
        installed.wheel,
        device=installed.device,
        expires_at=controller.clock() + 30,
        progress=lambda _: None,
    )
    backend = "ace_step-" + installed.device
    assert controller.engine_builds() == {backend: installed.wheel.build_identity}
    assert controller.library_probes() == {backend: prepared.probe}
    assert installed.installs == 1 and installed.probes == 1
    await controller.close()
    assert controller.library_probes() == {} and controller.engine_builds() == {}
    assert prepared.generation.is_dir()


async def test_cancelled_ace_preparation_never_installs_or_probes(
    tmp_path: Path,
    installed: cache_fixtures.NativeInstall,
) -> None:
    """Cancellation arriving before its ordered request cannot cause late native work."""
    controller = AceStepPreparationController(
        manager(tmp_path / "cache", installed), offline=True
    )
    identifier = CommandId("cancel-before-prepare")
    controller.cancel(identifier)
    with pytest.raises(DemucsPreparationCancelledError):
        await controller.prepare(
            identifier,
            installed.wheel,
            device=installed.device,
            expires_at=controller.clock() + 30,
            progress=lambda _: None,
        )
    assert (
        installed.installs == 0 and installed.probes == 0 and installed.downloads == 0
    )
    assert controller.library_probes() == {}
    await controller.close()
