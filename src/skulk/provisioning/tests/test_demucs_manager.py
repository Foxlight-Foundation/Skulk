"""Preparation readiness survives actual ownership races without inventing compute."""

import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from skulk.provisioning.demucs_manager import DemucsRuntimeManager
from skulk.provisioning.demucs_runtime import (
    DemucsBuildMismatchError,
    DemucsDevice,
    DemucsPreparationCancelledError,
    DemucsWheel,
    PreparationPhase,
    PreparedDemucsRuntime,
)


def wheel() -> DemucsWheel:
    """Return data-only pins; these tests never execute or qualify package bytes."""
    return DemucsWheel(
        platform_tag="macosx_14_0_arm64",
        filename="skulk_demucs_cpu-0.1.0-cp312-cp312-macosx_14_0_arm64.whl",
        sha256="a" * 64, payload_sha256="b" * 64, size_bytes=100,
    )


class ControlledCache:
    """Replace cache effects while exercising the manager's real synchronization."""

    def __init__(self, root: Path) -> None:
        """Keep private observations, without processes, files or network access."""
        self.root = root
        self.calls: list[tuple[DemucsWheel, DemucsDevice, bool, float]] = []
        self.entered = threading.Event()
        self.release: threading.Event | None = None
        self.finished = threading.Event()
        self.after_probe: Callable[[], None] = lambda: None
        self.return_device: DemucsDevice | None = None
        self.return_wheel: DemucsWheel | None = None
        self.fail_prepare = False
        self.fail_close = False
        self.close_calls = 0
        self.generation = "1" * 32

    def prepare(self, selected: DemucsWheel, *, device: DemucsDevice, offline: bool,
                is_cancelled: Callable[[], bool], progress: Callable[[PreparationPhase], None],
                timeout_seconds: float = 30 * 60) -> PreparedDemucsRuntime:
        """Pause a controlled cache effect until cancellation or explicit release."""
        self.finished.clear()
        self.calls.append((selected, device, offline, timeout_seconds))
        self.entered.set()
        try:
            progress("probing")
            while self.release is not None and not self.release.wait(.01):
                if is_cancelled():
                    raise DemucsPreparationCancelledError("cache effect cancelled")
            if self.fail_prepare:
                raise ValueError("controlled verification failure")
            self.after_probe()
            generation = self.root / "generations" / self.generation
            return PreparedDemucsRuntime(
                wheel=self.return_wheel or selected, generation=generation,
                interpreter=generation / "runtime/bin/python", device=self.return_device or device,
            )
        finally:
            self.finished.set()

    def close(self) -> None:
        """Model real cache drain and one retained cleanup failure, when requested."""
        if self.calls and not self.finished.is_set():
            raise RuntimeError("cancel and join preparation before cache close")
        self.close_calls += 1
        if self.fail_close:
            self.fail_close = False
            raise RuntimeError("retained cache cleanup")


def manager(root: Path, cache: ControlledCache) -> DemucsRuntimeManager:
    """Use the real manager while injecting only the underlying cache effects."""
    def factory(selected_root: Path) -> ControlledCache:
        assert selected_root == root
        return cache

    return DemucsRuntimeManager(root, (wheel(),), frozenset({"macosx_14_0_arm64"}), create_cache=factory)


def prepare(owner: DemucsRuntimeManager, *, device: DemucsDevice = "cpu",
            cancelled: Callable[[], bool] = lambda: False, budget: float = 30.0) -> PreparedDemucsRuntime:
    """Issue an explicit offline request with finite whole-operation budget."""
    return owner.prepare(wheel(), device=device, offline=True, is_cancelled=cancelled,
                         progress=lambda _phase: None, timeout_seconds=budget)


def test_read_only_inventory_does_not_prepare_and_explicit_probes_preserve_both_lanes(tmp_path: Path) -> None:
    """Installability is never readiness; the same verified generation keeps CPU/MPS."""
    root = tmp_path / "not-created"
    cache = ControlledCache(root)
    owner = manager(root, cache)
    assert owner.engine_builds() == {} and cache.calls == [] and not root.exists()
    prepare(owner)
    assert owner.engine_builds() == {"demucs-cpu": wheel().build_identity}
    prepare(owner, device="mps")
    assert owner.engine_builds() == {"demucs-cpu": wheel().build_identity, "demucs-metal": wheel().build_identity}
    mutated = owner.engine_builds()
    mutated.clear()
    assert len(owner.engine_builds()) == 2
    assert all(call[2] and 0 < call[3] <= 30 for call in cache.calls)
    owner.close()
    assert owner.engine_builds() == {} and cache.close_calls == 1
    with pytest.raises(DemucsPreparationCancelledError):
        prepare(owner)


@pytest.mark.parametrize("change", ["pins", "host", "mps-linux"])
def test_wrong_package_host_or_unimplemented_device_never_runs_cache(tmp_path: Path, change: str) -> None:
    """Real ABI/device requirements cannot be bypassed by a caller's chosen pin."""
    root = tmp_path / "cache"
    cache = ControlledCache(root)
    selected = wheel()
    if change == "mps-linux":
        selected = DemucsWheel(platform_tag="manylinux_2_28_x86_64",
            filename="skulk_demucs_cpu-0.1.0-cp312-cp312-manylinux_2_28_x86_64.whl",
            sha256="a" * 64, payload_sha256="b" * 64, size_bytes=100)
    owner = DemucsRuntimeManager(root, (selected,),
        frozenset({"unrelated" if change == "host" else selected.platform_tag}), create_cache=lambda _root: cache)
    if change == "pins":
        selected = selected.model_copy(update={"payload_sha256": "f" * 64})
    with pytest.raises(DemucsBuildMismatchError):
        owner.prepare(selected, device="mps" if change == "mps-linux" else "cpu",
                      offline=True, is_cancelled=lambda: False, progress=lambda _phase: None)
    assert cache.calls == [] and owner.engine_builds() == {}
    owner.close()


def test_cancellation_after_native_probe_cannot_publish_late_readiness(tmp_path: Path) -> None:
    """The owner's final cancellation checkpoint covers native completion races."""
    root = tmp_path / "cache"
    cache = ControlledCache(root)
    owner = manager(root, cache)
    cancelled = threading.Event()
    cache.after_probe = cancelled.set
    with pytest.raises(DemucsPreparationCancelledError):
        prepare(owner, cancelled=cancelled.is_set)
    assert owner.engine_builds() == {}
    owner.close()


@pytest.mark.parametrize("change", ["device", "payload"])
def test_returned_native_receipt_must_match_admitted_device_and_build(tmp_path: Path, change: str) -> None:
    """A successful cache call cannot substitute another compute lane or payload."""
    root = tmp_path / "cache"
    cache = ControlledCache(root)
    cache.return_device = "mps" if change == "device" else None
    cache.return_wheel = wheel().model_copy(update={"payload_sha256": "f" * 64}) if change == "payload" else None
    owner = manager(root, cache)
    with pytest.raises(DemucsBuildMismatchError):
        prepare(owner)
    assert owner.engine_builds() == {}
    owner.close()


def test_whole_budget_includes_wait_without_cancelling_the_current_owner(tmp_path: Path) -> None:
    """A queued request expires without running another cache or killing active work."""
    root = tmp_path / "cache"
    cache = ControlledCache(root)
    cache.release = threading.Event()
    owner = manager(root, cache)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            active = pool.submit(prepare, owner)
            assert cache.entered.wait(2)
            try:
                with pytest.raises(TimeoutError):
                    prepare(owner, budget=.1)
                assert len(cache.calls) == 1 and owner.engine_builds() == {}
            finally:
                cache.release.set()
            assert active.result(timeout=2).device == "cpu"
        assert owner.engine_builds() == {"demucs-cpu": wheel().build_identity}
    finally:
        owner.close()


def test_shutdown_cancels_current_and_waiting_requests_and_drains_original_cache(tmp_path: Path) -> None:
    """No request may restore capacity after the worker begins shutdown."""
    root = tmp_path / "cache"
    cache = ControlledCache(root)
    cache.release = threading.Event()
    owner = manager(root, cache)
    with ThreadPoolExecutor(max_workers=2) as pool:
        active = pool.submit(prepare, owner)
        assert cache.entered.wait(2)
        waiting = pool.submit(prepare, owner)
        owner.close()
        for future in (active, waiting):
            with pytest.raises(DemucsPreparationCancelledError):
                future.result(timeout=2)
    assert len(cache.calls) == cache.close_calls == 1 and owner.engine_builds() == {}


def test_failed_close_withdraws_and_retains_the_same_cleanup_handle(tmp_path: Path) -> None:
    """Retry drains the original cache; it cannot create a replacement over live writers."""
    root = tmp_path / "cache"
    cache = ControlledCache(root)
    owner = manager(root, cache)
    prepare(owner)
    cache.fail_close = True
    with pytest.raises(RuntimeError, match="retained"):
        owner.close()
    assert owner.engine_builds() == {}
    with pytest.raises(DemucsPreparationCancelledError):
        prepare(owner)
    owner.close()
    owner.close()
    assert cache.close_calls == 2 and len(cache.calls) == 1


def test_failed_revalidation_withdraws_and_explicit_retry_can_restore_capacity(tmp_path: Path) -> None:
    """A preparation failure does not permanently blacklist a compatible machine."""
    root = tmp_path / "cache"
    cache = ControlledCache(root)
    owner = manager(root, cache)
    prepare(owner)
    cache.fail_prepare = True
    with pytest.raises(ValueError, match="verification"):
        prepare(owner)
    assert owner.engine_builds() == {}
    cache.fail_prepare = False
    prepare(owner)
    assert owner.engine_builds() == {"demucs-cpu": wheel().build_identity}
    # A repaired installation has no inherited MPS device observation.
    prepare(owner, device="mps")
    cache.generation = "2" * 32
    prepare(owner)
    assert owner.engine_builds() == {"demucs-cpu": wheel().build_identity}
    owner.close()


def test_exact_runtime_lookup_has_no_effect_and_rejects_withdrawn_or_wrong_build(tmp_path: Path) -> None:
    """Only a currently observed exact lane can cross the private runner boundary."""
    cache = ControlledCache(tmp_path)
    owner = manager(tmp_path, cache)
    with pytest.raises(DemucsBuildMismatchError):
        owner.prepared_runtime("demucs-cpu", wheel().build_identity)
    assert cache.calls == []
    prepared = prepare(owner)
    assert owner.prepared_runtime("demucs-cpu", wheel().build_identity) == prepared
    for backend, build in (("demucs-metal", wheel().build_identity), ("demucs-cuda", wheel().build_identity), ("demucs-cpu", "other-build")):
        with pytest.raises(DemucsBuildMismatchError):
            owner.prepared_runtime(backend, build)
    owner.close()
    with pytest.raises(DemucsBuildMismatchError):
        owner.prepared_runtime("demucs-cpu", wheel().build_identity)
    assert len(cache.calls) == 1
