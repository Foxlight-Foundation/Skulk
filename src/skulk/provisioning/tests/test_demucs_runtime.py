"""Cache policy uses sealed synthetic bytes; candidate Python never executes."""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
import zipfile
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import httpx
import pytest

from skulk.provisioning import demucs_runtime as runtime
from skulk.provisioning.demucs_package import Checkpoint, CommandRunner
from skulk.provisioning.native_install_process import NativeInstallCommandRunner
from skulk.provisioning.tests import test_demucs_package as package_fixtures
from skulk.utils.runtime_files import RuntimeLock, private_directory

# Reuse data-only fixtures without importing names that pytest parameters shadow.
interpreter_fixture = package_fixtures.interpreter_fixture
cached_fixture = package_fixtures.cached_fixture


class DiskUsage(NamedTuple):
    """Controlled free space without allocating large files in unit tests."""

    total: int
    used: int
    free: int


@dataclass
class ControlledNativeInstall:
    """Replace native execution, retaining real package/tree integrity checks."""

    source: Path
    original: Path
    wheel: runtime.DemucsWheel
    free_bytes: int = 100 * 1024**3
    downloads: int = 0
    installs: int = 0
    probes: int = 0
    cancelled: bool = False
    requested: runtime.DemucsDevice = "cpu"
    native: runtime.DemucsDevice = "cpu"
    reported_payload: str | None = None
    fail_probe: bool = False
    cancel_after_install: bool = False
    cancel_after_probe: bool = False
    entered: threading.Event = field(default_factory=threading.Event)
    release: threading.Event | None = None

    def download(
        self, wheel: runtime.DemucsWheel, destination: Path, checkpoint: Checkpoint
    ) -> None:
        """Copy the exact tiny original object; no network or package resolution."""
        assert wheel.sha256 == self.wheel.sha256
        self.downloads += 1
        checkpoint()
        shutil.copyfile(self.source, destination)
        checkpoint()

    def bootstrap(
        self,
        package: Path,
        output: Path,
        expected_sha256: str,
        *,
        run: CommandRunner,
        checkpoint: Checkpoint | None = None,
    ) -> Path:
        """Copy verified synthetic trees to their final path, never run their code."""
        assert run is not None
        assert hashlib.sha256(package.read_bytes()).hexdigest() == expected_sha256
        self.installs += 1
        shutil.copytree(self.original, output, symlinks=True)
        base = output / "interpreter/python/bin/python3.12"
        native = output / "runtime/bin/python"
        native.unlink()
        native.symlink_to(base)
        (output / "runtime/pyvenv.cfg").write_text(
            f"home = {base.parent}\ninclude-system-site-packages = false\n"
            f"version = 3.12.13\nexecutable = {base}\n"
        )
        self.entered.set()
        if self.release is not None:
            assert self.release.wait(5)
        if self.cancel_after_install:
            self.cancelled = True
        if checkpoint is not None:
            checkpoint()
        return native


@pytest.fixture
def controlled(
    cached_fixture: tuple[Path, Path, str, str], monkeypatch: pytest.MonkeyPatch
) -> ControlledNativeInstall:
    """Keep real byte custody and fake only native execution and host observations."""
    source, original, digest, payload = cached_fixture
    wheel = runtime.DemucsWheel(
        platform_tag="macosx_14_0_arm64",
        filename="skulk_demucs_cpu-0.1.0-cp312-cp312-macosx_14_0_arm64.whl",
        sha256=digest,
        payload_sha256=payload,
        size_bytes=source.stat().st_size,
    )
    controlled = ControlledNativeInstall(source=source, original=original, wheel=wheel)
    monkeypatch.setattr(runtime, "platform_tags", lambda: iter(("macosx_14_0_arm64",)))

    def capacity(_path: Path) -> DiskUsage:
        return DiskUsage(200 * 1024**3, 0, controlled.free_bytes)

    monkeypatch.setattr(
        runtime,
        "shutil",
        SimpleNamespace(
            disk_usage=capacity,
            rmtree=shutil.rmtree,
        ),
    )
    monkeypatch.setattr(runtime, "install_fresh_runtime", controlled.bootstrap)

    def probe(
        _self: NativeInstallCommandRunner,
        arguments: Sequence[str],
        *,
        environment: dict[str, str],
        timeout_seconds: int,
        capture: bool,
    ) -> str:
        assert arguments[1:3] == ["-I", "-B"]
        assert f"requested={controlled.requested!r}" in arguments[-1]
        assert capture and timeout_seconds == 300
        assert environment["PYTORCH_ENABLE_MPS_FALLBACK"] == "0"
        assert environment["HF_HUB_OFFLINE"] == "1"
        assert "PYTHONPATH" not in environment
        controlled.probes += 1
        if controlled.fail_probe:
            raise ValueError("controlled native probe failure")
        if controlled.cancel_after_probe:
            controlled.cancelled = True
        return json.dumps(
            {
                "payload_sha256": controlled.reported_payload
                or controlled.wheel.payload_sha256,
                "requested": controlled.requested,
                "native": controlled.native,
            }
        )

    monkeypatch.setattr(NativeInstallCommandRunner, "__call__", probe)
    return controlled


def prepare(
    cache: runtime.DemucsRuntimeCache,
    controlled: ControlledNativeInstall,
    *,
    offline: bool = False,
    device: runtime.DemucsDevice = "cpu",
) -> runtime.PreparedDemucsRuntime:
    """Invoke the real cache controller with controlled effects and sealed fixtures."""
    return cache.prepare(
        controlled.wheel,
        device=device,
        offline=offline,
        is_cancelled=lambda: controlled.cancelled,
        progress=lambda _: None,
        download=controlled.download,
    )


def seed_package(root: Path, controlled: ControlledNativeInstall) -> Path:
    """Retain complete original bytes with no installed generation or receipt."""
    objects = root / controlled.wheel.sha256 / "objects"
    private_directory(root)
    private_directory(objects.parent)
    private_directory(objects)
    destination = objects / controlled.wheel.filename
    shutil.copyfile(controlled.source, destination)
    return destination


def test_exact_offline_reuse_checks_tree_and_reprobes_device(
    tmp_path: Path, controlled: ControlledNativeInstall
) -> None:
    """Selecting a complete cache never downloads or replays installation."""
    cache = runtime.DemucsRuntimeCache(tmp_path / "cache")
    first = prepare(cache, controlled)
    second = prepare(cache, controlled, offline=True)
    assert first == second
    assert (controlled.downloads, controlled.installs, controlled.probes) == (1, 1, 2)
    assert (
        first.interpreter.resolve()
        == first.generation / "interpreter/python/bin/python3.12"
    )
    cache.close()


def test_offline_can_bootstrap_from_verified_object_without_network(
    tmp_path: Path, controlled: ControlledNativeInstall
) -> None:
    """A sealed cached object suffices even when a previous install never started."""
    root = tmp_path / "cache"
    seed_package(root, controlled)
    result = prepare(runtime.DemucsRuntimeCache(root), controlled, offline=True)
    assert result.interpreter.is_file()
    assert (controlled.downloads, controlled.installs, controlled.probes) == (0, 1, 1)


@pytest.mark.parametrize("corrupt", (False, True))
def test_missing_or_corrupt_offline_package_never_downloads(
    tmp_path: Path, controlled: ControlledNativeInstall, corrupt: bool
) -> None:
    """Offline errors are actionable rather than an implicit remote fetch."""
    root = tmp_path / "cache"
    if corrupt:
        seed_package(root, controlled).write_bytes(b"changed")
    with pytest.raises(ValueError, match="offline.*intact pinned package"):
        prepare(runtime.DemucsRuntimeCache(root), controlled, offline=True)
    assert (controlled.downloads, controlled.installs, controlled.probes) == (0, 0, 0)


def test_whole_object_and_payload_pins_agree_before_any_execution(
    tmp_path: Path, controlled: ControlledNativeInstall
) -> None:
    """A valid package cannot authorize a differently selected engine payload."""
    root = tmp_path / "cache"
    seed_package(root, controlled)
    wrong = controlled.wheel.model_copy(update={"payload_sha256": "f" * 64})
    with pytest.raises(runtime.DemucsBuildMismatchError):
        runtime.DemucsRuntimeCache(root).prepare(
            wrong,
            device="cpu",
            offline=False,
            is_cancelled=lambda: False,
            progress=lambda _: None,
            download=controlled.download,
        )
    assert (controlled.downloads, controlled.installs, controlled.probes) == (0, 0, 0)


@pytest.mark.parametrize(
    "missing",
    ["audio.separate", "four_stems", "pcm24", "offline_loader", "bundled_python"],
)
def test_required_runtime_features_are_checked_before_execution(
    tmp_path: Path, controlled: ControlledNativeInstall, missing: str
) -> None:
    """A correctly sealed but incomplete ABI cannot start even its interpreter."""
    variant = tmp_path / "missing-feature.whl"
    with (
        zipfile.ZipFile(controlled.source) as source,
        zipfile.ZipFile(variant, "w") as target,
    ):
        manifest = runtime.BuildManifest.model_validate_json(
            source.read("skulk_demucs/build.json")
        )
        content = manifest.model_copy(
            update={
                "features": tuple(
                    value for value in manifest.features if value != missing
                )
            }
        ).model_dump(mode="json", exclude={"payload_sha256"})
        payload = hashlib.sha256(
            json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        for entry in source.infolist():
            target.writestr(
                entry,
                json.dumps(content | {"payload_sha256": payload}).encode()
                if entry.filename == "skulk_demucs/build.json"
                else source.read(entry),
            )
    controlled.source = variant
    controlled.wheel = controlled.wheel.model_copy(
        update={
            "sha256": hashlib.sha256(variant.read_bytes()).hexdigest(),
            "payload_sha256": payload,
            "size_bytes": variant.stat().st_size,
        }
    )
    root = tmp_path / "cache"
    seed_package(root, controlled)
    with pytest.raises(runtime.DemucsBuildMismatchError):
        prepare(runtime.DemucsRuntimeCache(root), controlled, offline=True)
    assert (controlled.downloads, controlled.installs, controlled.probes) == (0, 0, 0)


def test_bad_cached_runtime_gets_new_generation_preserving_old(
    tmp_path: Path, controlled: ControlledNativeInstall
) -> None:
    """Repair never overwrites a path that an existing mounted instance may use."""
    cache = runtime.DemucsRuntimeCache(tmp_path / "cache")
    first = prepare(cache, controlled)
    changed = (
        first.generation
        / "runtime/lib/python3.12/site-packages/skulk_demucs/__init__.py"
    )
    changed.write_bytes(b"foreign code")
    second = prepare(cache, controlled, offline=True)
    assert second.generation != first.generation
    assert changed.read_bytes() == b"foreign code"
    assert first.generation.is_dir() and second.generation.is_dir()
    assert (controlled.downloads, controlled.installs) == (1, 2)


def test_unpublished_complete_generation_is_recovered_without_reinstall(
    tmp_path: Path, controlled: ControlledNativeInstall
) -> None:
    """Power loss before publication leaves a recoverable stable private venv."""
    root = tmp_path / "cache"
    first = prepare(runtime.DemucsRuntimeCache(root), controlled)
    (root / controlled.wheel.sha256 / "selected.json").unlink()
    recovered = prepare(runtime.DemucsRuntimeCache(root), controlled, offline=True)
    assert recovered == first
    assert controlled.installs == 1 and controlled.probes == 2


def test_mps_probe_cannot_silently_fall_back_and_change_cpu_selection(
    tmp_path: Path, controlled: ControlledNativeInstall
) -> None:
    """Unsupported requested hardware leaves the prior healthy CPU cache intact."""
    root = tmp_path / "cache"
    cache = runtime.DemucsRuntimeCache(root)
    first = prepare(cache, controlled)
    pointer = root / controlled.wheel.sha256 / "selected.json"
    original = pointer.read_bytes()
    controlled.requested = "mps"
    with pytest.raises(ValueError, match="different build or device"):
        prepare(cache, controlled, offline=True, device="mps")
    assert pointer.read_bytes() == original
    assert first.interpreter.is_file() and controlled.installs == 1


@pytest.mark.parametrize("during_install", (False, True))
def test_cancellation_before_activation_cleans_only_new_generation(
    tmp_path: Path, controlled: ControlledNativeInstall, during_install: bool
) -> None:
    """Cancellation drains the owner before cleaning an unpublished installation."""
    root = tmp_path / "cache"
    controlled.cancel_after_install = during_install
    controlled.cancel_after_probe = not during_install
    with pytest.raises(runtime.DemucsPreparationCancelledError):
        prepare(runtime.DemucsRuntimeCache(root), controlled)
    build = root / controlled.wheel.sha256
    assert not (build / "selected.json").exists()
    assert not tuple((build / "generations").iterdir())
    assert (build / "objects" / controlled.wheel.filename).is_file()


def test_post_replace_fsync_failure_preserves_possibly_published_generation(
    tmp_path: Path, controlled: ControlledNativeInstall, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ambiguous commit cannot delete a generation already selected by others."""
    original = runtime.write_private
    failed = False

    def write(path: Path, content: bytes) -> None:
        nonlocal failed
        original(path, content)
        if path.name == "selected.json" and not failed:
            failed = True
            raise OSError("controlled directory sync failure after replacement")

    monkeypatch.setattr(runtime, "write_private", write)
    cache = runtime.DemucsRuntimeCache(tmp_path / "cache")
    with pytest.raises(OSError, match="after replacement"):
        prepare(cache, controlled)
    recovered = prepare(cache, controlled, offline=True)
    assert recovered.interpreter.is_file()
    assert controlled.installs == 1


def test_failed_cleanup_retains_runner_for_explicit_close_retry(
    tmp_path: Path, controlled: ControlledNativeInstall, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed drain leaves the exact owner available and preserves partial files."""
    original = NativeInstallCommandRunner.drain
    attempts: list[NativeInstallCommandRunner] = []

    def drain(runner: NativeInstallCommandRunner) -> None:
        attempts.append(runner)
        if len(attempts) == 1:
            raise PermissionError("controlled drain failure")
        original(runner)

    monkeypatch.setattr(NativeInstallCommandRunner, "drain", drain)
    controlled.fail_probe = True
    root = tmp_path / "cache"
    cache = runtime.DemucsRuntimeCache(root)
    with pytest.raises(PermissionError, match="drain failure"):
        prepare(cache, controlled)
    generations = root / controlled.wheel.sha256 / "generations"
    assert len(tuple(generations.iterdir())) == 1
    cache.close()
    assert attempts[0] is attempts[1]
    assert not tuple(generations.iterdir())


def test_disk_admission_refuses_before_bootstrap(
    tmp_path: Path, controlled: ControlledNativeInstall
) -> None:
    """An available sealed object does not grant space for an expanded runtime."""
    controlled.free_bytes = 11 * 1024**3
    with pytest.raises(runtime.DemucsStorageError, match="fresh isolated"):
        prepare(runtime.DemucsRuntimeCache(tmp_path / "cache"), controlled)
    assert controlled.installs == 0


def test_incompatible_abi_refuses_before_cache_or_download(
    tmp_path: Path, controlled: ControlledNativeInstall, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Eligibility uses the platform ABI, not package existence or a model claim."""
    monkeypatch.setattr(runtime, "platform_tags", lambda: iter(()))
    root = tmp_path / "cache"
    with pytest.raises(ValueError, match="incompatible.*ABI"):
        prepare(runtime.DemucsRuntimeCache(root), controlled)
    assert not root.exists() and controlled.downloads == 0


def test_two_controllers_serialize_one_installation(
    tmp_path: Path, controlled: ControlledNativeInstall
) -> None:
    """Filesystem fencing prevents duplicate installs across independent owners."""
    root = tmp_path / "cache"
    controlled.release = threading.Event()
    second_entered = threading.Event()
    first = runtime.DemucsRuntimeCache(root)
    second = runtime.DemucsRuntimeCache(root)
    with ThreadPoolExecutor(max_workers=2) as executor:
        created = executor.submit(prepare, first, controlled)
        try:
            assert controlled.entered.wait(5)
            waiting = executor.submit(
                second.prepare,
                controlled.wheel,
                device="cpu",
                offline=True,
                is_cancelled=lambda: False,
                progress=lambda phase: second_entered.set()
                if phase == "locking"
                else None,
                download=controlled.download,
            )
            assert second_entered.wait(5)
            assert not waiting.done()
        finally:
            controlled.release.set()
        assert created.result(timeout=10) == waiting.result(timeout=10)
    assert controlled.downloads == 1 and controlled.installs == 1


def test_waiting_for_another_owner_is_cancellable(
    tmp_path: Path, controlled: ControlledNativeInstall
) -> None:
    """An install fence never turns cancellation into an unbounded wait."""
    root = tmp_path / "cache"
    held = RuntimeLock(root)
    try:
        cache = runtime.DemucsRuntimeCache(root)

        def progress(phase: runtime.PreparationPhase) -> None:
            if phase == "locking":
                controlled.cancelled = True

        with pytest.raises(runtime.DemucsPreparationCancelledError):
            cache.prepare(
                controlled.wheel,
                device="cpu",
                offline=False,
                is_cancelled=lambda: controlled.cancelled,
                progress=progress,
                download=controlled.download,
            )
    finally:
        held.close()
    assert controlled.downloads == 0


@pytest.mark.parametrize(
    "mode",
    (
        "normal",
        "over",
        "under",
        "digest",
        "encoding",
        "header",
        "duplicate-header",
        "redirect",
    ),
)
def test_download_binds_exact_bytes_and_refuses_transport_variants(
    tmp_path: Path, controlled: ControlledNativeInstall, mode: str
) -> None:
    """No compressed, redirected, truncated or corrupt body becomes a package."""
    original = controlled.source.read_bytes()
    body = original
    headers: list[tuple[str, str]] = []
    status = 200
    if mode == "over":
        body += b"x"
    elif mode == "under":
        body = body[:-1]
    elif mode == "digest":
        body = bytes((body[0] ^ 1,)) + body[1:]
    elif mode == "encoding":
        headers.append(("Content-Encoding", "gzip"))
    elif mode == "header":
        headers.append(("Content-Length", str(len(body) + 1)))
    elif mode == "duplicate-header":
        headers = [("Content-Length", str(len(body)))] * 2
    elif mode == "redirect":
        status = 302
        headers.append(("Location", "https://unexpected.invalid/package"))
    else:
        headers.append(("Content-Length", str(len(body))))
    observed: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        observed.append(str(request.url))
        assert request.headers["Accept-Encoding"] == "identity"
        return httpx.Response(status, headers=headers, stream=httpx.ByteStream(body))

    destination = tmp_path / "download"
    transport = httpx.MockTransport(respond)
    if mode == "normal":
        runtime.download_demucs_package(
            controlled.wheel, destination, lambda: None, transport=transport
        )
        assert destination.read_bytes() == original
    else:
        with pytest.raises((ValueError, httpx.HTTPStatusError)):
            runtime.download_demucs_package(
                controlled.wheel, destination, lambda: None, transport=transport
            )
        assert (
            not destination.exists()
            or destination.stat().st_size <= controlled.wheel.size_bytes
        )
    assert observed == [controlled.wheel.url]


def test_download_failure_removes_only_its_partial_object(
    tmp_path: Path, controlled: ControlledNativeInstall
) -> None:
    """A failed transfer never replaces a prior object or advertises an engine."""
    root = tmp_path / "cache"
    package = seed_package(root, controlled)
    package.write_bytes(b"old corrupt object")

    def broken(
        _wheel: runtime.DemucsWheel, destination: Path, _checkpoint: Checkpoint
    ) -> None:
        destination.write_bytes(b"partial transfer")
        raise OSError("controlled transport interruption")

    cache = runtime.DemucsRuntimeCache(root)
    with pytest.raises(OSError, match="transport interruption"):
        cache.prepare(
            controlled.wheel,
            device="cpu",
            offline=False,
            is_cancelled=lambda: False,
            progress=lambda _: None,
            download=broken,
        )
    assert package.read_bytes() == b"old corrupt object"
    assert not tuple(package.parent.glob(".download-*"))
    assert controlled.installs == 0


def test_corrupt_cached_object_is_replaced_only_by_verified_bytes(
    tmp_path: Path, controlled: ControlledNativeInstall
) -> None:
    """An online retry can repair the original package without changing its pin."""
    root = tmp_path / "cache"
    package = seed_package(root, controlled)
    package.write_bytes(b"corrupt")
    result = prepare(runtime.DemucsRuntimeCache(root), controlled)
    assert package.read_bytes() == controlled.source.read_bytes()
    assert result.interpreter.is_file() and controlled.downloads == 1


def test_bad_selection_hint_cannot_escape_owned_generations(
    tmp_path: Path, controlled: ControlledNativeInstall
) -> None:
    """A local receipt cannot authorize a caller-selected path outside the cache."""
    root = tmp_path / "cache"
    cache = runtime.DemucsRuntimeCache(root)
    first = prepare(cache, controlled)
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "retain").write_bytes(b"unrelated")
    pointer = root / controlled.wheel.sha256 / "selected.json"
    pointer.write_text(
        json.dumps(
            {
                "generation": "../../foreign",
                "package_sha256": controlled.wheel.sha256,
                "payload_sha256": controlled.wheel.payload_sha256,
            }
        )
    )
    recovered = prepare(cache, controlled, offline=True)
    assert recovered == first and controlled.installs == 1
    assert (foreign / "retain").read_bytes() == b"unrelated"


@pytest.mark.parametrize("timeout", (0.0, -1.0, float("inf"), float("nan"), 7201.0))
def test_invalid_deadline_refuses_before_storage(
    tmp_path: Path, controlled: ControlledNativeInstall, timeout: float
) -> None:
    """A caller cannot disable the total preparation deadline."""
    root = tmp_path / "cache"
    with pytest.raises(ValueError, match="finite.*two hours"):
        runtime.DemucsRuntimeCache(root).prepare(
            controlled.wheel,
            device="cpu",
            offline=False,
            is_cancelled=lambda: False,
            progress=lambda _: None,
            timeout_seconds=timeout,
            download=controlled.download,
        )
    assert not root.exists() and controlled.downloads == 0
