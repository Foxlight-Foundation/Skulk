"""Native ACE preparation keeps real byte custody while controlling execution.

Synthetic interpreter and package contents deliberately cannot execute. GPU
parameter cases prove profile/receipt policy only, never hardware qualification.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import stat
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import pytest

from skulk.provisioning import ace_step_runtime as ace
from skulk.provisioning import demucs_runtime as cache_runtime
from skulk.provisioning.demucs_package import (
    Checkpoint,
    CommandRunner,
    NativePackageLimits,
    NativePackageVariant,
    PythonArchivePin,
    native_package_limits,
    read_python_catalog,
    unpack_candidate,
)
from skulk.provisioning.native_install_process import NativeInstallCommandRunner
from skulk.provisioning.tests import test_demucs_package as package_fixtures
from skulk.shared.types.native_music import (
    NativeMusicDevice,
    NativeMusicLibraryProbe,
    NativeMusicMemoryCapacity,
)

interpreter_fixture = package_fixtures.interpreter_fixture
pytestmark = pytest.mark.parametrize("variant", ("cpu", "cuda", "rocm"))


class DiskUsage(NamedTuple):
    """An injected filesystem observation without allocating gigabyte fixtures."""

    total: int
    used: int
    free: int


@dataclass
class NativeInstall:
    """Keep real original-byte verification; replace only native execution."""

    source: Path
    template: Path
    wheel: ace.AceStepWheel
    device: NativeMusicDevice
    free_bytes: int = 100 * 1024**3
    downloads: int = 0
    installs: int = 0
    probes: int = 0
    cancelled: bool = False
    cancel_after_probe: bool = False
    foreign_payload: bool = False
    foreign_device: bool = False
    missing_memory: bool = False
    unsampled_memory: bool = False

    def download(
        self, wheel: ace.AceStepWheel, destination: Path, checkpoint: Checkpoint
    ) -> None:
        """Copy the exact controlled original; no network or dependency solver."""
        assert wheel == self.wheel
        self.downloads += 1
        checkpoint()
        shutil.copyfile(self.source, destination)

    def install(
        self,
        package: Path,
        output: Path,
        expected_sha256: str,
        *,
        run: CommandRunner,
        checkpoint: Checkpoint | None = None,
        limits: NativePackageLimits,
    ) -> Path:
        """Copy the synthetic tree; the real cache verifier still checks it all."""
        assert run is not None
        assert hashlib.sha256(package.read_bytes()).hexdigest() == expected_sha256
        assert limits == native_package_limits(self.wheel.variant)
        self.installs += 1
        shutil.copytree(self.template, output, symlinks=True)
        base = output / "interpreter/python/bin/python3.12"
        native = output / "runtime/bin/python"
        native.unlink()
        native.symlink_to(base)
        (output / "runtime/pyvenv.cfg").write_text(
            f"home = {base.parent}\ninclude-system-site-packages = false\n"
            f"version = 3.12.13\nexecutable = {base}\n"
        )
        if checkpoint is not None:
            checkpoint()
        return native

    def probe(
        self,
        arguments: Sequence[str],
        *,
        environment: dict[str, str],
        timeout_seconds: int,
        capture: bool,
    ) -> str:
        """Return a controlled observation after checking the real launch policy."""
        assert arguments[1:4] == ["-I", "-B", "-c"]
        assert "--serve" not in arguments
        assert f"probe_libraries({self.device!r}" in arguments[-1]
        assert environment["HF_HUB_OFFLINE"] == "1"
        assert environment["TRANSFORMERS_OFFLINE"] == "1"
        assert environment["PYTORCH_ENABLE_MPS_FALLBACK"] == "0"
        assert environment["MIOPEN_FIND_MODE"] == "FAST"
        assert "PYTHONPATH" not in environment and capture and timeout_seconds == 300
        self.probes += 1
        if self.cancel_after_probe:
            self.cancelled = True
        device = "mps" if self.foreign_device else self.device
        return NativeMusicLibraryProbe(
            payload_sha256="f" * 64
            if self.foreign_payload
            else self.wheel.payload_sha256,
            device=device,
            precision="bfloat16" if device == "cuda" else "float32",
            autoencoder_precision=(
                "float16"
                if device == "mps"
                else "bfloat16"
                if device == "cuda"
                else "float32"
            ),
            cuda_compute_capability=(12, 1) if device == "cuda" else None,
            torch_version="2.10.0",
            memory_capacity=None
            if self.missing_memory
            else NativeMusicMemoryCapacity(
                sampled_at_unix_ms=0 if self.unsampled_memory else 1,
                host_total_bytes=8 * 1024**3,
                host_available_bytes=6 * 1024**3,
                device_index=None if device == "cpu" else 0,
                device_memory_kind=None
                if device == "cpu"
                else "unified"
                if device in {"mps", "rocm"}
                else "discrete",
                device_total_bytes=None if device == "cpu" else 8 * 1024**3,
                device_available_bytes=None if device == "cpu" else 6 * 1024**3,
                shared_total_bytes=8 * 1024**3 if device in {"mps", "rocm"} else None,
                shared_available_bytes=6 * 1024**3
                if device in {"mps", "rocm"}
                else None,
            ),
        ).model_dump_json()


@pytest.fixture
def installed(
    variant: NativePackageVariant,
    interpreter_fixture: tuple[Path, Path, PythonArchivePin],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> NativeInstall:
    """Build independent sealed fixtures for all three dependency profiles."""
    platform: ace.AceStepPlatform = (
        "macosx_14_0_arm64"
        if variant == "cpu"
        else "manylinux_2_28_aarch64"
        if variant == "cuda"
        else "linux_x86_64"
    )
    device: NativeMusicDevice = "cpu" if variant == "cpu" else variant
    python_archive, expanded, _ = interpreter_fixture
    catalog = read_python_catalog(
        Path(__file__).parents[4] / "packaging/skulk-ace-step"
    )
    pin = catalog.for_target(
        "manylinux_2_28_x86_64" if platform == "linux_x86_64" else platform
    ).model_copy(
        update={
            "sha256": hashlib.sha256(python_archive.read_bytes()).hexdigest(),
            "size_bytes": python_archive.stat().st_size,
        }
    )
    template = tmp_path / "native-template"
    template.mkdir()
    expanded.rename(template / "interpreter")
    members = {
        "skulk_ace_step/__init__.py": b"known native music adapter bytes",
        "skulk_ace_step/python/" + pin.filename: python_archive.read_bytes(),
    }
    content = dict(
        protocol_version=1,
        python="3.12.13",
        python_archive=pin.model_dump(mode="json"),
        adapter_revision="1" * 40,
        source_revision="ca1e85fe9430179831e6bc6be790c332190a3866",
        source_repository="https://github.com/ace-step/ACE-Step-1.5.git",
        platform_tag=platform,
        variant=variant,
        members={
            name: hashlib.sha256(data).hexdigest() for name, data in members.items()
        },
        features=[
            "music.generate",
            "music.edit",
            "source_conditioning",
            "pcm24",
            "offline_loader",
            "bundled_python",
            "resident_model",
            "library_probe",
            "memory_probe",
        ],
    )
    payload = hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    members["skulk_ace_step/build.json"] = json.dumps(
        content | {"payload_sha256": payload}
    ).encode()
    for name in ("METADATA", "WHEEL", "RECORD"):
        members[f"skulk_ace_step_{variant}-0.1.0.dist-info/{name}"] = (
            b"fixture metadata"
        )
    source = tmp_path / "native-fixture.whl"
    with zipfile.ZipFile(source, "w") as archive:
        for name, data in members.items():
            entry = zipfile.ZipInfo(name)
            entry.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(entry, data)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    unpack_candidate(
        source, template / "payload", digest, limits=native_package_limits(variant)
    )
    site = template / "runtime/lib/python3.12/site-packages"
    for name, data in members.items():
        path = site / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    base = template / "interpreter/python/bin/python3.12"
    binary = template / "runtime/bin/python"
    binary.parent.mkdir()
    binary.symlink_to(base)
    (template / "runtime/pyvenv.cfg").write_text(
        f"home = {base.parent}\ninclude-system-site-packages = false\n"
        f"version = 3.12.13\nexecutable = {base}\n"
    )
    wheel = ace.AceStepWheel(
        variant=variant,
        platform_tag=platform,
        filename=f"skulk_ace_step_{variant}-0.1.0-cp312-cp312-{platform}.whl",
        sha256=digest,
        payload_sha256=payload,
        size_bytes=source.stat().st_size,
    )
    controlled = NativeInstall(source, template, wheel, device)
    monkeypatch.setattr(
        ace, "platform_tags", lambda: iter(wheel.required_platform_tags)
    )

    def capacity(_path: Path) -> DiskUsage:
        return DiskUsage(200 * 1024**3, 0, controlled.free_bytes)

    monkeypatch.setattr(
        cache_runtime,
        "shutil",
        SimpleNamespace(
            disk_usage=capacity,
            rmtree=shutil.rmtree,
        ),
    )
    monkeypatch.setattr(cache_runtime, "install_fresh_runtime", controlled.install)

    def probe(
        _runner: NativeInstallCommandRunner,
        arguments: Sequence[str],
        *,
        environment: dict[str, str],
        timeout_seconds: int,
        capture: bool,
    ) -> str:
        return controlled.probe(
            arguments,
            environment=environment,
            timeout_seconds=timeout_seconds,
            capture=capture,
        )

    monkeypatch.setattr(NativeInstallCommandRunner, "__call__", probe)
    return controlled


def prepare(
    cache: ace.AceStepRuntimeCache, installed: NativeInstall, *, offline: bool = False
) -> ace.PreparedAceStepRuntime:
    """Use the actual cache and exact bytes with controlled native execution."""
    return cache.prepare(
        installed.wheel,
        device=installed.device,
        offline=offline,
        is_cancelled=lambda: installed.cancelled,
        progress=lambda _: None,
        download=installed.download,
    )


def test_exact_profile_offline_reuse_rechecks_bytes_and_device(
    tmp_path: Path, installed: NativeInstall
) -> None:
    """Every profile keeps its original object and successful probe authoritative."""
    cache = ace.AceStepRuntimeCache(tmp_path / "cache")
    first = prepare(cache, installed)
    second = prepare(cache, installed, offline=True)
    assert first == second
    assert first.device == installed.device
    assert first.wheel.build_identity == "ace_step@sha256:" + first.probe.payload_sha256
    assert (installed.downloads, installed.installs, installed.probes) == (1, 1, 2)
    cache.close()


def test_offline_without_original_bytes_never_downloads(
    tmp_path: Path, installed: NativeInstall
) -> None:
    """An installable package cannot become a verified offline engine."""
    with pytest.raises(ValueError, match="offline.*intact pinned package"):
        prepare(ace.AceStepRuntimeCache(tmp_path / "cache"), installed, offline=True)
    assert (installed.downloads, installed.installs, installed.probes) == (0, 0, 0)


def test_same_size_changed_code_is_not_reused_offline(
    tmp_path: Path, installed: NativeInstall
) -> None:
    """A complete local receipt cannot authorize modified cached executable bytes."""
    cache = ace.AceStepRuntimeCache(tmp_path / "cache")
    first = prepare(cache, installed)
    code = (
        first.generation
        / "runtime/lib/python3.12/site-packages/skulk_ace_step/__init__.py"
    )
    original = code.read_bytes()
    code.write_bytes(bytes((original[0] ^ 1,)) + original[1:])
    second = prepare(cache, installed, offline=True)
    assert second.generation != first.generation
    assert first.generation.is_dir()
    assert (installed.downloads, installed.installs, installed.probes) == (1, 2, 2)


@pytest.mark.parametrize("foreign", ("payload", "device"))
def test_foreign_probe_never_publishes_a_generation(
    tmp_path: Path, installed: NativeInstall, foreign: str
) -> None:
    """Successful math for a different build/lane is not requested readiness."""
    installed.foreign_payload = foreign == "payload"
    installed.foreign_device = foreign == "device"
    root = tmp_path / "cache"
    with pytest.raises(
        cache_runtime.DemucsBuildMismatchError,
        match="different payload or compute device",
    ):
        prepare(ace.AceStepRuntimeCache(root), installed)
    build = root / installed.wheel.sha256
    assert not (build / "selected.json").exists()
    assert not tuple((build / "generations").iterdir())


@pytest.mark.parametrize("observation", ["missing", "unsampled"])
def test_incomplete_memory_probe_never_publishes_a_generation(
    tmp_path: Path,
    installed: NativeInstall,
    observation: str,
) -> None:
    """A declared feature and successful math cannot substitute for a real sample."""
    installed.missing_memory = observation == "missing"
    installed.unsampled_memory = observation == "unsampled"
    root = tmp_path / "cache"
    with pytest.raises(
        cache_runtime.DemucsBuildMismatchError, match="memory observation"
    ):
        prepare(ace.AceStepRuntimeCache(root), installed)
    build = root / installed.wheel.sha256
    assert not (build / "selected.json").exists()
    assert not tuple((build / "generations").iterdir())


def test_cancellation_after_probe_cannot_publish_late_success(
    tmp_path: Path, installed: NativeInstall
) -> None:
    """Cancellation wins before publication, and owned temporary files drain."""
    installed.cancel_after_probe = True
    root = tmp_path / "cache"
    cache = ace.AceStepRuntimeCache(root)
    with pytest.raises(cache_runtime.DemucsPreparationCancelledError):
        prepare(cache, installed)
    build = root / installed.wheel.sha256
    assert not (build / "selected.json").exists()
    assert not tuple((build / "generations").iterdir())
    cache.close()


def test_profile_reservation_precedes_expensive_install(
    tmp_path: Path, installed: NativeInstall
) -> None:
    """GPU expansion cannot slip through the smaller former CPU disk allowance."""
    limits = native_package_limits(installed.wheel.variant)
    installed.free_bytes = (
        10 * 1024**3
        + limits.package_bytes
        + 512 * 1024**2
        + limits.installed_bytes
        + 256 * 1024**2
        - 1
    )
    with pytest.raises(cache_runtime.DemucsStorageError, match="fresh isolated"):
        prepare(ace.AceStepRuntimeCache(tmp_path / "cache"), installed)
    assert (installed.downloads, installed.installs, installed.probes) == (1, 0, 0)


def test_incompatible_host_fails_without_cache_effects(
    tmp_path: Path, installed: NativeInstall, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real ABI availability is distinct from package publication/qualification."""
    # A raw ROCm wheel tag alone cannot bypass its known interpreter ABI.
    unavailable = (
        (installed.wheel.platform_tag,) if installed.wheel.variant == "rocm" else ()
    )
    monkeypatch.setattr(ace, "platform_tags", lambda: iter(unavailable))
    root = tmp_path / "cache"
    with pytest.raises(cache_runtime.DemucsBuildMismatchError, match="host/device"):
        prepare(ace.AceStepRuntimeCache(root), installed)
    assert not root.exists()
    assert (installed.downloads, installed.installs, installed.probes) == (0, 0, 0)
