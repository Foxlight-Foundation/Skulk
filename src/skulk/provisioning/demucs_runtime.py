"""Verified offline native separation cache, separate from model readiness.

No package inventory is published or backend advertised here. Callers supply a
reviewed immutable CPU package pin, then use ordinary model admission separately.
The same CPU package can prove native CPU or Apple MPS library availability.
"""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from itertools import islice
from pathlib import Path
from typing import Literal, final
from uuid import uuid4

import httpx
from packaging.tags import platform_tags
from pydantic import Field, model_validator

from skulk.provisioning.demucs_package import (
    MAX_EXPANDED_PYTHON_BYTES,
    MAX_INSTALLED_BYTES,
    MAX_PACKAGE_BYTES,
    BuildManifest,
    Checkpoint,
    Digest,
    inspect_candidate,
    install_fresh_runtime,
    verify_cached_runtime,
)
from skulk.provisioning.native_install_process import NativeInstallCommandRunner
from skulk.utils.pydantic_ext import FrozenModel
from skulk.utils.runtime_files import (
    RuntimeLock,
    private_directory,
    read_private,
    write_private,
)

DemucsPlatform = Literal[
    "macosx_14_0_arm64", "manylinux_2_28_x86_64", "manylinux_2_28_aarch64"
]
DemucsDevice = Literal["cpu", "mps"]
PreparationPhase = Literal[
    "locking", "verifying", "downloading", "installing", "probing", "prepared"
]
_RESERVE_BYTES = 10 * 1024**3
_GENERATION_LIMIT = 16
_REQUIRED_FEATURES = frozenset(
    {"audio.separate", "four_stems", "pcm24", "offline_loader", "bundled_python"}
)


@final
class DemucsWheel(FrozenModel):
    """One reviewed immutable offline CPU package, with weights kept separate."""

    platform_tag: DemucsPlatform = Field(
        description="Minimum OS/libc and CPU architecture of the bundled runtime."
    )
    filename: str = Field(
        description="Closed CPU wheel filename in the Foxlight package channel."
    )
    sha256: Digest = Field(description="SHA-256 of the entire original package.")
    payload_sha256: Digest = Field(
        description="Manifest payload identity required before interpreter execution."
    )
    size_bytes: int = Field(
        gt=0, le=MAX_PACKAGE_BYTES, description="Exact bounded package download size."
    )

    @model_validator(mode="after")
    def validate_filename(self) -> DemucsWheel:
        """Bind the filename to the implemented package version and target."""
        if (
            self.filename
            != f"skulk_demucs_cpu-0.1.0-cp312-cp312-{self.platform_tag}.whl"
        ):
            raise ValueError("separation package filename does not match its target")
        return self

    @property
    def url(self) -> str:
        """Return the fixed immutable Foxlight distribution location."""
        return "https://wheels.foxlight.ai/wheels/" + self.filename

    @property
    def build_identity(self) -> str:
        """Return the payload identity, without claiming loaded-model readiness."""
        return "demucs@sha256:" + self.payload_sha256


@final
class PreparedDemucsRuntime(FrozenModel):
    """Verified libraries and selected device; not a mounted ready model."""

    wheel: DemucsWheel = Field(
        description="Exact complete package and payload pins that were verified."
    )
    generation: Path = Field(
        description="Stable private installation path; older generations remain untouched."
    )
    interpreter: Path = Field(
        description="Verified private Python for an admitted native task."
    )
    device: DemucsDevice = Field(
        description="Lane exercised by the actual finite FP32 library probe."
    )


class _CompletedGeneration(FrozenModel):
    """A bounded selection hint; full byte verification still authorizes reuse."""

    generation: str = Field(
        pattern=r"^[a-f0-9]{32}$", description="Owned stable generation basename."
    )
    package_sha256: Digest = Field(
        description="Original package expected by the completing preparation."
    )
    payload_sha256: Digest = Field(
        description="Expected immutable payload, not a local integrity authority."
    )


class _DeviceProbe(FrozenModel):
    """Closed native observation from already verified adapter/library bytes."""

    payload_sha256: Digest = Field(
        description="Actual installed adapter manifest identity."
    )
    requested: DemucsDevice = Field(
        description="Explicit lane requested by the controller."
    )
    native: DemucsDevice = Field(
        description="Native lane exercised without automatic CPU fallback."
    )


@final
class DemucsPreparationCancelledError(Exception):
    """An owner cancelled preparation; no new runtime may be published."""


@final
class DemucsStorageError(RuntimeError):
    """Preparation would consume the host's required free-disk reserve."""


@final
class DemucsBuildMismatchError(RuntimeError):
    """Verified package bytes disagree with the selected build's policy pins."""


@contextmanager
def _owned_lock(root: Path, checkpoint: Checkpoint) -> Iterator[RuntimeLock]:
    lock: RuntimeLock | None = None
    while lock is None:
        checkpoint()
        try:
            lock = RuntimeLock(root)
        except BlockingIOError:
            time.sleep(0.05)
    try:
        yield lock
    finally:
        lock.close()


def download_demucs_package(
    wheel: DemucsWheel,
    destination: Path,
    checkpoint: Checkpoint,
    *,
    transport: httpx.BaseTransport | None = None,
) -> None:
    """Stream one exact package with bounded bytes and cancellable network waits.

    The destination is new and owned by the cache caller. This never follows
    redirects, resolves package indexes, requests model weights or expands an
    encoded HTTP body. Interrupted bytes remain the caller's cleanup duty.
    ``transport`` is an optional injected network effect for isolated validation.
    """
    checkpoint()
    timeout = httpx.Timeout(connect=10.0, read=2.0, write=10.0, pool=5.0)
    with (
        httpx.Client(
            timeout=timeout, follow_redirects=False, transport=transport
        ) as client,
        client.stream(
            "GET", wheel.url, headers={"Accept-Encoding": "identity"}
        ) as response,
    ):
        response.raise_for_status()
        encodings = [
            value.lower() for value in response.headers.get_list("Content-Encoding")
        ]
        if encodings and encodings != ["identity"]:
            raise ValueError("engine channel returned an encoded package body")
        lengths = response.headers.get_list("Content-Length")
        if lengths and lengths != [str(wheel.size_bytes)]:
            raise ValueError("engine package response size differs from its pin")
        size = 0
        hasher = hashlib.sha256()
        descriptor = os.open(
            destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
        )
        with os.fdopen(descriptor, "wb") as target:
            # A large iterator chunk can buffer a slow continuous response for
            # minutes without tripping its per-read timeout. Check the owner on
            # each transport read instead of waiting to fill a MiB buffer.
            for block in response.iter_raw():
                checkpoint()
                size += len(block)
                if size > wheel.size_bytes:
                    raise ValueError("engine package download exceeds its pinned size")
                hasher.update(block)
                target.write(block)
            if size != wheel.size_bytes:
                raise ValueError("engine package download is incomplete")
            if hasher.hexdigest() != wheel.sha256:
                raise ValueError("engine package SHA-256 differs from its pin")
            target.flush()
            os.fsync(target.fileno())


def _inspect_package(
    package: Path, wheel: DemucsWheel, checkpoint: Checkpoint
) -> BuildManifest:
    if (
        package.is_symlink()
        or not package.is_file()
        or package.stat().st_size != wheel.size_bytes
    ):
        raise ValueError(
            "cached separation package is missing, redirected or incomplete"
        )
    manifest = inspect_candidate(package, wheel.sha256, checkpoint=checkpoint)
    if (
        manifest.payload_sha256 != wheel.payload_sha256
        or manifest.platform_tag != wheel.platform_tag
        or manifest.variant != "cpu"
        or not _REQUIRED_FEATURES.issubset(manifest.features)
    ):
        raise DemucsBuildMismatchError(
            "separation package manifest does not match the selected build"
        )
    return manifest


def _selection(root: Path, wheel: DemucsWheel) -> str | None:
    try:
        record = _CompletedGeneration.model_validate_json(
            read_private(root / "selected.json", 4096)
        )
    except (FileNotFoundError, ValueError):
        return None
    if (record.package_sha256, record.payload_sha256) != (
        wheel.sha256,
        wheel.payload_sha256,
    ):
        return None
    return record.generation


def _probe(
    interpreter: Path,
    wheel: DemucsWheel,
    device: DemucsDevice,
    runner: NativeInstallCommandRunner,
) -> None:
    # Both the adapter and its manifest are independently verified before this
    # import. The request contains no arbitrary upstream option or server path.
    code = (
        "import json;from importlib.resources import files;"
        "from skulk_demucs.engine import probe_device;"
        "manifest=json.loads(files('skulk_demucs').joinpath('build.json').read_text());"
        f"print(json.dumps(dict(payload_sha256=manifest['payload_sha256'],requested={device!r},native=probe_device({device!r}))))"
    )
    environment = {
        name: value
        for name, value in os.environ.items()
        if name in {"PATH", "HOME", "SYSTEMROOT", "TMPDIR", "TMP", "TEMP", "LANG"}
    } | {
        "HF_HUB_OFFLINE": "1",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "PYTORCH_ENABLE_MPS_FALLBACK": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    result = _DeviceProbe.model_validate_json(
        runner(
            [str(interpreter), "-I", "-B", "-c", code],
            environment=environment,
            timeout_seconds=300,
            capture=True,
        )
    )
    if (result.payload_sha256, result.requested, result.native) != (
        wheel.payload_sha256,
        device,
        device,
    ):
        raise ValueError("native separation probe selected a different build or device")


@final
class DemucsRuntimeCache:
    """Own serialized preparation and retained cleanup handles for one cache.

    Keep this controller for the worker lifetime. A failed process or file drain
    remains owned here; the next preparation or close retries the original
    handle. No local PID record can substitute for that unreaped ownership.
    """

    def __init__(self, root: Path) -> None:
        """Select the private cache without creating it or preparing any package."""
        self.root = root.absolute()
        self._guard = threading.Lock()
        self._pending: tuple[NativeInstallCommandRunner, Path | None, bool] | None = (
            None
        )

    def _finish(
        self,
        runner: NativeInstallCommandRunner,
        generation: Path | None,
        activation_attempted: bool,
    ) -> None:
        self._pending = (runner, generation, activation_attempted)
        runner.drain()
        self._remove_unpublished(generation, activation_attempted)
        self._pending = None

    @staticmethod
    def _remove_unpublished(
        generation: Path | None, activation_attempted: bool
    ) -> None:
        # Callers hold the global reservation fence. Presence is conservative
        # deletion protection only, never an integrity or execution authority.
        # Keep publication history even after the current selection moves away.
        if activation_attempted or generation is None:
            return
        marker = generation.parent.parent / "published" / generation.name
        try:
            marker.lstat()
        except FileNotFoundError:
            pass
        else:
            return
        if generation.is_dir() and not generation.is_symlink():
            shutil.rmtree(generation)

    def _retry_pending(self, checkpoint: Checkpoint) -> None:
        if self._pending is None:
            return
        runner, generation, activation_attempted = self._pending
        # Drain first: a surviving guardian may itself retain this global lock.
        # Then reacquire the fence before touching files, because a different
        # controller may have recovered them since the original call unwound.
        runner.drain()
        with _owned_lock(self.root, checkpoint):
            self._remove_unpublished(generation, activation_attempted)
        self._pending = None

    def close(self) -> None:
        """Retry retained cleanup after the owner cancels and joins preparation.

        Refuse while a preparation call is active. Failed cleanup stays retained
        for another explicit retry, and its generation is never deleted before
        complete process drain. Retry reacquires the cache fence after draining;
        waiting for another owner is bounded at five seconds. Published and
        older generations remain intact, even if the selection hint disappeared.
        """
        if not self._guard.acquire(blocking=False):
            raise RuntimeError(
                "cancel and join native preparation before closing its cache"
            )
        try:
            deadline = time.monotonic() + 5.0

            def checkpoint() -> None:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "separation cleanup is waiting for another preparation; retry close"
                    )

            self._retry_pending(checkpoint)
        finally:
            self._guard.release()

    def prepare(
        self,
        wheel: DemucsWheel,
        *,
        device: DemucsDevice,
        offline: bool,
        is_cancelled: Callable[[], bool],
        progress: Callable[[PreparationPhase], None],
        timeout_seconds: float = 30 * 60,
        download: Callable[
            [DemucsWheel, Path, Checkpoint], None
        ] = download_demucs_package,
    ) -> PreparedDemucsRuntime:
        """Verify, recover or bootstrap a pinned runtime with bounded ownership.

        Wait for this controller's prior call within the same cancellation and
        deadline budget. Retained cleanup is retried before any new installer
        starts. The result proves bytes and a device probe, not a loaded model.
        """
        if not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 2 * 60 * 60:
            raise ValueError(
                "separation preparation deadline must be finite and within two hours"
            )
        deadline = time.monotonic() + timeout_seconds

        def checkpoint() -> None:
            if is_cancelled():
                raise DemucsPreparationCancelledError(
                    "native separation preparation cancelled"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "native separation preparation exceeded its deadline"
                )

        while True:
            if is_cancelled():
                raise DemucsPreparationCancelledError(
                    "native separation preparation cancelled"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "native separation preparation exceeded its deadline"
                )
            if self._guard.acquire(timeout=0.05):
                break
        try:
            self._retry_pending(checkpoint)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    "native separation preparation exceeded its deadline"
                )
            return self._prepare(
                wheel,
                device=device,
                offline=offline,
                is_cancelled=is_cancelled,
                progress=progress,
                timeout_seconds=remaining,
                download=download,
            )
        finally:
            self._guard.release()

    def _prepare(
        self,
        wheel: DemucsWheel,
        *,
        device: DemucsDevice,
        offline: bool,
        is_cancelled: Callable[[], bool],
        progress: Callable[[PreparationPhase], None],
        timeout_seconds: float = 30 * 60,
        download: Callable[
            [DemucsWheel, Path, Checkpoint], None
        ] = download_demucs_package,
    ) -> PreparedDemucsRuntime:
        """Prepare an exact native runtime without replacing any mounted generation.

        Reuse and interrupted-generation recovery verify sealed source, the complete
        interpreter/venv/import tree, and the requested device. Offline calls never
        download, but may bootstrap from a verified cached object. New installations
        remain at their final UUID path before creating a venv; only an atomic synced
        selection record publishes success. Failed newly owned files are removed
        only after complete process drain; previously retained generations survive.
        Preparation does not download weights, load a model or advertise readiness.
        """
        if wheel.platform_tag not in set(platform_tags()):
            raise ValueError(
                "separation package is incompatible with this host OS/CPU ABI"
            )
        if device not in ("cpu", "mps") or (
            device == "mps" and not wheel.platform_tag.startswith("macosx_")
        ):
            raise ValueError(
                "this separation package cannot provide the requested device"
            )
        if not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 2 * 60 * 60:
            raise ValueError(
                "separation preparation deadline must be finite and within two hours"
            )
        root = self.root
        private_directory(root)
        deadline = time.monotonic() + timeout_seconds
        last_disk_check = 0.0

        def checkpoint(*, force_disk: bool = False) -> None:
            nonlocal last_disk_check
            if is_cancelled():
                raise DemucsPreparationCancelledError(
                    "native separation preparation cancelled"
                )
            now = time.monotonic()
            if now >= deadline:
                raise TimeoutError(
                    "native separation preparation exceeded its deadline"
                )
            if force_disk or now - last_disk_check >= 0.5:
                last_disk_check = now
                if shutil.disk_usage(root).free < _RESERVE_BYTES:
                    raise DemucsStorageError(
                        "native separation cache needs at least 10 GiB free for the host reserve"
                    )

        progress("locking")
        # The global cache fence serializes disk reservations across builds. The
        # guardian inherits it, so controller death cannot release it before its
        # installer group is drained. Each exact build also has its own fence.
        with _owned_lock(root, checkpoint) as reservation:
            build = root / wheel.sha256
            private_directory(build)
            with _owned_lock(build, checkpoint):
                objects = build / "objects"
                generations = build / "generations"
                publications = build / "published"
                private_directory(objects)
                private_directory(generations)
                private_directory(publications)
                package = objects / wheel.filename
                progress("verifying")
                try:
                    manifest = _inspect_package(package, wheel, checkpoint)
                except (FileNotFoundError, ValueError):
                    if offline:
                        raise ValueError(
                            "offline separation preparation needs the intact pinned package in its cache"
                        ) from None
                    if shutil.disk_usage(root).free < _RESERVE_BYTES + wheel.size_bytes:
                        raise DemucsStorageError(
                            "insufficient disk space to download the separation package"
                        ) from None
                    progress("downloading")
                    temporary = objects / (".download-" + uuid4().hex)
                    try:
                        download(wheel, temporary, checkpoint)
                        manifest = _inspect_package(temporary, wheel, checkpoint)
                        checkpoint()
                        os.replace(temporary, package)
                        descriptor = os.open(objects, os.O_RDONLY | os.O_DIRECTORY)
                        try:
                            os.fsync(descriptor)
                        finally:
                            os.close(descriptor)
                    finally:
                        temporary.unlink(missing_ok=True)

                selected = _selection(build, wheel)
                candidates: list[str] = [] if selected is None else [selected]
                entries = tuple(islice(generations.iterdir(), _GENERATION_LIMIT + 1))
                if len(entries) > _GENERATION_LIMIT:
                    raise ValueError(
                        "separation cache has too many retained generations; review inactive cache entries"
                    )
                # A complete but unpublished installation can survive interruption.
                # Recover it only after full independent verification, never from a
                # local completion receipt or a partial interpreter's own report.
                candidates.extend(
                    path.name
                    for path in sorted(entries)
                    if len(path.name) == 32
                    and all(character in "0123456789abcdef" for character in path.name)
                    and path.name not in candidates
                )
                for name in candidates:
                    checkpoint()
                    generation = generations / name
                    try:
                        interpreter, _ = verify_cached_runtime(
                            package,
                            generation,
                            wheel.sha256,
                            wheel.payload_sha256,
                            checkpoint=checkpoint,
                        )
                    except (FileNotFoundError, ValueError):
                        continue
                    runner = NativeInstallCommandRunner(
                        build / "diagnostics",
                        checkpoint=checkpoint,
                        lock_descriptor=reservation.descriptor,
                        guardian_interpreter=generation
                        / "interpreter"
                        / manifest.python_archive.executable,
                    )
                    try:
                        progress("probing")
                        _probe(interpreter, wheel, device, runner)
                        checkpoint(force_disk=True)
                        # Mark potential publication before replacing selection.
                        # A former installer's deferred cleanup must preserve
                        # this generation even if selection changes or disappears.
                        write_private(publications / generation.name, b"1\n")
                        write_private(
                            build / "selected.json",
                            _CompletedGeneration(
                                generation=name,
                                package_sha256=wheel.sha256,
                                payload_sha256=wheel.payload_sha256,
                            )
                            .model_dump_json()
                            .encode(),
                        )
                        progress("prepared")
                        return PreparedDemucsRuntime(
                            wheel=wheel,
                            generation=generation,
                            interpreter=interpreter,
                            device=device,
                        )
                    finally:
                        self._finish(runner, None, True)

                if len(entries) >= _GENERATION_LIMIT:
                    raise ValueError(
                        "separation cache reached its retained generation bound; review inactive entries"
                    )
                required = (
                    _RESERVE_BYTES
                    + MAX_PACKAGE_BYTES
                    + MAX_EXPANDED_PYTHON_BYTES
                    + MAX_INSTALLED_BYTES
                    + 256 * 1024**2
                )
                if shutil.disk_usage(root).free < required:
                    raise DemucsStorageError(
                        "insufficient disk space for a fresh isolated separation runtime and host reserve"
                    )
                generation = generations / uuid4().hex
                runner = NativeInstallCommandRunner(
                    build / "diagnostics",
                    checkpoint=checkpoint,
                    lock_descriptor=reservation.descriptor,
                    guardian_interpreter=generation
                    / "interpreter"
                    / manifest.python_archive.executable,
                )
                activation_attempted = False
                try:
                    progress("installing")
                    interpreter = install_fresh_runtime(
                        package,
                        generation,
                        wheel.sha256,
                        run=runner,
                        checkpoint=checkpoint,
                    )
                    # The independently supplied payload identity remains required
                    # after bootstrap, as well as in the pre-execution inspection.
                    interpreter, _ = verify_cached_runtime(
                        package,
                        generation,
                        wheel.sha256,
                        wheel.payload_sha256,
                        checkpoint=checkpoint,
                    )
                    progress("probing")
                    _probe(interpreter, wheel, device, runner)
                    checkpoint(force_disk=True)
                    # A directory fsync can fail after atomic replacement. Retain
                    # the generation once activation is attempted; never delete a
                    # possibly published interpreter underneath a concurrent user.
                    activation_attempted = True
                    write_private(publications / generation.name, b"1\n")
                    write_private(
                        build / "selected.json",
                        _CompletedGeneration(
                            generation=generation.name,
                            package_sha256=wheel.sha256,
                            payload_sha256=wheel.payload_sha256,
                        )
                        .model_dump_json()
                        .encode(),
                    )
                    progress("prepared")
                    return PreparedDemucsRuntime(
                        wheel=wheel,
                        generation=generation,
                        interpreter=interpreter,
                        device=device,
                    )
                finally:
                    self._finish(runner, generation, activation_attempted)
