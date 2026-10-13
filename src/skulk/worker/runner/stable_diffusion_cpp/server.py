"""Own one authenticated loopback server and release every terminal native job."""

from __future__ import annotations

import contextlib
import json
import os
import re
import secrets
import signal
import socket
import subprocess
import threading
import time
from collections.abc import Generator, Sequence
from pathlib import Path
from typing import Literal, cast, final

import httpx
import psutil
from pydantic import ConfigDict, Field

from skulk.provisioning.stable_diffusion_cpp import probe_native_image
from skulk.utils.pydantic_ext import FrozenModel

_MAX_RESPONSE_BYTES = 64 * 1024 * 1024
_JOB_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


class NativeImageCancelledError(Exception):
    """The native producer has stopped the cancelled generation."""


@final
class NativePreview(FrozenModel):
    """One bounded latest-value preview from a native sampling pass."""

    pass_number: int = Field(alias="pass", ge=1, description="Sampling pass number.")
    step: int = Field(ge=1, description="Actual logical sampling step.")
    total_steps: int = Field(
        ge=1, le=100, description="Actual steps in this sampling pass."
    )
    b64_json: str = Field(
        max_length=_MAX_RESPONSE_BYTES, description="Encoded PNG preview."
    )


@final
class NativeImageResult(FrozenModel):
    """One final encoded image in a native terminal job."""

    index: int = Field(ge=0, description="Image index in this native request.")
    b64_json: str = Field(
        max_length=_MAX_RESPONSE_BYTES, description="Encoded image bytes."
    )


@final
class NativeResult(FrozenModel):
    """Native image result container; unrelated video results are refused."""

    output_format: Literal["png", "jpeg", "webp"] = Field(
        description="Encoded media format."
    )
    images: list[NativeImageResult] = Field(
        min_length=1, max_length=1, description="Exactly one image per native request."
    )


@final
class NativeFailure(FrozenModel):
    """Bounded native terminal error details."""

    code: str = Field(max_length=256, description="Native error category.")
    message: str = Field(max_length=4096, description="Private diagnostic detail.")


@final
class NativeJob(FrozenModel):
    """Validated native lifecycle state, never retained after collection."""

    model_config = ConfigDict(frozen=True, strict=True, extra="ignore")
    id: str = Field(
        pattern=r"^[A-Za-z0-9_-]{1,128}$", description="Opaque native job ID."
    )
    kind: Literal["img_gen"] = Field(
        description="Only image jobs belong to this runner."
    )
    status: Literal["queued", "generating", "completed", "failed", "cancelled"] = Field(
        description="Native lifecycle phase."
    )
    preview: NativePreview | None = Field(
        default=None, description="Latest available preview."
    )
    result: NativeResult | None = Field(
        default=None, description="Terminal image result."
    )
    error: NativeFailure | None = Field(default=None, description="Terminal failure.")


@final
class NativeImageServer:
    """A single resident native process with bounded requests and deterministic teardown."""

    def __init__(
        self,
        binary: Path,
        expected_build: str,
        backend: str,
        arguments: Sequence[str],
        log_path: Path,
    ):
        """Bind an exact verified GPU, spawn a private server, and wait for readiness.

        Arguments are built from immutable card components by the runner. This
        object owns its log file, client, process group and all native job results.
        """
        observed = probe_native_image(binary)
        compute = backend.removeprefix("stable_diffusion_cpp-")
        if (
            observed.build_identity != expected_build
            or compute not in observed.computes
        ):
            raise RuntimeError("native image build/device differs from placement")
        if len(observed.devices) != 1:
            raise RuntimeError(
                "native image admission requires one observed GPU device"
            )
        self._lock = threading.Lock()
        self._process: subprocess.Popen[bytes] | None = None
        self._client: httpx.Client | None = None
        self._log = log_path.open("wb")
        with socket.socket() as lease:
            lease.bind(("127.0.0.1", 0))
            port = cast("tuple[str, int]", lease.getsockname())[1]
        token = secrets.token_urlsafe(32)
        environment = dict(os.environ)
        environment["SKULK_SDCPP_AUTH_TOKEN"] = token
        environment["SKULK_SDCPP_PARENT_PID"] = str(os.getpid())
        try:
            self._process = subprocess.Popen(
                [
                    str(binary),
                    *arguments,
                    "--backend",
                    observed.devices[0],
                    "--params-backend",
                    "cpu",
                    "--listen-ip",
                    "127.0.0.1",
                    "--listen-port",
                    str(port),
                ],
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=self._log,
                stderr=self._log,
                start_new_session=True,
            )
            self._client = httpx.Client(
                base_url=f"http://127.0.0.1:{port}",
                headers={"Authorization": f"Bearer {token}"},
                timeout=15,
                trust_env=False,
            )
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                self.check_alive()
                try:
                    self._request("GET", "/sdcpp/v1/capabilities")
                    return
                except httpx.TransportError:
                    time.sleep(0.25)
            raise TimeoutError("native image server exceeded its readiness deadline")
        except BaseException:
            self.close()
            raise

    def check_alive(self) -> None:
        """Fail immediately when the owned native process has exited."""
        if self._process is None or self._process.poll() is not None:
            raise RuntimeError("native image server exited")

    def resident_memory_bytes(self) -> int | None:
        """Read owned-process RSS; return unavailable after producer exit.

        RSS describes resident host memory. GPU allocations require separate
        hardware telemetry and are not inferred from this measurement.
        """
        process = self._process
        if process is None:
            return None
        try:
            return psutil.Process(process.pid).memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None

    def _request(
        self, method: str, path: str, body: dict[str, object] | None = None
    ) -> bytes:
        if self._client is None:
            raise RuntimeError("native image server client is closed")
        data = None if body is None else json.dumps(body).encode()
        if data is not None and len(data) > _MAX_RESPONSE_BYTES:
            raise ValueError("native image request exceeds its byte limit")
        with self._client.stream(
            method, path, content=data, headers={"Content-Type": "application/json"}
        ) as response:
            response.raise_for_status()
            result = bytearray()
            for chunk in response.iter_bytes():
                if len(result) + len(chunk) > _MAX_RESPONSE_BYTES:
                    raise ValueError("native image response exceeds its byte limit")
                result.extend(chunk)
            return bytes(result)

    def generate(
        self, body: dict[str, object], cancelled: threading.Event
    ) -> Generator[NativePreview | NativeResult, None, None]:
        """Yield requested previews and one final image; cancel and delete every owned job."""
        with self._lock:
            self.check_alive()
            if cancelled.is_set():
                raise NativeImageCancelledError()
            try:
                raw = cast(
                    "object",
                    json.loads(self._request("POST", "/sdcpp/v1/img_gen", body)),
                )
                if not isinstance(raw, dict):
                    raise ValueError("native submission response is not an object")
                identifier = cast("dict[str, object]", raw).get("id")
                if (
                    not isinstance(identifier, str)
                    or _JOB_ID.fullmatch(identifier) is None
                ):
                    raise ValueError(
                        "native submission did not return a bounded job ID"
                    )
            except BaseException:
                # A lost or malformed submission reply leaves job ownership uncertain.
                self.close()
                raise
            path = f"/sdcpp/v1/jobs/{identifier}"
            terminal = False
            cancellation_deadline: float | None = None
            deadline = time.monotonic() + 1800
            last_preview: tuple[int, int] | None = None
            try:
                while time.monotonic() < deadline:
                    self.check_alive()
                    if cancelled.is_set():
                        if cancellation_deadline is None:
                            cancellation_deadline = time.monotonic() + 30
                        # Reassert after the native per-request reset, including cancellation during encoding.
                        self._request("POST", path + "/cancel", {})
                        if time.monotonic() >= cancellation_deadline:
                            self.close()
                            raise NativeImageCancelledError()
                    job = NativeJob.model_validate_json(self._request("GET", path))
                    if job.id != identifier:
                        raise ValueError("native response changed job ownership")
                    if job.status in {"completed", "failed", "cancelled"}:
                        terminal = True
                        if cancelled.is_set() or job.status == "cancelled":
                            raise NativeImageCancelledError()
                        if job.status == "failed" or job.result is None:
                            raise RuntimeError("native image generation failed")
                        yield job.result
                        return
                    preview = job.preview
                    if (
                        preview is not None
                        and (preview.pass_number, preview.step) != last_preview
                    ):
                        last_preview = (preview.pass_number, preview.step)
                        yield preview
                    time.sleep(0.25)
                raise TimeoutError("native image generation exceeded its deadline")
            finally:
                # A consumer may stop iteration or lose transport while native compute is still active.
                if terminal:
                    try:
                        self._request("DELETE", path)
                    except (httpx.HTTPError, ValueError, RuntimeError):
                        self.close()
                else:
                    self.close()

    def close(self) -> None:
        """Release the HTTP client, whole process group, and log, including partial startup."""
        client, self._client = self._client, None
        if client is not None:
            client.close()
        process = self._process
        try:
            if process is not None:
                # A failure observer may already have reaped the native process.
                # Its former process group is no longer ours to signal.
                if process.poll() is None:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGTERM)
                try:
                    # Large BF16 CUDA contexts can take longer than five seconds
                    # to reclaim. Retain ownership until wait confirms exit.
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    if process.poll() is None:
                        with contextlib.suppress(ProcessLookupError):
                            os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=15)
                self._process = None
        finally:
            self._log.close()
