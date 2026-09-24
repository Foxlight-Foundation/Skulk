"""Supervise one loopback-only audio.cpp server for one mounted music model."""

from __future__ import annotations

import contextlib
import ctypes
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO, Final, cast, final

import httpx

from skulk.shared.models.model_cards import MusicCardConfig, MusicModelFamily
from skulk.worker.runner.audio_cpp.adapter import (
    MAX_AUDIO_CPP_RESPONSE_BYTES,
    MusicWav,
    decode_audio_cpp_music_response,
)

_STARTUP_SECONDS: Final = 600.0
_REQUEST_SECONDS: Final = 45 * 60.0


def _parent_death_signal() -> None:  # pyright: ignore[reportUnusedFunction]
    """Kill the sidecar if its runner process dies abruptly on Linux."""
    if sys.platform != "linux":
        return
    with contextlib.suppress(OSError, AttributeError):
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL, 0, 0, 0)


def _free_loopback_port() -> int:
    """Choose an ephemeral loopback port; health polling detects a lost race."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        address = cast("tuple[str, int]", probe.getsockname())
        return address[1]


def server_config(
    *,
    port: int,
    backend: str,
    model_dir: Path,
    music: MusicCardConfig,
    model_specs: Path,
) -> dict[str, object]:
    """Build a fixed, single-model server config with no management surface."""
    if backend not in {"cpu", "metal", "vulkan", "cuda", "hip"}:
        raise ValueError(f"unsupported audio.cpp backend {backend}")
    if music.family == MusicModelFamily.MiniMaxMusic3:
        session_options = {
            "minimax_music3.language_model_gguf": music.language_model_gguf,
            "minimax_music3.rvq_depth_decoder_gguf": music.rvq_depth_decoder_gguf,
            "minimax_music3.flow_transformer_gguf": music.flow_transformer_gguf,
        }
    else:
        session_options = {}
    return {
        "host": "127.0.0.1",
        "port": port,
        "backend": backend,
        "threads": 1,
        "ui": False,
        "ui_management": False,
        "log_request_body": False,
        "lazy_load": False,
        "max_loaded_models": 1,
        # JSON escaping can expand the public 8k prompt and 20k lyrics beyond
        # their character counts; keep the sidecar bound above that maximum.
        "max_request_body_bytes": 256 * 1024,
        "busy_timeout_ms": 30 * 60 * 1000,
        "model_spec_override": str(model_specs),
        "models": [
            {
                "id": "skulk-music",
                "family": (
                    "minimax_music3"
                    if music.family == MusicModelFamily.MiniMaxMusic3
                    else "ace_step"
                ),
                "path": str(model_dir),
                "task": "gen",
                "mode": "offline",
                "session_options": session_options,
            }
        ],
    }


@final
class AudioCppServer:
    """One supervised sidecar, restricted to the runner's model and backend."""

    def __init__(
        self,
        *,
        binary: Path,
        model_specs: Path,
        model_dir: Path,
        music: MusicCardConfig,
        backend: str,
        work_dir: Path,
    ) -> None:
        self.binary = binary
        self.model_specs = model_specs
        self.model_dir = model_dir
        self.music = music
        self.backend = backend
        self.work_dir = work_dir
        self.base_url: str | None = None
        self.process: subprocess.Popen[bytes] | None = None
        self._log: BinaryIO | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start the sidecar, requiring a healthy model load before ready."""
        self.work_dir.mkdir(parents=True, exist_ok=True)
        for _ in range(3):
            port = _free_loopback_port()
            config_path = self.work_dir / "server.json"
            config_path.write_text(
                json.dumps(
                    server_config(
                        port=port,
                        backend=self.backend,
                        model_dir=self.model_dir,
                        music=self.music,
                        model_specs=self.model_specs,
                    )
                )
            )
            self.base_url = f"http://127.0.0.1:{port}"
            log = (self.work_dir / "server.log").open("ab")
            self._log = log
            self.process = subprocess.Popen(  # noqa: S603 - pinned executable
                [str(self.binary), "--config", str(config_path), "--no-ui"],
                cwd=self.work_dir,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                preexec_fn=_parent_death_signal if sys.platform == "linux" else None,
            )
            deadline = time.monotonic() + _STARTUP_SECONDS
            with httpx.Client(timeout=5.0) as client:
                while time.monotonic() < deadline:
                    if not self.alive():
                        tail = self.log_tail()
                        self.teardown()
                        if "address already in use" in tail.lower():
                            break
                        raise RuntimeError(f"audio.cpp exited during load: {tail}")
                    try:
                        if client.get(f"{self.base_url}/health").status_code == 200:
                            return
                    except httpx.HTTPError:
                        pass
                    time.sleep(0.5)
                else:
                    self.teardown()
                    raise RuntimeError("audio.cpp did not become ready before the load deadline")
        raise RuntimeError("audio.cpp could not claim a loopback port")

    def alive(self) -> bool:
        """Whether this server process is still running."""
        with self._lock:
            return self.process is not None and self.process.poll() is None

    def log_tail(self) -> str:
        """Return a bounded diagnostic tail without request bodies."""
        with contextlib.suppress(OSError):
            return "\n".join(
                (self.work_dir / "server.log").read_text(errors="replace").splitlines()[-30:]
            )[-4000:]
        return "(no server log)"

    def generate(
        self,
        body: dict[str, object],
        *,
        is_cancelled: Callable[[], bool],
    ) -> MusicWav | None:
        """Run one request, killing this model's server as soon as it is cancelled."""
        if self.base_url is None:
            raise RuntimeError("audio.cpp server has not started")
        done = threading.Event()

        def watch_cancel() -> None:
            while not done.wait(0.1):
                if is_cancelled():
                    self.teardown()
                    return

        watcher = threading.Thread(target=watch_cancel, name="audio-cpp-cancel", daemon=True)
        watcher.start()
        response = bytearray()
        try:
            with (
                httpx.Client(timeout=httpx.Timeout(_REQUEST_SECONDS)) as client,
                client.stream("POST", f"{self.base_url}/v1/tasks/run", json=body) as stream,
            ):
                stream.raise_for_status()
                for chunk in stream.iter_bytes(1024 * 1024):
                    if is_cancelled():
                        self.teardown()
                        return None
                    if len(response) + len(chunk) > MAX_AUDIO_CPP_RESPONSE_BYTES:
                        self.teardown()
                        raise ValueError("audio.cpp response exceeds the 90 MiB envelope limit")
                    response.extend(chunk)
            if is_cancelled():
                self.teardown()
                return None
            return decode_audio_cpp_music_response(bytes(response))
        except httpx.HTTPError as error:
            if is_cancelled():
                return None
            raise RuntimeError(f"audio.cpp generation failed: {error}") from error
        finally:
            done.set()
            watcher.join(timeout=2.0)

    def teardown(self) -> None:
        """Terminate this one model server and its process group."""
        with self._lock:
            process = self.process
            self.process = None
        if process is not None and process.poll() is None:
            with contextlib.suppress(OSError):
                os.killpg(process.pid, signal.SIGTERM)
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=5)
            if process.poll() is None:
                with contextlib.suppress(OSError):
                    os.killpg(process.pid, signal.SIGKILL)
                with contextlib.suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=5)
        log = self._log
        self._log = None
        if log is not None:
            log.close()
