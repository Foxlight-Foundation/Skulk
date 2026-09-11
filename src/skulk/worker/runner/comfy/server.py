"""Spawn, watch, drive, and tear down one headless ComfyUI server.

Process shape follows the vLLM runner: the server starts at the head of its
own process group with ``PR_SET_PDEATHSIG`` so a runner crash never orphans
it, health is polled over HTTP with the process watched for early exit, and
teardown signals the whole group. Rendering talks ComfyUI's own protocol:
``POST /prompt`` with a caller-minted ``prompt_id`` and ``client_id``, the
``/ws`` socket for ``executing`` / ``progress_state`` / terminal events
addressed to that client, ``POST /api/jobs/{id}/cancel`` for cancellation,
and ``GET /history/{id}`` for the finished outputs. Queue entries from other
clients (an operator using the ComfyUI frontend on the same server) are
tolerated: the render reports ``queued`` until its own prompt executes.
"""

from __future__ import annotations

import asyncio
import contextlib
import ctypes
import json
import os
import random
import signal
import socket
import subprocess
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Final, Literal, cast

import aiohttp
import httpx
from loguru import logger

from skulk.shared.types.video import VideoStage
from skulk.worker.runner.comfy.graph import NODE_SAMPLER, ComfyPrompt, stage_for_node

_HEALTH_DEADLINE_SECONDS: Final = 600.0
"""ComfyUI imports torch and scans model folders before listening; a cold
start on a busy node can take minutes, and a dead process is detected long
before the deadline."""
_PORT_COLLISION_ATTEMPTS: Final = 3
_ADDRESS_IN_USE_MARKER: Final = "address already in use"
_CANCEL_GRACE_SECONDS: Final = 3600.0
"""How long an accepted cancel may take to stop the render. ComfyUI applies
the interrupt between sampling steps, and one H3 step on the supported GPUs
runs minutes, so the grace is a safety net for a server that ignores the
interrupt, not a bound on step length."""
_JOB_POLL_SECONDS: Final = 5.0
_SOCKET_WAIT_SECONDS: Final = 0.5
_SAMPLING_SHARE: Final = (0.1, 0.9)
"""Fraction of the progress bar the sampling stage spans."""

ProgressCallback = Callable[[VideoStage, int | None, int | None, float], None]

SKULK_LAUNCH_MARKER_DIR: Final = "comfy"
"""Directory name under the node's cache home that only Skulk-launched
ComfyUI servers use as ``--user-directory``; the orphan sweep keys on it."""


class ComfyRenderError(ValueError):
    """ComfyUI refused or failed this render; the server itself is fine."""


def _set_pdeathsig() -> None:
    """Ask the kernel to SIGKILL the server when the runner dies (Linux)."""
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        pr_set_pdeathsig = 1
        libc.prctl(pr_set_pdeathsig, signal.SIGKILL, 0, 0, 0)
    except (OSError, AttributeError):
        pass


def pick_free_port() -> int:
    """Pick a free ephemeral loopback port, avoiding the API port."""
    for _ in range(30):
        port = random.randint(49153, 65535)
        if port == 52415:
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
        return port
    raise RuntimeError("could not find a free port for the ComfyUI server")


def server_environment(interpreter: Path, extra: dict[str, str]) -> dict[str, str]:
    """The environment one ComfyUI server is spawned with.

    The interpreter's own bin directory comes first on ``PATH``: the
    environment carries its tools (ffmpeg wheels, compilers for JIT kernels)
    beside python. ``PYTHONPATH`` from the Skulk process must not leak into
    ComfyUI's interpreter, since the two environments carry different torch
    builds. Lane-specific variables (``extra``) are layered last so they win.
    """
    env = dict(os.environ)
    env["PATH"] = f"{interpreter.parent}{os.pathsep}{env.get('PATH', '')}"
    env.pop("PYTHONPATH", None)
    env.update(extra)
    return env


def server_args(
    interpreter: Path,
    root: Path,
    *,
    port: int,
    extra_model_paths: Path,
    input_dir: Path,
    output_dir: Path,
    temp_dir: Path,
    user_dir: Path,
    extra: Sequence[str] = (),
) -> list[str]:
    """The headless launch line for one ComfyUI server.

    Custom and API nodes are disabled (the graph uses core nodes only and a
    node pack the operator installed must never change what a card renders),
    output metadata is off so the container carries no prompt text, and the
    input, output, temp, and user directories are Skulk's, not the checkout's.
    """
    return [
        str(interpreter),
        str(root / "main.py"),
        "--listen",
        "127.0.0.1",
        "--port",
        str(port),
        "--disable-auto-launch",
        "--disable-all-custom-nodes",
        "--disable-api-nodes",
        "--disable-metadata",
        "--extra-model-paths-config",
        str(extra_model_paths),
        "--input-directory",
        str(input_dir),
        "--output-directory",
        str(output_dir),
        "--temp-directory",
        str(temp_dir),
        "--user-directory",
        str(user_dir),
        *extra,
    ]


@dataclass
class ComfyServer:
    """One managed ComfyUI process and the loopback URL it serves."""

    interpreter: Path
    root: Path
    extra_model_paths: Path
    input_dir: Path
    output_dir: Path
    temp_dir: Path
    user_dir: Path
    log_path: Path
    extra_args: tuple[str, ...] = ()
    extra_env: dict[str, str] = field(default_factory=dict)
    """Lane-specific variables layered over the process environment."""
    base_url: str | None = None
    process: subprocess.Popen[bytes] | None = None
    _log: BinaryIO | None = field(default=None, repr=False)

    def start(self) -> None:
        """Spawn the server and wait for it to answer, retrying a lost port race."""
        for attempt in range(1, _PORT_COLLISION_ATTEMPTS + 1):
            self._spawn()
            try:
                self._await_health()
            except RuntimeError as error:
                lost_race = _ADDRESS_IN_USE_MARKER in str(error).lower() and attempt < _PORT_COLLISION_ATTEMPTS
                if not lost_race:
                    self.teardown()
                    raise
                logger.warning(
                    f"ComfyUI lost a port race on startup (attempt {attempt}/{_PORT_COLLISION_ATTEMPTS}); retrying"
                )
                self.teardown()
                continue
            return

    def _spawn(self) -> None:
        port = pick_free_port()
        self.base_url = f"http://127.0.0.1:{port}"
        for directory in (self.input_dir, self.output_dir, self.temp_dir, self.user_dir):
            directory.mkdir(parents=True, exist_ok=True)
        args = server_args(
            self.interpreter,
            self.root,
            port=port,
            extra_model_paths=self.extra_model_paths,
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            temp_dir=self.temp_dir,
            user_dir=self.user_dir,
            extra=self.extra_args,
        )
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(self.log_path, "wb")  # noqa: SIM115 - closed in teardown
        env = server_environment(self.interpreter, self.extra_env)
        logger.info(f"spawning ComfyUI: {' '.join(args)} (log={self.log_path})")
        self.process = subprocess.Popen(
            args,
            cwd=self.root,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=_set_pdeathsig if os.name == "posix" else None,
            start_new_session=True,
        )

    def alive(self) -> bool:
        """Whether the server process is still running."""
        return self.process is not None and self.process.poll() is None

    def _await_health(self) -> None:
        assert self.process is not None and self.base_url is not None
        deadline = time.monotonic() + _HEALTH_DEADLINE_SECONDS
        with httpx.Client(timeout=5.0) as client:
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(
                        f"ComfyUI exited during startup (code {self.process.returncode}); "
                        f"log tail:\n{self.log_tail()}"
                    )
                try:
                    if client.get(f"{self.base_url}/system_stats").status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                time.sleep(1.0)
        raise RuntimeError(
            f"ComfyUI did not answer within {_HEALTH_DEADLINE_SECONDS:.0f}s; log tail:\n{self.log_tail()}"
        )

    def log_tail(self, lines: int = 30) -> str:
        """The last lines of the server log, for error reports."""
        try:
            text = self.log_path.read_text(errors="replace")
        except OSError:
            return "(no log)"
        return "\n".join(text.splitlines()[-lines:])

    def teardown(self) -> None:
        """Stop the server and its whole process group; always safe to call."""
        process = self.process
        if process is not None:
            try:
                if process.poll() is None:
                    _signal_group(process, signal.SIGTERM)
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        _signal_group(process, signal.SIGKILL)
                        process.wait(timeout=5)
            except Exception:  # noqa: BLE001 - teardown is best-effort
                pass
            self.process = None
        if self._log is not None:
            with contextlib.suppress(Exception):
                self._log.close()
            self._log = None


def _signal_group(process: subprocess.Popen[bytes], sig: int) -> None:
    try:
        os.killpg(process.pid, sig)
    except (OSError, AttributeError):
        process.send_signal(sig)


@dataclass(frozen=True, slots=True)
class RenderResult:
    """What one submitted prompt came to."""

    state: Literal["completed", "cancelled"]
    outputs: dict[str, Any]
    """``history[prompt_id]["outputs"]`` for a completed render."""
    sampling_seconds: float
    """Wall time between the first and last sampling progress event."""
    steps_observed: int


class _RenderSession:
    """State for one prompt while its events stream in."""

    def __init__(self, prompt_id: str, on_progress: ProgressCallback, total_steps: int) -> None:
        self.prompt_id = prompt_id
        self.on_progress = on_progress
        self.total_steps = total_steps
        self.stage: VideoStage = "queued"
        self.sampling_started_at: float | None = None
        self.last_step_at: float | None = None
        self.steps_seen = 0
        self.terminal: Literal["success", "error", "interrupted"] | None = None
        self.error: str | None = None
        self.on_progress("queued", None, None, 0.0)

    def _report(self, stage: VideoStage, step: int | None, fraction: float) -> None:
        self.stage = stage
        self.on_progress(stage, step, self.total_steps if step is not None else None, fraction)

    def handle(self, message: dict[str, Any]) -> None:
        """Apply one WebSocket event; events for other prompts are ignored."""
        kind = message.get("type")
        data = cast("dict[str, object]", message.get("data") or {})
        if kind == "executing":
            if data.get("prompt_id") != self.prompt_id:
                return
            node = data.get("node")
            if node is None:
                return
            if node == NODE_SAMPLER and self.sampling_started_at is None:
                # The sampler node starting is the sampling clock's zero;
                # progress events mark completed steps, so timing from the
                # first of them would drop one step's worth of work.
                self.sampling_started_at = time.monotonic()
            stage = stage_for_node(_as_str(node) or "")
            if stage is not None and stage != self.stage and stage != "sampling":
                fraction = {"encoding": 0.05, "decoding": _SAMPLING_SHARE[1], "muxing": 0.95}.get(stage, 0.0)
                self._report(stage, None, fraction)
        elif kind == "progress_state":
            if data.get("prompt_id") != self.prompt_id:
                return
            nodes = cast("dict[str, object]", data.get("nodes") or {})
            sampler = nodes.get(NODE_SAMPLER)
            if not isinstance(sampler, dict):
                return
            progress = cast("dict[str, object]", sampler)
            value = progress.get("value")
            maximum = progress.get("max")
            if not isinstance(value, (int, float)) or not isinstance(maximum, (int, float)) or maximum <= 0:
                return
            step = int(value)
            now = time.monotonic()
            if step >= 1:
                if self.sampling_started_at is None:
                    self.sampling_started_at = now
                self.last_step_at = now
                self.steps_seen = max(self.steps_seen, step)
            low, high = _SAMPLING_SHARE
            self._report("sampling", step, low + (high - low) * min(1.0, value / maximum))
        elif kind == "execution_success":
            if data.get("prompt_id") == self.prompt_id:
                self.terminal = "success"
        elif kind == "execution_interrupted":
            if data.get("prompt_id") == self.prompt_id:
                self.terminal = "interrupted"
        elif kind == "execution_error":
            if data.get("prompt_id") == self.prompt_id:
                self.terminal = "error"
                node = data.get("node_type") or data.get("node_id")
                self.error = f"{node}: {data.get('exception_message', 'unknown error')}"

    @property
    def sampling_seconds(self) -> float:
        """Wall time from the sampler starting to its last reported step."""
        if self.sampling_started_at is None or self.last_step_at is None:
            return 0.0
        return self.last_step_at - self.sampling_started_at


def _as_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _node_error_summary(node: str, info: object) -> str:
    if not isinstance(info, dict):
        return node
    errors = cast("dict[str, Any]", info).get("errors")
    messages = [
        _as_str(cast("dict[str, Any]", entry).get("message")) or "invalid input"
        for entry in cast("list[object]", errors or [])
        if isinstance(entry, dict)
    ]
    return f"{node}: {', '.join(messages)}" if messages else node


async def _submit(session: aiohttp.ClientSession, prompt: ComfyPrompt, client_id: str) -> str:
    prompt_id = str(uuid.uuid4())
    async with session.post(
        "/prompt", json={"prompt": prompt, "client_id": client_id, "prompt_id": prompt_id}
    ) as response:
        body = cast("dict[str, Any]", await response.json(content_type=None))
        if response.status != 200:
            error = cast("dict[str, Any]", body.get("error") or {})
            node_errors = cast("dict[str, object]", body.get("node_errors") or {})
            details = "; ".join(_node_error_summary(node, info) for node, info in node_errors.items())
            raise ComfyRenderError(
                f"ComfyUI rejected the graph: {error.get('message', response.status)}"
                + (f" ({details})" if details else "")
            )
        return _as_str(body.get("prompt_id")) or prompt_id


async def _job_status(session: aiohttp.ClientSession, prompt_id: str) -> tuple[str | None, str | None]:
    """The job's status and error text from the jobs endpoint, if it knows it."""
    try:
        async with session.get(f"/api/jobs/{prompt_id}") as response:
            if response.status != 200:
                return None, None
            job = cast("dict[str, Any]", await response.json(content_type=None))
    except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError):
        return None, None
    error = job.get("execution_error")
    message = None
    if isinstance(error, dict):
        failed = cast("dict[str, Any]", error)
        message = f"{failed.get('node_type') or failed.get('node_id')}: {failed.get('exception_message')}"
    return _as_str(job.get("status")), message


async def _history_outputs(session: aiohttp.ClientSession, prompt_id: str) -> dict[str, Any]:
    async with session.get(f"/history/{prompt_id}") as response:
        response.raise_for_status()
        history = cast("dict[str, Any]", await response.json(content_type=None))
    entry = cast("dict[str, Any] | None", history.get(prompt_id))
    if entry is None:
        raise RuntimeError(f"ComfyUI reported success for {prompt_id} but has no history entry")
    return cast("dict[str, Any]", entry.get("outputs") or {})


async def _cancel(session: aiohttp.ClientSession, prompt_id: str) -> bool:
    """Ask the server to stop the prompt; True when it was running or queued."""
    async with session.post(f"/api/jobs/{prompt_id}/cancel") as response:
        if response.status == 404:
            # Older servers without the jobs API: a global interrupt is the
            # only lever, and this runner submits one prompt at a time.
            async with session.post("/interrupt", json={"prompt_id": prompt_id}):
                pass
            return True
        body = cast("dict[str, Any]", await response.json(content_type=None))
        return bool(body.get("cancelled"))


async def run_prompt(
    base_url: str,
    prompt: ComfyPrompt,
    *,
    total_steps: int,
    on_progress: ProgressCallback,
    is_cancelled: Callable[[], bool],
    server_alive: Callable[[], bool],
) -> RenderResult:
    """Submit one prompt and follow it to completion, cancellation, or failure.

    Raises :class:`ComfyRenderError` when ComfyUI rejects the graph or the
    execution fails (the server is still healthy) and :class:`RuntimeError`
    when the server dies or stops answering (the runner must restart it).
    """
    client_id = uuid.uuid4().hex
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=60)
    async with (
        aiohttp.ClientSession(base_url, timeout=timeout) as session,
        session.ws_connect(f"/ws?clientId={client_id}", heartbeat=30) as socket_,
    ):
        prompt_id = await _submit(session, prompt, client_id)
        state = _RenderSession(prompt_id, on_progress, total_steps)
        cancel_sent_at: float | None = None
        last_poll = time.monotonic()
        while state.terminal is None:
            try:
                message = await asyncio.wait_for(socket_.receive(), timeout=_SOCKET_WAIT_SECONDS)
            except asyncio.TimeoutError:
                message = None
            if message is not None:
                if message.type == aiohttp.WSMsgType.TEXT:
                    try:
                        payload = cast("dict[str, Any]", json.loads(cast("str", message.data)))
                    except json.JSONDecodeError:
                        payload = {}
                    state.handle(payload)
                elif message.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    raise RuntimeError("ComfyUI closed the progress socket mid-render")
                # Binary frames are previews; the runner asked for none.
            # Cancellation and liveness are checked on every pass, not only
            # when the socket is quiet: a render that streams a progress
            # event per step would otherwise never see its cancel.
            if not server_alive():
                raise RuntimeError("ComfyUI exited mid-render")
            now = time.monotonic()
            if cancel_sent_at is None and is_cancelled():
                cancel_sent_at = now
                if not await _cancel(session, prompt_id):
                    # Nothing to interrupt: the prompt already reached a
                    # terminal state the socket has not told us about yet.
                    last_poll = 0.0
            elif cancel_sent_at is not None and now - cancel_sent_at > _CANCEL_GRACE_SECONDS:
                raise RuntimeError("ComfyUI did not stop the render after cancellation")
            if now - last_poll >= _JOB_POLL_SECONDS:
                last_poll = now
                status, error = await _job_status(session, prompt_id)
                if status == "completed":
                    state.terminal = "success"
                elif status == "cancelled":
                    state.terminal = "interrupted"
                elif status == "failed":
                    state.terminal = "error"
                    state.error = error or "execution failed"
        if state.terminal == "error":
            raise ComfyRenderError(f"ComfyUI failed the render ({state.error})")
        if state.terminal == "interrupted":
            return RenderResult("cancelled", {}, state.sampling_seconds, state.steps_seen)
        outputs = await _history_outputs(session, prompt_id)
        return RenderResult("completed", outputs, state.sampling_seconds, state.steps_seen)
