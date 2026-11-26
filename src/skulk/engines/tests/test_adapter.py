"""Adapter session lifecycle against a real stdlib adapter process.

The fake adapter below speaks the protocol with its own framing, as a package
would. Each test chooses a behaviour with a mode argument, so the host is
exercised against real pipes, real exits and real process groups.
"""

import sys
import time
from pathlib import Path

import psutil
import pytest

from skulk.engines.adapter import (
    AdapterFailedError,
    AdapterRefusedError,
    AdapterSession,
)
from skulk.engines.protocol import LoadSettings, ModelIdentity

DIGEST = "c" * 64

FAKE_ADAPTER = r"""
import json, struct, subprocess, sys, time

mode, digest, pid_file = sys.argv[1], sys.argv[2], sys.argv[3]
reported_digest = "0" * 64 if mode == "wrong_digest" else digest


def read_frame():
    head = sys.stdin.buffer.read(4)
    if not head:
        return None
    (length,) = struct.unpack(">I", head)
    return json.loads(sys.stdin.buffer.read(length))


def send(message):
    body = json.dumps(message, separators=(",", ":")).encode()
    sys.stdout.buffer.write(struct.pack(">I", len(body)) + body)
    sys.stdout.buffer.flush()


while True:
    request = read_frame()
    if request is None:
        break
    kind = request["type"]
    base = {"type": kind, "sequence": request["sequence"], "build_digest": reported_digest}
    if kind == "hello":
        send({**base, "version": max(request["versions"])})
    elif kind == "describe":
        if mode == "hang":
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
            with open(pid_file, "w") as handle:
                handle.write(str(child.pid))
            time.sleep(600)
        if mode == "exit":
            sys.exit(0)
        if mode == "junk":
            sys.stdout.buffer.write(struct.pack(">I", 5) + b"nope!")
            sys.stdout.buffer.flush()
            continue
        if mode == "wrong_sequence":
            base["sequence"] += 1
        send({**base, "engine_version": "0.1.0", "model_families": ["qwen"]})
    elif kind == "probe":
        send({**base, "usable": True, "detail": "ok"})
    elif kind == "estimate":
        send({**base, "pool": "gpu", "load_bytes": None, "steady_bytes": 512})
    elif kind == "load":
        if mode == "refuse_load":
            send({**base, "type": "error", "code": "load_refused",
                  "message": "checkpoint is not a supported layout"})
        else:
            applied = dict(request["settings"])
            if mode == "shrink":
                applied["context_tokens"] //= 2
            send({**base, "endpoint_port": 9999, "applied": applied})
    elif kind == "health":
        send({**base, "ready": True})
    elif kind == "cancel":
        send({**base, "request_id": request["request_id"], "acknowledged": True})
    elif kind == "unload":
        send({**base, "released": True})
    elif kind == "stop":
        with open(pid_file, "w") as handle:
            handle.write("stop received")
        send(base)
        break
"""

MODEL = ModelIdentity(
    model_id="Qwen/Example", source_revision="d" * 40, directory="/models/example"
)
SETTINGS = LoadSettings(
    context_tokens=32768, parallel_streams=4, cache_precision="f16", drafting=False
)


def _session(
    tmp_path: Path, mode: str, digest: str = DIGEST
) -> tuple[AdapterSession, Path]:
    script = tmp_path / "adapter.py"
    script.write_text(FAKE_ADAPTER)
    pid_file = tmp_path / "grandchild.pid"
    stderr = (tmp_path / "stderr.log").open("wb")
    session = AdapterSession(
        command=[sys.executable, "-I", "-B", str(script), mode, digest, str(pid_file)],
        environment={"PATH": "/usr/bin:/bin"},
        working_directory=tmp_path,
        stderr=stderr,
        build_digest=DIGEST,
    )
    return session, pid_file


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


def test_full_lifecycle_and_clean_stop(tmp_path: Path) -> None:
    """Every call is answered, the version is negotiated and close stops the process."""
    session, stop_marker = _session(tmp_path, "normal")
    assert session.start() == 1 and session.version == 1
    described = session.describe()
    assert described.model_families == ("qwen",)
    assert session.probe().usable
    assert (
        session.estimate(instance_id="i1", model=MODEL, settings=SETTINGS).steady_bytes
        == 512
    )
    loaded = session.load(instance_id="i1", model=MODEL, settings=SETTINGS)
    assert loaded.endpoint_port == 9999 and loaded.applied == SETTINGS
    assert session.health(instance_id="i1").ready
    assert session.cancel(instance_id="i1", request_id="r1").acknowledged
    assert session.unload(instance_id="i1").released
    process_id = session.process_id
    assert process_id is not None
    session.close()
    session.close()
    assert not session.alive
    assert _gone(process_id)
    # The adapter was asked to stop before its group was terminated.
    assert stop_marker.read_text() == "stop received"


def test_a_refused_load_keeps_the_session_usable(tmp_path: Path) -> None:
    """An engine's typed refusal is an answer, not a channel failure."""
    session, _ = _session(tmp_path, "refuse_load")
    session.start()
    try:
        with pytest.raises(AdapterRefusedError) as refused:
            session.load(instance_id="i1", model=MODEL, settings=SETTINGS)
        assert refused.value.code == "load_refused"
        assert session.alive
        assert session.health(instance_id="i1").ready
    finally:
        session.close()


def test_the_engines_applied_settings_are_reported_not_hidden(tmp_path: Path) -> None:
    """A load that took other settings is returned as is; the caller must compare."""
    session, _ = _session(tmp_path, "shrink")
    session.start()
    try:
        loaded = session.load(instance_id="i1", model=MODEL, settings=SETTINGS)
        assert loaded.applied.context_tokens == SETTINGS.context_tokens // 2
    finally:
        session.close()


def test_a_timed_out_call_kills_the_whole_process_group(tmp_path: Path) -> None:
    """A deadline ends the session and takes engine children down with the adapter."""
    session, pid_file = _session(tmp_path, "hang")
    session.start()
    with pytest.raises(AdapterFailedError) as failed:
        session.describe(deadline_seconds=0.5)
    assert failed.value.reason == "timeout"
    assert not session.alive
    grandchild = int(pid_file.read_text())
    assert _gone(grandchild)
    session.close()


def test_an_adapter_that_exits_is_reported_as_exited(tmp_path: Path) -> None:
    """Early exit surfaces as ``exited`` and the session refuses further calls."""
    session, _ = _session(tmp_path, "exit")
    session.start()
    with pytest.raises(AdapterFailedError) as failed:
        session.describe()
    assert failed.value.reason == "exited"
    with pytest.raises(AdapterFailedError):
        session.probe()
    session.close()


def test_a_malformed_frame_is_a_protocol_failure(tmp_path: Path) -> None:
    """Garbage on the channel is fatal: the session cannot be trusted to resynchronize."""
    session, _ = _session(tmp_path, "junk")
    session.start()
    with pytest.raises(AdapterFailedError) as failed:
        session.describe()
    assert failed.value.reason == "protocol"
    session.close()


def test_a_mismatched_sequence_is_fatal(tmp_path: Path) -> None:
    """A response answering a different request means the channel is out of step."""
    session, _ = _session(tmp_path, "wrong_sequence")
    session.start()
    with pytest.raises(AdapterFailedError) as failed:
        session.describe()
    assert failed.value.reason == "sequence"
    session.close()


def test_a_different_build_digest_is_refused_at_start(tmp_path: Path) -> None:
    """An adapter answering for another package build never becomes usable."""
    session, _ = _session(tmp_path, "wrong_digest")
    with pytest.raises(AdapterFailedError) as failed:
        session.start()
    assert failed.value.reason == "identity"
    assert not session.alive
    session.close()
