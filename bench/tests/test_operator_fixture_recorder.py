"""Recorder pipe safety and opt-in interoperability with the real relay reducer."""

import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import cast

import pytest

from bench.operator_fixture_client import request_fixture
from bench.operator_fixture_observer import (
    FixtureObservationError,
    FixtureObserver,
    ObservationFlow,
)
from bench.operator_fixture_recorder import (
    FixtureRecorder,
    RecorderSettings,
    aggregate_recorder,
    verified_recorder_copy,
)
from bench.operator_workload_fixture import FixtureSettings, isolated_fixture


def test_verified_copy_binds_bytes_and_rejects_wrong_digest(tmp_path: Path) -> None:
    """Replacing the mutable source does not replace the already-verified copy."""
    source = tmp_path / "source.js"
    source.write_bytes(b"original")
    destination = tmp_path / "verified.js"
    digest = hashlib.sha256(b"original").hexdigest()
    verified_recorder_copy(source, digest, destination)
    source.write_bytes(b"replacement")
    assert destination.read_bytes() == b"original"
    assert destination.stat().st_mode & 0o777 == 0o600
    with pytest.raises(FixtureObservationError):
        verified_recorder_copy(source, digest, tmp_path / "rejected.js")
    assert not (tmp_path / "rejected.js").exists()


async def test_full_sink_queue_never_silently_drops_or_recovers() -> None:
    """A stalled pipe cannot turn a lossy sample into valid aggregate evidence."""
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import time; time.sleep(10)",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        recorder = FixtureRecorder(process)
        for _ in range(512):
            recorder.record({"type": "flow-start", "at": 0, "flow": "chat"})
        with pytest.raises(FixtureObservationError):
            recorder.record({"type": "flow-start", "at": 0, "flow": "chat"})
        with pytest.raises(FixtureObservationError):
            await recorder.finish()
    finally:
        process.kill()
        await process.wait()


def _settings() -> RecorderSettings:
    configured = os.environ.get("SKULK_FIXTURE_RECORDER_DIRECTORY")
    node = os.environ.get("SKULK_FIXTURE_NODE_BINARY")
    if configured is None or node is None:
        pytest.skip("requires explicitly selected local compiled recorder and Node")
    directory = Path(configured)
    return RecorderSettings(
        node_binary=Path(node),
        recorder_module=directory / "workload-observation.js",
        recorder_sha256="a5e2cfce7bffe7490ef034cba167623c04ab8bd0d8bba6f6de76b2c69320a759",
        cli_module=directory / "workload-observation-cli.js",
        cli_sha256="7e7362772c875957e0e80403444dc43371fd9ceecf9e4845f62e4b470a8e512e",
    )


async def test_real_recorder_accepts_adapter_grammar_and_exports_only_aggregate() -> (
    None
):
    """Generated lifecycle interoperability is not a released-client profile."""
    settings = _settings()
    flows: tuple[ObservationFlow, ...] = (
        "cold-launch",
        "foreground-refresh",
        "settled-foreground",
        "background-resume",
        "reconnect",
        "chat",
        "speech",
    )
    now = 0.0
    async with aggregate_recorder(settings) as recorder:
        observer = FixtureObserver(recorder.record, clock=lambda: now)
        for flow in flows:
            observer.begin(flow)
            connection = observer.accepted()
            observer.bind_peer(connection, ("127.0.0.1", 1))
            request = observer.started(("127.0.0.1", 1), "/state")
            observer.body_bytes(request, 32, response=True)
            now += 0.1
            observer.finished(request, successful=True)
            observer.closed(connection)
            observer.end()
        output = await recorder.finish()
    document = cast(dict[str, object], json.loads(output))
    assert document["evidence"] == "unattested-aggregate"
    assert document["totalApplicationBytes"] == 224
    assert all(
        value not in output for value in (b"127.0.0.1", b"/state", b"request-start")
    )


async def test_real_recorder_rejects_incomplete_capture() -> None:
    """Missing app flows yield no partial success, and the process is reaped."""
    settings = _settings()
    with pytest.raises(ExceptionGroup) as failed:
        async with aggregate_recorder(settings) as recorder:
            await recorder.finish()
    assert all(
        isinstance(error, FixtureObservationError) for error in failed.value.exceptions
    )


async def test_real_relay_socket_events_reach_the_real_aggregate_recorder() -> None:
    """Joined generated failure flows verify wiring, not mobile semantic coverage."""
    settings = _settings()
    configured = os.environ.get("SKULK_PAIRED_RELAY_BINARY")
    if configured is None:
        pytest.skip("requires explicitly selected local relay")
    binary = Path(configured)
    with binary.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    fixture_settings = FixtureSettings(
        relay_binary=binary, relay_sha256=digest, lifetime_seconds=60
    )
    flows: tuple[ObservationFlow, ...] = (
        "cold-launch",
        "foreground-refresh",
        "settled-foreground",
        "background-resume",
        "reconnect",
        "chat",
        "speech",
    )
    async with aggregate_recorder(settings) as recorder:
        observer = FixtureObserver(recorder.record)
        async with isolated_fixture(fixture_settings, observer=observer) as fixture:
            package = fixture.pairing_service.create_session()
            remote = package.remote_access
            assert remote is not None
            for flow in flows:
                observer.begin(flow)
                # Deliberately denied generated requests prove failures survive
                # the entire TCP -> ASGI -> pipe -> aggregate path. These flow
                # labels alone do not attest actual chat/speech behavior.
                assert (await request_fixture(remote, "GET", "/state")).status == 401
                async with asyncio.timeout(3):
                    while not observer.idle():
                        await asyncio.sleep(0.01)
                observer.end()
            result = await recorder.finish()
    document = cast(dict[str, object], json.loads(result))
    assert document["evidence"] == "unattested-aggregate"
    observed_flows = cast(dict[str, dict[str, object]], document["flows"])
    assert len(observed_flows) == 7
    assert b"request-start" not in result
