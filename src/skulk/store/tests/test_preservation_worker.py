"""Publication acknowledgement and pinned storage survive real process failures."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest
from pydantic import JsonValue, TypeAdapter

from skulk.store.preservation_export import preservation_generation
from skulk.store.preservation_worker import WorkerRequest
from skulk.store.tests.test_preservation_export import make_preservation_installation


@pytest.mark.asyncio
@pytest.mark.parametrize("parent_lost", [False, True])
async def test_pinned_archive_and_no_publication_without_durable_ack(
    tmp_path: Path, parent_lost: bool
) -> None:
    installed = make_preservation_installation(tmp_path, "assistant")
    descriptor = os.open(
        installed.archive, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    )
    original = installed.archive.with_name("original-archive")
    installed.archive.rename(original)
    installed.archive.mkdir()
    staging = tmp_path / "staging"
    staging.mkdir(mode=0o700)
    info = os.fstat(descriptor)
    request = WorkerRequest(
        kind="preserve",
        archive_descriptor=descriptor,
        archive_device=info.st_dev,
        archive_inode=info.st_ino,
        model_directory=installed.primary,
        source_root=installed.primary.parent,
        generation_id=preservation_generation(
            installed.primary, artifact_roots=(installed.primary.parent,)
        ),
        snapshot_parent=staging,
        key_path=installed.key,
        trust=installed.trust,
    )
    request_path = tmp_path / "request.json"
    request_path.write_text(request.model_dump_json())
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "skulk.store.preservation_worker",
        str(request_path),
        pass_fds=(descriptor,),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    os.close(descriptor)
    assert process.stdin is not None and process.stdout is not None
    try:
        async with asyncio.timeout(15):
            expected: str | None = None
            while line := await process.stdout.readline():
                value = TypeAdapter(dict[str, JsonValue]).validate_json(line)
                if value["event"] == "expected":
                    candidate = value["manifest_id"]
                    assert isinstance(candidate, str)
                    expected = candidate
                    break
            assert expected is not None
            assert not (original / "manifests" / (expected + ".json")).exists()
            if parent_lost:
                process.stdin.close()
                assert await process.wait() == 75
                assert not (original / "manifests" / (expected + ".json")).exists()
            else:
                process.stdin.write(("publish:" + expected + "\n").encode())
                await process.stdin.drain()
                output = await process.stdout.read()
                assert await process.wait() == 0, (
                    await process.stderr.read() if process.stderr is not None else b""
                )
                assert b'"event":"result"' in output
                assert (original / "manifests" / (expected + ".json")).is_file()
            assert not tuple(installed.archive.iterdir())
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()


def test_frozen_worker_uses_the_known_internal_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A frozen Core executable cannot run Python's source-only -m invocation."""
    from skulk.store.preservation_host import preservation_worker_command

    request = tmp_path / "private request.json"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert preservation_worker_command(request) == (
        sys.executable,
        "_preservation-worker",
        str(request.absolute()),
    )
