"""Signed release downloads, durable acceptance and refusal before execution."""

import asyncio
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from skulk.extensions.runtime_artifacts import RuntimeTrust
from skulk.extensions.runtime_download import (
    InstallRequest,
    ReleaseSource,
    RuntimeDownloads,
    SourceUpdate,
)
from skulk.extensions.runtime_files import read_private, write_private
from skulk.extensions.tests.test_runtime_install import artifacts


def prepared(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[RuntimeDownloads, bytes, Path, list[str]]:
    """Supply an authenticated mock HTTPS origin and real signed offline artifacts."""
    source = tmp_path / "source"
    metadata, trust, host = artifacts(source)
    root = tmp_path / "installation"
    write_private(root / "publisher-trust.json", trust.model_dump_json().encode())
    write_private(
        root / "release-source.json",
        ReleaseSource(
            revision=1,
            base_url="https://releases.example.test/private/",
            metadata_filename="runtime.json",
            credential_reference="a" * 32,
        )
        .model_dump_json()
        .encode(),
    )
    write_private(root / "feed-credentials" / ("a" * 32), b"fixture-private-token")
    monkeypatch.setattr("skulk.extensions.runtime_install.measure_host", lambda: host)
    calls: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "releases.example.test"
        assert request.headers["Authorization"] == "Bearer fixture-private-token"
        calls.append(request.url.path)
        name = request.url.path.rsplit("/", 1)[1]
        return httpx.Response(
            200,
            content=metadata if name == "runtime.json" else read_private(source / name),
        )

    return (
        RuntimeDownloads(root, transport=httpx.MockTransport(respond)),
        metadata,
        source,
        calls,
    )


async def test_review_and_owned_staging_survive_request_lifetime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review precedes artifacts; accepted staging installs exact bytes and never enables."""
    downloads, _, _, calls = prepared(tmp_path, monkeypatch)
    before = await downloads.inspect()
    assert calls == ["/private/runtime.json"]
    assert before.version == "1.0.0" and before.permissions == (
        "local synthetic operation",
    )
    assert "fixture-private-token" not in before.model_dump_json()
    request = InstallRequest(
        operation_id="b" * 32,
        runtime_digest=before.runtime_digest,
        expected_source_revision=1,
    )
    accepted = await downloads.submit(request)
    assert accepted.state == "accepted"
    assert (await downloads.submit(request)).request == request
    assert downloads.work is not None
    await downloads.work
    staged = downloads.operation(request.operation_id)
    assert staged.state == "staged" and staged.downloaded_bytes == before.artifact_bytes
    assert len(calls) == 3
    assert not (downloads.root / "selected-runtime.json").exists()
    async with downloads.installer.locked_generation(before.runtime_digest) as (
        verified,
        _,
    ):
        assert verified.digest == before.runtime_digest
    await downloads.close()
    restarted = RuntimeDownloads(downloads.root)
    assert restarted.current() == staged
    assert await restarted.submit(request) == staged
    assert restarted.work is None
    await restarted.close()


@pytest.mark.parametrize("failure", ["redirect", "truncated", "tampered", "oversized"])
async def test_bad_artifacts_never_stage_or_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """Reject signed-byte failures without activating or replaying a retained operation."""
    downloads, _, source, calls = prepared(tmp_path, monkeypatch)
    reviewed = await downloads.inspect()

    def bad(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if failure == "redirect":
            return httpx.Response(
                302, headers={"Location": "https://untrusted.example.test/stolen"}
            )
        original = read_private(source / "bundle.pyz")
        body = (
            original[:-1]
            if failure == "truncated"
            else (original + b"x" if failure == "oversized" else b"x" * len(original))
        )
        return httpx.Response(200, content=body)

    downloads.transport = httpx.MockTransport(bad)
    request = InstallRequest(
        operation_id="b" * 32,
        runtime_digest=reviewed.runtime_digest,
        expected_source_revision=1,
    )
    await downloads.submit(request)
    assert downloads.work is not None
    await downloads.work
    failed = downloads.current()
    assert failed is not None and failed.state == "recovery_required"
    assert failed.error_code == "download_failed"
    assert not (downloads.root / "generations").exists()
    count = len(calls)
    assert await downloads.submit(request) == failed
    assert len(calls) == count == 2
    assert "fixture-private-token" not in failed.model_dump_json()
    await downloads.close()


async def test_missing_credential_and_untrusted_metadata_refuse_before_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No vault fallback, redirect or artifact I/O precedes source/signature validation."""
    downloads, metadata, _, calls = prepared(tmp_path, monkeypatch)
    credential = downloads.root / "feed-credentials" / ("a" * 32)
    credential.unlink()
    with pytest.raises(FileNotFoundError):
        await downloads.inspect()
    assert not calls
    write_private(credential, b"fixture-private-token")
    downloads.transport = httpx.MockTransport(
        lambda _: httpx.Response(
            200, content=metadata.replace(b'"fixture"', b'"unknown"')
        )
    )
    with pytest.raises(ValueError):
        await downloads.inspect()
    assert not (downloads.directory / "reviews").exists()
    await downloads.close()


async def test_interruption_and_revision_conflict_do_not_restart_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancellation leaves retained intent and a new manager does not repeat downloads."""
    downloads, _, _, calls = prepared(tmp_path, monkeypatch)
    review = await downloads.inspect()
    request = InstallRequest(
        operation_id="b" * 32,
        runtime_digest=review.runtime_digest,
        expected_source_revision=2,
    )
    with pytest.raises(ValueError, match="revision"):
        await downloads.submit(request)
    assert len(calls) == 1
    request = request.model_copy(update={"expected_source_revision": 1})
    started = asyncio.Event()

    async def hang(_: httpx.Request) -> httpx.Response:
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("cancelled download returned")

    downloads.transport = httpx.MockTransport(hang)
    await downloads.submit(request)
    await started.wait()
    await downloads.close()
    restarted = RuntimeDownloads(downloads.root)
    prior = restarted.current()
    assert prior is not None and prior.state == "recovery_required"
    assert await restarted.submit(request) == prior and restarted.work is None
    with pytest.raises(ValueError, match="identity"):
        await restarted.submit(request.model_copy(update={"runtime_digest": "c" * 64}))
    await restarted.close()


@pytest.mark.parametrize(
    "url",
    [
        "http://releases.example.test/",
        "https://secret@releases.example.test/",
        "https://releases.example.test/?token=secret",
        "https://releases.example.test/#fragment",
        "https://releases.example.test/missing-slash",
    ],
)
def test_source_endpoint_is_unambiguous(url: str) -> None:
    """Sources use one explicit HTTPS directory without URL credentials or redirects."""
    with pytest.raises(ValueError):
        ReleaseSource(revision=1, base_url=url, metadata_filename="runtime.json")


async def test_source_rotation_preserves_old_credentials_and_fences_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Source changes cannot silently forward a retained bearer to a new origin."""
    downloads, _, _, _ = prepared(tmp_path, monkeypatch)
    trust = RuntimeTrust.model_validate_json(
        read_private(downloads.root / "publisher-trust.json")
    )
    update = SourceUpdate(
        expected_revision=1,
        base_url="https://different.example.test/",
        metadata_filename="runtime.json",
        trust=trust,
    )
    with pytest.raises(ValueError, match="credential replacement"):
        await downloads.configure(update)
    assert downloads.source().revision == 1
    rotated = await downloads.configure(
        update.model_copy(update={"token": SecretStr("replacement-secret")})
    )
    assert rotated.revision == 2 and rotated.credential_ready
    assert rotated.credential_reference != "a" * 32
    assert "replacement-secret" not in rotated.model_dump_json()
    assert (
        read_private(downloads.root / "feed-credentials" / ("a" * 32))
        == b"fixture-private-token"
    )
    assert rotated.credential_reference is not None
    assert (
        read_private(downloads.root / "feed-credentials" / rotated.credential_reference)
        == b"replacement-secret"
    )
    with pytest.raises(ValueError, match="revision"):
        await downloads.configure(update)
    await downloads.close()
