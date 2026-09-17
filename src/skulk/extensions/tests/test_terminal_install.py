"""Guided terminal installation against real manager IPC and signed offline wheels."""

import asyncio
import getpass
import hashlib
import io
import json
import time
import warnings
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import JsonValue, SecretStr, TypeAdapter

from skulk.extensions.runtime_artifacts import RuntimeTrust, canonical_json
from skulk.extensions.runtime_attachment import HostSettings, ServiceConnection
from skulk.extensions.runtime_catalog import (
    CatalogEntryReview,
    CatalogReview,
    CatalogSourceUpdate,
    HostCatalog,
)
from skulk.extensions.runtime_download import RuntimeDownloads, SourceUpdate
from skulk.extensions.runtime_files import (
    private_directory,
    read_private,
    write_private,
)
from skulk.extensions.runtime_manager import (
    CatalogInstall,
    CatalogInstallRequest,
    CatalogRegistration,
    CatalogRequest,
    InstallationRequest,
    InstallSubmission,
    ManagerRequest,
    RuntimeManager,
    SourceRegistration,
    SubmitRequest,
    manager_request,
)
from skulk.extensions.terminal_install import TerminalInstaller
from skulk.extensions.tests.test_runtime_install import artifacts
from skulk.extensions.tests.test_runtime_service import OWNER_SOURCE


@dataclass
class Journey:
    """External HTTPS/terminal boundaries around the real manager and installer."""

    manager: RuntimeManager
    trust: RuntimeTrust
    source: Path
    metadata: bytes
    signing_key: Ed25519PrivateKey
    releases: dict[int, tuple[Path, bytes]] = field(default_factory=dict)
    fail_response: str | None = None
    fail_download: bool = False
    requests: list[ManagerRequest] = field(default_factory=list)
    output: list[str] = field(default_factory=list)
    urls: list[str] = field(default_factory=list)
    catalog: bytes | None = None
    authorized: list[tuple[str, str, bool]] = field(default_factory=list)

    async def request(self, request: ManagerRequest) -> dict[str, JsonValue]:
        """Lose a chosen accepted response without cancelling its manager task."""
        self.requests.append(request)
        response = await manager_request(self.manager.root, request)
        if request.action == self.fail_response:
            self.fail_response = None
            raise OSError("external connection lost")
        return response

    def fields(self, *decisions: str) -> Iterator[str]:
        """Provide external trust/source inputs, without any internal identifiers."""
        return iter(
            [
                "https://releases.example.test/",
                "",
                "fixture",
                self.trust.publishers["fixture"],
                datetime.fromtimestamp(self.trust.expires_at, UTC).isoformat(),
                *decisions,
            ]
        )

    def terminal(self, answers: Iterator[str]) -> TerminalInstaller:
        """Retain displayed output while never reflecting hidden credential input."""

        def prompt(_question: str) -> str:
            return next(answers)

        return TerminalInstaller(
            self.request,
            prompt,
            lambda _question: "hidden-test-feed-token",
            self.output.append,
            # Real downloads, venv creation and pip keep wall-clock timing.
            # Accelerating only the poller exhausts its budget on slower hosts.
        )

    def publish(self, sequence: int, *, bundle: str | None = None) -> None:
        """Serve ``sequence`` from the feed: built once per sequence, then reused.

        Re-serving an earlier sequence reuses its exact metadata, so the feed
        rolls back rather than equivocating (the same sequence with different
        bytes, which inspection refuses on its own). ``bundle`` publishes the
        sequence under another bundle identity.
        """
        if sequence not in self.releases:
            source = self.source.parent / f"source-{sequence}"
            metadata, _, _ = artifacts(
                source,
                owner_source=OWNER_SOURCE,
                signing_key=self.signing_key,
                sequence=sequence,
                bundle_id=bundle if bundle is not None else "example.plugin",
            )
            self.releases[sequence] = (source, metadata)
        self.source, self.metadata = self.releases[sequence]

    @property
    def identifier(self) -> str:
        """Read the generated identity from the first registration request."""
        request = self.requests[0]
        assert isinstance(request, InstallationRequest)
        return request.plugin_id

    def list_releases(
        self,
        feeds: dict[int, str],
        *,
        revision: int,
        digests: dict[int, str] | None = None,
        overrides: dict[int, dict[str, JsonValue]] | None = None,
    ) -> None:
        """Serve a signed catalog listing the built sequences at the given feeds.

        Each entry is derived from the sequence's real signed record the way
        the publisher's tool derives it; ``digests`` lets a listing claim a
        record other than the one its feed serves.
        """
        now = int(time.time())
        entries: list[JsonValue] = []
        for sequence, feed in sorted(feeds.items()):
            _, metadata = self.releases[sequence]
            objects = TypeAdapter(dict[str, JsonValue])
            signed = objects.validate_json(metadata)
            runtime = objects.validate_python(signed["runtime"])
            release = objects.validate_python(runtime["release"])
            manifest = objects.validate_python(release["manifest"])
            artifact_size = TypeAdapter(int).validate_python(release["artifact_size"])
            wheel_sizes = TypeAdapter(list[int]).validate_python(
                [
                    objects.validate_python(wheel)["size"]
                    for wheel in TypeAdapter(list[JsonValue]).validate_python(
                        runtime["wheels"]
                    )
                ]
            )
            entries.append(
                {
                    "bundle_id": manifest["bundle_id"],
                    "bundle_version": manifest["bundle_version"],
                    "title": "Example plugin",
                    "publisher": release["publisher"],
                    "sequence": release["sequence"],
                    "feed_url": feed,
                    "release_sha256": hashlib.sha256(metadata).hexdigest(),
                    "release_digest": (digests or {}).get(
                        sequence,
                        hashlib.sha256(canonical_json(runtime)).hexdigest(),
                    ),
                    "runtime_platform": runtime["platform"],
                    "artifact_sha256": manifest["executable_sha256"],
                    "artifact_size": artifact_size,
                    "transfer_bytes": artifact_size + sum(wheel_sizes),
                    "platforms": release["platforms"],
                    "skulk_build_sha256": release["skulk_build_sha256"],
                    "permissions": release["permissions"],
                    "descriptors": ["example.echo@1.0.0"],
                    "surfaces": [],
                    "operations": False,
                    "steward_risks": [],
                    "expires_at": release["expires_at"],
                    **(overrides or {}).get(sequence, {}),
                }
            )
        catalog: JsonValue = {
            "protocol": 1,
            "publisher": "fixture",
            "revision": revision,
            "created_at": now - 1,
            "expires_at": now + 3600,
            "entries": entries,
        }
        self.catalog = json.dumps(
            {
                "catalog": catalog,
                "signature": self.signing_key.sign(canonical_json(catalog)).hex(),
            }
        ).encode()

    async def configure_catalog(self) -> None:
        """Point the host's catalog at the same protected origin as the feed."""
        response = await self.request(
            CatalogRegistration(
                request=CatalogSourceUpdate(
                    expected_revision=0,
                    base_url="https://releases.example.test/",
                    trust=self.trust,
                    token=SecretStr("hidden-test-feed-token"),
                )
            )
        )
        assert "result" in response

    async def catalog_review(self) -> CatalogReview:
        """Read the catalog as the terminal does."""
        response = await self.request(CatalogRequest(action="read_catalog"))
        return CatalogReview.model_validate_json(json.dumps(response["result"]))


@asynccontextmanager
async def journey(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, artifact_delay: float = 0.0
) -> AsyncIterator[Journey]:
    """Use real private sockets, dependency installation and supervised processes."""
    source = tmp_path / "source"
    signing_key = Ed25519PrivateKey.generate()
    metadata, trust, host = artifacts(
        source, owner_source=OWNER_SOURCE, signing_key=signing_key
    )
    monkeypatch.setattr("skulk.extensions.runtime_install.measure_host", lambda: host)
    # The catalog review compares listings with the host the manager measures.
    monkeypatch.setattr("skulk.extensions.runtime_manager.measure_host", lambda: host)
    root = tmp_path / "manager"
    private_directory(root)
    write_private(
        root / "host.json",
        HostSettings(transport_node_id="fixture-peer").model_dump_json().encode(),
    )
    # The transports are patched before the manager exists: it opens its
    # catalog in its constructor, and the fixture the responder reads is
    # bound by name once the manager is built.
    fixture: Journey

    async def respond(request: httpx.Request) -> httpx.Response:
        authorized = (
            "Authorization" in request.headers
            and request.headers["Authorization"] == "Bearer hidden-test-feed-token"
        )
        fixture.authorized.append((request.url.host, request.url.path, authorized))
        if request.url.host != "releases.example.test":
            # Another origin: nothing is served there, and the credential
            # given for the feed's origin must not have been presented.
            return httpx.Response(404)
        assert authorized
        fixture.urls.append(request.url.path)
        name = request.url.path.rsplit("/", 1)[-1]
        if name == "catalog.json":
            assert fixture.catalog is not None
            return httpx.Response(200, content=fixture.catalog)
        # A catalog lists each sequence in its own directory; the plain
        # root keeps serving whatever is published, as before.
        source, metadata = fixture.source, fixture.metadata
        first = request.url.path.strip("/").split("/")[0]
        if first.isdigit() and int(first) in fixture.releases:
            source, metadata = fixture.releases[int(first)]
        if fixture.fail_download and name != "release.json":
            return httpx.Response(503)
        if name != "release.json":
            await asyncio.sleep(artifact_delay)
        return httpx.Response(
            200,
            content=metadata if name == "release.json" else read_private(source / name),
        )

    def downloads(root: Path) -> RuntimeDownloads:
        return RuntimeDownloads(root, transport=httpx.MockTransport(respond))

    def catalog(root: Path) -> HostCatalog:
        return HostCatalog(root, transport=httpx.MockTransport(respond))

    monkeypatch.setattr("skulk.extensions.runtime_manager.RuntimeDownloads", downloads)
    monkeypatch.setattr("skulk.extensions.runtime_manager.HostCatalog", catalog)
    manager = RuntimeManager(root)
    fixture = Journey(manager, trust, source, metadata, signing_key)
    fixture.releases[1] = (source, metadata)
    await manager.start()
    started = time.monotonic()
    try:
        yield fixture
    finally:
        # Pytest retains this on failure, including failures raised by an
        # expected-exception assertion before the intended operation is reached.
        print(
            f"Terminal journey elapsed: {time.monotonic() - started:.2f}s; "
            f"last terminal states: {json.dumps(fixture.output[-8:])}"
        )
        await manager.close()


@pytest.mark.parametrize("artifact_delay", [0.0, 4.0])
async def test_generated_identity_trust_permissions_and_real_installation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, artifact_delay: float
) -> None:
    """External inputs reach verified staging and activation without JSON or paths."""
    # The delayed case exceeds the old 240 * 10ms test polling budget while
    # preserving real manager IPC, signed artifacts, staging and activation.
    async with journey(tmp_path, monkeypatch, artifact_delay=artifact_delay) as fixture:
        identifier = await fixture.terminal(fixture.fields("y", "y", "y")).run()
        assert identifier.startswith("managed.") and len(identifier) == 40
        selection = fixture.manager.controllers[identifier].selector.current()
        assert selection is not None and selection.enabled
        assert selection.revision == 1
        assert len(fixture.urls) == 3
        assert "hidden-test-feed-token" not in "\n".join(fixture.output)
        assert fixture.output[0].endswith(identifier)
        assert any("local synthetic operation" in line for line in fixture.output)
        assert not any("RunPod" in line for line in fixture.output)
        await fixture.terminal(iter(())).run(identifier)
        assert sum(isinstance(r, InstallSubmission) for r in fixture.requests) == 1
        assert sum(isinstance(r, SubmitRequest) for r in fixture.requests) == 1
        assert sum(isinstance(r, SourceRegistration) for r in fixture.requests) == 1


async def test_a_newer_release_at_the_source_upgrades_the_installation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sequence N+1 is staged beside N and activated over it with one command."""
    async with journey(tmp_path, monkeypatch) as fixture:
        identifier = await fixture.terminal(fixture.fields("y", "y", "y")).run()
        selector = fixture.manager.controllers[identifier].selector
        first = selector.current()
        assert first is not None and first.sequence == 1 and first.revision == 1
        fixture.publish(2)
        # The retained operation is not resumed: the newer release is
        # reviewed, staged and confirmed as a replacement.
        await fixture.terminal(iter(("y", "y"))).run(identifier)
        second = selector.current()
        assert second is not None and second.enabled
        assert second.sequence == 2 and second.revision == 2
        assert second.runtime_digest != first.runtime_digest
        assert sum(isinstance(r, InstallSubmission) for r in fixture.requests) == 2
        assert sum(isinstance(r, SubmitRequest) for r in fixture.requests) == 2
        assert any("Selected release sequence: 1" in line for line in fixture.output)
        # Running again with nothing new at the source changes nothing.
        await fixture.terminal(iter(())).run(identifier)
        assert sum(isinstance(r, InstallSubmission) for r in fixture.requests) == 2
        assert sum(isinstance(r, SubmitRequest) for r in fixture.requests) == 2
        # The feed rolled back to the exact sequence 1 it served before: the
        # terminal refuses the rollback before any transfer, nothing is
        # submitted or staged, and the selection stands.
        fixture.publish(1)
        with pytest.raises(ValueError, match="rollback"):
            await fixture.terminal(iter(("y",))).run(identifier)
        assert selector.current() == second
        assert sum(isinstance(r, InstallSubmission) for r in fixture.requests) == 2
        # A selection with no retained download operation (selected outside
        # this wizard) is compared the same way: nothing to stage at the
        # selected release, a rollback refused before transfer, and a newer
        # release upgraded over it.
        retained = (
            fixture.manager.root
            / "installations"
            / identifier
            / "downloads"
            / "current.json"
        )
        assert retained.exists()
        retained.unlink()
        fixture.publish(2)
        await fixture.terminal(iter(())).run(identifier)
        assert any("retained at sequence 2" in line for line in fixture.output)
        fixture.publish(1)
        with pytest.raises(ValueError, match="rollback"):
            await fixture.terminal(iter(("y",))).run(identifier)
        assert sum(isinstance(r, InstallSubmission) for r in fixture.requests) == 2
        fixture.publish(3)
        await fixture.terminal(iter(("y", "y"))).run(identifier)
        third = selector.current()
        assert third is not None and third.sequence == 3 and third.revision == 3
        assert sum(isinstance(r, InstallSubmission) for r in fixture.requests) == 3
        # A staged release whose activation was declined still raises the
        # high-water mark: the feed going back below it is a rollback.
        fixture.publish(4)
        await fixture.terminal(iter(("y", "n"))).run(identifier)
        assert selector.current() == third
        assert sum(isinstance(r, InstallSubmission) for r in fixture.requests) == 4
        fixture.publish(3)
        with pytest.raises(ValueError, match="rollback"):
            await fixture.terminal(iter(("y",))).run(identifier)
        assert sum(isinstance(r, InstallSubmission) for r in fixture.requests) == 4
        # A different bundle at the source is refused before any transfer.
        fixture.publish(5, bundle="example.other")
        with pytest.raises(ValueError, match="another bundle"):
            await fixture.terminal(iter(("y",))).run(identifier)
        assert sum(isinstance(r, InstallSubmission) for r in fixture.requests) == 4


async def test_a_listed_release_installs_and_upgrades_from_the_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The catalog supplies feed and publisher; consent, staging and activation stay."""
    async with journey(tmp_path, monkeypatch) as fixture:
        fixture.list_releases({1: "https://releases.example.test/1/"}, revision=1)
        await fixture.configure_catalog()
        identifier = await fixture.terminal(iter(("y", "y", "y"))).run_from_catalog(
            "example.plugin"
        )
        assert identifier.startswith("managed.") and len(identifier) == 40
        selector = fixture.manager.controllers[identifier].selector
        first = selector.current()
        assert first is not None and first.enabled and first.sequence == 1
        # No source prompt was answered: the listing bound the source, and
        # the record, artifact and wheel came from the listed directory with
        # the catalog's credential (same origin).
        assert not any(isinstance(r, SourceRegistration) for r in fixture.requests)
        assert sum(isinstance(r, CatalogInstallRequest) for r in fixture.requests) == 1
        assert fixture.urls[0] == "/catalog.json"
        assert all(path.startswith("/1/") for path in fixture.urls[1:])
        assert all(authorized for _, _, authorized in fixture.authorized)
        assert "hidden-test-feed-token" not in "\n".join(fixture.output)
        assert "releases.example.test" not in "\n".join(fixture.output)
        assert any('"title": "Example plugin"' in line for line in fixture.output)
        assert fixture.output[1].endswith(identifier)
        # The newest listing that fits this host upgrades the installation
        # in place; the older listing is refused as a rollback before the
        # source is touched.
        fixture.publish(2)
        fixture.list_releases(
            {
                1: "https://releases.example.test/1/",
                2: "https://releases.example.test/2/",
            },
            revision=2,
        )
        await fixture.terminal(iter(("y", "y", "y"))).run_from_catalog(
            "example.plugin", plugin_id=identifier
        )
        second = selector.current()
        assert second is not None and second.enabled and second.sequence == 2
        downloads = fixture.manager.downloads[identifier]
        revision = downloads.source_status().revision
        with pytest.raises(ValueError, match="manager request incomplete"):
            await fixture.terminal(iter(("y",))).run_from_catalog(
                "example.plugin", sequence=1, plugin_id=identifier
            )
        assert downloads.source_status().revision == revision
        assert selector.current() == second
        # Running the ordinary command afterwards resumes on the bound source.
        await fixture.terminal(iter(())).run(identifier)
        assert sum(isinstance(r, InstallSubmission) for r in fixture.requests) == 2


async def test_catalog_installs_refuse_by_name_before_any_transfer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unlisted, superseded, mismatched, foreign-bundle and foreign-origin listings."""
    async with journey(tmp_path, monkeypatch) as fixture:
        fixture.list_releases({1: "https://releases.example.test/1/"}, revision=1)
        await fixture.configure_catalog()
        stale = await fixture.catalog_review()
        with pytest.raises(ValueError, match="no listed release"):
            await fixture.terminal(iter(())).run_from_catalog(
                "example.plugin", sequence=2
            )
        with pytest.raises(ValueError, match="no listed release"):
            await fixture.terminal(iter(())).run_from_catalog("example.other")
        assert not any(isinstance(r, CatalogInstallRequest) for r in fixture.requests)
        identifier = await fixture.terminal(iter(("y", "y", "y"))).run_from_catalog(
            "example.plugin"
        )
        selector = fixture.manager.controllers[identifier].selector
        installed = selector.current()
        assert installed is not None and installed.sequence == 1
        # A listing claiming a record other than the one its feed serves is
        # refused after inspection; nothing is staged.
        fixture.publish(2)
        fixture.list_releases(
            {
                1: "https://releases.example.test/1/",
                2: "https://releases.example.test/2/",
            },
            revision=2,
            digests={2: "f" * 64},
        )
        downloads = fixture.manager.downloads[identifier]
        before = downloads.source()
        with pytest.raises(ValueError, match="manager request incomplete"):
            await fixture.terminal(iter(("y",))).run_from_catalog(
                "example.plugin", plugin_id=identifier
            )
        assert sum(isinstance(r, InstallSubmission) for r in fixture.requests) == 1
        # The refused binding put the installation back on the source it had,
        # with the same credential under a fresh reference, so the plain
        # command still resumes there and nothing is staged.
        restored = downloads.source()
        assert (restored.base_url, restored.metadata_filename) == (
            before.base_url,
            before.metadata_filename,
        )
        assert restored.revision == before.revision + 2
        assert restored.credential_reference not in (None, before.credential_reference)
        await fixture.terminal(iter(())).run(identifier)
        assert fixture.urls[-1] == "/1/release.json"
        assert sum(isinstance(r, InstallSubmission) for r in fixture.requests) == 1
        assert sum(isinstance(r, SubmitRequest) for r in fixture.requests) == 1
        # The installer fence held by another operation is a refusal like
        # any other: the source goes back, and the fence's refusal is raised.
        before = downloads.source()

        async def fenced() -> object:
            raise BlockingIOError("installer fence busy")

        downloads.inspect = fenced  # type: ignore[method-assign]
        try:
            with pytest.raises(ValueError, match="manager request incomplete"):
                await fixture.terminal(iter(("y",))).run_from_catalog(
                    "example.plugin", plugin_id=identifier
                )
        finally:
            del downloads.inspect
        restored = downloads.source()
        assert (restored.base_url, restored.revision) == (
            before.base_url,
            before.revision + 2,
        )
        # The request deadline cancelling a slow inspection rolls back the
        # same way, and the deadline still reports.
        before = downloads.source()
        current = await fixture.catalog_review()

        async def slow() -> object:
            await asyncio.sleep(30)
            raise AssertionError("unreachable")

        downloads.inspect = slow  # type: ignore[method-assign]
        try:
            with pytest.raises(TimeoutError):
                async with asyncio.timeout(0.5):
                    await fixture.manager.dispatch(
                        CatalogInstallRequest(
                            request=CatalogInstall(
                                catalog_sha256=current.catalog_sha256,
                                bundle_id="example.plugin",
                                sequence=2,
                                runtime_platform="macos-arm64",
                                plugin_id=identifier,
                            )
                        )
                    )
        finally:
            del downloads.inspect
        restored = downloads.source()
        assert (restored.base_url, restored.revision) == (
            before.base_url,
            before.revision + 2,
        )
        # The digest of a superseded listing is refused: consent names the
        # listing the operator reviewed, and that one is no longer current.
        response = await fixture.request(
            CatalogInstallRequest(
                request=CatalogInstall(
                    catalog_sha256=stale.catalog_sha256,
                    bundle_id="example.plugin",
                    sequence=1,
                    runtime_platform="macos-arm64",
                    plugin_id=identifier,
                )
            )
        )
        assert "error" in response
        # Another bundle cannot take over an installation.
        fixture.publish(3, bundle="example.other")
        fixture.list_releases(
            {
                1: "https://releases.example.test/1/",
                3: "https://releases.example.test/3/",
            },
            revision=3,
        )
        revision = fixture.manager.downloads[identifier].source_status().revision
        with pytest.raises(ValueError, match="manager request incomplete"):
            await fixture.terminal(iter(("y",))).run_from_catalog(
                "example.other", plugin_id=identifier
            )
        assert (
            fixture.manager.downloads[identifier].source_status().revision == revision
        )
        assert selector.current() == installed
        # A feed at another origin is read without the catalog's credential.
        fixture.list_releases(
            {
                1: "https://releases.example.test/1/",
                3: "https://elsewhere.example.test/3/",
            },
            revision=4,
        )
        with pytest.raises(ValueError, match="manager request incomplete"):
            await fixture.terminal(iter(("y",))).run_from_catalog("example.other")
        elsewhere = [
            authorized
            for host, _, authorized in fixture.authorized
            if host == "elsewhere.example.test"
        ]
        assert elsewhere == [False]
        assert sum(isinstance(r, InstallSubmission) for r in fixture.requests) == 1
        # A listing whose consent facts differ from the verified record, or
        # whose document hash differs, is refused with the source restored.
        mismatches: list[tuple[int, dict[str, JsonValue]]] = [
            (10, {"permissions": ["something else"]}),
            (11, {"release_sha256": "e" * 64}),
        ]
        for revision, override in mismatches:
            fixture.list_releases(
                {
                    1: "https://releases.example.test/1/",
                    2: "https://releases.example.test/2/",
                },
                revision=revision,
                overrides={2: override},
            )
            before = downloads.source()
            with pytest.raises(ValueError, match="manager request incomplete"):
                await fixture.terminal(iter(("y",))).run_from_catalog(
                    "example.plugin", plugin_id=identifier
                )
            restored = downloads.source()
            assert (restored.base_url, restored.revision) == (
                before.base_url,
                before.revision + 2,
            )
        # A feed that serves another record after the binding reply is
        # refused before any transfer: staging is pinned to the bound digest.
        fixture.publish(4)
        fixture.list_releases(
            {
                1: "https://releases.example.test/1/",
                2: "https://releases.example.test/2/",
            },
            revision=12,
        )
        second = fixture.releases[2]

        async def swapping(request: ManagerRequest) -> dict[str, JsonValue]:
            response = await fixture.request(request)
            if isinstance(request, CatalogInstallRequest):
                fixture.releases[2] = fixture.releases[4]
            return response

        with pytest.raises(ValueError, match="changed since the listing was bound"):
            await TerminalInstaller(
                swapping, lambda _q: "y", lambda _q: "", fixture.output.append
            ).run_from_catalog("example.plugin", plugin_id=identifier)
        fixture.releases[2] = second
        assert sum(isinstance(r, InstallSubmission) for r in fixture.requests) == 1
        # A staged release whose activation was declined leaves no selection
        # but is still the installation's bundle and high-water mark.
        fixture.list_releases(
            {
                1: "https://releases.example.test/1/",
                2: "https://releases.example.test/2/",
                3: "https://releases.example.test/3/",
            },
            revision=13,
        )
        staged_only = await fixture.terminal(iter(("y", "y", "n"))).run_from_catalog(
            "example.plugin"
        )
        assert fixture.manager.controllers[staged_only].selector.current() is None
        held = fixture.manager.downloads[staged_only]
        retained = held.current()
        assert retained is not None and retained.state == "staged"
        bound = held.source().revision
        with pytest.raises(ValueError, match="manager request incomplete"):
            await fixture.terminal(iter(("y",))).run_from_catalog(
                "example.other", plugin_id=staged_only
            )
        with pytest.raises(ValueError, match="manager request incomplete"):
            await fixture.terminal(iter(("y",))).run_from_catalog(
                "example.plugin", sequence=1, plugin_id=staged_only
            )
        assert held.source().revision == bound
        # A publisher the installation trusts under another key is never
        # swapped by a binding: refused before the source is touched.
        keyed = "managed." + "d" * 32
        assert "result" in await fixture.request(
            InstallationRequest(action="register", plugin_id=keyed)
        )
        assert "result" in await fixture.request(
            SourceRegistration(
                plugin_id=keyed,
                request=SourceUpdate(
                    expected_revision=0,
                    base_url="https://releases.example.test/",
                    metadata_filename="release.json",
                    trust=RuntimeTrust(
                        revision=1,
                        expires_at=fixture.trust.expires_at,
                        publishers={
                            "fixture": Ed25519PrivateKey.generate()
                            .public_key()
                            .public_bytes_raw()
                            .hex()
                        },
                    ),
                ),
            )
        )
        with pytest.raises(ValueError, match="manager request incomplete"):
            await fixture.terminal(iter(("y",))).run_from_catalog(
                "example.plugin", plugin_id=keyed
            )
        assert fixture.manager.downloads[keyed].source().revision == 1


async def test_catalog_binding_rebases_trust_onto_the_installations_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An installation trusting another publisher gains the listed one, keeps its own."""
    async with journey(tmp_path, monkeypatch) as fixture:
        fixture.list_releases({1: "https://releases.example.test/1/"}, revision=1)
        await fixture.configure_catalog()
        identifier = "managed." + "c" * 32
        assert "result" in await fixture.request(
            InstallationRequest(action="register", plugin_id=identifier)
        )
        other = Ed25519PrivateKey.generate().public_key().public_bytes_raw().hex()
        configured = await fixture.request(
            SourceRegistration(
                plugin_id=identifier,
                request=SourceUpdate(
                    expected_revision=0,
                    base_url="https://elsewhere.example.test/",
                    metadata_filename="release.json",
                    trust=RuntimeTrust(
                        revision=2,
                        expires_at=fixture.trust.expires_at + 60,
                        publishers={"other": other},
                        revoked_artifacts=("9" * 64,),
                    ),
                ),
            )
        )
        assert "result" in configured
        await fixture.terminal(iter(("y", "y", "y"))).run_from_catalog(
            "example.plugin", plugin_id=identifier
        )
        selection = fixture.manager.controllers[identifier].selector.current()
        assert selection is not None and selection.enabled
        trust = RuntimeTrust.model_validate_json(
            read_private(
                fixture.manager.root
                / "installations"
                / identifier
                / "publisher-trust.json"
            )
        )
        assert trust.revision == 3
        assert trust.publishers == {
            "other": other,
            "fixture": fixture.trust.publishers["fixture"],
        }
        # The earlier expiry wins: a binding never lengthens a window.
        assert trust.expires_at == fixture.trust.expires_at
        assert trust.revoked_artifacts == ("9" * 64,)
        # Bound again at the same listing, the trust already authorizes the
        # publisher and carries every discovery revocation: left alone.
        await fixture.terminal(iter(("y",))).run_from_catalog(
            "example.plugin", plugin_id=identifier
        )
        installed = fixture.manager.root / "installations" / identifier
        assert (
            RuntimeTrust.model_validate_json(
                read_private(installed / "publisher-trust.json")
            ).revision
            == 3
        )
        # A wheel revoked for discovery is not named by any listing, so the
        # binding carries the revocation into the installation's trust,
        # where the release path enforces it.
        response = await fixture.request(
            CatalogRegistration(
                request=CatalogSourceUpdate(
                    expected_revision=1,
                    trust=RuntimeTrust(
                        revision=2,
                        expires_at=fixture.trust.expires_at,
                        publishers=fixture.trust.publishers,
                        revoked_artifacts=("8" * 64,),
                    ),
                )
            )
        )
        assert "result" in response
        await fixture.terminal(iter(("y",))).run_from_catalog(
            "example.plugin", plugin_id=identifier
        )
        carried = RuntimeTrust.model_validate_json(
            read_private(installed / "publisher-trust.json")
        )
        assert carried.revision == 4
        assert carried.revoked_artifacts == ("8" * 64, "9" * 64)
        # An expired record is not revived: its other publishers are dropped
        # and only its revocations carry, under the discovery expiry.
        write_private(
            installed / "publisher-trust.json",
            RuntimeTrust(
                revision=4,
                expires_at=int(time.time()) - 1,
                publishers={"other": other, "fixture": carried.publishers["fixture"]},
                revoked_publishers=("gone",),
                revoked_artifacts=carried.revoked_artifacts,
            )
            .model_dump_json()
            .encode(),
        )
        await fixture.terminal(iter(("y",))).run_from_catalog(
            "example.plugin", plugin_id=identifier
        )
        revived = RuntimeTrust.model_validate_json(
            read_private(installed / "publisher-trust.json")
        )
        assert revived.revision == 5
        assert revived.publishers == {"fixture": fixture.trust.publishers["fixture"]}
        assert revived.revoked_publishers == ("gone",)
        assert revived.revoked_artifacts == ("8" * 64, "9" * 64)


def test_two_artifacts_at_one_sequence_need_the_family_named() -> None:
    """The terminal never guesses between listings that both fit the host."""
    from skulk.extensions.terminal_install import TerminalInstaller

    def entry(sequence: int, platform: str | None, fits: bool) -> CatalogEntryReview:
        return CatalogEntryReview(
            bundle_id="example.plugin",
            bundle_version="1.0.0",
            title="Example plugin",
            publisher="fixture",
            sequence=sequence,
            release_digest="1" * 64,
            runtime_platform=platform,
            artifact_sha256="2" * 64,
            artifact_bytes=1,
            transfer_bytes=1,
            platforms=("darwin",),
            skulk_build_sha256="a" * 64,
            permissions=(),
            descriptors=(),
            surfaces=(),
            operations=False,
            steward_risks=(),
            expires_at=2,
            matches_host=fits,
        )

    def review(*entries: CatalogEntryReview) -> CatalogReview:
        return CatalogReview(
            publisher="fixture",
            revision=1,
            created_at=1,
            expires_at=2,
            catalog_sha256="3" * 64,
            entries=entries,
        )

    listed = TerminalInstaller._listed  # pyright: ignore[reportPrivateUsage]
    both = review(
        entry(2, None, True), entry(2, "macos-arm64", True), entry(1, None, True)
    )
    with pytest.raises(ValueError, match="name the family"):
        listed(both, "example.plugin", None, None)
    assert (
        listed(both, "example.plugin", None, "macos-arm64").runtime_platform
        == "macos-arm64"
    )
    assert listed(both, "example.plugin", 1, None).sequence == 1
    assert listed(both, "example.plugin", None, "plain").runtime_platform is None
    # An explicit family still has to fit this host: the newer listing that
    # does not is passed over for the older one that does.
    builds = review(entry(2, "macos-arm64", False), entry(1, "macos-arm64", True))
    assert listed(builds, "example.plugin", None, "macos-arm64").sequence == 1
    with pytest.raises(ValueError, match="no listed release"):
        listed(review(entry(1, "macos-arm64", True)), "example.plugin", None, "plain")
    with pytest.raises(ValueError, match="no listed release"):
        listed(review(entry(1, None, False)), "example.plugin", None, None)
    # An alias names the same family.
    assert (
        listed(
            review(entry(1, "ubuntu-24.04-x86_64", True)),
            "example.plugin",
            None,
            "linux-glibc-x86_64",
        ).sequence
        == 1
    )


def test_install_plugin_arguments_select_a_listing_only_with_a_bundle() -> None:
    """The one command keeps its bare form and gains the catalog selection flags."""
    from skulk.extensions.service_setup import (
        _install_arguments,  # pyright: ignore[reportPrivateUsage]
    )

    assert _install_arguments([]).plugin_id is None
    assert _install_arguments(["managed.abc"]).plugin_id == "managed.abc"
    options = _install_arguments(
        [
            "--from-catalog",
            "example.plugin",
            "--sequence",
            "2",
            "--platform",
            "linux-glibc-x86_64",
            "managed.abc",
        ]
    )
    assert (
        options.bundle_id,
        options.sequence,
        options.platform,
        options.plugin_id,
    ) == ("example.plugin", 2, "linux-glibc-x86_64", "managed.abc")
    with pytest.raises(ValueError, match="select a catalog listing"):
        _install_arguments(["--sequence", "2"])
    with pytest.raises(SystemExit):
        _install_arguments(["managed.abc", "managed.def"])


@pytest.mark.parametrize(
    "lost_action", ["register", "configure_source", "install", "submit"]
)
async def test_lost_accepted_response_resumes_without_repeating_effect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lost_action: str
) -> None:
    """A reconnect discovers the original operation after its reply is lost."""
    async with journey(tmp_path, monkeypatch) as fixture:
        fixture.fail_response = lost_action
        answers = fixture.fields("y", "y", "y")
        with pytest.raises(OSError, match="connection lost"):
            await fixture.terminal(answers).run()
        identifier = fixture.identifier
        downloads = fixture.manager.downloads[identifier]
        if downloads.work is not None:
            await downloads.work
        controller = fixture.manager.controllers[identifier]
        await fixture.terminal(answers).run(identifier)
        assert sum(isinstance(r, InstallSubmission) for r in fixture.requests) == 1
        assert sum(isinstance(r, SubmitRequest) for r in fixture.requests) == 1
        assert sum(isinstance(r, SourceRegistration) for r in fixture.requests) == 1
        selection = controller.selector.current()
        assert selection is not None and selection.enabled and selection.revision == 1


@pytest.mark.parametrize("decline", ["trust", "install", "activate"])
async def test_distinct_owner_decisions_do_not_imply_later_permission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, decline: str
) -> None:
    """Declining trust, download or owner execution leaves subsequent effects absent."""
    decisions = {
        "trust": ("n",),
        "install": ("y", "n"),
        "activate": ("y", "y", "n"),
    }
    async with journey(tmp_path, monkeypatch) as fixture:
        identifier = await fixture.terminal(fixture.fields(*decisions[decline])).run()
        assert fixture.manager.controllers[identifier].selector.current() is None
        assert not any(isinstance(r, SubmitRequest) for r in fixture.requests)
        if decline == "trust":
            assert not any(isinstance(r, SourceRegistration) for r in fixture.requests)
            assert not fixture.urls
        if decline != "activate":
            assert not any(isinstance(r, InstallSubmission) for r in fixture.requests)


async def test_download_failure_requires_explicit_same_operation_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failure never starts another download until the owner approves recovery."""
    async with journey(tmp_path, monkeypatch) as fixture:
        fixture.fail_download = True
        with pytest.raises(ValueError, match="explicit recovery"):
            await fixture.terminal(fixture.fields("y", "y")).run()
        identifier = fixture.identifier
        downloads = fixture.manager.downloads[identifier]
        original = downloads.current()
        assert original is not None and original.state == "recovery_required"
        before = list(fixture.urls)
        await fixture.terminal(iter(["n"])).run(identifier)
        assert fixture.urls == before
        assert downloads.current() == original
        fixture.fail_download = False
        await fixture.terminal(iter(["y", "y"])).run(identifier)
        recovered = downloads.current()
        assert recovered is not None and recovered.state == "staged"
        assert recovered.request == original.request and recovered.attempt == 1
        assert sum(isinstance(r, InstallSubmission) for r in fixture.requests) == 1


async def test_invalid_source_credential_and_unknown_identifier_are_not_disclosed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validation rejects unsafe inputs without displaying credential contents."""
    async with journey(tmp_path, monkeypatch) as fixture:
        terminal = fixture.terminal(fixture.fields("y"))
        terminal = TerminalInstaller(
            terminal.request,
            terminal.prompt,
            lambda _: "secret\ninvalid",
            terminal.output,
        )
        with pytest.raises(ValueError):
            await terminal.run()
        assert not any(isinstance(r, SourceRegistration) for r in fixture.requests)
        assert "secret" not in json.dumps(fixture.output)
        with pytest.raises(ValueError):
            await fixture.terminal(iter(())).run("../../foreign")


def test_credential_prompt_refuses_echo_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unavailable secure terminal cannot silently turn a secret prompt into input."""
    from skulk.extensions.service_setup import read_hidden_credential

    read = False

    def fallback(_prompt: str) -> str:
        nonlocal read
        warnings.warn("cannot disable echo", getpass.GetPassWarning, stacklevel=2)
        read = True
        return "must-not-be-read"

    monkeypatch.setattr("skulk.extensions.service_setup.getpass.getpass", fallback)
    with pytest.raises(getpass.GetPassWarning):
        read_hidden_credential("Credential: ")
    assert not read


def test_cli_lost_response_directs_owner_to_generated_resume_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The entrypoint must not recommend a new installation after an uncertain reply."""
    from skulk.extensions import service_setup

    class Terminal(io.StringIO):
        def isatty(self) -> bool:
            return True

    connection = ServiceConnection(manager_root=str(tmp_path), profile_id="a" * 32)
    monkeypatch.setattr(
        service_setup.sys, "argv", ["skulk-plugin-service", "install-plugin"]
    )
    monkeypatch.setattr(service_setup.sys, "stdin", Terminal())

    def read_connection(_path: Path, _limit: int) -> bytes:
        return connection.model_dump_json().encode()

    monkeypatch.setattr(service_setup, "read_private", read_connection)

    async def lost(_root: Path, _request: ManagerRequest) -> dict[str, JsonValue]:
        raise OSError("sensitive connection details")

    monkeypatch.setattr(service_setup, "manager_request", lost)
    with pytest.raises(SystemExit) as stopped:
        service_setup.main()
    assert stopped.value.code == 1
    captured = capsys.readouterr()
    assert "Resume: skulk-plugin-service install-plugin managed." in captured.out
    assert "printed resume command with its installation ID" in captured.err
    assert "Rerun the same local command" not in captured.err
    assert "sensitive connection details" not in captured.err


async def test_a_protocol_refusal_is_the_one_manager_error_the_terminal_names() -> None:
    """The installer says what to do about a release outside the window, and nothing else."""
    from skulk.extensions.terminal_install import protocol_refusal

    refusal: dict[str, JsonValue] = {
        "error": "release_protocol_unsupported",
        "kind": "release",
        "offered": 2,
        "accepted": [1],
    }
    named = protocol_refusal(refusal)
    assert named is not None
    assert "accepts release protocol 1; the release offers 2" in named
    assert protocol_refusal({"error": "manager_operation_refused"}) is None
    assert (
        protocol_refusal({"error": "release_protocol_unsupported", "kind": "x"}) is None
    )
    # bool is an int to Python and not to the fixed vocabulary.
    assert protocol_refusal({**refusal, "offered": True}) is None
    assert protocol_refusal({**refusal, "accepted": [True]}) is None
    output: list[str] = []

    async def request(_: ManagerRequest) -> dict[str, JsonValue]:
        return dict(refusal)

    terminal = TerminalInstaller(request, lambda _: "", lambda _: "", output.append)
    with pytest.raises(ValueError, match="manager request incomplete"):
        await terminal.run("managed." + "a" * 32)
    assert any("the release offers 2" in line for line in output)
