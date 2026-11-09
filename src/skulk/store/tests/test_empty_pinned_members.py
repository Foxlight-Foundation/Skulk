# pyright: reportPrivateUsage=false
"""Empty immutable bundle members remain files through admission and transfer."""

import base64
import hashlib
import json
from pathlib import Path
from typing import Literal

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from skulk.download import download_utils
from skulk.shared.models.model_cards import (
    ArtifactBundleConfig,
    ArtifactBundleFile,
    ModelCard,
    ModelTask,
)
from skulk.shared.types.common import ModelId
from skulk.shared.types.memory import Memory
from skulk.shared.types.worker.downloads import FileListEntry
from skulk.store.installed_cards import read_installed_card, verify_installed_card
from skulk.store.model_store import ModelStore, _remaining_store_download_bytes
from skulk.store.tests.test_native_music_store import _finished

Algorithm = Literal["sha256", "git-sha1"]


@pytest.mark.parametrize(
    "mode", ["zero", "redirect_zero", "missing", "negative", "invalid"]
)
async def test_http_metadata_distinguishes_empty_objects_from_absent_sizes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    """Real HEAD headers retain a required size declaration while admitting zero."""
    digest = identity("sha256", b"").split(":", 1)[1]

    async def headers() -> dict[str, str]:
        return {}

    async def respond(request: web.Request) -> web.StreamResponse:
        if mode == "missing":
            response = web.StreamResponse(headers={"ETag": digest})
            await response.prepare(request)
            await response.write_eof()
            return response
        if mode == "redirect_zero":
            return web.Response(
                status=307,
                headers={
                    "Location": "/unused",
                    "x-linked-size": "0",
                    "x-linked-etag": digest,
                },
            )
        extra = (
            {"x-linked-size": "-1" if mode == "negative" else "invalid"}
            if mode in {"negative", "invalid"}
            else {}
        )
        return web.Response(
            body=b"", headers={"ETag": digest, "Content-Length": "0", **extra}
        )

    app = web.Application()
    app.router.add_route("*", "/test/empty/resolve/{revision}/empty.bin", respond)
    monkeypatch.setattr(download_utils, "get_download_headers", headers)
    async with TestServer(app) as server:
        endpoint = str(server.make_url("/")).rstrip("/")
        monkeypatch.setattr(download_utils, "get_hf_endpoint", lambda: endpoint)
        if mode in {"zero", "redirect_zero"}:
            assert await download_utils.file_meta(
                ModelId("test/empty"), "a" * 40, "empty.bin"
            ) == (0, digest)
        else:
            with pytest.raises(
                ValueError,
                match="No content length"
                if mode == "missing"
                else "Invalid content length",
            ):
                await download_utils.file_meta(
                    ModelId("test/empty"), "a" * 40, "empty.bin"
                )


def identity(algorithm: Algorithm, data: bytes) -> str:
    """Calculate the actual original object identity, including Git's blob header."""
    digest = (
        hashlib.sha256(data).hexdigest()
        if algorithm == "sha256"
        else hashlib.sha1(
            f"blob {len(data)}\0".encode() + data, usedforsecurity=False
        ).hexdigest()
    )
    return algorithm + ":" + digest


def card(algorithm: Algorithm) -> tuple[ModelCard, dict[str, bytes]]:
    """Select one nonempty config and one exactly pinned zero-byte member."""
    objects = {"config.json": b"{}", "empty.bin": b""}
    files = tuple(
        ArtifactBundleFile(
            path=path,
            size_bytes=len(data),
            object_id=identity(algorithm if path == "empty.bin" else "git-sha1", data),
        )
        for path, data in sorted(objects.items())
    )
    manifest = [
        {
            "relative_path": item.path,
            "size_bytes": item.size_bytes,
            "object_id": item.object_id,
        }
        for item in files
    ]
    digest = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).digest()
    return ModelCard(
        model_id=ModelId("test/empty-members"),
        source_repository=ModelId("test/empty-source"),
        source_revision="a" * 40,
        is_custom=True,
        n_layers=1,
        hidden_size=8,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
        storage_size=Memory.from_bytes(2),
        artifact_bundle=ArtifactBundleConfig(
            bundle_id="bundle_" + base64.b32encode(digest).decode().rstrip("=").lower(),
            files=files,
            download_size=2,
        ),
    ), objects


@pytest.mark.parametrize("algorithm", ["sha256", "git-sha1"])
async def test_canonical_download_creates_reuses_and_repairs_empty_pinned_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    algorithm: Algorithm,
) -> None:
    """Real HEAD/GET and canonical registration preserve empty files and their pins."""
    selected, objects = card(algorithm)
    gets: list[str] = []

    async def listing(
        repository: ModelId, revision: str, **_options: object
    ) -> list[FileListEntry]:
        assert (
            repository == selected.artifact_repository
            and revision == selected.source_revision
        )
        return [
            FileListEntry(type="file", path=path, size=len(data))
            for path, data in objects.items()
        ]

    async def headers() -> dict[str, str]:
        return {}

    async def respond(request: web.Request) -> web.Response:
        path = request.match_info["path"]
        data = objects.get(path)
        if data is None:
            return web.Response(status=404)
        if request.method == "GET":
            gets.append(path)
        algorithm_for_file = algorithm if path == "empty.bin" else "git-sha1"
        return web.Response(
            body=data,
            headers={
                "ETag": identity(algorithm_for_file, data).split(":", 1)[1],
                "Content-Length": str(len(data)),
            },
        )

    app = web.Application()
    app.router.add_route(
        "*", "/test/empty-source/resolve/{revision}/{path:.*}", respond
    )
    monkeypatch.setattr(download_utils, "fetch_file_list_with_cache", listing)
    monkeypatch.setattr(download_utils, "get_download_headers", headers)
    async with TestServer(app) as hub:
        endpoint = str(hub.make_url("/")).rstrip("/")
        monkeypatch.setattr(download_utils, "get_hf_endpoint", lambda: endpoint)
        store = ModelStore(tmp_path / "store", recover_installed_cards=False)

        async def request(owner: ModelStore) -> Path:
            status = await owner.request_download(
                str(selected.model_id),
                model_card=selected,
                source_repository=str(selected.artifact_repository),
                source_revision=selected.source_revision,
            )
            await _finished(status)
            assert status.status == "complete", status.error
            directory = owner.get_store_path(str(selected.model_id))
            assert directory is not None
            return directory

        directory = await request(store)
        empty = directory / "empty.bin"
        assert empty.is_file() and empty.read_bytes() == b""
        assert sorted(gets) == ["config.json", "empty.bin"]
        installed = read_installed_card(directory)
        assert installed is not None and verify_installed_card(
            directory, installed, verify_hashes=True
        )
        gets.clear()
        assert await request(store) == directory
        assert gets == []

        empty.write_bytes(b"wrong nonempty cached object")
        retained = tmp_path / "retained-invalid-copy"
        retained.hardlink_to(empty)
        reopened = ModelStore(tmp_path / "store")
        assert await request(reopened) == directory
        assert empty.is_file() and empty.read_bytes() == b""
        assert (
            gets == ["empty.bin"]
            and retained.read_bytes() == b"wrong nonempty cached object"
        )
        repaired = read_installed_card(directory)
        assert repaired is not None and verify_installed_card(
            directory, repaired, verify_hashes=True
        )


@pytest.mark.parametrize("algorithm", ["sha256", "git-sha1"])
@pytest.mark.parametrize("existing", [None, b"", b"wrong"])
async def test_zero_byte_capacity_and_explicit_offline_custody_keep_existence_distinct(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    algorithm: Algorithm,
    existing: bytes | None,
) -> None:
    """Zero additional bytes neither invents a missing file nor blesses bad cache."""
    target = tmp_path / "empty.bin"
    if existing is not None:
        target.write_bytes(existing)
    expected = identity(algorithm, b"")
    assert (
        _remaining_store_download_bytes(
            tmp_path,
            [FileListEntry(type="file", path="empty.bin", size=0)],
            {"empty.bin": (0, expected)},
        )
        == 0
    )

    async def refuse_network(
        _model: ModelId, _revision: str, _path: str
    ) -> tuple[int, str]:
        raise AssertionError("Explicit offline reuse must not contact the network")

    monkeypatch.setattr(download_utils, "file_meta", refuse_network)
    if existing is None:
        with pytest.raises(FileNotFoundError, match="offline mode"):
            await download_utils._download_file(
                ModelId("test/empty"),
                "a" * 40,
                "empty.bin",
                tmp_path,
                skip_internet=True,
                expected_size=0,
                expected_object_id=expected,
            )
        assert not target.exists()
    elif existing:
        with pytest.raises(ValueError, match="object mismatch"):
            await download_utils._download_file(
                ModelId("test/empty"),
                "a" * 40,
                "empty.bin",
                tmp_path,
                skip_internet=True,
                expected_size=0,
                expected_object_id=expected,
            )
        assert target.read_bytes() == existing
    else:
        assert (
            await download_utils._download_file(
                ModelId("test/empty"),
                "a" * 40,
                "empty.bin",
                tmp_path,
                skip_internet=True,
                expected_size=0,
                expected_object_id=expected,
            )
            == target
        )
        assert target.is_file() and target.read_bytes() == b""
