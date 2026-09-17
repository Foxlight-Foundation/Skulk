"""The host reads a signed catalog for discovery; it authorizes nothing."""

import hashlib
import json
import time
from pathlib import Path

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import JsonValue, SecretStr, TypeAdapter

from skulk.extensions.runtime_artifacts import (
    ProtocolUnsupportedError,
    RuntimeTrust,
    canonical_json,
)
from skulk.extensions.runtime_catalog import (
    ACCEPTED_CATALOG_PROTOCOLS,
    CatalogSourceUpdate,
    HostCatalog,
    verify_catalog,
)
from skulk.extensions.runtime_files import private_directory, read_private

PUBLISHER = "fixture"
_RECORD: TypeAdapter[dict[str, JsonValue]] = TypeAdapter(dict[str, JsonValue])


def _floors(raw: bytes) -> dict[str, JsonValue]:
    floors = _RECORD.validate_json(raw)["floors"]
    assert isinstance(floors, dict)
    return floors


def _entry(sequence: int, **overrides: object) -> dict[str, object]:
    return {
        "bundle_id": "example.plugin",
        "bundle_version": "1.0.0",
        "title": "Example plugin",
        "publisher": PUBLISHER,
        "sequence": sequence,
        "feed_url": f"https://releases.example.test/example/{sequence}/",
        "release_sha256": "1" * 64,
        "release_digest": "2" * 64,
        "artifact_sha256": "3" * 64,
        "artifact_size": 4096,
        "transfer_bytes": 4096,
        "platforms": ["darwin", "linux"],
        "skulk_build_sha256": "a" * 64,
        "permissions": ["local synthetic operation"],
        "descriptors": ["example.echo@1.0.0"],
        "surfaces": ["Example studio"],
        "operations": True,
        "steward_risks": ["observation"],
        "expires_at": int(time.time()) + 86400,
        **overrides,
    }


def _catalog(
    key: Ed25519PrivateKey, entries: list[dict[str, object]], **overrides: object
) -> bytes:
    now = int(time.time())
    catalog: dict[str, object] = {
        "protocol": 1,
        "publisher": PUBLISHER,
        "revision": 1,
        "created_at": now - 1,
        "expires_at": now + 3600,
        "entries": entries,
        **overrides,
    }
    payload = canonical_json(catalog)  # type: ignore[arg-type]
    return json.dumps(
        {"catalog": catalog, "signature": key.sign(payload).hex()}
    ).encode()


def _trust(key: Ed25519PrivateKey, **overrides: object) -> RuntimeTrust:
    return RuntimeTrust.model_validate(
        {
            "revision": 1,
            "expires_at": int(time.time()) + 3600,
            "publishers": {PUBLISHER: key.public_key().public_bytes_raw().hex()},
            **overrides,
        }
    )


def test_a_catalog_verifies_against_discovery_trust_and_names_refusals() -> None:
    key = Ed25519PrivateKey.generate()
    now = int(time.time())
    document = _catalog(key, [_entry(1), _entry(2)])
    verified = verify_catalog(document, _trust(key), now=now)
    assert verified.sha256 == hashlib.sha256(document).hexdigest()
    assert verified.entry("example.plugin", 2, None) is not None
    assert verified.entry("example.plugin", 3, None) is None
    review = verified.review(skulk_build_sha256="a" * 64, platform="macos-arm64")
    assert [e.sequence for e in review.entries] == [1, 2]
    assert all(e.matches_host for e in review.entries)
    assert "feed_url" not in review.model_dump_json()
    assert "releases.example.test" not in review.model_dump_json()
    other = verified.review(skulk_build_sha256="b" * 64, platform="linux-x86_64")
    assert not any(e.matches_host for e in other.entries)
    # A runtime-bearing entry names its artifact family; one sequence may be
    # listed once per family, and only the matching family fits this host.
    families = verify_catalog(
        _catalog(
            key,
            [
                _entry(3, runtime_platform="macos-arm64"),
                _entry(3, runtime_platform="linux-x86_64"),
            ],
        ),
        _trust(key),
        now=now,
    ).review(skulk_build_sha256="a" * 64, platform="macos-arm64")
    assert [e.matches_host for e in families.entries] == [True, False]
    listed = verify_catalog(
        _catalog(
            key,
            [
                _entry(3, runtime_platform="macos-arm64"),
                _entry(3, runtime_platform="linux-x86_64"),
            ],
        ),
        _trust(key),
        now=now,
    )
    found = listed.entry("example.plugin", 3, "linux-x86_64")
    assert found is not None and found.runtime_platform == "linux-x86_64"
    assert listed.entry("example.plugin", 3, None) is None
    # An alias of a listed family finds the same listing.
    glibc = verify_catalog(
        _catalog(key, [_entry(3, runtime_platform="linux-glibc-x86_64")]),
        _trust(key),
        now=now,
    )
    assert glibc.entry("example.plugin", 3, "ubuntu-24.04-x86_64") is not None
    with pytest.raises(ValueError, match="twice"):
        verify_catalog(
            _catalog(
                key,
                [
                    _entry(3, runtime_platform="macos-arm64"),
                    _entry(3, runtime_platform="macos-arm64"),
                ],
            ),
            _trust(key),
            now=now,
        )
    # Aliases of one family are the same listing.
    with pytest.raises(ValueError, match="twice"):
        verify_catalog(
            _catalog(
                key,
                [
                    _entry(3, runtime_platform="linux-glibc-x86_64"),
                    _entry(3, runtime_platform="ubuntu-24.04-x86_64"),
                ],
            ),
            _trust(key),
            now=now,
        )
    stranger = Ed25519PrivateKey.generate()
    with pytest.raises(ValueError, match="trust refused"):
        verify_catalog(document, _trust(stranger, publishers={"x": "0" * 64}), now=now)
    with pytest.raises(ValueError, match="signature refused"):
        verify_catalog(document, _trust(stranger), now=now)
    with pytest.raises(ValueError, match="trust refused"):
        verify_catalog(document, _trust(key, revoked_publishers=(PUBLISHER,)), now=now)
    with pytest.raises(ValueError, match="window refused"):
        verify_catalog(document, _trust(key, expires_at=now + 10**6), now=now + 7200)
    beyond = max(ACCEPTED_CATALOG_PROTOCOLS) + 1
    with pytest.raises(ProtocolUnsupportedError) as refusal:
        verify_catalog(_catalog(key, [], protocol=beyond), _trust(key), now=now)
    assert refusal.value.kind == "catalog" and refusal.value.offered == beyond
    with pytest.raises(ValueError, match="twice"):
        verify_catalog(_catalog(key, [_entry(1), _entry(1)]), _trust(key), now=now)
    with pytest.raises(ValueError, match="catalog publisher"):
        verify_catalog(
            _catalog(key, [_entry(1, publisher="someone")]), _trust(key), now=now
        )
    with pytest.raises(ValueError, match="HTTPS directory"):
        verify_catalog(
            _catalog(key, [_entry(1, feed_url="http://plain.example/x/")]),
            _trust(key),
            now=now,
        )
    with pytest.raises(ValueError, match="exceeds bound"):
        verify_catalog(b"{" + b" " * 262144 + b"}", _trust(key), now=now)
    with pytest.raises(ValueError, match="below its artifact"):
        verify_catalog(
            _catalog(key, [_entry(1, transfer_bytes=1)]), _trust(key), now=now
        )


async def test_the_host_catalog_configures_fetches_and_retains_without_disclosure(
    tmp_path: Path,
) -> None:
    key = Ed25519PrivateKey.generate()
    document = _catalog(key, [_entry(1)])
    served = [document]
    calls: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        assert request.headers["Authorization"] == "Bearer hidden-catalog-token"
        if request.url.path.endswith("/redirect.json"):
            return httpx.Response(302, headers={"Location": "https://x.test/"})
        return httpx.Response(200, content=served[0])

    private_directory(tmp_path)
    catalog = HostCatalog(tmp_path, transport=httpx.MockTransport(respond))
    assert not catalog.source_status().configured
    with pytest.raises(ValueError, match="requires directory"):
        await catalog.configure(CatalogSourceUpdate(expected_revision=0))
    status = await catalog.configure(
        CatalogSourceUpdate(
            expected_revision=0,
            base_url="https://catalog.example.test/foxlight/",
            trust=_trust(key),
            token=SecretStr("hidden-catalog-token"),
        )
    )
    assert status.configured and status.credential_ready and status.revision == 1
    assert status.credential_reference is not None
    with pytest.raises(ValueError, match="revision conflict"):
        await catalog.configure(CatalogSourceUpdate(expected_revision=0))
    verified = await catalog.fetch()
    assert calls == ["/foxlight/catalog.json"]
    retained = read_private(tmp_path / "catalog" / (verified.sha256 + ".json"))
    assert retained == document
    assert catalog.retained(verified.sha256) == document
    with pytest.raises(ValueError, match="digest required"):
        catalog.retained("../catalog-source")
    assert "hidden-catalog-token" not in json.dumps(
        verified.review(
            skulk_build_sha256="a" * 64, platform="macos-arm64"
        ).model_dump()
    )
    # The accepted revision only moves forward: another document at the
    # same revision, or an older revision, is refused; a newer one accepted.
    served[0] = _catalog(key, [_entry(1), _entry(2)], revision=1)
    with pytest.raises(ValueError, match="rollback"):
        await catalog.fetch()
    served[0] = _catalog(key, [_entry(1), _entry(2)], revision=2)
    assert (await catalog.fetch()).claims.revision == 2
    served[0] = document
    with pytest.raises(ValueError, match="rollback"):
        await catalog.fetch()
    served[0] = _catalog(key, [_entry(1), _entry(2)], revision=2)
    # A damaged credential file reads as not ready, as the fetch would find.
    reference = catalog.source().credential_reference
    assert reference is not None
    (tmp_path / "catalog-credentials" / reference).write_bytes(b"bad token\n")
    assert not catalog.source_status().credential_ready
    (tmp_path / "catalog-credentials" / reference).write_bytes(b"hidden-catalog-token")
    assert catalog.source_status().credential_ready
    # Moving the catalog without supplying its credential again is refused;
    # a redirect is never followed.
    with pytest.raises(ValueError, match="credential replacement"):
        await catalog.configure(
            CatalogSourceUpdate(
                expected_revision=1, base_url="https://elsewhere.example.test/"
            )
        )
    await catalog.configure(
        CatalogSourceUpdate(expected_revision=1, document_filename="redirect.json")
    )
    with pytest.raises(ValueError, match="download refused"):
        await catalog.fetch()
    # A newer trust revision keeps prior revocations.
    await catalog.configure(
        CatalogSourceUpdate(
            expected_revision=2,
            document_filename="catalog.json",
            trust=_trust(key, revision=2, revoked_publishers=("gone",)),
        )
    )
    await catalog.configure(
        CatalogSourceUpdate(expected_revision=3, trust=_trust(key, revision=3))
    )
    assert catalog.trust().revoked_publishers == ("gone",)
    # A restored older trust file (one that could drop a revocation) is
    # refused against the floor the host recorded; the current one reads,
    # and a damaged floor fails the read closed.
    current_trust = (tmp_path / "catalog-trust.json").read_bytes()
    (tmp_path / "catalog-trust.json").write_bytes(
        _trust(key, revision=2).model_dump_json().encode()
    )
    with pytest.raises(ValueError, match="trust rollback"):
        catalog.trust()
    (tmp_path / "catalog-trust.json").write_bytes(current_trust)
    assert catalog.trust().revision == 3
    floor_record = (tmp_path / "catalog-trust-floor.json").read_bytes()
    (tmp_path / "catalog-trust-floor.json").write_bytes(
        b'{"revision": 0, "sha256": "short"}'
    )
    with pytest.raises(ValueError, match="local maintenance"):
        catalog.trust()
    (tmp_path / "catalog-trust-floor.json").write_bytes(floor_record)
    # Moving to another catalog (with its credential supplied again) starts
    # a new revision history: revision 1 there is not a rollback.
    await catalog.configure(
        CatalogSourceUpdate(
            expected_revision=4,
            base_url="https://catalog.example.test/other/",
            token=SecretStr("hidden-catalog-token"),
        )
    )
    served[0] = document
    assert (await catalog.fetch()).claims.revision == 1
    assert (tmp_path / "catalog-revision.json").exists()
    # Two trusted publishers at one address keep separate floors: serving
    # the other publisher does not erase the first one's history.
    other_key = Ed25519PrivateKey.generate()
    both = RuntimeTrust.model_validate(
        {
            "revision": 5,
            "expires_at": int(time.time()) + 3600,
            "publishers": {
                PUBLISHER: key.public_key().public_bytes_raw().hex(),
                "second": other_key.public_key().public_bytes_raw().hex(),
            },
            "revoked_publishers": ("gone",),
        }
    )
    await catalog.configure(CatalogSourceUpdate(expected_revision=5, trust=both))
    served[0] = _catalog(key, [_entry(1), _entry(2)], revision=10)
    assert (await catalog.fetch()).claims.revision == 10
    served[0] = _catalog(
        other_key, [_entry(1, publisher="second")], publisher="second", revision=1
    )
    assert (await catalog.fetch()).claims.publisher == "second"
    served[0] = document
    with pytest.raises(ValueError, match="rollback"):
        await catalog.fetch()
    # A damaged revision record fails closed rather than reading as first use,
    # whether the file or one publisher's floor is the damaged part.
    record = (tmp_path / "catalog-revision.json").read_bytes()
    (tmp_path / "catalog-revision.json").write_bytes(b"{not json")
    served[0] = _catalog(key, [_entry(1), _entry(2)], revision=11)
    with pytest.raises(ValueError, match="local maintenance"):
        await catalog.fetch()
    damaged = _RECORD.validate_json(record)
    _floors(record)[PUBLISHER] = {"revision": 0, "sha256": "short"}
    damaged_floors = damaged["floors"]
    assert isinstance(damaged_floors, dict)
    damaged_floors[PUBLISHER] = {"revision": 0, "sha256": "short"}
    (tmp_path / "catalog-revision.json").write_bytes(_RECORD.dump_json(damaged))
    with pytest.raises(ValueError, match="local maintenance"):
        await catalog.fetch()
    moved = _RECORD.validate_json(record)
    moved["base_url"] = "https://elsewhere.example.test/"
    (tmp_path / "catalog-revision.json").write_bytes(_RECORD.dump_json(moved))
    with pytest.raises(ValueError, match="local maintenance"):
        await catalog.fetch()
    (tmp_path / "catalog-revision.json").write_bytes(record)
    assert (await catalog.fetch()).claims.revision == 11
    # Retention keeps the newest documents and every accepted floor; a
    # regularly updated catalog never runs out of room.
    for revision in range(12, 24):
        served[0] = _catalog(key, [_entry(1), _entry(2)], revision=revision)
        await catalog.fetch()
    floors = _floors((tmp_path / "catalog-revision.json").read_bytes())
    retained_documents = list((tmp_path / "catalog").iterdir())
    # The newest eight plus one accepted document per publisher, at most.
    assert len(retained_documents) <= 8 + len(floors)
    for entry in floors.values():
        assert isinstance(entry, dict)
        sha256 = entry["sha256"]
        assert isinstance(sha256, str)
        assert (tmp_path / "catalog" / (sha256 + ".json")).exists()
    await catalog.close()
    with pytest.raises(ValueError, match="closed"):
        await catalog.fetch()


def test_a_revoked_release_or_artifact_is_not_offered() -> None:
    key = Ed25519PrivateKey.generate()
    now = int(time.time())
    document = _catalog(
        key,
        [
            _entry(1, release_digest="4" * 64),
            _entry(2, artifact_sha256="5" * 64),
            _entry(3),
            _entry(4, release_sha256="6" * 64),
        ],
    )
    verified = verify_catalog(
        document,
        _trust(key, revoked_artifacts=("4" * 64, "5" * 64, "6" * 64)),
        now=now,
    )
    assert [e.sequence for e in verified.entries] == [3]
    assert verified.entry("example.plugin", 1, None) is None
    review = verified.review(skulk_build_sha256="a" * 64, platform="macos-arm64")
    assert [e.sequence for e in review.entries] == [3]
