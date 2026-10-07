"""The host reads a signed catalog for discovery; it authorizes nothing."""

import hashlib
import json
import time
from collections.abc import Awaitable, Callable
from functools import partial
from pathlib import Path
from typing import get_args

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
    CatalogRefusal,
    CatalogRefusedError,
    CatalogSourceUpdate,
    HostCatalog,
    catalog_refusal,
    catalog_refusal_payload,
    catalog_refusal_sentence,
    verify_catalog,
)
from skulk.extensions.runtime_files import (
    private_directory,
    read_private,
    write_private,
)

PUBLISHER = "fixture"
_RECORD: TypeAdapter[dict[str, JsonValue]] = TypeAdapter(dict[str, JsonValue])


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
    review = verified.review(skulk_version="2.0.1", skulk_build_sha256="a" * 64, platform="macos-arm64")
    assert [e.sequence for e in review.entries] == [1, 2]
    assert all(e.matches_host for e in review.entries)
    assert "feed_url" not in review.model_dump_json()
    assert "releases.example.test" not in review.model_dump_json()
    # A protocol 1 listing was published for one exact Skulk build and keeps
    # that rule: another build does not fit it.
    other = verified.review(
        skulk_version="2.0.1", skulk_build_sha256="b" * 64, platform="linux-x86_64"
    )
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
    ).review(skulk_version="2.0.1", skulk_build_sha256="a" * 64, platform="macos-arm64")
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
    with pytest.raises(ValueError, match="unconfigured"):
        await catalog.fetch()
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
        catalog.retained("../catalog-state")
    assert "hidden-catalog-token" not in json.dumps(
        verified.review(skulk_version="2.0.1", skulk_build_sha256="a" * 64, platform="macos-arm64").model_dump()
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
    # A newer trust revision keeps prior revocations; the trust floor refuses
    # a lower revision even through a reconfiguration.
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
    with pytest.raises(ValueError, match="revision conflict"):
        await catalog.configure(
            CatalogSourceUpdate(expected_revision=4, trust=_trust(key, revision=2))
        )
    # Another address has its own history: revision 1 there is accepted, and
    # coming back to the first address meets its floor again.
    await catalog.configure(
        CatalogSourceUpdate(
            expected_revision=4,
            base_url="https://catalog.example.test/other/",
            token=SecretStr("hidden-catalog-token"),
        )
    )
    served[0] = document
    assert (await catalog.fetch()).claims.revision == 1
    await catalog.configure(
        CatalogSourceUpdate(
            expected_revision=5,
            base_url="https://catalog.example.test/foxlight/",
            token=SecretStr("hidden-catalog-token"),
        )
    )
    with pytest.raises(ValueError, match="rollback"):
        await catalog.fetch()
    served[0] = _catalog(key, [_entry(1), _entry(2)], revision=3)
    assert (await catalog.fetch()).claims.revision == 3
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
    await catalog.configure(CatalogSourceUpdate(expected_revision=6, trust=both))
    served[0] = _catalog(key, [_entry(1), _entry(2)], revision=10)
    assert (await catalog.fetch()).claims.revision == 10
    served[0] = _catalog(
        other_key, [_entry(1, publisher="second")], publisher="second", revision=1
    )
    assert (await catalog.fetch()).claims.publisher == "second"
    served[0] = document
    with pytest.raises(ValueError, match="rollback"):
        await catalog.fetch()
    # A damaged state document fails every read closed rather than reading
    # as first use; a tampered retained document is refused by its digest.
    state = (tmp_path / "catalog-state.json").read_bytes()
    (tmp_path / "catalog-state.json").write_bytes(b"{not json")
    with pytest.raises(ValueError, match="local maintenance"):
        await catalog.fetch()
    with pytest.raises(ValueError, match="local maintenance"):
        catalog.trust()
    (tmp_path / "catalog-state.json").unlink()
    with pytest.raises(ValueError, match="local maintenance"):
        await catalog.fetch()
    # A document whose trust sits below its own floor is a rolled-back view.
    rolled = _RECORD.validate_json(state)
    trust_view = rolled["trust"]
    assert isinstance(trust_view, dict)
    trust_view["revision"] = 1
    write_private(tmp_path / "catalog-state.json", _RECORD.dump_json(rolled))
    with pytest.raises(ValueError, match="trust rollback"):
        await catalog.fetch()
    write_private(tmp_path / "catalog-state.json", state)
    served[0] = _catalog(key, [_entry(1), _entry(2)], revision=11)
    listed = await catalog.fetch()
    (tmp_path / "catalog" / (listed.sha256 + ".json")).write_bytes(b"{}")
    with pytest.raises(ValueError, match="differs from its digest"):
        catalog.retained(listed.sha256)
    # Retention keeps the newest documents and every accepted floor; a
    # regularly updated catalog never runs out of room.
    for revision in range(12, 24):
        served[0] = _catalog(key, [_entry(1), _entry(2)], revision=revision)
        await catalog.fetch()
    floors = _RECORD.validate_json((tmp_path / "catalog-state.json").read_bytes())[
        "floors"
    ]
    assert isinstance(floors, dict)
    retained_documents = list((tmp_path / "catalog").iterdir())
    assert len(retained_documents) <= 8 + len(floors)
    catalog_floors = {
        key: floor for key, floor in floors.items() if not key.startswith("trust\n")
    }
    # Each address this host read keeps its discovery trust floor in the map.
    assert len(catalog_floors) < len(floors)
    for floor in catalog_floors.values():
        assert isinstance(floor, dict)
        sha256 = floor["sha256"]
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
    review = verified.review(skulk_version="2.0.1", skulk_build_sha256="a" * 64, platform="macos-arm64")
    assert [e.sequence for e in review.entries] == [3]


def test_the_floor_map_refuses_a_new_history_at_the_bound() -> None:
    from skulk.extensions.runtime_catalog import AcceptedFloor, bounded_floors

    prefix = "https://catalog.example.test/foxlight/\ncatalog.json\n"
    floors = {
        f"{prefix}publisher-{i}": AcceptedFloor(revision=i + 1, sha256="b" * 64)
        for i in range(32)
    }
    # Existing histories keep recording at the bound; a new one is refused.
    assert bounded_floors(floors, f"{prefix}publisher-3", prefix) == floors
    floors[f"{prefix}publisher-32"] = AcceptedFloor(revision=1, sha256="c" * 64)
    with pytest.raises(ValueError, match="local maintenance"):
        bounded_floors(floors, f"{prefix}publisher-32", prefix)


async def _refused(call: Awaitable[object]) -> CatalogRefusedError:
    with pytest.raises(CatalogRefusedError) as refused:
        await call
    return refused.value


def _refused_now(call: Callable[[], object]) -> CatalogRefusedError:
    with pytest.raises(CatalogRefusedError) as refused:
        call()
    return refused.value


def test_catalog_refusals_are_named_from_a_fixed_vocabulary() -> None:
    """Each code has one plain sentence; the wire form is a code and a status only."""
    codes: tuple[CatalogRefusal, ...] = get_args(CatalogRefusal)
    for code in codes:
        sentence = catalog_refusal_sentence(code)
        assert sentence.endswith(".") and "\u2014" not in sentence
        wire = _RECORD.validate_json(
            catalog_refusal_payload(CatalogRefusedError(code, "local detail"))
        )
        assert wire == {"error": "catalog_refused", "code": code}
        rebuilt = catalog_refusal(wire)
        assert rebuilt is not None
        assert (rebuilt.code, rebuilt.status) == (code, None)
    missing = CatalogRefusedError(
        "catalog_download_refused", "catalog download refused", status=404
    )
    wire = _RECORD.validate_json(catalog_refusal_payload(missing))
    assert wire == {
        "error": "catalog_refused",
        "code": "catalog_download_refused",
        "status": 404,
    }
    rebuilt = catalog_refusal(wire)
    assert rebuilt is not None and rebuilt.status == 404
    assert "HTTP 404" in catalog_refusal_sentence(rebuilt.code, rebuilt.status)
    # Nothing outside the vocabulary is rebuilt.
    assert catalog_refusal({"error": "manager_operation_refused"}) is None
    assert catalog_refusal({"error": "catalog_refused", "code": "elsewhere"}) is None
    assert catalog_refusal({**wire, "status": True}) is None
    assert catalog_refusal({**wire, "status": "404"}) is None
    assert catalog_refusal({**wire, "status": 99}) is None


async def test_the_host_catalog_names_why_it_refused(tmp_path: Path) -> None:
    """Every refusal an operator can act on carries its code."""
    key = Ed25519PrivateKey.generate()
    mode = ["catalog"]
    served = [_catalog(key, [_entry(1)])]

    def respond(request: httpx.Request) -> httpx.Response:
        if mode[0] == "unreachable":
            raise httpx.ConnectError("name resolution failed", request=request)
        if mode[0] == "missing":
            return httpx.Response(404)
        if mode[0] == "compressed":
            # A stream, not content: content would be decoded on construction.
            return httpx.Response(
                200,
                headers={"Content-Encoding": "gzip"},
                stream=httpx.ByteStream(served[0]),
            )
        if mode[0] == "page":
            return httpx.Response(200, content=b"<!doctype html><title>Sign in")
        return httpx.Response(200, content=served[0])

    private_directory(tmp_path)
    catalog = HostCatalog(tmp_path, transport=httpx.MockTransport(respond))
    now = int(time.time())
    assert (await _refused(catalog.fetch())).code == "catalog_unconfigured"
    assert (
        _refused_now(lambda: catalog.accepted("a" * 64, now=now)).code
        == "catalog_unconfigured"
    )
    incomplete = CatalogSourceUpdate(expected_revision=0)
    assert (
        await _refused(catalog.configure(incomplete))
    ).code == "catalog_setup_incomplete"
    await catalog.configure(
        CatalogSourceUpdate(
            expected_revision=0,
            base_url="https://catalog.example.test/foxlight/",
            trust=_trust(key),
            token=SecretStr("hidden-catalog-token"),
        )
    )
    assert (
        await _refused(catalog.configure(incomplete))
    ).code == "catalog_source_conflict"
    expired = CatalogSourceUpdate(
        expected_revision=1, trust=_trust(key, revision=2, expires_at=now - 1)
    )
    assert (
        await _refused(catalog.configure(expired))
    ).code == "catalog_trust_update_refused"
    moved = CatalogSourceUpdate(
        expected_revision=1, base_url="https://elsewhere.example.test/"
    )
    assert (
        await _refused(catalog.configure(moved))
    ).code == "catalog_credential_required"
    # What the server did, as the operator can act on it.
    for current, code in (
        ("unreachable", "catalog_unreachable"),
        ("compressed", "catalog_download_refused"),
        ("page", "catalog_invalid"),
    ):
        mode[0] = current
        refused = await _refused(catalog.fetch())
        assert (refused.code, refused.status) == (code, None)
    mode[0] = "missing"
    refused = await _refused(catalog.fetch())
    assert (refused.code, refused.status) == ("catalog_download_refused", 404)
    # What the document is: signed by another key, by an untrusted publisher,
    # outside its validity period, or signed yet not a catalog this host reads.
    mode[0] = "catalog"
    for document, code in (
        (
            _catalog(Ed25519PrivateKey.generate(), [_entry(1)]),
            "catalog_signature_refused",
        ),
        (_catalog(key, [_entry(1)], publisher="stranger"), "catalog_trust_refused"),
        (
            _catalog(key, [_entry(1)], created_at=now - 7200, expires_at=now - 3600),
            "catalog_window_refused",
        ),
        (_catalog(key, [_entry(1), _entry(1)]), "catalog_invalid"),
    ):
        served[0] = document
        assert (await _refused(catalog.fetch())).code == code
    # A listing a later read superseded, or one this host never held, is
    # refused with the same remedy: read the catalog again.
    served[0] = _catalog(key, [_entry(1)])
    first = await catalog.fetch()
    served[0] = _catalog(key, [_entry(1), _entry(2)], revision=2)
    await catalog.fetch()
    now = int(time.time())
    for digest in (first.sha256, "f" * 64):
        refused = _refused_now(partial(catalog.accepted, digest, now=now))
        assert refused.code == "catalog_superseded"
    served[0] = _catalog(key, [_entry(1)])
    assert (await _refused(catalog.fetch())).code == "catalog_rollback_refused"
    async with catalog.guard:
        assert (await _refused(catalog.fetch())).code == "catalog_busy"
    reference = catalog.source().credential_reference
    assert reference is not None
    (tmp_path / "catalog-credentials" / reference).unlink()
    assert (await _refused(catalog.fetch())).code == "catalog_credential_unavailable"


def test_a_protocol_2_listing_fits_by_skulk_version_whatever_the_build() -> None:
    """A Skulk update keeps a listing installable; a version outside its range does not."""
    key = Ed25519PrivateKey.generate()
    now = int(time.time())
    verified = verify_catalog(
        _catalog(
            key,
            [
                _entry(1, skulk_requires=">=2.0.0,<3"),
                _entry(2, skulk_requires=">=2.1.0,<3"),
            ],
            protocol=2,
        ),
        _trust(key),
        now=now,
    )
    updated = verified.review(
        skulk_version="2.0.1", skulk_build_sha256="f" * 64, platform="macos-arm64"
    )
    # The newer release needs a later Skulk: it is listed, but not as fitting,
    # so a consumer picking the newest fitting release picks the older one.
    assert [(e.sequence, e.matches_host) for e in updated.entries] == [(1, True), (2, False)]
    assert updated.entries[0].skulk_requires == ">=2.0.0,<3"
    later = verified.review(
        skulk_version="2.4.0", skulk_build_sha256="e" * 64, platform="linux-x86_64"
    )
    assert all(e.matches_host for e in later.entries)
    major = verified.review(
        skulk_version="3.0.0", skulk_build_sha256="a" * 64, platform="macos-arm64"
    )
    assert not any(e.matches_host for e in major.entries)


@pytest.mark.parametrize(
    ("protocol", "entry_overrides"),
    [
        (2, {}),
        (1, {"skulk_requires": ">=2.0.0,<3"}),
        (2, {"skulk_requires": "two point oh"}),
    ],
)
def test_a_listing_states_its_skulk_range_exactly_when_its_protocol_does(
    protocol: int, entry_overrides: dict[str, object]
) -> None:
    """Protocol 2 entries carry a valid range and protocol 1 entries carry none."""
    key = Ed25519PrivateKey.generate()
    document = _catalog(key, [_entry(1, **entry_overrides)], protocol=protocol)
    with pytest.raises(CatalogRefusedError) as refused:
        verify_catalog(document, _trust(key), now=int(time.time()))
    assert refused.value.code == "catalog_invalid"

