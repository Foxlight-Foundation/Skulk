"""The signed package index: what verifies, what is kept, and what a host is offered."""

import dataclasses
import json
from pathlib import Path

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import JsonValue

from skulk.extensions.runtime_artifacts import RuntimeTrust, canonical_json
from skulk.provisioning.engine_index import (
    ENGINE_INDEX_SIGNATURE_DOMAIN,
    MAX_ENGINE_INDEX_BYTES,
    EngineIndexClient,
    EngineIndexRefusedError,
    installable_engine_packages,
    verify_engine_index,
)
from skulk.provisioning.engine_packages import ENGINE_SIGNATURE_DOMAIN, EngineHost
from skulk.provisioning.tests.engine_package_fixtures import (
    NOW,
    PUBLISHER,
    Claims,
    public_hex,
)

HOST = EngineHost(
    platform="linux-glibc-x86_64",
    lanes=frozenset({"rocm", "vulkan", "cpu"}),
    compute_targets=frozenset({"gfx1151"}),
    skulk_version="2.0.3",
    control_protocols=(1, 1),
    python_version="3.13.1",
)


def _entry(**changes: JsonValue) -> Claims:
    """One valid entry: a TensorFold build for Strix Halo on the stable channel."""
    entry: Claims = {
        "engine_id": "foxlight.tensorfold",
        "engine_version": "1.3.2",
        "generation": 1,
        "channel": "stable",
        "platform": "linux-glibc-x86_64",
        "lane": "rocm",
        "compute_targets": ["gfx1151"],
        "skulk_versions": ">=2.0.3,<3",
        "control_protocol": {"minimum": 1, "maximum": 2},
        "adapter_python_requires": ">=3.12",
        "manifest_digest": "d" * 64,
        "directory": "foxlight.tensorfold/rocm/" + "d" * 64,
        "download_bytes": 924_000_000,
        "conservative_overhead_bytes": 2 * 1024**3,
    }
    entry.update(changes)
    return entry


def _claims(*entries: Claims, **changes: JsonValue) -> Claims:
    claims: Claims = {
        "schema_version": 1,
        "publisher": PUBLISHER,
        "revision": 1,
        "issued_at": NOW - 60,
        "expires_at": NOW + 86_400,
        "entries": list(entries),
    }
    claims.update(changes)
    return claims


def _signed(
    key: Ed25519PrivateKey,
    claims: Claims,
    *,
    domain: bytes = ENGINE_INDEX_SIGNATURE_DOMAIN,
) -> bytes:
    signature = key.sign(domain + canonical_json(claims)).hex()
    return json.dumps({"index": claims, "signature": signature}).encode()


def _trust(key: Ed25519PrivateKey, **changes: JsonValue) -> RuntimeTrust:
    document: Claims = {
        "revision": 1,
        "expires_at": NOW + 86_400,
        "publishers": {PUBLISHER: public_hex(key)},
    }
    document.update(changes)
    return RuntimeTrust.model_validate_json(json.dumps(document))


def _refusal(document: bytes, trust: RuntimeTrust, now: int = NOW) -> str:
    with pytest.raises(EngineIndexRefusedError) as caught:
        verify_engine_index(document, trust, now=now)
    return caught.value.code


@pytest.fixture
def key() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.generate()


def test_a_signed_index_verifies(key: Ed25519PrivateKey) -> None:
    index = verify_engine_index(_signed(key, _claims(_entry())), _trust(key), now=NOW)

    assert index.claims.revision == 1
    assert [entry.backend_tag for entry in index.claims.entries] == ["tensorfold-rocm"]


def test_a_manifest_signature_is_not_an_index_signature(key: Ed25519PrivateKey) -> None:
    """The domain prefix keeps the two kinds of signed document apart."""
    document = _signed(key, _claims(_entry()), domain=ENGINE_SIGNATURE_DOMAIN)

    assert _refusal(document, _trust(key)) == "signature_refused"


def test_a_changed_entry_fails_the_signature(key: Ed25519PrivateKey) -> None:
    signature = key.sign(
        ENGINE_INDEX_SIGNATURE_DOMAIN + canonical_json(_claims(_entry()))
    ).hex()
    altered = _claims(_entry(manifest_digest="e" * 64))
    document = json.dumps({"index": altered, "signature": signature}).encode()

    assert _refusal(document, _trust(key)) == "signature_refused"


def test_untrusted_revoked_and_foreign_publishers_are_refused(
    key: Ed25519PrivateKey,
) -> None:
    document = _signed(key, _claims(_entry()))
    other = Ed25519PrivateKey.generate()

    assert _refusal(document, _trust(key, publishers={"someone": public_hex(key)})) == (
        "trust_refused"
    )
    assert _refusal(document, _trust(key, revoked_publishers=[PUBLISHER])) == (
        "trust_refused"
    )
    assert _refusal(document, _trust(other)) == "signature_refused"


def test_expired_trust_and_an_index_outside_its_window_are_refused(
    key: Ed25519PrivateKey,
) -> None:
    document = _signed(key, _claims(_entry()))

    assert _refusal(document, _trust(key, expires_at=NOW)) == "trust_refused"
    assert _refusal(document, _trust(key), now=NOW - 61) == "window_refused"
    assert (
        _refusal(document, _trust(key, expires_at=NOW + 200_000), now=NOW + 86_400)
        == "window_refused"
    )


@pytest.mark.parametrize(
    "claims",
    [
        _claims(_entry(directory="../elsewhere")),
        _claims(_entry(directory="/absolute")),
        _claims(_entry(lane="quantum")),
        _claims(_entry(), _entry()),
        _claims(_entry(), unexpected=True),
        _claims(_entry(), schema_version=2),
    ],
)
def test_signed_claims_outside_the_schema_are_refused(
    key: Ed25519PrivateKey, claims: Claims
) -> None:
    assert _refusal(_signed(key, claims), _trust(key)) == "index_invalid"


def test_oversized_and_malformed_documents_are_refused(key: Ed25519PrivateKey) -> None:
    assert _refusal(b" " * (MAX_ENGINE_INDEX_BYTES + 1), _trust(key)) == "index_invalid"
    assert _refusal(b"not json", _trust(key)) == "index_invalid"


def test_a_revoked_build_is_left_out_and_the_rest_stays(key: Ed25519PrivateKey) -> None:
    kept = _entry(
        manifest_digest="e" * 64, lane="vulkan", compute_targets=[], directory="x/y"
    )
    document = _signed(key, _claims(_entry(), kept))

    index = verify_engine_index(
        document, _trust(key, revoked_artifacts=["d" * 64]), now=NOW
    )

    assert [entry.manifest_digest for entry in index.claims.entries] == ["e" * 64]


def _offers(
    key: Ed25519PrivateKey,
    *entries: Claims,
    host: EngineHost = HOST,
    follow_candidates: bool = False,
) -> dict[str, str]:
    index = verify_engine_index(_signed(key, _claims(*entries)), _trust(key), now=NOW)
    chosen = (
        installable_engine_packages(
            index, host, channels=frozenset({"stable", "candidate"})
        )
        if follow_candidates
        else installable_engine_packages(index, host)
    )
    return {tag: entry.manifest_digest for tag, entry in chosen.items()}


def test_a_fitting_build_is_offered_under_its_backend_tag(
    key: Ed25519PrivateKey,
) -> None:
    assert _offers(key, _entry()) == {"tensorfold-rocm": "d" * 64}


@pytest.mark.parametrize(
    "host",
    [
        dataclasses.replace(HOST, platform="macos-arm64"),
        dataclasses.replace(HOST, lanes=frozenset({"vulkan", "cpu"})),
        dataclasses.replace(HOST, compute_targets=frozenset({"gfx1150"})),
        dataclasses.replace(HOST, skulk_version="3.0.0"),
        dataclasses.replace(HOST, control_protocols=(3, 4)),
        dataclasses.replace(HOST, python_version="3.11.9"),
    ],
)
def test_a_build_the_host_cannot_run_is_not_offered(
    key: Ed25519PrivateKey, host: EngineHost
) -> None:
    assert _offers(key, _entry(), host=host) == {}


def test_a_host_with_an_undetected_device_is_still_offered_the_build(
    key: Ed25519PrivateKey,
) -> None:
    """Failed detection is not evidence of a mismatch."""
    host = dataclasses.replace(HOST, compute_targets=frozenset())

    assert _offers(key, _entry(), host=host) == {"tensorfold-rocm": "d" * 64}


def test_a_candidate_build_is_offered_only_to_a_node_that_follows_candidates(
    key: Ed25519PrivateKey,
) -> None:
    candidate = _entry(channel="candidate", generation=2, manifest_digest="e" * 64)

    assert _offers(key, _entry(), candidate) == {"tensorfold-rocm": "d" * 64}
    assert _offers(key, _entry(), candidate, follow_candidates=True) == {
        "tensorfold-rocm": "e" * 64
    }
    assert _offers(key, candidate) == {}


def test_the_highest_generation_wins_and_an_exact_device_build_breaks_a_tie(
    key: Ed25519PrivateKey,
) -> None:
    newer = _entry(generation=3, manifest_digest="e" * 64)
    generic = _entry(generation=3, manifest_digest="f" * 64, compute_targets=[])

    assert _offers(key, _entry(), generic, newer) == {"tensorfold-rocm": "e" * 64}
    assert _offers(key, _entry(), generic) == {"tensorfold-rocm": "f" * 64}


def test_a_tag_two_engine_ids_claim_is_offered_by_neither(
    key: Ed25519PrivateKey,
) -> None:
    rival = _entry(engine_id="someone.tensorfold", manifest_digest="e" * 64)

    assert _offers(key, _entry(), rival) == {}


def test_a_build_named_after_a_built_in_engine_is_never_offered(
    key: Ed25519PrivateKey,
) -> None:
    assert _offers(key, _entry(engine_id="foxlight.llama_server")) == {}


def _client(
    tmp_path: Path, served: list[bytes | int | Exception]
) -> EngineIndexClient:
    """A client whose store answers with the next item each time it is asked."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://store.invalid/engines/index-v1.json"
        answer = served.pop(0)
        if isinstance(answer, Exception):
            raise answer
        if isinstance(answer, int):
            return httpx.Response(answer)
        return httpx.Response(200, content=answer)

    return EngineIndexClient(
        tmp_path / "engine-index",
        base_url="https://store.invalid/engines",
        transport=httpx.MockTransport(handler),
    )


def test_a_fetched_index_is_kept_and_read_back_without_the_network(
    key: Ed25519PrivateKey, tmp_path: Path
) -> None:
    client = _client(tmp_path, [_signed(key, _claims(_entry()))])

    fetched = client.load(_trust(key), offline=False, now=NOW)
    kept = client.load(_trust(key), offline=True, now=NOW)

    assert fetched.digest == kept.digest
    assert client.kept(_trust(key), now=NOW).claims.revision == 1


@pytest.mark.parametrize(
    "answer", [503, httpx.ConnectError("no route"), b"not an index"]
)
def test_a_failed_fetch_falls_back_to_the_kept_index(
    key: Ed25519PrivateKey, tmp_path: Path, answer: bytes | int | Exception
) -> None:
    client = _client(tmp_path, [_signed(key, _claims(_entry())), answer])
    client.load(_trust(key), offline=False, now=NOW)

    assert client.load(_trust(key), offline=False, now=NOW).claims.revision == 1


def test_an_older_index_or_a_different_one_at_the_same_revision_is_not_taken(
    key: Ed25519PrivateKey, tmp_path: Path
) -> None:
    current = _signed(key, _claims(_entry(), revision=5))
    older = _signed(key, _claims(revision=4))
    altered = _signed(key, _claims(revision=5))
    client = _client(tmp_path, [current, older, altered])
    client.load(_trust(key), offline=False, now=NOW)

    for _ in range(2):
        index = client.load(_trust(key), offline=False, now=NOW)
        assert index.claims.revision == 5
        assert len(index.claims.entries) == 1


def test_a_newer_index_replaces_the_kept_one(
    key: Ed25519PrivateKey, tmp_path: Path
) -> None:
    client = _client(
        tmp_path,
        [_signed(key, _claims(_entry())), _signed(key, _claims(revision=2))],
    )
    client.load(_trust(key), offline=False, now=NOW)
    client.load(_trust(key), offline=False, now=NOW)

    assert client.kept(_trust(key), now=NOW).claims.revision == 2


def test_with_nothing_kept_and_no_store_there_is_no_index(
    key: Ed25519PrivateKey, tmp_path: Path
) -> None:
    client = _client(tmp_path, [503])

    with pytest.raises(EngineIndexRefusedError) as caught:
        client.load(_trust(key), offline=False, now=NOW)
    assert caught.value.code == "index_unavailable"
    with pytest.raises(EngineIndexRefusedError):
        client.kept(_trust(key), now=NOW)


def test_a_kept_index_stops_being_used_when_it_expires_or_loses_its_publisher(
    key: Ed25519PrivateKey, tmp_path: Path
) -> None:
    client = _client(tmp_path, [_signed(key, _claims(_entry()))])
    client.load(_trust(key), offline=False, now=NOW)
    later = _trust(key, expires_at=NOW + 900_000)

    with pytest.raises(EngineIndexRefusedError):
        client.kept(later, now=NOW + 86_400)
    with pytest.raises(EngineIndexRefusedError):
        client.kept(_trust(key, revoked_publishers=[PUBLISHER]), now=NOW)


def test_a_damaged_kept_copy_is_not_used_and_does_not_block_renewal(
    key: Ed25519PrivateKey, tmp_path: Path
) -> None:
    client = _client(
        tmp_path,
        [_signed(key, _claims(_entry(), revision=5)), _signed(key, _claims(revision=2))],
    )
    client.load(_trust(key), offline=False, now=NOW)
    kept = client.retained_path.read_text()
    client.retained_path.write_text(kept.replace("tensorfold", "tensorfolx"))

    with pytest.raises(EngineIndexRefusedError):
        client.kept(_trust(key), now=NOW)
    assert client.load(_trust(key), offline=False, now=NOW).claims.revision == 2


def test_the_store_address_must_be_https(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        EngineIndexClient(tmp_path, base_url="http://store.invalid/engines")
