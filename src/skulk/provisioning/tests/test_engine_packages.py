"""Engine package manifests: signature, trust, validity, host fit and file integrity."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import JsonValue

from skulk.extensions.runtime_artifacts import RuntimeTrust, canonical_json
from skulk.provisioning import engine_packages
from skulk.provisioning.engine_packages import (
    EngineHost,
    EnginePackageRefusedError,
    check_engine_host,
    select_control_protocol,
    verify_engine_file,
    verify_engine_manifest,
)
from skulk.provisioning.tests.engine_package_fixtures import (
    ADAPTER_SHA256,
    ARCHIVE_SHA256,
    NOW,
    PUBLISHER,
    REQUIREMENT_SHA256,
    Claims,
)
from skulk.provisioning.tests.engine_package_fixtures import (
    engine_claims as _claims,
)
from skulk.provisioning.tests.engine_package_fixtures import (
    engine_envelope as _envelope,
)
from skulk.provisioning.tests.engine_package_fixtures import (
    engine_signature as _signature,
)
from skulk.provisioning.tests.engine_package_fixtures import (
    public_hex as _public_hex,
)
from skulk.provisioning.tests.engine_package_fixtures import (
    sign_engine_claims as _sign,
)

ClaimPath = tuple[str | int, ...]


def _with(claims: Claims, path: ClaimPath, value: JsonValue) -> Claims:
    """Return a copy of the claims with one nested value replaced or added."""
    copied = copy.deepcopy(claims)
    node: JsonValue = copied
    for step in path[:-1]:
        if isinstance(node, dict) and isinstance(step, str):
            node = node[step]
            continue
        assert isinstance(node, list) and isinstance(step, int)
        node = node[step]
    last = path[-1]
    assert isinstance(node, dict) and isinstance(last, str)
    node[last] = value
    return copied


def _trust(key: Ed25519PrivateKey, **changes: JsonValue) -> RuntimeTrust:
    document: Claims = {
        "revision": 1,
        "expires_at": NOW + 86_400,
        "publishers": {PUBLISHER: _public_hex(key)},
    }
    document.update(changes)
    return RuntimeTrust.model_validate_json(json.dumps(document))


HOST = EngineHost(
    platform="linux-glibc-x86_64",
    lanes=frozenset({"rocm", "vulkan", "cpu"}),
    compute_targets=frozenset({"gfx1151"}),
    skulk_version="2.0.3",
    control_protocols=(1, 1),
    python_version="3.13.1",
)


def _refusal(document: bytes, trust: RuntimeTrust, now: int = NOW) -> str:
    with pytest.raises(EnginePackageRefusedError) as caught:
        verify_engine_manifest(document, trust, now=now)
    return caught.value.code


@pytest.fixture
def key() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.generate()


def test_valid_manifest_verifies_and_names_its_build(key: Ed25519PrivateKey) -> None:
    claims = _claims()
    package = verify_engine_manifest(_sign(key, claims), _trust(key), now=NOW)

    assert package.claims.engine_id == "foxlight.tensorfold"
    assert package.claims.backend_tag == "tensorfold-rocm"
    assert package.digest == hashlib.sha256(canonical_json(claims)).hexdigest()


def test_a_requirement_may_name_the_members_a_node_unpacks(
    key: Ed25519PrivateKey,
) -> None:
    """A manifest without the claim keeps its identity; one with it is a new build."""
    plain = _claims()
    named = _with(
        plain,
        ("external_requirements", 0, "unpack"),
        {
            "members": ["_rocm_sdk_core/lib/libamdhip64.so.7"],
            "unpacked_size": 28_303_953,
        },
    )

    whole = verify_engine_manifest(_sign(key, plain), _trust(key), now=NOW)
    unpacked = verify_engine_manifest(_sign(key, named), _trust(key), now=NOW)

    assert whole.claims.external_requirements[0].unpack is None
    assert whole.digest == hashlib.sha256(canonical_json(plain)).hexdigest()
    claim = unpacked.claims.external_requirements[0].unpack
    assert claim is not None
    assert claim.members == ("_rocm_sdk_core/lib/libamdhip64.so.7",)
    assert unpacked.digest != whole.digest


def test_a_build_may_say_that_it_drafts(key: Ed25519PrivateKey) -> None:
    """A manifest without the claim keeps its identity and states drafting off."""
    plain = _claims()
    drafting = _with(plain, ("drafting",), True)

    without = verify_engine_manifest(_sign(key, plain), _trust(key), now=NOW)
    stated = verify_engine_manifest(_sign(key, drafting), _trust(key), now=NOW)

    assert without.claims.drafting is False
    assert without.digest == hashlib.sha256(canonical_json(plain)).hexdigest()
    assert stated.claims.drafting is True
    assert stated.digest != without.digest
    # The claim is a JSON boolean and nothing else.
    assert _refusal(_sign(key, _with(plain, ("drafting",), "yes")), _trust(key)) == (
        "manifest_invalid"
    )


def test_build_identity_ignores_key_order_and_whitespace(
    key: Ed25519PrivateKey,
) -> None:
    claims = _claims()
    reordered = Claims(reversed(list(claims.items())))
    pretty = json.dumps(
        {"signature": _signature(key, claims), "engine": reordered}, indent=2
    )

    first = verify_engine_manifest(_sign(key, claims), _trust(key), now=NOW)
    second = verify_engine_manifest(pretty.encode(), _trust(key), now=NOW)

    assert first.digest == second.digest


def test_changed_claim_fails_the_signature(key: Ed25519PrivateKey) -> None:
    claims = _claims()
    changed = _with(claims, ("archive", "sha256"), "d" * 64)
    document = _envelope(changed, _signature(key, claims))

    assert _refusal(document, _trust(key)) == "signature_refused"


def test_signature_without_the_engine_domain_is_refused(
    key: Ed25519PrivateKey,
) -> None:
    # A capability release signs its canonical JSON with no prefix. The same
    # key must not be able to produce an engine manifest that way.
    document = _sign(key, _claims(), domain=b"")

    assert _refusal(document, _trust(key)) == "signature_refused"


def test_unknown_and_revoked_publishers_are_refused(key: Ed25519PrivateKey) -> None:
    other = Ed25519PrivateKey.generate()
    document = _sign(key, _claims())

    unknown = _trust(other, publishers={"someone-else": _public_hex(other)})
    revoked = _trust(key, revoked_publishers=[PUBLISHER])

    assert _refusal(document, unknown) == "trust_refused"
    assert _refusal(document, revoked) == "trust_refused"


def test_key_of_another_publisher_does_not_verify(key: Ed25519PrivateKey) -> None:
    other = Ed25519PrivateKey.generate()

    assert _refusal(_sign(other, _claims()), _trust(key)) == "signature_refused"


def test_expired_trust_is_refused_before_the_signature(key: Ed25519PrivateKey) -> None:
    trust = _trust(key, expires_at=NOW - 1)

    assert _refusal(_sign(key, _claims()), trust) == "trust_refused"


@pytest.mark.parametrize("now", [NOW - 61, NOW + 86_400])
def test_manifest_outside_its_window_is_refused(
    key: Ed25519PrivateKey, now: int
) -> None:
    trust = _trust(key, expires_at=NOW + 10 * 86_400)

    assert _refusal(_sign(key, _claims()), trust, now=now) == "window_refused"


@pytest.mark.parametrize("which", ["manifest", "archive", "adapter", "requirement"])
def test_revoked_package_or_file_is_refused(
    key: Ed25519PrivateKey, which: str
) -> None:
    claims = _claims()
    digests = {
        "manifest": hashlib.sha256(canonical_json(claims)).hexdigest(),
        "archive": ARCHIVE_SHA256,
        "adapter": ADAPTER_SHA256,
        "requirement": REQUIREMENT_SHA256,
    }
    trust = _trust(key, revoked_artifacts=[digests[which]])

    assert _refusal(_sign(key, claims), trust) == "artifact_revoked"


_SCHEMA_VIOLATIONS: list[tuple[ClaimPath, JsonValue]] = [
    (("unknown_field",), True),
    (("schema_version",), 2),
    (("engine_id",), "tensorfold"),
    (("lane",), "quantum"),
    (("skulk_versions",), "not a specifier"),
    (("expires_at",), NOW - 60),
    (("compute_targets",), ["gfx1151", "gfx1151"]),
    (("archive", "executable"), "../bin/engine"),
    (("archive", "executable"), "/usr/bin/engine"),
    (("archive", "filename"), "../engine.tar"),
    (("adapter", "python_requires"), None),
    (("adapter", "kind"), "native"),
    (("control_protocol", "minimum"), 3),
    (("external_requirements", 0, "url"), "http://example.invalid/file.whl"),
    (("archive", "filename"), "engine.rar"),
    # Members can be unpacked only out of a zip distribution, each named once,
    # into a folder the requirement's own name can be.
    (("external_requirements", 0, "unpack"), {"members": [], "unpacked_size": 1}),
    (
        ("external_requirements", 0, "unpack"),
        {"members": ["lib/a.so", "lib/a.so"], "unpacked_size": 1},
    ),
    (
        ("external_requirements", 0, "unpack"),
        {"members": ["../lib/a.so"], "unpacked_size": 1},
    ),
    (
        ("external_requirements", 0, "unpack"),
        {"members": ["lib/a.so"], "unpacked_size": 0},
    ),
    (
        ("external_requirements",),
        [
            {
                "name": "vendor-runtime",
                "version": "1.0",
                "filename": "vendor_runtime-1.0.tar.gz",
                "size": 10,
                "sha256": "a" * 64,
                "url": "https://vendor.example.invalid/vendor_runtime-1.0.tar.gz",
                "unpack": {"members": ["lib/a.so"], "unpacked_size": 1},
            }
        ],
    ),
    (
        ("external_requirements",),
        [
            {
                "name": "..",
                "version": "1.0",
                "filename": "vendor_runtime-1.0-py3-none-any.whl",
                "size": 10,
                "sha256": "a" * 64,
                "url": "https://vendor.example.invalid/vendor_runtime-1.0-py3-none-any.whl",
                "unpack": {"members": ["lib/a.so"], "unpacked_size": 1},
            }
        ],
    ),
    # A package carries an engine and an adapter. Weights stay in the model
    # store, so the schema has no place to put them.
    (("weights",), {"filename": "model.safetensors", "sha256": "f" * 64}),
]


@pytest.mark.parametrize(("path", "value"), _SCHEMA_VIOLATIONS)
def test_signed_claims_outside_the_schema_are_refused(
    key: Ed25519PrivateKey, path: ClaimPath, value: JsonValue
) -> None:
    claims = _with(_claims(), path, value)

    # The signature is valid, so the refusal comes from the schema alone.
    assert _refusal(_sign(key, claims), _trust(key)) == "manifest_invalid"


def test_oversized_and_malformed_documents_are_refused(key: Ed25519PrivateKey) -> None:
    trust = _trust(key)
    oversized = b" " * (engine_packages.MAX_ENGINE_MANIFEST_BYTES + 1)

    assert _refusal(oversized, trust) == "manifest_invalid"
    assert _refusal(b"not json", trust) == "manifest_invalid"
    assert _refusal(b'{"engine": {}, "signature": "zz"}', trust) == "manifest_invalid"
    assert _refusal(b'{"runtime": {}, "signature": "' + b"0" * 128 + b'"}', trust) == (
        "manifest_invalid"
    )


def test_fitting_host_gets_the_highest_common_protocol(key: Ed25519PrivateKey) -> None:
    package = verify_engine_manifest(_sign(key, _claims()), _trust(key), now=NOW)

    wider = dataclasses.replace(HOST, control_protocols=(1, 5))
    newer = dataclasses.replace(HOST, control_protocols=(2, 2))

    assert check_engine_host(package, HOST) == 1
    assert check_engine_host(package, wider) == 2
    assert select_control_protocol(package.claims, newer) == 2


@pytest.mark.parametrize(
    ("host", "code"),
    [
        (dataclasses.replace(HOST, platform="linux-glibc-aarch64"), "platform_mismatch"),
        (dataclasses.replace(HOST, platform="macos-arm64"), "platform_mismatch"),
        (dataclasses.replace(HOST, platform="linux-musl-x86_64"), "platform_mismatch"),
        (
            dataclasses.replace(HOST, lanes=frozenset({"vulkan", "cpu"})),
            "lane_unavailable",
        ),
        (
            dataclasses.replace(HOST, compute_targets=frozenset({"gfx1100"})),
            "compute_target_mismatch",
        ),
        (dataclasses.replace(HOST, skulk_version="2.0.2"), "skulk_version_unsupported"),
        (dataclasses.replace(HOST, skulk_version="3.0.0"), "skulk_version_unsupported"),
        (
            dataclasses.replace(HOST, skulk_version="not-a-version"),
            "skulk_version_unsupported",
        ),
        (dataclasses.replace(HOST, control_protocols=(3, 4)), "protocol_unsupported"),
        (
            dataclasses.replace(HOST, python_version="3.11.9"),
            "adapter_python_unsupported",
        ),
    ],
)
def test_host_that_cannot_run_the_build_is_refused_by_name(
    key: Ed25519PrivateKey, host: EngineHost, code: str
) -> None:
    package = verify_engine_manifest(_sign(key, _claims()), _trust(key), now=NOW)

    with pytest.raises(EnginePackageRefusedError) as caught:
        check_engine_host(package, host)

    assert caught.value.code == code


def test_host_with_undetected_device_target_stays_eligible(
    key: Ed25519PrivateKey,
) -> None:
    # Failed detection is not evidence of a mismatch; the engine's own
    # start-up is the final check.
    package = verify_engine_manifest(_sign(key, _claims()), _trust(key), now=NOW)

    undetected = dataclasses.replace(HOST, compute_targets=frozenset())

    assert check_engine_host(package, undetected) == 1


def test_build_without_compiled_targets_fits_any_device_of_its_lane(
    key: Ed25519PrivateKey,
) -> None:
    claims = _claims()
    claims.update(platform="macos-arm64", lane="metal", compute_targets=[])
    package = verify_engine_manifest(_sign(key, claims), _trust(key), now=NOW)
    host = dataclasses.replace(
        HOST,
        platform="macos-arm64",
        lanes=frozenset({"metal"}),
        compute_targets=frozenset({"apple-m4"}),
    )

    assert package.claims.backend_tag == "tensorfold-metal"
    assert check_engine_host(package, host) == 1


def test_host_with_undetected_c_library_stays_eligible(
    key: Ed25519PrivateKey,
) -> None:
    package = verify_engine_manifest(_sign(key, _claims()), _trust(key), now=NOW)
    undetected = dataclasses.replace(HOST, platform="linux-unknown-x86_64")

    assert check_engine_host(package, undetected) == 1


def test_file_must_match_signed_size_and_digest(tmp_path: Path) -> None:
    content = b"engine archive bytes"
    path = tmp_path / "engine.tar.xz"
    path.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()

    verify_engine_file(path, size=len(content), sha256=digest)

    for size, sha256 in ((len(content) + 1, digest), (len(content), "e" * 64)):
        with pytest.raises(EnginePackageRefusedError) as caught:
            verify_engine_file(path, size=size, sha256=sha256)
        assert caught.value.code == "archive_corrupt"


def test_missing_file_and_symlink_are_refused(tmp_path: Path) -> None:
    content = b"engine archive bytes"
    real = tmp_path / "real.tar.xz"
    real.write_bytes(content)
    link = tmp_path / "link.tar.xz"
    link.symlink_to(real)
    digest = hashlib.sha256(content).hexdigest()

    for path in (tmp_path / "absent.tar.xz", link, tmp_path):
        with pytest.raises(EnginePackageRefusedError) as caught:
            verify_engine_file(path, size=len(content), sha256=digest)
        assert caught.value.code == "archive_corrupt"
