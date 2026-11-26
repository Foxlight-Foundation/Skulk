"""Engine packages a node can install: what it offers, and installing one for a launch."""

import hashlib
import io
import json
import tarfile
import threading
import time
from pathlib import Path

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from skulk.engines.inventory import read_engine_package_inventory
from skulk.engines.on_demand import (
    EnginePackageInstallError,
    engine_host_of,
    engine_package_download_bytes,
    engine_package_offers,
    followed_channels,
    install_engine_package_on_demand,
    offered_engine_package_reports,
    on_demand_gates_pass,
    published_engine_package,
    refresh_engine_package_list,
)
from skulk.extensions.capability_store import StoreTrustClient
from skulk.extensions.runtime_artifacts import canonical_json
from skulk.extensions.tests.store_repository import StoreRepository, trust_document
from skulk.facts.testing import NVIDIA_A40, make_facts
from skulk.provisioning.engine_index import (
    ENGINE_INDEX_SIGNATURE_DOMAIN,
    EngineChannel,
    EngineIndexClient,
)
from skulk.provisioning.engine_trust import ENGINE_TRUST_TARGET, engine_trust_client
from skulk.provisioning.llama_server import AUTOPROVISION_OPT_OUT_ENV
from skulk.provisioning.tests.engine_package_fixtures import (
    PUBLISHER,
    Claims,
    engine_claims,
    sign_engine_claims,
)
from skulk.shared.backends import ENGINE_PACKAGES_DISABLE_ENV
from skulk.shared.types.node_facts import GpuDeviceFact, NodeFacts
from skulk.store.config import SkulkConfig

STORE = "https://store.invalid/engines/"
PLATFORM = "linux-glibc-x86_64"
TAG = "tensorfold-rocm"
ENGINE = b"#!/bin/sh\necho engine\n"
ADAPTER = b"print('adapter')\n"

STRIX = GpuDeviceFact(
    vendor="amd",
    name="AMD GPU",
    detection_source="amdgpu_sysfs",
    gpu_architecture="gfx1151",
)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _archive() -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        info = tarfile.TarInfo("bin/engine")
        info.mode = 0o755
        info.size = len(ENGINE)
        archive.addfile(info, io.BytesIO(ENGINE))
    return stream.getvalue()


class _Store:
    """A store with engine trust, one published package and a package list."""

    def __init__(self, root: Path) -> None:
        self.now = int(time.time())
        self.key = Ed25519PrivateKey.generate()
        self.repository = StoreRepository()
        embedded = root / "capability_store_root.json"
        embedded.write_bytes(self.repository.embedded_root)
        self.repository.publish_trust(
            trust_document(self.key, revision=1, publisher=PUBLISHER),
            trust_target=ENGINE_TRUST_TARGET,
        )
        self.trust: StoreTrustClient = engine_trust_client(
            root / "engine-trust",
            embedded_root=embedded,
            fetcher=self.repository.fetcher,
        )
        self.served: dict[str, bytes] = {}
        self.seen: list[str] = []
        self.index = EngineIndexClient(
            root / "engine-index",
            base_url=STORE,
            transport=httpx.MockTransport(self._respond),
        )
        self.packages = root / "packages"
        self.revision = 0
        self.entries: list[Claims] = []
        self.digest = self.publish_package(generation=1)

    def _respond(self, request: httpx.Request) -> httpx.Response:
        self.seen.append(str(request.url))
        content = self.served.get(str(request.url))
        return httpx.Response(404) if content is None else httpx.Response(200, content=content)

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._respond)

    def publish_package(self, *, generation: int, channel: str = "stable") -> str:
        """Sign a manifest for real files, serve them, and list the build."""
        archive = _archive()
        claims = engine_claims()
        claims.update(
            generation=generation,
            skulk_versions=">=2,<99",
            control_protocol={"minimum": 1, "maximum": 1},
            issued_at=self.now - 60,
            expires_at=self.now + 3600,
            external_requirements=[],
            archive={
                "filename": "engine.tar.gz",
                "size": len(archive),
                "sha256": _sha256(archive),
                "unpacked_size": 1_000_000,
                "executable": "bin/engine",
            },
            adapter={
                "filename": "adapter.py",
                "size": len(ADAPTER),
                "sha256": _sha256(ADAPTER),
                "kind": "python",
                "python_requires": ">=3.12",
            },
        )
        digest = _sha256(canonical_json(claims))
        directory = f"foxlight.tensorfold/rocm/{digest}"
        self.served[f"{STORE}{directory}/manifest.json"] = sign_engine_claims(
            self.key, claims
        )
        self.served[f"{STORE}{directory}/engine.tar.gz"] = archive
        self.served[f"{STORE}{directory}/adapter.py"] = ADAPTER
        self.entries.append(
            {
                "engine_id": "foxlight.tensorfold",
                "engine_version": "1.3.2",
                "generation": generation,
                "channel": channel,
                "platform": PLATFORM,
                "lane": "rocm",
                "compute_targets": ["gfx1151"],
                "skulk_versions": ">=2,<99",
                "control_protocol": {"minimum": 1, "maximum": 1},
                "adapter_python_requires": ">=3.12",
                "manifest_digest": digest,
                "directory": directory,
                "download_bytes": len(archive) + len(ADAPTER),
                "conservative_overhead_bytes": 2 * 1024**3,
            }
        )
        self.publish_list()
        return digest

    def publish_list(self) -> None:
        self.revision += 1
        claims: Claims = {
            "schema_version": 1,
            "publisher": PUBLISHER,
            "revision": self.revision,
            "issued_at": self.now - 60,
            "expires_at": self.now + 3600,
            "entries": list(self.entries),
        }
        signature = self.key.sign(
            ENGINE_INDEX_SIGNATURE_DOMAIN + canonical_json(claims)
        ).hex()
        self.served[f"{STORE}index-v1.json"] = json.dumps(
            {"index": claims, "signature": signature}
        ).encode()

    def refresh(self) -> bool:
        return refresh_engine_package_list(
            offline=False, environ={}, trust_client=self.trust, index_client=self.index
        )

    def offers(
        self,
        facts: NodeFacts,
        *,
        offline: bool = False,
        environ: dict[str, str] | None = None,
        channels: frozenset[EngineChannel] = frozenset({"stable"}),
    ) -> dict[str, str]:
        return {
            tag: entry.manifest_digest
            for tag, entry in engine_package_offers(
                facts,
                offline=offline,
                environ=environ if environ is not None else {},
                channels=channels,
                trust_client=self.trust,
                index_client=self.index,
                platform_name=PLATFORM,
            ).items()
        }

    def install(self, facts: NodeFacts, build: str, tag: str = TAG) -> None:
        install_engine_package_on_demand(
            tag,
            build,
            facts,
            offline=False,
            stop=threading.Event(),
            environ={},
            trust_client=self.trust,
            index_client=self.index,
            transport=self.transport,
            store_url=STORE,
            root=self.packages,
            platform_name=PLATFORM,
        )


@pytest.fixture
def store(tmp_path: Path) -> _Store:
    return _Store(tmp_path)


@pytest.fixture
def strix() -> NodeFacts:
    return make_facts(platform="linux", gpus=(STRIX,))


def test_a_node_is_described_by_its_observed_hardware() -> None:
    strix = engine_host_of(
        make_facts(platform="linux", gpus=(STRIX,)), platform_name=PLATFORM
    )
    nvidia = engine_host_of(
        make_facts(platform="linux", gpus=(NVIDIA_A40,)), platform_name=PLATFORM
    )
    mac = engine_host_of(make_facts(platform="darwin"), platform_name="macos-arm64")
    plain = engine_host_of(make_facts(platform="linux"), platform_name=PLATFORM)

    assert strix is not None and nvidia is not None and mac is not None
    assert plain is not None
    assert strix.lanes == {"cpu", "rocm", "vulkan"}
    assert strix.compute_targets == {"gfx1151"}
    assert nvidia.lanes == {"cpu", "cuda", "vulkan"}
    assert nvidia.compute_targets == {"sm_86"}
    assert mac.lanes == {"cpu", "metal"}
    assert plain.lanes == {"cpu"} and plain.compute_targets == frozenset()


def test_nothing_is_offered_before_the_node_has_a_package_list(
    store: _Store, strix: NodeFacts
) -> None:
    assert store.offers(strix) == {}
    assert store.seen == []


def test_a_fitting_published_package_is_offered_from_the_kept_list(
    store: _Store, strix: NodeFacts
) -> None:
    assert store.refresh()
    store.seen.clear()

    assert store.offers(strix) == {TAG: store.digest}
    # Reading the offers uses the kept list only.
    assert store.seen == []


def test_an_offered_build_is_reported_with_what_installing_it_means(
    store: _Store, strix: NodeFacts
) -> None:
    """The engines report says which build, from which channel, and how big."""
    store.refresh()
    offers = engine_package_offers(
        strix,
        offline=False,
        environ={},
        channels=frozenset({"stable"}),
        trust_client=store.trust,
        index_client=store.index,
        platform_name=PLATFORM,
    )

    (report,) = offered_engine_package_reports(offers)

    entry = offers[TAG]
    assert report.status == "offered"
    assert report.backend_tag == TAG
    assert report.engine_id == entry.engine_id
    assert report.lane == entry.lane
    assert report.build == store.digest
    assert report.generation == 1
    assert report.channel == "stable"
    assert report.download_bytes == entry.download_bytes
    assert report.overhead_bytes == entry.conservative_overhead_bytes
    assert report.installed_builds == () and report.publisher is None


def test_a_node_the_package_does_not_fit_is_offered_nothing(store: _Store) -> None:
    store.refresh()

    assert store.offers(make_facts(platform="linux", gpus=(NVIDIA_A40,))) == {}
    assert store.offers(make_facts(platform="linux")) == {}


@pytest.mark.parametrize(
    "environ",
    [
        {AUTOPROVISION_OPT_OUT_ENV: "1"},
        {"SKULK_NODE_PARTICIPATION": "management"},
    ],
)
def test_a_node_that_may_not_install_is_offered_nothing(
    store: _Store, strix: NodeFacts, environ: dict[str, str]
) -> None:
    store.refresh()

    assert store.offers(strix, environ=environ) == {}
    assert not on_demand_gates_pass(environ)


def test_an_offline_node_is_offered_nothing(store: _Store, strix: NodeFacts) -> None:
    store.refresh()

    assert store.offers(strix, offline=True) == {}


def test_packages_turned_off_offer_and_fetch_nothing(
    store: _Store, strix: NodeFacts, monkeypatch: pytest.MonkeyPatch
) -> None:
    store.refresh()
    store.seen.clear()
    monkeypatch.setenv(ENGINE_PACKAGES_DISABLE_ENV, "1")

    assert store.offers(strix) == {}
    assert not store.refresh()
    assert store.seen == []


def test_a_package_the_node_already_holds_is_not_offered_again(
    store: _Store, strix: NodeFacts
) -> None:
    store.refresh()
    holding = strix.model_copy(update={"engine_package_builds": {TAG: "0" * 64}})

    assert store.offers(holding) == {}


def test_a_candidate_build_is_offered_only_when_the_node_follows_candidates(
    store: _Store, strix: NodeFacts
) -> None:
    candidate = store.publish_package(generation=2, channel="candidate")
    store.refresh()

    assert store.offers(strix) == {TAG: store.digest}
    assert store.offers(strix, channels=frozenset({"stable", "candidate"})) == {
        TAG: candidate
    }


def test_the_candidate_choice_is_a_cluster_setting() -> None:
    def following(channel: str) -> SkulkConfig:
        return SkulkConfig.model_validate({"engines": {"package_channel": channel}})

    assert followed_channels(lambda: None) == {"stable"}
    assert followed_channels(lambda: SkulkConfig()) == {"stable"}
    assert followed_channels(lambda: following("stable")) == {"stable"}
    assert followed_channels(lambda: following("candidate")) == {
        "stable",
        "candidate",
    }


def test_the_launch_environment_cannot_choose_candidate_builds(
    store: _Store, strix: NodeFacts, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A packaged Skulk has no launch environment a person can change, so the
    # choice must not depend on one.
    _ = store.publish_package(generation=2, channel="candidate")
    store.refresh()
    monkeypatch.setenv("SKULK_ENGINE_PACKAGE_CHANNEL", "candidate")

    assert store.offers(
        strix, environ={"SKULK_ENGINE_PACKAGE_CHANNEL": "candidate"}
    ) == {TAG: store.digest}
    assert followed_channels(lambda: None) == {"stable"}


def test_settings_that_cannot_be_read_mean_released_builds_only() -> None:
    def damaged() -> SkulkConfig | None:
        raise OSError("settings file is unreadable")

    assert followed_channels(damaged) == {"stable"}


def test_a_published_build_is_found_by_its_identity_for_its_size(
    store: _Store,
) -> None:
    store.refresh()

    found = published_engine_package(
        store.digest, trust_client=store.trust, index_client=store.index
    )
    missing = published_engine_package(
        "0" * 64, trust_client=store.trust, index_client=store.index
    )

    assert found is not None and found.download_bytes == len(_archive()) + len(ADAPTER)
    assert missing is None
    assert engine_package_download_bytes(None) == 0


def test_a_launch_installs_and_activates_the_exact_build(
    store: _Store, strix: NodeFacts
) -> None:
    store.install(strix, store.digest)

    active = read_engine_package_inventory(root=store.packages).active[TAG]
    assert active.digest == store.digest
    assert active.executable.read_bytes() == ENGINE
    # The node now holds the package, so it is no longer offered on demand.
    holding = strix.model_copy(update={"engine_package_builds": {TAG: store.digest}})
    assert store.offers(holding) == {}


def test_an_older_build_a_placement_names_is_installed_while_it_stays_listed(
    store: _Store, strix: NodeFacts
) -> None:
    """A placement is stamped before the install; a newer build must not replace it."""
    first = store.digest
    store.publish_package(generation=2)

    store.install(strix, first)

    assert read_engine_package_inventory(root=store.packages).active[TAG].digest == first


def _failed_step(store: _Store, facts: NodeFacts, build: str, tag: str = TAG) -> str:
    with pytest.raises(EnginePackageInstallError) as caught:
        store.install(facts, build, tag)
    assert read_engine_package_inventory(root=store.packages).active == {}
    return caught.value.step


def test_a_build_that_is_not_published_is_not_installed(
    store: _Store, strix: NodeFacts
) -> None:
    assert _failed_step(store, strix, "0" * 64) == "the package list check"
    assert _failed_step(store, strix, store.digest, "tensorfold-vulkan") == (
        "the package list check"
    )


def test_a_manifest_other_than_the_one_listed_is_not_installed(
    store: _Store, strix: NodeFacts
) -> None:
    """The list names a build; the store cannot serve a different signed one in its place."""
    listed = store.digest
    other = store.publish_package(generation=2)
    store.served[f"{STORE}foxlight.tensorfold/rocm/{listed}/manifest.json"] = (
        store.served[f"{STORE}foxlight.tensorfold/rocm/{other}/manifest.json"]
    )

    assert _failed_step(store, strix, listed) == "the manifest check"


def test_a_list_and_manifest_that_disagree_about_memory_install_nothing(
    store: _Store, strix: NodeFacts
) -> None:
    """A launch is admitted on the list's figure, so the manifest must state the same."""
    store.entries[0]["conservative_overhead_bytes"] = 1024**3
    store.publish_list()

    with pytest.raises(EnginePackageInstallError, match="different memory needs"):
        store.install(strix, store.digest)
    assert _failed_step(store, strix, store.digest) == "the manifest check"


def test_a_missing_manifest_or_file_installs_nothing(
    store: _Store, strix: NodeFacts
) -> None:
    directory = f"{STORE}foxlight.tensorfold/rocm/{store.digest}/"
    archive = store.served.pop(directory + "engine.tar.gz")

    assert _failed_step(store, strix, store.digest) == "the package download"

    store.served[directory + "engine.tar.gz"] = archive
    del store.served[directory + "manifest.json"]
    assert _failed_step(store, strix, store.digest) == "the manifest download"


def test_a_node_the_build_does_not_fit_refuses_to_install_it(store: _Store) -> None:
    nvidia = make_facts(platform="linux", gpus=(NVIDIA_A40,))

    assert _failed_step(store, nvidia, store.digest) == "the manifest check"


def test_an_offline_or_opted_out_node_installs_nothing(
    store: _Store, strix: NodeFacts
) -> None:
    for offline, environ, step in (
        (True, {}, "the network check"),
        (False, {AUTOPROVISION_OPT_OUT_ENV: "1"}, "the install check"),
    ):
        with pytest.raises(EnginePackageInstallError) as caught:
            install_engine_package_on_demand(
                TAG,
                store.digest,
                strix,
                offline=offline,
                stop=threading.Event(),
                environ=environ,
                trust_client=store.trust,
                index_client=store.index,
                transport=store.transport,
                store_url=STORE,
                root=store.packages,
                platform_name=PLATFORM,
            )
        assert caught.value.step == step
    assert store.seen == []
