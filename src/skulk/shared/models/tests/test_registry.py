# pyright: reportPrivateUsage=false
"""Tests for signed registry loading and artifact identity separation."""

import json
from pathlib import Path
from typing import cast

import pytest
from anyio import Path as AsyncPath

import skulk.download.download_utils as download_utils
import skulk.shared.constants as constants_module
import skulk.shared.models.model_cards as model_cards_module
import skulk.shared.models.registry as registry_module
from skulk.shared.models.model_cards import ModelCard, ModelTask, registry_model_cards
from skulk.shared.models.registry import (
    RegistryCatalog,
    RegistryUnavailableError,
    TufRegistryClient,
)
from skulk.shared.types.common import ModelId
from skulk.shared.types.memory import Memory
from skulk.shared.types.worker.shards import PipelineShardMetadata
from skulk.store.installed_cards import (
    build_installed_card_record,
    write_installed_card,
)


def _catalog_payload() -> bytes:
    return json.dumps(
        {
            "schema_version": 2,
            "snapshot_id": "snapshot_1_test",
            "generated_at": "2026-08-08T12:00:00Z",
            "published_by": "validator@example.com",
            "note": "test",
            "card_metadata": {f"card_{'a' * 52}": {"provenance": "foxlight"}},
            "cards": [
                {
                    "schema_version": 1,
                    "card_id": f"card_{'a' * 52}",
                    "alias": "org/multi-gguf@q4-k-m",
                    "model_ref": "org/multi-gguf@q4-k-m",
                    "artifact": {
                        "repository": "org/multi-gguf",
                        "revision": "b" * 40,
                        "selected_file": "model-Q4_K_M.gguf",
                        "format": "gguf",
                        "quantization": "Q4_K_M",
                    },
                    "card": {
                        "model_id": "org/multi-gguf",
                        "source_revision": "b" * 40,
                        "storage_size": {"in_bytes": 1024},
                        "n_layers": 4,
                        "hidden_size": 64,
                        "supports_tensor": False,
                        "tasks": ["TextGeneration"],
                        "gguf_file": "model-Q4_K_M.gguf",
                        "quantization": "Q4_K_M",
                        "placement": {"compatible_backends": ["llama_server"]},
                    },
                }
            ],
        }
    ).encode()


def _advisories_payload() -> bytes:
    """Build one active signed-warning target for cache recovery tests."""

    return json.dumps(
        {
            "schema_version": 1,
            "generated_at": "2026-08-10T12:00:00Z",
            "enforcement": "warn",
            "advisories": [
                {
                "schema_version": 1,
                "advisory_id": "FLA-2026-0001",
                "severity": "critical",
                "title": "Test model warning",
                "description": "Retain this warning during a registry outage.",
                "affected_card_ids": (f"card_{'a' * 52}",),
                "affected_model_aliases": (),
                "active": True,
                "created_at": "2026-08-10T12:00:00Z",
                "updated_at": "2026-08-10T12:00:00Z",
                "enforcement": "warn",
                }
            ],
        }
    ).encode()


def test_registry_alias_is_separate_from_artifact_repository() -> None:
    """Two quants can use distinct runtime ids while sharing one Hub repo."""
    catalog = RegistryCatalog.model_validate_json(_catalog_payload(), strict=False)

    card = registry_model_cards(catalog)[0]

    assert str(card.model_id) == "org/multi-gguf@q4-k-m"
    assert str(card.artifact_repository) == "org/multi-gguf"
    assert card.gguf_file == "model-Q4_K_M.gguf"
    assert card.registry_snapshot_id == "snapshot_1_test"
    assert card.registry_provenance == "foxlight"


@pytest.mark.parametrize("location", ["catalog", "card"])
def test_registry_rejects_unknown_schema_versions(location: str) -> None:
    """A client never interprets a future signed schema with v1 semantics."""
    payload = cast("dict[str, object]", json.loads(_catalog_payload()))
    if location == "catalog":
        payload["schema_version"] = 3
    else:
        cards = cast("list[dict[str, object]]", payload["cards"])
        cards[0]["schema_version"] = 2

    with pytest.raises(ValueError, match="schema_version"):
        RegistryCatalog.model_validate(payload, strict=False)


@pytest.mark.parametrize("alias", [".", "..", "org/..", "../model", "org\\model"])
def test_registry_rejects_path_like_aliases(alias: str) -> None:
    """Signed aliases can never address a staging root or its parent."""
    payload = cast("dict[str, object]", json.loads(_catalog_payload()))
    cards = cast("list[dict[str, object]]", payload["cards"])
    cards[0]["alias"] = alias

    with pytest.raises(ValueError, match="safe repository identifier"):
        RegistryCatalog.model_validate(payload, strict=False)


def test_registry_forces_signed_cards_to_non_custom() -> None:
    """Signed payload content cannot acquire local override semantics."""
    payload = cast("dict[str, object]", json.loads(_catalog_payload()))
    cards = cast("list[dict[str, object]]", payload["cards"])
    card_payload = cast("dict[str, object]", cards[0]["card"])
    card_payload["is_custom"] = True
    catalog = RegistryCatalog.model_validate(payload, strict=False)

    assert not registry_model_cards(catalog)[0].is_custom


def test_registry_rejects_unpinned_separate_processor_repository() -> None:
    """A signed card cannot approve code that remains mutable upstream."""
    payload = cast("dict[str, object]", json.loads(_catalog_payload()))
    cards = cast("list[dict[str, object]]", payload["cards"])
    card_payload = cast("dict[str, object]", cards[0]["card"])
    card_payload["vision"] = {
        "model_type": "test_vlm",
        "processor_repo": "org/processor",
    }
    catalog = RegistryCatalog.model_validate(payload, strict=False)

    with pytest.raises(ValueError, match="processor_revision"):
        registry_model_cards(catalog)


@pytest.mark.parametrize(
    ("section", "expected_field"),
    [
        (
            {"vision": {"model_type": "vlm", "weights_repo": "org/vision"}},
            "vision.weights_repo",
        ),
        (
            {"runtime": {"mtp_heads": True, "mtp_sidecar_repo": "org/mtp"}},
            "runtime.mtp_sidecar_repo",
        ),
        (
            {"runtime": {"assistant_model_repo": "org/assistant"}},
            "runtime.assistant_model_repo",
        ),
        (
            {
                "runtime": {
                    "served_spec_draft_repo": "org/draft",
                    "served_spec_draft_file": "draft.gguf",
                }
            },
            "runtime.served_spec_draft_repo",
        ),
        (
            {
                "runtime": {
                    "vllm_spec_method": "dflash",
                    "vllm_spec_draft_repo": "org/dflash",
                }
            },
            "runtime.vllm_spec_draft_repo",
        ),
    ],
)
def test_registry_rejects_unpinned_separate_companion_repository(
    section: dict[str, object], expected_field: str
) -> None:
    """Every companion source participates in signed artifact identity."""
    payload = cast("dict[str, object]", json.loads(_catalog_payload()))
    cards = cast("list[dict[str, object]]", payload["cards"])
    card_payload = cast("dict[str, object]", cards[0]["card"])
    card_payload.update(section)
    catalog = RegistryCatalog.model_validate(payload, strict=False)

    with pytest.raises(ValueError, match=expected_field):
        registry_model_cards(catalog)


def test_registry_rejects_catalog_metadata_outside_card_identity() -> None:
    """Every published card has exactly one signed provenance record."""
    payload = cast("dict[str, object]", json.loads(_catalog_payload()))
    payload["card_metadata"] = {}
    catalog = RegistryCatalog.model_validate(payload, strict=False)

    with pytest.raises(ValueError, match="metadata does not match"):
        registry_model_cards(catalog)


def test_offline_mode_disables_registry_network_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Air-gapped nodes use bundled cards without contacting public TUF."""
    monkeypatch.delenv("SKULK_TESTS", raising=False)
    monkeypatch.setattr(model_cards_module, "SKULK_MODEL_REGISTRY_ENABLED", True)
    monkeypatch.setattr(model_cards_module, "SKULK_OFFLINE", True)

    assert not model_cards_module._registry_enabled()


@pytest.mark.asyncio
async def test_air_gap_restart_loads_installed_card_without_registry_lkg(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Complete installed bytes remain usable after registry cache expiry."""
    card = registry_model_cards(
        RegistryCatalog.model_validate_json(_catalog_payload(), strict=False)
    )[0]
    artifact = tmp_path / card.model_id.normalize()
    artifact.mkdir()
    (artifact / "model-Q4_K_M.gguf").write_bytes(b"weights")
    (artifact / ".skulk-source-revision").write_text(f"{card.source_revision}\n")
    write_installed_card(
        artifact,
        build_installed_card_record(artifact, card),
    )

    async def _registry_unavailable() -> bool:
        return False

    async def _no_cards(_path: object, *, is_custom: bool) -> None:
        del is_custom

    original_cache = dict(model_cards_module._card_cache)
    original_installed = dict(model_cards_module._installed_card_cache)
    model_cards_module._card_cache.clear()
    monkeypatch.setattr(constants_module, "SKULK_MODELS_DIR", tmp_path)
    monkeypatch.setattr(constants_module, "SKULK_MODELS_PATH", None)
    monkeypatch.setattr(
        model_cards_module, "_load_cards_from_registry", _registry_unavailable
    )
    monkeypatch.setattr(model_cards_module, "_load_cards_from_dir", _no_cards)
    try:
        await model_cards_module._refresh_card_cache()
        assert model_cards_module.get_card(card.model_id) == card
        installed = model_cards_module.get_installed_card_record(card.model_id)
        assert installed is not None
        assert installed.installed_identity == card.registry_card_id
    finally:
        model_cards_module._card_cache.clear()
        model_cards_module._card_cache.update(original_cache)
        model_cards_module._installed_card_cache.clear()
        model_cards_module._installed_card_cache.update(original_installed)


def test_completed_stage_converges_installed_cache_without_registry_refresh(
    tmp_path: Path,
) -> None:
    """Offline model metadata changes immediately after durable staging."""

    card = registry_model_cards(
        RegistryCatalog.model_validate_json(_catalog_payload(), strict=False)
    )[0]
    artifact = tmp_path / card.model_id.normalize()
    artifact.mkdir()
    (artifact / "model-Q4_K_M.gguf").write_bytes(b"weights")
    (artifact / ".skulk-source-revision").write_text(f"{card.source_revision}\n")
    record = build_installed_card_record(artifact, card)

    original_cache = dict(model_cards_module._card_cache)
    original_installed = dict(model_cards_module._installed_card_cache)
    original_installed_current = dict(
        model_cards_module._installed_current_registry_ids
    )
    model_cards_module._card_cache.clear()
    model_cards_module._installed_card_cache.clear()
    model_cards_module._installed_current_registry_ids.clear()
    try:
        model_cards_module.register_installed_card_record(record)

        assert model_cards_module.get_installed_card_record(card.model_id) == record
        assert model_cards_module.get_card(card.model_id) == card
    finally:
        model_cards_module._card_cache.clear()
        model_cards_module._card_cache.update(original_cache)
        model_cards_module._installed_card_cache.clear()
        model_cards_module._installed_card_cache.update(original_installed)
        model_cards_module._installed_current_registry_ids.clear()
        model_cards_module._installed_current_registry_ids.update(
            original_installed_current
        )


def test_artifact_eviction_unregisters_installed_generation(tmp_path: Path) -> None:
    """Deleted local bytes stop being active installed truth immediately."""

    card = registry_model_cards(
        RegistryCatalog.model_validate_json(_catalog_payload(), strict=False)
    )[0]
    artifact = tmp_path / card.model_id.normalize()
    artifact.mkdir()
    (artifact / "model-Q4_K_M.gguf").write_bytes(b"weights")
    record = build_installed_card_record(artifact, card)

    original_cache = dict(model_cards_module._card_cache)
    original_installed = dict(model_cards_module._installed_card_cache)
    original_installed_current = dict(
        model_cards_module._installed_current_registry_ids
    )
    original_dirty = model_cards_module._card_cache_dirty
    model_cards_module._card_cache.clear()
    model_cards_module._installed_card_cache.clear()
    model_cards_module._installed_current_registry_ids.clear()
    try:
        model_cards_module.register_installed_card_record(record)
        model_cards_module.unregister_installed_card_record(card.model_id)

        assert model_cards_module.get_installed_card_record(card.model_id) is None
        assert model_cards_module.get_card(card.model_id) is None
        assert model_cards_module._card_cache_dirty
    finally:
        model_cards_module._card_cache.clear()
        model_cards_module._card_cache.update(original_cache)
        model_cards_module._installed_card_cache.clear()
        model_cards_module._installed_card_cache.update(original_installed)
        model_cards_module._installed_current_registry_ids.clear()
        model_cards_module._installed_current_registry_ids.update(
            original_installed_current
        )
        model_cards_module._card_cache_dirty = original_dirty


def test_client_uses_hash_bound_last_known_good(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A verified catalog survives an outage but not local byte tampering."""
    payload_path = tmp_path / "downloaded.json"
    payload_path.write_bytes(_catalog_payload())
    embedded_root = tmp_path / "embedded-root.json"
    embedded_root.write_text("{}")
    cached_root = tmp_path / "cache/metadata/root.json"
    cached_root.parent.mkdir(parents=True)
    cached_root.write_text('{"attacker":"preseeded"}')
    observed_trusted_roots: list[bytes] = []

    class WorkingUpdater:
        def __init__(self, **kwargs: object) -> None:
            observed_trusted_roots.append(cast("bytes", kwargs["bootstrap"]))

        def refresh(self) -> None:
            pass

        def get_targetinfo(self, _: str) -> object:
            return object()

        def download_target(self, _: object) -> str:
            return str(payload_path)

    monkeypatch.setattr(registry_module, "Updater", WorkingUpdater)
    client = TufRegistryClient(
        base_url="https://registry.example/",
        cache_dir=tmp_path / "cache",
        embedded_root=embedded_root,
        timeout_seconds=1,
        max_stale_days=30,
    )
    assert client.load_catalog(registry_model_cards).snapshot_id == "snapshot_1_test"
    assert observed_trusted_roots == [embedded_root.read_bytes()]

    malformed_payload = cast("dict[str, object]", json.loads(_catalog_payload()))
    malformed_cards = cast("list[dict[str, object]]", malformed_payload["cards"])
    malformed_card = cast("dict[str, object]", malformed_cards[0]["card"])
    malformed_card["n_layers"] = "not-an-integer"
    payload_path.write_text(json.dumps(malformed_payload))
    assert client.load_catalog(registry_model_cards).snapshot_id == "snapshot_1_test"
    assert (tmp_path / "cache/last-known-good-catalog.json").read_bytes() == (
        _catalog_payload()
    )

    class FailingUpdater(WorkingUpdater):
        def refresh(self) -> None:
            raise OSError("offline")

    monkeypatch.setattr(registry_module, "Updater", FailingUpdater)
    assert client.load_catalog(registry_model_cards).snapshot_id == "snapshot_1_test"

    (tmp_path / "cache/last-known-good-catalog.json").write_bytes(b"tampered")
    with pytest.raises(RegistryUnavailableError):
        client.load_catalog()


def test_client_uses_hash_bound_last_known_good_advisories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verified warnings survive restart outages but not cache tampering."""

    payload_path = tmp_path / "advisories.json"
    payload_path.write_bytes(_advisories_payload())
    embedded_root = tmp_path / "embedded-root.json"
    embedded_root.write_text("{}")

    class WorkingUpdater:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def refresh(self) -> None:
            pass

        def get_targetinfo(self, target_path: str) -> object:
            assert target_path == "v1/advisories.json"
            return object()

        def download_target(self, _target: object) -> str:
            return str(payload_path)

    monkeypatch.setattr(registry_module, "Updater", WorkingUpdater)
    client = TufRegistryClient(
        base_url="https://registry.example/",
        cache_dir=tmp_path / "cache",
        embedded_root=embedded_root,
        timeout_seconds=1,
        max_stale_days=30,
    )
    assert client.load_advisories().advisories[0].advisory_id == "FLA-2026-0001"

    class FailingUpdater(WorkingUpdater):
        def refresh(self) -> None:
            raise OSError("offline")

    monkeypatch.setattr(registry_module, "Updater", FailingUpdater)
    assert client.load_advisories().advisories[0].severity == "critical"

    (tmp_path / "cache/last-known-good-advisories.json").write_bytes(b"tampered")
    with pytest.raises(RegistryUnavailableError):
        client.load_advisories()


def test_embedded_roots_match_release_resources() -> None:
    """Package and frozen-app trust anchors cannot drift independently."""
    package_root = registry_module.EMBEDDED_REGISTRY_ROOT.read_bytes()
    release_root = (
        Path(__file__).parents[5] / "resources/model_registry/root.json"
    ).read_bytes()
    assert package_root == release_root


@pytest.mark.asyncio
async def test_failed_refresh_removes_previous_registry_cards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An expired in-memory catalog cannot outlive the configured LKG bound."""
    registry_card = registry_model_cards(
        RegistryCatalog.model_validate_json(_catalog_payload(), strict=False)
    )[0]
    bundled_card = registry_card.model_copy(
        update={
            "model_id": ModelId("org/bundled"),
            "source_repository": None,
            "registry_card_id": None,
            "registry_snapshot_id": None,
        }
    )

    class FailingClient:
        def load_catalog(self, _catalog_validator: object = None) -> RegistryCatalog:
            raise OSError("registry offline and LKG expired")

    original_cache = dict(model_cards_module._card_cache)
    model_cards_module._card_cache.clear()
    model_cards_module._card_cache[registry_card.model_id] = registry_card
    model_cards_module._card_cache[bundled_card.model_id] = bundled_card
    monkeypatch.setattr(model_cards_module, "_registry_enabled", lambda: True)
    monkeypatch.setattr(model_cards_module, "_registry_client", FailingClient())
    try:
        await model_cards_module._load_cards_from_registry()
        assert registry_card.model_id not in model_cards_module._card_cache
        assert model_cards_module._card_cache[bundled_card.model_id] == bundled_card
    finally:
        model_cards_module._card_cache.clear()
        model_cards_module._card_cache.update(original_cache)


@pytest.mark.asyncio
async def test_successful_refresh_excludes_unlisted_bundled_cards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A signed snapshot can revoke a card that the distribution once bundled."""
    catalog = RegistryCatalog.model_validate_json(_catalog_payload(), strict=False)
    registry_card = registry_model_cards(catalog)[0]
    bundled_card = registry_card.model_copy(
        update={
            "model_id": ModelId("org/revoked-bundled"),
            "source_repository": None,
            "registry_card_id": None,
            "registry_snapshot_id": None,
            "registry_provenance": None,
        }
    )
    custom_card = bundled_card.model_copy(
        update={"model_id": ModelId("org/custom"), "is_custom": True}
    )

    class WorkingClient:
        def load_catalog(self, _catalog_validator: object = None) -> RegistryCatalog:
            return catalog

    original_cache = dict(model_cards_module._card_cache)
    model_cards_module._card_cache.clear()
    model_cards_module._card_cache[bundled_card.model_id] = bundled_card
    model_cards_module._card_cache[custom_card.model_id] = custom_card
    monkeypatch.setattr(model_cards_module, "_registry_enabled", lambda: True)
    monkeypatch.setattr(model_cards_module, "_registry_client", WorkingClient())
    try:
        assert await model_cards_module._load_cards_from_registry()
        assert bundled_card.model_id not in model_cards_module._card_cache
        assert model_cards_module._card_cache[registry_card.model_id] == registry_card
        assert model_cards_module._card_cache[custom_card.model_id] == custom_card
    finally:
        model_cards_module._card_cache.clear()
        model_cards_module._card_cache.update(original_cache)


@pytest.mark.asyncio
async def test_current_custom_file_supersedes_retained_custom_sidecar(
    tmp_path: Path,
) -> None:
    """Operator edits remain authoritative after installed-card recovery."""

    catalog = RegistryCatalog.model_validate_json(_catalog_payload(), strict=False)
    base = registry_model_cards(catalog)[0].model_copy(
        update={
            "registry_card_id": None,
            "registry_snapshot_id": None,
            "registry_provenance": None,
            "is_custom": True,
        }
    )
    retained = base.model_copy(update={"hidden_size": 64})
    edited = base.model_copy(update={"hidden_size": 128})
    custom_directory = AsyncPath(tmp_path)
    await edited.save(custom_directory / "edited.toml")

    original_cache = dict(model_cards_module._card_cache)
    model_cards_module._card_cache.clear()
    model_cards_module._card_cache[retained.model_id] = retained
    try:
        await model_cards_module._load_cards_from_dir(
            custom_directory,
            is_custom=True,
        )

        selected = model_cards_module._card_cache[retained.model_id]
        assert selected.hidden_size == 128
        assert selected.is_custom
    finally:
        model_cards_module._card_cache.clear()
        model_cards_module._card_cache.update(original_cache)


@pytest.mark.asyncio
async def test_complete_catalog_is_available_without_image_ui_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Store authority can resolve signed image cards on a non-image host."""
    catalog = RegistryCatalog.model_validate_json(_catalog_payload(), strict=False)
    text_card = registry_model_cards(catalog)[0]
    image_card = text_card.model_copy(
        update={
            "model_id": ModelId("org/image"),
            "tasks": [model_cards_module.ModelTask.TextToImage],
        }
    )
    original_cache = dict(model_cards_module._card_cache)
    model_cards_module._card_cache.clear()
    model_cards_module._card_cache[text_card.model_id] = text_card
    model_cards_module._card_cache[image_card.model_id] = image_card
    monkeypatch.setattr(model_cards_module, "SKULK_ENABLE_IMAGE_MODELS", False)
    monkeypatch.setattr(model_cards_module, "_last_registry_refresh", 100.0)
    monkeypatch.setattr(model_cards_module.time, "monotonic", lambda: 101.0)
    try:
        assert await model_cards_module.get_all_model_cards() == [
            text_card,
            image_card,
        ]
        assert await model_cards_module.get_model_cards() == [text_card]
    finally:
        model_cards_module._card_cache.clear()
        model_cards_module._card_cache.update(original_cache)


@pytest.mark.asyncio
async def test_registry_refresh_helper_throttles_repeated_cache_misses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown model requests cannot refresh TUF more than once per interval."""
    refreshes = 0

    async def refresh() -> None:
        nonlocal refreshes
        refreshes += 1
        monkeypatch.setattr(model_cards_module, "_last_registry_refresh", 100.0)

    original_cache = dict(model_cards_module._card_cache)
    model_cards_module._card_cache.clear()
    catalog = RegistryCatalog.model_validate_json(_catalog_payload(), strict=False)
    card = registry_model_cards(catalog)[0]
    model_cards_module._card_cache[card.model_id] = card
    monkeypatch.setattr(model_cards_module, "_registry_enabled", lambda: True)
    monkeypatch.setattr(model_cards_module, "_last_registry_refresh", 0.0)
    monkeypatch.setattr(model_cards_module.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(model_cards_module, "_refresh_card_cache", refresh)
    try:
        await model_cards_module._refresh_card_cache_if_due()
        await model_cards_module._refresh_card_cache_if_due()
        assert refreshes == 1
    finally:
        model_cards_module._card_cache.clear()
        model_cards_module._card_cache.update(original_cache)


@pytest.mark.asyncio
async def test_registry_id_miss_forces_one_serialized_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A store can bridge snapshot skew without allowing refresh storms."""
    refreshes = 0
    requested_id = f"card_{'z' * 52}"

    async def refresh() -> None:
        nonlocal refreshes
        refreshes += 1

    original_cache = dict(model_cards_module._card_cache)
    model_cards_module._card_cache.clear()
    catalog = RegistryCatalog.model_validate_json(_catalog_payload(), strict=False)
    model_cards_module._card_cache.update(
        {card.model_id: card for card in registry_model_cards(catalog)}
    )
    monkeypatch.setattr(model_cards_module, "_registry_enabled", lambda: True)
    monkeypatch.setattr(model_cards_module, "_last_registry_refresh", 100.0)
    monkeypatch.setattr(model_cards_module, "_last_registry_miss_refresh", 0.0)
    monkeypatch.setattr(model_cards_module.time, "monotonic", lambda: 101.0)
    monkeypatch.setattr(model_cards_module, "_refresh_card_cache", refresh)
    try:
        assert (
            await model_cards_module.get_registry_card_by_id(
                requested_id,
                refresh_on_miss=True,
            )
            is None
        )
        assert (
            await model_cards_module.get_registry_card_by_id(
                requested_id,
                refresh_on_miss=True,
            )
            is None
        )
        assert refreshes == 1
    finally:
        model_cards_module._card_cache.clear()
        model_cards_module._card_cache.update(original_cache)


@pytest.mark.asyncio
async def test_current_registry_id_is_visible_behind_installed_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Store authorization can resolve an update hidden by installed alias truth."""

    catalog = RegistryCatalog.model_validate_json(_catalog_payload(), strict=False)
    current = registry_model_cards(catalog)[0]
    installed = current.model_copy(
        update={"registry_card_id": f"card_{'z' * 52}"}
    )
    original_cache = dict(model_cards_module._card_cache)
    original_current = dict(model_cards_module._registry_current_cards)
    model_cards_module._card_cache.clear()
    model_cards_module._registry_current_cards.clear()
    model_cards_module._card_cache[current.model_id] = installed
    model_cards_module._registry_current_cards[current.model_id] = current
    monkeypatch.setattr(model_cards_module, "_registry_enabled", lambda: False)
    try:
        assert (
            await model_cards_module.get_registry_card_by_id(
                str(current.registry_card_id)
            )
            == current
        )
    finally:
        model_cards_module._card_cache.clear()
        model_cards_module._card_cache.update(original_cache)
        model_cards_module._registry_current_cards.clear()
        model_cards_module._registry_current_cards.update(original_current)


def test_installed_startup_selects_current_generation_deterministically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Directory order cannot replace current installed truth with stale bytes."""

    catalog = RegistryCatalog.model_validate_json(_catalog_payload(), strict=False)
    current = registry_model_cards(catalog)[0]
    stale = current.model_copy(
        update={"registry_card_id": f"card_{'z' * 52}"}
    )
    for directory_name, card in (("aaa-current", current), ("zzz-stale", stale)):
        artifact = tmp_path / directory_name
        artifact.mkdir()
        (artifact / "model-Q4_K_M.gguf").write_bytes(directory_name.encode())
        (artifact / ".skulk-source-revision").write_text(
            f"{card.source_revision}\n"
        )
        write_installed_card(artifact, build_installed_card_record(artifact, card))

    original_cache = dict(model_cards_module._card_cache)
    original_installed = dict(model_cards_module._installed_card_cache)
    original_installed_current = dict(
        model_cards_module._installed_current_registry_ids
    )
    original_current = dict(model_cards_module._registry_current_cards)
    model_cards_module._card_cache.clear()
    model_cards_module._installed_card_cache.clear()
    model_cards_module._installed_current_registry_ids.clear()
    model_cards_module._registry_current_cards.clear()
    model_cards_module._registry_current_cards[current.model_id] = current
    monkeypatch.setattr(constants_module, "SKULK_MODELS_DIR", tmp_path)
    monkeypatch.setattr(constants_module, "SKULK_MODELS_PATH", None)
    try:
        model_cards_module._load_cards_from_installed_artifacts()

        assert model_cards_module.get_card(current.model_id) == current
        installed = model_cards_module.get_installed_card_record(current.model_id)
        assert installed is not None
        assert installed.installed_identity == current.registry_card_id
        assert (
            model_cards_module.get_current_registry_card_id(current.model_id)
            == current.registry_card_id
        )
    finally:
        model_cards_module._card_cache.clear()
        model_cards_module._card_cache.update(original_cache)
        model_cards_module._installed_card_cache.clear()
        model_cards_module._installed_card_cache.update(original_installed)
        model_cards_module._installed_current_registry_ids.clear()
        model_cards_module._installed_current_registry_ids.update(
            original_installed_current
        )
        model_cards_module._registry_current_cards.clear()
        model_cards_module._registry_current_cards.update(original_current)


@pytest.mark.asyncio
async def test_downloader_fetches_source_repository_under_alias_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Network origin uses artifact truth while local state keeps the alias."""
    observed: list[ModelId] = []

    async def fake_models_dir() -> Path:
        return tmp_path

    async def fake_file_list(
        model_id: ModelId, *_: object, **__: object
    ) -> list[object]:
        observed.append(model_id)
        return []

    async def ignore_progress(*_: object) -> None:
        pass

    monkeypatch.setattr(download_utils, "ensure_models_dir", fake_models_dir)
    monkeypatch.setattr(download_utils, "fetch_file_list_with_cache", fake_file_list)
    card = ModelCard(
        model_id=ModelId("org/multi@q4"),
        source_repository=ModelId("org/multi"),
        source_revision="b" * 40,
        storage_size=Memory.from_bytes(1024),
        n_layers=4,
        hidden_size=64,
        supports_tensor=False,
        tasks=[ModelTask.TextGeneration],
    )
    shard = PipelineShardMetadata(
        model_card=card,
        device_rank=0,
        world_size=1,
        start_layer=0,
        end_layer=4,
        n_layers=4,
    )

    path, _ = await download_utils.download_shard(
        shard,
        ignore_progress,
        skip_download=True,
        skip_internet=True,
        allow_patterns=["config.json"],
    )

    assert observed == [ModelId("org/multi")]
    assert path.name == "org--multi@q4"
