"""Keep the public route inventory, generated schemas, and guide in agreement."""

import re
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute, APIWebSocketRoute
from pydantic import JsonValue, TypeAdapter

from skulk.api.main import API
from skulk.operator.authority import EncryptedAuthorityStore
from skulk.operator.key_provider import LocalFileAuthorityKeyProvider
from skulk.operator.pairing import OperatorPairingService
from skulk.shared.election import ElectionMessage
from skulk.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from skulk.shared.types.common import NodeId
from skulk.shared.types.events import IndexedEvent
from skulk.utils.channels import channel

# These operations intentionally accept no request document. New bodyless verbs
# need an explicit decision here, so a raw Request handler cannot accidentally
# lose its public request schema without making the omission visible in review.
_BODYLESS_MUTATIONS = {
    ("POST", "/models/remote-code-approvals/{card_id}"),
    ("POST", "/v1/videos/{video_id}/cancel"),
    ("POST", "/v1/cancel/{command_id}"),
    ("POST", "/onboarding"),
    ("POST", "/store/reconciliation/rescan"),
    ("POST", "/admin/restart"),
    (
        "POST",
        "/v1/plugins/managed/installations/{plugin_id}/operations/{operation_id}/recover",
    ),
}


@pytest.fixture
def documented_app(tmp_path: Path) -> FastAPI:
    """Build all public routes without listeners, dashboard assets, or host keys."""
    command_sender, _ = channel[ForwarderCommand]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    _, event_receiver = channel[IndexedEvent]()
    _, election_receiver = channel[ElectionMessage]()
    provider = LocalFileAuthorityKeyProvider(tmp_path / "key.bin")
    pairing = OperatorPairingService(
        EncryptedAuthorityStore(provider, tmp_path / "authority.sqlite3"), provider
    )
    return API(
        NodeId("documentation-test"),
        port=52415,
        event_receiver=event_receiver,
        command_sender=command_sender,
        download_command_sender=download_sender,
        election_receiver=election_receiver,
        enable_event_log=False,
        mount_dashboard=False,
        operator_pairing_service=pairing,
    ).app


def test_every_http_operation_has_a_complete_openapi_entry(
    documented_app: FastAPI,
) -> None:
    """Every registered API verb must retain its schema and discovery metadata."""
    schema = TypeAdapter(dict[str, JsonValue]).validate_python(documented_app.openapi())
    paths = schema["paths"]
    assert isinstance(paths, dict)
    registered: set[tuple[str, str]] = set()
    for route in documented_app.routes:
        if not isinstance(route, APIRoute):
            continue
        assert route.include_in_schema, f"Hidden API route: {route.path}"
        path_item = paths.get(route.path_format)
        assert isinstance(path_item, dict), route.path
        for method in route.methods:
            registered.add((method.lower(), route.path_format))
            operation = path_item.get(method.lower())
            assert isinstance(operation, dict), (method, route.path)
            for field in ("summary", "description", "tags", "responses"):
                assert operation.get(field), (method, route.path, field)

    published = {
        (method, path)
        for path, path_item in paths.items()
        if isinstance(path_item, dict)
        for method in path_item
        if method in {"get", "post", "put", "patch", "delete", "head", "options"}
    }
    assert published == registered


def test_mutation_request_schemas_are_not_silently_omitted(
    documented_app: FastAPI,
) -> None:
    """Raw Request handlers need explicit schemas or a reviewed bodyless entry."""
    schema = TypeAdapter(dict[str, JsonValue]).validate_python(documented_app.openapi())
    paths = schema["paths"]
    assert isinstance(paths, dict)
    bodyless: set[tuple[str, str]] = set()
    for path, path_item in paths.items():
        assert isinstance(path_item, dict)
        for method in ("post", "put", "patch"):
            operation = path_item.get(method)
            if not isinstance(operation, dict):
                continue
            request_body = operation.get("requestBody")
            if request_body is None:
                bodyless.add((method.upper(), path))
                continue
            assert isinstance(request_body, dict), (method, path)
            content = request_body.get("content")
            assert isinstance(content, dict) and content, (method, path)
            for media_type, media in content.items():
                assert isinstance(media, dict) and media.get("schema"), (
                    method,
                    path,
                    media_type,
                )
    assert bodyless == _BODYLESS_MUTATIONS


def test_public_route_paths_are_present_in_the_manual_guide(
    documented_app: FastAPI,
) -> None:
    """Require exact normalized HTTP and WebSocket paths, not particular prose."""
    repository = Path(__file__).resolve().parents[4]
    guide = (repository / "website/docs/api-guide.md").read_text()
    documented_paths = set(
        re.findall(r"`(?:[A-Z]+\s+)?(/[^`\s?]+)(?:[?][^`]*)?`", guide)
    )
    for route in documented_app.routes:
        if isinstance(route, APIRoute | APIWebSocketRoute):
            assert route.path_format in documented_paths, route.path_format


async def test_all_references_resolve_in_live_openapi(
    documented_app: FastAPI,
) -> None:
    """Resolve every reference against the HTTP schema without exporter repair."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=documented_app),
        base_url="http://localhost",
    ) as client:
        response = await client.get("/api/openapi.json")
    assert response.status_code == 200
    schema = TypeAdapter(dict[str, JsonValue]).validate_json(response.content)
    pending: list[JsonValue] = [schema]
    visited: set[str] = set()
    while pending:
        value = pending.pop()
        if isinstance(value, list):
            pending.extend(value)
        elif isinstance(value, dict):
            reference = value.get("$ref")
            if isinstance(reference, str) and reference not in visited:
                assert reference.startswith("#/"), reference
                visited.add(reference)
                target: JsonValue = schema
                for encoded_part in reference[2:].split("/"):
                    part = encoded_part.replace("~1", "/").replace("~0", "~")
                    assert isinstance(target, dict) and part in target, reference
                    target = target[part]
                pending.append(target)
            pending.extend(value.values())


def test_config_request_retains_direct_and_wrapped_nested_fields(
    documented_app: FastAPI,
) -> None:
    """Inlining must retain configuration structure, constraints, and both forms."""
    schema = TypeAdapter(dict[str, JsonValue]).validate_python(documented_app.openapi())
    value: JsonValue = schema
    for key in (
        "paths",
        "/config",
        "put",
        "requestBody",
        "content",
        "application/json",
        "schema",
    ):
        assert isinstance(value, dict)
        value = value[key]
    assert isinstance(value, dict)
    alternatives = value["anyOf"]
    assert isinstance(alternatives, list) and len(alternatives) == 2
    direct, wrapped = alternatives
    assert isinstance(direct, dict) and isinstance(wrapped, dict)
    wrapped_properties = wrapped["properties"]
    assert isinstance(wrapped_properties, dict)
    assert wrapped["required"] == ["config"]
    assert wrapped_properties["config"] == direct
    properties = direct["properties"]
    assert isinstance(properties, dict)
    intelligent_fabric = properties["intelligentFabric"]
    assert isinstance(intelligent_fabric, dict)
    variants = intelligent_fabric["anyOf"]
    assert isinstance(variants, list)
    settings = variants[0]
    assert isinstance(settings, dict) and settings["additionalProperties"] is False
    settings_properties = settings["properties"]
    assert isinstance(settings_properties, dict)
    enabled = settings_properties["enabled"]
    assert isinstance(enabled, dict)
    assert enabled["type"] == "boolean" and enabled["default"] is False
