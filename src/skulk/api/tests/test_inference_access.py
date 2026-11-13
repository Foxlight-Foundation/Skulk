"""Real HTTP owner/client authentication and protected inference enrollment."""

from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI, Request
from pydantic import JsonValue, TypeAdapter

from skulk.api.inference_access import (
    ClientCreated,
    authenticate_inference_request,
    create_inference_access_router,
)
from skulk.extensions.inference_callers import InferenceCallers
from skulk.extensions.runtime_attachment import HostSettings, ServiceConnection
from skulk.extensions.runtime_files import write_private
from skulk.shared.types.common import NodeId


def application(root: Path) -> tuple[FastAPI, InferenceCallers]:
    """Use production routers and real custody, with a test-only identity observer."""
    write_private(
        root / "host.json",
        HostSettings(profile_id="a" * 32, transport_node_id="access-core")
        .model_dump_json()
        .encode(),
    )
    write_private(
        root / "connection.json",
        ServiceConnection(manager_root=str(root), profile_id="a" * 32)
        .model_dump_json()
        .encode(),
    )
    service = InferenceCallers(root, "a" * 32, "access-core")
    app = FastAPI()
    app.include_router(
        create_inference_access_router(
            NodeId("access-core"), connection_path=root / "connection.json"
        )
    )

    async def caller(request: Request) -> dict[str, JsonValue]:
        authenticated = await authenticate_inference_request(
            request, service, NodeId("access-core")
        )
        return authenticated.principal.model_dump(mode="json")

    app.get("/caller")(caller)
    return app, service


def client_for(
    app: FastAPI,
    *,
    peer: str = "127.0.0.1",
    scheme: str = "http",
    dashboard: bool = True,
) -> httpx.AsyncClient:
    """Create a real ASGI socket/origin boundary without authentication stubs."""
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, client=(peer, 1)),
        base_url=f"{scheme}://localhost",
        headers={"origin": f"{scheme}://localhost", "x-skulk-dashboard": "pairing-v1"}
        if dashboard
        else {},
    )


async def test_owner_can_enroll_once_list_metadata_and_revoke(tmp_path: Path) -> None:
    """The bearer exists only in the creation response; revocation invalidates it."""
    app, service = application(tmp_path)
    async with client_for(app) as client:
        listed = await client.get("/v1/inference-access/clients")
        assert listed.status_code == 200 and listed.json()["clients"] == []
        created = await client.post(
            "/v1/inference-access/clients",
            json={
                "expected_revision": listed.json()["revision"],
                "label": "AnythingLLM",
            },
        )
        assert created.status_code == 201
        assert created.headers["cache-control"] == "no-store"
        issued = ClientCreated.model_validate_json(created.content)
        token = issued.token
        identifier = issued.client.identifier
        assert (
            service.authenticate(token).principal.identifier == f"client:{identifier}"
        )
        visible = await client.get("/v1/inference-access/clients")
        assert (
            token not in visible.text
            and "token_digest" not in visible.text
            and "digest_key" not in visible.text
        )
        original = await client.get("/caller")
        assert original.json()["identifier"] == "direct-owner"
        bearer = await client.get(
            "/caller",
            headers={"authorization": f"Bearer {token}", "x-caller-id": "direct-owner"},
        )
        assert bearer.json()["identifier"] == f"client:{identifier}"
        assert bearer.json()["kind"] == "client"
        stale = await client.post(
            "/v1/inference-access/clients",
            json={"expected_revision": listed.json()["revision"], "label": "Stale"},
        )
        assert stale.status_code == 409
        revoked = await client.request(
            "DELETE",
            f"/v1/inference-access/clients/{identifier}",
            json={"expected_revision": visible.json()["revision"]},
        )
        assert revoked.status_code == 200 and revoked.json()["clients"] == []
        denied = await client.get(
            "/caller", headers={"authorization": f"Bearer {token}"}
        )
        assert denied.status_code == 401
        assert service.policy().enabled is False


@pytest.mark.parametrize(
    "fault",
    ["invalid", "duplicate", "empty", "paired", "forwarded", "remote-cleartext"],
)
async def test_invalid_client_never_falls_back_to_owner(
    tmp_path: Path, fault: str
) -> None:
    """Real credentials cannot be substituted with caller text or unsafe transport."""
    app, service = application(tmp_path)
    _, token = service.create(service.inventory().revision, "API", None)
    auth = f"Bearer {token}"
    headers: list[tuple[str, str]] = [("authorization", auth)]
    if fault == "invalid":
        headers = [("authorization", "Bearer invalid")]
    elif fault == "duplicate":
        headers.append(("authorization", auth))
    elif fault == "empty":
        headers = [("authorization", "")]
    elif fault == "paired":
        headers = [("authorization", "Bearer operator-credential")]
    elif fault == "forwarded":
        headers.append(("x-forwarded-for", "127.0.0.1"))
    async with client_for(
        app, peer="192.0.2.1" if fault == "remote-cleartext" else "127.0.0.1"
    ) as client:
        result = await client.get("/caller", headers=headers)
    assert result.status_code in {401, 403}


async def test_client_and_unverified_browser_cannot_administer_enrollment(
    tmp_path: Path,
) -> None:
    """Execution authentication never grants owner management access."""
    app, service = application(tmp_path)
    _, token = service.create(service.inventory().revision, "API", None)
    async with client_for(app) as client:
        result = await client.get(
            "/v1/inference-access/clients", headers={"authorization": f"Bearer {token}"}
        )
        assert result.status_code == 403
    async with client_for(app, dashboard=False) as client:
        result = await client.get("/v1/inference-access/clients")
        assert result.status_code == 403
    async with client_for(app, peer="192.0.2.1") as client:
        result = await client.get("/v1/inference-access/clients")
        assert result.status_code == 403


async def test_bearer_requires_no_browser_origin_on_https(tmp_path: Path) -> None:
    """Stock clients use a standard bearer over a protected direct transport."""
    app, service = application(tmp_path)
    _, token = service.create(service.inventory().revision, "API", None)
    async with client_for(
        app, peer="192.0.2.1", scheme="https", dashboard=False
    ) as client:
        result = await client.get(
            "/caller", headers={"authorization": f"Bearer {token}"}
        )
    assert result.status_code == 200 and result.json()["kind"] == "client"


async def test_management_authentication_precedes_body_parsing_and_bounds_it(
    tmp_path: Path,
) -> None:
    """An unauthenticated caller cannot force parsing; an owner has a fixed byte cap."""
    app, service = application(tmp_path)
    async with client_for(app, dashboard=False) as client:
        refused = await client.post(
            "/v1/inference-access/clients",
            content=b"{" * 150000,
            headers={"content-type": "application/json"},
        )
        assert refused.status_code == 403
    async with client_for(app) as client:
        oversized = await client.post(
            "/v1/inference-access/clients",
            content=b"{" * 150000,
            headers={"content-type": "application/json"},
        )
        assert oversized.status_code == 413
        malformed = await client.post(
            "/v1/inference-access/clients",
            json={
                "expected_revision": service.inventory().revision,
                "label": "API",
                "expires_unix_ms": True,
            },
        )
        assert malformed.status_code == 422
    assert service.inventory().clients == ()


async def test_policy_has_distinct_default_deny_and_cas(tmp_path: Path) -> None:
    """Policy administration stays separate from credentials and legacy read enrollment."""
    app, _ = application(tmp_path)
    async with client_for(app) as client:
        initial = await client.get("/v1/inference-access/policy")
        assert initial.status_code == 200 and initial.json()["enabled"] is False
        updated = await client.put(
            "/v1/inference-access/policy",
            json={
                "expected_revision": initial.json()["revision"],
                "enabled": True,
                "callers": [],
            },
        )
        assert updated.status_code == 200 and updated.json()["callers"] == []
        stale = await client.put(
            "/v1/inference-access/policy",
            json={
                "expected_revision": initial.json()["revision"],
                "enabled": False,
                "callers": [],
            },
        )
        assert stale.status_code == 409
    schema = TypeAdapter(dict[str, JsonValue]).validate_python(app.openapi())
    paths = TypeAdapter(dict[str, dict[str, dict[str, JsonValue]]]).validate_python(
        schema["paths"]
    )
    for path, methods in paths.items():
        if path.startswith("/v1/inference-access"):
            assert all(
                operation["summary"] and operation["description"] and operation["tags"]
                for operation in methods.values()
            )
