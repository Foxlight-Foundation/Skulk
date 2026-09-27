"""The dashboard SPA's client routes must serve the app shell, not a 404.

The dashboard restores its active view from the URL path, so deep links and
browser refreshes hit these paths directly. StaticFiles(html=True) only
serves index.html at "/"; the API registers explicit fallbacks for the
client routes. Keep the route list here in sync with NavRoute in
dashboard-react/src/components/layout/HeaderNav.tsx.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from skulk.api.main import API
from skulk.shared.election import ElectionMessage
from skulk.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from skulk.shared.types.common import NodeId
from skulk.shared.types.events import IndexedEvent
from skulk.utils.channels import channel

SPA_ROUTES = (
    "/cluster",
    "/model-store",
    "/chat",
    "/steward",
    "/integrations",
    "/plugins",
    "/operator",
)


@contextmanager
def _client_with_dashboard(dashboard_dir: Path) -> Iterator[TestClient]:
    """Yield a TestClient whose API serves the dashboard from *dashboard_dir*.

    The patch stays active for the client's lifetime: the SPA fallback reads
    ``DASHBOARD_DIR`` at request time, not at construction.
    """
    (dashboard_dir / "index.html").write_text(
        "<!doctype html><title>skulk-test-shell</title>"
    )
    command_sender, _ = channel[ForwarderCommand]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    _, event_receiver = channel[IndexedEvent]()
    _, election_receiver = channel[ElectionMessage]()
    with patch("skulk.api.main.DASHBOARD_DIR", str(dashboard_dir)):
        api = API(
            NodeId("test-node"),
            port=52415,
            event_receiver=event_receiver,
            command_sender=command_sender,
            download_command_sender=download_sender,
            election_receiver=election_receiver,
            enable_event_log=False,
            mount_dashboard=True,
        )
        yield TestClient(api.app)


def test_spa_client_routes_serve_the_app_shell(tmp_path: Path) -> None:
    with _client_with_dashboard(tmp_path) as client:
        # Both slash forms: the StaticFiles mount at "/" swallows unmatched
        # paths before FastAPI's redirect-slashes logic can normalize.
        for route in [*SPA_ROUTES, *(f"{r}/" for r in SPA_ROUTES)]:
            response = client.get(route)
            assert response.status_code == 200, route
            assert response.headers["content-type"].startswith("text/html"), route
            assert "skulk-test-shell" in response.text, route


def test_app_shell_is_revalidated_on_every_load(tmp_path: Path) -> None:
    # The shell names the build's hashed bundles; a cached shell after an
    # update asks for bundles that no longer exist and renders a blank page.
    with _client_with_dashboard(tmp_path) as client:
        for route in ["/", *SPA_ROUTES, *(f"{r}/" for r in SPA_ROUTES)]:
            response = client.get(route)
            assert response.status_code == 200, route
            assert response.headers["cache-control"] == "no-cache", route


def test_hashed_bundles_are_immutable_and_other_files_revalidate(
    tmp_path: Path,
) -> None:
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "index-Ab12Cd34.js").write_text("console.log(1)")
    (tmp_path / "favicon.svg").write_text("<svg/>")
    with _client_with_dashboard(tmp_path) as client:
        bundle = client.get("/assets/index-Ab12Cd34.js")
        assert bundle.status_code == 200
        assert bundle.headers["cache-control"] == "public, max-age=31536000, immutable"
        favicon = client.get("/favicon.svg")
        assert favicon.status_code == 200
        assert favicon.headers["cache-control"] == "no-cache"
        # A bundle from a replaced build is simply gone.
        assert client.get("/assets/index-Old00000.js").status_code == 404


def test_revalidated_bundles_stay_immutable(tmp_path: Path) -> None:
    # A cache replaces its stored headers with the 304's, so a bundle
    # revalidated once must come back immutable, not demoted to no-cache.
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "index-Ab12Cd34.js").write_text("console.log(1)")
    with _client_with_dashboard(tmp_path) as client:
        first = client.get("/assets/index-Ab12Cd34.js")
        revalidated = client.get(
            "/assets/index-Ab12Cd34.js",
            headers={"If-None-Match": first.headers["etag"]},
        )
        assert revalidated.status_code == 304
        assert (
            revalidated.headers["cache-control"]
            == "public, max-age=31536000, immutable"
        )
        shell_etag = client.get("/").headers["etag"]
        shell = client.get("/", headers={"If-None-Match": shell_etag})
        assert shell.status_code == 304
        assert shell.headers["cache-control"] == "no-cache"


def test_unknown_path_still_404s(tmp_path: Path) -> None:
    # The fallback is scoped to the known client routes — arbitrary paths
    # must keep 404ing so typos and probes don't silently render the app.
    with _client_with_dashboard(tmp_path) as client:
        assert client.get("/definitely-not-a-route").status_code == 404


def test_headless_api_boots_without_dashboard_assets() -> None:
    # A headless/non-Mac node has no built dashboard (DASHBOARD_DIR is None).
    # mount_dashboard=True must then NOT register the StaticFiles mount or the
    # SPA fallbacks, and the API must still boot and serve (#333).
    command_sender, _ = channel[ForwarderCommand]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    _, event_receiver = channel[IndexedEvent]()
    _, election_receiver = channel[ElectionMessage]()
    with patch("skulk.api.main.DASHBOARD_DIR", None):
        api = API(
            NodeId("test-node"),
            port=52415,
            event_receiver=event_receiver,
            command_sender=command_sender,
            download_command_sender=download_sender,
            election_receiver=election_receiver,
            enable_event_log=False,
            mount_dashboard=True,
        )
        # The dashboard StaticFiles mount is absent.
        assert "dashboard" not in {
            getattr(route, "name", None) for route in api.app.routes
        }
        client = TestClient(api.app)
        # No SPA fallback and no "/" mount when assets are missing.
        assert client.get("/cluster").status_code == 404
