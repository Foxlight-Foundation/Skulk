# pyright: reportPrivateUsage=false
"""Stored-image URL construction must match the registered fetch route."""

from types import SimpleNamespace
from typing import cast

from starlette.requests import Request

from skulk.api.main import API
from skulk.shared.types.common import Id


def _fake_request(host: str = "node.example:52415", scheme: str = "http") -> Request:
    return cast(
        "Request",
        cast(
            object,
            SimpleNamespace(
                headers={"host": host}, url=SimpleNamespace(scheme=scheme)
            ),
        ),
    )


def _fake_api() -> API:
    return cast("API", cast(object, SimpleNamespace(port=52415)))


def test_build_image_url_matches_the_registered_route() -> None:
    # The fetch route is GET /images/{image_id}; a /v1 prefix here returned
    # URLs that always 404ed (found by the 1.5.0 docs review).
    url = API._build_image_url(_fake_api(), _fake_request(), Id("img-123"))
    assert url == "http://node.example:52415/images/img-123"


def test_build_image_url_preserves_https() -> None:
    url = API._build_image_url(_fake_api(), _fake_request(scheme="https"), Id("img-9"))
    assert url.startswith("https://")
