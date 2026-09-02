"""``POST /models/add`` failures must name the operator's actual fix.

A gated or private Hugging Face repository fails the Add flow at the Hub
metadata fetch with a 401/403 whose raw text tells the operator to "log in",
which is the wrong remediation for a Skulk node. ``_describe_hf_fetch_failure``
routes those statuses through the download layer's auth explainer (configure a
token, accept the model terms, match the accepting account to the token) and
passes every other failure through unchanged.
"""

import pytest

from skulk.api.main import API
from skulk.shared.types.common import ModelId

_MODEL = ModelId("meta-llama/Llama-3.2-1B-Instruct")


class _FakeResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class _FakeHubError(Exception):
    def __init__(self, message: str, status_code: int | None) -> None:
        super().__init__(message)
        self.response = (
            _FakeResponse(status_code) if status_code is not None else None
        )


@pytest.mark.parametrize("status_code", [401, 403])
async def test_auth_status_routes_through_explainer(
    monkeypatch: pytest.MonkeyPatch, status_code: int
) -> None:
    async def fake_explainer(code: int, model_id: ModelId) -> str:
        assert code == status_code
        assert model_id == _MODEL
        return f"explained {code} for {model_id}"

    monkeypatch.setattr(
        "skulk.download.download_utils.build_auth_error_message",
        fake_explainer,
    )

    detail = await API._describe_hf_fetch_failure(  # pyright: ignore[reportPrivateUsage]
        _MODEL, _FakeHubError("401 Client Error: please log in", status_code)
    )

    assert detail == f"explained {status_code} for {_MODEL}"


async def test_auth_explainer_names_the_remediation() -> None:
    """Unpatched, a 403 without a token points at terms + token setup."""
    detail = await API._describe_hf_fetch_failure(  # pyright: ignore[reportPrivateUsage]
        _MODEL, _FakeHubError("403 Client Error", 403)
    )

    assert str(_MODEL) in detail
    assert "log in" not in detail


async def test_non_auth_errors_pass_through() -> None:
    detail = await API._describe_hf_fetch_failure(  # pyright: ignore[reportPrivateUsage]
        _MODEL, _FakeHubError("500 Server Error", 500)
    )
    assert detail == "Failed to fetch model: 500 Server Error"


async def test_errors_without_response_pass_through() -> None:
    detail = await API._describe_hf_fetch_failure(  # pyright: ignore[reportPrivateUsage]
        _MODEL, ValueError("no GGUF weights")
    )
    assert detail == "Failed to fetch model: no GGUF weights"
