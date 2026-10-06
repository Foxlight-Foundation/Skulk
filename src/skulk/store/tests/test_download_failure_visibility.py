"""Failed store downloads must stay visible with their recorded reason.

``list_active_downloads`` feeds ``GET /store/downloads``, the dashboard's only
window into store transfer activity. It used to filter to pending/downloading,
so a failed download (for example a gated Hugging Face repository without a
token) silently vanished from the UI along with the actionable ``error``
explanation the store had just recorded. These tests pin the listing contract:
pending, downloading, and failed entries are reported; cancelled and complete
entries are not.
"""

from pathlib import Path

from skulk.store.model_store import ModelStore, StoreDownloadStatus


def _seed(store: ModelStore, model_id: str, status: str, error: str | None = None) -> None:
    store._active_downloads[model_id] = StoreDownloadStatus(  # pyright: ignore[reportPrivateUsage]
        model_id=model_id,
        status=status,  # pyright: ignore[reportArgumentType]
        progress=0.5,
        error=error,
    )


def test_failed_downloads_stay_listed_with_reason(tmp_path: Path) -> None:
    store = ModelStore(tmp_path)
    _seed(store, "org/pending", "pending")
    _seed(store, "org/live", "downloading")
    _seed(
        store,
        "org/gated",
        "failed",
        error=(
            "HuggingFaceAuthenticationError: Access to 'org/gated' is "
            "restricted and this node sent no Hugging Face token."
        ),
    )
    _seed(store, "org/cancelled", "cancelled")
    _seed(store, "org/done", "complete")

    listed = {s.model_id: s for s in store.list_active_downloads()}

    assert set(listed) == {"org/pending", "org/live", "org/gated"}
    failed = listed["org/gated"]
    assert failed.status == "failed"
    assert failed.error is not None
    assert "Hugging Face token" in failed.error


def test_empty_store_lists_nothing(tmp_path: Path) -> None:
    store = ModelStore(tmp_path)
    assert store.list_active_downloads() == []
