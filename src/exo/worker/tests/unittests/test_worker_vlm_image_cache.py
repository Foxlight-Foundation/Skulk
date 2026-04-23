from exo.worker.main import resolve_cached_vlm_images


def test_resolve_cached_vlm_images_reports_missing_hashes() -> None:
    resolved, missing = resolve_cached_vlm_images(
        {"hash-a": "image-a"},
        {0: "hash-a", 1: "hash-b"},
    )

    assert resolved == {0: "image-a"}
    assert missing == [(1, "hash-b")]
