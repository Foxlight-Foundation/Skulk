# pyright: reportPrivateUsage=false
"""Engine wheel publication keeps digest-pinned filenames immutable."""

from importlib.util import module_from_spec, spec_from_file_location
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Protocol, cast

import pytest


class PublisherModule(Protocol):
    """Typed contract loaded from the standalone wheel publication script."""

    _sha256: Callable[[Path], str]
    _upload_wheels: Callable[[object, str, list[Path], dict[str, str]], None]


def _publisher() -> PublisherModule:
    """Load the CI-only script without adding it to the Skulk runtime package."""

    script = Path(__file__).resolve().parents[4] / "scripts" / "publish_wheel_index.py"
    spec = spec_from_file_location("wheel_publisher_under_test", script)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return cast(PublisherModule, cast(object, module))


class MissingWheelError(Exception):
    """The fake bucket has no object at the requested wheel key."""


class WheelBucket:
    """Minimal object store for the publisher collision contract."""

    exceptions = SimpleNamespace(NoSuchKey=MissingWheelError)

    def __init__(self, objects: dict[str, bytes] | None = None) -> None:
        self.objects = objects or {}
        self.uploaded: list[str] = []

    def get_object(self, **kwargs: str) -> dict[str, BytesIO]:
        """Return stored bytes or the fake missing-key error."""

        key = kwargs["Key"]
        if key not in self.objects:
            raise MissingWheelError(key)
        return {"Body": BytesIO(self.objects[key])}

    def upload_file(
        self, filename: str, bucket: str, key: str,
        **_kwargs: object,
    ) -> None:
        """Record one wheel write under its object key."""

        self.uploaded.append(key)
        self.objects[key] = Path(filename).read_bytes()


def test_wheel_publisher_refuses_a_digest_collision_before_any_upload(
    tmp_path: Path,
) -> None:
    """A rerun cannot replace a published wheel or partially publish a batch."""

    fresh = tmp_path / "fresh.whl"
    collision = tmp_path / "pinned.whl"
    fresh.write_bytes(b"new")
    collision.write_bytes(b"different")
    bucket = WheelBucket({"wheels/pinned.whl": b"original"})
    publisher = _publisher()
    checksums = {wheel.name: publisher._sha256(wheel) for wheel in (fresh, collision)}

    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        publisher._upload_wheels(bucket, "engine-bucket", [fresh, collision], checksums)

    assert bucket.uploaded == []
    assert bucket.objects["wheels/pinned.whl"] == b"original"


def test_wheel_publisher_skips_identical_bytes_and_uploads_new_names(
    tmp_path: Path,
) -> None:
    """Idempotent reruns leave pinned bytes intact while adding new wheels."""

    same = tmp_path / "same.whl"
    fresh = tmp_path / "fresh.whl"
    same.write_bytes(b"original")
    fresh.write_bytes(b"new")
    bucket = WheelBucket({"wheels/same.whl": b"original"})
    publisher = _publisher()
    checksums = {wheel.name: publisher._sha256(wheel) for wheel in (same, fresh)}

    publisher._upload_wheels(bucket, "engine-bucket", [same, fresh], checksums)

    assert bucket.uploaded == ["wheels/fresh.whl"]
    assert bucket.objects["wheels/same.whl"] == b"original"
