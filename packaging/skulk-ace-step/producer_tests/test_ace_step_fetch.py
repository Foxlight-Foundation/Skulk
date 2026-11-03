"""Selected immutable archive intake with real cache files and controlled transport."""

from __future__ import annotations

import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.request import Request

import fetch_archives
from build_ace_step import ArchiveInventory
from fetch_archives import fetch_object, fetch_profile
from test_ace_step_producer import fixture


class ArchiveFetchTests(unittest.TestCase):
    """A network success or matching filename cannot replace immutable object truth."""

    def setUp(self) -> None:
        """Give each intake case an exclusive small local cache."""
        self.temporary = tempfile.TemporaryDirectory(prefix="native-ace-intake-")
        self.root = Path(self.temporary.name)
        self.destination = self.root / "object.whl"
        self.content = b"selected immutable bytes"
        self.digest = hashlib.sha256(self.content).hexdigest()

    def tearDown(self) -> None:
        """Remove only the cache owned by this test."""
        self.temporary.cleanup()

    def test_download_and_verified_cache_reuse_preserve_the_same_inode(self) -> None:
        """Cache reuse verifies existing bytes without performing a second request."""

        def network(request: Request, *, timeout: int) -> io.BytesIO:
            self.assertEqual(
                request.full_url, "https://files.pythonhosted.org/selected.whl"
            )
            self.assertEqual(timeout, 120)
            return io.BytesIO(self.content)

        with patch.object(fetch_archives, "urlopen", network):
            fetch_object(
                "https://files.pythonhosted.org/selected.whl",
                self.destination,
                len(self.content),
                self.digest,
            )
        inode = self.destination.stat().st_ino

        def forbidden(request: Request, *, timeout: int) -> io.BytesIO:
            raise AssertionError("verified cache reuse must not download")

        with patch.object(fetch_archives, "urlopen", forbidden):
            fetch_object(
                "https://files.pythonhosted.org/selected.whl",
                self.destination,
                len(self.content),
                self.digest,
            )
        self.assertEqual(self.destination.stat().st_ino, inode)
        self.assertEqual(self.destination.read_bytes(), self.content)

    def test_wrong_and_overlimit_network_objects_leave_no_published_or_partial_file(
        self,
    ) -> None:
        """An interrupted/bad object cannot masquerade as a cached selection."""
        for content in (b"x" * len(self.content), self.content + b"too large"):

            def network(
                request: Request, *, timeout: int, selected: bytes = content
            ) -> io.BytesIO:
                return io.BytesIO(selected)

            with (
                self.subTest(content=content),
                patch.object(fetch_archives, "urlopen", network),
                self.assertRaises(ValueError),
            ):
                fetch_object(
                    "https://files.pythonhosted.org/selected.whl",
                    self.destination,
                    len(self.content),
                    self.digest,
                )
            self.assertFalse(self.destination.exists())
            self.assertFalse(self.destination.with_name("object.whl.part").exists())

    def test_changed_existing_cache_is_not_overwritten(self) -> None:
        """Preserve unexpected bytes for diagnosis instead of silently repairing a cache."""
        changed = b"x" * len(self.content)
        self.destination.write_bytes(changed)
        with self.assertRaisesRegex(ValueError, "cached native archive differs"):
            fetch_object(
                "https://files.pythonhosted.org/selected.whl",
                self.destination,
                len(self.content),
                self.digest,
            )
        self.assertEqual(self.destination.read_bytes(), changed)

    def test_another_owners_partial_is_never_removed(self) -> None:
        """Exclusive staging failure cannot clean another concurrent intake's work."""
        partial = self.destination.with_name("object.whl.part")
        partial.write_bytes(b"other owner")
        with self.assertRaises(FileExistsError):
            fetch_object(
                "https://files.pythonhosted.org/selected.whl",
                self.destination,
                len(self.content),
                self.digest,
            )
        self.assertEqual(partial.read_bytes(), b"other owner")

    def test_selected_profile_uses_exact_urls_without_index_resolution(self) -> None:
        """The intake receipt binds exactly its committed wheel, lock and interpreter."""
        inputs = fixture(self.root / "source-fixtures")
        source = inputs.wrapper.parent
        inventory = ArchiveInventory.model_validate_json(
            (source / "archives/macos-arm64-cpu-py312.json").read_bytes()
        )
        responses = {
            inventory.objects[0].source_url: inputs.wheel.read_bytes(),
            inventory.python_archive.url: inputs.interpreter.read_bytes(),
        }
        observed: list[str] = []

        def network(request: Request, *, timeout: int) -> io.BytesIO:
            observed.append(request.full_url)
            return io.BytesIO(responses[request.full_url])

        output = self.root / "intake"
        with patch.object(fetch_archives, "urlopen", network):
            fetch_profile(source, "macos-arm64", output)
        self.assertEqual(set(observed), set(responses))
        self.assertEqual(
            ArchiveInventory.model_validate_json(
                (output / "wheel-receipt.json").read_bytes()
            ),
            inventory,
        )
        self.assertEqual(
            (output / "wheels" / inputs.wheel.name).read_bytes(),
            inputs.wheel.read_bytes(),
        )
        self.assertEqual(
            (output / "python" / inputs.interpreter.name).read_bytes(),
            inputs.interpreter.read_bytes(),
        )
