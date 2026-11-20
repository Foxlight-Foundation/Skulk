"""Real pre-write directory ownership checks without weights or model imports."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import check_support
from skulk_ace_step.custody import verify_lifetime, verify_request_directory


class NativeCustodyTests(unittest.TestCase):
    """Keep foreign, incomplete and replaced paths out of native write admission."""

    def setUp(self) -> None:
        """Observe real Core-side fixture inodes before any native operation."""
        self.temporary = tempfile.TemporaryDirectory(prefix="native-music-custody-")
        self.root = Path(self.temporary.name)
        self.config = check_support.configuration(self.root)

    def tearDown(self) -> None:
        """Remove this test's isolated files only after every assertion completed."""
        self.temporary.cleanup()

    def test_parent_created_request_can_be_reserved_during_model_loading(self) -> None:
        """Preallocated results do not confer permission to write before admission."""
        request = check_support.request(self.config)
        verify_lifetime(self.config, initial=True)
        self.assertEqual(
            verify_request_directory(self.config, request, empty=True),
            Path(request.output_directory),
        )

    def test_absent_result_is_not_created_by_the_native_process(self) -> None:
        """A plausible pathname cannot manufacture cancellation cleanup authority."""
        request = check_support.request(self.config)
        directory = Path(request.output_directory)
        directory.rmdir()
        with self.assertRaises(FileNotFoundError):
            verify_request_directory(self.config, request, empty=True)
        self.assertFalse(directory.exists())

    def test_replaced_result_inode_and_foreign_contents_are_preserved(self) -> None:
        """A matching path cannot transfer the parent's ownership to a new inode."""
        request = check_support.request(self.config)
        directory = Path(request.output_directory)
        retained = directory.with_name("retained-original")
        directory.rename(retained)
        directory.mkdir(mode=0o700)
        foreign = directory / "operator-note.txt"
        foreign.write_bytes(b"must survive")
        with self.assertRaisesRegex(ValueError, "Core-owned custody"):
            verify_request_directory(self.config, request, empty=False)
        self.assertEqual(foreign.read_bytes(), b"must survive")
        self.assertEqual(
            check_support.directory_identity(retained),
            request.output_directory_identity,
        )

    def test_same_inode_with_partial_output_cannot_be_replayed(self) -> None:
        """Correct ownership does not authorize overwriting an unsettled result."""
        request = check_support.request(self.config)
        output = Path(request.output_directory) / "result.wav"
        output.write_bytes(b"partial")
        with self.assertRaisesRegex(ValueError, "Core-owned custody"):
            verify_request_directory(self.config, request, empty=True)
        self.assertEqual(output.read_bytes(), b"partial")

    def test_symlink_to_an_owned_result_is_not_an_admitted_directory(self) -> None:
        """Even an inode-equivalent final symlink cannot change the write location."""
        request = check_support.request(self.config)
        directory = Path(request.output_directory)
        retained = directory.with_name("retained-original")
        directory.rename(retained)
        directory.symlink_to(retained, target_is_directory=True)
        with self.assertRaises(OSError):
            verify_request_directory(self.config, request, empty=True)
        self.assertTrue(directory.is_symlink())
        self.assertTrue(retained.is_dir())

    def test_private_modes_are_required_again_at_request_admission(self) -> None:
        """A permissive request directory is refused before any native output."""
        request = check_support.request(self.config)
        directory = Path(request.output_directory)
        directory.chmod(0o750)
        with self.assertRaisesRegex(ValueError, "Core-owned custody"):
            verify_request_directory(self.config, request, empty=True)
        self.assertFalse(tuple(directory.iterdir()))

    def test_replaced_lifetime_is_refused_without_touching_the_replacement(
        self,
    ) -> None:
        """A stale configuration cannot repurpose another native model's root."""
        lifetime = Path(self.config.directory)
        lifetime.rename(self.root / "retained-lifetime")
        lifetime.mkdir(mode=0o700)
        foreign = lifetime / "operator-note.txt"
        foreign.write_bytes(b"must survive")
        with self.assertRaisesRegex(ValueError, "Core-owned custody"):
            verify_lifetime(self.config, initial=True)
        self.assertEqual(foreign.read_bytes(), b"must survive")

    def test_parent_alias_resolving_to_the_same_owned_inodes_is_valid(self) -> None:
        """Platform temporary-directory aliases preserve real ownership identity."""
        alias = self.root / "alias"
        alias.symlink_to(self.root, target_is_directory=True)
        config = self.config.model_copy(update={"directory": str(alias / "model")})
        verify_lifetime(config, initial=True)

    def test_existing_loader_state_is_not_a_fresh_native_lifetime(self) -> None:
        """A failed initialization must retire instead of quietly reusing its view."""
        unexpected = Path(self.config.directory) / "loader" / "checkpoints"
        unexpected.mkdir(mode=0o700)
        with self.assertRaisesRegex(ValueError, "Core-owned custody"):
            verify_lifetime(self.config, initial=True)
        self.assertTrue(unexpected.is_dir())

    def test_replaced_loader_is_refused_before_model_initialization(self) -> None:
        """Retained requests do not authorize writes into another loader inode."""
        loader = Path(self.config.directory) / "loader"
        loader.rename(self.root / "retained-loader")
        loader.mkdir(mode=0o700)
        with self.assertRaisesRegex(ValueError, "Core-owned custody"):
            verify_lifetime(self.config, initial=True)
        self.assertFalse(tuple(loader.iterdir()))

    def test_retained_request_does_not_make_the_new_loader_nonempty(self) -> None:
        """Restart can keep completed output while initializing a clean model view."""
        request = check_support.request(self.config)
        original = Path(request.output_directory) / "result.wav"
        original.write_bytes(b"result still awaiting delivery")
        verify_lifetime(self.config, initial=True)
        self.assertEqual(original.read_bytes(), b"result still awaiting delivery")
