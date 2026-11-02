"""Offline package refusal tests; no models, interpreters or network are executed."""

from __future__ import annotations

import hashlib
import io
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

from package_inputs import PythonArchivePin, read_python_catalog
from verify_package import extract_python, safe_member, verify_site_packages


class InstallationTests(unittest.TestCase):
    """Exercise archive boundaries and expanded-tree integrity independently."""

    temporary: tempfile.TemporaryDirectory[str]
    root: Path

    def setUp(self) -> None:
        """Give each malformed archive a new private test directory."""
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        """Remove only this test's owned files."""
        self.temporary.cleanup()

    def python_archive(
        self, entries: tuple[tarfile.TarInfo, ...]
    ) -> tuple[Path, PythonArchivePin]:
        """Pin the exact synthetic bytes so archive checks follow hash validation."""
        pin = read_python_catalog(Path(__file__).parents[1]).for_target(
            "macosx_14_0_arm64"
        )
        path = self.root / pin.filename
        with tarfile.open(path, "w:gz") as archive:
            for entry in entries:
                archive.addfile(entry, io.BytesIO(b"x" * entry.size))
        return path, pin.model_copy(
            update={
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size_bytes": path.stat().st_size,
            }
        )

    def test_regular_interpreter_and_in_root_link_survive(self) -> None:
        """The admitted standalone layout needs its legitimate relative links."""
        executable = tarfile.TarInfo("python/bin/python3.12")
        executable.mode = 0o755
        executable.size = 1
        link = tarfile.TarInfo("python/bin/python3")
        link.type = tarfile.SYMTYPE
        link.linkname = "python3.12"
        path, pin = self.python_archive((executable, link))
        native = extract_python(path, self.root / "expanded", pin)
        self.assertEqual(native.read_bytes(), b"x")
        self.assertEqual(native.with_name("python3").resolve(), native.resolve())

    def test_escaping_interpreter_link_is_refused(self) -> None:
        """A verified archive hash cannot authorize writing outside owned output."""
        link = tarfile.TarInfo("python/bin")
        link.type = tarfile.SYMTYPE
        link.linkname = "../../escaped"
        path, pin = self.python_archive((link,))
        with self.assertRaises(tarfile.FilterError):
            extract_python(path, self.root / "expanded", pin)
        self.assertFalse((self.root / "escaped").exists())

    def test_unsafe_archive_paths_are_refused(self) -> None:
        """Traversal, absolute names and alternate separators fail before writes."""
        for name in (
            "/tmp/foreign",
            "../foreign",
            "python/../foreign",
            "python\\foreign",
            "python//foreign",
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                safe_member(name)

    def test_duplicate_interpreter_member_is_refused(self) -> None:
        """Later archive entries cannot overwrite an already extracted executable."""
        entry = tarfile.TarInfo("python/bin/python3.12")
        entry.size = 1
        entry.mode = 0o755
        path, pin = self.python_archive((entry, entry))
        with self.assertRaisesRegex(ValueError, "duplicate member"):
            extract_python(path, self.root / "expanded", pin)

    def test_installed_tree_uses_wheel_bytes_instead_of_mutable_records(self) -> None:
        """Changed code is rejected even when its installed RECORD claims success."""
        wheel = self.root / "fixture.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr("fixture.py", b"original\n")
            archive.writestr("fixture-1.dist-info/RECORD", b"original record\n")
        site = self.root / "site"
        (site / "fixture-1.dist-info").mkdir(parents=True)
        (site / "fixture.py").write_bytes(b"original\n")
        (site / "fixture-1.dist-info" / "RECORD").write_text("installer rewrite\n")
        verify_site_packages(site, (wheel,))
        (site / "fixture.py").write_text("changed\n")
        (site / "fixture-1.dist-info" / "RECORD").write_text(
            "claims changed bytes are fine\n"
        )
        with self.assertRaisesRegex(ValueError, "differs from trusted package bytes"):
            verify_site_packages(site, (wheel,))

    def test_unexpected_import_path_and_missing_library_are_refused(self) -> None:
        """An injected .pth or removed extension invalidates the expanded runtime."""
        wheel = self.root / "fixture.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr("fixture.so", b"native library\n")
        site = self.root / "site"
        site.mkdir()
        (site / "fixture.so").write_bytes(b"native library\n")
        (site / "injected.pth").write_text("foreign import path\n")
        with self.assertRaisesRegex(ValueError, "differs from trusted package bytes"):
            verify_site_packages(site, (wheel,))
        (site / "injected.pth").unlink()
        (site / "fixture.so").unlink()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            verify_site_packages(site, (wheel,))


if __name__ == "__main__":
    unittest.main()
