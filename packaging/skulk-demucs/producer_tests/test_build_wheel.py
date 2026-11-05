"""Offline producer regressions with tiny committed fixture repositories."""

from __future__ import annotations

import hashlib
import io
import json
import stat
import subprocess
import tarfile
import tempfile
import unittest
import zipfile
from dataclasses import dataclass
from pathlib import Path

from build_wheel import (
    ADAPTER_MODULES,
    UPSTREAM_MODULES,
    assemble_wheel,
    dependency_matches,
    locked_dependencies,
)
from package_inputs import read_python_catalog
from verify_package import unpack_candidate


@dataclass(frozen=True)
class Fixture:
    """Independent source repositories and a one-distribution offline wheel set."""

    root: Path
    repository: Path
    source: Path
    wrapper: Path
    wheelhouse: Path
    lock: Path
    wheel: Path
    python_archive: Path


def git(root: Path, *arguments: str) -> str:
    """Run an isolated test repository command without changing user Git config."""
    return (
        subprocess.check_output(
            ["git", "-C", str(root), *arguments], stderr=subprocess.PIPE
        )
        .decode()
        .strip()
    )


def commit_fixture(root: Path) -> str:
    """Record known tiny test inputs for the producer's dirty-source gate."""
    git(root, "add", ".")
    git(
        root,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-qm",
        "fixture",
    )
    return git(root, "rev-parse", "HEAD")


def fixture(root: Path) -> Fixture:
    """Create valid inventories without downloading or executing model code."""
    repository = root / "adapter-repo"
    source = root / "author-repo"
    wrapper = repository / "packaging" / "skulk-demucs" / "skulk_demucs"
    wheelhouse = root / "wheels"
    for directory in (repository, source, wheelhouse, wrapper, source / "demucs"):
        directory.mkdir(parents=True, exist_ok=True)
    for directory in (repository, source):
        git(directory, "init", "-q")
        (directory / "LICENSE").write_text("fixture license\n")
    for filename in UPSTREAM_MODULES:
        (source / "demucs" / filename).write_text('"""Test module."""\n')
    revision = commit_fixture(source)
    contract = repository / "src" / "contract.py"
    contract.parent.mkdir()
    contract.write_text('"""Canonical fixture contract."""\n')
    for filename in ADAPTER_MODULES:
        if filename == "contracts.py":
            (wrapper / filename).symlink_to("../../../src/contract.py")
        else:
            (wrapper / filename).write_text('"""Test adapter module."""\n')
    (wrapper.parent / "source.json").write_text(
        json.dumps(
            {
                "repository": "https://github.com/adefossez/demucs.git",
                "revision": revision,
            }
        )
    )
    (wrapper.parent / "requirements.in").write_text("fixture-dependency==1.0.0\n")
    catalog = read_python_catalog(Path(__file__).parents[1])
    python_filename = (
        "cpython-3.12.13+20260807-aarch64-apple-darwin-install_only_stripped.tar.gz"
    )
    python_archive = root / python_filename
    with tarfile.open(python_archive, "w:gz") as archive:
        content = b"fixture interpreter, never executed\n"
        entry = tarfile.TarInfo("python/bin/python3.12")
        entry.size = len(content)
        entry.mode = 0o755
        archive.addfile(entry, io.BytesIO(content))
    fixture_pin = catalog.for_target("macosx_14_0_arm64").model_copy(
        update={
            "size_bytes": python_archive.stat().st_size,
            "sha256": hashlib.sha256(python_archive.read_bytes()).hexdigest(),
        }
    )
    fixture_catalog = catalog.model_copy(
        update={"archives": (fixture_pin, *catalog.archives[1:])}
    )
    (wrapper.parent / "python.json").write_text(fixture_catalog.model_dump_json())
    commit_fixture(repository)
    wheel = wheelhouse / "fixture_dependency-1.0.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("fixture_dependency/__init__.py", b"")
        archive.writestr(
            "fixture_dependency-1.0.0.dist-info/METADATA",
            "Metadata-Version: 2.4\nName: fixture-dependency\nVersion: 1.0.0\nRequires-Python: >=3.12\n\n",
        )
    lock = root / "requirements.lock"
    lock.write_text(
        f"fixture-dependency==1.0.0 --hash=sha256:{hashlib.sha256(wheel.read_bytes()).hexdigest()}\n"
    )
    return Fixture(
        root, repository, source, wrapper, wheelhouse, lock, wheel, python_archive
    )


class ProducerTests(unittest.TestCase):
    """Prove identity, provenance, completeness and architecture refusal."""

    temporary: tempfile.TemporaryDirectory[str]
    inputs: Fixture

    def setUp(self) -> None:
        """Create a separate complete input set for every mutation test."""
        self.temporary = tempfile.TemporaryDirectory()
        self.inputs = fixture(Path(self.temporary.name))

    def tearDown(self) -> None:
        """Remove only the test's own temporary repositories."""
        self.temporary.cleanup()

    def build(self, name: str = "output", wrapper: Path | None = None) -> Path:
        """Assemble fixture inputs into a fresh explicitly owned output location."""
        value = self.inputs
        return assemble_wheel(
            value.repository,
            value.source,
            wrapper if wrapper is not None else value.wrapper,
            value.wheelhouse,
            value.lock,
            value.root / name,
            target="macosx_14_0_arm64",
            python_archive=value.python_archive,
        )

    def replace_metadata(
        self,
        extra_headers: str,
        *,
        metadata_version: str = "2.4",
        python_requirement: str = ">=3.12",
    ) -> None:
        """Re-pin a fixture archive to exercise metadata after content verification."""
        wheel = self.inputs.wheel
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr("fixture_dependency/__init__.py", b"")
            archive.writestr(
                "fixture_dependency-1.0.0.dist-info/METADATA",
                f"Metadata-Version: {metadata_version}\nName: fixture-dependency\nVersion: 1.0.0\nRequires-Python: {python_requirement}\n{extra_headers}\n",
            )
        self.inputs.lock.write_text(
            f"fixture-dependency==1.0.0 --hash=sha256:{hashlib.sha256(wheel.read_bytes()).hexdigest()}\n"
        )

    def test_independent_builds_are_identical_and_contracts_are_regular_bytes(
        self,
    ) -> None:
        """No timestamp, path or symlink dependency may enter the final payload."""
        first = self.build("first")
        second = self.build("second")
        self.assertEqual(first.read_bytes(), second.read_bytes())
        with zipfile.ZipFile(first) as archive:
            contract = archive.getinfo("skulk_demucs/contracts.py")
            self.assertTrue(stat.S_ISREG(contract.external_attr >> 16))
            self.assertEqual(
                archive.read(contract),
                (self.inputs.repository / "src" / "contract.py").read_bytes(),
            )
            manifest = archive.read("skulk_demucs/build.json")
            self.assertNotIn(str(self.inputs.root).encode(), manifest)
            self.assertIn(b'"source_revision"', manifest)
            self.assertIn(b'"payload_sha256"', manifest)
            self.assertIn("demucs/htdemucs.py", archive.namelist())
            self.assertEqual(
                archive.read(f"skulk_demucs/python/{self.inputs.python_archive.name}"),
                self.inputs.python_archive.read_bytes(),
            )
            self.assertIn(
                "skulk_demucs/licenses/DEMUCS_LICENSE.txt", archive.namelist()
            )
        with self.assertRaises(FileExistsError):
            self.build("first")

    def test_candidate_unpack_requires_complete_package_and_manifest_identity(
        self,
    ) -> None:
        """An exact sealed package extracts without needing developer source paths."""
        wheel = self.build()
        digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
        manifest = unpack_candidate(wheel, self.inputs.root / "unpacked", digest)
        self.assertEqual(
            manifest.python_archive.filename, self.inputs.python_archive.name
        )
        self.assertEqual(
            manifest.adapter_revision, git(self.inputs.repository, "rev-parse", "HEAD")
        )
        with self.assertRaisesRegex(ValueError, "expected digest"):
            unpack_candidate(wheel, self.inputs.root / "changed", "0" * 64)

    def test_dirty_source_is_refused(self) -> None:
        """A committed identity cannot label locally changed adapter source."""
        (self.inputs.wrapper / "engine.py").write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "clean immutable"):
            self.build()

    def test_interpreter_archive_bytes_are_pinned(self) -> None:
        """A cached interpreter with the right label but changed bytes is refused."""
        original = self.inputs.python_archive.read_bytes()
        self.inputs.python_archive.write_bytes(bytes([original[0] ^ 1]) + original[1:])
        with self.assertRaisesRegex(ValueError, "standalone Python differs"):
            self.build()

    def test_interpreter_target_cannot_be_substituted(self) -> None:
        """A catalog cannot relabel an ARM interpreter as an x86 package."""
        path = self.inputs.wrapper.parent / "python.json"
        path.write_text(
            path.read_text().replace("macosx_14_0_arm64", "manylinux_2_28_x86_64", 1)
        )
        commit_fixture(self.inputs.repository)
        with self.assertRaisesRegex(ValueError, "identity does not match"):
            self.build()

    def test_interpreter_catalog_must_cover_three_distinct_targets(self) -> None:
        """A duplicated profile cannot hide a missing supported architecture."""
        path = self.inputs.wrapper.parent / "python.json"
        catalog = read_python_catalog(path.parent)
        path.write_text(
            catalog.model_copy(
                update={
                    "archives": (
                        catalog.archives[0],
                        catalog.archives[0],
                        catalog.archives[2],
                    )
                }
            ).model_dump_json()
        )
        commit_fixture(self.inputs.repository)
        with self.assertRaisesRegex(ValueError, "all three targets"):
            self.build()

    def test_foreign_adapter_root_is_refused(self) -> None:
        """An unrelated clean repository cannot provide the provenance identity."""
        with self.assertRaisesRegex(ValueError, "declared clean source"):
            self.build(wrapper=self.inputs.source)

    def test_source_revision_mismatch_is_refused(self) -> None:
        """Even clean library inputs must match the adapter's pinned source."""
        (self.inputs.source / "LICENSE").write_text("changed fixture license\n")
        commit_fixture(self.inputs.source)
        with self.assertRaisesRegex(ValueError, "reviewed source pin"):
            self.build()

    def test_changed_archive_is_refused(self) -> None:
        """Exact version labels do not authorize changed dependency bytes."""
        self.inputs.wheel.write_bytes(b"changed archive")
        with self.assertRaisesRegex(ValueError, "unlocked, changed"):
            self.build()

    def test_missing_archive_is_refused(self) -> None:
        """The producer cannot seal a partial offline dependency set."""
        self.inputs.wheel.unlink()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.build()

    def test_foreign_architecture_is_refused_even_with_a_locked_hash(self) -> None:
        """A valid content hash cannot substitute for a compatible wheel ABI."""
        self.inputs.wheel.rename(
            self.inputs.wheel.with_name(
                "fixture_dependency-1.0.0-cp312-cp312-manylinux_2_28_x86_64.whl"
            )
        )
        with self.assertRaisesRegex(ValueError, "foreign target"):
            self.build()

    def test_contract_symlink_cannot_escape_declared_source(self) -> None:
        """A clean tracked link must still dereference into its source repository."""
        outside = self.inputs.root / "outside.py"
        outside.write_text("outside\n")
        link = self.inputs.wrapper / "contracts.py"
        link.unlink()
        link.symlink_to(outside)
        commit_fixture(self.inputs.repository)
        with self.assertRaisesRegex(ValueError, "unexpected file"):
            self.build()

    def test_lock_requires_exact_unique_hash_pins(self) -> None:
        """No URL, range, marker, unhashed or duplicate input enters the package."""
        cases = (
            b"numpy>=2\n",
            b"numpy==2.2.6\n",
            b"--index-url https://example.invalid\n",
            self.inputs.lock.read_bytes() * 2,
        )
        for invalid in cases:
            with self.subTest(lock=invalid), self.assertRaises(ValueError):
                locked_dependencies(invalid)

    def test_omitted_transitive_dependency_is_refused(self) -> None:
        """A mutually matching lock and wheel set can still omit a real dependency."""
        self.replace_metadata("Requires-Dist: absent-child>=1\n")
        with self.assertRaisesRegex(ValueError, "metadata closure is incomplete"):
            self.build()

    def test_source_dependency_change_requires_an_updated_lock(self) -> None:
        """An older complete lock cannot label newly committed adapter requirements."""
        (self.inputs.wrapper.parent / "requirements.in").write_text(
            "fixture-dependency==1.0.0\nnew-dependency==1.0\n"
        )
        commit_fixture(self.inputs.repository)
        with self.assertRaisesRegex(ValueError, "does not satisfy adapter source"):
            self.build()

    def test_invalid_requirement_header_is_not_silently_dropped(self) -> None:
        """The metadata parser's unparsed requirements must fail the producer."""
        self.replace_metadata("Requires-Python: >=3.13\n")
        with self.assertRaisesRegex(ValueError, "requirements are malformed"):
            self.build()

    def test_wrong_interpreter_requirement_is_refused(self) -> None:
        """A correctly hashed and tagged wheel must also admit native Python 3.12."""
        self.replace_metadata("", python_requirement=">=3.13")
        with self.assertRaisesRegex(
            ValueError, "does not support the pinned native interpreter"
        ):
            self.build()

    def test_notice_metadata_does_not_disable_dependency_validation(self) -> None:
        """Older upstream notice syntax is preserved without ignoring requirements."""
        self.replace_metadata("License-File: LICENSE\n", metadata_version="2.1")
        self.assertTrue(self.build("notice-only").is_file())
        self.replace_metadata(
            "License-File: LICENSE\nRequires-Dist: absent-child>=1\n",
            metadata_version="2.1",
        )
        with self.assertRaisesRegex(ValueError, "metadata closure is incomplete"):
            self.build("missing-child")

    def test_interpreter_and_platform_compatibility(self) -> None:
        """Stable ABI wheels remain eligible; foreign architectures do not."""
        self.assertTrue(
            dependency_matches(
                "a-1.0-cp310-abi3-manylinux_2_17_aarch64.whl", "manylinux_2_28_aarch64"
            )
        )
        self.assertTrue(
            dependency_matches("a-1.0-py3-none-any.whl", "macosx_14_0_arm64")
        )
        self.assertFalse(
            dependency_matches(
                "a-1.0-cp313-cp313-macosx_14_0_arm64.whl", "macosx_14_0_arm64"
            )
        )
        self.assertFalse(
            dependency_matches(
                "a-1.0-cp312-cp312-manylinux_2_28_x86_64.whl", "manylinux_2_28_aarch64"
            )
        )


if __name__ == "__main__":
    unittest.main()
