"""Real sealed-byte production with tiny committed fixtures, never music weights."""

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
from dataclasses import dataclass, replace
from pathlib import Path
from unittest.mock import patch

import build_ace_step
from build_ace_step import (
    ADAPTER_MODULES,
    ArchiveInventory,
    SourceInventory,
    assemble_wheel,
)
from package_inputs import (
    NativePackageVariant,
    inspect_candidate,
    locked_dependencies,
    native_package_limits,
    read_python_catalog,
    unpack_candidate,
)
from pydantic import JsonValue, TypeAdapter
from wheel_inputs import validate_dependency_closure


def document(path: Path) -> dict[str, JsonValue]:
    """Read a fixture metadata document without untyped JSON access."""
    return TypeAdapter[dict[str, JsonValue]](dict[str, JsonValue]).validate_json(
        path.read_bytes()
    )


def selected_object(data: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """Narrow the fixture's one archive object before a deliberate mutation."""
    objects = data["objects"]
    assert isinstance(objects, list) and len(objects) == 1
    result = objects[0]
    assert isinstance(result, dict)
    return result


def git(root: Path, *arguments: str) -> str:
    """Run a fixture-local Git command without modifying user configuration."""
    return subprocess.check_output(
        ["git", "-C", str(root), *arguments], stderr=subprocess.PIPE, text=True
    ).strip()


def commit(root: Path) -> str:
    """Assign immutable provenance to tiny source inputs, not an engine build."""
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


@dataclass(frozen=True)
class Fixture:
    """Independent repositories and pinned archive/lock inputs for one target."""

    root: Path
    repository: Path
    source: Path
    wrapper: Path
    wheelhouse: Path
    wheel: Path
    lock: Path
    interpreter: Path


def fixture(root: Path) -> Fixture:
    """Create a complete test selection without downloading or running model code."""
    repository, source, wheelhouse = (
        root / "adapter",
        root / "upstream",
        root / "wheels",
    )
    wrapper = repository / "packaging/skulk-ace-step/skulk_ace_step"
    for directory in (repository, source, wheelhouse, wrapper):
        directory.mkdir(parents=True, exist_ok=True)
    for directory in (repository, source):
        git(directory, "init", "-q")
        (directory / "LICENSE").write_text("fixture license\n")
    paths = [
        "acestep/__init__.py",
        "acestep/training/__init__.py",
        *[
            "acestep/models/base/" + name
            for name in (
                "configuration_acestep_v15.py",
                "modeling_acestep_v15_base.py",
                "apg_guidance.py",
            )
        ],
    ]
    files: list[JsonValue] = []
    for name in paths:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        data = (
            b""
            if name == "acestep/__init__.py"
            else ('"""Fixture source: ' + name + '."""\n').encode()
        )
        path.write_bytes(data)
        files.append(
            dict(
                path=name, size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest()
            )
        )
    (source / "unselected.py").write_text("must not be included\n")
    source_revision = commit(source)
    canonical = repository / "src/contract.py"
    canonical.parent.mkdir()
    canonical.write_text('"""Canonical source contract fixture."""\n')
    for name in ADAPTER_MODULES:
        if name in ("contracts.py", "audio_separation.py"):
            (wrapper / name).symlink_to("../../../src/contract.py")
        else:
            (wrapper / name).write_bytes(
                b"" if name == "py.typed" else b'"""Adapter fixture."""\n'
            )
    owned = wrapper.parent
    override = owned / "patches/inference-training-namespace.py"
    override.parent.mkdir()
    override.write_bytes(b'"""Owned inference-only namespace."""\n')
    pin = dict(
        repository="https://github.com/ace-step/ACE-Step-1.5.git",
        revision=source_revision,
    )
    (owned / "source.json").write_text(json.dumps(pin))
    (owned / "source-files.json").write_text(
        json.dumps(
            pin
            | dict(
                files=files,
                overrides=[
                    dict(
                        path="acestep/training/__init__.py",
                        source="patches/inference-training-namespace.py",
                        sha256=hashlib.sha256(override.read_bytes()).hexdigest(),
                        reason="Avoid eager trainer imports",
                    )
                ],
                effective_model_code=[
                    dict(source=p, destination=Path(p).name)
                    for p in paths
                    if p.startswith("acestep/models/")
                ],
            )
        )
    )
    catalog = read_python_catalog(Path(__file__).parents[1])
    interpreter = root / catalog.for_target("macosx_14_0_arm64").filename
    with tarfile.open(interpreter, "w:gz") as archive:
        data = b"fixture interpreter, never executed\n"
        entry = tarfile.TarInfo("python/bin/python3.12")
        entry.size, entry.mode = len(data), 0o755
        archive.addfile(entry, io.BytesIO(data))
    python_pin = catalog.for_target("macosx_14_0_arm64").model_copy(
        update={
            "size_bytes": interpreter.stat().st_size,
            "sha256": hashlib.sha256(interpreter.read_bytes()).hexdigest(),
        }
    )
    (owned / "python.json").write_text(
        catalog.model_copy(
            update={"archives": (python_pin, *catalog.archives[1:])}
        ).model_dump_json()
    )
    wheel = wheelhouse / "fixture_dependency-1.0.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("fixture_dependency/__init__.py", b"")
        archive.writestr(
            "fixture_dependency-1.0.0.dist-info/METADATA",
            "Metadata-Version: 2.4\nName: fixture-dependency\nVersion: 1.0.0\nRequires-Python: >=3.12\n\n",
        )
    lock = owned / "locks/macos-arm64-cpu-py312.lock"
    lock.parent.mkdir()
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    lock.write_text(f"fixture-dependency==1.0.0 --hash=sha256:{digest}\n")
    (owned / "requirements.in").write_text("fixture-dependency==1.0.0\n")
    (owned / "archives").mkdir()
    (owned / "archives/macos-arm64-cpu-py312.json").write_text(
        json.dumps(
            dict(
                target="macosx_14_0_arm64",
                lock_sha256=hashlib.sha256(lock.read_bytes()).hexdigest(),
                python_archive=python_pin.model_dump(mode="json"),
                objects=[
                    dict(
                        distribution="fixture-dependency",
                        version="1.0.0",
                        filename=wheel.name,
                        size_bytes=wheel.stat().st_size,
                        sha256=digest,
                        source_url="https://files.pythonhosted.org/fixtures/fixture.whl",
                    )
                ],
            )
        )
    )
    commit(repository)
    return Fixture(
        root, repository, source, wrapper, wheelhouse, wheel, lock, interpreter
    )


class NativeProducerTests(unittest.TestCase):
    """Immutable provenance, exact target selection and deterministic regular bytes."""

    def setUp(self) -> None:
        """Give every producer mutation its own committed repositories and archives."""
        self.temporary = tempfile.TemporaryDirectory(prefix="native-ace-producer-")
        self.inputs = fixture(Path(self.temporary.name))

    def tearDown(self) -> None:
        """Remove only this test's owned source/output trees."""
        self.temporary.cleanup()

    def build(
        self,
        name: str = "output",
        *,
        variant: NativePackageVariant = "cpu",
        target: str = "macosx_14_0_arm64",
    ) -> Path:
        """Seal the actual fixture bytes through the production entry point."""
        value = self.inputs
        return assemble_wheel(
            value.repository,
            value.source,
            value.wrapper,
            value.wheelhouse,
            value.lock,
            value.root / name,
            target=target,
            python_archive=value.interpreter,
            variant=variant,
        )

    def repin_wheel(self) -> None:
        """Update complete selected archive provenance after a deliberate fixture mutation."""
        value = self.inputs
        digest = hashlib.sha256(value.wheel.read_bytes()).hexdigest()
        value.lock.write_text(f"fixture-dependency==1.0.0 --hash=sha256:{digest}\n")
        path = value.wrapper.parent / "archives/macos-arm64-cpu-py312.json"
        data = document(path)
        selected_object(data).update(
            filename=value.wheel.name,
            size_bytes=value.wheel.stat().st_size,
            sha256=digest,
        )
        data["lock_sha256"] = hashlib.sha256(value.lock.read_bytes()).hexdigest()
        path.write_text(json.dumps(data))
        commit(value.repository)

    def test_independent_builds_are_identical_and_links_become_regular_bytes(
        self,
    ) -> None:
        """The sealed object has no timestamps, local paths or unresolved developer links."""
        first, second = self.build("first"), self.build("second")
        self.assertEqual(first.read_bytes(), second.read_bytes())
        with zipfile.ZipFile(first) as archive:
            for name in ("contracts.py", "audio_separation.py"):
                info = archive.getinfo("skulk_ace_step/" + name)
                self.assertTrue(stat.S_ISREG(info.external_attr >> 16))
                self.assertEqual(
                    archive.read(info),
                    (self.inputs.repository / "src/contract.py").read_bytes(),
                )
            self.assertEqual(
                archive.read("acestep/training/__init__.py"),
                (
                    self.inputs.wrapper.parent
                    / "patches/inference-training-namespace.py"
                ).read_bytes(),
            )
            self.assertNotIn("unselected.py", archive.namelist())
            code = TypeAdapter[list[dict[str, JsonValue]]](
                list[dict[str, JsonValue]]
            ).validate_json(archive.read("skulk_ace_step/model-code.json"))
            self.assertEqual(len(code), 3)
            self.assertNotIn(
                str(self.inputs.root).encode(),
                archive.read("skulk_ace_step/build.json"),
            )
            self.assertIn(
                "skulk_ace_step/licenses/ACE_STEP_LICENSE.txt", archive.namelist()
            )
        with self.assertRaises(FileExistsError):
            self.build("first")
        self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_real_package_verifier_accepts_exact_native_layout(self) -> None:
        """Shared bootstrap recognizes the ACE owner and verifies every selected member."""
        wheel = self.build()
        digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
        manifest = unpack_candidate(wheel, self.inputs.root / "unpacked", digest)
        self.assertEqual(
            manifest.source_repository, "https://github.com/ace-step/ACE-Step-1.5.git"
        )
        self.assertIn("music.edit", manifest.features)
        self.assertEqual(
            manifest.adapter_revision, git(self.inputs.repository, "rev-parse", "HEAD")
        )

    def test_outputs_cannot_be_hidden_inside_either_source_repository(self) -> None:
        """Ignored output paths cannot evade the producer's immutable source boundary."""
        value = self.inputs
        for root in (value.repository, value.source):
            with (
                self.subTest(root=root),
                self.assertRaisesRegex(ValueError, "outside both clean"),
            ):
                assemble_wheel(
                    value.repository,
                    value.source,
                    value.wrapper,
                    value.wheelhouse,
                    value.lock,
                    root / "ignored-output",
                    target="macosx_14_0_arm64",
                    python_archive=value.interpreter,
                )
            self.assertFalse((root / "ignored-output").exists())

    def test_dirty_adapter_and_extra_committed_module_are_refused(self) -> None:
        """Neither a stale source label nor an undeclared module can enter the wheel."""
        path = self.inputs.wrapper / "unexpected.py"
        path.write_text("unexpected\n")
        with self.assertRaisesRegex(ValueError, "clean immutable"):
            self.build()
        commit(self.inputs.repository)
        with self.assertRaisesRegex(ValueError, "closed module inventory"):
            self.build()

    def test_original_source_hash_is_checked_after_an_updated_revision(self) -> None:
        """Matching Git commit labels cannot substitute for every selected file digest."""
        value = self.inputs
        (value.source / "acestep/__init__.py").write_text("changed\n")
        revision = commit(value.source)
        for name in ("source.json", "source-files.json"):
            path = value.wrapper.parent / name
            data = document(path)
            data["revision"] = revision
            path.write_text(json.dumps(data))
        commit(value.repository)
        with self.assertRaisesRegex(ValueError, "selected source bytes"):
            self.build()

    def test_owned_override_has_its_own_digest(self) -> None:
        """A reviewed upstream commit does not authenticate a changed local patch."""
        (
            self.inputs.wrapper.parent / "patches/inference-training-namespace.py"
        ).write_text("changed\n")
        commit(self.inputs.repository)
        with self.assertRaisesRegex(ValueError, "override differs"):
            self.build()

    def test_source_contract_link_cannot_escape_its_clean_repository(self) -> None:
        """A committed link is still rejected if its bytes come from outside provenance."""
        outside = self.inputs.root / "outside.py"
        outside.write_text("not owned by source commit\n")
        link = self.inputs.wrapper / "contracts.py"
        link.unlink()
        link.symlink_to(outside)
        commit(self.inputs.repository)
        with self.assertRaisesRegex(ValueError, "escaped"):
            self.build()

    def test_changed_and_missing_dependency_archives_are_refused(self) -> None:
        """The exact intake inventory cannot be satisfied by a label or partial wheel set."""
        self.inputs.wheel.write_bytes(b"changed archive")
        with self.assertRaisesRegex(ValueError, "exact compatible archive"):
            self.build()
        self.inputs.wheel.unlink()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.build()

    def test_interpreter_bytes_remain_independently_pinned(self) -> None:
        """The containing wheel manifest cannot bless a substituted interpreter."""
        data = self.inputs.interpreter.read_bytes()
        self.inputs.interpreter.write_bytes(bytes([data[0] ^ 1]) + data[1:])
        with self.assertRaisesRegex(ValueError, "standalone Python differs"):
            self.build()

    def test_locked_foreign_architecture_still_cannot_enter_selected_target(
        self,
    ) -> None:
        """An exact hash and metadata version do not make an x86 wheel ARM-compatible."""
        foreign = self.inputs.wheel.with_name(
            "fixture_dependency-1.0.0-cp312-cp312-manylinux_2_28_x86_64.whl"
        )
        self.inputs.wheel.rename(foreign)
        path = self.inputs.wrapper.parent / "archives/macos-arm64-cpu-py312.json"
        data = document(path)
        selected_object(data)["filename"] = foreign.name
        path.write_text(json.dumps(data))
        commit(self.inputs.repository)
        with self.assertRaisesRegex(ValueError, "exact compatible archive"):
            self.build()

    def test_mutually_relocked_metadata_cannot_omit_a_transitive_dependency(
        self,
    ) -> None:
        """Actual Requires-Dist closure is checked after all object hashes match."""
        with zipfile.ZipFile(self.inputs.wheel, "w") as archive:
            archive.writestr("fixture_dependency/__init__.py", b"")
            archive.writestr(
                "fixture_dependency-1.0.0.dist-info/METADATA",
                "Metadata-Version: 2.4\nName: fixture-dependency\nVersion: 1.0.0\nRequires-Python: >=3.12\nRequires-Dist: absent-child>=1\n\n",
            )
        self.repin_wheel()
        with self.assertRaisesRegex(ValueError, "metadata closure is incomplete"):
            self.build()

    def test_streamed_wheel_substitution_cannot_leave_an_advertised_package(
        self,
    ) -> None:
        """Recheck bytes during copying, after earlier hash and metadata validation."""
        original = build_ace_step.write_member

        def changed(
            archive: zipfile.ZipFile, name: str, member: build_ace_step.Member
        ) -> None:
            if name == "skulk_ace_step/wheels/" + self.inputs.wheel.name:
                data = self.inputs.wheel.read_bytes()
                self.inputs.wheel.write_bytes(bytes([data[0] ^ 1]) + data[1:])
            original(archive, name, member)

        with (
            patch.object(build_ace_step, "write_member", changed),
            self.assertRaisesRegex(ValueError, "changed during sealing"),
        ):
            self.build()
        output = self.inputs.root / "output"
        self.assertEqual(list(output.glob("*.whl")), [])
        self.assertEqual(list(output.glob("*.json")), [])

    def test_shared_metadata_checker_accepts_identical_bytes_and_paths(self) -> None:
        """The streaming extension preserves the existing Demucs in-memory interface."""
        required = locked_dependencies(self.inputs.lock.read_bytes())
        direct = (self.inputs.wrapper.parent / "requirements.in").read_bytes()
        for value in (self.inputs.wheel, self.inputs.wheel.read_bytes()):
            validate_dependency_closure(
                {self.inputs.wheel.name: value}, required, direct, "macosx_14_0_arm64"
            )

    def test_actual_committed_inputs_cover_all_targets_and_empty_source_namespaces(
        self,
    ) -> None:
        """Tiny fixtures cannot conceal a schema that rejects the real selected archives."""
        root = Path(__file__).parents[1]
        source = SourceInventory.model_validate_json(
            (root / "source-files.json").read_bytes()
        )
        self.assertTrue(any(item.size_bytes == 0 for item in source.files))
        targets: set[str] = set()
        for path in (root / "archives").glob("*-cpu-py312.json"):
            inventory = ArchiveInventory.model_validate_json(path.read_bytes())
            targets.add(inventory.target)
            self.assertEqual(len(inventory.objects), 53)
        self.assertEqual(
            targets,
            {"macosx_14_0_arm64", "manylinux_2_28_x86_64", "manylinux_2_28_aarch64"},
        )

    def test_gpu_profiles_seal_and_verify_actual_candidate_bytes(self) -> None:
        """Tiny real packages exercise all GPU layouts without native execution."""
        profiles: tuple[tuple[NativePackageVariant, str], ...] = (
            ("cuda", "manylinux_2_28_x86_64"),
            ("cuda", "manylinux_2_28_aarch64"),
            ("rocm", "linux_x86_64"),
        )
        for variant, target in profiles:
            with self.subTest(variant=variant, target=target):
                owned = self.inputs.wrapper.parent
                catalog = read_python_catalog(owned)
                interpreter_target = (
                    "manylinux_2_28_x86_64" if target == "linux_x86_64" else target
                )
                pin = catalog.for_target(interpreter_target).model_copy(
                    update={
                        "sha256": hashlib.sha256(
                            self.inputs.interpreter.read_bytes()
                        ).hexdigest(),
                        "size_bytes": self.inputs.interpreter.stat().st_size,
                    }
                )
                interpreter = self.inputs.root / pin.filename
                interpreter.write_bytes(self.inputs.interpreter.read_bytes())
                updated = catalog.model_copy(
                    update={
                        "archives": tuple(
                            pin if item.profile == pin.profile else item
                            for item in catalog.archives
                        )
                    }
                )
                (owned / "python.json").write_text(updated.model_dump_json())
                inventory = document(owned / "archives/macos-arm64-cpu-py312.json")
                lock = owned / f"locks/{pin.profile}-{variant}-py312.lock"
                lock.write_bytes(self.inputs.lock.read_bytes())
                inventory.update(
                    variant=variant,
                    target=target,
                    python_archive=pin.model_dump(mode="json"),
                    lock_sha256=hashlib.sha256(lock.read_bytes()).hexdigest(),
                )
                (owned / f"archives/{pin.profile}-{variant}-py312.json").write_text(
                    json.dumps(inventory)
                )
                commit(self.inputs.repository)
                self.inputs = replace(self.inputs, lock=lock, interpreter=interpreter)
                wheel = self.build(
                    f"{variant}-{target}", variant=variant, target=target
                )
                digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
                manifest = inspect_candidate(
                    wheel, digest, limits=native_package_limits(variant)
                )
                self.assertEqual(
                    (manifest.variant, manifest.platform_tag), (variant, target)
                )
                self.assertEqual(
                    manifest.python_archive.platform_tag, interpreter_target
                )
                self.assertIn("library_probe", manifest.features)
                self.assertIn("memory_probe", manifest.features)
                self.assertIn("live_resource_guard", manifest.features)
                with self.assertRaises(ValueError):
                    inspect_candidate(wheel, digest)

    def test_cuda_profile_cannot_be_mislabeled_as_an_apple_package(self) -> None:
        """A variant switch cannot invent a CPU architecture or accelerator ABI."""
        with self.assertRaisesRegex(ValueError, "target"):
            self.build(variant="cuda", target="macosx_14_0_arm64")

    def test_actual_gpu_inputs_bind_all_archives_and_complete_selected_locks(
        self,
    ) -> None:
        """The real GPU inventories must satisfy the same source schema as fixtures."""
        root = Path(__file__).parents[1]
        profiles: tuple[tuple[str, NativePackageVariant, str, int], ...] = (
            ("linux-amd64", "cuda", "manylinux_2_28_x86_64", 71),
            ("linux-arm64", "cuda", "manylinux_2_28_aarch64", 71),
            ("linux-amd64", "rocm", "linux_x86_64", 57),
        )
        for profile, variant, target, count in profiles:
            name = f"{profile}-{variant}-py312"
            inventory = ArchiveInventory.model_validate_json(
                (root / f"archives/{name}.json").read_bytes()
            )
            lock = (root / f"locks/{name}.lock").read_bytes()
            required = locked_dependencies(lock)
            self.assertEqual((inventory.variant, inventory.target), (variant, target))
            self.assertEqual(len(inventory.objects), count)
            self.assertEqual(inventory.lock_sha256, hashlib.sha256(lock).hexdigest())
            self.assertEqual(
                set(required), {item.distribution for item in inventory.objects}
            )
            limits = native_package_limits(variant)
            for item in inventory.objects:
                version, hashes = required[item.distribution]
                self.assertEqual(str(version), item.version)
                self.assertIn(item.sha256, hashes)
                self.assertLessEqual(item.size_bytes, limits.object_bytes)
            self.assertEqual(
                sum(item.source_build is not None for item in inventory.objects),
                1 if variant == "rocm" else 0,
            )
