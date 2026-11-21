"""Strict transitive extras against real tiny wheel metadata, not a resolver."""

from __future__ import annotations

import hashlib
import io
import tempfile
import unittest
import zipfile
from pathlib import Path

from packaging.version import Version
from wheel_inputs import dependency_matches, validate_dependency_closure


def wheel(name: str, headers: str = "", *, version: str = "1.0") -> bytes:
    """Encode independent distribution metadata inside an actual wheel archive."""
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w") as archive:
        archive.writestr(
            f"{name}-{version}.dist-info/METADATA",
            f"Metadata-Version: 2.4\nName: {name}\nVersion: {version}\n{headers}\n",
        )
    return target.getvalue()


def closure(
    metadata: dict[str, bytes], *, allow_extras: bool = True, paths: Path | None = None
) -> None:
    """Validate a fixed selected closure using actual archive bytes or paths."""
    wheels: dict[str, bytes | Path] = {}
    required: dict[str, tuple[Version, frozenset[str]]] = {}
    for name, content in metadata.items():
        filename = f"{name}-1.0-py3-none-any.whl"
        if paths is None:
            wheels[filename] = content
        else:
            target = paths / filename
            target.write_bytes(content)
            wheels[filename] = target
        required[name] = (
            Version("1.0"),
            frozenset((hashlib.sha256(content).hexdigest(),)),
        )
    validate_dependency_closure(
        wheels,
        required,
        b"root==1.0\n",
        "manylinux_2_28_x86_64",
        allow_dependency_extras=allow_extras,
    )


class DependencyExtraTests(unittest.TestCase):
    """A declared extra grants no exemption from exact complete transitive closure."""

    def fixture(self) -> dict[str, bytes]:
        """Model the actual Torch-to-ROCm-to-platform-library dependency shape."""
        return {
            "root": wheel("root", "Requires-Dist: meta[libraries]==1.0\n"),
            "meta": wheel(
                "meta",
                "Provides-Extra: libraries\nRequires-Dist: library==1.0; extra == 'libraries'\n",
            ),
            "library": wheel("library"),
        }

    def test_declared_extra_closes_for_original_bytes_and_streamed_archive_paths(
        self,
    ) -> None:
        """The selected ROCm-style extra is checked in both producer input forms."""
        closure(self.fixture())
        with tempfile.TemporaryDirectory() as directory:
            closure(self.fixture(), paths=Path(directory))

    def test_cpu_policy_still_refuses_dependency_extras(self) -> None:
        """ROCm enablement cannot silently broaden the shared CPU producer policy."""
        with self.assertRaisesRegex(ValueError, "closure is incomplete"):
            closure(self.fixture(), allow_extras=False)

    def test_missing_extra_library_is_refused(self) -> None:
        """Ignoring the activated extra would falsely accept this incomplete set."""
        selected = self.fixture()
        del selected["library"]
        with self.assertRaisesRegex(ValueError, "closure is incomplete"):
            closure(selected)

    def test_activated_extra_preserves_exact_version_constraints(self) -> None:
        """An existing distribution is insufficient if the extra needs other bytes."""
        selected = self.fixture()
        selected["meta"] = wheel(
            "meta",
            "Provides-Extra: libraries\nRequires-Dist: library==2.0; extra == 'libraries'\n",
        )
        with self.assertRaisesRegex(ValueError, "closure is incomplete"):
            closure(selected)

    def test_requested_extra_must_be_declared_by_its_distribution(self) -> None:
        """Unknown or undeclared extras never become a successful partial install."""
        for headers in ("", "Provides-Extra: other\n"):
            selected = self.fixture()
            selected["meta"] = wheel("meta", headers)
            with (
                self.subTest(headers=headers),
                self.assertRaisesRegex(ValueError, "unknown extra"),
            ):
                closure(selected)

    def test_new_extra_contexts_propagate_across_shared_dependencies_and_cycles(
        self,
    ) -> None:
        """An early visit cannot hide another parent's later compiler dependency."""
        selected = self.fixture()
        selected["root"] = wheel(
            "root",
            "Provides-Extra: follow\nRequires-Dist: meta[libraries]==1.0\nRequires-Dist: helper==1.0\n",
        )
        selected["helper"] = wheel("helper", "Requires-Dist: meta[compiler]==1.0\n")
        selected["meta"] = wheel(
            "meta",
            "Provides-Extra: libraries\nProvides-Extra: compiler\nRequires-Dist: root[follow]==1.0\nRequires-Dist: library==1.0; extra == 'libraries'\nRequires-Dist: compiler==1.0; extra == 'compiler'\n",
        )
        selected["compiler"] = wheel("compiler")
        closure(selected)

        del selected["compiler"]
        with self.assertRaisesRegex(ValueError, "closure is incomplete"):
            closure(selected)

    def test_direct_url_in_activated_extra_is_refused(self) -> None:
        """The extra cannot introduce network resolution during offline installation."""
        selected = self.fixture()
        selected["meta"] = wheel(
            "meta",
            "Provides-Extra: libraries\nRequires-Dist: library @ https://example.invalid/library.whl ; extra == 'libraries'\n",
        )
        with self.assertRaisesRegex(ValueError, "closure is incomplete"):
            closure(selected)

    def test_extra_names_are_normalized_and_platform_markers_remain_target_specific(
        self,
    ) -> None:
        """PEP 685 names match while unrelated Windows dependencies stay inactive."""
        selected = {
            "root": wheel("root", "Requires-Dist: meta[gpu_libs]==1.0\n"),
            "meta": wheel(
                "meta",
                "Provides-Extra: gpu-libs\nRequires-Dist: absent==1.0; extra == 'gpu_libs' and sys_platform == 'win32'\n",
            ),
        }
        closure(selected)


class PlatformBaselineTests(unittest.TestCase):
    """Older valid dependencies cannot silently change the package's host floor."""

    def test_x86_target_accepts_older_glibc_baselines_and_historical_aliases(
        self,
    ) -> None:
        """These actual NVIDIA tag families work above their glibc requirement."""
        for platform in (
            "manylinux_2_5_x86_64",
            "manylinux1_x86_64",
            "manylinux_2_12_x86_64",
            "manylinux2010_x86_64",
            "manylinux_2_17_x86_64",
            "manylinux2014_x86_64",
        ):
            with self.subTest(platform=platform):
                self.assertTrue(
                    dependency_matches(
                        f"library-1.0-cp312-cp312-{platform}.whl",
                        "manylinux_2_28_x86_64",
                    )
                )

    def test_newer_glibc_wrong_python_and_foreign_architectures_remain_refused(
        self,
    ) -> None:
        """Compatibility with older same-architecture wheels is not an ABI bypass."""
        for filename in (
            "library-1.0-cp312-cp312-manylinux_2_29_x86_64.whl",
            "library-1.0-cp313-cp313-manylinux_2_17_x86_64.whl",
            "library-1.0-cp312-cp312-manylinux_2_17_aarch64.whl",
            "library-1.0-cp312-cp312-macosx_14_0_arm64.whl",
        ):
            with self.subTest(filename=filename):
                self.assertFalse(dependency_matches(filename, "manylinux_2_28_x86_64"))

    def test_arm_target_still_starts_at_its_actual_first_manylinux_baseline(
        self,
    ) -> None:
        """AArch64 has no historical manylinux1 or manylinux2010 contract."""
        self.assertTrue(
            dependency_matches(
                "library-1.0-cp312-cp312-manylinux2014_aarch64.whl",
                "manylinux_2_28_aarch64",
            )
        )
        for platform in (
            "manylinux_2_12_aarch64",
            "manylinux2010_aarch64",
            "manylinux1_aarch64",
        ):
            with self.subTest(platform=platform):
                self.assertFalse(
                    dependency_matches(
                        f"library-1.0-cp312-cp312-{platform}.whl",
                        "manylinux_2_28_aarch64",
                    )
                )
