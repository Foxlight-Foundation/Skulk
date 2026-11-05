"""Actual complete file preparation before importing native model constructors."""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

import check_support
import check_weights
from skulk_ace_step.contracts import NativeMusicProgress
from skulk_ace_step.engine import load_music
from skulk_ace_step.upstream_bindings import NativeAPI, native_api


class StopBeforeInferenceError(RuntimeError):
    """Explicit test boundary: no model constructor or music inference runs."""


class NativeLoadingTests(unittest.TestCase):
    """Run real copy/hash/role guards through the production load entry point."""

    def setUp(self) -> None:
        """Reuse tiny data/code fixtures; they are never interpreted as model tensors."""
        self.fixture = check_weights.NativeWeightsTests()
        self.fixture.setUp()
        self.config = check_support.configuration(self.fixture.root).model_copy(
            update={"weights": self.fixture.weights}
        )
        self.imports = 0
        self.progress: list[NativeMusicProgress] = []

    def tearDown(self) -> None:
        """Remove only this test's completed/partial private preparation."""
        self.fixture.tearDown()

    def import_boundary(self) -> NativeAPI:
        """Inspect complete verified custody at the first possible model import."""
        self.imports += 1
        view = Path(self.config.directory) / "loader" / "checkpoints"
        self.assertEqual(len([p for p in view.rglob("*") if p.is_file()]), 17)
        self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
        self.assertEqual(os.environ["ACESTEP_GENERATION_TIMEOUT"], "3600")
        self.assertEqual(os.environ["ACESTEP_CHECKPOINTS_DIR"], str(view))
        self.assertEqual(
            os.environ["ACESTEP_PROJECT_ROOT"],
            str(Path(self.config.directory) / "loader"),
        )
        self.assertEqual(os.environ["ACESTEP_VAE_ON_CPU"], "0")
        self.assertEqual(os.environ["ACESTEP_MPS_DECODE_OFFLOAD"], "0")
        self.assertNotIn("ACESTEP_VAE_DECODE_CHUNK_SIZE", os.environ)
        self.assertEqual(os.environ["MIOPEN_FIND_MODE"], "FAST")
        self.assertEqual(os.environ["ACESTEP_ROCM_DTYPE"], "float32")
        raise StopBeforeInferenceError(
            "complete bytes; stop before native import/model loading"
        )

    def invoke(self) -> None:
        """Use the actual engine loader while retaining this interpreter's environment."""
        with patch.dict(
            os.environ,
            {
                "ACESTEP_VAE_ON_CPU": "1",
                "ACESTEP_MPS_DECODE_OFFLOAD": "1",
                "ACESTEP_GENERATION_TIMEOUT": "37",
                "ACESTEP_PROJECT_ROOT": "/unexpected/ui",
                "ACESTEP_VAE_DECODE_CHUNK_SIZE": "bogus",
            },
        ):
            load_music(
                self.config,
                "b" * 64,
                self.progress.append,
                runtime_root=self.fixture.runtime,
                code=self.fixture.code,
                api_loader=self.import_boundary,
            )

    def test_complete_three_component_and_owned_code_closure_precedes_native_import(
        self,
    ) -> None:
        """No implicit cache, data fragment or repository Python reaches a constructor."""
        with self.assertRaises(StopBeforeInferenceError):
            self.invoke()
        self.assertEqual(self.imports, 1)
        self.assertEqual([p.stage for p in self.progress], ["verifying_weights"])
        self.assertTrue(
            all(
                path.read_bytes() == data
                for path, data in self.fixture.originals.items()
            )
        )

    def test_changed_late_component_prevents_even_importing_native_model_api(
        self,
    ) -> None:
        """A base-only view cannot be advertised after a VAE object fails its digest."""
        item = self.fixture.weights.autoencoder.files[-1]
        (Path(self.fixture.weights.autoencoder.directory) / item.path).write_bytes(
            b"x" * item.size_bytes
        )
        with self.assertRaises(ValueError):
            self.invoke()
        self.assertEqual(self.imports, 0)
        self.assertFalse((Path(self.config.directory) / "ready.json").exists())

    def test_real_initializer_admits_verified_view_through_a_parent_path_alias(
        self,
    ) -> None:
        """A platform's temporary-directory alias preserves the exact verified view."""
        alias = self.fixture.root / "parent-alias"
        alias.symlink_to(self.fixture.root, target_is_directory=True)
        configuration = self.config.model_copy(
            update={"directory": str(alias / "model")}
        )
        api = native_api()
        handler = api.handler()
        with (
            patch.dict(os.environ),
            patch.object(handler, "_sync_model_code_if_needed", return_value=None),
            patch.object(
                handler,
                "_load_main_model_from_checkpoint",
                side_effect=StopBeforeInferenceError(
                    "controlled stop at first native tensor loader"
                ),
            ) as tensor_loader,
        ):
            with self.assertRaisesRegex(
                RuntimeError, "controlled stop at first native tensor loader"
            ):
                load_music(
                    configuration,
                    "b" * 64,
                    self.progress.append,
                    runtime_root=self.fixture.runtime,
                    code=self.fixture.code,
                    api_loader=lambda: NativeAPI(
                        handler=lambda: handler,
                        constructors=api.constructors,
                        generate=api.generate,
                    ),
                )
            tensor_loader.assert_called_once()
