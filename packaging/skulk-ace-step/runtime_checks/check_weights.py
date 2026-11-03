"""Complete data custody and runtime-owned model-code preparation, without inference."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from skulk_ace_step.contracts import (
    NativeMusicComponent,
    NativeMusicWeights,
)
from skulk_ace_step.weights import ModelCodeFile, prepare_weights


def object_identity(data: bytes) -> str:
    """Compute the actual Git blob identity used by small component metadata."""
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {len(data)}\0".encode() + data)
    return "git-sha1:" + digest.hexdigest()


class NativeWeightsTests(unittest.TestCase):
    """Real file selection/copy checks; tiny fixture tensors are never loaded."""

    def setUp(self) -> None:
        """Create separate immutable-source and runtime-source fixture roots."""
        self.temporary = tempfile.TemporaryDirectory(prefix="native-music-weights-")
        self.root = Path(self.temporary.name)
        self.runtime = self.root / "runtime"
        self.originals: dict[Path, bytes] = {}
        configuration = json.dumps(
            dict(
                model_type="acestep",
                hidden_size=2048,
                num_hidden_layers=24,
                architectures=["AceStepConditionGenerationModel"],
                auto_map={
                    "AutoConfig": "configuration_acestep_v15.AceStepConfig",
                    "AutoModel": "modeling_acestep_v15_base.AceStepConditionGenerationModel",
                },
            )
        ).encode()

        def component(kind: str, names: tuple[str, ...]) -> NativeMusicComponent:
            directory = self.root / "sources" / kind
            prefix = kind + "-root/"
            files = []
            for name in names:
                data = (
                    configuration
                    if kind == "base" and name == "config.json"
                    else (kind + ":" + name).encode()
                )
                path = directory / (prefix + name)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                self.originals[path] = data
                identity = (
                    "sha256:" + hashlib.sha256(data).hexdigest()
                    if name.endswith((".safetensors", ".pt"))
                    else object_identity(data)
                )
                files.append(
                    dict(path=prefix + name, size_bytes=len(data), object_id=identity)
                )
            return NativeMusicComponent.model_validate_json(
                json.dumps(
                    dict(
                        kind=kind,
                        directory=str(directory),
                        root=kind + "-root",
                        files=files,
                    )
                )
            )

        self.weights = NativeMusicWeights(
            base=component(
                "base", ("config.json", "model.safetensors", "silence_latent.pt")
            ),
            text_encoder=component(
                "text_encoder",
                (
                    "added_tokens.json",
                    "chat_template.jinja",
                    "config.json",
                    "merges.txt",
                    "model.safetensors",
                    "special_tokens_map.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "vocab.json",
                ),
            ),
            autoencoder=component(
                "autoencoder", ("config.json", "diffusion_pytorch_model.safetensors")
            ),
        )
        self.code: list[ModelCodeFile] = []
        for name in (
            "configuration_acestep_v15.py",
            "modeling_acestep_v15_base.py",
            "apg_guidance.py",
        ):
            relative = "acestep/models/base/" + name
            data = ("# reviewed runtime file: " + name + "\n").encode()
            path = self.runtime / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            self.code.append(
                ModelCodeFile.model_validate_json(
                    json.dumps(
                        dict(
                            source=relative,
                            destination=name,
                            size_bytes=len(data),
                            sha256=hashlib.sha256(data).hexdigest(),
                        )
                    )
                )
            )

    def tearDown(self) -> None:
        """Remove only the fixture's private temporary tree."""
        self.temporary.cleanup()

    def test_all_components_are_verified_and_hub_python_is_not_copied(self) -> None:
        """Mirrored loader roots preserve roles; only runtime-owned Python enters the view."""
        extra = (
            Path(self.weights.base.directory) / "base-root/modeling_acestep_v15_base.py"
        )
        extra.write_bytes(b"untrusted repository Python")
        view = prepare_weights(
            self.root / "view", self.weights, self.runtime, self.code
        )
        files = [path for path in view.rglob("*") if path.is_file()]
        self.assertEqual(len(files), 17)
        self.assertTrue((view / "Qwen3-Embedding-0.6B/tokenizer.json").is_file())
        self.assertTrue((view / "vae/diffusion_pytorch_model.safetensors").is_file())
        self.assertEqual(
            (view / "acestep-v15-base/modeling_acestep_v15_base.py").read_bytes(),
            (
                self.runtime / "acestep/models/base/modeling_acestep_v15_base.py"
            ).read_bytes(),
        )
        self.assertEqual(extra.read_bytes(), b"untrusted repository Python")
        self.assertTrue(
            all(path.read_bytes() == data for path, data in self.originals.items())
        )

    def test_changed_component_fails_without_reusing_partial_preparation(self) -> None:
        """A late digest failure remains owned partial work, never a ready receipt."""
        item = self.weights.autoencoder.files[-1]
        path = Path(self.weights.autoencoder.directory) / item.path
        path.write_bytes(b"x" * item.size_bytes)
        target = self.root / "partial"
        with self.assertRaises(ValueError):
            prepare_weights(target, self.weights, self.runtime, self.code)
        self.assertTrue(target.is_dir())
        with self.assertRaises(FileExistsError):
            prepare_weights(target, self.weights, self.runtime, self.code)

    def test_runtime_code_hash_is_checked_independently_of_model_data(self) -> None:
        """Complete valid model files do not authorize changed implementation code."""
        item = self.code[0]
        (self.runtime / item.source).write_bytes(b"x" * item.size_bytes)
        with self.assertRaises(ValueError):
            prepare_weights(
                self.root / "changed-code", self.weights, self.runtime, self.code
            )

    def test_external_model_code_reference_is_rejected_even_with_matching_data_hash(
        self,
    ) -> None:
        """A signed JSON object's identity does not permit arbitrary cached/remote Python."""
        body = self.weights.model_dump(mode="json")
        component = body["base"]
        config = next(
            item for item in component["files"] if item["path"].endswith("/config.json")
        )
        path = Path(component["directory"]) / config["path"]
        payload = json.loads(path.read_bytes())
        payload["auto_map"]["AutoModel"] = "untrusted/repository--other.Model"
        data = json.dumps(payload).encode()
        path.write_bytes(data)
        config.update(size_bytes=len(data), object_id=object_identity(data))
        changed = NativeMusicWeights.model_validate_json(json.dumps(body))
        with self.assertRaises(ValueError):
            prepare_weights(
                self.root / "external-code", changed, self.runtime, self.code
            )

    def test_missing_code_is_refused_before_creating_a_view(self) -> None:
        """A valid complete model cannot load with an incomplete reviewed implementation."""
        target = self.root / "missing-code"
        with self.assertRaises(ValueError):
            prepare_weights(target, self.weights, self.runtime, self.code[:2])
        self.assertFalse(target.exists())

    def test_symbolic_component_is_refused(self) -> None:
        """Private model preparation never follows redirected source files."""
        item = self.weights.base.files[1]
        path = Path(self.weights.base.directory) / item.path
        original = path.with_name("saved.safetensors")
        path.rename(original)
        path.symlink_to(original)
        with self.assertRaises(OSError):
            prepare_weights(
                self.root / "symbolic", self.weights, self.runtime, self.code
            )


if __name__ == "__main__":
    unittest.main()
