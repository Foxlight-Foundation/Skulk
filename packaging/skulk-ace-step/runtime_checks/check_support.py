"""Small contract-only process fixtures; none represents loaded music weights."""

import json
from pathlib import Path

from skulk_ace_step.contracts import (
    NativeMusicComponent,
    NativeMusicDirectoryIdentity,
    NativeMusicExecution,
    NativeMusicRequest,
    NativeMusicServeConfig,
    NativeMusicWeights,
)


def directory_identity(path: Path) -> NativeMusicDirectoryIdentity:
    """Observe a fixture-created private inode before the native wrapper writes."""
    facts = path.stat()
    return NativeMusicDirectoryIdentity(device=facts.st_dev, inode=facts.st_ino)


def configuration(root: Path) -> NativeMusicServeConfig:
    """Build a complete typed inventory for wrapper tests, never a loader fixture."""

    def component(kind: str, names: tuple[str, ...]) -> NativeMusicComponent:
        return NativeMusicComponent.model_validate_json(
            json.dumps(
                {
                    "kind": kind,
                    "directory": str(root / "weights" / kind),
                    "files": [
                        {
                            "path": name,
                            "size_bytes": 1,
                            "object_id": "sha256:" + "a" * 64,
                        }
                        for name in names
                    ],
                }
            )
        )

    lifetime = root / "model"
    lifetime.mkdir(mode=0o700)
    requests = lifetime / "requests"
    requests.mkdir(mode=0o700)
    loader = lifetime / "loader"
    loader.mkdir(mode=0o700)
    return NativeMusicServeConfig(
        model_identity="a" * 64,
        directory=str(lifetime),
        directory_identity=directory_identity(lifetime),
        requests_identity=directory_identity(requests),
        loader_identity=directory_identity(loader),
        device="cpu",
        precision="float32",
        weights=NativeMusicWeights(
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
        ),
    )


def execution() -> NativeMusicExecution:
    """Return the explicit CPU component policy used by small wrapper fixtures."""
    return NativeMusicExecution(
        device="cpu",
        precision="float32",
        text_encoder_device="cpu",
        text_encoder_precision="float32",
        autoencoder_device="cpu",
        autoencoder_precision="float32",
    )


def request(
    config: NativeMusicServeConfig, identity: str = "take-1"
) -> NativeMusicRequest:
    """Create an exact private output path for a one-second generate request."""
    directory = Path(config.directory) / "requests" / identity
    directory.mkdir(mode=0o700, exist_ok=True)
    return NativeMusicRequest(
        request_id=identity,
        operation="generate",
        prompt="Warm folk guitar",
        seed=12025,
        target_frames=48000,
        output_directory=str(directory),
        output_directory_identity=directory_identity(directory),
    )
