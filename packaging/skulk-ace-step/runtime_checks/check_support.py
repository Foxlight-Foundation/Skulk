"""Small contract-only process fixtures; none represents loaded music weights."""

import json
from pathlib import Path

from skulk_ace_step.contracts import (
    NativeMusicComponent,
    NativeMusicExecution,
    NativeMusicRequest,
    NativeMusicServeConfig,
    NativeMusicWeights,
)


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

    return NativeMusicServeConfig(
        model_identity="a" * 64,
        directory=str(root / "model"),
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
    return NativeMusicRequest(
        request_id=identity,
        operation="generate",
        prompt="Warm folk guitar",
        seed=12025,
        target_frames=48000,
        output_directory=str(Path(config.directory) / "requests" / identity),
    )
