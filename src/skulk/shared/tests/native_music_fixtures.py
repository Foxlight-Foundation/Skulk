"""Tiny native ACE data closure for byte-custody tests, never inference weights."""

import base64
import hashlib
import json
import tomllib
from pathlib import Path
from typing import cast

from skulk.shared.models.model_cards import ModelCard


def native_music_fixture(
    *, shared_primary: bool = False
) -> tuple[ModelCard, dict[tuple[str, str, str], bytes]]:
    """Return a validated exact fourteen-file card and its small original objects.

    ``shared_primary`` mirrors every role into the primary repository; otherwise
    the encoder and VAE share an external source, matching the initial card.
    Every tensor is opaque fixture data. This proves delivery, never model load.
    """
    source = Path(__file__).parents[1] / "models/tests/ace-native-card.toml"
    body = cast("dict[str, object]", tomllib.loads(source.read_text()))
    body["is_custom"] = True
    body["model_id"] = "test/native-music"
    body["source_repository"] = "test/native-base"
    body["source_revision"] = "a" * 40
    native = cast("dict[str, object]", cast("dict[str, object]", body["music"])["native"])
    selections = [("test/native-base", "a" * 40, cast("dict[str, object]", body["artifact_bundle"]))]
    for role in ("text_encoder", "autoencoder"):
        component = cast("dict[str, object]", native[role])
        component["repo"] = "test/native-base" if shared_primary else "test/native-components"
        component["revision"] = "a" * 40 if shared_primary else "b" * 40
        selections.append((str(component["repo"]), str(component["revision"]), cast("dict[str, object]", component["artifact_bundle"])))
    objects: dict[tuple[str, str, str], bytes] = {}
    total = 0
    for repository, revision, bundle in selections:
        manifest: list[dict[str, object]] = []
        size = 0
        for item in cast("list[dict[str, object]]", bundle["files"]):
            path = str(item["path"])
            data = b'{"model_type":"acestep","fixture":true}' if path.endswith("config.json") else ("fixture:" + path).encode()
            objects[repository, revision, path] = data
            item["size_bytes"] = len(data)
            if path.endswith((".safetensors", ".pt")):
                item["object_id"] = "sha256:" + hashlib.sha256(data).hexdigest()
            else:
                item["object_id"] = "git-sha1:" + hashlib.sha1(f"blob {len(data)}\0".encode() + data, usedforsecurity=False).hexdigest()
            size += len(data)
            manifest.append({"relative_path": path, "size_bytes": len(data), "object_id": item["object_id"]})
        digest = hashlib.sha256(json.dumps(sorted(manifest, key=lambda item: str(item["relative_path"])), sort_keys=True, separators=(",", ":")).encode()).digest()
        bundle["bundle_id"] = "bundle_" + base64.b32encode(digest).decode().rstrip("=").lower()
        bundle["download_size"] = size
        total += size
    body["storage_size"] = {"in_bytes": total}
    return ModelCard.model_validate(body), objects
