# pyright: reportPrivateUsage=false
"""Provisioning tests: variant selection, verification, wiring, overrides."""

import hashlib
import io
import os
import tarfile
from pathlib import Path

import pytest

import skulk.provisioning.llama_server as provisioning
from skulk.facts.testing import AMD_STRIX, NVIDIA_A40, make_facts, ok_bin
from skulk.provisioning.llama_server import (
    ensure_llama_server,
    provision_llama_server,
    select_variant,
    select_variant_chain,
)
from skulk.provisioning.manifest import (
    LLAMA_SERVER_ARTIFACTS,
    LLAMA_SERVER_PIN,
    EngineArtifact,
)
from skulk.shared.backends import LLAMA_SERVER_BIN_ENV


def test_select_variant_chain_by_gpu_vendor() -> None:
    # NVIDIA prefers the Foxlight CUDA build (container clouds have no usable
    # Vulkan ICD) with Vulkan as the bare-metal-friendly fallback; AMD uses
    # the fleet-proven RADV Vulkan path.
    assert select_variant_chain(make_facts(gpus=(NVIDIA_A40,))) == ("cuda", "vulkan")
    assert select_variant_chain(make_facts(gpus=(AMD_STRIX,))) == ("vulkan",)
    assert select_variant(make_facts(gpus=(NVIDIA_A40,))) == "cuda"
    assert select_variant(make_facts(gpus=(AMD_STRIX,))) == "vulkan"


def test_select_variant_cpu_without_gpu() -> None:
    assert select_variant(make_facts()) == "cpu"


def test_select_variant_none_on_darwin() -> None:
    assert select_variant(make_facts(platform="darwin")) is None


def _fake_archive(binary_relpath: str = "build/bin/llama-server") -> bytes:
    """A gzipped tarball containing a fake llama-server binary."""
    payload = b"#!/bin/sh\necho fake llama-server\n"
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        info = tarfile.TarInfo(binary_relpath)
        info.size = len(payload)
        info.mode = 0o755
        tar.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def _pin_fake_artifact(
    monkeypatch: pytest.MonkeyPatch, archive: bytes, *, corrupt_checksum: bool = False
) -> None:
    """Point the manifest at a synthetic artifact and stub the download."""
    sha256 = hashlib.sha256(archive).hexdigest()
    if corrupt_checksum:
        sha256 = "0" * 64
    artifact = EngineArtifact(asset_name="fake.tar.gz", sha256=sha256)
    import platform as platform_module

    monkeypatch.setitem(
        LLAMA_SERVER_ARTIFACTS, (platform_module.machine(), "vulkan"), artifact
    )

    def _fake_download(art: EngineArtifact, destination: Path) -> None:
        destination.write_bytes(archive)

    monkeypatch.setattr(provisioning, "_download", _fake_download)


def _isolate_engines_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    engines = tmp_path / "engines"
    monkeypatch.setattr(provisioning, "SKULK_ENGINES_DIR", engines)
    return engines


def test_provision_downloads_verifies_and_installs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engines = _isolate_engines_dir(monkeypatch, tmp_path)
    _pin_fake_artifact(monkeypatch, _fake_archive())
    binary = provision_llama_server("vulkan")
    assert binary.is_file()
    assert os.access(binary, os.X_OK)
    assert binary.is_relative_to(engines / "llama-server" / LLAMA_SERVER_PIN)


def test_provision_is_idempotent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_engines_dir(monkeypatch, tmp_path)
    _pin_fake_artifact(monkeypatch, _fake_archive())
    first = provision_llama_server("vulkan")

    def _must_not_download(art: EngineArtifact, destination: Path) -> None:
        raise AssertionError("re-downloaded an already-provisioned build")

    monkeypatch.setattr(provisioning, "_download", _must_not_download)
    assert provision_llama_server("vulkan") == first


def test_provision_rejects_checksum_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_engines_dir(monkeypatch, tmp_path)
    _pin_fake_artifact(monkeypatch, _fake_archive(), corrupt_checksum=True)
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        provision_llama_server("vulkan")


def test_provision_rejects_archive_without_binary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_engines_dir(monkeypatch, tmp_path)
    _pin_fake_artifact(monkeypatch, _fake_archive(binary_relpath="build/bin/other"))
    with pytest.raises(RuntimeError, match="no llama-server binary"):
        provision_llama_server("vulkan")


def test_ensure_honors_explicit_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # An operator's SKULK_LLAMA_SERVER_BIN wins; provisioning never runs.
    _isolate_engines_dir(monkeypatch, tmp_path)
    facts = make_facts(
        gpus=(NVIDIA_A40,), llama_server_bin=ok_bin(LLAMA_SERVER_BIN_ENV)
    )
    assert ensure_llama_server(facts) is None


def test_ensure_honors_opt_out(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_engines_dir(monkeypatch, tmp_path)
    monkeypatch.setenv(provisioning.AUTOPROVISION_OPT_OUT_ENV, "1")
    assert ensure_llama_server(make_facts(gpus=(NVIDIA_A40,))) is None


def test_ensure_provisions_and_exports_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The end-to-end just-works wiring: no override, Linux GPU node ->
    # download, verify, and export SKULK_LLAMA_SERVER_BIN for this process.
    _isolate_engines_dir(monkeypatch, tmp_path)
    _pin_fake_artifact(monkeypatch, _fake_archive())
    monkeypatch.delenv(provisioning.AUTOPROVISION_OPT_OUT_ENV, raising=False)
    monkeypatch.delenv(LLAMA_SERVER_BIN_ENV, raising=False)
    binary = ensure_llama_server(make_facts(gpus=(NVIDIA_A40,)))
    assert binary is not None
    assert os.environ[LLAMA_SERVER_BIN_ENV] == str(binary)


def test_ensure_swallows_download_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A node must start without network: provisioning failure degrades to
    # "no served engine" with a warning, never a crash.
    _isolate_engines_dir(monkeypatch, tmp_path)

    def _fail(art: EngineArtifact, destination: Path) -> None:
        raise OSError("network down")

    monkeypatch.setattr(provisioning, "_download", _fail)
    monkeypatch.delenv(provisioning.AUTOPROVISION_OPT_OUT_ENV, raising=False)
    assert ensure_llama_server(make_facts(gpus=(NVIDIA_A40,))) is None
