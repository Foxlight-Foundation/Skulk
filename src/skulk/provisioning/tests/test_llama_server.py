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


def test_ensure_offline_wires_existing_managed_install(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # An --offline restart must keep its served capability: the managed build
    # already on disk wires without any network touch (PR #615 review).
    _isolate_engines_dir(monkeypatch, tmp_path)
    _pin_fake_artifact(monkeypatch, _fake_archive())
    provisioned = provision_llama_server("vulkan")

    def _must_not_download(art: EngineArtifact, destination: Path) -> None:
        raise AssertionError("offline ensure touched the network")

    monkeypatch.setattr(provisioning, "_download", _must_not_download)
    monkeypatch.delenv(provisioning.AUTOPROVISION_OPT_OUT_ENV, raising=False)
    monkeypatch.delenv(LLAMA_SERVER_BIN_ENV, raising=False)
    wired = ensure_llama_server(make_facts(gpus=(NVIDIA_A40,)), allow_download=False)
    assert wired == provisioned
    assert os.environ[LLAMA_SERVER_BIN_ENV] == str(provisioned)


def test_ensure_offline_without_install_is_absent_not_downloading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_engines_dir(monkeypatch, tmp_path)

    def _must_not_download(art: EngineArtifact, destination: Path) -> None:
        raise AssertionError("offline ensure touched the network")

    monkeypatch.setattr(provisioning, "_download", _must_not_download)
    monkeypatch.delenv(provisioning.AUTOPROVISION_OPT_OUT_ENV, raising=False)
    assert (
        ensure_llama_server(make_facts(gpus=(NVIDIA_A40,)), allow_download=False)
        is None
    )


def test_ensure_offline_prefers_capable_variant_over_cpu(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # With a stale cpu install AND a vulkan install on disk, an offline GPU
    # node must wire the vulkan build, not the alphabetically-first cpu one
    # (PR #615 review).
    engines = _isolate_engines_dir(monkeypatch, tmp_path)
    for variant in ("cpu", "vulkan"):
        vdir = engines / "llama-server" / LLAMA_SERVER_PIN / variant
        vdir.mkdir(parents=True)
        binary = vdir / "llama-server"
        binary.write_text("#!/bin/sh\n")
        binary.chmod(0o755)
    monkeypatch.delenv(provisioning.AUTOPROVISION_OPT_OUT_ENV, raising=False)
    monkeypatch.delenv(LLAMA_SERVER_BIN_ENV, raising=False)
    wired = ensure_llama_server(make_facts(gpus=(NVIDIA_A40,)), allow_download=False)
    assert wired is not None
    assert "vulkan" in str(wired)


def test_interrupted_extraction_leaves_no_partial_install(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # An extraction failure must not leave a partial variant dir that would
    # satisfy the fast path unverified on the next start (PR #615 review).
    engines = _isolate_engines_dir(monkeypatch, tmp_path)
    _pin_fake_artifact(monkeypatch, _fake_archive())

    def _boom(archive: Path, destination: Path) -> None:
        (destination / "partial-file").write_text("half")
        raise OSError("disk full mid-extract")

    monkeypatch.setattr(provisioning, "_safe_extract", _boom)
    with pytest.raises(OSError):
        provision_llama_server("vulkan")
    target = engines / "llama-server" / LLAMA_SERVER_PIN / "vulkan"
    assert not target.exists()
    # Recovery: a later attempt with a working extractor installs cleanly.
    monkeypatch.undo()
    _isolate_engines_dir(monkeypatch, tmp_path)
    _pin_fake_artifact(monkeypatch, _fake_archive())
    assert provision_llama_server("vulkan").is_file()


def _fake_wheel(tmp_path: Path, name: str) -> Path:
    shim = tmp_path / name
    shim.write_text("#!/bin/sh\n")
    shim.chmod(0o755)
    return shim


def test_wheel_outranks_tarball_provisioning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A pip-installed engine wheel is the preferred managed source: no
    # download may happen when one is present, and its RPC donor shim is
    # exported alongside.
    _isolate_engines_dir(monkeypatch, tmp_path)
    shim = _fake_wheel(tmp_path, "llama-server-cuda")
    rpc = _fake_wheel(tmp_path, "ggml-rpc-server-cuda")

    def _fake_lookup(vendor: str) -> tuple[Path, Path | None] | None:
        return (shim, rpc) if vendor == "nvidia" else None

    monkeypatch.setattr(provisioning, "wheel_llama_server", _fake_lookup)

    def _must_not_download(art: EngineArtifact, destination: Path) -> None:
        raise AssertionError("downloaded despite an installed engine wheel")

    monkeypatch.setattr(provisioning, "_download", _must_not_download)
    monkeypatch.delenv(provisioning.AUTOPROVISION_OPT_OUT_ENV, raising=False)
    monkeypatch.delenv(LLAMA_SERVER_BIN_ENV, raising=False)
    monkeypatch.delenv(provisioning.RPC_SERVER_BIN_ENV, raising=False)
    wired = ensure_llama_server(make_facts(gpus=(NVIDIA_A40,)))
    assert wired == shim
    assert os.environ[LLAMA_SERVER_BIN_ENV] == str(shim)
    assert os.environ[provisioning.RPC_SERVER_BIN_ENV] == str(rpc)


def test_vulkan_wheel_wired_on_amd_nodes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # AMD nodes use the Vulkan wheel when installed; the tarball path is the
    # fallback, not the preference.
    _isolate_engines_dir(monkeypatch, tmp_path)
    shim = _fake_wheel(tmp_path, "llama-server-vulkan")

    def _fake_lookup(vendor: str) -> tuple[Path, Path | None] | None:
        return (shim, None) if vendor == "amd" else None

    monkeypatch.setattr(provisioning, "wheel_llama_server", _fake_lookup)
    monkeypatch.delenv(provisioning.AUTOPROVISION_OPT_OUT_ENV, raising=False)
    monkeypatch.delenv(LLAMA_SERVER_BIN_ENV, raising=False)
    wired = ensure_llama_server(make_facts(gpus=(AMD_STRIX,)))
    assert wired == shim


def test_tarball_fallback_when_no_wheel_installed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _isolate_engines_dir(monkeypatch, tmp_path)

    def _no_wheel(vendor: str) -> tuple[Path, Path | None] | None:
        return None

    monkeypatch.setattr(provisioning, "wheel_llama_server", _no_wheel)
    _pin_fake_artifact(monkeypatch, _fake_archive())
    monkeypatch.delenv(provisioning.AUTOPROVISION_OPT_OUT_ENV, raising=False)
    monkeypatch.delenv(LLAMA_SERVER_BIN_ENV, raising=False)
    wired = ensure_llama_server(make_facts(gpus=(AMD_STRIX,)))
    assert wired is not None
    assert "vulkan" in str(wired)
