"""Exact-byte overlays: isolation, atomic installation, offline reuse and cancellation."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import threading
import zipfile
from collections.abc import Sequence
from pathlib import Path

import pytest

from skulk.provisioning import comfy_extension as extension
from skulk.provisioning.manifest import COMFY_PIN

BASE_BUILD = f"comfy@{COMFY_PIN}/torch@2.14.0+cu130"
SOURCE = Path(__file__).resolve().parents[4] / "packaging/skulk-comfy-ltx"


def candidate(tmp_path: Path) -> tuple[extension.ComfyExtensionWheel, Path]:
    """Build actual reviewed package bytes, without publishing or executing them."""
    result = subprocess.run([sys.executable, str(SOURCE.parents[1] / "scripts/build_comfy_ltx_wheel.py"), "--source", str(SOURCE), "--output", str(tmp_path / "candidate")], check=True, capture_output=True, text=True)
    path = Path(result.stdout.strip())
    assert path.parent == tmp_path / "candidate"
    with zipfile.ZipFile(path) as archive:
        manifest = extension.ComfyExtensionManifest.model_validate_json(archive.read("skulk_comfy_ltx/manifest.json"))
    wheel = extension.ComfyExtensionWheel(filename=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), payload_sha256=manifest.payload_sha256, size_bytes=path.stat().st_size, features=manifest.features, comfy_revision=manifest.comfy_revision, ltx_revision=manifest.ltx_revision)
    return wheel, path


def test_candidate_admission_requires_its_exact_published_descriptor(tmp_path: Path) -> None:
    """Source candidates stay unavailable until exact-byte promotion registers them."""
    built, _ = candidate(tmp_path)
    issued = extension.published_extensions()
    promoted = next((wheel for wheel in issued if wheel.filename == built.filename), None)
    if promoted is not None:
        assert promoted == built
    else:
        # CI must assemble and attest new versions before a descriptor can
        # truthfully name their published bytes. The actual candidate must
        # remain inadmissible while the previous issued version still works.
        version = tuple(int(part) for part in built.filename.split("-")[1].split("."))
        assert all(version > tuple(int(part) for part in wheel.filename.split("-")[1].split(".")) for wheel in issued)
        build = built.build_identity(BASE_BUILD)
        assert extension.extension_for_build(build) is None
        assert extension.extension_features_for_build(build) == frozenset()


def install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, offline: bool = False, stop: threading.Event | None = None, fail_probe: bool = False) -> tuple[extension.ComfyExtensionWheel, Path, list[str]]:
    """Drive the real installer with bounded local download and probe effects."""
    wheel, archive = candidate(tmp_path)
    monkeypatch.setattr(extension, "SKULK_ENGINES_DIR", tmp_path / "engines")
    calls: list[str] = []

    def download(url: str, destination: Path, size: int, stopping: threading.Event) -> None:
        assert url == wheel.url and size == wheel.size_bytes and not stopping.is_set()
        calls.append("download")
        shutil.copyfile(archive, destination)

    def probe(interpreter: Path, package: Path, base: str, backend: str, stopping: threading.Event) -> None:
        assert interpreter == Path(sys.executable) and backend == "comfy-cuda" and base == BASE_BUILD
        assert extension.verify_extension_root(package.parents[1], wheel, base, interpreter) == package
        assert not stopping.is_set()
        calls.append("probe")
        if fail_probe:
            raise ValueError("controlled GPU probe failure")

    path = extension.install_extension_payload(wheel, BASE_BUILD, Path(sys.executable), "comfy-cuda", offline=offline, stop=stop or threading.Event(), download=download, probe=probe)
    return wheel, path, calls


def test_install_is_atomic_and_bound_to_archive_base_and_interpreter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wheel, package, calls = install(tmp_path, monkeypatch)
    root = package.parents[1]
    assert calls == ["download", "probe"]
    assert extension.verify_extension_root(root, wheel, BASE_BUILD, Path(sys.executable)) == package
    assert extension.verify_extension_root(root, wheel, BASE_BUILD.replace("cu130", "cu131"), Path(sys.executable)) is None
    assert not list(root.parent.glob(f".{root.name}-*"))
    assert not (Path(sys.executable).parent / "skulk_comfy_ltx").exists()


@pytest.mark.parametrize("mutation", ["source", "manifest", "archive", "bytecode", "library_symlink"])
def test_modified_or_expanded_installation_never_reports_ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str) -> None:
    wheel, package, _ = install(tmp_path, monkeypatch)
    root = package.parents[1]
    if mutation == "source":
        path = package / "nodes.py"
        path.chmod(0o644)
        _ = path.write_text("raise RuntimeError('modified')")
    elif mutation == "manifest":
        path = package / "manifest.json"
        path.chmod(0o644)
        _ = path.write_text(path.read_text().replace(extension.LTX_REFERENCE_REVISION, "a" * 40))
    elif mutation == "archive":
        _ = (root / wheel.filename).write_bytes(b"different archive")
    elif mutation == "bytecode":
        (package / "__pycache__").mkdir()
        _ = (package / "__pycache__/nodes.pyc").write_bytes(b"unchecked bytecode")
    else:
        library = root / "lib"
        library.rename(root / "other")
        library.symlink_to(root / "other", target_is_directory=True)
    assert extension.verify_extension_root(root, wheel, BASE_BUILD, Path(sys.executable)) is None


def test_offline_cache_reuse_still_probes_the_selected_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wheel, package, _ = install(tmp_path, monkeypatch)
    calls: list[Path] = []

    def download(_url: str, _destination: Path, _size: int, _stop: threading.Event) -> None:
        raise AssertionError("offline reuse must not download")

    def probe(interpreter: Path, selected: Path, _base: str, _backend: str, _stop: threading.Event) -> None:
        assert interpreter == Path(sys.executable)
        calls.append(selected)

    observed = extension.install_extension_payload(wheel, BASE_BUILD, Path(sys.executable), "comfy-cuda", offline=True, stop=threading.Event(), download=download, probe=probe)
    assert observed == package and calls == [package]
    shutil.rmtree(package)
    with pytest.raises(ValueError, match="offline"):
        extension.install_extension_payload(wheel, BASE_BUILD, Path(sys.executable), "comfy-cuda", offline=True, stop=threading.Event(), download=download, probe=probe)


def test_probe_failure_and_cancellation_leave_no_ready_or_staged_overlay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="controlled GPU"):
        install(tmp_path / "probe", monkeypatch, fail_probe=True)
    assert not list((tmp_path / "probe/engines").rglob("installed.json"))
    assert not list((tmp_path / "probe/engines").rglob(".*-*"))
    stopped = threading.Event()
    stopped.set()
    with pytest.raises(InterruptedError, match="cancelled"):
        install(tmp_path / "cancel", monkeypatch, stop=stopped)
    assert not list((tmp_path / "cancel/engines").rglob("installed.json"))


def test_unpublished_build_never_executes_or_downloads(tmp_path: Path) -> None:
    build = BASE_BUILD + "/extension@sha256:" + "a" * 64
    assert extension.extension_payload(build) == "a" * 64
    assert extension.extension_features_for_build(build) == frozenset()

    def run(_command: Sequence[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise AssertionError("unpublished candidates must not execute")

    with pytest.raises(ValueError, match="unavailable"):
        extension.install_comfy_extension(build, tmp_path / "python", "comfy-cuda", offline=False, stop=threading.Event(), run=run)


@pytest.mark.parametrize("offline,opt_out", [(False, False), (True, False), (False, True)])
def test_overlay_upgrade_offer_does_not_hide_the_installed_base_or_older_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, offline: bool, opt_out: bool) -> None:
    """New installs use the offered build; retained instances keep their exact bytes."""
    from skulk.facts import inventory
    from skulk.facts.derive import derive_node_backends
    from skulk.facts.testing import NVIDIA_A40, make_facts
    from skulk.shared import constants
    from skulk.shared.types.node_facts import EngineBinaryFact

    older, _, _ = install(tmp_path, monkeypatch)
    newer = older.model_copy(update={"filename": "skulk_comfy_ltx-0.2.0-py3-none-any.whl", "payload_sha256": "b" * 64})
    monkeypatch.setattr(extension, "PUBLISHED_EXTENSIONS", (newer, older))
    monkeypatch.setattr(constants, "SKULK_ENABLE_VIDEO_MODELS", True)

    def measured_base(_facts: object) -> str:
        return BASE_BUILD

    monkeypatch.setattr(inventory, "comfy_base_build", measured_base)
    facts = make_facts(platform="linux", gpus=(NVIDIA_A40.model_copy(update={"name": "Previously untested NVIDIA GPU"}),)).model_copy(update={"comfy_binary": EngineBinaryFact(env_var="SKULK_COMFY_BIN", configured_path=sys.executable, state="ok"), "comfy_root": str(tmp_path), "comfy_root_state": "ok", "declared_comfy_backends": "cuda"})
    offers, installed = extension.comfy_extension_inventory(facts, offline=offline, environ={"SKULK_NO_AUTO_PROVISION": "1" if opt_out else "0"})
    expected = older if offline or opt_out else newer
    assert offers == {"comfy-cuda": expected.build_identity(BASE_BUILD)}
    assert installed == {"comfy-cuda": older.build_identity(BASE_BUILD)}
    facts = facts.model_copy(update={"comfy_extension_installable_builds": offers, "comfy_extension_installed_builds": installed})
    derivation = derive_node_backends(facts)
    assert "comfy-cuda" in derivation.installed_backends and not derivation.on_demand_backends
    assert inventory.engine_build_inventory(derivation.installed_backends, facts, environ={"SKULK_ENGINE_BUILDS": '{"comfy-cuda":"fake"}'})["comfy-cuda"] == installed["comfy-cuda"]


def test_issued_overlay_retains_its_own_source_pins_when_defaults_advance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A newer Core default cannot rewrite a retained payload's provenance."""
    wheel, package, _ = install(tmp_path, monkeypatch)
    monkeypatch.setattr(extension, "PUBLISHED_EXTENSIONS", (wheel,))
    build = wheel.build_identity(BASE_BUILD)
    monkeypatch.setattr(extension, "COMFY_PIN", "f" * 40)
    monkeypatch.setattr(extension, "LTX_REFERENCE_REVISION", "e" * 40)
    assert extension.extension_for_build(build) == wheel
    assert extension.extension_features_for_build(build) == frozenset(wheel.features)
    assert extension.verify_extension_root(package.parents[1], wheel, BASE_BUILD, Path(sys.executable)) == package
    wrong_base = BASE_BUILD.replace(wheel.comfy_revision, "f" * 40)
    assert extension.extension_for_build(wrong_base + "/extension@sha256:" + wheel.payload_sha256) is None
    with pytest.raises(ValueError, match="compatible"):
        wheel.build_identity(wrong_base)
