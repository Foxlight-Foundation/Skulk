"""Native video package integrity, host eligibility and offline/cancel boundaries."""

from __future__ import annotations

import hashlib
import json
import sys
import threading
import zipfile
from pathlib import Path

import httpx
import pytest
from packaging.tags import Tag

from skulk.facts.derive import derive_node_backends
from skulk.facts.inventory import engine_build_inventory
from skulk.facts.testing import APPLE_GPU, make_facts
from skulk.provisioning import mlx_video


def _cache(root: Path) -> mlx_video.MlxVideoWheel:
    payload = {"skulk_mlx_video/__init__.py": b'"""Test engine."""\n'}
    members = {name: hashlib.sha256(data).hexdigest() for name, data in payload.items()}
    identity = hashlib.sha256(json.dumps(members, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    payload["skulk_mlx_video/build.json"] = json.dumps({"source_revision": mlx_video.SOURCE_REVISION, "payload_sha256": identity, "members": members}).encode()
    name = "skulk_mlx_video-0.1.0-cp313-cp313-macosx_15_0_arm64.whl"
    root.mkdir(parents=True)
    package = root / name
    with zipfile.ZipFile(package, "w") as archive:
        for member, data in payload.items():
            archive.writestr(member, data)
    libraries = root / "venv/lib/python3.13/site-packages"
    for member, data in payload.items():
        target = libraries / member
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    interpreter = root / "venv/bin/python"
    interpreter.parent.mkdir()
    interpreter.symlink_to(sys.executable)
    (root / "venv/pyvenv.cfg").write_text(f"home = {Path(sys.executable).resolve().parent}\nimplementation = CPython\nversion_info = 3.13\ninclude-system-site-packages = false\n")
    return mlx_video.MlxVideoWheel(filename=name, sha256=hashlib.sha256(package.read_bytes()).hexdigest(), payload_sha256=identity, size_bytes=package.stat().st_size)


def _host(monkeypatch: pytest.MonkeyPatch, wheel: mlx_video.MlxVideoWheel) -> None:
    monkeypatch.setattr(mlx_video, "PUBLISHED_WHEELS", (wheel,))
    monkeypatch.setattr(mlx_video.sys, "platform", "darwin")
    monkeypatch.setattr(mlx_video.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(mlx_video, "sys_tags", lambda: iter([Tag("cp313", "cp313", "macosx_15_0_arm64")]))


def test_untested_apple_gpu_is_eligible_for_published_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wheel = _cache(tmp_path / "cache")
    _host(monkeypatch, wheel)
    facts = make_facts(platform="darwin", gpus=(APPLE_GPU.model_copy(update={"name": "Previously untested Apple GPU"}),))
    monkeypatch.setenv("SKULK_NODE_PARTICIPATION", "full")
    monkeypatch.delenv("SKULK_NO_AUTO_PROVISION", raising=False)
    monkeypatch.setattr(mlx_video, "SKULK_ENABLE_VIDEO_MODELS", True)
    def available_tool(_program: str) -> str:
        return "/tools/uv"
    monkeypatch.setattr(mlx_video.shutil, "which", available_tool)
    builds = mlx_video.installable_video_builds(facts, offline=False)
    assert builds == {"mlx_video-metal": wheel.build_identity}
    offered = facts.model_copy(update={"mlx_video_installable_builds": builds})
    derivation = derive_node_backends(offered)
    assert derivation.on_demand_backends == frozenset(builds)
    assert "mlx_video-metal" not in engine_build_inventory(derivation.installed_backends, offered, environ={})
    ready = offered.model_copy(update={"mlx_video_build_identity": wheel.build_identity})
    assert "mlx_video-metal" in derive_node_backends(ready).installed_backends
    assert engine_build_inventory(frozenset({"mlx_video-metal"}), ready, environ={"SKULK_ENGINE_BUILDS": '{"mlx_video-metal":"fake"}'}) == {**builds, "mlx_video": wheel.build_identity}
    assert mlx_video.installable_video_builds(facts, offline=True) == {}


def test_real_abi_and_os_requirements_remain_enforced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wheel = _cache(tmp_path / "cache")
    _host(monkeypatch, wheel)
    assert mlx_video.wheel_for_host() == wheel
    monkeypatch.setattr(mlx_video, "sys_tags", lambda: iter([Tag("cp313", "cp313", "macosx_14_0_arm64")]))
    assert mlx_video.wheel_for_host() is None


def test_compatible_interpreter_upgrade_preserves_previous_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wheel = _cache(tmp_path / "cache")
    original = mlx_video.runtime_root(wheel)
    monkeypatch.setattr(mlx_video.sys, "executable", str(tmp_path / "new-python"))
    assert mlx_video.runtime_root(wheel) != original
    monkeypatch.setattr(mlx_video.sys, "platform", "linux")
    assert mlx_video.wheel_for_host() is None


@pytest.mark.parametrize("corruption", ["code", "manifest", "archive", "extra_pth", "extra_python", "symlink"])
def test_cache_integrity_uses_trusted_package_bytes(tmp_path: Path, corruption: str) -> None:
    root = tmp_path / "cache"
    wheel = _cache(root)
    assert mlx_video.verified_interpreter(root, wheel) is not None
    libraries = root / "venv/lib/python3.13/site-packages"
    if corruption == "code":
        (libraries / "skulk_mlx_video/__init__.py").write_text("raise RuntimeError('changed')")
    elif corruption == "manifest":
        (libraries / "skulk_mlx_video/build.json").write_text("{}")
    elif corruption == "archive":
        (root / wheel.filename).write_bytes(b"changed")
    elif corruption in {"extra_pth", "extra_python"}:
        (libraries / ("extra.pth" if corruption == "extra_pth" else "sitecustomize.py")).write_text("import os")
    else:
        path = libraries / "skulk_mlx_video/__init__.py"
        path.unlink()
        path.symlink_to(__file__)
    assert mlx_video.verified_interpreter(root, wheel) is None


def test_offline_reuse_and_cancel_do_not_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "cache"
    wheel = _cache(root)
    _host(monkeypatch, wheel)
    def cached_root(_wheel: mlx_video.MlxVideoWheel) -> Path:
        return root
    monkeypatch.setattr(mlx_video, "runtime_root", cached_root)
    probed: list[str] = []
    def record_probe(_interpreter: Path, build: str) -> None:
        probed.append(build)
    monkeypatch.setattr(mlx_video, "probe_runtime", record_probe)
    path = mlx_video.install_mlx_video("mlx_video-metal", wheel.build_identity, offline=True, stop=threading.Event())
    assert path == root / "venv/bin/python" and probed == [wheel.build_identity]
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(mlx_video.MlxVideoInstallError, match="cancelled"):
        mlx_video.install_mlx_video("mlx_video-metal", wheel.build_identity, offline=False, stop=cancelled)
    def absent_root(_wheel: mlx_video.MlxVideoWheel) -> Path:
        return tmp_path / "absent"
    monkeypatch.setattr(mlx_video, "runtime_root", absent_root)
    with pytest.raises(mlx_video.MlxVideoInstallError, match="not cached"):
        mlx_video.install_mlx_video("mlx_video-metal", wheel.build_identity, offline=True, stop=threading.Event())


def test_runtime_environment_blocks_independent_downloads_and_injection() -> None:
    environment = mlx_video.runtime_environment({"HF_TOKEN": "private", "PYTHONPATH": "/other", "DYLD_INSERT_LIBRARIES": "other", "PATH": "/usr/bin"})
    assert "HF_TOKEN" not in environment and "PYTHONPATH" not in environment and "DYLD_INSERT_LIBRARIES" not in environment
    assert environment["HF_HUB_OFFLINE"] == "1" and environment["TRANSFORMERS_OFFLINE"] == "1"


def test_corrupt_download_never_promotes_or_executes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    wheel = _cache(tmp_path / "trusted")
    _host(monkeypatch, wheel)
    root = tmp_path / "runtime"
    def destination(_wheel: mlx_video.MlxVideoWheel) -> Path:
        return root
    monkeypatch.setattr(mlx_video, "runtime_root", destination)
    client = httpx.Client
    def respond(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == wheel.url
        return httpx.Response(200, content=b"x" * wheel.size_bytes)
    def create_client(*, timeout: httpx.Timeout) -> httpx.Client:
        return client(timeout=timeout, transport=httpx.MockTransport(respond))
    monkeypatch.setattr(mlx_video.httpx, "Client", create_client)
    with pytest.raises(mlx_video.MlxVideoInstallError, match="integrity check"):
        mlx_video.install_mlx_video("mlx_video-metal", wheel.build_identity, offline=False, stop=threading.Event())
    assert not root.exists() and list(tmp_path.glob(".mlx-video-*")) == []
