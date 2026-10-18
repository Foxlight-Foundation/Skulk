"""A source upgrade preserves immutable engine pins for existing image instances."""

import hashlib
import json
import subprocess
import threading
import zipfile
from pathlib import Path

import httpx
import pytest

from skulk.provisioning import stable_diffusion_cpp as engines
from skulk.shared.backends import runtime_backends_for_builds

_OLD_SOURCE = "d637cfb935feb2254898a8f1524046c958f3df80"
_BACKEND = "stable_diffusion_cpp-metal"
_HARDWARE = frozenset({"apple:compatible-untested-gpu"})


@pytest.fixture
def packages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[
    engines.StableDiffusionCppWheel, engines.StableDiffusionCppWheel, dict[str, bytes]
]:
    """Build two independently hashed runtime payloads with different source pins."""
    wheels: list[engines.StableDiffusionCppWheel] = []
    archives: dict[str, bytes] = {}
    for version, source in (("post1", _OLD_SOURCE), ("post2", engines.SOURCE_REVISION)):
        prefix = "skulk_stable_diffusion_cpp_metal"
        executable = f"native-{version}".encode()
        members = {f"{prefix}/bin/sd-server": hashlib.sha256(executable).hexdigest()}
        payload = hashlib.sha256(json.dumps(members, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        filename = f"{prefix}-0.1.0.{version}-py3-none-macosx_15_0_arm64.whl"
        archive_path = tmp_path / filename
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr(f"{prefix}/bin/sd-server", executable)
            archive.writestr(f"{prefix}/build.json", json.dumps({"members": members, "payload_sha256": payload}))
        wheels.append(engines.StableDiffusionCppWheel(
            filename=filename, source_revision=source,
            sha256=engines.file_sha256(archive_path), binary_sha256=members[f"{prefix}/bin/sd-server"],
            payload_sha256=payload, compute="metal", system="darwin", architecture="arm64"))
        archives[wheels[-1].url] = archive_path.read_bytes()
    old, new = wheels
    monkeypatch.setattr(engines, "RETAINED_WHEELS", (old,))
    monkeypatch.setattr(engines, "PUBLISHED_WHEELS", (new,))
    monkeypatch.setattr(engines, "SKULK_ENGINES_DIR", tmp_path / "engines")
    monkeypatch.setattr(engines.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(engines.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(engines, "platform_tags", lambda: iter(("macosx_15_0_arm64",)))
    monkeypatch.delenv(engines.BINARY_ENV, raising=False)

    def probe(arguments: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        digest = engines.file_sha256(Path(arguments[0]))
        source = next(wheel.source_revision for wheel in wheels if wheel.binary_sha256 == digest)
        return subprocess.CompletedProcess(arguments, 0, json.dumps({
            "protocol": 1, "source_revision": source, "device_report": "MTL0\tApple GPU\n"}), "")

    monkeypatch.setattr(engines.subprocess, "run", probe)
    return old, new, archives


def _cache(wheel: engines.StableDiffusionCppWheel, archives: dict[str, bytes]) -> Path:
    root = engines.SKULK_ENGINES_DIR / "stable_diffusion_cpp" / wheel.source_revision / wheel.sha256
    root.mkdir(parents=True)
    archive_path = root / wheel.filename
    archive_path.write_bytes(archives[wheel.url])
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(root)
    binary = root / wheel.prefix / "bin/sd-server"
    binary.chmod(0o755)
    return binary


def test_offline_restart_restores_old_exact_cache_without_changing_default(
    packages: tuple[engines.StableDiffusionCppWheel, engines.StableDiffusionCppWheel, dict[str, bytes]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old, new, archives = packages
    old_binary = _cache(old, archives)
    new_binary = _cache(new, archives)
    assert engines.rehydrate_native_image(_HARDWARE) == new_binary
    assert engines.wheel_for_host(_BACKEND, _HARDWARE) == new
    assert engines.prepared_native_image(_BACKEND, old.build_identity, _HARDWARE) == old_binary
    assert engines.install_native_image(_BACKEND, old.build_identity, _HARDWARE,
        offline=True, stop=threading.Event()) == old_binary
    assert engines.probe_native_image(old_binary).source_revision == _OLD_SOURCE
    assert engines.os.environ[engines.BINARY_ENV] == str(new_binary)
    # The previous protocol also remains valid for explicit operator builds.
    monkeypatch.setattr(engines, "RETAINED_WHEELS", ())
    assert engines.probe_native_image(old_binary).build_identity == old.build_identity


def test_previous_source_operator_override_retains_exact_build_enforcement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An independently built old executable survives upgrade without joining the wheel catalog."""
    binary = tmp_path / "operator-server"
    binary.write_bytes(b"independent operator executable")
    expected = "stable_diffusion_cpp@sha256:" + engines.file_sha256(binary)
    source = _OLD_SOURCE

    def probe(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 0, json.dumps({
            "protocol": 1, "source_revision": source, "device_report": "MTL0\tApple GPU\n"}), "")

    monkeypatch.setattr(engines.subprocess, "run", probe)
    assert engines.probe_native_image(binary).build_identity == expected
    override = {engines.BINARY_ENV: str(binary)}
    assert engines.install_native_image(_BACKEND, expected, _HARDWARE,
        offline=True, stop=threading.Event(), environ=override) == binary
    with pytest.raises(engines.ImageEngineInstallError, match="override build"):
        engines.install_native_image(_BACKEND, "different-stamped-build", _HARDWARE,
            offline=True, stop=threading.Event(), environ=override)
    binary.write_bytes(b"replaced operator executable")
    with pytest.raises(engines.ImageEngineInstallError, match="override build"):
        engines.install_native_image(_BACKEND, expected, _HARDWARE,
            offline=True, stop=threading.Event(), environ=override)
    source = "f" * 40
    with pytest.raises(ValueError, match="unrecognized protocol or source"):
        engines.probe_native_image(binary)


def test_cold_retained_preparation_downloads_exact_old_bytes_beside_ready_new_default(
    packages: tuple[engines.StableDiffusionCppWheel, engines.StableDiffusionCppWheel, dict[str, bytes]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old, new, archives = packages
    new_binary = _cache(new, archives)
    monkeypatch.setenv(engines.BINARY_ENV, str(new_binary))
    requests: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(200, content=archives[str(request.url)])

    original_client = httpx.Client

    def client(*, timeout: httpx.Timeout) -> httpx.Client:
        return original_client(timeout=timeout, transport=httpx.MockTransport(respond))

    monkeypatch.setattr(engines.httpx, "Client", client)
    binary = engines.install_native_image(_BACKEND, old.build_identity, _HARDWARE,
        offline=False, stop=threading.Event())
    assert requests == [old.url]
    assert engines.probe_native_image(binary).build_identity == old.build_identity
    assert _OLD_SOURCE in binary.parts
    assert engines.os.environ[engines.BINARY_ENV] == str(new_binary)
    assert engines.prepared_native_image(_BACKEND, new.build_identity, _HARDWARE) == new_binary
    # Explicit operator overrides retain their exact-build constraint.
    override = new_binary.parent / "override"
    override.write_bytes(new_binary.read_bytes())
    with pytest.raises(engines.ImageEngineInstallError, match="override build"):
        engines.install_native_image(_BACKEND, old.build_identity, _HARDWARE,
            offline=True, stop=threading.Event(), environ={engines.BINARY_ENV: str(override)})


def test_retained_build_family_limit_does_not_disqualify_untested_hardware() -> None:
    for wheel in engines.RETAINED_WHEELS:
        builds = {f"stable_diffusion_cpp-{wheel.compute}": wheel.build_identity}
        backends = frozenset(builds)
        # Genuine implementation limits apply to both new model variants.
        for family in ("flux_3_klein", "flux_3_klein_base"):
            permitted = runtime_backends_for_builds(backends, builds, frozenset(), image_family=family)
            assert bool(permitted) is (wheel.source_revision != _OLD_SOURCE)
        assert not runtime_backends_for_builds(backends, builds, frozenset(), image_family="moss_image")
        assert runtime_backends_for_builds(backends, builds, frozenset(), image_family="flux_2_klein_4b") == backends
    for build in (engines.PUBLISHED_WHEELS[0].build_identity, "operator-compatible-build"):
        assert runtime_backends_for_builds(frozenset({_BACKEND}), {_BACKEND: build}, _HARDWARE,
            image_family="flux_3_klein") == frozenset({_BACKEND})
