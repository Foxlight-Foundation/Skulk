# pyright: reportPrivateUsage=false
"""Pinned audio.cpp preparation, cache reuse, and offline failure behavior."""

import hashlib
import json
import os
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from skulk.provisioning import audio_cpp

_SPECS_SOURCE = (
    Path(__file__).resolve().parents[4]
    / "packaging/skulk-audio-cpp-cpu/src/skulk_audio_cpp_cpu/model_specs"
)


def _wheel(tmp_path: Path) -> tuple[audio_cpp.AudioCppWheel, Path]:
    wheel_path = tmp_path / "skulk_audio_cpp_cpu-0.8.2.post1-py3-none-macosx_15_0_arm64.whl"
    with zipfile.ZipFile(wheel_path, "w") as archive:
        for member in audio_cpp._REQUIRED_MEMBERS:
            if member.endswith("audiocpp_server"):
                payload = b"#!/bin/sh\n"
            elif "/model_specs/" in member:
                payload = (_SPECS_SOURCE / Path(member).name).read_bytes()
            else:
                payload = b"{}"
            archive.writestr(member, payload)
    wheel = audio_cpp.AudioCppWheel(
        filename=wheel_path.name,
        sha256=hashlib.sha256(wheel_path.read_bytes()).hexdigest(),
    )
    return wheel, wheel_path


def _vulkan_wheel(tmp_path: Path) -> tuple[audio_cpp.AudioCppWheel, Path]:
    """Make a complete independent Vulkan payload for cache isolation tests."""
    wheel_path = tmp_path / "skulk_audio_cpp_vulkan-0.8.2.post1-py3-none-manylinux_2_35_x86_64.whl"
    provisional = audio_cpp.AudioCppWheel(filename=wheel_path.name, sha256="0" * 64)
    with zipfile.ZipFile(wheel_path, "w") as archive:
        for member in audio_cpp._required_members(provisional):
            if member.endswith("audiocpp_server"):
                payload = b"#!/bin/sh\n# vulkan\n"
            elif "/model_specs/" in member:
                payload = (_SPECS_SOURCE / Path(member).name).read_bytes()
            else:
                payload = b"license\n"
            archive.writestr(member, payload)
    wheel = audio_cpp.AudioCppWheel(
        filename=wheel_path.name,
        sha256=hashlib.sha256(wheel_path.read_bytes()).hexdigest(),
    )
    return wheel, wheel_path


def _fixture_downloader(source: Path) -> Callable[[audio_cpp.AudioCppWheel, Path], None]:
    """Return a typed local substitute for the package-channel downloader."""

    def download(_wheel: audio_cpp.AudioCppWheel, destination: Path) -> None:
        destination.write_bytes(source.read_bytes())

    return download


def test_prepare_download_then_offline_cache_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One verified download gives an offline-ready executable thereafter."""
    wheel, source = _wheel(tmp_path)
    monkeypatch.setattr(audio_cpp, "SKULK_ENGINES_DIR", tmp_path / "cache")
    monkeypatch.setattr(audio_cpp, "audio_cpp_wheel_for_host", lambda: wheel)
    monkeypatch.setattr(
        audio_cpp,
        "_download_wheel",
        _fixture_downloader(source),
    )
    monkeypatch.delenv("SKULK_AUDIO_CPP_BIN", raising=False)
    installed = audio_cpp.prepare_audio_cpp(allow_download=True, environ={})
    assert installed.is_file()
    monkeypatch.delenv("SKULK_AUDIO_CPP_BIN", raising=False)
    assert audio_cpp.prepare_audio_cpp(allow_download=False, environ={}) == installed
    assert audio_cpp.verified_cached_audio_cpp_binary(installed)
    monkeypatch.delenv("SKULK_AUDIO_CPP_BIN", raising=False)
    assert audio_cpp.rehydrate_cached_audio_cpp(environ={}) == installed
    assert os.environ["SKULK_AUDIO_CPP_BIN"] == str(installed)


def test_vulkan_variant_coexists_with_cpu_and_rehydrates_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preparing the GPU wheel cannot replace a pinned executable in use by CPU."""
    cpu_wheel, cpu_source = _wheel(tmp_path)
    vulkan_wheel, vulkan_source = _vulkan_wheel(tmp_path)
    monkeypatch.setattr(audio_cpp, "SKULK_ENGINES_DIR", tmp_path / "cache")
    monkeypatch.setattr(audio_cpp, "audio_cpp_wheel_for_host", lambda: cpu_wheel)
    monkeypatch.setattr(
        audio_cpp, "audio_cpp_vulkan_wheel_for_host", lambda: vulkan_wheel
    )
    sources = {cpu_wheel.filename: cpu_source, vulkan_wheel.filename: vulkan_source}

    def download(wheel: audio_cpp.AudioCppWheel, destination: Path) -> None:
        destination.write_bytes(sources[wheel.filename].read_bytes())

    monkeypatch.setattr(audio_cpp, "_download_wheel", download)
    monkeypatch.delenv("SKULK_AUDIO_CPP_BIN", raising=False)
    monkeypatch.delenv("SKULK_AUDIO_CPP_VULKAN_BIN", raising=False)
    cpu = audio_cpp.prepare_audio_cpp(allow_download=True, environ={})
    vulkan = audio_cpp.prepare_audio_cpp(
        allow_download=True, variant="vulkan",
        environ={"SKULK_AUDIO_CPP_BIN": str(cpu)},
    )
    assert cpu != vulkan
    assert cpu.is_file() and vulkan.is_file()
    assert audio_cpp.verified_cached_audio_cpp_binary(cpu)
    assert audio_cpp.verified_cached_audio_cpp_binary(vulkan)
    monkeypatch.delenv("SKULK_AUDIO_CPP_BIN", raising=False)
    monkeypatch.delenv("SKULK_AUDIO_CPP_VULKAN_BIN", raising=False)
    assert audio_cpp.rehydrate_cached_audio_cpp(environ={}) == cpu
    assert os.environ["SKULK_AUDIO_CPP_BIN"] == str(cpu)
    assert os.environ["SKULK_AUDIO_CPP_VULKAN_BIN"] == str(vulkan)
    assert audio_cpp.prepare_audio_cpp(
        allow_download=False, variant="vulkan", environ={}
    ) == vulkan


def test_offline_miss_and_tampered_cache_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An offline miss or changed binary cannot become ready by cache presence."""
    wheel, source = _wheel(tmp_path)
    monkeypatch.setattr(audio_cpp, "SKULK_ENGINES_DIR", tmp_path / "cache")
    monkeypatch.setattr(audio_cpp, "audio_cpp_wheel_for_host", lambda: wheel)
    monkeypatch.delenv("SKULK_AUDIO_CPP_BIN", raising=False)
    with pytest.raises(RuntimeError, match="not in the verified local cache"):
        audio_cpp.prepare_audio_cpp(allow_download=False, environ={})
    monkeypatch.setattr(audio_cpp, "_download_wheel", _fixture_downloader(source))
    installed = audio_cpp.prepare_audio_cpp(allow_download=True, environ={})
    installed.write_bytes(b"replaced executable")
    monkeypatch.delenv("SKULK_AUDIO_CPP_BIN", raising=False)
    with pytest.raises(RuntimeError, match="not in the verified local cache"):
        audio_cpp.prepare_audio_cpp(allow_download=False, environ={})


def test_editing_cache_record_cannot_approve_replaced_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rewritten cache record cannot attest to bytes absent from the pinned wheel."""
    wheel, source = _wheel(tmp_path)
    monkeypatch.setattr(audio_cpp, "SKULK_ENGINES_DIR", tmp_path / "cache")
    monkeypatch.setattr(audio_cpp, "audio_cpp_wheel_for_host", lambda: wheel)
    monkeypatch.setattr(audio_cpp, "_download_wheel", _fixture_downloader(source))
    monkeypatch.delenv("SKULK_AUDIO_CPP_BIN", raising=False)
    installed = audio_cpp.prepare_audio_cpp(allow_download=True, environ={})
    installed.write_bytes(b"#!/bin/sh\n# replaced\n")
    record_path = installed.parents[2] / "provisioned.json"
    record = cast("dict[str, object]", json.loads(record_path.read_text()))
    record["binary_sha256"] = hashlib.sha256(installed.read_bytes()).hexdigest()
    member_hashes = cast("dict[str, str]", record["member_sha256"])
    member_hashes[audio_cpp._BINARY_MEMBER] = hashlib.sha256(installed.read_bytes()).hexdigest()
    record_path.write_text(json.dumps(record))
    assert not audio_cpp.verified_cached_audio_cpp_binary(installed)
    monkeypatch.delenv("SKULK_AUDIO_CPP_BIN", raising=False)
    with pytest.raises(RuntimeError, match="not in the verified local cache"):
        audio_cpp.prepare_audio_cpp(allow_download=False, environ={})


def test_explicit_override_cannot_hide_invalid_path(tmp_path: Path) -> None:
    """A bad operator override remains a mount error, without downloading."""
    with pytest.raises(RuntimeError, match="names no executable"):
        audio_cpp.prepare_audio_cpp(
            allow_download=True,
            environ={"SKULK_AUDIO_CPP_BIN": str(tmp_path / "missing")},
        )


def test_standalone_override_requires_pinned_model_specs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An operator binary may keep its specs outside wheel layout, but not drift."""
    binary = tmp_path / "build" / "bin" / "audiocpp_server"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    specs = tmp_path / "source" / "model_specs"
    specs.mkdir(parents=True)
    for source in _SPECS_SOURCE.glob("*.json"):
        (specs / source.name).write_bytes(source.read_bytes())
    monkeypatch.setattr(
        audio_cpp, "audio_cpp_wheel_for_host",
        lambda: audio_cpp.AudioCppWheel(
            filename="skulk_audio_cpp_cpu-0.8.2.post1-py3-none-macosx_15_0_arm64.whl",
            sha256="a" * 64,
        ),
    )
    env = {
        "SKULK_AUDIO_CPP_BIN": str(binary),
        "SKULK_AUDIO_CPP_SPECS_DIR": str(specs),
    }
    assert audio_cpp.prepare_audio_cpp(allow_download=False, environ=env) == binary
    with pytest.raises(RuntimeError, match="paths must be absolute"):
        audio_cpp.prepare_audio_cpp(
            allow_download=False,
            environ={**env, "SKULK_AUDIO_CPP_SPECS_DIR": "relative/specs"},
        )
    (specs / "ace_step.json").write_text("{}")
    with pytest.raises(RuntimeError, match="differs from the pin"):
        audio_cpp.prepare_audio_cpp(allow_download=False, environ=env)


@pytest.mark.parametrize("compute", ["vulkan", "cpu"])
def test_primary_override_preserves_qualified_vulkan_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, compute: str,
) -> None:
    """A standalone primary Vulkan build remains usable after adding the GPU wheel."""
    binary = tmp_path / "operator" / "bin" / "audiocpp_server"
    binary.parent.mkdir(parents=True)
    specs = binary.parent.parent / "model_specs"
    specs.mkdir()
    for source in _SPECS_SOURCE.glob("*.json"):
        (specs / source.name).write_bytes(source.read_bytes())
    device = "VK:0 Radeon" if compute == "vulkan" else "CPU:0 Host"
    binary.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "--version" ]; then\n'
        f"  echo 'git: {audio_cpp.AUDIO_CPP_SOURCE_REVISION}'\n"
        f"  echo 'backends: {compute}'\n"
        'elif [ "$1" = "--list-devices" ]; then\n'
        f"  echo '{device}'\n"
        "fi\n"
    )
    binary.chmod(0o755)

    def managed_fallback(**_kwargs: object) -> Path:
        raise RuntimeError("managed Vulkan fallback invoked")

    monkeypatch.setattr(audio_cpp, "_prepare_pinned_audio_cpp", managed_fallback)
    environment = {"SKULK_AUDIO_CPP_BIN": str(binary)}
    if compute == "vulkan":
        assert audio_cpp.prepare_audio_cpp(
            allow_download=False, variant="vulkan", environ=environment,
        ) == binary
    else:
        with pytest.raises(RuntimeError, match="managed Vulkan fallback invoked"):
            audio_cpp.prepare_audio_cpp(
                allow_download=False, variant="vulkan", environ=environment,
            )


def test_standalone_vulkan_override_outweighs_cached_vulkan_wheel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Startup must not mask an explicit Vulkan build with a cached package."""
    wheel, source = _vulkan_wheel(tmp_path)
    monkeypatch.setattr(audio_cpp, "SKULK_ENGINES_DIR", tmp_path / "cache")
    monkeypatch.setattr(audio_cpp, "audio_cpp_vulkan_wheel_for_host", lambda: wheel)
    monkeypatch.setattr(audio_cpp, "_download_wheel", _fixture_downloader(source))
    monkeypatch.delenv("SKULK_AUDIO_CPP_BIN", raising=False)
    monkeypatch.delenv("SKULK_AUDIO_CPP_VULKAN_BIN", raising=False)
    cached = audio_cpp.prepare_audio_cpp(
        allow_download=True, variant="vulkan", environ={}
    )
    assert cached.is_file()
    monkeypatch.delenv("SKULK_AUDIO_CPP_VULKAN_BIN", raising=False)

    binary = tmp_path / "operator" / "bin" / "audiocpp_server"
    binary.parent.mkdir(parents=True)
    specs = binary.parent.parent / "model_specs"
    specs.mkdir()
    for pinned in _SPECS_SOURCE.glob("*.json"):
        (specs / pinned.name).write_bytes(pinned.read_bytes())
    binary.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "--version" ]; then\n'
        f"  echo 'git: {audio_cpp.AUDIO_CPP_SOURCE_REVISION}'\n"
        "  echo 'backends: vulkan'\n"
        'elif [ "$1" = "--list-devices" ]; then\n'
        "  echo 'VK:0 Radeon'\n"
        "fi\n"
    )
    binary.chmod(0o755)
    environment = {"SKULK_AUDIO_CPP_BIN": str(binary)}

    assert audio_cpp.rehydrate_cached_audio_cpp(environ=environment) is None
    assert "SKULK_AUDIO_CPP_VULKAN_BIN" not in os.environ
    assert audio_cpp.prepare_audio_cpp(
        allow_download=False, variant="vulkan", environ=environment,
    ) == binary


def test_cached_spec_mutation_invalidates_even_selected_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cache record covers required model specs as well as the executable."""
    wheel, source = _wheel(tmp_path)
    monkeypatch.setattr(audio_cpp, "SKULK_ENGINES_DIR", tmp_path / "cache")
    monkeypatch.setattr(audio_cpp, "audio_cpp_wheel_for_host", lambda: wheel)
    monkeypatch.setattr(audio_cpp, "_download_wheel", _fixture_downloader(source))
    monkeypatch.delenv("SKULK_AUDIO_CPP_BIN", raising=False)
    installed = audio_cpp.prepare_audio_cpp(allow_download=True, environ={})
    (installed.parent.parent / "model_specs" / "ace_step.json").write_text("tampered")
    with pytest.raises(RuntimeError, match="integrity verification"):
        audio_cpp.prepare_audio_cpp(
            allow_download=False,
            environ={"SKULK_AUDIO_CPP_BIN": str(installed)},
        )
