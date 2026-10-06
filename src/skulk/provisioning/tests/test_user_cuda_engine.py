# pyright: reportPrivateUsage=false
"""The CUDA engine installed for the running user (operating-system packages).

A packaged Skulk runtime is read-only and ships pip but no uv, so the CUDA
engine wheel cannot join its environment. It is installed into the user's
engines directory instead, behind launchers that put NVIDIA's runtime
libraries on the loader path. These tests drive that path with a fake index,
a fake wheel download, and a fake pip that lays out what a real
``pip install --target`` produces.
"""

import hashlib
import os
import subprocess
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest
from packaging.tags import Tag

import skulk.provisioning.llama_server as provisioning
from skulk.facts.testing import AMD_STRIX, NVIDIA_A40, make_facts
from skulk.provisioning.manifest import (
    LLAMA_SERVER_CUDA_MIN_REVISION,
    LLAMA_SERVER_PIN,
)
from skulk.shared.backends import LLAMA_SERVER_BIN_ENV, RPC_SERVER_BIN_ENV
from skulk.shared.types.node_facts import NodeFacts

_BUILD = int(LLAMA_SERVER_PIN.removeprefix("b"))
_CURRENT = f"0.{_BUILD}.{LLAMA_SERVER_CUDA_MIN_REVISION}"
_INDEX_ROOT = "https://wheels.example/wheels/"
_WHEEL_BYTES = b"pretend CUDA engine wheel"
_WHEEL_DIGEST = hashlib.sha256(_WHEEL_BYTES).hexdigest()
_X86_64_TAGS = (
    Tag("py3", "none", "manylinux_2_35_x86_64"),
    Tag("py3", "none", "linux_x86_64"),
)
# Captured before any test patches the module, to execute real launchers.
_REAL_RUN = subprocess.run

PipRun = Callable[..., subprocess.CompletedProcess[str]]


@pytest.fixture(autouse=True)
def _isolated_host(  # pyright: ignore[reportUnusedFunction] - autouse
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An x86_64 Linux host with a private engines directory and clean env.

    The engine variables are set and then deleted so that monkeypatch
    restores their absence after a test that wires them.
    """
    monkeypatch.setattr(provisioning.platform_module, "machine", lambda: "x86_64")
    monkeypatch.setattr(provisioning, "SKULK_ENGINES_DIR", tmp_path / "engines")
    monkeypatch.setattr(provisioning, "sys_tags", lambda: iter(_X86_64_TAGS))
    monkeypatch.delenv(provisioning.AUTOPROVISION_OPT_OUT_ENV, raising=False)
    for variable in (LLAMA_SERVER_BIN_ENV, RPC_SERVER_BIN_ENV):
        monkeypatch.setenv(variable, "unset by the fixture")
        monkeypatch.delenv(variable)


def _wheel_name(version: str, platform_tag: str = "manylinux_2_35_x86_64") -> str:
    return f"skulk_llama_server_cuda-{version}-py3-none-{platform_tag}.whl"


def _serve_index(
    monkeypatch: pytest.MonkeyPatch, links: list[tuple[str, str | None]]
) -> None:
    """Serve a PEP 503 page listing ``(filename, sha256_or_None)`` links."""
    anchors = "\n".join(
        f'<a href="{_INDEX_ROOT}{filename}'
        f'{f"#sha256={digest}" if digest else ""}">{filename}</a><br/>'
        for filename, digest in links
    )
    page = f"<!DOCTYPE html><html><body>{anchors}</body></html>"

    def _get(url: str, **kwargs: object) -> httpx.Response:
        assert url == provisioning.FOXLIGHT_WHEEL_INDEX + "skulk-llama-server-cuda/"
        return httpx.Response(200, text=page, request=httpx.Request("GET", url))

    monkeypatch.setattr(provisioning.httpx, "get", _get)


def _serve_wheel(monkeypatch: pytest.MonkeyPatch, downloads: list[str]) -> None:
    """Serve ``_WHEEL_BYTES`` for any wheel URL, recording what was fetched."""

    @contextmanager
    def _stream(method: str, url: str, **kwargs: object) -> Iterator[httpx.Response]:
        downloads.append(url)
        yield httpx.Response(
            200, content=_WHEEL_BYTES, request=httpx.Request(method, url)
        )

    monkeypatch.setattr(provisioning.httpx, "stream", _stream)


def _lay_out_pip_target(tree: Path, version: str = _CURRENT) -> None:
    """What ``pip install --target`` leaves for the CUDA engine wheel."""
    binaries = tree / "skulk_llama_server_cuda" / "bin"
    binaries.mkdir(parents=True)
    for name in ("llama-server", "ggml-rpc-server"):
        binary = binaries / name
        binary.write_text('#!/bin/sh\necho "loader=$LD_LIBRARY_PATH"\necho "args=$*"\n')
        binary.chmod(0o755)
    for package in ("cuda_runtime", "cublas", "nccl"):
        (tree / "nvidia" / package / "lib").mkdir(parents=True)
    info = tree / f"skulk_llama_server_cuda-{version}.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: skulk-llama-server-cuda\nVersion: {version}\n"
    )


def _fake_pip(calls: list[list[str]], environments: list[dict[str, str]]) -> PipRun:
    def _run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(list(argv))
        environment = kwargs["env"]
        assert isinstance(environment, dict)
        environments.append(dict(environment))  # pyright: ignore[reportUnknownArgumentType]
        _lay_out_pip_target(Path(argv[argv.index("--target") + 1]))
        return subprocess.CompletedProcess(argv, 0, "", "")

    return _run


def _install_user_engine(version: str = _CURRENT) -> Path:
    """Put a user engine on disk the way a completed install leaves it."""
    directory = provisioning._user_cuda_engine_dir()
    _lay_out_pip_target(directory, version)
    provisioning._write_cuda_launchers(directory)
    return directory


def _nvidia() -> NodeFacts:
    return make_facts(gpus=(NVIDIA_A40,))


def test_index_lookup_picks_the_newest_pinned_wheel_for_this_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only a hash-carrying, pin-matched, platform-compatible wheel qualifies."""
    floor = LLAMA_SERVER_CUDA_MIN_REVISION
    _serve_index(
        monkeypatch,
        [
            (_wheel_name("0.10068.0"), "1" * 64),
            (_wheel_name(_CURRENT), _WHEEL_DIGEST),
            (
                _wheel_name(f"0.{_BUILD}.{floor + 1}", "manylinux_2_35_aarch64"),
                "2" * 64,
            ),
            (_wheel_name(f"0.{_BUILD}.{floor + 2}"), None),
            (_wheel_name(f"0.{_BUILD}.{floor + 3}rc1"), "3" * 64),
            (
                f"skulk_llama_server_vulkan-0.{_BUILD}.{floor + 4}"
                "-py3-none-manylinux_2_35_x86_64.whl",
                "4" * 64,
            ),
            (_wheel_name(f"0.{_BUILD + 1}.0"), "5" * 64),
            ("not-a-wheel.tar.gz", "6" * 64),
        ],
    )
    assert provisioning._foxlight_cuda_wheel() == (
        _INDEX_ROOT + _wheel_name(_CURRENT),
        _WHEEL_DIGEST,
    )


def test_index_lookup_degrades_when_the_index_is_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _unreachable(url: str, **kwargs: object) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    monkeypatch.setattr(provisioning.httpx, "get", _unreachable)
    assert provisioning._foxlight_cuda_wheel() is None


def test_user_install_lays_out_a_launchable_cuda_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The installed launcher runs the engine with NVIDIA's libraries wired."""
    downloads: list[str] = []
    calls: list[list[str]] = []
    environments: list[dict[str, str]] = []
    _serve_index(monkeypatch, [(_wheel_name(_CURRENT), _WHEEL_DIGEST)])
    _serve_wheel(monkeypatch, downloads)
    monkeypatch.setattr(provisioning.subprocess, "run", _fake_pip(calls, environments))
    monkeypatch.setenv("PIP_INDEX_URL", "https://mirror.corp.example/simple")

    assert provisioning._install_cuda_wheel_for_user(_nvidia())

    assert downloads == [_INDEX_ROOT + _wheel_name(_CURRENT)]
    (argv,) = calls
    assert argv[:4] == [sys.executable, "-m", "pip", "install"]
    assert "--isolated" in argv
    assert "--no-cache-dir" in argv
    # NVIDIA's runtime wheels resolve from PyPI alone; the engine wheel is
    # the verified local file, never re-resolved from an index.
    assert argv[argv.index("--index-url") + 1] == "https://pypi.org/simple/"
    assert "--extra-index-url" not in argv
    assert argv[-1].endswith(_wheel_name(_CURRENT))
    assert not any(key.startswith("PIP_") for key in environments[0])
    # pip stages beside the engine, not in a possibly small system TMPDIR.
    target = Path(argv[argv.index("--target") + 1])
    assert environments[0]["TMPDIR"] == str(target.parent)

    directory = provisioning._user_cuda_engine_dir()
    found = provisioning.user_cuda_llama_server(_nvidia())
    assert found is not None
    assert found == (
        directory / "launchers" / "llama-server-cuda",
        directory / "launchers" / "ggml-rpc-server-cuda",
    )
    # The staging directory and the downloaded wheel are gone.
    assert [entry.name for entry in directory.parent.iterdir()] == [directory.name]

    server, _ = found
    launched = _REAL_RUN(
        [str(server), "--list-devices"],
        env={"PATH": os.environ["PATH"], "LD_LIBRARY_PATH": "/existing/runtime"},
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    loader_line, arguments_line = launched.stdout.splitlines()
    assert arguments_line == "args=--list-devices"
    entries = loader_line.removeprefix("loader=").split(":")
    assert entries[-1] == "/existing/runtime"
    assert [Path(entry).resolve() for entry in entries[:-1]] == [
        (directory / relative).resolve()
        for relative in (
            "nvidia/cublas/lib",
            "nvidia/cuda_runtime/lib",
            "nvidia/nccl/lib",
            "skulk_llama_server_cuda/bin",
        )
    ]


def test_user_install_refuses_a_wheel_that_differs_from_the_index_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    downloads: list[str] = []
    _serve_index(
        monkeypatch,
        [(_wheel_name(_CURRENT), hashlib.sha256(b"the published wheel").hexdigest())],
    )
    _serve_wheel(monkeypatch, downloads)

    def _unexpected_pip(*args: object, **kwargs: object) -> None:
        raise AssertionError("pip must not install an unverified wheel")

    monkeypatch.setattr(provisioning.subprocess, "run", _unexpected_pip)
    assert not provisioning._install_cuda_wheel_for_user(_nvidia())
    assert downloads
    directory = provisioning._user_cuda_engine_dir()
    assert not directory.exists()
    assert list(directory.parent.iterdir()) == []


def test_user_install_degrades_when_pip_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    _serve_index(monkeypatch, [(_wheel_name(_CURRENT), _WHEEL_DIGEST)])
    _serve_wheel(monkeypatch, [])

    def _failing_pip(
        argv: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            argv, 1, "", "ERROR: No matching distribution found for nvidia-nccl-cu12"
        )

    monkeypatch.setattr(provisioning.subprocess, "run", _failing_pip)
    assert not provisioning._install_cuda_wheel_for_user(_nvidia())
    assert not provisioning._user_cuda_engine_dir().exists()


def test_user_install_degrades_when_the_engines_directory_is_unwritable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _serve_index(monkeypatch, [(_wheel_name(_CURRENT), _WHEEL_DIGEST)])

    def _read_only(*args: object, **kwargs: object) -> str:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(provisioning.tempfile, "mkdtemp", _read_only)
    assert not provisioning._install_cuda_wheel_for_user(_nvidia())


def test_user_install_replaces_a_broken_previous_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale = provisioning._user_cuda_engine_dir()
    stale.mkdir(parents=True)
    (stale / "leftover").write_text("from an interrupted engine")
    _serve_index(monkeypatch, [(_wheel_name(_CURRENT), _WHEEL_DIGEST)])
    _serve_wheel(monkeypatch, [])
    monkeypatch.setattr(provisioning.subprocess, "run", _fake_pip([], []))

    assert provisioning._install_cuda_wheel_for_user(_nvidia())
    assert not (stale / "leftover").exists()
    assert provisioning.user_cuda_llama_server(_nvidia()) is not None


@pytest.mark.parametrize(
    ("version", "facts", "found"),
    [
        (_CURRENT, make_facts(gpus=(NVIDIA_A40,)), True),
        (
            f"0.{_BUILD}.{LLAMA_SERVER_CUDA_MIN_REVISION - 1}",
            make_facts(gpus=(NVIDIA_A40,)),
            False,
        ),
        (f"0.{_BUILD + 1}.0", make_facts(gpus=(NVIDIA_A40,)), False),
        (
            _CURRENT,
            make_facts(
                gpus=(NVIDIA_A40.model_copy(update={"compute_capability": "7.5"}),)
            ),
            False,
        ),
        (_CURRENT, make_facts(gpus=(AMD_STRIX,)), False),
    ],
)
def test_user_engine_lookup_requires_the_pin_and_a_capable_gpu(
    version: str, facts: NodeFacts, found: bool
) -> None:
    _install_user_engine(version)
    assert (provisioning.user_cuda_llama_server(facts) is not None) is found


@pytest.mark.parametrize("metadata", [b"", b"\xff\xfe not text"])
def test_a_damaged_user_engine_reads_as_absent(metadata: bytes) -> None:
    """A corrupt install must send startup to reinstall, never crash it."""
    directory = _install_user_engine()
    (
        directory / f"skulk_llama_server_cuda-{_CURRENT}.dist-info" / "METADATA"
    ).write_bytes(metadata)
    assert provisioning.user_cuda_llama_server(_nvidia()) is None


def test_user_engine_lookup_requires_the_launcher() -> None:
    directory = _install_user_engine()
    (directory / "launchers" / "llama-server-cuda").unlink()
    assert provisioning.user_cuda_llama_server(_nvidia()) is None


def test_startup_wires_the_user_engine_ahead_of_the_vulkan_wheel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An installed user CUDA engine outranks the Vulkan wheel without reinstalling."""
    directory = _install_user_engine()
    vulkan = tmp_path / "llama-server-vulkan"
    vulkan.write_text("#!/bin/sh\n")
    vulkan.chmod(0o755)

    def _vulkan_wheel(vendor: str, facts: object) -> tuple[Path, Path | None]:
        return vulkan, None

    def _unexpected_install(facts: object) -> bool:
        raise AssertionError("an installed user engine must not be reinstalled")

    monkeypatch.setattr(provisioning, "wheel_llama_server", _vulkan_wheel)
    monkeypatch.setattr(provisioning, "_cuda_wheel_usable", lambda: False)
    monkeypatch.setattr(provisioning, "try_install_cuda_wheel", _unexpected_install)

    launcher = directory / "launchers" / "llama-server-cuda"
    assert provisioning.dormant_llama_server(_nvidia()) == launcher
    assert LLAMA_SERVER_BIN_ENV not in os.environ
    assert provisioning.ensure_llama_server(_nvidia()) == launcher
    assert os.environ[LLAMA_SERVER_BIN_ENV] == str(launcher)
    assert os.environ[RPC_SERVER_BIN_ENV] == str(
        directory / "launchers" / "ggml-rpc-server-cuda"
    )


def test_startup_installs_the_user_engine_when_only_vulkan_is_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A packaged NVIDIA node completes its CUDA lane at startup."""
    vulkan = tmp_path / "llama-server-vulkan"
    vulkan.write_text("#!/bin/sh\n")
    vulkan.chmod(0o755)

    def _vulkan_wheel(vendor: str, facts: object) -> tuple[Path, Path | None]:
        return vulkan, None

    def _install(facts: object) -> bool:
        _install_user_engine()
        return True

    monkeypatch.setattr(provisioning, "wheel_llama_server", _vulkan_wheel)
    monkeypatch.setattr(provisioning, "_cuda_wheel_usable", lambda: False)
    monkeypatch.setattr(provisioning, "try_install_cuda_wheel", _install)

    wired = provisioning.ensure_llama_server(_nvidia())
    assert (
        wired
        == provisioning._user_cuda_engine_dir() / "launchers" / "llama-server-cuda"
    )
