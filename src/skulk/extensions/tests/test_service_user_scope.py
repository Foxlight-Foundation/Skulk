"""Per-user plugin manager services the node sets up without elevation."""

import asyncio
import os
import plistlib
import subprocess
from pathlib import Path
from typing import Literal, cast

import httpx
import pytest
from fastapi import FastAPI

from skulk.api.plugins import create_plugins_router
from skulk.extensions import (
    LoadedExtensions,
    service_autosetup,
    service_registration,
    service_setup,
)
from skulk.extensions.runtime_files import write_private
from skulk.extensions.service_autosetup import ServiceSetupRunner
from skulk.extensions.service_registration import ServiceLayout, register_user
from skulk.extensions.service_setup import ServiceConnection, SetupOperation


def _user_layout(tmp_path: Path, platform: str) -> ServiceLayout:
    return ServiceLayout(
        platform,
        os.getuid(),
        os.getgid(),
        "owner",
        "owners",
        "user",
        tmp_path / "state",
        tmp_path / "units",
    )


@pytest.mark.parametrize("platform", ["macos-arm64", "linux-glibc-x86_64"])
def test_user_scope_lives_in_the_owners_directories(
    tmp_path: Path, platform: str
) -> None:
    layout = _user_layout(tmp_path, platform)

    assert layout.root == tmp_path / "state" / "plugin-service"
    assert layout.label == "foundation.foxlight.skulk.plugins"
    suffix = ".plist" if platform.startswith("macos") else ".service"
    assert layout.unit == tmp_path / "units" / ("foundation.foxlight.skulk.plugins" + suffix)


def test_user_scope_requires_absolute_owner_directories() -> None:
    with pytest.raises(ValueError, match="absolute owner directories"):
        ServiceLayout("macos-arm64", 501, 20, "owner", "owners", "user")
    with pytest.raises(ValueError, match="absolute owner directories"):
        ServiceLayout(
            "macos-arm64", 501, 20, "owner", "owners", "user", Path("state"), Path("/u")
        )


def test_user_agent_runs_as_its_owner_without_naming_one(tmp_path: Path) -> None:
    layout = _user_layout(tmp_path, "macos-arm64")
    python = Path("/opt/python/bin/python3.13")

    agent = cast(dict[str, object], plistlib.loads(layout.definition(python)))

    assert "UserName" not in agent and "GroupName" not in agent
    assert cast(list[str], agent["ProgramArguments"])[0] == str(python)
    assert agent["WorkingDirectory"] == str(layout.root)
    assert agent["KeepAlive"] is True and agent["RunAtLoad"] is True
    layout.verify_existing(layout.definition(python))


def test_user_unit_starts_with_the_user_manager(tmp_path: Path) -> None:
    layout = _user_layout(tmp_path, "linux-glibc-x86_64")
    unit = layout.definition(Path("/opt/python/bin/python3.13")).decode()

    assert "User=" not in unit and "Group=" not in unit
    assert "WantedBy=default.target" in unit
    assert "multi-user.target" not in unit
    assert unit.startswith('# skulk-plugin-service-v1 "/opt/python/bin/python3.13"\n')
    layout.verify_existing(unit.encode())


def test_local_user_layout_honors_xdg_on_linux(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        service_registration, "service_platform", lambda: "linux-glibc-x86_64"
    )
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg-state"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg-config"))

    layout = service_registration.local_layout(os.getuid(), "user")

    assert layout.root == tmp_path / "xdg-state" / "skulk" / "plugin-service"
    assert layout.unit.parent == tmp_path / "xdg-config" / "systemd" / "user"

    # A relative XDG value is not a base directory; the default applies.
    monkeypatch.setenv("XDG_STATE_HOME", "relative")
    fallback = service_registration.local_layout(os.getuid(), "user")
    assert fallback.root.parts[-4:] == (".local", "state", "skulk", "plugin-service")


def test_prepare_creates_a_private_root_and_refuses_a_shared_one(
    tmp_path: Path,
) -> None:
    layout = _user_layout(tmp_path, "macos-arm64")

    register_user(layout, "prepare")

    info = layout.root.stat()
    assert info.st_mode & 0o777 == 0o700 and info.st_uid == os.getuid()
    layout.root.chmod(0o750)
    with pytest.raises(ValueError, match="ownership differs"):
        register_user(layout, "prepare")


def test_user_registration_refuses_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(service_registration.os, "geteuid", lambda: 0)
    with pytest.raises(ValueError, match="as its owner"):
        register_user(_user_layout(tmp_path, "macos-arm64"), "prepare")


def _record_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, ...]]:
    commands: list[tuple[str, ...]] = []

    def run(
        arguments: tuple[str, ...], **_: object
    ) -> subprocess.CompletedProcess[bytes]:
        commands.append(tuple(arguments))
        output = b"ActiveState=inactive\nMainPID=0\n" if "show" in arguments else b""
        return subprocess.CompletedProcess(arguments, 0, output, b"")

    monkeypatch.setattr(service_registration.subprocess, "run", run)
    return commands


@pytest.mark.parametrize("platform", ["macos-arm64", "linux-glibc-x86_64"])
def test_install_and_stop_use_the_owners_service_manager(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, platform: str
) -> None:
    layout = _user_layout(tmp_path, platform)
    commands = _record_commands(monkeypatch)

    register_user(layout, "install")

    content = layout.unit.read_bytes()
    layout.verify_existing(content)
    assert layout.unit.stat().st_mode & 0o777 == 0o644
    uid = os.getuid()
    if platform.startswith("macos"):
        assert commands == [
            ("/bin/launchctl", "enable", f"gui/{uid}/{layout.label}"),
            ("/bin/launchctl", "bootstrap", f"gui/{uid}", str(layout.unit)),
        ]
    else:
        assert commands == [
            ("/usr/bin/systemctl", "--user", "daemon-reload"),
            ("/usr/bin/systemctl", "--user", "enable", "--now", layout.unit.name),
        ]

    commands.clear()
    register_user(layout, "stop")
    if platform.startswith("macos"):
        assert commands == [
            ("/bin/launchctl", "bootout", f"gui/{uid}/{layout.label}")
        ]
    else:
        assert commands[0] == ("/usr/bin/systemctl", "--user", "stop", layout.unit.name)
        assert commands[1][:3] == ("/usr/bin/systemctl", "--user", "show")


def test_stop_without_a_definition_does_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    commands = _record_commands(monkeypatch)
    register_user(_user_layout(tmp_path, "macos-arm64"), "stop")
    assert commands == []


def test_a_foreign_definition_under_the_reserved_name_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    layout = _user_layout(tmp_path, "linux-glibc-x86_64")
    layout.unit.parent.mkdir(parents=True)
    layout.unit.write_text("[Service]\nExecStart=/bin/true\n")
    commands = _record_commands(monkeypatch)

    with pytest.raises(ValueError):
        register_user(layout, "install")
    assert commands == []


def _connect(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, root: Path | None
) -> None:
    configuration = tmp_path / "config"
    monkeypatch.setattr(service_setup, "SKULK_CONFIG_HOME", configuration)
    monkeypatch.setattr(service_autosetup, "SKULK_CONFIG_HOME", configuration)
    if root is not None:
        write_private(
            configuration / "managed-service" / "connection.json",
            ServiceConnection(manager_root=str(root), profile_id="a" * 32)
            .model_dump_json()
            .encode(),
        )


def _layouts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> dict[str, ServiceLayout]:
    layouts = {
        "user": _user_layout(tmp_path, "macos-arm64"),
        "system": ServiceLayout("macos-arm64", os.getuid(), os.getgid(), "o", "g"),
    }

    def local_layout(
        _user_id: int, scope: Literal["system", "user"] = "system"
    ) -> ServiceLayout:
        return layouts[scope]

    monkeypatch.setattr(service_setup, "local_layout", local_layout)
    return layouts


@pytest.mark.parametrize("connected", [None, "user", "system"])
def test_connected_scope_names_the_service_this_configuration_uses(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    connected: Literal["user", "system"] | None,
) -> None:
    layouts = _layouts(monkeypatch, tmp_path)
    _connect(
        monkeypatch, tmp_path, None if connected is None else layouts[connected].root
    )
    assert service_setup.connected_scope() == connected


async def test_user_registration_never_elevates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []

    async def elevate(_layout: ServiceLayout, action: str) -> None:
        calls.append("sudo " + action)

    def owner(_layout: ServiceLayout, action: str) -> None:
        calls.append("owner " + action)

    monkeypatch.setattr(service_setup, "_elevate", elevate)
    monkeypatch.setattr(service_registration, "register_user", owner)
    layouts = _layouts(monkeypatch, tmp_path)

    await service_setup._register(layouts["user"], "install")  # pyright: ignore[reportPrivateUsage]
    await service_setup._register(layouts["system"], "install")  # pyright: ignore[reportPrivateUsage]

    assert calls == ["owner install", "sudo install"]


def _operation() -> SetupOperation:
    return SetupOperation(
        operation_id="b" * 32,
        profile_id="a" * 32,
        skulk_build_sha256="c" * 64,
        source_sha256="d" * 64,
        configuration_directory="/tmp/config",
        phase="preparing",
    )


async def _settle(runner: ServiceSetupRunner) -> None:
    for _ in range(100):
        if (await runner.status()).state != "setting_up":
            return
        await asyncio.sleep(0.01)


async def test_runner_sets_up_a_user_service_and_reports_ready(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    layouts = _layouts(monkeypatch, tmp_path)
    _connect(monkeypatch, tmp_path, None)
    scopes: list[str | None] = []
    release = asyncio.Event()

    async def setup(scope: str | None, report: object) -> SetupOperation:
        scopes.append(scope)
        assert callable(report)
        report("Preparing verified independent manager runtime...")
        await release.wait()
        _connect(monkeypatch, tmp_path, layouts["user"].root)
        return _operation()

    async def answers(_root: Path) -> bool:
        return True

    monkeypatch.setattr(service_autosetup, "setup_service", setup)
    monkeypatch.setattr(service_autosetup, "_manager_answers", answers)
    runner = ServiceSetupRunner()
    assert (await runner.status()).state == "absent"

    runner.start()
    runner.start()  # a running setup is not started twice
    await asyncio.sleep(0)
    running = await runner.status()
    assert running.state == "setting_up"
    assert running.progress == "Preparing verified independent manager runtime..."
    release.set()
    await _settle(runner)

    done = await runner.status()
    assert (done.state, done.scope, done.error) == ("ready", "user", None)
    assert scopes == ["user"]


async def test_runner_reports_why_setup_failed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _layouts(monkeypatch, tmp_path)
    _connect(monkeypatch, tmp_path, None)

    async def setup(_scope: str | None, _report: object) -> SetupOperation:
        raise ValueError("service interpreter and Skulk configuration must be outside Git checkouts")

    monkeypatch.setattr(service_autosetup, "setup_service", setup)
    runner = ServiceSetupRunner()
    runner.start()
    await _settle(runner)

    failed = await runner.status()
    assert failed.state == "failed"
    assert failed.error is not None and "outside Git checkouts" in failed.error


async def test_runner_leaves_a_system_service_to_the_terminal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    layouts = _layouts(monkeypatch, tmp_path)
    _connect(monkeypatch, tmp_path, layouts["system"].root)

    async def setup(_scope: str | None, _report: object) -> SetupOperation:
        raise AssertionError("a system service needs elevation the node never asks for")

    async def silent(_root: Path) -> bool:
        return False

    monkeypatch.setattr(service_autosetup, "setup_service", setup)
    monkeypatch.setattr(service_autosetup, "_manager_answers", silent)
    runner = ServiceSetupRunner()
    runner.start()

    status = await runner.status()
    assert (status.state, status.scope) == ("failed", "system")
    assert status.error is not None and "setup --system" in status.error


async def test_setup_routes_answer_before_setup_and_start_it_for_the_owner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _layouts(monkeypatch, tmp_path)
    _connect(monkeypatch, tmp_path, None)
    started: list[str | None] = []

    async def setup(scope: str | None, _report: object) -> SetupOperation:
        started.append(scope)
        raise ValueError("stopped by the test")

    monkeypatch.setattr(service_autosetup, "setup_service", setup)
    app = FastAPI()
    app.include_router(create_plugins_router(LoadedExtensions([]), None))
    transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 52000))
    owner = {"X-Skulk-Dashboard": "pairing-v1", "Origin": "https://localhost"}
    prefix = "/v1/plugins/managed/service"
    async with httpx.AsyncClient(
        transport=transport, base_url="https://localhost"
    ) as client:
        status = await client.get(prefix, headers=owner)
        assert status.status_code == 200, status.text
        assert status.json()["state"] == "absent"
        assert status.headers["Cache-Control"] == "no-store"

        foreign = await client.post(
            prefix + "/setup", headers={**owner, "Origin": "https://foreign.example"}
        )
        assert foreign.status_code == 403
        assert started == []

        accepted = await client.post(prefix + "/setup", headers=owner)
        assert accepted.status_code == 200, accepted.text
        for _ in range(100):
            if started:
                break
            await asyncio.sleep(0.01)
        assert started == ["user"]
