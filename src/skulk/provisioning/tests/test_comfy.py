# pyright: reportPrivateUsage=false, reportAny=false
"""ComfyUI provisioning: variant selection, staged install, gates, wiring."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

import skulk.provisioning.comfy as comfy
from skulk.facts.testing import AMD_STRIX, NVIDIA_A40, make_facts, ok_bin
from skulk.provisioning.comfy import (
    RECORD_FILENAME,
    dormant_comfy,
    ensure_comfy,
    managed_comfy_install,
    provision_comfy,
    select_comfy_variant_chain,
)
from skulk.provisioning.manifest import COMFY_PIN, COMFY_TORCH_WHEELS, PinnedWheel
from skulk.shared.backends import COMFY_BIN_ENV, COMFY_ROOT_ENV


@pytest.fixture(autouse=True)
def isolated_comfy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(comfy, "SKULK_ENGINES_DIR", tmp_path / "engines")
    monkeypatch.setattr(comfy.platform_module, "machine", lambda: "aarch64")
    monkeypatch.setattr(comfy, "_video_models_enabled", lambda: True)
    monkeypatch.delenv(COMFY_BIN_ENV, raising=False)
    monkeypatch.delenv(COMFY_ROOT_ENV, raising=False)
    monkeypatch.delenv("SKULK_NO_ENGINE_AUTOPROVISION", raising=False)
    monkeypatch.setattr(comfy, "_require_tool", _fake_tool)
    monkeypatch.setattr(comfy, "provision_unasked_allowed", lambda: True)


def _fake_tool(name: str) -> str:
    return f"/fake/bin/{name}"


def test_manifest_records_a_hashed_cu130_wheel_set() -> None:
    assert set(COMFY_TORCH_WHEELS) == {("aarch64", "cuda"), ("x86_64", "cuda")}
    assert len(COMFY_PIN) == 40
    for wheels in COMFY_TORCH_WHEELS.values():
        assert [wheel.name for wheel in wheels] == ["torch", "torchvision", "torchaudio"]
        for wheel in wheels:
            assert len(wheel.sha256) == 64
            assert wheel.version.endswith("+cu130")
            assert wheel.filename.startswith(f"{wheel.name}-") and "cp313" in wheel.filename
            assert wheel.url().startswith("https://download.pytorch.org/whl/cu130/")
            assert wheel.requirement().endswith(f"--hash=sha256:{wheel.sha256}")
            assert wheel.constraint() == f"{wheel.name}=={wheel.version}"


def test_variant_chain_needs_linux_a_gpu_and_a_recorded_wheel_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert select_comfy_variant_chain(make_facts(gpus=(NVIDIA_A40,))) == ("cuda",)
    # AMD has no recorded wheel set yet, so nothing is offered.
    assert select_comfy_variant_chain(make_facts(gpus=(AMD_STRIX,))) == ()
    assert select_comfy_variant_chain(make_facts()) == ()
    assert select_comfy_variant_chain(make_facts(platform="darwin", gpus=(NVIDIA_A40,))) == ()
    monkeypatch.setattr(comfy.platform_module, "machine", lambda: "riscv64")
    assert select_comfy_variant_chain(make_facts(gpus=(NVIDIA_A40,))) == ()


class _FakeRun:
    """Stands in for subprocess.run: records commands and fakes their effects."""

    def __init__(self, *, head: str = COMFY_PIN, fail_on: str | None = None) -> None:
        self.commands: list[list[str]] = []
        self.head = head
        self.fail_on = fail_on

    def __call__(self, args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        self.commands.append(args)
        joined = " ".join(args)
        if self.fail_on is not None and self.fail_on in joined:
            return subprocess.CompletedProcess(args, 1, "", "boom")
        if args[1] == "clone":
            checkout = Path(args[-1])
            checkout.mkdir(parents=True)
            (checkout / "requirements.txt").write_text("torch\naiohttp\n")
        elif args[1] == "-C" and args[3] == "checkout":
            (Path(args[2]) / "main.py").write_text("# comfy\n")
        elif args[1] == "-C" and args[3] == "rev-parse":
            return subprocess.CompletedProcess(args, 0, self.head + "\n", "")
        elif args[1] == "venv":
            binary = Path(args[-1]) / "bin" / "python"
            binary.parent.mkdir(parents=True)
            binary.write_text("#!/bin/sh\n")
            binary.chmod(0o755)
        elif args[1:3] == ["pip", "freeze"]:
            return subprocess.CompletedProcess(args, 0, "torch==2.14.0+cu130\n", "")
        return subprocess.CompletedProcess(args, 0, "", "")


def test_provision_builds_the_install_in_order_and_records_it(tmp_path: Path) -> None:
    run = _FakeRun()
    root = provision_comfy("cuda", run=run)
    assert root == tmp_path / "engines" / "comfy" / COMFY_PIN / "cuda"
    assert (root / "ComfyUI" / "main.py").is_file()
    assert os.access(root / "venv" / "bin" / "python", os.X_OK)
    record = json.loads((root / RECORD_FILENAME).read_text())
    assert record["pin"] == COMFY_PIN and record["variant"] == "cuda"
    assert [entry["filename"] for entry in record["wheels"]] == [
        wheel.filename for wheel in COMFY_TORCH_WHEELS[("aarch64", "cuda")]
    ]
    assert (root / "freeze.txt").read_text().startswith("torch==")
    verbs = [tuple(command[1:4]) for command in run.commands]
    assert verbs[0][0] == "clone" and "--filter=blob:none" in run.commands[0]
    assert verbs[1][1:] == ("checkout",) or verbs[1][2] == "checkout"
    assert run.commands[2][3] == "rev-parse"
    assert run.commands[3][1] == "venv" and "3.13" in run.commands[3]
    torch_install = run.commands[4]
    assert "--require-hashes" in torch_install and "--no-deps" in torch_install
    # The staging tree was renamed into place, so the files the commands named
    # now live under the final root.
    requirements = (root / "torch-requirements.txt").read_text()
    assert all(f"--hash=sha256:{wheel.sha256}" in requirements for wheel in COMFY_TORCH_WHEELS[("aarch64", "cuda")])
    comfy_install = run.commands[5]
    assert "--extra-index-url" in comfy_install and "-c" in comfy_install
    assert "torch==2.14.0+cu130" in (root / "torch-constraints.txt").read_text()
    assert run.commands[6][1:3] == ["pip", "freeze"]
    # Idempotent: a complete install runs nothing.
    again = _FakeRun()
    assert provision_comfy("cuda", run=again) == root and again.commands == []


def test_provision_refuses_a_checkout_off_the_pin(tmp_path: Path) -> None:
    run = _FakeRun(head="0" * 40)
    with pytest.raises(RuntimeError, match="not the pinned"):
        provision_comfy("cuda", run=run)
    assert not (tmp_path / "engines" / "comfy" / COMFY_PIN / "cuda").exists()
    assert list((tmp_path / "engines" / "comfy" / COMFY_PIN).iterdir()) == []


def test_provision_leaves_nothing_behind_when_an_install_step_fails(tmp_path: Path) -> None:
    run = _FakeRun(fail_on="--require-hashes")
    with pytest.raises(RuntimeError, match="pinned torch wheel install failed"):
        provision_comfy("cuda", run=run)
    assert list((tmp_path / "engines" / "comfy" / COMFY_PIN).iterdir()) == []


def test_provision_needs_a_recorded_wheel_set() -> None:
    with pytest.raises(RuntimeError, match="no ComfyUI torch wheel set"):
        provision_comfy("rocm", run=_FakeRun())


def test_ensure_provisions_and_exports_both_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_provision(variant: str, *, run: object = None) -> Path:
        return provision_comfy(variant, run=_FakeRun())  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(comfy, "provision_comfy", fake_provision)
    root = ensure_comfy(make_facts(gpus=(NVIDIA_A40,)))
    assert root is not None
    assert os.environ[COMFY_BIN_ENV] == str(root / "venv" / "bin" / "python")
    assert os.environ[COMFY_ROOT_ENV] == str(root / "ComfyUI")


def test_ensure_honors_override_opt_out_and_the_video_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    facts = make_facts(gpus=(NVIDIA_A40,))
    assert ensure_comfy(facts.model_copy(update={"comfy_binary": ok_bin(COMFY_BIN_ENV)})) is None
    # A half-set override is an operator decision too; never overwrite it.
    assert ensure_comfy(facts.model_copy(update={"comfy_root": "/opt/ComfyUI", "comfy_root_state": "ok"})) is None
    monkeypatch.setenv("SKULK_NO_ENGINE_AUTOPROVISION", "1")
    assert ensure_comfy(facts) is None
    monkeypatch.delenv("SKULK_NO_ENGINE_AUTOPROVISION")
    monkeypatch.setattr(comfy, "_video_models_enabled", lambda: False)
    assert ensure_comfy(facts) is None
    assert COMFY_BIN_ENV not in os.environ


def test_ensure_offline_wires_only_an_existing_install(monkeypatch: pytest.MonkeyPatch) -> None:
    facts = make_facts(gpus=(NVIDIA_A40,))
    assert ensure_comfy(facts, allow_download=False) is None
    root = provision_comfy("cuda", run=_FakeRun())
    assert dormant_comfy(facts) == root and managed_comfy_install(facts) == root
    assert COMFY_BIN_ENV not in os.environ
    assert ensure_comfy(facts, allow_download=False) == root
    assert os.environ[COMFY_ROOT_ENV] == str(root / "ComfyUI")


def test_ensure_degrades_when_provisioning_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(variant: str, *, run: object = None) -> Path:
        raise RuntimeError("no network")

    monkeypatch.setattr(comfy, "provision_comfy", broken)
    assert ensure_comfy(make_facts(gpus=(NVIDIA_A40,))) is None
    assert COMFY_BIN_ENV not in os.environ


def test_sanitized_environment_drops_index_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UV_INDEX_URL", "https://mirror.example")
    monkeypatch.setenv("PIP_EXTRA_INDEX_URL", "https://mirror.example")
    monkeypatch.setenv("UV_NO_INDEX", "1")
    monkeypatch.setenv("PIP_NO_INDEX", "1")
    monkeypatch.setenv("KEEP_ME", "1")
    env = comfy.sanitized_index_environment()
    assert "UV_INDEX_URL" not in env and "PIP_EXTRA_INDEX_URL" not in env
    assert "UV_NO_INDEX" not in env and "PIP_NO_INDEX" not in env
    assert env["KEEP_ME"] == "1"


def test_pinned_wheel_requirement_shape() -> None:
    wheel = PinnedWheel(name="torch", version="1.0+cu130", filename="torch-1.0.whl", sha256="a" * 64, index="https://x/")
    assert wheel.url() == "https://x/torch-1.0.whl"
    assert wheel.requirement() == "torch @ https://x/torch-1.0.whl --hash=sha256:" + "a" * 64


def test_startup_never_provisions_before_the_runner_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(comfy, "provision_unasked_allowed", lambda: False)
    calls: list[str] = []

    def fake_provision(variant: str, *, run: object = None) -> Path:
        calls.append(variant)
        return provision_comfy(variant, run=_FakeRun())  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(comfy, "provision_comfy", fake_provision)
    facts = make_facts(gpus=(NVIDIA_A40,))
    assert ensure_comfy(facts) is None and calls == []
    # An operator's doctor --fix is explicit and may provision.
    assert ensure_comfy(facts, explicit=True) is not None and calls == ["cuda"]
