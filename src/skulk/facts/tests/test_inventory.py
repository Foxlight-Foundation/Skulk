# pyright: reportPrivateUsage=false
"""Tests for exact open engine and hardware inventory."""

import hashlib
import json
from pathlib import Path

import pytest

from skulk.facts.inventory import engine_build_inventory, hardware_class_inventory
from skulk.facts.testing import AMD_STRIX, NVIDIA_A40, make_facts
from skulk.shared.types.node_facts import EngineBinaryFact


def test_engine_build_override_supports_open_backend_tags() -> None:
    """Operators can name exact builds without closing future identifier space."""
    facts = make_facts()
    inventory = engine_build_inventory(
        frozenset({"mlx", "mlx-metal"}),
        facts,
        environ={
            "SKULK_ENGINE_BUILDS": json.dumps(
                {"mlx": "mlx@future.1+sha.abc"}
            )
        },
    )

    assert inventory == {
        "mlx": "mlx@future.1+sha.abc",
        "mlx-metal": "mlx@future.1+sha.abc",
    }


def test_served_binary_inventory_is_content_bound(tmp_path: Path) -> None:
    """A served engine build changes identity when its executable bytes change."""
    binary = tmp_path / "llama-server"
    binary.write_bytes(b"exact engine bytes")
    facts = make_facts(
        llama_server_bin=EngineBinaryFact(
            env_var="SKULK_LLAMA_SERVER_BIN",
            configured_path=str(binary),
            state="ok",
        )
    )

    inventory = engine_build_inventory(
        frozenset({"llama_server", "llama_server-vulkan"}),
        facts,
        environ={},
    )

    expected = "llama.cpp@sha256:" + hashlib.sha256(binary.read_bytes()).hexdigest()
    assert inventory["llama_server"] == expected
    assert inventory["llama_server-vulkan"] == expected


def test_vllm_inventory_reads_configured_environment_version(
    tmp_path: Path,
) -> None:
    """A separate vLLM virtual environment reports its own package version."""
    binary = tmp_path / "vllm"
    binary.write_text("#!/bin/sh\nprintf 'vllm 0.26.1+cu129\\n'\n")
    binary.chmod(0o755)
    facts = make_facts(
        vllm_bin=EngineBinaryFact(
            env_var="SKULK_VLLM_BIN",
            configured_path=str(binary),
            state="ok",
        )
    )
    inventory = engine_build_inventory(
        frozenset({"vllm", "vllm-cuda"}), facts, environ={}
    )

    assert inventory == {
        "vllm": "vllm@0.26.1+cu129",
        "vllm-cuda": "vllm@0.26.1+cu129",
    }


def test_hardware_classes_preserve_vendor_model_and_compute() -> None:
    """Observed hardware becomes open stable constraint identifiers."""
    classes = hardware_class_inventory(
        make_facts(gpus=(NVIDIA_A40, AMD_STRIX))
    )

    assert classes == frozenset(
        {
            "platform:linux",
            "nvidia",
            "nvidia:nvidia-a40",
            "nvidia:sm-8.6",
            "amd",
            "amd:amd-gpu",
            "amd:pci-1002-1586",
        }
    )


def test_invalid_engine_build_override_fails_closed() -> None:
    """Malformed operator inventory cannot silently admit signed support."""
    with pytest.raises(ValueError, match="JSON object of strings"):
        engine_build_inventory(
            frozenset({"mlx"}),
            make_facts(),
            environ={"SKULK_ENGINE_BUILDS": '["mlx@1"]'},
        )


def test_comfy_build_is_the_checkout_head(monkeypatch: pytest.MonkeyPatch) -> None:
    from skulk.facts import inventory as inventory_module
    from skulk.facts.testing import NVIDIA_A40, make_facts, ok_bin

    def fake_head(checkout: str) -> str | None:
        return "a" * 40 if checkout == "/opt/ComfyUI" else None

    monkeypatch.setattr(inventory_module, "_git_head", fake_head)

    def fake_torch(interpreter: str) -> str | None:
        return "2.9.1+cu130" if interpreter == "/opt/SKULK_COMFY_BIN" else None

    monkeypatch.setattr(inventory_module, "_comfy_torch", fake_torch)
    facts = make_facts(gpus=(NVIDIA_A40,)).model_copy(
        update={"comfy_binary": ok_bin("SKULK_COMFY_BIN"), "comfy_root": "/opt/ComfyUI", "comfy_root_state": "ok"}
    )
    builds = inventory_module.engine_build_inventory(frozenset({"comfy", "comfy-cuda"}), facts, environ={})
    # The torch build is part of the identity: the same checkout on another
    # torch is another engine for the support matrix.
    expected = "comfy@" + "a" * 40 + "/torch@2.9.1+cu130"
    assert builds == {"comfy": expected, "comfy-cuda": expected}
    # A checkout whose interpreter cannot answer keeps the commit-only form.
    def no_torch(_interpreter: str) -> str | None:
        return None

    monkeypatch.setattr(inventory_module, "_comfy_torch", no_torch)
    builds = inventory_module.engine_build_inventory(frozenset({"comfy"}), facts, environ={})
    assert builds == {"comfy": "comfy@" + "a" * 40}


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [
        ("skulk-torch=2.9.1+cu130\n", "2.9.1+cu130"),
        ("Welcome banner\nskulk-torch=2.9.1+rocm7.2\nsome warning\n", "2.9.1+rocm7.2"),
        ("2.9.1+cu130\n", None),
        ("skulk-torch=\n", None),
        ("skulk-torch=2.9.1\nskulk-torch=2.9.2\n", None),
        ("skulk-torch=not a version\n", None),
    ],
)
def test_comfy_torch_reads_only_the_sentinel_line(
    monkeypatch: pytest.MonkeyPatch, stdout: str, expected: str | None
) -> None:
    """Only the sentinel line of the interpreter's output names the torch build."""
    import subprocess

    from skulk.facts import inventory as inventory_module

    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(inventory_module.subprocess, "run", fake_run)
    inventory_module._comfy_torch.cache_clear()
    assert inventory_module._comfy_torch(f"/opt/python-{hash(stdout)}") == expected
