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
