"""Pip-installable Vulkan llama-server for Skulk's served GGUF engine.

Sibling of ``skulk_llama_server_cuda`` with a simpler runtime story: the
wheel bundles the Khronos Vulkan loader next to the binary, so the shim only
needs the wheel's own ``bin`` directory on the loader path. The system's
remaining prerequisite is the GPU driver's Vulkan ICD (``mesa-vulkan-drivers``
on AMD; the NVIDIA driver's ICD on bare metal), which only the driver vendor
can ship; Skulk's facts probe and doctor surface its absence loudly.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent
_BIN_DIR = _PACKAGE_DIR / "bin"


def binary_path() -> Path:
    """Absolute path of the bundled ``llama-server`` binary.

    Raises:
        FileNotFoundError: When the wheel payload is missing (a source
            checkout rather than a built wheel).
    """
    binary = _BIN_DIR / "llama-server"
    if not binary.is_file():
        raise FileNotFoundError(
            f"no llama-server payload at {binary}; this is a source checkout, "
            "not a built wheel"
        )
    return binary


def rpc_server_path() -> Path | None:
    """Absolute path of the bundled ``ggml-rpc-server``, or ``None``."""
    rpc = _BIN_DIR / "ggml-rpc-server"
    return rpc if rpc.is_file() else None


def launch_environment(base: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for exec'ing the binary: the wheel's lib dir prepended.

    The bundled Vulkan loader and the ggml shared libraries sit next to the
    binary; ``$ORIGIN`` rpath already resolves the ggml libraries, so the
    explicit entry mainly guarantees the bundled ``libvulkan`` wins over a
    missing or stale system loader.

    Args:
        base: Environment to extend (defaults to ``os.environ``).

    Returns:
        A copy with ``LD_LIBRARY_PATH`` prefixed by the wheel's ``bin`` dir.
    """
    env = dict(os.environ if base is None else base)
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(
        [str(_BIN_DIR)] + ([existing] if existing else [])
    )
    return env


def main() -> int:
    """Console entry point: exec llama-server with the bundled loader wired.

    Forwards all arguments verbatim, so ``llama-server-vulkan --list-devices``
    behaves exactly like the underlying binary (which is what lets Skulk's
    facts probe validate the wheel like any other engine binary).
    """
    binary = binary_path()
    os.execve(str(binary), [str(binary), *sys.argv[1:]], launch_environment())
    return 1  # pragma: no cover - execve does not return on success


def rpc_main() -> int:
    """Console entry point for the bundled ``ggml-rpc-server`` (RPC donor)."""
    rpc = rpc_server_path()
    if rpc is None:
        raise FileNotFoundError(
            "no ggml-rpc-server payload in this wheel; this is a source "
            "checkout, not a built wheel"
        )
    os.execve(str(rpc), [str(rpc), *sys.argv[1:]], launch_environment())
    return 1  # pragma: no cover - execve does not return on success
