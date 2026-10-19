"""Install immutable vLLM wheel recipes in isolated environments on first use.

Startup only adopts completed caches. Explicit operator binaries always win.
Each recipe binds every dependency by version/hash and selects a GPU runtime,
independently of the hardware on which Foxlight has collected qualifications.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import re
import shutil
import subprocess
import tempfile
import threading
import time
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Final, Literal, cast, final

import httpx
from filelock import FileLock
from filelock import Timeout as FileLockTimeout
from packaging.tags import platform_tags
from pydantic import Field

from skulk.provisioning.comfy import (
    ComfyInstallCancelledError,
    Runner,
    run_runtime_install_command,
    runtime_install_tool,
    runtime_install_tool_available,
    sanitized_index_environment,
)
from skulk.provisioning.llama_server import AUTOPROVISION_OPT_OUT_ENV
from skulk.shared.backends import VLLM_BIN_ENV
from skulk.shared.constants import SKULK_ENGINES_DIR
from skulk.shared.types.node_facts import NodeFacts
from skulk.utils.pydantic_ext import FrozenModel

VLLM_PIN: Final = "0.32.0"
VLLM_PYTHON: Final = "3.12.13"
VLLM_INSTALL_FREE_BYTES: Final = 20 * 1024**3
_INSTALL_TIMEOUT_SECONDS: Final = 14400.0
_RECIPE_LIMIT_BYTES: Final = 2 * 1024**2
_INDEX_ROOT: Final = "https://wheels.foxlight.ai/wheels/"


@final
class VllmRecipe(FrozenModel):
    """A complete, immutable Python/GPU runtime recipe, separate from model data."""

    variant: Literal["cuda", "rocm"] = Field(
        description="GPU runtime in the upstream wheels."
    )
    architecture: Literal["aarch64", "x86_64"] = Field(description="Host CPU ABI.")
    platform_tag: str = Field(
        pattern=r"^manylinux_2_(28|39)_(aarch64|x86_64)$",
        description="Minimum host libc and CPU ABI.",
    )
    filename: str = Field(
        pattern=r"^skulk_vllm_runtime_(cuda|rocm)-[a-zA-Z0-9_.-]+\.whl$",
        description="Append-only recipe package filename.",
    )
    sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$", description="Digest of the entire recipe wheel."
    )
    size_bytes: int = Field(
        gt=0, le=_RECIPE_LIMIT_BYTES, description="Exact bounded recipe archive size."
    )
    lock_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$", description="Digest of the complete dependency lock."
    )
    upstream_filename: str = Field(
        description="Unmodified upstream vLLM wheel mirrored on the index."
    )
    upstream_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="Expected complete upstream wheel digest.",
    )
    source_revision: str = Field(
        pattern=r"^[0-9a-f]{40}$",
        description="Immutable upstream vLLM source revision.",
    )
    version: str = Field(
        description="Exact vLLM distribution version, including GPU build suffix."
    )

    required_system_libraries: tuple[str, ...] = Field(
        default=(),
        description="Shared-library SONAMEs the host ROCm/OpenMPI runtime must provide before download eligibility.",
    )

    @property
    def build_identity(self) -> str:
        """Return an identity binding vLLM and its complete dependency recipe."""
        return f"vllm@{self.version}/recipe@sha256:{self.sha256}"


PUBLISHED_RECIPES: Final = tuple(
    VllmRecipe.model_validate_json(json.dumps(entry))
    for entry in cast(
        list[object],
        json.loads(Path(__file__).with_name("vllm_manifest.json").read_text()),
    )
)


@final
class VllmInstallError(RuntimeError):
    """Installation failure with a safe operator step and private diagnostic detail."""

    def __init__(self, step: str, detail: str) -> None:
        """Store the failing step and diagnostic output for worker error handling."""
        super().__init__(f"{step}: {detail}")
        self.step = step
        self.detail = detail


def select_vllm_recipe(facts: NodeFacts) -> VllmRecipe | None:
    """Match real runtime ABI and observed GPU vendor, without a test-card allowlist."""
    if facts.platform != "linux":
        return None
    tags = set(platform_tags())
    variants: set[str] = set()
    if facts.gpus_of("nvidia"):
        variants.add("cuda")
    if facts.gpus_of("amd"):
        variants.add("rocm")
    declared = (
        facts.declared_vllm_backends
        or facts.declared_llama_server_backends
        or facts.declared_llama_cpp_backends
    )
    allowed = (
        None
        if not declared or not declared.strip()
        else {item.strip().lower() for item in declared.split(",")}
    )
    return next(
        (
            recipe
            for recipe in PUBLISHED_RECIPES
            if recipe.architecture == platform.machine()
            and recipe.variant in variants
            and recipe.platform_tag in tags
            and (allowed is None or recipe.variant in allowed)
        ),
        None,
    )


def managed_vllm_root(recipe: VllmRecipe) -> Path:
    """Return the immutable recipe-keyed cache root, preserving earlier runtimes."""
    return (
        SKULK_ENGINES_DIR
        / "vllm"
        / recipe.version
        / f"{recipe.variant}-{recipe.architecture}-{recipe.sha256}"
    )


def _binary(root: Path) -> Path:
    return root / "venv/bin/vllm"


def _complete(root: Path, recipe: VllmRecipe) -> bool:
    try:
        record = cast(object, json.loads((root / "provisioned.json").read_text()))
        lock = (root / "requirements.txt").read_bytes()
        return (
            record
            == {
                "build": recipe.build_identity,
                "python": VLLM_PYTHON,
                "gpu_verified": True,
            }
            and hashlib.sha256(lock).hexdigest() == recipe.lock_sha256
            and _binary(root).is_file()
            and os.access(_binary(root), os.X_OK)
            and (root / "venv/bin/python").is_file()
        )
    except (OSError, ValueError):
        return False


def installed_managed_vllm(
    resolved_backend: str | None = None, expected_build: str | None = None
) -> Path | None:
    """Find a complete runtime from a child runner, which predates worker installation."""
    for recipe in PUBLISHED_RECIPES:
        if recipe.architecture != platform.machine():
            continue
        if resolved_backend not in (None, "vllm", f"vllm-{recipe.variant}"):
            continue
        if expected_build is not None and expected_build != recipe.build_identity:
            continue
        root = managed_vllm_root(recipe)
        if _complete(root, recipe):
            return _binary(root)
    return None


def managed_vllm_build(binary: str) -> str | None:
    """Identify only a verified managed cache; explicit runtimes retain their own identity."""
    for recipe in PUBLISHED_RECIPES:
        root = managed_vllm_root(recipe)
        if str(_binary(root)) == binary and _complete(root, recipe):
            return recipe.build_identity
    return None


def managed_vllm_variant(binary: str) -> Literal["cuda", "rocm"] | None:
    """Read the verified managed binary's compute variant for the facts snapshot.

    A mixed-vendor host can see more GPUs than one installed runtime can drive.
    The observed recipe limits advertised tags without probing inside derivation.
    """
    for recipe in PUBLISHED_RECIPES:
        root = managed_vllm_root(recipe)
        if str(_binary(root)) == binary and _complete(root, recipe):
            return recipe.variant
    return None


def runtime_prerequisites_present(recipe: VllmRecipe) -> bool:
    """Check host-supplied ROCm/OpenMPI libraries before promising a download.

    The upstream ROCm wheel links the system SDK. This passive file check does
    not load those libraries into Skulk; first-use GPU verification checks ABI.
    """
    if not recipe.required_system_libraries:
        return True
    roots = [
        Path(item) for item in os.environ.get("LD_LIBRARY_PATH", "").split(":") if item
    ]
    roots += [
        Path("/lib"),
        Path("/lib64"),
        Path("/usr/lib"),
        Path("/usr/lib64"),
        Path("/usr/lib64/openmpi/lib"),
        Path("/usr/lib/x86_64-linux-gnu"),
        Path("/usr/local/lib"),
        Path("/opt/rocm/lib"),
        Path("/opt/rocm/lib64"),
    ]
    if path := os.environ.get("ROCM_PATH"):
        roots += [Path(path) / "lib", Path(path) / "lib64"]
    return all(
        any((root / library).is_file() for root in roots)
        for library in recipe.required_system_libraries
    )


def _gates_pass(facts: NodeFacts, env: Mapping[str, str]) -> bool:
    return (
        facts.vllm_binary.state == "not_configured"
        and not env.get(VLLM_BIN_ENV, "").strip()
        and env.get(AUTOPROVISION_OPT_OUT_ENV, "").strip() != "1"
        and env.get("SKULK_NODE_PARTICIPATION", "").strip().lower() in ("", "full")
    )


def rehydrate_vllm(
    facts: NodeFacts, *, environ: dict[str, str] | None = None
) -> Path | None:
    """Wire a completed cache without network access, including on offline restarts."""
    env = os.environ if environ is None else environ
    if not _gates_pass(facts, env):
        return None
    recipe = select_vllm_recipe(facts)
    if (
        recipe is None
        or not runtime_prerequisites_present(recipe)
        or not _complete(managed_vllm_root(recipe), recipe)
    ):
        return None
    binary = _binary(managed_vllm_root(recipe))
    env[VLLM_BIN_ENV] = str(binary)
    return binary


def installable_vllm_builds(
    facts: NodeFacts, *, offline: bool, environ: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Advertise first-use availability only for a matching published runtime and policy."""
    env = os.environ if environ is None else environ
    if offline or not _gates_pass(facts, env):
        return {}
    # uv is shipped with packaged Skulk as well as source installations.
    if not runtime_install_tool_available("uv"):
        return {}
    recipe = select_vllm_recipe(facts)
    return (
        {}
        if recipe is None or not runtime_prerequisites_present(recipe)
        else {f"vllm-{recipe.variant}": recipe.build_identity}
    )


def _recipe_lock(recipe: VllmRecipe, stop: threading.Event) -> bytes:
    content = bytearray()
    with httpx.stream(
        "GET", _INDEX_ROOT + recipe.filename, timeout=10, follow_redirects=True
    ) as response:
        response.raise_for_status()
        for chunk in response.iter_bytes():
            if stop.is_set():
                raise ComfyInstallCancelledError("vLLM install cancelled")
            content.extend(chunk)
            if len(content) > recipe.size_bytes:
                raise VllmInstallError(
                    "the package verification",
                    "recipe archive exceeds its recorded size",
                )
    if (
        len(content) != recipe.size_bytes
        or hashlib.sha256(content).hexdigest() != recipe.sha256
    ):
        raise VllmInstallError(
            "the package verification", "recipe archive checksum mismatch"
        )
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        name = f"skulk_vllm_runtime_{recipe.variant}/requirements.txt"
        info = archive.getinfo(name)
        if info.file_size > _RECIPE_LIMIT_BYTES:
            raise VllmInstallError(
                "the package verification", "dependency lock exceeds limit"
            )
        lock = archive.read(info)
    if hashlib.sha256(lock).hexdigest() != recipe.lock_sha256:
        raise VllmInstallError(
            "the package verification", "dependency lock checksum mismatch"
        )
    return lock


def _verify_runtime(
    root: Path, recipe: VllmRecipe, run: Runner, environment: dict[str, str]
) -> None:
    # Validate dependency completeness before touching GPU memory. Installation
    # never changes Skulk's environment or an operator's existing vLLM runtime.
    expected_versions = dict(
        re.findall(
            r"^([A-Za-z0-9_.-]+)==([^\s\\]+)",
            (root / "requirements.txt").read_text(),
            flags=re.MULTILINE,
        )
    )
    expected_versions["vllm"] = recipe.version
    # NVIDIA's official ARM cuSPARSELt wheel advertises the non-standard
    # manylinux2014_sbsa tag inside WHEEL, even though its filename and ELF
    # target are aarch64. uv pip check rejects that metadata on working GB10
    # runtimes. Compare every locked distribution directly instead; the
    # resolver established closure and --no-deps installed exactly that set.
    script = (
        "import importlib.metadata as m,sys,torch; "
        f"expected={expected_versions!r}; "
        "assert all(m.version(name)==version for name,version in expected.items()), 'locked dependency version mismatch'; "
        f"assert sys.version_info[:3] == {tuple(int(part) for part in VLLM_PYTHON.split('.'))!r}; "
        "assert torch.cuda.is_available(), 'no usable GPU'; "
        f"assert bool(torch.version.hip) == {recipe.variant == 'rocm'!r}, 'GPU runtime mismatch'; "
        "x=torch.ones(1,device='cuda'); assert x.item()==1; "
        "torch.cuda.synchronize(); "
        "from vllm.platforms import current_platform; "
        "assert current_platform.get_device_capability() is not None; "
        "print('skulk-vllm-runtime-ready')"
    )
    output = run_runtime_install_command(
        run,
        [str(root / "venv/bin/python"), "-c", script],
        timeout=300,
        what="vLLM GPU check",
        env=environment,
    )
    if "skulk-vllm-runtime-ready" not in output.splitlines():
        raise VllmInstallError("the GPU check", "runtime verification did not complete")


def install_vllm_on_demand(
    facts: NodeFacts,
    *,
    offline: bool,
    stop: threading.Event,
    run: Runner = subprocess.run,
    expected_build: str | None = None,
) -> Path:
    """Prepare, verify and atomically adopt one immutable isolated runtime.

    The caller supplies a cancellable runner. Failed installations never replace
    a ready environment; concurrent processes share a file lock and recheck it.
    Returns the CLI path and wires it for fresh node-fact publication.
    """
    configured = os.environ.get(VLLM_BIN_ENV, "").strip()
    if not configured and facts.vllm_binary.state == "ok":
        configured = facts.vllm_binary.configured_path or ""
    if configured:
        binary = Path(configured)
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise VllmInstallError(
                "the explicit runtime check", "configured vLLM CLI is not executable"
            )
        if (
            expected_build is not None
            and managed_vllm_build(str(binary)) != expected_build
        ):
            raise VllmInstallError(
                "the placement check",
                "the configured runtime differs from the planned immutable recipe; place the model again",
            )
        return binary
    recipe = select_vllm_recipe(facts)
    if expected_build is not None and (
        recipe is None or expected_build != recipe.build_identity
    ):
        raise VllmInstallError(
            "the placement check", "vLLM runtime recipe differs from placement"
        )
    cached = rehydrate_vllm(facts)
    if cached is not None:
        return cached
    if recipe is not None and not runtime_prerequisites_present(recipe):
        raise VllmInstallError(
            "the native runtime prerequisite check",
            "the host must supply the ROCm 7.2.3 SDK and OpenMPI 4.x libraries listed in the runtime guide",
        )
    if not installable_vllm_builds(facts, offline=offline):
        raise VllmInstallError(
            "the eligibility check",
            "no installable vLLM runtime for this node and policy",
        )
    recipe = select_vllm_recipe(facts)
    assert recipe is not None
    target = managed_vllm_root(recipe)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(target) + ".lock")
    deadline = time.monotonic() + _INSTALL_TIMEOUT_SECONDS
    while True:
        if stop.is_set():
            raise ComfyInstallCancelledError("vLLM install cancelled while waiting")
        try:
            lock.acquire(timeout=0.5)
            break
        except FileLockTimeout:
            if time.monotonic() >= deadline:
                raise VllmInstallError(
                    "the installation lock", "another installation did not finish"
                ) from None
    try:
        if not _complete(target, recipe):
            # Holding the cross-process recipe lock proves no other installer
            # owns these same-prefix staging trees. Recover hard-kill leftovers
            # before the disk check without touching any other runtime recipe.
            for abandoned in target.parent.glob(f".{target.name}-*"):
                if abandoned.is_dir() and not abandoned.is_symlink():
                    shutil.rmtree(abandoned)
            if shutil.disk_usage(target.parent).free < VLLM_INSTALL_FREE_BYTES:
                raise VllmInstallError(
                    "the disk-space check", "vLLM installation requires 20 GiB free"
                )
            environment = {
                key: value
                for key, value in sanitized_index_environment().items()
                if not key.startswith(("UV_", "PIP_"))
            }
            with tempfile.TemporaryDirectory(
                dir=target.parent, prefix=f".{target.name}-"
            ) as staging:
                root = Path(staging) / "runtime"
                root.mkdir()
                (root / "requirements.txt").write_bytes(_recipe_lock(recipe, stop))
                uv = runtime_install_tool("uv")
                run_runtime_install_command(
                    run,
                    [
                        uv,
                        "venv",
                        "--no-config",
                        "--python",
                        VLLM_PYTHON,
                        "--python-preference",
                        "only-managed",
                        str(root / "venv"),
                    ],
                    timeout=600,
                    what="vLLM Python environment",
                    env=environment,
                )
                arguments = [
                    uv,
                    "pip",
                    "install",
                    "--no-config",
                    "--python",
                    str(root / "venv/bin/python"),
                    "--require-hashes",
                    "--only-binary",
                    ":all:",
                    "--no-deps",
                    "-r",
                    str(root / "requirements.txt"),
                ]
                if recipe.variant == "cuda":
                    arguments += ["--torch-backend", "cu129"]
                run_runtime_install_command(
                    run,
                    arguments,
                    timeout=_INSTALL_TIMEOUT_SECONDS,
                    what="vLLM wheel installation",
                    env=environment,
                )
                _verify_runtime(root, recipe, run, environment)
                # Console scripts contain absolute shebangs. Correct them for
                # their final path before publishing the staged environment.
                for script in (root / "venv/bin").iterdir():
                    if script.is_symlink() or not script.is_file():
                        continue
                    with script.open("rb") as stream:
                        header = stream.read(512)
                    if header.startswith(b"#!"):
                        # uv uses a shell trampoline for long interpreter
                        # paths; relocating only a Python shebang misses it.
                        content = script.read_bytes()
                        script.write_bytes(
                            content.replace(str(root).encode(), str(target).encode())
                        )
                (root / "provisioned.json").write_text(
                    json.dumps(
                        {
                            "build": recipe.build_identity,
                            "python": VLLM_PYTHON,
                            "gpu_verified": True,
                        }
                    )
                )
                if stop.is_set():
                    raise ComfyInstallCancelledError(
                        "vLLM install cancelled before adoption"
                    )
                if target.exists():
                    # Only the current recipe's incomplete cache is replaceable.
                    # Completed older recipes live at different immutable paths.
                    shutil.rmtree(target)
                root.rename(target)
        os.environ[VLLM_BIN_ENV] = str(_binary(target))
        return _binary(target)
    except (
        OSError,
        ValueError,
        httpx.HTTPError,
        subprocess.SubprocessError,
        RuntimeError,
    ) as error:
        if isinstance(error, VllmInstallError):
            raise
        raise VllmInstallError("the engine download and install", str(error)) from error
    finally:
        lock.release()


def main() -> None:
    """Eager installer compatibility entry point; ordinary nodes install on first use."""
    from skulk.facts import gather_node_facts
    from skulk.provisioning.comfy import stoppable_runner
    from skulk.shared.constants import offline_mode

    stop = threading.Event()
    print(
        install_vllm_on_demand(
            gather_node_facts(),
            offline=offline_mode(),
            stop=stop,
            run=stoppable_runner(stop),
        )
    )


if __name__ == "__main__":
    main()
