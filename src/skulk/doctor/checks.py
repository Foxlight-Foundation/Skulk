"""The node environment contract, executable: skulk doctor's check registry.

Every check reads the same :class:`~skulk.shared.types.node_facts.NodeFacts`
snapshot the capability pipeline uses (#614: nobody probes ad hoc) plus a few
injected system probes (disk, directory writability). A check yields zero or
more :class:`CheckResult` verdicts; every non-OK verdict states its
CONSEQUENCE (what serving behavior degrades) and its remediation, and marks
whether ``skulk doctor --fix`` can remediate it safely.

The registry is also the source the user-facing platform documentation is
generated from (``scripts/generate_doctor_docs.py``), so the docs and the
checks cannot drift apart.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, final

from pydantic import ConfigDict

from skulk.facts import derive_node_backends
from skulk.shared.types.node_facts import (
    CONFLICT_ERROR_CODES,
    NodeFacts,
)
from skulk.utils.pydantic_ext import CamelCaseModel

CheckVerdict = Literal["ok", "degraded", "fail"]
"""Outcome of one doctor check.

``ok`` means the contract holds. ``degraded`` means serving works but below
the hardware's capability or with reduced observability. ``fail`` means
serving is broken or misconfigured in a way that will visibly hurt.
"""


@final
class CheckResult(CamelCaseModel):
    """One verdict from one doctor check, self-contained for rendering."""

    model_config = ConfigDict(frozen=True)

    check_id: str
    """Stable identifier of the producing check."""

    title: str
    """Short human title for the finding."""

    verdict: CheckVerdict
    """The outcome (see :data:`CheckVerdict`)."""

    detail: str
    """What was observed, with concrete values."""

    consequence: str = ""
    """What this means for serving when non-OK (empty for OK verdicts)."""

    remediation: str = ""
    """The operator's path to resolve it (empty for OK verdicts)."""

    fix_available: bool = False
    """Whether ``skulk doctor --fix`` can remediate this safely."""


@final
@dataclass(frozen=True)
class DoctorCheck:
    """One entry in the check registry.

    ``run`` is a pure function over the facts snapshot (plus any injected
    probes bound at construction); ``fix`` is an optional idempotent
    remediation returning a one-line description of what it did.
    """

    check_id: str
    title: str
    docs: str
    """One-paragraph description for the generated platform documentation:
    what the check verifies and why it matters."""

    run: Callable[[NodeFacts], Sequence[CheckResult]]
    fix: Callable[[NodeFacts], str | None] | None = None


def _ok(check_id: str, title: str, detail: str) -> CheckResult:
    return CheckResult(check_id=check_id, title=title, verdict="ok", detail=detail)


# --- engine availability ---------------------------------------------------


def _declared_participation() -> str:
    """The node's declared participation (management nodes need no engine).

    Normalized exactly like ``NodeResources.gather()``: an unrecognized value
    (a typo like ``managment``) falls back to ``full``, so the doctor judges
    the same effective role the runtime will actually use for placement.
    """
    declared = os.environ.get("SKULK_NODE_PARTICIPATION", "full").strip().lower()
    return declared if declared in ("full", "management", "ffn_only") else "full"


def _provisioning_fix_applicable(facts: NodeFacts) -> bool:
    """Whether --fix can actually provision an engine on this node.

    Mirrors ensure_llama_server's own gates so the doctor never promises a
    remediation that would inevitably fail: Linux only, no explicit override
    (valid or not; an invalid override is its own conflict, not something to
    paper over), and auto-provisioning not opted out.
    """
    from skulk.provisioning.llama_server import AUTOPROVISION_OPT_OUT_ENV

    return (
        facts.platform == "linux"
        and _declared_participation() == "full"
        and facts.llama_server_binary.state == "not_configured"
        and os.environ.get(AUTOPROVISION_OPT_OUT_ENV, "").strip() != "1"
    )


def _check_engine_available(facts: NodeFacts) -> Sequence[CheckResult]:
    """At least one inference engine must be usable on this node.

    Usability is judged from the derived backends, not raw binary presence:
    a vllm CLI on a node with no GPU derives no tags and serves nothing, so
    it must not count as an available engine.
    """
    check_id = "engine-available"
    title = "Inference engine availability"
    if _declared_participation() != "full":
        return [
            _ok(
                check_id,
                title,
                f"declared participation is {_declared_participation()!r}; "
                "this node serves no inference shards and needs no engine",
            )
        ]
    derived = derive_node_backends(facts).backends
    engines: list[str] = []
    if "mlx" in derived:
        engines.append("mlx (in-process)")
    if "mlx_audio" in derived:
        engines.append("mlx_audio (speech)")
    if "llama_cpp" in derived:
        engines.append("llama_cpp (in-process GGUF)")
    if "llama_server" in derived:
        engines.append(f"llama_server ({facts.llama_server_binary.configured_path})")
    if "vllm" in derived:
        engines.append(f"vllm ({facts.vllm_binary.configured_path})")
    if engines:
        return [_ok(check_id, title, "available: " + ", ".join(engines))]
    # A wheel-provisioned or previously provisioned managed engine derives no
    # tags until node startup exports its path; without this read-only lookup
    # plain doctor reports FAIL on exactly the box the installer just set up
    # (#628). --fix and startup share the same discovery via
    # ensure_llama_server. The candidate is judged by REPLAYING the real
    # derivation over a hypothetical facts snapshot with the binary wired:
    # any candidate startup would disable (failed device probe from a dead
    # Vulkan ICD or missing CUDA loader, a CPU-only build on a GPU node's
    # gpu_serving_disabled, ...) must not read as available, or #628's false
    # negative becomes a false positive (PR #634 review, both rounds).
    from skulk.facts.probe import probe_llama_server_devices
    from skulk.provisioning import dormant_llama_server
    from skulk.shared.backends import LLAMA_SERVER_BIN_ENV
    from skulk.shared.types.node_facts import EngineBinaryFact

    dormant = dormant_llama_server(facts)
    broken_dormant_detail: str | None = None
    if dormant is not None:
        hypothetical = facts.model_copy(
            update={
                "llama_server_binary": EngineBinaryFact(
                    env_var=LLAMA_SERVER_BIN_ENV,
                    configured_path=str(dormant),
                    state="ok",
                ),
                "llama_server_device_probe": probe_llama_server_devices(
                    str(dormant)
                ),
            }
        )
        replay = derive_node_backends(hypothetical)
        if "llama_server" in replay.backends:
            # Derivation can advertise the engine AND raise conflicts (the
            # #609 cpu-only-on-GPU shape keeps llama_server-cpu with an
            # error-level gpu_serving_disabled). The capability-conflicts
            # check reads the unwired facts and cannot see them before
            # startup, so each replayed conflict becomes its own verdict here
            # with the same error->fail mapping; the audit's exit code then
            # matches what startup will flag instead of reading healthy.
            results = [
                _ok(
                    check_id,
                    title,
                    f"llama_server ({dormant}) is installed and wires "
                    "automatically at node startup",
                )
            ]
            for conflict in replay.conflicts:
                conflict_verdict: CheckVerdict = (
                    "fail" if conflict.code in CONFLICT_ERROR_CODES else "degraded"
                )
                results.append(
                    CheckResult(
                        check_id=check_id,
                        title=f"Startup capability conflict: {conflict.code}",
                        verdict=conflict_verdict,
                        detail=conflict.message,
                        consequence=(
                            "node startup will wire this engine and raise "
                            "this conflict into nodeHealth"
                        ),
                        remediation=conflict.remediation,
                    )
                )
            return results
        reason = "; ".join(c.message for c in replay.conflicts) or (
            "the binary derives no usable backend on this hardware"
        )
        broken_dormant_detail = (
            f"an installed managed llama-server at {dormant} would be "
            f"disabled at startup: {reason}"
        )
    return [
        CheckResult(
            check_id=check_id,
            title=title,
            verdict="fail",
            detail=broken_dormant_detail
            or "no inference engine is importable or configured on this node",
            consequence=(
                "the node advertises no backends and will never be selected to "
                "serve a model; it participates in the cluster as management "
                "only"
            ),
            remediation=(
                "`skulk doctor --fix` provisions the pinned llama-server build "
                "on Linux (on an NVIDIA node this installs the "
                "skulk-llama-server-cuda wheel from the Foxlight index, then "
                "falls back to the managed Vulkan/CPU lanes); alternatively install "
                "a llama-cpp-python build, set SKULK_LLAMA_SERVER_BIN to a "
                "custom llama-server, or set SKULK_VLLM_BIN to a vllm CLI"
            ),
            fix_available=_provisioning_fix_applicable(facts),
        )
    ]


def _fix_engine_available(facts: NodeFacts) -> str | None:
    """Provision the pinned llama-server build when no engine is available."""
    if not _provisioning_fix_applicable(facts):
        return None
    if derive_node_backends(facts).backends:
        # Some engine already derives usable tags; nothing to provision.
        return None
    from skulk.provisioning import ensure_llama_server

    binary = ensure_llama_server(facts)
    if binary is None:
        raise RuntimeError(
            "engine provisioning did not produce a binary (override present, "
            "opted out, or download failed; see the log)"
        )
    return f"provisioned pinned llama-server at {binary}"


# --- capability conflicts --------------------------------------------------


def _check_capability_conflicts(facts: NodeFacts) -> Sequence[CheckResult]:
    """Surface every backend-derivation conflict as a doctor verdict."""
    check_id = "capability-conflicts"
    derivation = derive_node_backends(facts)
    if not derivation.conflicts:
        return [
            _ok(
                check_id,
                "Capability conflicts",
                f"none; advertising {sorted(derivation.backends) or 'no backends'}",
            )
        ]
    results: list[CheckResult] = []
    for conflict in derivation.conflicts:
        verdict: CheckVerdict = (
            "fail" if conflict.code in CONFLICT_ERROR_CODES else "degraded"
        )
        results.append(
            CheckResult(
                check_id=check_id,
                title=f"Capability conflict: {conflict.code}",
                verdict=verdict,
                detail=conflict.message,
                consequence=(
                    "serving runs far below hardware capability"
                    if conflict.code == "gpu_serving_disabled"
                    else "capability or observability is degraded on this node"
                ),
                remediation=conflict.remediation,
                # nvidia-ml-py installation is the one conflict --fix owns.
                fix_available=(
                    conflict.code == "gpu_detection_degraded"
                    and not facts.pynvml_importable
                ),
            )
        )
    return results


def _fix_capability_conflicts(facts: NodeFacts) -> str | None:
    """Install nvidia-ml-py when detection is degraded by its absence."""
    degraded = any(
        gpu.detection_source == "nvidia_device_node" for gpu in facts.gpus
    )
    if not degraded or facts.pynvml_importable:
        return None
    # uv-managed environments ship no pip module in the interpreter, so
    # prefer `uv pip install --python <this interpreter>` and fall back to
    # `python -m pip` for conventional venvs.
    uv = shutil.which("uv")
    command = (
        [uv, "pip", "install", "--python", sys.executable, "nvidia-ml-py"]
        if uv is not None
        else [sys.executable, "-m", "pip", "install", "nvidia-ml-py"]
    )
    completed = subprocess.run(  # noqa: S603 - fixed, known-safe command
        command,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"nvidia-ml-py install failed: {completed.stderr.strip()[-300:]}"
        )
    return "installed nvidia-ml-py (restart skulk to pick it up)"


# --- storage ---------------------------------------------------------------

# Disk thresholds mirror node_health's models-volume thresholds.
_DISK_LOW_GB = 10.0
_DISK_FULL_GB = 2.0


def _check_models_storage(facts: NodeFacts) -> Sequence[CheckResult]:
    """The models volume must exist, be writable, and have headroom."""
    del facts
    check_id = "models-storage"
    title = "Model storage"
    from skulk.shared.constants import SKULK_MODELS_DIR

    models_dir = Path(SKULK_MODELS_DIR)
    if not models_dir.exists():
        return [
            CheckResult(
                check_id=check_id,
                title=title,
                verdict="degraded",
                detail=f"models directory {models_dir} does not exist yet",
                consequence="it will be created on first download",
                remediation="run skulk once, or `skulk doctor --fix` creates it",
                fix_available=True,
            )
        ]
    if not models_dir.is_dir():
        return [
            CheckResult(
                check_id=check_id,
                title=title,
                verdict="fail",
                detail=f"models path {models_dir} exists but is not a directory",
                consequence="every model download on this node will fail",
                remediation=(
                    f"move or remove the file at {models_dir} (or point "
                    "SKULK_MODELS_DIR at a directory)"
                ),
            )
        ]
    if not os.access(models_dir, os.W_OK):
        return [
            CheckResult(
                check_id=check_id,
                title=title,
                verdict="fail",
                detail=f"models directory {models_dir} is not writable",
                consequence="every model download on this node will fail",
                remediation=f"fix ownership/permissions on {models_dir}",
            )
        ]
    usage = shutil.disk_usage(models_dir)
    free_gb = usage.free / 2**30
    if free_gb <= _DISK_FULL_GB:
        return [
            CheckResult(
                check_id=check_id,
                title=title,
                verdict="fail",
                detail=f"models volume effectively full: {free_gb:.1f} GB free",
                consequence="the next model download is certain to fail",
                remediation="free disk space or lower staging_keep_recent_gb",
            )
        ]
    if free_gb < _DISK_LOW_GB:
        return [
            CheckResult(
                check_id=check_id,
                title=title,
                verdict="degraded",
                detail=f"models volume low on space: {free_gb:.1f} GB free",
                consequence="large model downloads may fail",
                remediation="free disk space before pulling a large model",
            )
        ]
    return [_ok(check_id, title, f"{models_dir} writable, {free_gb:.0f} GB free")]


def _fix_models_storage(facts: NodeFacts) -> str | None:
    """Create the models directory when it does not exist yet."""
    del facts
    from skulk.shared.constants import SKULK_MODELS_DIR

    models_dir = Path(SKULK_MODELS_DIR)
    if models_dir.exists():
        return None
    models_dir.mkdir(parents=True, exist_ok=True)
    return f"created models directory {models_dir}"


# --- dashboard -------------------------------------------------------------


def _check_dashboard_assets(facts: NodeFacts) -> Sequence[CheckResult]:
    """Whether the built dashboard is present (the API serves without it)."""
    del facts
    check_id = "dashboard-assets"
    title = "Dashboard assets"
    from skulk.shared.constants import DASHBOARD_DIR

    if DASHBOARD_DIR is not None:
        return [_ok(check_id, title, f"built dashboard at {DASHBOARD_DIR}")]
    return [
        CheckResult(
            check_id=check_id,
            title=title,
            verdict="degraded",
            detail="no built dashboard assets found",
            consequence=(
                "the API serves normally but this node hosts no web UI "
                "(expected on headless workers)"
            ),
            remediation=(
                "build the dashboard (`cd dashboard-react && npm install && "
                "npm run build`) or browse another node's dashboard"
            ),
        )
    ]


# --- hugging face token ----------------------------------------------------


def _is_peer_id_store_host(store_host: str) -> bool:
    """Whether ``store_host`` looks like a libp2p peer ID rather than a hostname.

    Peer IDs are base58 and carry no dots; hostnames in this field are short
    names or ``.local`` spellings. The distinction matters because hostname
    matching is decidable from doctor while peer-ID matching is not.
    """
    return "." not in store_host and store_host.startswith(("12D3KooW", "Qm"))


def _fetching_role(facts: NodeFacts) -> tuple[bool, str]:
    """Whether this node performs Hugging Face fetches, and why.

    A token only matters on the node that actually reaches out to Hugging
    Face. With a model store enabled that is the store host; without one,
    every node downloads for itself. Returning the reason lets the verdict
    explain itself instead of asserting a role the operator cannot see.
    """
    del facts
    from skulk.shared.constants import SKULK_OFFLINE
    from skulk.store.config import (
        load_skulk_config,
        node_matches_store_host,
        resolve_config_path,
    )

    if SKULK_OFFLINE:
        # Offline mode is an explicit declaration that this node fetches
        # nothing from the network, so a missing token cannot bite.
        return False, "this node runs in offline mode and downloads nothing"

    participation = _declared_participation()

    def _participation_exempt() -> tuple[bool, str]:
        # placement.py hard-filters every participation value other than
        # "full", so neither a management node nor an ffn_only one is ever
        # assigned an inference shard, and neither downloads weights. A
        # permanent degraded verdict there would be pure noise. Checked only
        # after the store-host question, because a non-serving node can still
        # be the configured store host and would then fetch for the fleet.
        return False, (
            f"this node declares {participation} participation, so the planner "
            "assigns it no inference shard and it downloads no models"
        )

    config_path = resolve_config_path()
    if not config_path.exists():
        if participation != "full":
            return _participation_exempt()
        # skulk.yaml is resolved relative to the working directory, so this is
        # either a genuinely zero-config node (which does download for itself)
        # or doctor being run from somewhere other than the install directory.
        # Say which, rather than asserting a store layout we cannot see.
        return True, (
            f"no {config_path} in the working directory, so this reads as a "
            "zero-config node that downloads directly; if this node uses a "
            "model store, re-run doctor from its install directory"
        )
    try:
        config = load_skulk_config()
    except Exception:  # noqa: BLE001 - a broken config is another check's job
        # Unreadable config: assume this node fetches, because warning about a
        # token that turns out to be unnecessary is far cheaper than staying
        # silent on the node that actually needed one.
        return True, f"{config_path} could not be read, assuming direct downloads"
    store = config.model_store if config is not None else None
    if store is None or not store.enabled:
        if participation != "full":
            return _participation_exempt()
        return True, "no model store is configured, so this node downloads directly"
    if node_matches_store_host(store.store_host, node_id="", hostname=None):
        # Deliberately ahead of the participation exemption: hosting the store
        # is not an inference role, so a management or ffn_only node can be the
        # store host and would then fetch for the whole fleet.
        return True, f"this node is the model store host ({store.store_host})"
    if _is_peer_id_store_host(store.store_host):
        # store_host may be a libp2p peer ID, which only the running node can
        # match against its own ephemeral ID. Doctor has no node ID, so it
        # cannot rule out that this node is the store host. Claiming "a worker,
        # no token needed" here would reintroduce exactly the silent gap this
        # check exists to close, so report the ambiguity instead.
        return True, (
            f"the model store host is configured as a node ID "
            f"({store.store_host}), which doctor cannot match against this "
            "node; if this node is the store host, it needs the token"
        )
    if participation != "full":
        return _participation_exempt()
    if store.download.allow_hf_fallback:
        # #657: a worker that cannot reach the store falls back to downloading
        # from Hugging Face itself, so "only the store host fetches" is not
        # strictly true. Not a warning, because that fallback may never fire
        # and yellowing every worker in the fleet would drown the signal, but
        # the caveat belongs in the detail.
        return False, (
            f"the model store host ({store.store_host}) performs downloads; "
            "this node would need its own token only if it falls back to "
            "downloading directly because the store host is unreachable "
            "(allow_hf_fallback is on)"
        )
    return False, f"the model store host ({store.store_host}) performs downloads"


def _check_hf_token(facts: NodeFacts) -> Sequence[CheckResult]:
    """Whether this node can authenticate to Hugging Face, if it needs to."""
    check_id = "hf-token"
    title = "Hugging Face token"
    from skulk.download.huggingface_utils import (
        get_hf_token_path,
        resolve_hf_token_source,
    )

    _token, source = resolve_hf_token_source()
    fetches, reason = _fetching_role(facts)

    if source == "env":
        return [
            _ok(
                check_id,
                title,
                "token configured via the HF_TOKEN environment variable "
                "(set directly, or from hf_token in skulk.yaml at startup)",
            )
        ]
    if source == "service_env":
        from skulk.download.huggingface_utils import get_service_env_path

        return [
            _ok(
                check_id,
                title,
                f"token configured as HF_TOKEN in {get_service_env_path()}, "
                "which the service startup wrapper exports; a node launched "
                "directly with `uv run skulk` does not read that file and "
                "would need HF_TOKEN in its own environment",
            )
        ]
    if source == "config":
        return [
            _ok(
                check_id,
                title,
                "token configured via hf_token in skulk.yaml (what the "
                "dashboard writes); node startup copies it into HF_TOKEN",
            )
        ]
    if source == "file":
        return [_ok(check_id, title, f"token configured at {get_hf_token_path()}")]

    if not fetches:
        # No token here is entirely normal on a worker: it never talks to
        # Hugging Face. Saying so beats a warning the operator cannot act on.
        return [
            _ok(
                check_id,
                title,
                f"no token on this node, which is expected: {reason}",
            )
        ]
    return [
        CheckResult(
            check_id=check_id,
            title=title,
            verdict="degraded",
            detail=f"no Hugging Face token is configured, and {reason}",
            consequence=(
                "public models download normally, but every gated or private "
                "repository (Llama and Gemma among them) fails to download "
                "on this node"
            ),
            remediation=(
                "on a formed cluster, enter the token once in any node's "
                "dashboard Settings; it propagates to every node including "
                "this one. On a single node, run `hf auth login` (writes "
                f"{get_hf_token_path()} and is picked up without a restart) "
                "or set HF_TOKEN in ~/.skulk/skulk.env and restart."
            ),
        )
    ]


# --- vllm prerequisites ------------------------------------------------------

_CXX_COMPILERS = ("g++", "clang++", "c++")
"""C++ compiler names Inductor will look for on PATH, in no particular order.

Inductor drives a **C++** compiler, not a C one: ``torch._inductor.cpp_builder``
resolves ``$CXX`` and otherwise joins ``bin/g++``. A box carrying ``gcc`` with
no ``g++`` therefore still fails, which is why checking for ``cc``/``gcc``
would report a toolchain that cannot actually build the kernels.
"""


def _vllm_interpreter(vllm_binary_path: str) -> Path | None:
    """Locate the interpreter that runs the configured vLLM entry point.

    The adjacent ``python`` covers the venv layout the installer creates. A
    console script installed elsewhere (a user install under ``~/.local/bin``,
    a pipx shim) has no sibling interpreter, so fall back to the shebang pip
    wrote into the script, which names the real one.
    """
    adjacent = Path(vllm_binary_path).with_name("python")
    if adjacent.exists():
        return adjacent
    try:
        with Path(vllm_binary_path).open("rb") as handle:
            first_line = handle.readline(512).decode("utf-8", "replace").strip()
    except OSError:
        return None
    if not first_line.startswith("#!"):
        return None
    tokens = first_line[2:].split()
    if not tokens:
        return None
    first = Path(tokens[0])
    # "#!/usr/bin/env python3" and its "env -S" form name the interpreter as an
    # argument, so the leading path is env itself. Returning that would probe
    # `env -c ...`, which fails and silently skips header verification.
    if first.name == "env":
        for argument in tokens[1:]:
            if argument.startswith("-"):
                continue
            resolved = shutil.which(argument)
            return Path(resolved) if resolved else None
        return None
    return first if first.is_absolute() and first.exists() else None


def _vllm_include_dir(vllm_binary_path: str) -> Path | None:
    """Ask the vLLM venv's own interpreter where its C headers would live.

    The vLLM engine runs in its own virtualenv with its own Python version, so
    the headers that matter are that interpreter's, not the ones Skulk is
    running under. Asking the interpreter itself avoids guessing a version.
    """
    interpreter = _vllm_interpreter(vllm_binary_path)
    if interpreter is None:
        return None
    try:
        completed = subprocess.run(  # noqa: S603 - fixed, known-safe command
            [str(interpreter), "-c", "import sysconfig;print(sysconfig.get_paths()['include'])"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    include = completed.stdout.strip()
    return Path(include) if include else None


def _check_vllm_prerequisites(facts: NodeFacts) -> Sequence[CheckResult]:
    """vLLM's Triton JIT needs a C toolchain that the wheel does not install."""
    check_id = "vllm-prerequisites"
    title = "vLLM build prerequisites"

    binary = facts.vllm_binary
    if binary.state != "ok" or binary.configured_path is None:
        # No usable vLLM on this node: engine-available already reports on the
        # configured-but-broken states, and there is nothing to prepare for.
        return [_ok(check_id, title, "no vLLM engine configured on this node")]

    missing: list[str] = []
    if not any(shutil.which(compiler) for compiler in _CXX_COMPILERS):
        missing.append("a C++ compiler (g++/clang++/c++) on PATH")
    include_dir = _vllm_include_dir(binary.configured_path)
    if include_dir is None:
        # Not knowing is not the same as knowing it is broken. Reporting fail
        # here would tell an operator their working node cannot serve, which is
        # worse than staying quiet about the half we could not determine.
        if not missing:
            return [
                _ok(
                    check_id,
                    title,
                    "C++ toolchain present; could not resolve the vLLM "
                    "interpreter's include directory, so headers were not "
                    "verified",
                )
            ]
    elif not (include_dir / "Python.h").exists():
        missing.append(f"Python development headers (no Python.h in {include_dir})")

    if not missing:
        return [
            _ok(
                check_id,
                title,
                "C++ toolchain and Python development headers present for the "
                "vLLM engine",
            )
        ]
    return [
        CheckResult(
            check_id=check_id,
            title=title,
            verdict="fail",
            detail="vLLM cannot JIT-compile its kernels: missing " + "; ".join(missing),
            consequence=(
                "the node advertises vLLM capacity and accepts placements, but "
                "every engine start fails during initialization with an "
                "InductorError, so the model never serves"
            ),
            remediation=(
                "install the Python development headers and a C++ compiler "
                "for the vLLM interpreter (Debian and Ubuntu: "
                "`sudo apt install python3-dev build-essential`; RHEL family: "
                "`sudo dnf install python3-devel gcc-c++`), then retry the "
                "placement"
            ),
        )
    ]


# --- registry --------------------------------------------------------------

REGISTRY: tuple[DoctorCheck, ...] = (
    DoctorCheck(
        check_id="engine-available",
        title="Inference engine availability",
        docs=(
            "Verifies at least one inference engine is usable: in-process MLX "
            "on macOS, an importable llama-cpp-python build, a llama-server "
            "binary (SKULK_LLAMA_SERVER_BIN), or a vllm CLI (SKULK_VLLM_BIN). "
            "A node with none advertises no backends and can only participate "
            "as management."
        ),
        run=_check_engine_available,
        fix=_fix_engine_available,
    ),
    DoctorCheck(
        check_id="capability-conflicts",
        title="Capability conflicts",
        docs=(
            "Runs backend derivation over the node facts snapshot and surfaces "
            "every observation-vs-declaration conflict: a GPU that no engine "
            "would use (silent CPU serving), degraded NVIDIA detection "
            "(missing nvidia-ml-py or a driver mismatch), an engine binary "
            "override pointing at an unusable path, or a declared backend the "
            "observed hardware cannot support."
        ),
        run=_check_capability_conflicts,
        fix=_fix_capability_conflicts,
    ),
    DoctorCheck(
        check_id="models-storage",
        title="Model storage",
        docs=(
            "Verifies the models directory exists, is writable, and has "
            "download headroom (warns under 10 GB free, fails at 2 GB or less)."
        ),
        run=_check_models_storage,
        fix=_fix_models_storage,
    ),
    DoctorCheck(
        check_id="dashboard-assets",
        title="Dashboard assets",
        docs=(
            "Reports whether the built web dashboard is present. The API "
            "serves without it; headless workers are expected to run this way."
        ),
        run=_check_dashboard_assets,
    ),
    DoctorCheck(
        check_id="hf-token",
        title="Hugging Face token",
        docs=(
            "Reports whether this node can authenticate to Hugging Face, and "
            "whether it is the node that needs to. A token entered in any "
            "node's dashboard Settings propagates over the encrypted cluster "
            "fabric to every node, and joining nodes adopt it at bootstrap, "
            "so one entry covers the fleet; this check verifies it actually "
            "arrived on the node that performs downloads (the model store "
            "host when a store is configured, otherwise this node itself). "
            "Without one, public models still download and only gated or "
            "private repositories fail."
        ),
        run=_check_hf_token,
    ),
    DoctorCheck(
        check_id="vllm-prerequisites",
        title="vLLM build prerequisites",
        docs=(
            "When a vLLM engine is configured, verifies the node can actually "
            "compile its kernels. vLLM JITs Triton and torch.compile kernels "
            "at runtime, shelling out to a C++ compiler (Inductor drives g++, "
            "so gcc alone is not enough) against the Python development "
            "headers; neither is a dependency of the vLLM wheel. Without them "
            "the node advertises vLLM capacity and accepts placements, then "
            "fails every engine start with an InductorError."
        ),
        run=_check_vllm_prerequisites,
    ),
)


def run_checks(facts: NodeFacts) -> list[CheckResult]:
    """Run every registered check against one facts snapshot.

    A crashing check must not take the audit down: it degrades into a ``fail``
    verdict naming the check, because a doctor that dies mid-audit on the very
    machines it exists for is worse than useless.
    """
    results: list[CheckResult] = []
    for check in REGISTRY:
        try:
            results.extend(check.run(facts))
        except Exception as error:  # noqa: BLE001 - audit must complete
            results.append(
                CheckResult(
                    check_id=check.check_id,
                    title=check.title,
                    verdict="fail",
                    detail=f"check crashed: {error}",
                    consequence="this aspect of the environment is unverified",
                    remediation="report this as a skulk bug with the error above",
                )
            )
    return results


def run_fixes(facts: NodeFacts) -> list[str]:
    """Run every check's idempotent remediation; returns what was done.

    Fix failures are reported, not raised: one broken remediation must not
    stop the rest.
    """
    actions: list[str] = []
    for check in REGISTRY:
        if check.fix is None:
            continue
        try:
            action = check.fix(facts)
        except Exception as error:  # noqa: BLE001 - keep fixing the rest
            actions.append(f"[{check.check_id}] fix failed: {error}")
            continue
        if action is not None:
            actions.append(f"[{check.check_id}] {action}")
    return actions
