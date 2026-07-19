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
    return [
        CheckResult(
            check_id=check_id,
            title=title,
            verdict="fail",
            detail="no inference engine is importable or configured on this node",
            consequence=(
                "the node advertises no backends and will never be selected to "
                "serve a model; it participates in the cluster as management "
                "only"
            ),
            remediation=(
                "`skulk doctor --fix` provisions the pinned llama-server build "
                "on Linux; alternatively install a llama-cpp-python build, set "
                "SKULK_LLAMA_SERVER_BIN to a custom llama-server, or set "
                "SKULK_VLLM_BIN to a vllm CLI"
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
