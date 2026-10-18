# pyright: reportPrivateUsage=false
"""Immutable vLLM install acceptance, rollback, cache reuse and policy regression."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import threading
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import httpx
import pytest

import skulk.provisioning.vllm_runtime as runtime
from skulk.facts.derive import derive_node_backends
from skulk.facts.inventory import engine_build_inventory
from skulk.facts.testing import AMD_STRIX, NVIDIA_A40, bad_bin, make_facts
from skulk.provisioning.comfy import ComfyInstallCancelledError
from skulk.shared.backends import VLLM_BIN_ENV


@pytest.fixture(autouse=True)
def isolated_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Use synthetic hardware, an isolated cache and a deterministic recipe channel."""
    monkeypatch.setattr(runtime, "SKULK_ENGINES_DIR", tmp_path / "engines")
    monkeypatch.setattr(runtime.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(
        runtime, "platform_tags", lambda: iter(("manylinux_2_28_aarch64",))
    )

    def tool_available(_name: str) -> bool:
        return True

    def tool(_name: str) -> str:
        return "/fake/uv"

    monkeypatch.setattr(runtime, "runtime_install_tool_available", tool_available)
    monkeypatch.setattr(runtime, "runtime_install_tool", tool)
    monkeypatch.setattr(runtime, "VLLM_INSTALL_FREE_BYTES", 0)
    for variable in (
        VLLM_BIN_ENV,
        runtime.AUTOPROVISION_OPT_OUT_ENV,
        "SKULK_NODE_PARTICIPATION",
    ):
        monkeypatch.delenv(variable, raising=False)


def _channel(
    monkeypatch: pytest.MonkeyPatch, *, corrupt: bool = False
) -> runtime.VllmRecipe:
    original = runtime.PUBLISHED_RECIPES[0]
    lock = b"--index-url https://pypi.org/simple/\nninja==1.13.2 --hash=sha256:abcd\n"
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w") as archive:
        archive.writestr("skulk_vllm_runtime_cuda/requirements.txt", lock)
    payload = archive_bytes.getvalue()
    recipe = original.model_copy(
        update={
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
            "lock_sha256": hashlib.sha256(lock).hexdigest(),
        }
    )
    monkeypatch.setattr(runtime, "PUBLISHED_RECIPES", (recipe,))

    @contextmanager
    def stream(method: str, url: str, **_kwargs: object) -> Iterator[httpx.Response]:
        assert method == "GET" and url.endswith(recipe.filename)
        yield httpx.Response(
            200,
            request=httpx.Request(method, url),
            content=b"invalid" if corrupt else payload,
        )

    monkeypatch.setattr(runtime.httpx, "stream", stream)
    return recipe


def _run(
    calls: list[list[str]],
    *,
    gpu_fails: bool = False,
    stop: threading.Event | None = None,
) -> runtime.Runner:
    def run(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(arguments)
        if arguments[1] == "venv":
            binary_dir = Path(arguments[-1]) / "bin"
            binary_dir.mkdir(parents=True)
            (binary_dir / "python").symlink_to(sys.executable)
            script = binary_dir / "vllm"
            script.write_text(f"#!{binary_dir}/python\nprint('ready')\n")
            script.chmod(0o755)
        if "install" in arguments:
            assert "--require-hashes" in arguments and "--no-deps" in arguments
            assert arguments[arguments.index("--only-binary") + 1] == ":all:"
            environment = cast(dict[str, str], kwargs["env"])
            assert not any(key.startswith(("UV_", "PIP_")) for key in environment)
            if stop is not None:
                stop.set()
                raise ComfyInstallCancelledError("worker shut down")
        if "-c" in arguments:
            assert "locked dependency version mismatch" in arguments[-1]
            return subprocess.CompletedProcess(
                arguments,
                1 if gpu_fails else 0,
                "skulk-vllm-runtime-ready\n",
                "no GPU" if gpu_fails else "",
            )
        return subprocess.CompletedProcess(arguments, 0, "", "")

    return run


def test_untested_hardware_uses_runtime_abi_instead_of_qualification() -> None:
    facts = make_facts(gpus=(NVIDIA_A40,))
    recipe = runtime.select_vllm_recipe(facts)
    assert recipe is not None and recipe.variant == "cuda"
    # No tested GPU name or hardware class participates in eligibility.
    assert runtime.installable_vllm_builds(facts, offline=False) == {
        "vllm-cuda": recipe.build_identity
    }
    assert (
        runtime.select_vllm_recipe(make_facts(platform="darwin", gpus=(NVIDIA_A40,)))
        is None
    )


def test_rocm_matches_published_libc_and_cpu_abi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        runtime, "platform_tags", lambda: iter(("manylinux_2_28_x86_64",))
    )
    assert runtime.select_vllm_recipe(make_facts(gpus=(AMD_STRIX,))) is None
    monkeypatch.setattr(
        runtime, "platform_tags", lambda: iter(("manylinux_2_39_x86_64",))
    )
    recipe = runtime.select_vllm_recipe(make_facts(gpus=(AMD_STRIX,)))
    assert recipe is not None and recipe.variant == "rocm"


@pytest.mark.parametrize(
    "environment",
    [
        {"SKULK_NO_ENGINE_AUTOPROVISION": "1"},
        {"SKULK_NODE_PARTICIPATION": "management"},
        {VLLM_BIN_ENV: "/operator/missing"},
    ],
)
def test_policy_and_explicit_overrides_prevent_installation(
    environment: dict[str, str],
) -> None:
    facts = make_facts(gpus=(NVIDIA_A40,))
    assert (
        runtime.installable_vllm_builds(facts, offline=False, environ=environment) == {}
    )
    assert runtime.rehydrate_vllm(facts, environ=environment) is None


def test_offline_or_broken_explicit_binary_never_advertises_download() -> None:
    facts = make_facts(gpus=(NVIDIA_A40,))
    assert runtime.installable_vllm_builds(facts, offline=True) == {}
    assert (
        runtime.installable_vllm_builds(
            make_facts(gpus=(NVIDIA_A40,), vllm_bin=bad_bin(VLLM_BIN_ENV)),
            offline=False,
        )
        == {}
    )


def test_first_use_publishes_only_verified_runtime_then_reuses_it_offline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe = _channel(monkeypatch)
    calls: list[list[str]] = []
    monkeypatch.setenv("UV_INDEX_URL", "https://untrusted.invalid/")
    monkeypatch.setenv("PIP_CONSTRAINT", "/untrusted-constraints")
    facts = make_facts(gpus=(NVIDIA_A40,))
    binary = runtime.install_vllm_on_demand(
        facts,
        offline=False,
        stop=threading.Event(),
        run=_run(calls),
        expected_build=recipe.build_identity,
    )
    root = runtime.managed_vllm_root(recipe)
    assert binary == root / "venv/bin/vllm"
    assert binary.read_text().startswith(f"#!{root}/venv/bin/python\n")
    assert subprocess.check_output([str(binary)], text=True).strip() == "ready"
    assert runtime.managed_vllm_build(str(binary)) == recipe.build_identity
    assert runtime.installed_managed_vllm("vllm-cuda", recipe.build_identity) == binary
    assert runtime.installed_managed_vllm("vllm-rocm") is None
    assert runtime.installed_managed_vllm("vllm-cuda", "different-build") is None
    monkeypatch.delenv(VLLM_BIN_ENV)
    before = len(calls)
    assert (
        runtime.install_vllm_on_demand(
            facts, offline=True, stop=threading.Event(), run=_run(calls)
        )
        == binary
    )
    assert len(calls) == before
    with pytest.raises(runtime.VllmInstallError, match="placement check"):
        runtime.install_vllm_on_demand(
            facts,
            offline=True,
            stop=threading.Event(),
            run=_run(calls),
            expected_build="different-recipe",
        )


@pytest.mark.parametrize("failure", ["checksum", "gpu", "cancel"])
def test_failed_install_is_absent_and_a_retry_recovers(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    recipe = _channel(monkeypatch, corrupt=failure == "checksum")
    facts = make_facts(gpus=(NVIDIA_A40,))
    stop = threading.Event()
    with pytest.raises((runtime.VllmInstallError, ComfyInstallCancelledError)):
        runtime.install_vllm_on_demand(
            facts,
            offline=False,
            stop=stop,
            run=_run(
                [],
                gpu_fails=failure == "gpu",
                stop=stop if failure == "cancel" else None,
            ),
        )
    assert not runtime.managed_vllm_root(recipe).exists()
    assert runtime.installed_managed_vllm() is None
    assert VLLM_BIN_ENV not in runtime.os.environ
    _channel(monkeypatch)
    binary = runtime.install_vllm_on_demand(
        facts, offline=False, stop=threading.Event(), run=_run([])
    )
    assert binary.is_file()


def test_stamped_build_mismatch_cannot_install_a_newer_recipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _channel(monkeypatch)
    with pytest.raises(runtime.VllmInstallError, match="placement check"):
        runtime.install_vllm_on_demand(
            make_facts(gpus=(NVIDIA_A40,)),
            offline=False,
            stop=threading.Event(),
            run=_run([]),
            expected_build="vllm@other",
        )


def test_partial_or_changed_receipt_is_never_adopted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _channel(monkeypatch)
    facts = make_facts(gpus=(NVIDIA_A40,))
    binary = runtime.install_vllm_on_demand(
        facts, offline=False, stop=threading.Event(), run=_run([])
    )
    monkeypatch.delenv(VLLM_BIN_ENV)
    (binary.parent.parent.parent / "provisioned.json").write_text(
        json.dumps({"build": "old"})
    )
    assert runtime.rehydrate_vllm(facts) is None


def test_installable_and_installed_inventory_are_distinct() -> None:
    recipe = runtime.PUBLISHED_RECIPES[0]
    facts = make_facts(gpus=(NVIDIA_A40,)).model_copy(
        update={"vllm_installable_builds": {"vllm-cuda": recipe.build_identity}}
    )
    derived = derive_node_backends(facts)
    assert {"vllm", "vllm-cuda"} <= derived.on_demand_backends
    assert "vllm-cuda" not in derived.installed_backends
    assert (
        engine_build_inventory(derived.backends, facts)["vllm-cuda"]
        == recipe.build_identity
    )


async def test_resource_publication_separates_installable_and_installed_builds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The master receives the recipe identity before the package exists locally."""
    import skulk.facts as facts_module
    from skulk.shared.types.profiling import NodeResources

    recipe = runtime.PUBLISHED_RECIPES[0]
    facts = make_facts(gpus=(NVIDIA_A40,)).model_copy(
        update={"vllm_installable_builds": {"vllm-cuda": recipe.build_identity}}
    )
    monkeypatch.setattr(facts_module, "current_node_facts", lambda: facts)
    monkeypatch.setattr(
        facts_module, "current_backend_derivation", lambda: derive_node_backends(facts)
    )
    resources = await NodeResources.gather()
    assert resources.installable_engine_builds == {"vllm-cuda": recipe.build_identity}
    assert resources.engine_builds == {}
    assert "vllm-cuda" in resources.on_demand_backends
