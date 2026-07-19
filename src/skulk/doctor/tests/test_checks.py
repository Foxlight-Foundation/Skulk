# pyright: reportPrivateUsage=false
"""Doctor registry tests: verdicts, consequences, and crash containment."""

from collections.abc import Sequence
from pathlib import Path

import pytest

from skulk.doctor.checks import (
    REGISTRY,
    CheckResult,
    DoctorCheck,
    _check_capability_conflicts,
    _check_engine_available,
    run_checks,
)
from skulk.facts.testing import (
    NVIDIA_A40,
    NVIDIA_PRESENCE_ONLY,
    make_facts,
    ok_bin,
)
from skulk.shared.types.node_facts import NodeFacts


def test_engine_available_ok_on_darwin() -> None:
    results = _check_engine_available(make_facts(platform="darwin"))
    assert [r.verdict for r in results] == ["ok"]
    assert "mlx" in results[0].detail


def test_engine_available_fails_on_bare_linux(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Hermetic: the dormant-engine lookup must not see a real managed install
    # or wheel on the developer machine.
    monkeypatch.setattr(
        "skulk.provisioning.llama_server.SKULK_ENGINES_DIR", tmp_path
    )

    def _no_wheel(vendor: str, facts: NodeFacts) -> tuple[Path, Path | None] | None:
        return None

    monkeypatch.setattr(
        "skulk.provisioning.llama_server.wheel_llama_server", _no_wheel
    )
    results = _check_engine_available(make_facts())
    assert [r.verdict for r in results] == ["fail"]
    assert "management" in results[0].consequence
    assert results[0].remediation != ""


def test_engine_available_ok_with_dormant_wheel(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The #628 shape: engine wheel installed, no override exported yet. Plain
    # doctor must report the engine that startup will wire, not FAIL.
    shim = tmp_path / "llama-server-cuda"

    def _dormant(facts: NodeFacts) -> Path:
        return shim

    monkeypatch.setattr("skulk.provisioning.dormant_llama_server", _dormant)

    def _probe_ok(binary: str) -> object:
        from skulk.shared.types.node_facts import LlamaServerDeviceProbe

        return LlamaServerDeviceProbe(outcome="devices", computes=("cuda",))

    monkeypatch.setattr(
        "skulk.facts.probe.probe_llama_server_devices", _probe_ok
    )
    results = _check_engine_available(make_facts(gpus=(NVIDIA_A40,)))
    assert [r.verdict for r in results] == ["ok"]
    assert str(shim) in results[0].detail
    assert "startup" in results[0].detail


def test_engine_available_fails_when_dormant_engine_is_broken(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A shim whose runtime probe fails (dead Vulkan ICD, missing CUDA loader)
    # would be disabled at startup; doctor must not claim it as available
    # (PR #634 review). The nonexistent tmp path makes the real probe fail
    # with a launch error, exercising the same outcome as a broken loader.
    shim = tmp_path / "llama-server-vulkan"

    def _dormant(facts: NodeFacts) -> Path:
        return shim

    monkeypatch.setattr("skulk.provisioning.dormant_llama_server", _dormant)
    results = _check_engine_available(make_facts(gpus=(NVIDIA_A40,)))
    assert [r.verdict for r in results] == ["fail"]
    assert str(shim) in results[0].detail
    assert "would be disabled at startup" in results[0].detail


def test_engine_available_flags_cpu_only_dormant_on_gpu_node(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A CPU-only build on a GPU node: derivation ADVERTISES llama_server-cpu
    # with an error-level gpu_serving_disabled conflict (#609), so the engine
    # is available but the doctor detail must surface the conflict, because
    # the capability-conflicts check reads unwired facts and cannot see it
    # before startup (PR #634 review round 2).
    shim = tmp_path / "llama-server-cpu"

    def _dormant(facts: NodeFacts) -> Path:
        return shim

    monkeypatch.setattr("skulk.provisioning.dormant_llama_server", _dormant)

    def _probe_cpu_only(binary: str) -> object:
        from skulk.shared.types.node_facts import LlamaServerDeviceProbe

        return LlamaServerDeviceProbe(outcome="devices", computes=())

    monkeypatch.setattr(
        "skulk.facts.probe.probe_llama_server_devices", _probe_cpu_only
    )
    results = _check_engine_available(make_facts(gpus=(NVIDIA_A40,)))
    assert [r.verdict for r in results] == ["ok"]
    assert "startup will flag" in results[0].detail
    assert "fraction of hardware speed" in results[0].detail


def test_engine_available_ok_with_served_binary() -> None:
    results = _check_engine_available(
        make_facts(llama_server_bin=ok_bin("SKULK_LLAMA_SERVER_BIN"))
    )
    assert [r.verdict for r in results] == ["ok"]


def test_capability_conflicts_ok_when_none() -> None:
    results = _check_capability_conflicts(make_facts(platform="darwin"))
    assert [r.verdict for r in results] == ["ok"]


def test_capability_conflicts_fail_for_error_codes() -> None:
    # GPU visible, served engine resolves cpu-only: the #609 shape must be a
    # FAIL verdict with the conflict's own remediation.
    facts = make_facts(
        gpus=(NVIDIA_A40,),
        llama_server_bin=ok_bin("SKULK_LLAMA_SERVER_BIN"),
        declared_llama_server="cpu",
    )
    results = _check_capability_conflicts(facts)
    assert [r.verdict for r in results] == ["fail"]
    assert "gpu_serving_disabled" in results[0].title


def test_capability_conflicts_degraded_detection_is_fixable() -> None:
    # nvidia present without pynvml: degraded, and --fix owns the install.
    facts = make_facts(
        gpus=(NVIDIA_PRESENCE_ONLY,),
        llama_server_bin=ok_bin("SKULK_LLAMA_SERVER_BIN"),
        declared_llama_server="cuda",
    )
    results = _check_capability_conflicts(facts)
    degraded = [r for r in results if r.verdict == "degraded"]
    assert any(r.fix_available for r in degraded)


def test_run_checks_contains_crashing_check(monkeypatch: object) -> None:
    # A crashing check degrades into a FAIL naming itself; the audit finishes.
    def _boom(facts: NodeFacts) -> Sequence[CheckResult]:
        raise RuntimeError("synthetic crash")

    import skulk.doctor.checks as checks_module

    broken = DoctorCheck(
        check_id="broken-check",
        title="Broken check",
        docs="synthetic",
        run=_boom,
    )
    original = checks_module.REGISTRY
    checks_module.REGISTRY = (*original, broken)
    try:
        results = run_checks(make_facts(platform="darwin"))
    finally:
        checks_module.REGISTRY = original
    crash = [r for r in results if r.check_id == "broken-check"]
    assert len(crash) == 1
    assert crash[0].verdict == "fail"
    assert "synthetic crash" in crash[0].detail


def test_registry_docs_are_nonempty() -> None:
    # The docs page is generated from the registry; every check must describe
    # itself well enough to stand alone there.
    for check in REGISTRY:
        assert check.docs.strip(), check.check_id
        assert check.title.strip(), check.check_id


def test_disabled_vllm_binary_does_not_count_as_available() -> None:
    # A vllm CLI on a GPU-less node derives no tags and serves nothing, so
    # the availability check must FAIL, keeping --fix provisioning eligible
    # (PR #615 review).
    from skulk.facts.testing import ok_bin as _ok_bin

    results = _check_engine_available(make_facts(vllm_bin=_ok_bin("SKULK_VLLM_BIN")))
    assert [r.verdict for r in results] == ["fail"]


def test_fix_not_promised_with_invalid_override() -> None:
    # An invalid SKULK_LLAMA_SERVER_BIN is its own conflict; --fix will not
    # provision over it, so the FAIL must not advertise fix_available.
    from skulk.facts.testing import bad_bin as _bad_bin

    results = _check_engine_available(
        make_facts(llama_server_bin=_bad_bin("SKULK_LLAMA_SERVER_BIN"))
    )
    assert [r.verdict for r in results] == ["fail"]
    assert results[0].fix_available is False


def test_models_storage_fails_when_path_is_a_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A file squatting on the models path must FAIL, not read as a healthy
    # existing directory (PR #615 review).
    import skulk.doctor.checks as checks_module

    squatter = tmp_path / "models"
    squatter.write_text("not a directory")
    monkeypatch.setattr(
        "skulk.shared.constants.SKULK_MODELS_DIR", squatter
    )
    results = checks_module._check_models_storage(make_facts(platform="darwin"))
    assert [r.verdict for r in results] == ["fail"]
    assert "not a directory" in results[0].detail


def test_mistyped_participation_still_requires_an_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A typo'd participation value normalizes to full at runtime, so the
    # doctor must judge the same effective role instead of skipping the
    # engine check (PR #615 review).
    monkeypatch.setenv("SKULK_NODE_PARTICIPATION", "managment")
    results = _check_engine_available(make_facts())
    assert [r.verdict for r in results] == ["fail"]


def test_declared_management_needs_no_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SKULK_NODE_PARTICIPATION", "management")
    results = _check_engine_available(make_facts())
    assert [r.verdict for r in results] == ["ok"]
