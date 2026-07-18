# pyright: reportPrivateUsage=false
"""Doctor registry tests: verdicts, consequences, and crash containment."""

from collections.abc import Sequence

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


def test_engine_available_fails_on_bare_linux() -> None:
    results = _check_engine_available(make_facts())
    assert [r.verdict for r in results] == ["fail"]
    assert "management" in results[0].consequence
    assert results[0].remediation != ""


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
