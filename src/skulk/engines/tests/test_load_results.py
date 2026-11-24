"""A node's record of how the last model load on each engine package build ended."""

from __future__ import annotations

import pytest

from skulk.engines.load_results import (
    EngineLoadResults,
    LoadObservation,
    observe_load,
    with_load_results,
)
from skulk.shared.backends import engine_load_refusal
from skulk.shared.types.engine_packages import (
    MAX_NODE_ENGINE_PACKAGES,
    MAX_REPORTED_ENGINE_BUILDS,
    NodeEngineLoadResult,
    NodeEnginePackage,
)
from skulk.shared.types.worker.runners import (
    RunnerConnected,
    RunnerFailed,
    RunnerIdle,
    RunnerLoaded,
    RunnerLoading,
    RunnerReady,
    RunnerRunning,
    RunnerStatus,
    RunnerWarmingUp,
)

TAG = "tensorfold-rocm"
BUILD_A = "a" * 64
BUILD_B = "b" * 64


def _result(
    build: str = BUILD_A, *, at: int = 1000, outcome: str = "loaded"
) -> NodeEngineLoadResult:
    return NodeEngineLoadResult.model_validate(
        {"build": build, "outcome": outcome, "modelId": "org/model", "atUnixMs": at}
    )


def _package(**changes: object) -> NodeEnginePackage:
    return NodeEnginePackage(
        engine_id="foxlight.tensorfold",
        lane="rocm",
        backend_tag=TAG,
        status="active",
        build=BUILD_A,
        installed_builds=(BUILD_A,),
    ).model_copy(update=changes)


@pytest.mark.parametrize("previous", [RunnerLoading(), RunnerLoaded(), RunnerWarmingUp()])
def test_ready_after_a_loading_status_is_a_load(previous: RunnerStatus) -> None:
    assert observe_load(previous, RunnerReady(), now_unix_ms=7) == LoadObservation(
        "loaded", None, 7
    )


def test_a_typed_refusal_while_loading_keeps_only_its_code() -> None:
    """The engine's own words can name local paths and never leave the node."""
    refused = RunnerFailed(
        error_message=engine_load_refusal("load_refused", "cannot map /private/w.bin")
    )

    assert observe_load(RunnerLoading(), refused, now_unix_ms=7) == LoadObservation(
        "refused", "load_refused", 7
    )


def test_any_other_end_while_loading_is_a_failed_load() -> None:
    crashed = RunnerFailed(error_message="Terminated (signal=9)")

    assert observe_load(RunnerLoading(), crashed, now_unix_ms=7) == LoadObservation(
        "failed", None, 7
    )


@pytest.mark.parametrize(
    ("previous", "new"),
    [
        # A runner that never began to load said nothing about the build.
        (RunnerIdle(), RunnerFailed(error_message="could not start")),
        (RunnerConnected(), RunnerFailed(error_message="could not start")),
        # A model that had loaded and later stopped serving is not a load result.
        (RunnerReady(), RunnerFailed(error_message="Terminated (signal=9)")),
        (RunnerRunning(), RunnerFailed(error_message="Terminated (signal=9)")),
        (RunnerRunning(), RunnerReady()),
        (RunnerLoading(), RunnerLoading(layers_loaded=1, total_layers=4)),
        (RunnerFailed(error_message="first"), RunnerFailed(error_message="second")),
    ],
)
def test_a_change_that_does_not_end_a_load_is_no_observation(
    previous: RunnerStatus, new: RunnerStatus
) -> None:
    assert observe_load(previous, new, now_unix_ms=7) is None


def test_the_record_keeps_the_newest_result_for_each_build() -> None:
    results = EngineLoadResults()
    results.record(TAG, _result(at=2000, outcome="refused"))
    results.record(TAG, _result(at=1000))  # an older reading arriving late
    results.record(TAG, _result(BUILD_B, at=3000))
    results.record(TAG, _result(BUILD_B, at=3000))  # the same reading again

    assert results.for_tag(TAG) == (
        _result(BUILD_B, at=3000),
        _result(at=2000, outcome="refused"),
    )
    assert results.for_tag("tensorfold-cuda") == ()


def test_the_record_is_bounded_and_the_oldest_result_leaves_first() -> None:
    results = EngineLoadResults()
    builds = [f"{index:064x}" for index in range(MAX_REPORTED_ENGINE_BUILDS + 3)]
    for index, build in enumerate(builds):
        results.record(TAG, _result(build, at=index))

    held = results.for_tag(TAG)
    assert len(held) == MAX_REPORTED_ENGINE_BUILDS
    assert [entry.build for entry in held] == list(reversed(builds[3:]))

    for index in range(MAX_NODE_ENGINE_PACKAGES + 5):
        results.record(f"engine{index}-cpu", _result())
    assert results.for_tag(f"engine{MAX_NODE_ENGINE_PACKAGES + 4}-cpu") == ()
    assert results.for_tag(TAG) == held


def test_a_node_that_loaded_nothing_reports_what_it_did_before() -> None:
    packages = (_package(), _package(lane="cuda", backend_tag="tensorfold-cuda"))

    reported = with_load_results(packages, EngineLoadResults())

    assert reported == packages
    assert all(after is before for after, before in zip(reported, packages, strict=True))


def test_results_join_the_lane_they_belong_to_and_only_builds_it_still_names() -> None:
    results = EngineLoadResults()
    results.record(TAG, _result(at=2000, outcome="failed"))
    results.record(TAG, _result(BUILD_B, at=3000))  # a build no longer on disk
    results.record("tensorfold-cuda", _result(at=4000))
    held = _package()
    offered = _package(status="offered", build=BUILD_B, installed_builds=())
    other_lane = _package(lane="vulkan", backend_tag="tensorfold-vulkan")

    reported = with_load_results((held, other_lane), results)
    assert reported[0].load_results == (_result(at=2000, outcome="failed"),)
    assert reported[1] is other_lane

    # An offered build is named by its lane, so a result from before it was
    # removed and offered again still describes it.
    assert with_load_results((offered,), results)[0].load_results == (
        _result(BUILD_B, at=3000),
    )


def test_a_load_result_travels_in_the_package_report() -> None:
    package = _package(load_results=(_result(at=2000, outcome="loaded"),))

    wire = package.model_dump(mode="json", by_alias=True)

    assert wire["loadResults"] == [
        {
            "build": BUILD_A,
            "outcome": "loaded",
            "code": None,
            "modelId": "org/model",
            "atUnixMs": 2000,
        }
    ]
    assert (
        NodeEnginePackage.model_validate_json(package.model_dump_json(by_alias=True))
        == package
    )
