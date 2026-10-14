"""Native control planning and process-group cancellation without model weights."""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import pytest

from skulk.api.types.api import VideoCapabilitySection
from skulk.shared.backends import ENGINE_UNAVAILABLE_FAILURE_MARKER
from skulk.shared.models.model_cards import ModelCard
from skulk.shared.tests.model_card_fixtures import load_fixture_card
from skulk.shared.types.common import ModelId
from skulk.shared.types.video import VideoGenerationTaskParams
from skulk.worker.runner.mlx_video import runner as runner_module
from skulk.worker.runner.mlx_video.plan import plan_native_job, selected_model_view
from skulk.worker.runner.mlx_video.process import NativeProgress, run_native_job


def _card() -> ModelCard:
    card = asyncio.run(load_fixture_card(ModelId("Lightricks/LTX-2.5-distilled-comfy-int8")))
    assert card is not None
    return card


def _params(**updates: object) -> VideoGenerationTaskParams:
    return VideoGenerationTaskParams.model_validate({"model": str(_card().model_id), "prompt": "A fox in a meadow", "seconds": 2, "size": "768x512", "seed": 4, **updates})


def test_native_plan_preserves_exact_requested_workload(tmp_path: Path) -> None:
    job, plan = plan_native_job(_params(), _card(), tmp_path / "model", tmp_path / "output", tmp_path / "input")
    assert (job.width, job.height, job.frame_count, job.seed) == (768, 512, 49, 4)
    assert job.pipeline == "distilled_two_stage" and job.base_steps == 8
    assert job.audio and plan.sample_rate == 48000


def test_native_distilled_honors_its_fixed_ancestral_sampler(tmp_path: Path) -> None:
    job, _ = plan_native_job(_params(sampler="euler_ancestral"), _card(), tmp_path, tmp_path, tmp_path)
    assert job.pipeline == "distilled_two_stage"


def test_native_refuses_ambiguous_comfy_artifact_layout(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="pack-root"):
        selected_model_view(_card(), tmp_path)


def test_missing_native_engine_fails_placement_without_restart_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(_build: str) -> None:
        return None
    monkeypatch.setattr(runner_module, "prepared_runtime", missing)
    with pytest.raises(RuntimeError, match=ENGINE_UNAVAILABLE_FAILURE_MARKER):
        runner_module.resolve_placed_runtime("mlx_video-metal", "expected-build")


@pytest.mark.parametrize("prerequisite", ["audio_output", "frame_grid_multiple"])
def test_metadata_does_not_offer_malformed_ltx_pipeline(prerequisite: str) -> None:
    card = _card()
    assert card.video is not None
    broken = card.model_copy(update={"video": card.video.model_copy(update={prerequisite: False if prerequisite == "audio_output" else 1})})
    metadata = VideoCapabilitySection.from_model_card(broken)
    assert metadata is not None
    assert metadata.pipelines == [] and metadata.decoders == []


def test_cancel_terminates_native_process_group(tmp_path: Path) -> None:
    interpreter = tmp_path / "interpreter"
    # A real isolated group with a child lets this test catch the orphaned
    # FFmpeg failure class, independent of Metal or model artifacts.
    interpreter.write_text(f"#!{sys.executable}\nimport os,time,json,subprocess\nfrom pathlib import Path\nchild=subprocess.Popen([{sys.executable!r},'-c','import time;time.sleep(60)'])\nPath({str(tmp_path / 'child.pid')!r}).write_text(str(child.pid))\nprint(json.dumps({{'stage':'sampling','step':1,'total_steps':8,'sampling_stage':1}}),flush=True)\ntime.sleep(60)\n")
    interpreter.chmod(0o755)
    job, _ = plan_native_job(_params(), _card(), tmp_path / "model", tmp_path / "output", tmp_path / "input")
    records: list[NativeProgress] = []
    started = time.monotonic()
    assert run_native_job(interpreter, job, progress=records.append, is_cancelled=lambda: bool(records)) is None
    assert time.monotonic() - started < 6
    child = int((tmp_path / "child.pid").read_text())
    # A terminated child may briefly remain a zombie until the OS reaps it.
    import psutil

    try:
        process = psutil.Process(child)
        assert process.status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        pass
    assert not (tmp_path / "output/job.json").exists()
