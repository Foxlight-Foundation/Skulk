"""Real subprocess evidence for bounded native completion and group cleanup."""

from __future__ import annotations

import io
import sys
import time
import venv
import wave
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

from skulk.shared.types.audio_separation import NativeSeparationJob, source_from_wav
from skulk.worker.runner.demucs import process as native_process
from skulk.worker.runner.demucs.process import (
    run_native_separation,
    separation_environment,
)


@pytest.fixture(scope="module")
def interpreter(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create an isolated test CLI without installing any inference dependency."""
    root = tmp_path_factory.mktemp("separation-native")
    venv.EnvBuilder(with_pip=False, system_site_packages=True, symlinks=True).create(
        root
    )
    site = (
        root
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    package = site / "skulk_demucs"
    package.mkdir()
    (package / "__init__.py").touch()
    (package / "__main__.py").write_text("""
import argparse, hashlib, json, os, signal, subprocess, sys, time, wave
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument("--job", type=Path)
parser.add_argument("--parent-pid")
args = parser.parse_args()
job = json.loads(args.job.read_text())
directory = Path(job["output_dir"])
mode = Path(job["weights_path"]).name
print(json.dumps({"stage":"loading","completed_frames":0}), flush=True)
if mode in ("slow", "crash", "malformed", "oversized"):
    (directory / "partial.wav").write_bytes(b"partial")
    if mode == "crash":
        sys.stderr.write("native load failed")
        sys.exit(9)
    if mode == "malformed":
        print("not JSON", flush=True)
    if mode == "oversized":
        (directory / "result.json").write_bytes(b"x" * 16385)
        sys.exit(0)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    child = subprocess.Popen([sys.executable,"-c","import time; time.sleep(120)"])
    Path(job["source_path"]).with_suffix(".pid").write_text(str(child.pid))
    while True: time.sleep(0.05)
stems=[]
frames=job["source"]["frames"]
rate=job["source"]["sample_rate"]
for name in ("drums","bass","other","vocals"):
    path=directory / (name+".wav")
    with wave.open(str(path), "wb") as wav:
        wav.setparams((2,3,rate,frames,"NONE","not compressed"))
        wav.writeframes(bytes(frames*6))
    data=path.read_bytes()
    stems.append(dict(name=name,content_type="audio/wav",size_bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),sample_rate=rate,frames=frames,channels=2,bit_depth=24,peak=0.0))
if mode == "missing": (directory / "vocals.wav").unlink()
if mode == "corrupt": (directory / "bass.wav").write_bytes(b"corrupt")
output=dict(source=job["source"],stems=stems,common_gain=1.0,pre_gain_peak=0.0)
result=dict(output=output,device=job["device"],load_seconds=0.0,separation_seconds=0.0,total_seconds=0.0,process_peak_rss_bytes=0,device_peak_allocated_bytes=None)
(directory / "result.json").write_text(json.dumps(result))
""")
    return root / "bin" / "python"


def _job(root: Path, mode: str) -> NativeSeparationJob:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as source:
        source.setparams((1, 2, 8000, 8000, "NONE", "not compressed"))
        source.writeframes(bytes(16000))
    path = root / "original.wav"
    path.write_bytes(buffer.getvalue())
    return NativeSeparationJob(
        weights_path=str(root / mode),
        source_path=str(path),
        output_dir=str(root / "output"),
        source=source_from_wav(buffer.getvalue()),
        device="cpu",
        threads=1,
    )


def test_complete_result_is_verified_and_original_is_unchanged(
    interpreter: Path, tmp_path: Path
) -> None:
    """Only the complete exact files survive a successful native subprocess."""
    job = _job(tmp_path, "complete")
    original = Path(job.source_path).read_bytes()
    result = run_native_separation(
        interpreter, job, progress=lambda _: None, is_cancelled=lambda: False
    )
    assert result is not None and len(result.output.stems) == 4
    assert Path(job.source_path).read_bytes() == original
    assert not (Path(job.output_dir) / "job.json").exists()


@pytest.mark.parametrize(
    "mode", ["missing", "corrupt", "crash", "malformed", "oversized"]
)
def test_partial_or_invalid_completion_removes_owned_outputs(
    interpreter: Path, tmp_path: Path, mode: str
) -> None:
    """No partial result survives a crash, protocol failure or digest mismatch."""
    job = _job(tmp_path, mode)
    with pytest.raises((ValueError, FileNotFoundError)):
        run_native_separation(
            interpreter, job, progress=lambda _: None, is_cancelled=lambda: False
        )
    assert not Path(job.output_dir).exists()
    assert Path(job.source_path).is_file()


@pytest.mark.parametrize("stop", ["cancel", "timeout", "pressure"])
def test_inference_stop_drains_descendants_and_removes_partial_files(
    interpreter: Path, tmp_path: Path, stop: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancellation/deadline/admission pressure stop even a TERM-resistant tree."""
    job = _job(tmp_path, "slow")
    child_record = Path(job.source_path).with_suffix(".pid")
    cancellation: Callable[[], bool] = (
        (lambda: child_record.exists()) if stop == "cancel" else (lambda: False)
    )
    resource_guard: Callable[[int], str | None] | None = (
        (lambda _: "resource pressure" if child_record.exists() else None)
        if stop == "pressure"
        else None
    )
    if stop == "timeout":
        started = time.monotonic()
        armed: float | None = None

        def inference_clock() -> float:
            nonlocal armed
            now = time.monotonic()
            if now - started >= 10:
                raise AssertionError(
                    "native test fixture did not reach its inference barrier"
                )
            if armed is None and child_record.exists():
                armed = now
            # This case proves descendant drain during inference. Keep startup
            # outside its half-second clock; a separate test covers expiry
            # before startup. Production's actual total deadline is unchanged.
            return started if armed is None else started + now - armed

        monkeypatch.setattr(
            native_process, "time", SimpleNamespace(monotonic=inference_clock)
        )
    if stop == "cancel":
        assert (
            run_native_separation(
                interpreter, job, progress=lambda _: None, is_cancelled=cancellation
            )
            is None
        )
    else:
        with pytest.raises(
            ValueError, match="deadline" if stop == "timeout" else "pressure"
        ):
            run_native_separation(
                interpreter,
                job,
                progress=lambda _: None,
                is_cancelled=cancellation,
                resource_guard=resource_guard,
                timeout_seconds=0.5 if stop == "timeout" else 30,
            )
    assert not Path(job.output_dir).exists()
    assert child_record.exists()
    import psutil

    try:
        child = psutil.Process(int(child_record.read_text()))
        assert child.status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        pass


def test_deadline_before_startup_drains_and_removes_partial_files(
    interpreter: Path, tmp_path: Path
) -> None:
    """An expired total budget does not require the child to reach inference."""
    job = _job(tmp_path, "slow")
    with pytest.raises(ValueError, match="deadline"):
        run_native_separation(
            interpreter,
            job,
            progress=lambda _: None,
            is_cancelled=lambda: False,
            timeout_seconds=0.0,
        )
    assert not Path(job.output_dir).exists()
    assert Path(job.source_path).is_file()


def test_runtime_environment_owns_offline_and_thread_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ambient Python paths, Hub tokens and tuning cannot change the native job."""
    monkeypatch.setenv("PYTHONPATH", "/untrusted")
    monkeypatch.setenv("HF_TOKEN", "not-a-real-token")
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    value = separation_environment(3)
    assert "PYTHONPATH" not in value and "HF_TOKEN" not in value
    assert (
        value["PYTORCH_ENABLE_MPS_FALLBACK"] == "0" and value["OMP_NUM_THREADS"] == "3"
    )
