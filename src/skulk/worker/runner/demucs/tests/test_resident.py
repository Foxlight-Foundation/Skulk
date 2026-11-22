"""Real child-process checks for resident control, lifetime and output ownership."""

from __future__ import annotations

import io
import os
import subprocess
import sys
import time
import venv
import wave
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from skulk.shared.types.audio_separation import (
    NativeSeparationJob,
    NativeSeparationResult,
    NativeSeparationServeConfig,
    source_from_wav,
)
from skulk.utils.process_groups import observe_process_exit
from skulk.utils.runtime_files import RuntimeLock
from skulk.worker.runner.demucs import resident as resident_module
from skulk.worker.runner.demucs.resident import (
    ResidentSeparationError,
    ResidentSeparator,
)


@pytest.fixture(scope="module")
def interpreter(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a bounded synthetic private CLI, independent of all inference libraries."""
    root = tmp_path_factory.mktemp("resident-native")
    venv.EnvBuilder(with_pip=False, system_site_packages=True, symlinks=True).create(
        root
    )
    package = (
        root
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "skulk_demucs"
    )
    package.mkdir()
    (package / "__init__.py").touch()
    (package / "__main__.py").write_text("""
import argparse,hashlib,json,os,signal,subprocess,sys,threading,time,wave
from pathlib import Path
parser=argparse.ArgumentParser()
parser.add_argument('--serve',type=Path)
parser.add_argument('--parent-pid')
args=parser.parse_args()
def watch_parent():
    while os.getppid()==int(args.parent_pid): time.sleep(.1)
    os.killpg(os.getpgrp(),signal.SIGKILL)
threading.Thread(target=watch_parent,daemon=True).start()
cfg=json.loads(args.serve.read_text())
root=args.serve.parent
mode=Path(cfg['weights_path']).name
(root/'native.pid').write_text(str(os.getpid()))
def emit(value): print(json.dumps(value),flush=True)
if mode=='startup-slow':
    while True: time.sleep(.05)
if mode=='malformed':
    print('not-json',flush=True)
    while True: time.sleep(.05)
ready=dict(kind='ready',protocol=1,payload_sha256='d'*64,weights_sha256='a'*64,device=cfg['device'],threads=cfg['threads'],load_seconds=0.0,sample_rate=44100,audio_channels=2,stems=['drums','bass','other','vocals'])
if mode=='payload': ready['payload_sha256']='f'*64
if mode=='device': ready['device']='mps'
emit(ready)
if mode=='idle-crash':
    def crash():
        while not (root/'crash-now').exists(): time.sleep(.01)
        os._exit(9)
    threading.Thread(target=crash,daemon=True).start()
for line in sys.stdin.buffer:
    request=json.loads(line)
    job=request['job'];rid=request['request_id'];directory=Path(job['output_dir'])
    (root/'request-count').write_text(str(request['sequence']))
    if mode=='slow':
        (directory/'partial.wav').write_bytes(b'partial')
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        child=subprocess.Popen([sys.executable,'-c','import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(120)'])
        (root/'child.pid').write_text(str(child.pid))
        while True: time.sleep(.05)
    count=2000 if mode=='progress' else 1
    for _ in range(count):
        emit(dict(kind='progress',request_id=rid,progress=dict(stage='separating',completed_frames=0)))
    rate=16000 if mode=='rate' else 44100
    frames=(2*job['source']['frames']*rate+job['source']['sample_rate'])//(2*job['source']['sample_rate'])
    stems=[]
    for name in ('drums','bass','other','vocals'):
        path=directory/(name+'.wav')
        with wave.open(str(path),'wb') as wav:
            wav.setparams((2,3,rate,frames,'NONE','not compressed'));wav.writeframes(bytes(frames*6))
        data=path.read_bytes()
        stems.append(dict(name=name,content_type='audio/wav',size_bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),sample_rate=rate,frames=frames,channels=2,bit_depth=24,peak=.5 if mode=='peak' and name=='bass' else 0.0))
    if mode=='missing': (directory/'vocals.wav').unlink()
    output=dict(source=job['source'],stems=stems,common_gain=1.0,pre_gain_peak=.5 if mode=='peak' else 0.0)
    result=dict(output=output,device=cfg['device'],load_seconds=0.0,separation_seconds=0.0,total_seconds=0.0,process_peak_rss_bytes=0,device_peak_allocated_bytes=None)
    (directory/'result.json').write_text(json.dumps(result))
    emit(dict(kind='completed',request_id='f'*32 if mode=='foreign' else rid,result=result))
""")
    return root / "bin" / "python"


def owner(interpreter: Path, root: Path, mode: str) -> ResidentSeparator:
    """Construct an actual supervised CLI owner with exact expected policy."""
    return ResidentSeparator(
        interpreter,
        root / "owner",
        NativeSeparationServeConfig(
            weights_path=str(root / mode), device="cpu", threads=1
        ),
        payload_sha256="d" * 64,
        weights_sha256="a" * 64,
    )


def job(control: ResidentSeparator, number: int = 1) -> NativeSeparationJob:
    """Build a real original and a fresh command-private result directory."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as source:
        source.setparams((1, 2, 8000, 8000, "NONE", "not compressed"))
        source.writeframes(bytes(16000))
    original = control.root / "original.wav"
    original.write_bytes(buffer.getvalue())
    return NativeSeparationJob(
        weights_path=control.configuration.weights_path,
        source_path=str(original),
        output_dir=str(control.jobs / f"{number:032x}"),
        source=source_from_wav(buffer.getvalue()),
        device="cpu",
        threads=1,
    )


def wait_for(path: Path) -> None:
    """Wait for a real fixture barrier without sleeping past a bounded deadline."""
    deadline = time.monotonic() + 10
    while not path.exists():
        assert time.monotonic() < deadline, f"missing native test barrier:{path.name}"
        time.sleep(0.01)


def test_two_jobs_reuse_native_pid_and_keep_verified_outputs(
    interpreter: Path,
    tmp_path: Path,
) -> None:
    """Verified completion and serial admission reuse one resident native lifetime."""
    control = owner(interpreter, tmp_path, "complete")
    try:
        ready = control.start(is_cancelled=lambda: False, timeout_seconds=10)
        pid = (control.root / "native.pid").read_text()
        assert control.is_ready() and control.start(is_cancelled=lambda: False) == ready
        for number in (1, 2):
            request = job(control, number)
            result = control.run(
                request,
                progress=lambda _: None,
                is_cancelled=lambda: False,
                timeout_seconds=10,
            )
            assert result is not None and len(result.output.stems) == 4
            assert Path(request.output_dir, "vocals.wav").is_file()
        assert (control.root / "native.pid").read_text() == pid
        assert (control.root / "request-count").read_text() == "2"
    finally:
        control.close()
    assert all(
        (control.jobs / f"{number:032x}" / "vocals.wav").exists() for number in (1, 2)
    )
    RuntimeLock(control.root, name="service.lock").close()


@pytest.mark.parametrize("mode", ["payload", "device", "malformed", "startup-slow"])
def test_invalid_or_unresponsive_load_never_becomes_ready(
    interpreter: Path,
    tmp_path: Path,
    mode: str,
) -> None:
    """A wrong identity/lane, bad response or load timeout drains before release."""
    control = owner(interpreter, tmp_path, mode)
    with pytest.raises((ResidentSeparationError, TimeoutError)):
        control.start(
            is_cancelled=lambda: False,
            timeout_seconds=0.3 if mode == "startup-slow" else 10,
        )
    assert not control.is_ready()
    control.close()
    RuntimeLock(control.root, name="service.lock").close()


@pytest.mark.parametrize("mode", ["missing", "peak", "rate", "foreign"])
def test_invalid_output_stops_model_and_cleans_only_its_job(
    interpreter: Path,
    tmp_path: Path,
    mode: str,
) -> None:
    """Valid-looking terminals cannot bypass byte, native-rate or request binding."""
    control = owner(interpreter, tmp_path, mode)
    try:
        control.start(is_cancelled=lambda: False, timeout_seconds=10)
        request = job(control)
        with pytest.raises((ValueError, FileNotFoundError, ResidentSeparationError)):
            control.run(
                request,
                progress=lambda _: None,
                is_cancelled=lambda: False,
                timeout_seconds=10,
            )
        assert not Path(request.output_dir).exists()
        assert Path(request.source_path).is_file()
        assert not control.is_ready()
    finally:
        control.close()


def test_progress_flood_cannot_drop_completion(
    interpreter: Path, tmp_path: Path
) -> None:
    """Replaceable progress shares no finite queue with required terminal frames."""
    control = owner(interpreter, tmp_path, "progress")
    try:
        control.start(is_cancelled=lambda: False, timeout_seconds=10)
        result = control.run(
            job(control),
            progress=lambda _: time.sleep(0.001),
            is_cancelled=lambda: False,
            timeout_seconds=10,
        )
        assert result is not None
    finally:
        control.close()


def test_idle_crash_withdraws_readiness_and_releases_fence(
    interpreter: Path,
    tmp_path: Path,
) -> None:
    """A remembered ready receipt cannot survive independent native process death."""
    control = owner(interpreter, tmp_path, "idle-crash")
    control.start(is_cancelled=lambda: False, timeout_seconds=10)
    (control.root / "crash-now").touch()
    deadline = time.monotonic() + 5
    while control.is_ready():
        assert time.monotonic() < deadline
        time.sleep(0.01)
    control.close()
    RuntimeLock(control.root, name="service.lock").close()


def test_concurrent_close_cancels_active_job_and_drains_resistant_tree(
    interpreter: Path,
    tmp_path: Path,
) -> None:
    """Close requests cancellation without racing active file verification/cleanup."""
    control = owner(interpreter, tmp_path, "slow")
    control.start(is_cancelled=lambda: False, timeout_seconds=10)
    request = job(control)
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(
            control.run,
            request,
            progress=lambda _: None,
            is_cancelled=lambda: False,
            timeout_seconds=15,
        )
        try:
            wait_for(control.root / "child.pid")
            with pytest.raises(ResidentSeparationError, match="active"):
                control.run(
                    request, progress=lambda _: None, is_cancelled=lambda: False
                )
            control.close()
            assert result.result(timeout=5) is None
        finally:
            control.close()
    assert not Path(request.output_dir).exists()
    assert Path(request.source_path).is_file()
    RuntimeLock(control.root, name="service.lock").close()


def test_blocked_control_write_observes_total_deadline(
    interpreter: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A full pipe cannot hide the owner deadline inside a buffered write/flush."""
    control = owner(interpreter, tmp_path, "complete")
    control.start(is_cancelled=lambda: False, timeout_seconds=10)
    calls: list[int] = []

    def blocked(_: int, data: memoryview[bytes]) -> int:
        calls.append(len(data))
        raise BlockingIOError("injected full native control pipe")

    monkeypatch.setattr(
        resident_module,
        "os",
        SimpleNamespace(
            set_blocking=os.set_blocking,
            write=blocked,
        ),
    )
    request = job(control)
    with pytest.raises(TimeoutError):
        control.run(
            request,
            progress=lambda _: None,
            is_cancelled=lambda: False,
            timeout_seconds=0.1,
        )
    assert calls and not Path(request.output_dir).exists()
    control.close()


@pytest.mark.parametrize("budget", [0.0, -1.0, float("inf"), float("nan"), 21601.0])
def test_invalid_total_budget_has_no_native_side_effect(
    interpreter: Path, tmp_path: Path, budget: float
) -> None:
    """Invalid budgets cannot take a fence, load native code or interrupt ready work."""
    control = owner(interpreter, tmp_path, "complete")
    with pytest.raises(ValueError, match="budget"):
        control.start(is_cancelled=lambda: False, timeout_seconds=budget)
    assert not control.root.exists()
    try:
        control.start(is_cancelled=lambda: False, timeout_seconds=10)
        with pytest.raises(ValueError, match="budget"):
            control.run(
                job(control),
                progress=lambda _: None,
                is_cancelled=lambda: False,
                timeout_seconds=budget,
            )
        assert control.is_ready()
    finally:
        control.close()


def test_cancel_during_complete_file_verification_discards_unsettled_result(
    interpreter: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A native terminal cannot outrun cancellation during independent WAV scanning."""
    control = owner(interpreter, tmp_path, "complete")
    verify = resident_module.verify_native_result
    checking = False
    cancelled = False
    inspections = 0

    def cancellation() -> bool:
        nonlocal cancelled, inspections
        if checking:
            inspections += 1
            # First checkpoint enters the scan; the next one observes a cancel
            # before the complete four-file receipt can become settled output.
            if inspections == 2:
                cancelled = True
        return cancelled

    def inspect(
        directory: Path,
        request: NativeSeparationJob,
        *,
        checkpoint: Callable[[], None],
    ) -> NativeSeparationResult:
        nonlocal checking
        checking = True
        # The real verifier supplies inter-file checkpoints and independent byte
        # facts; only the cancellation source changes at its second checkpoint.
        return verify(directory, request, checkpoint=checkpoint)

    monkeypatch.setattr(resident_module, "verify_native_result", inspect)
    try:
        control.start(is_cancelled=lambda: False, timeout_seconds=10)
        request = job(control)
        result = control.run(
            request,
            progress=lambda _: None,
            is_cancelled=cancellation,
            timeout_seconds=10,
        )
        assert result is None and cancelled and inspections == 2
        assert not control.is_ready() and not Path(request.output_dir).exists()
        assert Path(request.source_path).is_file()
        RuntimeLock(control.root, name="service.lock").close()
    finally:
        control.close()


def test_incomplete_drain_retains_fence_and_partial_files_for_close_retry(
    interpreter: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No file deletion or ownership release follows an unconfirmed native drain."""
    control = owner(interpreter, tmp_path, "slow")
    control.start(is_cancelled=lambda: False, timeout_seconds=10)
    request = job(control)
    stop = resident_module.terminate_process_group
    attempts: list[int] = []

    def incomplete(
        process: subprocess.Popen[bytes],
        *,
        grace_seconds: float,
        confirm_members: bool,
    ) -> None:
        attempts.append(process.pid)
        if len(attempts) == 1:
            assert observe_process_exit(process) is None
            raise subprocess.TimeoutExpired(process.args, 1)
        stop(process, grace_seconds=grace_seconds, confirm_members=confirm_members)

    monkeypatch.setattr(resident_module, "terminate_process_group", incomplete)
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            control.run(
                request,
                progress=lambda _: None,
                is_cancelled=lambda: (control.root / "child.pid").exists(),
                timeout_seconds=15,
            )
        assert Path(request.output_dir, "partial.wav").exists()
        with pytest.raises(BlockingIOError):
            RuntimeLock(control.root, name="service.lock")
        control.close()
        assert not Path(request.output_dir).exists()
        RuntimeLock(control.root, name="service.lock").close()
    finally:
        control.close()
