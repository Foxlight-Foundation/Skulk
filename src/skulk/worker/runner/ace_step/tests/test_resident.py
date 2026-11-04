"""Real synthetic CLI control checks; no inference or musical qualification."""

from __future__ import annotations

import sys
import threading
import time
import venv
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import psutil
import pytest

from skulk.shared.types.common import CommandId
from skulk.shared.types.native_music import NativeMusicRequest, NativeMusicServeConfig
from skulk.utils.runtime_files import RuntimeLock
from skulk.worker.runner.ace_step import resident as resident_module
from skulk.worker.runner.ace_step.resident import (
    NativeMusicProcessError,
    ResidentMusic,
    spawn_music_resident,
)
from skulk.worker.runner.ace_step.result_custody import (
    music_result_directory,
    prepare_result_directory,
    reclaim_result_directory,
)
from skulk.worker.runner.ace_step.tests.test_workspace import weights
from skulk.worker.runner.ace_step.workspace import NativeMusicWorkspace


@pytest.fixture(scope="module")
def interpreter(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Install a stdlib-only synthetic CLI in a private disposable interpreter."""
    root = tmp_path_factory.mktemp("native-music-resident")
    venv.EnvBuilder(with_pip=False, system_site_packages=True, symlinks=True).create(
        root
    )
    package = (
        root
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages/skulk_ace_step"
    )
    package.mkdir()
    (package / "__init__.py").touch()
    (package / "__main__.py").write_text("""
import argparse,hashlib,json,os,signal,subprocess,sys,threading,time,wave
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--serve',type=Path);parser.add_argument('--parent-pid',type=int)
args=parser.parse_args();cfg=json.loads(args.serve.read_text());root=args.serve.parent
mode=Path(cfg['weights']['base']['directory']).name
(root/'native.pid').write_text(str(os.getpid()))
def watch():
    while os.getppid()==args.parent_pid: time.sleep(.05)
    os.killpg(os.getpgrp(),signal.SIGKILL)
threading.Thread(target=watch,daemon=True).start()
def emit(record):print(json.dumps(record),flush=True)
def loading_copy():
    path=Path(cfg['directory'])/'loader/checkpoints/acestep-v15-base'
    path.mkdir(parents=True,mode=0o700);(path/'config.json').write_bytes(b'partial')
loading_copy()
if mode=='startup-slow':
    while True:time.sleep(.01)
if mode=='malformed':
    print('not-json',flush=True)
    while True:time.sleep(.01)
if mode=='oversized':
    print('x'*(256*1024+1),flush=True)
    while True:time.sleep(.01)
if mode=='premature':emit(dict(kind='progress',request_id='take',stage='generating'))
emit(dict(kind='progress',request_id=None,stage='loading'))
execution=dict(device=cfg['device'],precision=cfg['precision'],text_encoder_device=cfg['device'],text_encoder_precision=cfg['precision'],autoencoder_device=cfg['device'],autoencoder_precision='float16' if cfg['device']=='mps' else cfg['precision'])
ready=dict(protocol_version=1,model_identity='f'*64 if mode=='model' else cfg['model_identity'],payload_sha256='f'*64 if mode=='payload' else 'c'*64,execution=execution,load_seconds=.1)
if mode=='dtype':ready['execution']['autoencoder_precision']='float16'
emit(dict(kind='ready',ready=ready))
if mode=='idle-crash':
    def crash():
        while not (root/'crash-now').exists():time.sleep(.01)
        os._exit(9)
    threading.Thread(target=crash,daemon=True).start()
if mode=='duplicate-ready':emit(dict(kind='ready',ready=ready))
if mode=='no-read':
    while True:time.sleep(.01)
for line in sys.stdin.buffer:
    envelope=json.loads(line);request=envelope['request'];rid=request['request_id'];directory=Path(request['output_directory'])
    (root/'request-count').write_text(str(envelope['sequence']))
    active=(root/'mode').read_text() if (root/'mode').exists() else mode
    if active=='slow':
        (directory/'result.wav').write_bytes(b'partial')
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        child=subprocess.Popen([sys.executable,'-c',"import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);print('ready',flush=True);time.sleep(120)"],stdout=subprocess.PIPE)
        child.stdout.readline();child.stdout.close()
        (root/'child.pid').write_text(str(child.pid))
        while True:time.sleep(.01)
    for _ in range(2000 if active=='progress' else 1):emit(dict(kind='progress',request_id=rid,stage='generating'))
    if active=='progress':sys.stderr.write('diagnostic'*10000);sys.stderr.flush()
    path=directory/'result.wav';frames=request['target_frames']
    with wave.open(str(path),'wb') as wav:
        wav.setparams((2,3,48000,frames,'NONE','not compressed'));wav.writeframes(bytes(frames*6))
    data=path.read_bytes()
    output=dict(content_type='audio/wav',sha256=hashlib.sha256(data).hexdigest(),size_bytes=len(data),frames=frames,sample_rate=48000,channels=2,bit_depth=24,peak=.5 if active=='peak' else 0.,gain=1.)
    result=dict(request_id='foreign' if active=='foreign' else rid,model_identity=cfg['model_identity'],payload_sha256='c'*64,operation=request['operation'],seed=request['seed'],source_sha256=None,conditioning_sha256=None,conditioning_gain=None,conditioning_padding_frames=0,padding_frames_removed=0,execution=execution,output=output,generation_seconds=.1)
    record=dict(result)
    if active=='manifest':record['seed']+=1
    (directory/'result.json').write_text(json.dumps(record))
    if active=='missing':path.unlink()
    if active=='foreign-file':(directory/'operator-note').write_bytes(b'must survive')
    emit(dict(kind='completed',result=result))
""")
    return root / "bin/python"


def resident(
    root: Path,
    interpreter: Path,
    *,
    mode: str = "normal",
    resource_guard: Callable[[int], str | None] | None = None,
    source_fence: int | None = None,
) -> ResidentMusic:
    """Prepare real private inodes around tiny metadata, never actual model weights."""
    selected = weights()
    selected = selected.model_copy(
        update={
            "base": selected.base.model_copy(update={"directory": str(root / mode)})
        }
    )
    owner = NativeMusicWorkspace(root, selected)
    owner.open()
    configuration = NativeMusicServeConfig(
        model_identity="b" * 64,
        weights=selected,
        directory=str(owner.directory),
        directory_identity=owner.directory_identity,
        requests_identity=owner.requests_identity,
        loader_identity=owner.loader_identity,
        device="cpu",
        precision="float32",
    )
    return ResidentMusic(
        interpreter,
        owner,
        configuration,
        payload_sha256="c" * 64,
        spawn=partial(spawn_music_resident, source_fence=source_fence),
        resource_guard=resource_guard,
    )


def request(
    owner: ResidentMusic, identity: str = "take", *, large: bool = False
) -> NativeMusicRequest:
    """Journal one exact empty result before creating its private request."""
    directory = music_result_directory(owner.workspace.root, CommandId(identity))
    observed = prepare_result_directory(
        directory, owner.configuration.requests_identity
    )
    return NativeMusicRequest(
        request_id=identity,
        operation="generate",
        prompt="x" * 8000 if large else "Warm folk",
        lyrics="x" * 32000 if large else "[Instrumental]",
        seed=12025,
        target_frames=48000,
        output_directory=str(directory),
        output_directory_identity=observed,
    )


def wait_file(path: Path) -> None:
    """Bound synthetic-child observation without changing any production deadline."""
    deadline = time.monotonic() + 5
    while not path.exists():
        if time.monotonic() >= deadline:
            raise AssertionError(
                "synthetic native process did not reach its checkpoint"
            )
        time.sleep(0.01)


def process_alive(pid: int) -> bool:
    """Treat observed zombies as exited, while confirming live descendants are gone."""
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def test_one_loaded_model_serves_two_contiguous_calls_and_retains_outputs(
    tmp_path: Path, interpreter: Path
) -> None:
    """Output custody outlives model close; successful calls do not reload it."""
    owner = resident(tmp_path, interpreter)
    try:
        ready = owner.start(is_cancelled=lambda: False)
        pid = (tmp_path / "native.pid").read_text()
        results: list[Path] = []
        for name in ("take-one", "take-two"):
            call = request(owner, name)
            result = owner.run(
                call, progress=lambda _: None, is_cancelled=lambda: False
            )
            assert (
                result is not None
                and result.request_id == name
                and result.seed == 12025
            )
            assert result.execution == ready.execution
            results.append(Path(call.output_directory) / "result.wav")
        assert (tmp_path / "request-count").read_text() == "2"
        assert (tmp_path / "native.pid").read_text() == pid
        assert owner.is_ready()
        owner.close()
        assert all(path.exists() for path in results)
        assert not process_alive(int(pid))
        assert not tuple((tmp_path / "workspace/loader").iterdir())
    finally:
        owner.close()


@pytest.mark.parametrize(
    "mode", ["malformed", "oversized", "premature", "payload", "model", "dtype"]
)
def test_bad_native_readiness_drains_model_and_loading(
    tmp_path: Path, interpreter: Path, mode: str
) -> None:
    """A package/process observation cannot silently replace exact loaded readiness."""
    owner = resident(tmp_path, interpreter, mode=mode)
    try:
        with pytest.raises(NativeMusicProcessError):
            owner.start(is_cancelled=lambda: False, timeout_seconds=5)
        assert not owner.is_ready()
        assert not process_alive(int((tmp_path / "native.pid").read_text()))
        assert not tuple((tmp_path / "workspace/loader").iterdir())
    finally:
        owner.close()


@pytest.mark.parametrize("mode", ["foreign", "manifest", "peak", "missing"])
def test_invalid_completion_retires_and_reclaims_only_the_active_call(
    tmp_path: Path, interpreter: Path, mode: str
) -> None:
    """Native completion alone never becomes a successful result-return receipt."""
    owner = resident(tmp_path, interpreter, mode=mode)
    try:
        owner.start(is_cancelled=lambda: False)
        call = request(owner)
        with pytest.raises((NativeMusicProcessError, ValueError)):
            owner.run(call, progress=lambda _: None, is_cancelled=lambda: False)
        assert not owner.is_ready()
        assert not Path(call.output_directory).exists()
        assert not process_alive(int((tmp_path / "native.pid").read_text()))
    finally:
        owner.close()


def test_progress_flood_keeps_terminal_and_diagnostics_bounded(
    tmp_path: Path, interpreter: Path
) -> None:
    """Coalesced observation cannot consume the non-dropping terminal capacity."""
    owner = resident(tmp_path, interpreter, mode="progress")
    progress: list[str] = []
    try:
        owner.start(is_cancelled=lambda: False)
        result = owner.run(
            request(owner),
            progress=lambda record: progress.append(record.stage),
            is_cancelled=lambda: False,
        )
        assert result is not None and progress
        assert len(owner.diagnostic_tail.encode()) <= 4096
        assert owner.is_ready()
    finally:
        owner.close()


def test_cancel_drains_descendants_and_keeps_an_earlier_completed_result(
    tmp_path: Path, interpreter: Path
) -> None:
    """Cancelling new inference cannot erase audio still awaiting prior delivery."""
    owner = resident(tmp_path, interpreter)
    stopped = threading.Event()
    try:
        owner.start(is_cancelled=lambda: False)
        original = request(owner, "completed-take")
        assert (
            owner.run(original, progress=lambda _: None, is_cancelled=lambda: False)
            is not None
        )
        path = Path(original.output_directory) / "result.wav"
        data = path.read_bytes()
        (tmp_path / "mode").write_text("slow")
        pending = request(owner, "cancelled-take")
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                owner.run, pending, progress=lambda _: None, is_cancelled=stopped.is_set
            )
            wait_file(tmp_path / "child.pid")
            child = int((tmp_path / "child.pid").read_text())
            stopped.set()
            assert future.result(timeout=10) is None
        assert not process_alive(child)
        assert not process_alive(int((tmp_path / "native.pid").read_text()))
        assert path.read_bytes() == data
        assert not Path(pending.output_directory).exists()
        assert not tuple((tmp_path / "workspace/loader").iterdir())
    finally:
        stopped.set()
        owner.close()


def test_busy_call_refuses_without_cancelling_current_inference(
    tmp_path: Path, interpreter: Path
) -> None:
    """Serial admission protects the existing call and keeps the rejected copy owned."""
    owner = resident(tmp_path, interpreter, mode="slow")
    stopped = threading.Event()
    try:
        owner.start(is_cancelled=lambda: False)
        current, rejected = request(owner, "current"), request(owner, "rejected")
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                owner.run, current, progress=lambda _: None, is_cancelled=stopped.is_set
            )
            wait_file(tmp_path / "child.pid")
            with pytest.raises(NativeMusicProcessError, match="already owns"):
                owner.run(rejected, progress=lambda _: None, is_cancelled=lambda: False)
            assert owner.is_ready() and not future.done()
            assert (tmp_path / "request-count").read_text() == "1"
            stopped.set()
            assert future.result(timeout=10) is None
        assert Path(rejected.output_directory).exists()
        assert reclaim_result_directory(Path(rejected.output_directory))
    finally:
        stopped.set()
        owner.close()


def test_dead_idle_model_withdraws_and_restart_keeps_request_output(
    tmp_path: Path, interpreter: Path
) -> None:
    """Liveness is observed again; a cached ready receipt cannot survive a crash."""
    owner = resident(tmp_path, interpreter, mode="idle-crash")
    replacement: ResidentMusic | None = None
    try:
        owner.start(is_cancelled=lambda: False)
        call = request(owner)
        assert (
            owner.run(call, progress=lambda _: None, is_cancelled=lambda: False)
            is not None
        )
        path = Path(call.output_directory) / "result.wav"
        data = path.read_bytes()
        (tmp_path / "crash-now").touch()
        deadline = time.monotonic() + 5
        while owner.is_ready():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        replacement = resident(tmp_path, interpreter)
        replacement.start(is_cancelled=lambda: False)
        assert replacement.is_ready() and path.read_bytes() == data
    finally:
        owner.close()
        if replacement is not None:
            replacement.close()


def test_partial_cleanup_failure_retains_fence_until_explicit_retry(
    tmp_path: Path, interpreter: Path
) -> None:
    """Foreign files survive cancellation and prevent false cleanup success."""
    owner = resident(tmp_path, interpreter, mode="foreign-file")
    try:
        owner.start(is_cancelled=lambda: False)
        call = request(owner)
        with pytest.raises(NativeMusicProcessError, match="cleanup remains reserved"):
            owner.run(call, progress=lambda _: None, is_cancelled=lambda: False)
        with pytest.raises(BlockingIOError):
            RuntimeLock(tmp_path, name="service.lock")
        extra = Path(call.output_directory) / "operator-note"
        assert extra.read_bytes() == b"must survive"
        extra.unlink()
        owner.close()
        lock = RuntimeLock(tmp_path, name="service.lock")
        lock.close()
        assert not Path(call.output_directory).exists()
    finally:
        owner.close()


def test_load_deadline_drains_known_loading_before_releasing_owner(
    tmp_path: Path, interpreter: Path
) -> None:
    """A stalled startup cannot leave a readable loader or falsely ready model."""
    owner = resident(tmp_path, interpreter, mode="startup-slow")
    try:
        with pytest.raises(TimeoutError):
            owner.start(is_cancelled=lambda: False, timeout_seconds=0.3)
        assert owner.ready is None
        assert not tuple((tmp_path / "workspace/loader").iterdir())
        assert not process_alive(int((tmp_path / "native.pid").read_text()))
    finally:
        owner.close()


def test_blocked_control_peer_does_not_block_parent_deadline(
    tmp_path: Path, interpreter: Path
) -> None:
    """An unread maximum-size request remains cancellable under its hard budget."""
    owner = resident(tmp_path, interpreter, mode="no-read")
    try:
        owner.start(is_cancelled=lambda: False)
        call = request(owner, large=True)
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            owner.run(
                call,
                progress=lambda _: None,
                is_cancelled=lambda: False,
                timeout_seconds=0.3,
            )
        assert time.monotonic() - started < 5
        assert not Path(call.output_directory).exists()
    finally:
        owner.close()


def test_resource_refusal_retires_actual_native_group(
    tmp_path: Path, interpreter: Path
) -> None:
    """Parent resource policy remains effective while native inference is blocked."""
    refused = threading.Event()
    owner = resident(
        tmp_path,
        interpreter,
        mode="slow",
        resource_guard=lambda _pid: "native memory budget exceeded"
        if refused.is_set()
        else None,
    )
    try:
        owner.start(is_cancelled=lambda: False)
        call = request(owner)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                owner.run, call, progress=lambda _: None, is_cancelled=lambda: False
            )
            wait_file(tmp_path / "child.pid")
            refused.set()
            with pytest.raises(NativeMusicProcessError, match="memory budget"):
                future.result(timeout=10)
        assert not process_alive(int((tmp_path / "child.pid").read_text()))
        assert not Path(call.output_directory).exists()
    finally:
        owner.close()


def test_native_process_inherits_source_reader_fence(
    tmp_path: Path, interpreter: Path
) -> None:
    """A native child retains the real attachment lock after its parent copy closes."""
    fence = RuntimeLock(tmp_path, name="attachment.lock")
    owner = resident(tmp_path, interpreter, source_fence=fence.descriptor)
    try:
        owner.start(is_cancelled=lambda: False)
        fence.close()
        with pytest.raises(BlockingIOError):
            RuntimeLock(tmp_path, name="attachment.lock")
        owner.close()
        acquired = RuntimeLock(tmp_path, name="attachment.lock")
        acquired.close()
    finally:
        fence.close()
        owner.close()


def test_result_needs_real_prewrite_receipt_before_native_send(
    tmp_path: Path, interpreter: Path
) -> None:
    """Even Core-supplied inode facts cannot replace its durable ownership journal."""
    owner = resident(tmp_path, interpreter)
    try:
        owner.start(is_cancelled=lambda: False)
        call = request(owner)
        directory = Path(call.output_directory)
        directory.with_name(".take.owner.json").unlink()
        with pytest.raises(ValueError, match="ownership receipt"):
            owner.run(call, progress=lambda _: None, is_cancelled=lambda: False)
        assert not (tmp_path / "request-count").exists()
        assert directory.exists() and not tuple(directory.iterdir())
    finally:
        owner.close()


def test_failed_group_drain_keeps_service_fence_and_loader_for_retry(
    tmp_path: Path, interpreter: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unconfirmed native exit cannot release resources or remove its inputs."""
    owner = resident(tmp_path, interpreter)
    try:
        owner.start(is_cancelled=lambda: False)
        original = resident_module.terminate_process_group

        def refuse(
            _process: object, *, grace_seconds: float, confirm_members: bool
        ) -> None:
            assert grace_seconds == 1 and confirm_members
            raise TimeoutError("injected incomplete native group drain")

        monkeypatch.setattr(resident_module, "terminate_process_group", refuse)
        with pytest.raises(TimeoutError, match="group drain"):
            owner.close()
        assert process_alive(int((tmp_path / "native.pid").read_text()))
        assert (
            tmp_path / "workspace/loader/checkpoints/acestep-v15-base/config.json"
        ).read_bytes() == b"partial"
        with pytest.raises(BlockingIOError):
            RuntimeLock(tmp_path, name="service.lock")
        monkeypatch.setattr(resident_module, "terminate_process_group", original)
        owner.close()
        assert not process_alive(int((tmp_path / "native.pid").read_text()))
        assert not tuple((tmp_path / "workspace/loader").iterdir())
    finally:
        monkeypatch.undo()
        owner.close()


def test_repeated_readiness_withdraws_native_liveness(
    tmp_path: Path, interpreter: Path
) -> None:
    """A second ready frame cannot leave the first cached receipt advertised."""
    owner = resident(tmp_path, interpreter, mode="duplicate-ready")
    try:
        try:
            owner.start(is_cancelled=lambda: False)
        except NativeMusicProcessError:
            pass
        else:
            deadline = time.monotonic() + 5
            while owner.is_ready():
                assert time.monotonic() < deadline
                time.sleep(0.01)
        assert owner.ready is None
        assert not process_alive(int((tmp_path / "native.pid").read_text()))
    finally:
        owner.close()
