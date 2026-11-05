"""Live native resource ownership, joined failure and actual group retirement."""

from __future__ import annotations

import io
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import check_support
import psutil
import skulk_ace_step
from skulk_ace_step.contracts import (
    NativeMusicFrame,
    NativeMusicMemoryBudget,
    NativeMusicMemoryCapacity,
    NativeMusicMemoryFrame,
    NativeMusicMemoryObservation,
    NativeMusicServeConfig,
)
from skulk_ace_step.resources import NativeMusicMemoryObserver


class NativeResourceTests(unittest.TestCase):
    """Observe real thread/process lifetimes with controlled allocation readings."""

    def setUp(self) -> None:
        """Create owned metadata only; no music model or GPU load is claimed."""
        self.temporary = tempfile.TemporaryDirectory(prefix="native-resource-")
        self.root = Path(self.temporary.name)
        self.configuration = check_support.configuration(self.root).model_copy(
            update={
                "memory_budget": NativeMusicMemoryBudget(
                    host_peak_bytes=100, device_peak_bytes=0, shared_peak_bytes=100
                )
            }
        )
        self.frames: list[NativeMusicFrame] = []
        self.retired = threading.Event()

    def tearDown(self) -> None:
        """Remove only owned fixture data after observers and children have drained."""
        self.temporary.cleanup()

    def sample(
        self, configuration: NativeMusicServeConfig, payload: str
    ) -> NativeMusicMemoryObservation:
        """Return one bound actual-leader identity with explicit controlled capacity."""
        self.assertEqual(configuration, self.configuration)
        return NativeMusicMemoryObservation(
            process_id=os.getpid(),
            payload_sha256=payload,
            device="cpu",
            capacity=NativeMusicMemoryCapacity(
                sampled_at_unix_ms=1000, host_total_bytes=1000, host_available_bytes=900
            ),
            host_resident_bytes=10,
            device_allocated_bytes=0,
        )

    def observer(self) -> NativeMusicMemoryObserver:
        """Retain real timer/thread ownership while injecting sensing and retirement."""
        return NativeMusicMemoryObserver(
            self.configuration,
            "b" * 64,
            self.frames.append,
            sample=self.sample,
            retire=self.retired.set,
            clock_ms=lambda: 1000,
        )

    def test_initial_observation_precedes_any_background_work(self) -> None:
        """A synchronous sensor refusal cannot start loading without a guard."""
        observer = self.observer()
        observer.sample = lambda config, payload: self.sample(
            config, payload
        ).model_copy(update={"host_resident_bytes": 101})
        try:
            with self.assertRaises(MemoryError):
                observer.start()
            self.assertEqual(self.frames, [])
            self.assertFalse(self.retired.is_set())
        finally:
            observer.close()

    def test_preload_fit_reserves_the_same_physical_pressure_floor(self) -> None:
        """The native process independently refuses changed free capacity before load."""
        observer = self.observer()

        def pressured(
            config: NativeMusicServeConfig, payload: str
        ) -> NativeMusicMemoryObservation:
            actual = self.sample(config, payload)
            return actual.model_copy(
                update={
                    "capacity": actual.capacity.model_copy(
                        update={"host_available_bytes": 140}
                    )
                }
            )

        observer.sample = pressured
        try:
            with self.assertRaises(MemoryError):
                observer.start()
            self.assertEqual(self.frames, [])
        finally:
            observer.close()

    def test_missing_admitted_budget_is_not_native_serving(self) -> None:
        """Controlled protocol fixtures cannot silently become unguarded production."""
        with self.assertRaises(ValueError):
            NativeMusicMemoryObserver(
                self.configuration.model_copy(update={"memory_budget": None}),
                "b" * 64,
                self.frames.append,
            )

    def test_foreign_sampler_identity_cannot_emit_a_heartbeat(self) -> None:
        """Only this payload, process and selected device can attest live capacity."""
        observer = self.observer()
        observer.sample = lambda config, payload: self.sample(
            config, payload
        ).model_copy(update={"process_id": os.getpid() + 1})
        try:
            with self.assertRaises(ValueError):
                observer.start()
            self.assertEqual(self.frames, [])
        finally:
            observer.close()

    def test_background_sensor_failure_retires_even_with_closed_diagnostics(
        self,
    ) -> None:
        """Broken stderr must never bypass retirement of running native threads."""
        observer = self.observer()
        observer.start()

        def unavailable(
            _config: NativeMusicServeConfig, _payload: str
        ) -> NativeMusicMemoryObservation:
            raise OSError("sensor stopped")

        observer.sample = unavailable
        closed = io.StringIO()
        closed.close()
        try:
            with patch("sys.stderr", closed):
                self.assertTrue(self.retired.wait(3))
        finally:
            observer.close()
        self.assertEqual(len(self.frames), 1)

    def test_control_egress_failure_also_retires_inference(self) -> None:
        """A failed heartbeat output is not a reason to continue running unobserved."""
        observer = self.observer()
        observer.start()

        def broken(_frame: NativeMusicFrame) -> None:
            raise BrokenPipeError("reader closed")

        observer.emit = broken
        try:
            self.assertTrue(self.retired.wait(3))
        finally:
            observer.close()

    def test_close_joins_observation_and_retires_the_observer_lifetime(self) -> None:
        """No sample may race a successfully closed control/file owner."""
        observer = self.observer()
        second = threading.Event()

        def emit(frame: NativeMusicFrame) -> None:
            self.frames.append(frame)
            if isinstance(frame, NativeMusicMemoryFrame) and frame.sequence == 2:
                second.set()

        observer.emit = emit
        try:
            observer.start()
            self.assertTrue(second.wait(3))
        finally:
            observer.close()
        self.assertEqual(
            [
                frame.sequence
                for frame in self.frames
                if isinstance(frame, NativeMusicMemoryFrame)
            ],
            [1, 2],
        )
        with self.assertRaises(RuntimeError):
            observer.start()

    def test_actual_native_guard_kills_its_sigterm_ignoring_descendant(self) -> None:
        """The default guard effect retires a real isolated group without Core help."""
        configuration = self.root / "configuration.json"
        configuration.write_text(self.configuration.model_dump_json())
        child_code = """import json,os,signal,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from skulk_ace_step.lifetime import start_parent_guardian
start_parent_guardian(os.getppid())
from skulk_ace_step.contracts import NativeMusicServeConfig,NativeMusicMemoryObservation,NativeMusicMemoryCapacity
from skulk_ace_step.resources import NativeMusicMemoryObserver
config=NativeMusicServeConfig.model_validate_json(Path(sys.argv[2]).read_bytes())
root=Path(sys.argv[2]).parent
child=subprocess.Popen([sys.executable,'-c',"import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);print('ready',flush=True);time.sleep(120)"],stdout=subprocess.PIPE)
assert child.stdout.readline()==b'ready\\n'
(root/'child.pid').write_text(str(child.pid))
calls=0
def sample(_config,payload):
 global calls
 calls+=1
 return NativeMusicMemoryObservation(process_id=os.getpid(),payload_sha256=payload,device='cpu',capacity=NativeMusicMemoryCapacity(sampled_at_unix_ms=1000,host_total_bytes=1000,host_available_bytes=900 if calls==1 else 0),host_resident_bytes=10,device_allocated_bytes=0)
observer=NativeMusicMemoryObserver(config,'b'*64,lambda _frame:None,sample=sample,clock_ms=lambda:1000)
observer.start()
while True:time.sleep(.01)
"""
        package = str(Path(skulk_ace_step.__file__).resolve().parent.parent)
        child = subprocess.Popen(
            [sys.executable, "-I", "-B", "-c", child_code, package, str(configuration)],
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            _, diagnostic = child.communicate(timeout=15)
            self.assertEqual(child.returncode, -signal.SIGKILL, diagnostic.decode())
            descendant = int((self.root / "child.pid").read_text())
            deadline = time.monotonic() + 3
            while True:
                try:
                    if psutil.Process(descendant).status() == psutil.STATUS_ZOMBIE:
                        break
                except psutil.NoSuchProcess:
                    break
                if time.monotonic() >= deadline:
                    self.fail("native memory guard left its actual descendant alive")
                time.sleep(0.025)
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.communicate()
