"""Resident control/custody admission, independent of model/native imports."""

from __future__ import annotations

import io
import os
import tempfile
import unittest
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import check_support
from skulk_ace_step.contracts import (
    MAX_NATIVE_MUSIC_RECORD_BYTES,
    NativeMusicCompleted,
    NativeMusicFailed,
    NativeMusicFrame,
    NativeMusicProgress,
    NativeMusicReady,
    NativeMusicReadyFrame,
    NativeMusicRequest,
    NativeMusicResult,
    NativeMusicServeConfig,
    NativeMusicServeRequest,
    NativeMusicWav,
)
from skulk_ace_step.server import serve


@dataclass
class ModelFixture:
    """Controlled model interface; no audio generation or weight loading claimed."""

    configuration: NativeMusicServeConfig
    ready: NativeMusicReady
    calls: list[NativeMusicRequest] = field(default_factory=list)
    foreign: bool = False
    missing: bool = False
    foreign_progress: bool = False
    fail: bool = False

    def run(
        self,
        request: NativeMusicRequest,
        progress: Callable[[NativeMusicProgress], None],
    ) -> NativeMusicResult:
        """Retain real strict metadata to exercise completion-before-custody races."""
        self.calls.append(request)
        progress(
            NativeMusicProgress(
                request_id="foreign" if self.foreign_progress else request.request_id,
                stage="generating",
            )
        )
        if self.fail:
            raise TimeoutError("native timeout could leave an inference thread alive")
        result = NativeMusicResult(
            request_id="foreign" if self.foreign else request.request_id,
            model_identity=self.configuration.model_identity,
            payload_sha256=self.ready.payload_sha256,
            operation=request.operation,
            seed=request.seed,
            execution=self.ready.execution,
            output=NativeMusicWav(
                sha256="c" * 64,
                size_bytes=44 + 48000 * 6,
                frames=48000,
                peak=0.25,
                gain=1.0,
            ),
            generation_seconds=0.25,
        )
        if not self.missing:
            directory = Path(request.output_directory)
            with os.fdopen(
                os.open(
                    directory / "result.json",
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                ),
                "wb",
            ) as stream:
                stream.write(result.model_dump_json().encode())
        return result


class NativeServerTests(unittest.TestCase):
    """Exercise actual bounded records, sequence admission and terminal controls."""

    def setUp(self) -> None:
        """Create one loaded boundary fixture and an ordinary two-request stream."""
        self.temporary = tempfile.TemporaryDirectory(prefix="native-music-control-")
        self.config = check_support.configuration(Path(self.temporary.name))
        self.model = ModelFixture(
            self.config,
            NativeMusicReady(
                model_identity=self.config.model_identity,
                payload_sha256="b" * 64,
                execution=check_support.execution(),
                load_seconds=0.25,
            ),
        )
        self.frames: list[NativeMusicFrame] = []
        self.loads = 0

    def tearDown(self) -> None:
        """Remove private fixture files after the control loop has returned."""
        self.temporary.cleanup()

    def load(
        self,
        configuration: NativeMusicServeConfig,
        payload: str,
        progress: Callable[[NativeMusicProgress], None],
    ) -> ModelFixture:
        """Observe real configuration binding; emit only initialization progress."""
        self.loads += 1
        self.assertEqual(configuration, self.config)
        self.assertEqual(payload, "b" * 64)
        progress(NativeMusicProgress(stage="loading"))
        return self.model

    def wire(self, *sequences: int) -> io.BytesIO:
        """Encode the same strict newline request envelopes used by the CLI."""
        return io.BytesIO(
            b"".join(
                NativeMusicServeRequest(
                    sequence=s, request=check_support.request(self.config, f"take-{s}")
                )
                .model_dump_json()
                .encode()
                + b"\n"
                for s in sequences
            )
        )

    def run_server(self, stream: io.BytesIO) -> None:
        """Use the actual resident loop with controlled model effects."""
        serve(
            self.config,
            stream,
            self.frames.append,
            payload_sha256="b" * 64,
            load=self.load,
        )

    def test_two_requests_load_once_and_publish_only_retained_results(self) -> None:
        """Both strict request IDs stay correlated, without reloading the model."""
        self.run_server(self.wire(1, 2))
        self.assertEqual(self.loads, 1)
        self.assertEqual(
            sum(isinstance(f, NativeMusicReadyFrame) for f in self.frames), 1
        )
        completed = [
            f.result for f in self.frames if isinstance(f, NativeMusicCompleted)
        ]
        self.assertEqual([r.request_id for r in completed], ["take-1", "take-2"])
        for result in completed:
            path = (
                Path(self.config.directory)
                / "requests"
                / result.request_id
                / "result.json"
            )
            self.assertEqual(
                NativeMusicResult.model_validate_json(path.read_bytes()), result
            )

    def test_gap_replay_unknown_and_unbounded_records_are_terminal(self) -> None:
        """Invalid framing never reaches a second expensive model invocation."""
        streams = [
            self.wire(2),
            self.wire(1, 1),
            io.BytesIO(b"{}\n"),
            io.BytesIO(b"x" * (MAX_NATIVE_MUSIC_RECORD_BYTES + 1) + b"\n"),
            io.BytesIO(self.wire(1).getvalue().rstrip(b"\n")),
        ]
        for stream in streams:
            with self.subTest(prefix=stream.getvalue()[:24]):
                self.frames.clear()
                self.model.calls.clear()
                with self.assertRaises(ValueError):
                    self.run_server(stream)
                self.assertIsInstance(self.frames[-1], NativeMusicFailed)
                self.assertLessEqual(len(self.model.calls), 1)
                # Replayed take-1 may have created its fixture record. Preserve
                # each rejected stream independently instead of reusing files.
                path = Path(self.config.directory) / "requests/take-1/result.json"
                if path.exists():
                    path.unlink()
                    path.parent.rmdir()

    def test_foreign_ready_identity_never_advertises_capacity(self) -> None:
        """A load echoing another artifact cannot become ready."""
        self.model.ready = self.model.ready.model_copy(
            update={"model_identity": "d" * 64}
        )
        with self.assertRaises(ValueError):
            self.run_server(self.wire(1))
        self.assertFalse(any(isinstance(f, NativeMusicReadyFrame) for f in self.frames))
        self.assertEqual(self.model.calls, [])

    def test_foreign_completion_and_unretained_result_never_publish(self) -> None:
        """A tensor-success flag cannot substitute for correlated file custody."""
        for name in ("foreign", "missing", "foreign_progress"):
            with self.subTest(name=name):
                self.frames.clear()
                setattr(self.model, name, True)
                with self.assertRaises((ValueError, FileNotFoundError)):
                    self.run_server(self.wire(1))
                self.assertFalse(
                    any(isinstance(f, NativeMusicCompleted) for f in self.frames)
                )
                self.assertIsInstance(self.frames[-1], NativeMusicFailed)
                setattr(self.model, name, False)
                path = Path(self.config.directory) / "requests/take-1/result.json"
                if path.exists():
                    path.unlink()
                    path.parent.rmdir()

    def test_timeout_stops_before_next_queued_request(self) -> None:
        """The native loop never reuses a model with a potentially active thread."""
        self.model.fail = True
        with self.assertRaises(TimeoutError):
            self.run_server(self.wire(1, 2))
        self.assertEqual([r.request_id for r in self.model.calls], ["take-1"])
        self.assertIsInstance(self.frames[-1], NativeMusicFailed)
