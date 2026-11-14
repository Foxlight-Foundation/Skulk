"""Guarded same-host stream calls, without provider clients or routing policy."""

import asyncio
import json
from collections.abc import Callable
from typing import final

from pydantic import JsonValue, TypeAdapter

from skulk.extensions.native_host_protocol import Received, ResponseFrame
from skulk.extensions.native_host_rpc import Peer
from skulk.extensions.native_inference_contracts import (
    InferenceDescriptor,
    Schema,
    host_stream_kernel,
)

_OBJECT = TypeAdapter(dict[str, JsonValue])
_TEXT = TypeAdapter(str)


@final
class NativeInferenceOutput:
    """Own a native-admitted output and its byte lease until delivery/discard."""

    def __init__(
        self,
        response: Received,
        projection: str,
        media: bytes,
        authorize: Callable[[], None],
        release: Callable[["NativeInferenceOutput"], None],
    ) -> None:
        """Take a protected response only after complete frame/schema admission."""
        self._response, self._projection, self._media = response, projection, media
        self._authorize, self._release = authorize, release
        self._released = False

    def _check(self) -> None:
        if self._released:
            raise ValueError("native stream output was released")
        self._authorize()

    @property
    def projection(self) -> str:
        """Borrow canonical metadata under the current original-caller policy."""
        self._check()
        return self._projection

    @property
    def packet(self) -> bytes:
        """Borrow the exact native wire packet under the current delivery guard."""
        self._check()
        return self._response.payload

    @property
    def media(self) -> bytes:
        """Borrow separate raw media only while this delivery lease is live."""
        self._check()
        return self._media

    @property
    def kind(self) -> str:
        """Return the admitted output kind without retaining provider diagnostics."""
        return _TEXT.validate_python(
            _OBJECT.validate_json(self.projection)["kind"], strict=True
        )

    def release(self) -> None:
        """Discard content and return byte/window capacity exactly once."""
        if not self._released:
            self._response.release()
            self._released = True
            self._projection, self._media = "", b""
            self._release(self)


@final
class NativeInferenceStream:
    """One already dispatched exact call; polling cannot create provider work."""

    def __init__(
        self,
        peer: Peer,
        call_id: str,
        descriptor: InferenceDescriptor,
        control: asyncio.Task[Received],
        authorize: Callable[[], None],
        completed: Callable[[], None],
        write_seconds: float = 300,
        cleanup_seconds: float = 3,
    ) -> None:
        """Bind original-caller guards, declared schemas and the sole control call."""
        native = host_stream_kernel()
        if native is None or descriptor.io_mode == "unary":
            raise ValueError("native stream host is unavailable")
        self.kernel, self.peer, self._call_id = native, peer, call_id
        self._mode, self.write_seconds = descriptor.io_mode, write_seconds
        self.cleanup_seconds = cleanup_seconds
        self.control, self.authorize, self.completed = control, authorize, completed
        self.observer = native.TransportV2Observer(call_id, descriptor.io_mode)
        self.chunk_schema = self._schema(descriptor.output_chunk_schema)
        self.result_schema = self._schema(descriptor.output_schema)
        self.input_lock, self.output_lock = asyncio.Lock(), asyncio.Lock()
        self.borrowed: set[NativeInferenceOutput] = set()
        self._finished = self.closed = False

    @property
    def call_id(self) -> str:
        """Return the fixed host-assigned stream identity for caller input framing."""
        return self._call_id

    @property
    def mode(self) -> str:
        """Return the exact signed stream mode, independently of client input."""
        return self._mode

    @property
    def finished(self) -> bool:
        """Report terminal/end/control verification, independently of peer liveness."""
        return self._finished

    def _schema(self, value: dict[str, JsonValue] | bool | None) -> Schema | None:
        return None if value is None else self.kernel.PortableSchema(json.dumps(value))

    def _check(self) -> None:
        if self.closed:
            raise ValueError("native stream is closed")
        self.authorize()

    async def ready(self, seconds: float) -> None:
        """Observe the existing call within one fixed connection resource window."""
        async with asyncio.timeout(seconds):
            while True:
                self._check()
                if self.control.done():
                    raise ValueError("native stream completed before connection")
                response = await self.peer.request(
                    "stream_ready",
                    {"call_id": self.call_id},
                    check=self._check,
                    cleanup_seconds=self.cleanup_seconds,
                )
                try:
                    if (
                        not isinstance(response.frame, ResponseFrame)
                        or response.payload
                    ):
                        raise ValueError("native stream readiness refused")
                    if (
                        response.frame.outcome == "complete"
                        and response.frame.parameters == {"ready": True}
                    ):
                        return
                    if response.frame.outcome not in (
                        "complete",
                        "pre_dispatch",
                    ) or response.frame.parameters not in ({}, {"ready": False}):
                        raise ValueError("native stream readiness refused")
                finally:
                    response.release()
                await asyncio.sleep(0.01)

    async def send_input(self, wire: bytes) -> None:
        """Write one caller frame; failure permanently withdraws this stream handle."""
        try:
            await self._send_input(wire)
        except BaseException:
            self.close()
            raise

    async def _send_input(self, wire: bytes) -> None:
        """Send one exact native caller frame; completed half-closes input only."""
        async with self.input_lock:
            self._check()
            if self.finished:
                raise ValueError("native stream is finished")
            self.observer.accept_input(wire)
            response = await self.peer.request(
                "stream_input",
                {"call_id": self.call_id},
                wire,
                check=self._check,
                cleanup_seconds=self.cleanup_seconds,
            )
            try:
                if (
                    not isinstance(response.frame, ResponseFrame)
                    or response.frame.outcome != "complete"
                    or response.payload
                    or response.frame.parameters
                ):
                    raise ValueError("native stream input refused")
                self._check()
            finally:
                response.release()

    async def _next(self, *, closing: bool = False) -> NativeInferenceOutput:
        self._check()
        if not closing and len(self.borrowed) >= 8:
            raise ValueError("native stream output window unavailable")
        response = await self.peer.request(
            "stream_next",
            {"call_id": self.call_id},
            check=self._check,
            cleanup_seconds=self.cleanup_seconds,
        )
        transferred = False
        try:
            if (
                not isinstance(response.frame, ResponseFrame)
                or response.frame.outcome != "complete"
                or response.frame.parameters
            ):
                raise ValueError("native stream output unavailable")
            projection, media = self.observer.accept_output(response.payload)
            fields = _OBJECT.validate_json(projection)
            if closing and fields["kind"] != "end":
                raise ValueError("native stream closing acknowledgment refused")
            schema = (
                self.chunk_schema
                if fields["kind"] == "chunk"
                else self.result_schema
                if fields["kind"] == "completed"
                else None
            )
            if (
                schema is not None
                and (fields.get("payload") is not None or fields["kind"] == "completed")
                and not schema.matches(json.dumps(fields["payload"], allow_nan=False))
            ):
                raise ValueError("native stream output schema refused")
            self._check()
            value = NativeInferenceOutput(
                response, projection, media, self._check, self.borrowed.discard
            )
            self.borrowed.add(value)
            transferred = True
            return value
        finally:
            if not transferred:
                response.release()

    async def next_output(self) -> NativeInferenceOutput | None:
        """Borrow one output until released; failed or interrupted reads cannot resume."""
        try:
            return await self._next_output()
        except BaseException:
            self.close()
            raise

    async def _next_output(self) -> NativeInferenceOutput | None:
        """Borrow a frame; terminal delivery waits for end, control and scope cleanup.

        Release each returned frame after writing or discarding it. There is no
        output allowance; the eight-frame window bounds retained host memory.
        """
        async with self.output_lock:
            self._check()
            if self.finished:
                return None
            value = await self._next()
            transferred = False
            try:
                if value.kind != "chunk":
                    # The closing acknowledgment carries no media and uses one
                    # bounded header slot in addition to borrowed data frames.
                    end = await self._next(closing=True)
                    try:
                        if end.kind != "end" or not self.observer.finished:
                            raise ValueError("native stream end refused")
                    finally:
                        end.release()
                    result = await asyncio.shield(self.control)
                    if (
                        not isinstance(result.frame, ResponseFrame)
                        or result.frame.outcome != "complete"
                        or _OBJECT.validate_json(result.payload) != {}
                    ):
                        raise ValueError("native stream control completion refused")
                    self.completed()
                    self._finished = True
                self._check()
                transferred = True
                return value
            finally:
                if not transferred:
                    value.release()

    def close(self) -> None:
        """Withdraw borrowed output; the owning invocation drains control custody."""
        self.closed = True
        for value in list(self.borrowed):
            value.release()
