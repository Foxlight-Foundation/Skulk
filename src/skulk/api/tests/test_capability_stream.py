"""End-to-end provider server-streaming dispatch over the local DATA path."""

from collections.abc import AsyncIterator

import anyio
import pytest

from skulk.api.main import (
    API,
    _ActiveProviderStream,  # pyright: ignore[reportPrivateUsage]
)
from skulk.extensions import (
    CapabilityCall,
    CapabilityDescriptor,
    CapabilityError,
    CapabilityResult,
    CapabilityStreamFrame,
    ExtensionContext,
    InlineMediaAttachment,
    LoadedExtensions,
    descriptor_revision,
)
from skulk.routing.provider_streams import ProviderStreamPacket
from skulk.shared.election import ElectionMessage
from skulk.shared.types.commands import ForwarderCommand, ForwarderDownloadCommand
from skulk.shared.types.common import NodeId
from skulk.shared.types.events import IndexedEvent
from skulk.shared.types.telemetry import TelemetryView
from skulk.utils.channels import channel

_TTS = CapabilityDescriptor(
    id="tts",
    version="1.0.0",
    title="Test speech synthesis",
    description="Streams deterministic PCM bytes for provider transport tests.",
    input_schema={
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    },
    io_mode="server_streaming",
    output_chunk_schema={
        "type": "object",
        "properties": {"format": {"const": "pcm_s16le"}},
        "required": ["format"],
        "additionalProperties": False,
    },
)
_TTS_REVISION = descriptor_revision(_TTS)

_BIDIRECTIONAL = CapabilityDescriptor(
    id="realtime-stt",
    version="1.0.0",
    title="Future realtime STT",
    description="Discovery-only until caller input frames are implemented.",
    input_schema={"type": "object"},
    io_mode="bidirectional",
    input_chunk_schema={"type": "object"},
    output_chunk_schema={"type": "object"},
)


class _TtsProvider:
    name = "tts-test"
    skulk_requires = ">=0"

    def chat_middleware(self) -> None:
        return None

    def capabilities(self) -> list[CapabilityDescriptor]:
        return [_TTS]

    async def handle_stream(
        self,
        context: ExtensionContext,
        call: CapabilityCall,
    ) -> AsyncIterator[CapabilityStreamFrame]:
        yield CapabilityStreamFrame(
            call_id=call.call_id,
            direction="provider_to_caller",
            sequence=1,
            kind="chunk",
            payload={"format": "pcm_s16le"},
            media=InlineMediaAttachment(
                data=b"\x00\xff\x80\x7f",
                media_type="audio/pcm",
                codec="pcm_s16le",
                sample_rate=24000,
                channels=1,
            ),
        )
        yield CapabilityStreamFrame(
            call_id=call.call_id,
            direction="provider_to_caller",
            sequence=2,
            kind="completed",
        )


class _InvalidChunkProvider(_TtsProvider):
    async def handle_stream(
        self,
        context: ExtensionContext,
        call: CapabilityCall,
    ) -> AsyncIterator[CapabilityStreamFrame]:
        yield CapabilityStreamFrame(
            call_id=call.call_id,
            direction="provider_to_caller",
            sequence=1,
            kind="chunk",
            payload={"format": "not-pcm"},
        )


class _RejectingTtsProvider(_TtsProvider):
    def __init__(self) -> None:
        self.handler_called = False

    async def admit_stream(
        self,
        context: ExtensionContext,
        call: CapabilityCall,
    ) -> CapabilityError | None:
        return CapabilityError(
            code="not_found",
            message="no eligible model is mounted",
        )

    async def handle_stream(
        self,
        context: ExtensionContext,
        call: CapabilityCall,
    ) -> AsyncIterator[CapabilityStreamFrame]:
        self.handler_called = True
        if False:
            yield CapabilityStreamFrame(
                call_id=call.call_id,
                direction="provider_to_caller",
                sequence=1,
                kind="completed",
            )


class _BlockingAdmissionTtsProvider(_TtsProvider):
    def __init__(self) -> None:
        self.admission_started = anyio.Event()
        self.release_admission = anyio.Event()

    async def admit_stream(
        self,
        context: ExtensionContext,
        call: CapabilityCall,
    ) -> CapabilityError | None:
        self.admission_started.set()
        await self.release_admission.wait()
        return None


class _CancellableProvider(_TtsProvider):
    def __init__(self) -> None:
        self.cancelled = anyio.Event()

    async def handle_stream(
        self,
        context: ExtensionContext,
        call: CapabilityCall,
    ) -> AsyncIterator[CapabilityStreamFrame]:
        try:
            yield CapabilityStreamFrame(
                call_id=call.call_id,
                direction="provider_to_caller",
                sequence=1,
                kind="chunk",
                payload={"format": "pcm_s16le"},
            )
            await anyio.sleep_forever()
        finally:
            self.cancelled.set()


class _BurstProvider(_TtsProvider):
    async def handle_stream(
        self,
        context: ExtensionContext,
        call: CapabilityCall,
    ) -> AsyncIterator[CapabilityStreamFrame]:
        for sequence in range(1, 302):
            yield CapabilityStreamFrame(
                call_id=call.call_id,
                direction="provider_to_caller",
                sequence=sequence,
                kind="chunk",
                payload={"format": "pcm_s16le"},
            )
        yield CapabilityStreamFrame(
            call_id=call.call_id,
            direction="provider_to_caller",
            sequence=302,
            kind="completed",
        )


class _FloodProvider(_TtsProvider):
    def __init__(self) -> None:
        self.cancelled = anyio.Event()

    async def handle_stream(
        self,
        context: ExtensionContext,
        call: CapabilityCall,
    ) -> AsyncIterator[CapabilityStreamFrame]:
        sequence = 1
        try:
            while True:
                yield CapabilityStreamFrame(
                    call_id=call.call_id,
                    direction="provider_to_caller",
                    sequence=sequence,
                    kind="chunk",
                    payload={"format": "pcm_s16le"},
                )
                sequence += 1
                await anyio.sleep(0)
        finally:
            self.cancelled.set()


class _BidirectionalProvider(_TtsProvider):
    def capabilities(self) -> list[CapabilityDescriptor]:
        return [_BIDIRECTIONAL]


def _build_api(provider: object) -> API:
    command_sender, _ = channel[ForwarderCommand]()
    download_sender, _ = channel[ForwarderDownloadCommand]()
    _, event_receiver = channel[IndexedEvent]()
    _, election_receiver = channel[ElectionMessage]()
    provider_sender, provider_receiver = channel[ProviderStreamPacket](256)
    return API(
        NodeId("api-node"),
        port=52415,
        event_receiver=event_receiver,
        command_sender=command_sender,
        download_command_sender=download_sender,
        election_receiver=election_receiver,
        enable_event_log=False,
        mount_dashboard=False,
        telemetry_view=TelemetryView(),
        provider_stream_sender=provider_sender,
        provider_stream_receiver=provider_receiver,
        extensions=LoadedExtensions([provider]),  # pyright: ignore[reportArgumentType]
    )


async def _collect_local_stream(
    api: API,
) -> tuple[bool, list[CapabilityStreamFrame]]:
    context = api._extension_context  # pyright: ignore[reportPrivateUsage]
    opened = False
    frames: list[CapabilityStreamFrame] = []
    async with api._tg as task_group:  # pyright: ignore[reportPrivateUsage]
        task_group.start_soon(
            api._apply_provider_data  # pyright: ignore[reportPrivateUsage]
        )
        session = await context.stream_capability(
            NodeId("api-node"),
            "tts",
            "1.0.0",
            _TTS_REVISION,
            {"text": "hello"},
            timeout_seconds=2.0,
        )
        opened = session.open_result.ok
        frames = [frame async for frame in session.frames]
        task_group.cancel_scope.cancel()
    return opened, frames


async def test_local_provider_stream_preserves_lifecycle_and_binary_media() -> None:
    opened, frames = await _collect_local_stream(_build_api(_TtsProvider()))

    assert opened is True
    assert [frame.kind for frame in frames] == ["started", "chunk", "completed"]
    assert [frame.sequence for frame in frames] == [0, 1, 2]
    assert isinstance(frames[1].media, InlineMediaAttachment)
    assert frames[1].media.data == b"\x00\xff\x80\x7f"


async def test_remote_open_uses_peer_api_but_media_uses_provider_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fabric_sender, fabric_receiver = channel[ProviderStreamPacket](256)

    def build_node(
        node_id: str,
        *,
        provider: object | None,
        stream_sender: object | None,
        stream_receiver: object | None,
    ) -> API:
        command_sender, _ = channel[ForwarderCommand]()
        download_sender, _ = channel[ForwarderDownloadCommand]()
        _, event_receiver = channel[IndexedEvent]()
        _, election_receiver = channel[ElectionMessage]()
        return API(
            NodeId(node_id),
            port=52415,
            event_receiver=event_receiver,
            command_sender=command_sender,
            download_command_sender=download_sender,
            election_receiver=election_receiver,
            enable_event_log=False,
            mount_dashboard=False,
            telemetry_view=TelemetryView(),
            provider_stream_sender=stream_sender,  # type: ignore[arg-type]
            provider_stream_receiver=stream_receiver,  # type: ignore[arg-type]
            extensions=(
                LoadedExtensions([provider])  # pyright: ignore[reportArgumentType]
                if provider is not None
                else None
            ),
        )

    provider_api = build_node(
        "provider-node",
        provider=_TtsProvider(),
        stream_sender=fabric_sender,
        stream_receiver=None,
    )
    caller_api = build_node(
        "caller-node",
        provider=None,
        stream_sender=None,
        stream_receiver=fabric_receiver,
    )

    async def peer_url(node_id: NodeId) -> str | None:
        return "http://provider.test" if node_id == NodeId("provider-node") else None

    monkeypatch.setattr(caller_api, "_peer_api_url_for", peer_url)

    class _Response:
        def __init__(self, result: object) -> None:
            self._result = result

        def raise_for_status(self) -> None:
            return None

        def json(self) -> object:
            return self._result

    class _PeerClient:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def __aenter__(self) -> "_PeerClient":
            return self

        async def __aexit__(self, *_args: object) -> None:
            return None

        async def post(self, url: str, *, json: object) -> _Response:
            assert url.endswith("/v1/capabilities/stream")
            call = CapabilityCall.model_validate(json)
            result = await provider_api.serve_capability_stream(call)
            return _Response(result.model_dump(mode="json"))

    monkeypatch.setattr("skulk.api.main.httpx.AsyncClient", _PeerClient)

    opened = False
    frames: list[CapabilityStreamFrame] = []
    async with (
        provider_api._tg as provider_tasks,  # pyright: ignore[reportPrivateUsage]
        caller_api._tg as caller_tasks,  # pyright: ignore[reportPrivateUsage]
    ):
        caller_tasks.start_soon(
            caller_api._apply_provider_data  # pyright: ignore[reportPrivateUsage]
        )
        session = await caller_api._extension_context.stream_capability(  # pyright: ignore[reportPrivateUsage]
            NodeId("provider-node"),
            "tts",
            "1.0.0",
            _TTS_REVISION,
            {"text": "remote hello"},
            timeout_seconds=2.0,
        )
        opened = session.open_result.ok
        frames = [frame async for frame in session.frames]
        caller_tasks.cancel_scope.cancel()
        provider_tasks.cancel_scope.cancel()

    assert opened is True
    assert [frame.kind for frame in frames] == ["started", "chunk", "completed"]
    assert isinstance(frames[1].media, InlineMediaAttachment)
    assert frames[1].media.data == b"\x00\xff\x80\x7f"


async def test_invalid_provider_chunk_becomes_typed_failed_terminal() -> None:
    opened, frames = await _collect_local_stream(
        _build_api(_InvalidChunkProvider())
    )

    assert opened is True
    assert [frame.kind for frame in frames] == ["started", "failed"]
    assert frames[-1].error is not None
    assert frames[-1].error.code == "invalid_frame"


async def test_dynamic_admission_rejection_emits_no_started_frame() -> None:
    provider = _RejectingTtsProvider()
    api = _build_api(provider)
    call = CapabilityCall(
        call_id="rejected-before-start",
        capability_id="tts",
        version="1.0.0",
        descriptor_revision=_TTS_REVISION,
        caller_node="api-node",
        target_node="api-node",
        timeout_seconds=2.0,
        payload={"text": "hello"},
    )

    result: CapabilityResult | None = None
    async with api._tg:  # pyright: ignore[reportPrivateUsage]
        result = await api.serve_capability_stream(call)

    assert result is not None
    assert result.ok is False
    assert result.error is not None
    assert result.error.code == "not_found"
    assert provider.handler_called is False
    assert api._active_capability_streams == {}  # pyright: ignore[reportPrivateUsage]


async def test_dynamic_admission_reserves_concurrency_slot_before_await(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("skulk.api.main._MAX_CONCURRENT_CAPABILITY_STREAMS", 1)
    provider = _BlockingAdmissionTtsProvider()
    api = _build_api(provider)

    def call(call_id: str) -> CapabilityCall:
        return CapabilityCall(
            call_id=call_id,
            capability_id="tts",
            version="1.0.0",
            descriptor_revision=_TTS_REVISION,
            caller_node="api-node",
            target_node="api-node",
            timeout_seconds=2.0,
            payload={"text": "hello"},
        )

    first_results: list[CapabilityResult] = []

    async def open_first() -> None:
        first_results.append(await api.serve_capability_stream(call("first")))

    async with api._tg as task_group:  # pyright: ignore[reportPrivateUsage]
        task_group.start_soon(open_first)
        await provider.admission_started.wait()

        second_result = await api.serve_capability_stream(call("second"))
        assert second_result.ok is False
        assert second_result.error is not None
        assert second_result.error.code == "overloaded"
        assert list(api._active_capability_streams) == ["first"]  # pyright: ignore[reportPrivateUsage]

        provider.release_admission.set()
        while not first_results:
            await anyio.sleep(0)
        assert first_results[0].ok is True
        task_group.cancel_scope.cancel()


def test_bidirectional_descriptor_remains_discoverable_but_not_executable() -> None:
    loaded = LoadedExtensions([_BidirectionalProvider()])

    assert loaded.capability_descriptors == (_BIDIRECTIONAL,)
    assert loaded.stream_handler(_BIDIRECTIONAL.qualified_id) is None


async def test_early_caller_close_cancels_only_its_provider_stream() -> None:
    provider = _CancellableProvider()
    api = _build_api(provider)
    context = api._extension_context  # pyright: ignore[reportPrivateUsage]

    async with api._tg as task_group:  # pyright: ignore[reportPrivateUsage]
        task_group.start_soon(
            api._apply_provider_data  # pyright: ignore[reportPrivateUsage]
        )
        session = await context.stream_capability(
            NodeId("api-node"),
            "tts",
            "1.0.0",
            _TTS_REVISION,
            {"text": "cancel me"},
            timeout_seconds=5.0,
        )
        assert session.open_result.ok is True
        iterator = session.frames.__aiter__()
        assert (await iterator.__anext__()).kind == "started"
        assert (await iterator.__anext__()).kind == "chunk"
        await session.frames.aclose()  # type: ignore[attr-defined]
        with anyio.fail_after(1.0):
            await provider.cancelled.wait()
        task_group.cancel_scope.cancel()


async def test_caller_stream_can_be_finalized_by_a_different_task() -> None:
    """Async-generator finalization must not inherit an open deadline scope."""

    provider = _CancellableProvider()
    api = _build_api(provider)
    context = api._extension_context  # pyright: ignore[reportPrivateUsage]

    async with api._tg as task_group:  # pyright: ignore[reportPrivateUsage]
        task_group.start_soon(
            api._apply_provider_data  # pyright: ignore[reportPrivateUsage]
        )
        session = await context.stream_capability(
            NodeId("api-node"),
            "tts",
            "1.0.0",
            _TTS_REVISION,
            {"text": "finalize me elsewhere"},
            timeout_seconds=5.0,
        )
        assert session.open_result.ok is True
        iterator = session.frames.__aiter__()
        assert (await iterator.__anext__()).kind == "started"

        async def close_from_child_task() -> None:
            await session.frames.aclose()  # type: ignore[attr-defined]

        task_group.start_soon(close_from_child_task)
        with anyio.fail_after(1.0):
            await provider.cancelled.wait()
        task_group.cancel_scope.cancel()


async def test_cancel_racing_admission_still_emits_started_first() -> None:
    api = _build_api(_TtsProvider())
    call = CapabilityCall(
        call_id="cancel-before-start",
        capability_id="tts",
        version="1.0.0",
        descriptor_revision=_TTS_REVISION,
        caller_node="api-node",
        target_node="api-node",
        timeout_seconds=2.0,
        payload={"text": "cancel immediately"},
    )
    cancel_requested = anyio.Event()
    cancel_requested.set()
    active = _ActiveProviderStream(
        caller_node="api-node",
        cancel_requested=cancel_requested,
    )

    await api._run_capability_stream(  # pyright: ignore[reportPrivateUsage]
        call,
        "tts-test",
        _TtsProvider(),
        _TTS,
        active,
    )
    assert api._provider_stream_receiver is not None  # pyright: ignore[reportPrivateUsage]
    first = await api._provider_stream_receiver.receive()  # pyright: ignore[reportPrivateUsage]
    second = await api._provider_stream_receiver.receive()  # pyright: ignore[reportPrivateUsage]

    assert [first.frame.kind, second.frame.kind] == ["started", "cancelled"]
    assert [first.frame.sequence, second.frame.sequence] == [0, 1]


async def test_caller_queue_overflow_cannot_report_truncated_stream_complete() -> None:
    api = _build_api(_BurstProvider())
    context = api._extension_context  # pyright: ignore[reportPrivateUsage]
    opened = False
    frames: list[CapabilityStreamFrame] = []

    async with api._tg as task_group:  # pyright: ignore[reportPrivateUsage]
        task_group.start_soon(
            api._apply_provider_data  # pyright: ignore[reportPrivateUsage]
        )
        session = await context.stream_capability(
            NodeId("api-node"),
            "tts",
            "1.0.0",
            _TTS_REVISION,
            {"text": "overflow the bounded caller queue"},
            timeout_seconds=5.0,
        )
        opened = session.open_result.ok
        await anyio.sleep(0.1)
        frames = [frame async for frame in session.frames]
        task_group.cancel_scope.cancel()

    assert opened is True
    assert frames[0].kind == "started"
    assert frames[-1].kind == "failed"
    assert frames[-1].error is not None
    assert frames[-1].error.code == "transport_error"
    assert all(frame.kind != "completed" for frame in frames)
    assert [frame.sequence for frame in frames] == list(range(len(frames)))


async def test_non_consuming_caller_overflow_cancels_provider_immediately() -> None:
    provider = _FloodProvider()
    api = _build_api(provider)
    context = api._extension_context  # pyright: ignore[reportPrivateUsage]

    async with api._tg as task_group:  # pyright: ignore[reportPrivateUsage]
        task_group.start_soon(
            api._apply_provider_data  # pyright: ignore[reportPrivateUsage]
        )
        session = await context.stream_capability(
            NodeId("api-node"),
            "tts",
            "1.0.0",
            _TTS_REVISION,
            {"text": "do not consume this stream"},
            timeout_seconds=5.0,
        )
        assert session.open_result.ok is True
        with anyio.fail_after(1.0):
            await provider.cancelled.wait()
        await session.frames.aclose()  # type: ignore[attr-defined]
        task_group.cancel_scope.cancel()
