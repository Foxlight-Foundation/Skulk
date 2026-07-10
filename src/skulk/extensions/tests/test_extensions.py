"""Tests for extension discovery, version gating, and guarded dispatch."""

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from importlib.metadata import EntryPoint

import pytest

from skulk.extensions import (
    BaseChatMiddleware,
    CapabilityDescriptor,
    CapabilityResult,
    CapabilityStreamFrame,
    CapabilityStreamSession,
    ChatResponseSummary,
    ExtensionContext,
    LoadedExtensions,
    call_failure,
    load_extensions,
)
from skulk.extensions.loader import ChatStreamChunk
from skulk.shared.types.chunks import ErrorChunk, TokenChunk
from skulk.shared.types.common import ModelId, NodeId
from skulk.shared.types.text_generation import (
    InputMessage,
    TextGenerationTaskParams,
)

_TESTS_MODULE = "skulk.extensions.tests.test_extensions"


async def _embed_stub(
    texts: list[str], model_id: ModelId | None = None
) -> list[list[float]] | None:
    """Deterministic embeddings for tests."""
    return [[float(len(text))] for text in texts]


async def _describe_stub(node_id: NodeId) -> tuple[CapabilityDescriptor, ...]:
    """Empty describe surface for tests."""
    return ()


async def _call_stub(
    node_id: NodeId,
    capability_id: str,
    version: str,
    descriptor_revision: str,
    payload: dict[str, object],
    *,
    timeout_seconds: float | None = None,
) -> CapabilityResult:
    """Unreachable call surface for tests."""
    return call_failure("test-call", "unreachable", "no fabric in tests")


async def _empty_stream() -> AsyncIterator[CapabilityStreamFrame]:
    if False:
        yield CapabilityStreamFrame(
            call_id="unused",
            direction="provider_to_caller",
            sequence=0,
            kind="started",
        )


async def _stream_stub(
    node_id: NodeId,
    capability_id: str,
    version: str,
    descriptor_revision: str,
    payload: dict[str, object],
    *,
    timeout_seconds: float | None = None,
) -> CapabilityStreamSession:
    """Unreachable streaming surface for tests."""

    return CapabilityStreamSession(
        open_result=call_failure(
            "test-stream", "unreachable", "no fabric in tests"
        ),
        frames=_empty_stream(),
    )


def _context() -> ExtensionContext:
    return ExtensionContext(
        node_id=NodeId("test-node"),
        skulk_version="1.3.1",
        embed_texts=_embed_stub,
        read_cluster=lambda: (),
        advertise_capability=lambda capability: None,  # noqa: ARG005
        withdraw_capability=lambda capability: None,  # noqa: ARG005
        describe_node=_describe_stub,
        call_capability=_call_stub,
        stream_capability=_stream_stub,
    )


def _params() -> TextGenerationTaskParams:
    return TextGenerationTaskParams(
        model=ModelId("test-model"),
        input=[InputMessage(role="user", content="hello")],
    )


def _token(text: str, finish: bool = False) -> TokenChunk:
    return TokenChunk(
        model=ModelId("test-model"),
        text=text,
        token_id=0,
        usage=None,
        finish_reason="stop" if finish else None,
    )


class _RecordingMiddleware(BaseChatMiddleware):
    """Middleware that tags requests and records observed summaries."""

    def __init__(self) -> None:
        self.summaries: list[ChatResponseSummary] = []

    async def transform_chat_request(
        self,
        context: ExtensionContext,
        task_params: TextGenerationTaskParams,
    ) -> TextGenerationTaskParams:
        marker = f"[ext {context.skulk_version}]"
        instructions = (
            f"{task_params.instructions}\n{marker}"
            if task_params.instructions
            else marker
        )
        return task_params.model_copy(update={"instructions": instructions})

    async def observe_chat_response(
        self,
        context: ExtensionContext,
        task_params: TextGenerationTaskParams,
        summary: ChatResponseSummary,
    ) -> None:
        self.summaries.append(summary)


class _RaisingMiddleware(BaseChatMiddleware):
    """Middleware whose hooks always raise."""

    async def transform_chat_request(
        self,
        context: ExtensionContext,
        task_params: TextGenerationTaskParams,
    ) -> TextGenerationTaskParams:
        raise RuntimeError("boom")

    async def observe_chat_response(
        self,
        context: ExtensionContext,
        task_params: TextGenerationTaskParams,
        summary: ChatResponseSummary,
    ) -> None:
        raise RuntimeError("boom")


class _StubExtension:
    """Minimal SkulkExtension for direct LoadedExtensions construction."""

    def __init__(
        self,
        name: str = "stub",
        skulk_requires: str = ">=0",
        middleware: BaseChatMiddleware | None = None,
    ) -> None:
        self._name = name
        self._requires = skulk_requires
        self._middleware = middleware

    @property
    def name(self) -> str:
        return self._name

    @property
    def skulk_requires(self) -> str:
        return self._requires

    def chat_middleware(self) -> BaseChatMiddleware | None:
        return self._middleware


def make_compatible_extension() -> _StubExtension:
    """Entry-point factory: loads against any Skulk version."""
    return _StubExtension(name="compatible", skulk_requires=">=1.0")


def make_incompatible_extension() -> _StubExtension:
    """Entry-point factory: requires an ancient Skulk."""
    return _StubExtension(name="incompatible", skulk_requires="==0.0.1")


def make_broken_extension() -> _StubExtension:
    """Entry-point factory that raises."""
    raise RuntimeError("factory exploded")


class _FlakyPropertyExtension:
    """Extension whose metadata properties raise (not AttributeError)."""

    @property
    def name(self) -> str:
        raise RuntimeError("flaky name")

    @property
    def skulk_requires(self) -> str:
        return ">=1.0"

    def chat_middleware(self) -> None:
        return None


def make_flaky_property_extension() -> _FlakyPropertyExtension:
    """Entry-point factory: loads, but its name property raises."""
    return _FlakyPropertyExtension()


class _RaisingMiddlewareFactoryExtension(_StubExtension):
    """Extension whose chat_middleware() itself raises."""

    def chat_middleware(self) -> BaseChatMiddleware | None:
        raise RuntimeError("middleware factory exploded")


def _entry_point(name: str, attribute: str) -> EntryPoint:
    return EntryPoint(
        name=name, value=f"{_TESTS_MODULE}:{attribute}", group="skulk.extensions"
    )


async def _drain(
    stream: AsyncGenerator[ChatStreamChunk, None],
) -> list[ChatStreamChunk]:
    return [chunk async for chunk in stream]


async def _stream_of(
    chunks: list[ChatStreamChunk],
) -> AsyncGenerator[ChatStreamChunk, None]:
    for chunk in chunks:
        yield chunk


async def _settle() -> None:
    """Let scheduled observer tasks run."""
    for _ in range(5):
        await asyncio.sleep(0)


def test_load_extensions_discovers_and_version_gates() -> None:
    loaded = load_extensions(
        candidates=[
            _entry_point("ok", "make_compatible_extension"),
            _entry_point("old", "make_incompatible_extension"),
            _entry_point("broken", "make_broken_extension"),
            _entry_point("missing", "does_not_exist"),
        ],
        skulk_version="1.3.1",
    )
    # Only the compatible extension survives; the stale, broken, and missing
    # ones are skipped loudly rather than stopping the node.
    assert loaded.names == ["compatible"]


def test_raising_metadata_property_is_skipped_not_fatal() -> None:
    # hasattr() would only suppress AttributeError; a plugin property raising
    # anything else must still be skipped without breaking the "loader never
    # raises" contract (it runs at node startup).
    loaded = load_extensions(
        candidates=[
            _entry_point("flaky", "make_flaky_property_extension"),
            _entry_point("ok", "make_compatible_extension"),
        ],
        skulk_version="1.3.1",
    )
    assert loaded.names == ["compatible"]


def test_raising_chat_middleware_loads_without_hooks() -> None:
    # chat_middleware() raising must not crash startup; the extension loads
    # with no chat hooks and everything else proceeds.
    recording = _RecordingMiddleware()
    loaded = LoadedExtensions(
        [
            _RaisingMiddlewareFactoryExtension(name="exploder"),
            _StubExtension(name="recorder", middleware=recording),
        ]
    )
    assert loaded.names == ["exploder", "recorder"]
    assert loaded.has_chat_middleware


def test_kill_switch_skips_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SKULK_EXTENSIONS_DISABLE", "1")
    loaded = load_extensions(
        candidates=[_entry_point("ok", "make_compatible_extension")],
        skulk_version="1.3.1",
    )
    assert loaded.names == []


async def test_transform_applies_and_raising_middleware_is_skipped() -> None:
    recording = _RecordingMiddleware()
    loaded = LoadedExtensions(
        [
            _StubExtension(name="raiser", middleware=_RaisingMiddleware()),
            _StubExtension(name="recorder", middleware=recording),
        ]
    )
    params = await loaded.transform_chat_request(_context(), _params())
    # The raiser is skipped; the recorder's marker lands on the params.
    assert params.instructions is not None
    assert "[ext 1.3.1]" in params.instructions


async def test_tap_is_transparent_and_observers_get_summary() -> None:
    recording = _RecordingMiddleware()
    loaded = LoadedExtensions([_StubExtension(middleware=recording)])
    chunks: list[ChatStreamChunk] = [
        _token("hel"),
        _token("lo", finish=True),
    ]
    tapped = loaded.tap_chat_stream(_context(), _params(), _stream_of(chunks))
    seen = await _drain(tapped)
    await _settle()
    assert seen == chunks  # byte-for-byte passthrough
    assert len(recording.summaries) == 1
    summary = recording.summaries[0]
    assert summary.text == "hello"
    assert summary.finish_reason == "stop"
    assert summary.had_error is False


async def test_tap_summarizes_errors_and_survives_raising_observer() -> None:
    recording = _RecordingMiddleware()
    loaded = LoadedExtensions(
        [
            _StubExtension(name="raiser", middleware=_RaisingMiddleware()),
            _StubExtension(name="recorder", middleware=recording),
        ]
    )
    chunks: list[ChatStreamChunk] = [
        _token("partial"),
        ErrorChunk(model=ModelId("test-model"), error_message="exploded"),
    ]
    tapped = loaded.tap_chat_stream(_context(), _params(), _stream_of(chunks))
    seen = await _drain(tapped)
    await _settle()
    assert seen == chunks
    assert len(recording.summaries) == 1
    assert recording.summaries[0].had_error is True
    assert recording.summaries[0].finish_reason == "error"


async def test_tap_schedules_summary_on_early_close() -> None:
    """A client disconnect (generator closed mid-stream) still notifies."""
    recording = _RecordingMiddleware()
    loaded = LoadedExtensions([_StubExtension(middleware=recording)])
    chunks: list[ChatStreamChunk] = [_token("a"), _token("b"), _token("c")]
    tapped = loaded.tap_chat_stream(_context(), _params(), _stream_of(chunks))
    got_one = False
    async for _ in tapped:
        got_one = True
        break  # simulate the consumer going away after one chunk
    await tapped.aclose()
    await _settle()
    assert got_one
    assert len(recording.summaries) == 1
    assert recording.summaries[0].finish_reason is None
    assert recording.summaries[0].text == "a"


async def test_no_middleware_returns_stream_unwrapped() -> None:
    loaded = LoadedExtensions([_StubExtension(middleware=None)])
    stream = _stream_of([_token("x", finish=True)])
    assert loaded.tap_chat_stream(_context(), _params(), stream) is stream


# --- Provider facet (fabric-citizenship Phase 2a) ---------------------------

_ECHO_DESCRIPTOR = CapabilityDescriptor(
    id="echo",
    version="1.0.0",
    title="Echo",
    description="Returns the input text unchanged.",
    input_schema={"type": "object"},
)


class _ProviderExtension(_StubExtension):
    """Extension that serves one capability and records on_start calls."""

    def __init__(self, name: str = "provider") -> None:
        super().__init__(name=name)
        self.started_with: list[ExtensionContext] = []

    def capabilities(self) -> list[CapabilityDescriptor]:
        return [_ECHO_DESCRIPTOR]

    def on_start(self, context: ExtensionContext) -> None:
        self.started_with.append(context)


class _RaisingProviderExtension(_StubExtension):
    """Extension whose provider facet raises everywhere."""

    def capabilities(self) -> list[CapabilityDescriptor]:
        raise RuntimeError("capabilities exploded")

    def on_start(self, context: ExtensionContext) -> None:
        raise RuntimeError("on_start exploded")


def test_provider_descriptors_are_collected() -> None:
    loaded = LoadedExtensions([_ProviderExtension(), _StubExtension()])
    assert loaded.capability_descriptors == (_ECHO_DESCRIPTOR,)


def test_duplicate_qualified_ids_are_rejected() -> None:
    # One provider per id@version per node: a duplicate would make
    # describe/call ambiguous locally. First one wins; the duplicate is
    # skipped loudly.
    loaded = LoadedExtensions(
        [_ProviderExtension(name="first"), _ProviderExtension(name="second")]
    )
    assert loaded.capability_descriptors == (_ECHO_DESCRIPTOR,)


def test_raising_capabilities_loads_extension_without_them() -> None:
    loaded = LoadedExtensions([_RaisingProviderExtension()])
    assert loaded.names == ["stub"]
    assert loaded.capability_descriptors == ()


def test_run_startup_hooks_dispatches_and_guards() -> None:
    provider = _ProviderExtension()
    loaded = LoadedExtensions([provider, _RaisingProviderExtension()])
    context = _context()
    # The raising hook must not prevent the healthy one from running, and
    # must not raise out of the dispatch.
    loaded.run_startup_hooks(context)
    assert provider.started_with == [context]


def test_non_provider_extension_has_no_capabilities() -> None:
    loaded = LoadedExtensions([_StubExtension()])
    assert loaded.capability_descriptors == ()
