"""Tests for extension discovery, version gating, and guarded dispatch."""

import asyncio
from collections.abc import AsyncGenerator
from importlib.metadata import EntryPoint

import pytest

from skulk.extensions import (
    BaseChatMiddleware,
    ChatResponseSummary,
    ExtensionContext,
    LoadedExtensions,
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


def _context() -> ExtensionContext:
    return ExtensionContext(
        node_id=NodeId("test-node"),
        skulk_version="1.3.1",
        embed_texts=_embed_stub,
        read_cluster=lambda: (),
        advertise_capability=lambda capability: None,  # noqa: ARG005
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
