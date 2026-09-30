"""Dynamic provider availability and bounded resource ownership regressions."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import replace

import anyio
import pytest

from skulk.extensions import (
    CapabilityCall,
    CapabilityDescriptor,
    CapabilityStreamFrame,
    ExtensionContext,
    LoadedExtensions,
)
from skulk.extensions.tests.test_steward_tools import context as fixture_context


class Provider:
    """One instance serves all supported call modes with independently cached health."""

    name = "lifecycle-fixture"
    skulk_requires = ">=0"

    def __init__(self) -> None:
        self.ready = True
        self.broken = False
        self.stops = 0

    def chat_middleware(self) -> None:
        """Leave inference unmodified."""
        return None

    def capabilities(self) -> list[CapabilityDescriptor]:
        """Expose unary, output, input and bidirectional descriptor fixtures."""
        return [
            CapabilityDescriptor(
                id=mode,
                version="1.0.0",
                title=mode,
                description="Lifecycle fixture",
                input_schema={"type": "object"},
                output_schema={"type": "object"},
                io_mode=mode,
                input_chunk_schema={"type": "object"}
                if mode in ("client_streaming", "bidirectional")
                else None,
                output_chunk_schema={"type": "object"}
                if mode in ("server_streaming", "bidirectional")
                else None,
            )
            for mode in (
                "unary",
                "server_streaming",
                "client_streaming",
                "bidirectional",
            )
        ]

    def capability_ready(self, qualified_id: str) -> bool:
        """Return controlled cached truth or raise a simulated probe failure."""
        assert qualified_id.endswith("@1.0.0")
        if self.broken:
            raise RuntimeError("private provider detail")
        return self.ready

    async def handle_call(
        self, context: ExtensionContext, call: CapabilityCall
    ) -> dict[str, object]:
        """Serve a harmless response."""
        return {}

    async def handle_stream(
        self, context: ExtensionContext, call: CapabilityCall
    ) -> AsyncIterator[CapabilityStreamFrame]:
        """Provide the streaming facet without exercising payload framing here."""
        if False:
            yield CapabilityStreamFrame(
                call_id=call.call_id, sequence=1, kind="completed"
            )

    async def handle_input_stream(
        self,
        context: ExtensionContext,
        call: CapabilityCall,
        input_frames: AsyncIterator[CapabilityStreamFrame],
    ) -> AsyncIterator[CapabilityStreamFrame]:
        """Provide the caller-input facet for dynamic lookup coverage."""
        async for frame in self.handle_stream(context, call):
            yield frame

    async def on_stop(self) -> None:
        """Count cleanup completions across repeated shutdown requests."""
        await anyio.sleep(0)
        self.stops += 1


@pytest.mark.parametrize("composed", [False, True])
def test_dynamic_readiness_covers_discovery_and_every_call_mode(composed: bool) -> None:
    """Withdrawal, failure and recovery affect cached descriptors without reload."""
    provider = Provider()
    loaded = LoadedExtensions([provider])
    if composed:
        loaded = loaded.with_builtin_extensions([])
    for ready, broken, visible in (
        (True, False, True),
        (False, False, False),
        (True, True, False),
        (True, False, True),
    ):
        provider.ready, provider.broken = ready, broken
        assert bool(loaded.capability_descriptors) is visible
        assert (loaded.call_handler("unary@1.0.0") is not None) is visible
        for mode in ("server_streaming", "client_streaming", "bidirectional"):
            assert (loaded.stream_handler(mode + "@1.0.0") is not None) is visible
        assert bool(loaded.handled_capability_ids()) is visible
        assert bool(loaded.handled_stream_capability_ids()) is visible


def test_builtin_collision_keeps_readiness_bound_to_winning_provider() -> None:
    """A healthy duplicate cannot reopen a reserved but unhealthy built-in."""
    builtin, external = Provider(), Provider()
    builtin.ready = False
    loaded = LoadedExtensions([external]).with_builtin_extensions([builtin])
    assert loaded.capability_descriptors == ()
    builtin.ready = True
    external.ready = False
    assert len(loaded.capability_descriptors) == 4


class FailedCleanup(Provider):
    """Fail before another extension finishes cleanup."""

    async def on_stop(self) -> None:
        """Raise from a plugin boundary."""
        raise RuntimeError("cleanup failed")


class SlowCleanup(Provider):
    """Cooperate with the shutdown deadline while simulating a hung backend."""

    async def on_stop(self) -> None:
        """Wait until deadline cancellation is delivered."""
        try:
            await anyio.sleep_forever()
        finally:
            self.stops += 1


async def test_shutdown_withdraws_then_cleans_siblings_despite_failure_and_cancellation() -> (
    None
):
    """Outer cancellation cannot discard cleanup; repeated shutdown is inert."""
    provider = Provider()
    loaded = LoadedExtensions([provider, FailedCleanup()])
    with anyio.CancelScope() as scope:
        scope.cancel()
        await loaded.run_shutdown_hooks()
    assert loaded.capability_descriptors == ()
    assert loaded.call_handler("unary@1.0.0") is None
    assert provider.stops == 1
    await loaded.run_shutdown_hooks()
    assert provider.stops == 1


async def test_shutdown_deadline_does_not_prevent_healthy_sibling_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The configured deadline bounds all cooperative hooks collectively."""
    monkeypatch.setattr(
        "skulk.extensions.loader._EXTENSION_SHUTDOWN_TIMEOUT_SECONDS", 0.05
    )
    slow, healthy = SlowCleanup(), Provider()
    loaded = LoadedExtensions([slow, healthy])
    with anyio.fail_after(7):
        await loaded.run_shutdown_hooks()
    assert slow.stops == healthy.stops == 1


class DynamicProvider(Provider):
    """Publish cached contracts through the same facet as managed owners."""

    def capabilities(self) -> list[CapabilityDescriptor]:
        """Do not register a static descriptor snapshot."""
        return []

    def dynamic_capabilities(self) -> tuple[CapabilityDescriptor, ...]:
        """Updateable descriptors must retain all four executable call modes."""
        return tuple(super().capabilities())


@pytest.mark.parametrize("composed", [False, True])
def test_dynamic_contract_lookup_preserves_all_four_modes(composed: bool) -> None:
    """Live cached stream contracts are discoverable and routed without a reload."""
    provider = DynamicProvider()
    loaded = LoadedExtensions([provider])
    if composed:
        loaded = loaded.with_builtin_extensions([])
    assert len(loaded.capability_descriptors) == 4
    assert loaded.handled_capability_ids() == {"unary"}
    assert loaded.handled_stream_capability_ids() == {
        "server_streaming",
        "client_streaming",
        "bidirectional",
    }
    for mode in ("server_streaming", "client_streaming", "bidirectional"):
        assert loaded.stream_handler(f"{mode}@1.0.0") is not None
        assert loaded.call_handler(f"{mode}@1.0.0") is None
    provider.ready = False
    assert not loaded.capability_descriptors
    assert not loaded.handled_stream_capability_ids()
    assert loaded.stream_handler("bidirectional@1.0.0") is None


def test_dynamic_stream_conflict_does_not_transfer_ownership() -> None:
    """An unavailable duplicate reserves its identity across mode boundaries."""
    first, second = DynamicProvider(), DynamicProvider()
    second.name = "second-dynamic"
    second.ready = False
    assert not LoadedExtensions([first, second]).capability_descriptors
    assert not LoadedExtensions([first, second]).handled_stream_capability_ids()
    builtin = Provider()
    assert (
        len(
            LoadedExtensions([first])
            .with_builtin_extensions([builtin])
            .capability_descriptors
        )
        == 4
    )


async def test_late_dynamic_stream_readiness_advertises_and_withdraws_telemetry() -> None:
    """Peers discover late managed streams; withdrawal and shutdown remove every tag."""
    provider = DynamicProvider()
    provider.ready = False
    loaded = LoadedExtensions([provider])
    tags: set[str] = set()
    changed = asyncio.Event()

    def advertise(tag: str) -> None:
        tags.add(tag)
        changed.set()

    def withdraw(tag: str) -> None:
        tags.discard(tag)
        changed.set()

    async def wait_for(expected: set[str]) -> None:
        async with asyncio.timeout(4):
            while tags != expected:
                changed.clear()
                await changed.wait()

    context = replace(fixture_context(), advertise_capability=advertise, withdraw_capability=withdraw)
    loaded.run_startup_hooks(context)
    try:
        all_modes = {"unary", "server_streaming", "client_streaming", "bidirectional"}
        provider.ready = True
        await wait_for(all_modes)
        provider.ready = False
        await wait_for(set())
        provider.ready = True
        await wait_for(all_modes)
    finally:
        await loaded.run_shutdown_hooks()
    assert tags == set()
