"""Four new output roles cannot mutate the existing video/music assembly."""

import pytest

from skulk.api.main import API
from skulk.routing.output_media import OutputMediaPacket, OutputMediaPurpose
from skulk.shared.types.common import CommandId, ModelId, NodeId


@pytest.mark.parametrize("purpose", [
    "separation_drums", "separation_bass", "separation_other", "separation_vocals",
])
async def test_stem_open_is_refused_before_touching_video_music_state(
    monkeypatch: pytest.MonkeyPatch, purpose: OutputMediaPurpose,
) -> None:
    """A reused command ID never chooses another family's storage or lifecycle."""
    failures: list[OutputMediaPacket] = []

    async def capture(_self: API, packet: OutputMediaPacket) -> None:
        failures.append(packet)

    monkeypatch.setattr(API, "_send_output_media_terminal", capture)
    # Deliberately provide no video/music members: the guard must run before
    # reading or mutating either family's stores, deadlines or pending packets.
    api = object.__new__(API)
    packet = OutputMediaPacket(source_node=NodeId("producer"), target_node=NodeId("api"),
        command_id=CommandId("same-command"), model=ModelId("adefossez/HTDemucs"),
        purpose=purpose, sequence=0, kind="opened", total_chunks=1, total_bytes=128,
        content_type="audio/wav")
    await api._apply_output_media_packet(packet)  # pyright: ignore[reportPrivateUsage]
    assert len(failures) == 1
    failure = failures[0]
    assert failure.kind == "transport_failed" and failure.purpose == purpose
    assert (failure.source_node, failure.target_node) == (packet.target_node, packet.source_node)
    assert failure.command_id == packet.command_id
    assert not vars(api)
