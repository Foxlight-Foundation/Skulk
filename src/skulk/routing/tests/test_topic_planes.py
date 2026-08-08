"""Runtime topic plane census coverage."""

from skulk.routing.topics import TOPIC_PLANE_CENSUS, MessagePlane


def test_every_runtime_topic_has_one_explicit_plane() -> None:
    """The runtime census must remain complete as transport topics evolve."""

    assert {
        "global_events": MessagePlane.Control,
        "local_events": MessagePlane.Control,
        "commands": MessagePlane.Control,
        "election_messages": MessagePlane.Control,
        "authority_messages": MessagePlane.Authority,
        "connection_messages": MessagePlane.Control,
        "download_commands": MessagePlane.Control,
        "state_sync_messages": MessagePlane.Control,
        "telemetry": MessagePlane.Telemetry,
        "data": MessagePlane.Data,
        "provider_data": MessagePlane.Data,
        "realtime_audio": MessagePlane.Data,
        "speech_media": MessagePlane.Data,
        "trace_data": MessagePlane.Data,
        "vision_media": MessagePlane.Data,
    } == TOPIC_PLANE_CENSUS
