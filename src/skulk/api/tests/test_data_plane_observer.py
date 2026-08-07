"""DATA stream lifecycle and timing observability coverage."""

from skulk.api.data_plane import DataPlaneObserver
from skulk.shared.models.model_cards import ModelId
from skulk.shared.types.chunks import DataChunk, TokenChunk
from skulk.shared.types.common import CommandId


def _token_frame(
    command_id: CommandId,
    *,
    sequence: int,
    terminal: bool = False,
) -> DataChunk:
    return DataChunk(
        command_id=command_id,
        kind="completed" if terminal else "chunk",
        sequence=sequence,
        chunk=TokenChunk(
            model=ModelId("mlx-community/test"),
            text="done" if terminal else "hello",
            token_id=sequence,
            usage=None,
            finish_reason="stop" if terminal else None,
        ),
    )


def test_observer_measures_explicit_stream_lifecycle() -> None:
    observer = DataPlaneObserver(
        transport="zenoh",
        reorder_buffer_enabled=False,
    )
    command_id = CommandId("cmd-observed")

    observer.record_received()
    observer.record_dispatched(
        DataChunk(command_id=command_id, kind="started", sequence=0),
        observed_at=10.0,
    )
    observer.record_received()
    observer.record_dispatched(
        _token_frame(command_id, sequence=1),
        observed_at=12.5,
    )
    observer.record_received()
    observer.record_dispatched(
        _token_frame(command_id, sequence=2, terminal=True),
        observed_at=15.0,
    )

    diagnostics = observer.snapshot()
    assert diagnostics.frames_received == 3
    assert diagnostics.frames_dispatched == 3
    assert diagnostics.started_frames == 1
    assert diagnostics.chunk_frames == 1
    assert diagnostics.completed_frames == 1
    assert diagnostics.active_streams == 0
    assert diagnostics.missing_started_streams == 0
    assert diagnostics.missing_terminal_streams == 0
    assert diagnostics.first_byte_samples == 1
    assert diagnostics.first_byte_seconds_average == 2.5
    assert diagnostics.stream_span_samples == 1
    assert diagnostics.stream_span_seconds_average == 2.5


def test_observer_counts_ordering_and_missing_lifecycle_signals() -> None:
    observer = DataPlaneObserver(
        transport="gossipsub",
        reorder_buffer_enabled=True,
    )
    command_id = CommandId("cmd-incomplete")

    observer.record_received()
    observer.record_out_of_order()
    observer.record_duplicate()
    observer.record_skipped_sequences(2)
    observer.record_late()
    observer.record_dispatched(
        _token_frame(command_id, sequence=3),
        observed_at=20.0,
    )
    observer.finalize(command_id)

    diagnostics = observer.snapshot()
    assert diagnostics.out_of_order_frames == 1
    assert diagnostics.duplicate_frames == 1
    assert diagnostics.skipped_sequences == 2
    assert diagnostics.late_frames == 1
    assert diagnostics.missing_started_streams == 1
    assert diagnostics.missing_terminal_streams == 1
    assert diagnostics.active_streams == 0
