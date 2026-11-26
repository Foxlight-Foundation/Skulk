"""Each node's saved engine orders: written from a caught-up view, read back at a cold start."""

from pathlib import Path

import pytest

from skulk.engines import order_record
from skulk.engines.order_record import (
    ModelEngineOrderRecorder,
    orders_are_writable,
    read_model_engine_orders,
    write_model_engine_orders,
)
from skulk.shared.types.common import ModelId
from skulk.shared.types.state import State

MODEL = ModelId("test/text")
ORDER = ("tensorfold-rocm", "llama_cpp-vulkan")


def _caught_up(orders: dict[ModelId, tuple[str, ...]]) -> State:
    return State(model_engine_orders=orders, last_event_applied_idx=0)


def test_no_file_reads_as_nothing_saved(tmp_path: Path) -> None:
    assert read_model_engine_orders(tmp_path / "orders.json") is None


def test_saved_orders_read_back_exactly(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "orders.json"

    write_model_engine_orders(path, {MODEL: ORDER, ModelId("test/other"): ("mlx",)})

    assert read_model_engine_orders(path) == {
        MODEL: ORDER,
        ModelId("test/other"): ("mlx",),
    }
    assert not list(path.parent.glob(".*.tmp"))


def test_an_empty_save_reads_back_empty_not_missing(tmp_path: Path) -> None:
    path = tmp_path / "orders.json"

    write_model_engine_orders(path, {})

    assert read_model_engine_orders(path) == {}


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        '{"formatVersion": 2, "orders": {}}',
        '{"formatVersion": 1, "orders": {"test/text": ["Not A Tag"]}}',
        '{"formatVersion": 1, "orders": {"test/text": []}}',
        '{"formatVersion": 1, "orders": {}, "surprise": true}',
    ],
)
def test_a_damaged_or_malformed_file_is_ignored(tmp_path: Path, content: str) -> None:
    path = tmp_path / "orders.json"
    path.write_text(content)

    assert read_model_engine_orders(path) is None


def test_a_view_that_applied_no_event_may_not_replace_the_file() -> None:
    assert not orders_are_writable(None)
    assert not orders_are_writable(State())
    assert orders_are_writable(State(last_event_applied_idx=0))


async def test_a_node_that_never_held_an_order_writes_no_file(tmp_path: Path) -> None:
    path = tmp_path / "orders.json"
    recorder = ModelEngineOrderRecorder(
        state_provider=lambda: _caught_up({}), path=path
    )

    assert await recorder.record_once() is False
    assert not path.exists()


async def test_the_file_follows_the_view(tmp_path: Path) -> None:
    path = tmp_path / "orders.json"
    view = _caught_up({MODEL: ORDER})
    recorder = ModelEngineOrderRecorder(state_provider=lambda: view, path=path)

    assert await recorder.record_once() is True
    assert read_model_engine_orders(path) == {MODEL: ORDER}
    assert await recorder.record_once() is False

    view = _caught_up({})
    assert await recorder.record_once() is True
    assert read_model_engine_orders(path) == {}


async def test_a_node_still_catching_up_keeps_its_saved_orders(tmp_path: Path) -> None:
    path = tmp_path / "orders.json"
    write_model_engine_orders(path, {MODEL: ORDER})
    recorder = ModelEngineOrderRecorder(state_provider=lambda: State(), path=path)

    assert await recorder.record_once() is False
    assert read_model_engine_orders(path) == {MODEL: ORDER}


async def test_a_recorder_started_over_a_current_file_does_not_rewrite_it(
    tmp_path: Path,
) -> None:
    path = tmp_path / "orders.json"
    write_model_engine_orders(path, {MODEL: ORDER})
    recorder = ModelEngineOrderRecorder(
        state_provider=lambda: _caught_up({MODEL: ORDER}), path=path
    )

    assert await recorder.record_once() is False


async def test_a_disk_that_refuses_the_write_does_not_stop_the_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(_path: Path, _orders: object) -> None:
        raise OSError("read-only file system")

    monkeypatch.setattr(order_record, "write_model_engine_orders", refuse)
    recorder = ModelEngineOrderRecorder(
        state_provider=lambda: _caught_up({MODEL: ORDER}),
        path=tmp_path / "orders.json",
    )

    assert await recorder.record_once() is False
    # Nothing was saved, so the next pass tries again.
    monkeypatch.undo()
    assert await recorder.record_once() is True
