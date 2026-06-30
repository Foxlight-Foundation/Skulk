from skulk.shared.apply import apply_staged_model_evicted
from skulk.shared.models.model_cards import ModelId
from skulk.shared.tests.conftest import get_pipeline_shard_metadata
from skulk.shared.types.common import NodeId
from skulk.shared.types.events import StagedModelEvicted
from skulk.shared.types.memory import Memory
from skulk.shared.types.state import State
from skulk.shared.types.worker.downloads import DownloadCompleted
from skulk.worker.tests.constants import MODEL_A_ID, MODEL_B_ID


def _completed(model_id: ModelId, node: str) -> DownloadCompleted:
    return DownloadCompleted(
        node_id=NodeId(node),
        shard_metadata=get_pipeline_shard_metadata(
            model_id, device_rank=0, world_size=1
        ),
        total=Memory(),
    )


def test_staged_model_evicted_drops_model_across_nodes() -> None:
    # node-1 staged both models; node-2 staged only the evicted one.
    a1 = _completed(MODEL_A_ID, "node-1")
    b1 = _completed(MODEL_B_ID, "node-1")
    a2 = _completed(MODEL_A_ID, "node-2")
    state = State(downloads={NodeId("node-1"): [a1, b1], NodeId("node-2"): [a2]})

    new_state = apply_staged_model_evicted(
        StagedModelEvicted(model_id=MODEL_A_ID), state
    )

    # Model A's entries are gone everywhere; node-2 (left empty) drops its key;
    # node-1 keeps the unrelated model B.
    assert new_state.downloads == {NodeId("node-1"): [b1]}


def test_staged_model_evicted_no_match_returns_state_unchanged() -> None:
    b1 = _completed(MODEL_B_ID, "node-1")
    state = State(downloads={NodeId("node-1"): [b1]})

    new_state = apply_staged_model_evicted(
        StagedModelEvicted(model_id=MODEL_A_ID), state
    )

    # Nothing staged for the evicted model -> the exact same state object back.
    assert new_state is state
