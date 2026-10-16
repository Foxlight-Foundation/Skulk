"""Native image output ownership requires one pipeline rank."""

import json

import pytest

from skulk.master.placement import PlacementError, place_instance
from skulk.shared.models.model_cards import ModelCard
from skulk.shared.topology import Topology
from skulk.shared.types.commands import PlaceInstance
from skulk.shared.types.worker.instances import InstanceMeta
from skulk.shared.types.worker.shards import Sharding


@pytest.mark.parametrize("sharding,minimum_nodes", [(Sharding.Tensor, 1), (Sharding.Pipeline, 2)])
def test_native_image_rejects_tensor_or_multinode_before_placement(sharding: Sharding, minimum_nodes: int) -> None:
    """Reject a shape whose image stream has no primary pipeline output owner."""
    card = ModelCard.model_validate_json(json.dumps({
        "model_id": "test/native", "source_revision": "a" * 40,
        "gguf_file": "model.gguf", "storage_size": {"in_bytes": 1},
        "n_layers": 1, "hidden_size": 1, "supports_tensor": False, "tasks": ["TextToImage"],
        "image": {"family": "sdxl", "default_steps": 20, "components": [{
            "role": "model", "repository": "test/native", "revision": "a" * 40,
            "entry_file": "model.gguf", "files": [{"path": "model.gguf", "size_bytes": 1, "object_id": "sha256:" + "b" * 64}],
        }]},
    }))
    command = PlaceInstance(model_card=card, sharding=sharding, min_nodes=minimum_nodes, instance_meta=InstanceMeta.MlxRing)
    with pytest.raises(PlacementError, match="single-node pipeline"):
        place_instance(command, Topology(), {}, {}, {})
