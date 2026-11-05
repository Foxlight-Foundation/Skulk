"""Native ACE artifact truth stays exact without granting an unrelated runner."""

import hashlib
import json
import tomllib
from pathlib import Path
from typing import cast

import loguru
import pytest
from pydantic import ValidationError

from skulk.api.main import API
from skulk.master.placement import PlacementError, place_instance
from skulk.shared.models.model_cards import (
    ModelCard,
    ModelTask,
    model_engine_requirements,
)
from skulk.shared.tests.model_card_fixtures import fixture_card_path
from skulk.shared.topology import Topology
from skulk.shared.types.audio import RealtimeAudioInputFrame
from skulk.shared.types.commands import PlaceInstance
from skulk.shared.types.common import NodeId
from skulk.shared.types.diagnostics import RunnerDiagnosticUpdate
from skulk.shared.types.events import Event, RunnerStatusUpdated
from skulk.shared.types.profiling import MemoryUsage, NodeResources
from skulk.shared.types.tasks import Task, TaskId
from skulk.shared.types.worker.instances import (
    BoundInstance,
    InstanceId,
    InstanceMeta,
    MlxRingInstance,
)
from skulk.shared.types.worker.runners import RunnerFailed, RunnerId, ShardAssignments
from skulk.shared.types.worker.shards import PipelineShardMetadata, Sharding
from skulk.utils.channels import MpReceiver, MpSender, mp_channel
from skulk.worker.runner import bootstrap
from skulk.worker.runner.audio_cpp import runner as audio_cpp_runner
from skulk.worker.runner.bootstrap import (
    _resolve_text_engine,  # pyright: ignore[reportPrivateUsage]
)


def _body() -> dict[str, object]:
    return cast("dict[str, object]", tomllib.loads(
        Path(__file__).with_name("ace-native-card.toml").read_text()
    ))


def _section(body: dict[str, object], *keys: str) -> dict[str, object]:
    result = body
    for key in keys:
        result = cast("dict[str, object]", result[key])
    return result


def test_native_sources_group_shared_repository_without_losing_component_roots() -> None:
    """One pinned companion transaction retains all encoder and VAE data files."""
    card = ModelCard.model_validate(_body())
    sources = card.native_music_artifact_sources()
    assert sorted(len(source.files) for source in sources) == [3, 11]
    assert sum(item.size_bytes for source in sources for item in source.files) == card.storage_size.in_bytes
    for source in sources:
        assert tuple(item.path for item in source.files) == tuple(sorted(item.path for item in source.files))
        bundle = source.download_bundle()
        assert bundle.root is None
        assert bundle.files == source.files
        assert bundle == source.download_bundle()
        assert card.native_music_files_for_source(source.repository, source.revision) == source.files
        assert not card.native_music_files_for_source(source.repository, "f" * 40)
    assert card.music is not None and card.music.native is not None
    assert card.music.native.text_encoder.artifact_bundle.root == "Qwen3-Embedding-0.6B"
    assert card.music.native.autoencoder.artifact_bundle.root == "vae"


def test_native_sources_can_share_the_primary_repository() -> None:
    """A mirror may keep all roles at one immutable source without widening files."""
    body = _body()
    for role in ("text_encoder", "autoencoder"):
        component = _section(body, "music", "native", role)
        component["repo"] = body["source_repository"]
        component["revision"] = body["source_revision"]
    card = ModelCard.model_validate(body)
    sources = card.native_music_artifact_sources()
    assert len(sources) == 1 and len(sources[0].files) == 14
    assert card.artifact_bundle is not None
    assert len(card.artifact_bundle.files) == 3


def test_native_sources_reject_overlapping_role_paths() -> None:
    """Even equal pins must not let separate loader roles alias one configuration."""
    body = _body()
    encoder_root = _section(body, "music", "native", "text_encoder", "artifact_bundle")["root"]
    bundle = _section(body, "music", "native", "autoencoder", "artifact_bundle")
    old_prefix = str(bundle["root"]) + "/"
    bundle["root"] = encoder_root
    for item in cast("list[dict[str, object]]", bundle["files"]):
        item["path"] = str(encoder_root) + "/" + str(item["path"]).removeprefix(old_prefix)
    with pytest.raises(ValidationError, match="roles must not overlap"):
        ModelCard.model_validate(body)


def test_native_source_role_root_cannot_descend_through_another_selected_file() -> None:
    """A structurally named component directory cannot overwrite the base configuration."""
    body = _body()
    component = _section(body, "music", "native", "text_encoder")
    component["repo"] = body["source_repository"]
    component["revision"] = body["source_revision"]
    bundle = _section(component, "artifact_bundle")
    prefix = str(bundle["root"]) + "/"
    bundle["root"] = "config.json"
    for item in cast("list[dict[str, object]]", bundle["files"]):
        item["path"] = "config.json/" + str(item["path"]).removeprefix(prefix)
    with pytest.raises(ValidationError, match="roles must not overlap"):
        ModelCard.model_validate(body)


def test_native_card_retains_generation_editing_and_public_metadata() -> None:
    """All three component roots and both intrinsic tasks survive public projection."""
    card = ModelCard.model_validate(_body())
    assert card.tasks == [ModelTask.TextToMusic, ModelTask.MusicEditing]
    assert card.music is not None and card.music.native is not None
    native = card.music.native
    assert native.editing_modes == ("add_part", "repaint", "continue")
    assert native.text_encoder.artifact_bundle.root == "Qwen3-Embedding-0.6B"
    assert native.autoencoder.artifact_bundle.root == "vae"
    assert card.storage_size.in_bytes == 6336571578
    assert card.artifact_repository == "ACE-Step/acestep-v15-base"
    assert ModelCard.model_validate_json(card.model_dump_json()) == card
    entry = API._model_list_entry(card)  # pyright: ignore[reportPrivateUsage]
    assert entry.music is not None and entry.music.native is not None
    assert entry.music.native.editing_modes == native.editing_modes
    assert entry.music.native.part_classes == native.part_classes
    assert entry.music.native.sample_rate == 48000
    assert entry.music.native.model_dump(by_alias=True)["editing_modes"] == native.editing_modes
    assert entry.engine_support == [] and not card.placement.compatible_backends
    assert not model_engine_requirements(card)


@pytest.mark.parametrize(("name", "expected"), [
    ("audio-cpp/ACE-Step1.5-Turbo-BF16", "642dfe6b85f55e19d67a18dcad600dec46693306ff0e167aa41884b53a1e7cee"),
    ("audio-cpp/MiniMax-Music3-GGUF-Q4", "22e21e664fc0e57355279b52a613008f4cdcfb40ac52043ae088d0023bdfa0a4"),
])
def test_existing_signed_music_identity_is_byte_preserved(name: str, expected: str) -> None:
    """Match hashes measured with the unchanged reader, including omitted defaults."""
    card = ModelCard.model_validate(tomllib.loads(fixture_card_path(name).read_text()))
    assert card.music is not None and "native" not in card.music.model_dump()
    assert "native" not in card.music.model_dump(by_alias=True)
    content = json.dumps(card.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(content).hexdigest() == expected
    entry = API._model_list_entry(card)  # pyright: ignore[reportPrivateUsage]
    assert entry.music is not None and "native" not in entry.music.model_dump()
    assert model_engine_requirements(card) == frozenset({"audio_cpp"})


@pytest.mark.parametrize(("path", "value"), [
    (("tasks",), ["TextToMusic"]),
    (("tasks",), ["MusicEditing"]),
    (("tasks",), ["TextToMusic", "MusicEditing", "MusicEditing"]),
    (("tasks",), ["TextToMusic", "MusicEditing", "TextGeneration"]),
    (("music", "native"), None),
    (("music", "native", "variant"), "turbo"),
    (("music", "native", "sample_rate"), 44100),
    (("music", "native", "audio_channels"), 1),
    (("music", "native", "editing_modes"), ["complete"]),
    (("music", "native", "editing_modes"), ["repaint", "repaint"]),
    (("music", "native", "editing_modes"), []),
    (("music", "native", "part_classes"), ["cello"]),
    (("music", "native", "part_classes"), ["strings", "strings"]),
    (("music", "native", "server_path"), "/tmp/model"),
    (("music", "native", "text_encoder", "revision"), "main"),
    (("music", "native", "text_encoder", "repo"), "https://example.com/model"),
    (("music", "native", "text_encoder", "license"), ""),
    (("music", "native", "autoencoder", "revision"), "a" * 40),
    (("gguf_file",), "turbo.gguf"),
    (("source_revision",), None),
    (("trust_remote_code",), True),
    (("storage_size",), {"in_bytes": 1}),
    (("n_layers",), 1),
    (("hidden_size",), 1),
])
def test_native_card_rejects_inconsistent_or_unbounded_truth(path: tuple[str, ...], value: object) -> None:
    """Refuse missing tasks, unsupported variants, arbitrary paths and mutable origins."""
    body = _body()
    _section(body, *path[:-1])[path[-1]] = value
    with pytest.raises(ValidationError):
        ModelCard.model_validate(body)


@pytest.mark.parametrize("role", ["base", "text_encoder", "autoencoder"])
@pytest.mark.parametrize("mutation", ["missing", "executable", "identity", "weak_tensor_identity"])
def test_each_component_requires_complete_identified_non_executable_files(role: str, mutation: str) -> None:
    """Reject incomplete closure even when aggregate byte accounting is consistent."""
    body = _body()
    native = _section(body, "music", "native")
    bundle = _section(body, "artifact_bundle") if role == "base" else _section(native, role, "artifact_bundle")
    files = cast("list[dict[str, object]]", bundle["files"])
    if mutation == "missing":
        files.pop()
    elif mutation == "executable":
        prefix = str(bundle["root"]) + "/" if bundle.get("root") else ""
        files.append({"path": prefix + "modeling.py", "size_bytes": 1, "object_id": "git-sha1:" + "a" * 40})
    else:
        tensor = next(file for file in files if str(file["path"]).endswith(".safetensors"))
        tensor["object_id"] = None if mutation == "identity" else "git-sha1:" + "a" * 40
    bundle["download_size"] = sum(cast("int", file["size_bytes"]) for file in files)
    all_bundles = [_section(body, "artifact_bundle"), _section(native, "text_encoder", "artifact_bundle"), _section(native, "autoencoder", "artifact_bundle")]
    body["storage_size"] = {"in_bytes": sum(cast("int", item["download_size"]) for item in all_bundles)}
    with pytest.raises(ValidationError, match="native music"):
        ModelCard.model_validate(body)


def test_native_mirror_can_preserve_roles_at_different_loader_roots() -> None:
    """Compatibility follows exact selected files, not a repository permission list."""
    body = _body()
    for role, root in (("base", "base"), ("text_encoder", "encoder"), ("autoencoder", "decoder")):
        bundle = _section(body, "artifact_bundle") if role == "base" else _section(body, "music", "native", role, "artifact_bundle")
        old_root = str(bundle["root"]) + "/" if bundle.get("root") else ""
        bundle["root"] = root
        for item in cast("list[dict[str, object]]", bundle["files"]):
            item["path"] = root + "/" + str(item["path"]).removeprefix(old_root)
    body["source_repository"] = "a-mirror/native-base"
    card = ModelCard.model_validate(body)
    assert card.music is not None and card.music.native is not None
    assert card.music.native.text_encoder.artifact_bundle.root == "encoder"


@pytest.mark.parametrize("backend", ["audio_cpp-cpu", "audio_cpp-vulkan", "audio_cpp-cuda", "mlx-metal"])
def test_existing_engine_cannot_place_native_artifacts(backend: str) -> None:
    """Actual ordinary placement cannot confuse native base with an ACE Turbo GGUF."""
    body = _body()
    body["placement"] = {"compatible_backends": [backend]}
    card = ModelCard.model_validate(body)
    node = NodeId("native-test")
    topology = Topology()
    topology.add_node(node)
    with pytest.raises(PlacementError):
        place_instance(
            PlaceInstance(model_card=card, sharding=Sharding.Pipeline, instance_meta=InstanceMeta.MlxRing, min_nodes=1),
            topology, {}, {node: MemoryUsage.from_bytes(ram_total=64 * 1024**3, ram_available=64 * 1024**3, swap_total=0, swap_available=0)}, {},
            node_resources={node: NodeResources(backends=frozenset({backend}))},
        )


@pytest.mark.parametrize("backend", [None, "audio_cpp-cpu"])
def test_worker_refuses_native_artifacts_instead_of_falling_back_to_mlx(backend: str | None) -> None:
    """Manual and stamped incompatible assignments fail with an actionable loader error."""
    card = ModelCard.model_validate(_body())
    node, runner = NodeId("native-test"), RunnerId("native-test")
    instance = MlxRingInstance(
        instance_id=InstanceId("native-test"),
        shard_assignments=ShardAssignments(model_id=card.model_id, node_to_runner={node: runner}, runner_to_shard={runner: PipelineShardMetadata(
            model_card=card, device_rank=0, world_size=1, start_layer=0, end_layer=24, n_layers=24, resolved_backend=backend,
        )}), hosts_by_node={}, ephemeral_port=50000,
    )
    with pytest.raises(RuntimeError, match="native music runner"):
        _resolve_text_engine(BoundInstance(instance=instance, bound_runner_id=runner, bound_node_id=node))


def _skip_process_setup(*_arguments: object, **_keywords: object) -> None:
    """Isolate OS/watchdog/diagnostic effects while exercising actual dispatch."""


@pytest.mark.parametrize(("native", "backend"), [
    (True, None), (True, "audio_cpp-cpu"), (False, "audio_cpp-cpu"),
])
def test_actual_entrypoint_refuses_native_before_audio_cpp_dispatch(
    monkeypatch: pytest.MonkeyPatch, native: bool, backend: str | None,
) -> None:
    """Unprepared native paths fail; an existing GGUF still dispatches unchanged."""
    card = ModelCard.model_validate(_body() if native else tomllib.loads(
        fixture_card_path("audio-cpp/ACE-Step1.5-Turbo-BF16").read_text()
    ))
    node, runner = NodeId("entrypoint-node"), RunnerId("entrypoint-runner")
    bound = BoundInstance(
        instance=MlxRingInstance(
            instance_id=InstanceId("entrypoint-instance"),
            shard_assignments=ShardAssignments(
                model_id=card.model_id, node_to_runner={node: runner},
                runner_to_shard={runner: PipelineShardMetadata(
                    model_card=card, device_rank=0, world_size=1,
                    start_layer=0, end_layer=card.n_layers, n_layers=card.n_layers,
                    resolved_backend=backend,
                )},
            ), hosts_by_node={}, ephemeral_port=50000,
        ), bound_node_id=node, bound_runner_id=runner,
    )
    events: list[Event] = []
    dispatched: list[str] = []

    class ObservedAudioCppRunner:
        """Observe only the specialized dispatch, without loading a real model."""

        def __init__(
            self, _bound: BoundInstance, _events: MpSender[Event],
            _tasks: MpReceiver[Task], _cancellations: MpReceiver[TaskId],
        ) -> None:
            dispatched.append("constructed")

        def main(self) -> None:
            """Record actual runner-loop entry without inference side effects."""
            dispatched.append("main")

    event_sender, event_receiver = mp_channel[Event]()
    diagnostic_sender, diagnostic_receiver = mp_channel[RunnerDiagnosticUpdate]()
    task_sender, task_receiver = mp_channel[Task]()
    cancel_sender, cancel_receiver = mp_channel[TaskId]()
    audio_sender, audio_receiver = mp_channel[RealtimeAudioInputFrame]()
    monkeypatch.setattr(event_sender, "send", events.append)
    monkeypatch.setattr(audio_cpp_runner, "Runner", ObservedAudioCppRunner)
    monkeypatch.setattr(bootstrap.signal, "signal", _skip_process_setup)
    monkeypatch.setattr(bootstrap.resource, "setrlimit", _skip_process_setup)
    monkeypatch.setenv("MLX_METAL_FAST_SYNCH", "0")
    monkeypatch.setattr(bootstrap, "logger", bootstrap.logger)
    for name in (
        "install_stack_dump_signal", "_install_parent_death_watchdog",
        "configure_runner_diagnostics", "record_runner_phase", "_release_metal_resources",
    ):
        monkeypatch.setattr(bootstrap, name, _skip_process_setup)
    try:
        bootstrap.entrypoint(
            bound, event_sender, diagnostic_sender, task_receiver,
            cancel_receiver, audio_receiver, loguru.logger,
        )
        failures = [event.runner_status for event in events if isinstance(event, RunnerStatusUpdated)]
        if native:
            assert dispatched == []
            assert len(failures) == 1
            assert isinstance(failures[0], RunnerFailed)
            assert failures[0].error_message == "native music placement has no prepared worker context"
        else:
            assert dispatched == ["constructed", "main"]
            assert failures == []
    finally:
        event_sender.close()
        diagnostic_sender.close()
        task_sender.close()
        cancel_sender.close()
        audio_sender.close()
        event_receiver.close()
        diagnostic_receiver.close()
        task_receiver.close()
        cancel_receiver.close()
        audio_receiver.close()
        event_sender.join()
        diagnostic_sender.join()
        task_sender.join()
        cancel_sender.join()
        audio_sender.join()
