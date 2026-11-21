"""Native music uses the actual bounded original owner without separator tasks."""

from pathlib import Path

import pytest

from skulk.shared.tests.music_edit_fixtures import music_edit_fixture
from skulk.shared.types.common import CommandId, NodeId
from skulk.shared.types.tasks import MusicEditing, TaskStatus
from skulk.shared.types.worker.instances import InstanceId
from skulk.utils.runtime_files import RuntimeLock
from skulk.worker.separation_custody import SeparationSourceCustody
from skulk.worker.separation_sources import SeparationSourceStore


def task(identifier: str = "edit") -> tuple[MusicEditing, bytes]:
    """Return a real strict music task and its exact encoded source."""
    job, _, original, _ = music_edit_fixture(identifier)
    return MusicEditing(
        command_id=job.id, owner_node=NodeId("owner"),
        instance_id=InstanceId("native-music"), task_params=job.params,
    ), original


def test_native_edit_uses_source_store_with_its_actual_strict_task(tmp_path: Path) -> None:
    """Verified PCM is preserved byte for byte; no recast or path enters State."""
    admitted, original = task()
    store = SeparationSourceStore[MusicEditing](tmp_path / "inputs")
    try:
        store.admit(admitted)
        store.admit(admitted.model_copy(update={"task_status": TaskStatus.Running}))
        store.append(admitted.command_id, 1, original)
        path = store.complete(admitted.command_id)
        assert path is not None and path.read_bytes() == original
        assert path.stat().st_mode & 0o077 == 0
        assert store.ready_path(admitted.command_id) == path
        foreign = admitted.model_copy(update={"owner_node": NodeId("other-api")})
        with pytest.raises(ValueError, match="changed"):
            store.admit(foreign)
    finally:
        store.close()
    assert not path.exists()


def test_music_recovery_uses_injected_result_custody_after_reader_drain(tmp_path: Path) -> None:
    """ACE recovery cannot accidentally apply Demucs file-layout cleanup rules."""
    key = "e" * 64
    root, readers = tmp_path / "inputs", tmp_path / "readers"
    drained = False
    calls: list[tuple[Path, CommandId]] = []

    def reclaim(directory: Path, command: CommandId) -> bool:
        calls.append((directory, command))
        return drained

    def open_store() -> SeparationSourceStore[MusicEditing]:
        return SeparationSourceStore[MusicEditing](
            root,
            custody=SeparationSourceCustody[MusicEditing](
                root, readers, lambda _task: key, reclaim_result=reclaim,
            ),
        )

    admitted, original = task()
    first = open_store()
    first.admit(admitted)
    first.append(admitted.command_id, 1, original)
    path = first.complete(admitted.command_id)
    assert path is not None
    reader = RuntimeLock(readers / key, name="attachment.lock")
    first.detach()
    reopened = open_store()
    try:
        assert reopened.unresolved_source_count() == 1
        assert reopened.recover_stale() == 0 and path.read_bytes() == original
        assert calls == []
        reader.close()
        assert reopened.recover_stale() == 0 and path.read_bytes() == original
        assert calls == [(readers / key, admitted.command_id)]
        drained = True
        assert reopened.recover_stale() == 1
        assert not path.exists() and reopened.unresolved_source_count() == 0
    finally:
        reader.close()
        reopened.close()
