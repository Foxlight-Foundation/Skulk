"""Real loader cleanup, inherited fences and retained music result ownership."""

import os
import select
import stat
import subprocess
import sys
from pathlib import Path
from typing import Literal

import pytest

from skulk.shared.types.common import CommandId
from skulk.shared.types.native_music import (
    NativeMusicComponent,
    NativeMusicFile,
    NativeMusicWeights,
)
from skulk.worker.runner.ace_step.result_custody import (
    music_result_directory,
    prepare_result_directory,
)
from skulk.worker.runner.ace_step.tests.test_result_custody import write
from skulk.worker.runner.ace_step.workspace import (
    MAX_MUSIC_LOADER_CACHE_BYTES,
    NativeMusicWorkspace,
    music_loader_storage_budget,
)


def weights() -> NativeMusicWeights:
    """Validate a complete small role inventory; no actual model is loaded."""

    def component(
        kind: Literal["base", "text_encoder", "autoencoder"], names: tuple[str, ...]
    ) -> NativeMusicComponent:
        return NativeMusicComponent(
            kind=kind,
            directory="/private/fixture/" + kind,
            files=tuple(
                NativeMusicFile(path=name, size_bytes=8, object_id="sha256:" + "a" * 64)
                for name in names
            ),
        )

    return NativeMusicWeights(
        base=component(
            "base", ("config.json", "model.safetensors", "silence_latent.pt")
        ),
        text_encoder=component(
            "text_encoder",
            (
                "added_tokens.json",
                "chat_template.jinja",
                "config.json",
                "merges.txt",
                "model.safetensors",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
            ),
        ),
        autoencoder=component(
            "autoencoder", ("config.json", "diffusion_pytorch_model.safetensors")
        ),
    )


def loader_files(owner: NativeMusicWorkspace) -> tuple[Path, ...]:
    """Create ordinary partial loader/cache files, never real weights or code."""
    loader = owner.directory / "loader"
    files = (
        loader / "checkpoints/acestep-v15-base/model.safetensors",
        loader / "checkpoints/acestep-v15-base/configuration_acestep_v15.py",
        loader / "checkpoints/Qwen3-Embedding-0.6B/config.json",
        loader / "checkpoints/vae/diffusion_pytorch_model.safetensors",
        loader / "hf/hub/version.txt",
        loader / "modules/transformers_modules/model/__init__.py",
        loader / "torch-cache/kernel/test.bin",
    )
    for path in files:
        # Path.mkdir(parents=True) ignores mode for intermediate directories;
        # the native process instead creates them under its inherited 0077 umask.
        parent = loader
        for name in path.relative_to(loader).parts[:-1]:
            parent /= name
            parent.mkdir(mode=0o700, exist_ok=True)
        write(path, b"partial")
    return files


def retained_result(owner: NativeMusicWorkspace) -> tuple[Path, bytes]:
    """Represent a result whose native writer ended but whose transport has not."""
    directory = music_result_directory(owner.root, CommandId("retained-take"))
    prepare_result_directory(directory, owner.requests_identity)
    original = b"result still in addressed delivery"
    path = directory / "result.wav"
    write(path, original)
    return path, original


def test_parent_observes_and_syncs_all_inodes_before_native_spawn(
    tmp_path: Path,
) -> None:
    """Only a successfully journaled owner can supply native directory identities."""
    owner = NativeMusicWorkspace(tmp_path, weights())
    with pytest.raises(RuntimeError):
        _ = owner.descriptor
    owner.open()
    try:
        for path, identity in (
            (owner.directory, owner.directory_identity),
            (owner.directory / "requests", owner.requests_identity),
            (owner.directory / "loader", owner.loader_identity),
        ):
            facts = path.stat()
            assert (identity.device, identity.inode) == (facts.st_dev, facts.st_ino)
            assert stat.S_IMODE(facts.st_mode) == 0o700
        assert (tmp_path / "workspace-owner.json").stat().st_mode & 0o777 == 0o600
        assert not tuple((owner.directory / "loader").iterdir())
        assert (
            music_loader_storage_budget(weights())
            == 14 * 8 + 3 * 2 * 1024**2 + MAX_MUSIC_LOADER_CACHE_BYTES
        )
        assert owner.retire_loading()
    finally:
        owner.close()


def test_restart_reclaims_loader_but_keeps_original_result_and_receipt(
    tmp_path: Path,
) -> None:
    """A model's restart cannot delete audio whose output receipt is still pending."""
    owner = NativeMusicWorkspace(tmp_path, weights())
    owner.open()
    result, data = retained_result(owner)
    receipt = result.parent.with_name(".retained-take.owner.json")
    journal = receipt.read_bytes()
    requests = owner.requests_identity
    loader_files(owner)
    owner.close()
    resumed = NativeMusicWorkspace(tmp_path, weights())
    resumed.open()
    try:
        assert resumed.requests_identity == requests
        assert not tuple((resumed.directory / "loader").iterdir())
        assert result.read_bytes() == data and receipt.read_bytes() == journal
    finally:
        resumed.close()


@pytest.mark.parametrize("location", ["workspace", "requests", "loader"])
def test_replaced_lifetime_inode_preserves_both_old_and_foreign_trees(
    tmp_path: Path, location: str
) -> None:
    """An old journal cannot authorize recovery of an operator's replacement."""
    owner = NativeMusicWorkspace(tmp_path, weights())
    owner.open()
    result, data = retained_result(owner)
    owner.close()
    path = owner.directory if location == "workspace" else owner.directory / location
    retained = path.with_name("old-" + path.name)
    path.rename(retained)
    path.mkdir(mode=0o700)
    foreign = path / "operator-note.txt"
    write(foreign, b"must survive")
    refused = NativeMusicWorkspace(tmp_path, weights())
    with pytest.raises(ValueError):
        refused.open()
    assert foreign.read_bytes() == b"must survive"
    relocated = (
        retained / result.relative_to(path) if result.is_relative_to(path) else result
    )
    assert relocated.read_bytes() == data


@pytest.mark.parametrize(
    "foreign", ["note.txt", "checkpoints/acestep-v15-base/foreign.pt"]
)
def test_unknown_loader_entry_preserves_all_partial_copies(
    tmp_path: Path, foreign: str
) -> None:
    """The complete view is checked before deletion starts, including failed loads."""
    owner = NativeMusicWorkspace(tmp_path, weights())
    owner.open()
    try:
        partial = loader_files(owner)
        extra = owner.directory / "loader" / foreign
        write(extra, b"must survive")
        assert not owner.retire_loading()
        assert all(path.exists() for path in partial)
        assert extra.read_bytes() == b"must survive"
        extra.unlink()
        assert owner.retire_loading()
    finally:
        owner.close()


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo", "permissive"])
def test_loader_cache_links_and_foreign_modes_remain_reserved(
    tmp_path: Path, kind: str
) -> None:
    """Cache paths cannot grant authority over files outside the owned view."""
    owner = NativeMusicWorkspace(tmp_path, weights())
    owner.open()
    try:
        partial = loader_files(owner)
        extra = owner.directory / "loader/hf/hub/foreign"
        original = tmp_path / "creator-original"
        write(original, b"creator bytes")
        if kind == "symlink":
            extra.symlink_to(original)
        elif kind == "hardlink":
            os.link(original, extra)
        elif kind == "fifo":
            os.mkfifo(extra, 0o600)
        else:
            write(extra)
            extra.chmod(0o644)
        assert not owner.retire_loading()
        assert original.read_bytes() == b"creator bytes"
        assert all(path.exists() for path in partial)
    finally:
        owner.close()


@pytest.mark.parametrize("category", ["checkpoint", "cache"])
def test_over_limit_loading_retains_journal_and_result(
    tmp_path: Path, category: str
) -> None:
    """Unexpected copy/cache extents are preserved without claiming a free budget."""
    owner = NativeMusicWorkspace(tmp_path, weights())
    owner.open()
    try:
        result, data = retained_result(owner)
        partial = loader_files(owner)
        path = partial[0] if category == "checkpoint" else partial[-1]
        with path.open("r+b") as stream:
            stream.truncate(
                9 if category == "checkpoint" else MAX_MUSIC_LOADER_CACHE_BYTES + 1
            )
        assert not owner.retire_loading()
        assert result.read_bytes() == data
        assert (tmp_path / "workspace-owner.json").exists()
    finally:
        owner.close()


def test_failed_unlink_can_retry_without_touching_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Partial cleanup keeps its original receipt and the service owner for retry."""
    owner = NativeMusicWorkspace(tmp_path, weights())
    owner.open()
    try:
        result, data = retained_result(owner)
        loader_files(owner)
        unlink = os.unlink

        def refuse(path: str | os.PathLike[str], *, dir_fd: int | None = None) -> None:
            if path == "model.safetensors":
                raise PermissionError("injected loader unlink failure")
            unlink(path, dir_fd=dir_fd)

        monkeypatch.setattr(os, "unlink", refuse)
        assert not owner.retire_loading()
        assert owner.descriptor >= 0
        assert result.read_bytes() == data
        monkeypatch.undo()
        assert owner.retire_loading()
        assert result.read_bytes() == data
    finally:
        owner.close()


@pytest.mark.parametrize("has_output", [False, True])
def test_missing_receipt_accepts_only_empty_pre_spawn_creation_gap(
    tmp_path: Path, has_output: bool
) -> None:
    """Without a journal native execution cannot have been admitted by this owner."""
    owner = NativeMusicWorkspace(tmp_path, weights())
    owner.open()
    owner.close()
    (tmp_path / "workspace-owner.json").unlink()
    output = owner.directory / "requests/unproven.wav"
    if has_output:
        write(output)
        with pytest.raises(ValueError, match="not empty"):
            owner.open()
        assert output.exists()
    else:
        owner.open()
        owner.close()


def test_child_inherits_actual_service_lock_after_parent_copy_closes(
    tmp_path: Path,
) -> None:
    """A new owner cannot reclaim loader bytes while an orphan child still reads."""
    owner = NativeMusicWorkspace(tmp_path, weights())
    owner.open()
    partial = loader_files(owner)
    process = subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            "import sys;print('ready',flush=True);sys.stdin.buffer.read(1)",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        pass_fds=(owner.descriptor,),
    )
    try:
        assert process.stdout is not None
        assert select.select([process.stdout], [], [], 5)[0]
        assert process.stdout.readline() == b"ready\n"
        owner.close()
        replacement = NativeMusicWorkspace(tmp_path, weights())
        with pytest.raises(BlockingIOError):
            replacement.open()
        assert all(path.exists() for path in partial)
        process.communicate(b"x", timeout=5)
        assert process.returncode == 0
        replacement.open()
        replacement.close()
        assert not any(path.exists() for path in partial)
    finally:
        owner.close()
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        if process.stdin is not None:
            process.stdin.close()
        if process.stdout is not None:
            process.stdout.close()


def test_journal_sync_failure_never_returns_native_admission_facts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed pre-spawn receipt leaves only an empty, explicitly recoverable gap."""
    owner = NativeMusicWorkspace(tmp_path, weights())
    fsync = os.fsync

    def refuse_metadata(descriptor: int) -> None:
        if stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError("injected workspace journal sync failure")
        fsync(descriptor)

    monkeypatch.setattr(os, "fsync", refuse_metadata)
    with pytest.raises(OSError, match="journal sync failure"):
        owner.open()
    with pytest.raises(RuntimeError, match="not prepared"):
        _ = owner.directory_identity
    with pytest.raises(RuntimeError, match="not open"):
        _ = owner.descriptor
    assert not (tmp_path / "workspace-owner.json").exists()
    assert not tuple((owner.directory / "loader").iterdir())
    monkeypatch.undo()
    owner.open()
    owner.close()


@pytest.mark.parametrize("kind", ["malformed", "oversized", "fifo", "symlink"])
def test_unproven_workspace_receipt_never_authorizes_loader_cleanup(
    tmp_path: Path, kind: str
) -> None:
    """Invalid metadata cannot manufacture authority over populated loader bytes."""
    owner = NativeMusicWorkspace(tmp_path, weights())
    owner.open()
    partial = loader_files(owner)
    owner.close()
    receipt = tmp_path / "workspace-owner.json"
    data = receipt.read_bytes()
    receipt.unlink()
    if kind == "malformed":
        write(receipt, b"{")
    elif kind == "oversized":
        write(receipt, data + bytes(8193))
    elif kind == "fifo":
        os.mkfifo(receipt, 0o600)
    else:
        other = tmp_path / "other-owner.json"
        write(other, data)
        receipt.symlink_to(other)
    with pytest.raises((OSError, ValueError)):
        owner.open()
    assert all(path.exists() for path in partial)
