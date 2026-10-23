"""Frozen data closure includes exact converter evidence, without installer state."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import foxlight_model_preservation.peer_install as native
import foxlight_model_preservation.pmf_preservation as adapter
import pytest
from foxlight_model_preservation.signing import load_key

from skulk._pyinstaller import preservation_data_files
from skulk.store.preservation_export import preserve_installed
from skulk.store.tests.test_preservation_export import make_preservation_installation


def test_collected_frozen_resources_support_full_pmf_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen = tmp_path / "frozen"
    files = preservation_data_files()
    assert not any(
        Path(source).name in {"direct_url.json", "INSTALLER"} for source, _ in files
    )
    for source, target in files:
        selected = frozen / target
        selected.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, selected / Path(source).name)
    source = frozen / "foxlight_model_preservation/pmf_preservation.py"
    assert source.is_file()
    expected = hashlib.sha256(source.read_bytes()).hexdigest()

    # Model the resources visible beside PYZ modules, independently of the
    # source module's development checkout. The real frozen smoke is separate.
    def frozen_files(_package: str) -> Path:
        return frozen / "foxlight_model_preservation"

    monkeypatch.setattr(adapter, "files", frozen_files)
    monkeypatch.setattr(native, "files", frozen_files)
    assert native.runtime_plan().artifact.executable_bytes > 0
    assert (frozen / "foxlight_model_preservation/resources/kubo-NOTICE.txt").is_file()
    installed = make_preservation_installation(tmp_path)
    key = load_key(installed.key, None)
    identity = preserve_installed(installed.primary, installed.archive, (key,))
    portable = tmp_path / "portable"
    receipt = adapter.import_preservation(
        installed.archive, identity, installed.trust, portable, (key,)
    )
    assert receipt.adapter_source_sha256 == expected
    reverse = adapter.export_preservation(
        portable,
        receipt.output_identity,
        installed.trust,
        tmp_path / "reverse",
        source_trust=installed.trust,
        expected_source_manifest_id=identity,
    )
    assert (
        reverse.output_identity == identity
        and reverse.adapter_source_sha256 == expected
    )
