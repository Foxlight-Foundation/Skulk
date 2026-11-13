"""Native reader resource closure without build-machine installer metadata."""

import importlib.metadata
import shutil
from pathlib import Path

import pytest
from pydantic import JsonValue, TypeAdapter

import skulk._pyinstaller
from skulk._pyinstaller import native_contract_data_files
from skulk.extensions.native_call_contracts import call_cohort
from skulk.extensions.native_contracts import cohort
from skulk.extensions.native_inference_contracts import (
    INFERENCE_KERNEL_VERSION,
    inference_cohort,
)


def test_native_frozen_resource_closure_preserves_reader_identity(
    tmp_path: Path,
) -> None:
    """Collected data retain real cohort, version and licenses; no local URLs leak."""
    resources = native_contract_data_files()
    for source, destination in resources:
        target = tmp_path / destination
        target.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target / Path(source).name)
    observed = TypeAdapter(dict[str, JsonValue]).validate_json(
        (tmp_path / "skulk/extensions/native-cohort.json").read_bytes()
    )
    assert observed == cohort()
    calls = TypeAdapter(dict[str, JsonValue]).validate_json(
        (tmp_path / "skulk/extensions/native-call-cohort.json").read_bytes()
    )
    assert calls == call_cohort()
    inference = TypeAdapter(dict[str, JsonValue]).validate_json(
        (tmp_path / "skulk/extensions/native-inference-cohort.json").read_bytes()
    )
    assert inference == inference_cohort()
    metadata = next(tmp_path.glob("*.dist-info"))
    package = importlib.metadata.Distribution.at(metadata)
    assert package.version == INFERENCE_KERNEL_VERSION
    assert package.metadata["Name"] == "skulk-capability-kernel"
    for license_file in TypeAdapter(list[str]).validate_python(
        package.metadata.get_all("License-File") or ()
    ):
        assert (metadata / "licenses" / license_file).is_file()
    assert not list(tmp_path.rglob("direct_url.json"))
    assert not list(tmp_path.rglob("INSTALLER"))
    assert not list(tmp_path.rglob("RECORD"))


@pytest.mark.parametrize(
    "missing",
    ["native-cohort.json", "native-call-cohort.json", "native-inference-cohort.json"],
)
def test_native_build_refuses_either_missing_interpretation_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    """A frozen host retains every issued closure pin; no build-time fallback."""
    package = tmp_path / "skulk"
    resources = package / "extensions"
    resources.mkdir(parents=True)
    for name in (
        "native-cohort.json",
        "native-call-cohort.json",
        "native-inference-cohort.json",
    ):
        if name != missing:
            (resources / name).write_text("{}")
    monkeypatch.setattr(
        skulk._pyinstaller, "__file__", str(package / "_pyinstaller/__init__.py")
    )
    with pytest.raises(FileNotFoundError, match="cohort resource is missing"):
        native_contract_data_files()
