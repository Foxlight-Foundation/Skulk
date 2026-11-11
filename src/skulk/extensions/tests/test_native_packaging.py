"""Native reader resource closure without build-machine installer metadata."""

import importlib.metadata
import shutil
from pathlib import Path

from pydantic import JsonValue, TypeAdapter

from skulk._pyinstaller import native_contract_data_files
from skulk.extensions.native_contracts import KERNEL_VERSION, cohort


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
    metadata = next(tmp_path.glob("*.dist-info"))
    package = importlib.metadata.Distribution.at(metadata)
    assert package.version == KERNEL_VERSION
    assert package.metadata["Name"] == "skulk-capability-kernel"
    for license_file in TypeAdapter(list[str]).validate_python(
        package.metadata.get_all("License-File") or ()
    ):
        assert (metadata / "licenses" / license_file).is_file()
    assert not list(tmp_path.rglob("direct_url.json"))
    assert not list(tmp_path.rglob("INSTALLER"))
    assert not list(tmp_path.rglob("RECORD"))
