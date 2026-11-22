"""Inference registration preserves explicit new identities without advertising effects."""

import json
from pathlib import Path

from skulk.extensions.capabilities import descriptor_revision
from skulk.extensions.installed_capabilities import InstalledContract
from skulk.extensions.native_inference_contracts import InferenceManifest
from skulk.extensions.runtime_files import write_private
from skulk.extensions.runtime_manager import installed_release
from skulk.extensions.tests.test_inference_runtime import inference_fixture
from skulk.extensions.tests.test_native_runtime import OBJECT, sign


def test_installed_inference_registration_keeps_full_native_identity(
    tmp_path: Path,
) -> None:
    """Display metadata reads release3/runtime5; this is not ready capacity or caller admission."""
    payload, key, _, _ = inference_fixture(tmp_path / "fixture")
    path = tmp_path / "staged.json"
    write_private(path, sign(payload, key, b"skulk.runtime/5\0"))
    release = installed_release(path)
    assert release is not None and release.registration is not None
    wrapper = OBJECT.validate_python(
        OBJECT.validate_python(payload["release"], strict=True)["manifest"], strict=True
    )
    manifest = InferenceManifest.model_validate(wrapper)
    descriptor = manifest.descriptors[0]
    summary = release.registration.capabilities[0]
    assert summary.qualified_id == descriptor.qualified_id
    assert summary.revision == descriptor.revision == descriptor_revision(descriptor)
    assert len(summary.revision) == 64
    assert (
        descriptor.behavior == "inference" and descriptor.processing_domain == "local"
    )
    assert release.registration.configuration_required


def test_full_installed_contract_has_an_explicit_inference_view(tmp_path: Path) -> None:
    """No legacy descriptor validator or truncated revision stands in for portable2."""
    payload, _, _, _ = inference_fixture(tmp_path / "fixture")
    wrapper = OBJECT.validate_python(
        OBJECT.validate_python(payload["release"], strict=True)["manifest"], strict=True
    )
    descriptor = InferenceManifest.model_validate(wrapper).descriptors[0]
    observed = InstalledContract.model_validate_json(
        json.dumps(
            {
                "plugin_id": "managed.inference",
                "selected_digest": "a" * 64,
                "descriptor": descriptor.model_dump(mode="json"),
                "revision": descriptor.revision,
            }
        )
    )
    assert observed.descriptor == descriptor
    assert observed.revision == descriptor_revision(observed.descriptor)
