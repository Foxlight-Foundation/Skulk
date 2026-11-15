"""Trusted adapter selections bind fresh native generations without granting effects."""

import json
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest

from skulk.api.tests.test_inference_access import application
from skulk.extensions.caller_identity import AuthenticatedOwner
from skulk.extensions.inference_callers import AuthenticatedInferenceCaller
from skulk.extensions.native_host_protocol import Received
from skulk.extensions.native_inference_authority import (
    InferenceAccessPolicy,
    PrincipalAccess,
)
from skulk.extensions.native_inference_calls import NativeInferenceCalls
from skulk.extensions.native_inference_contracts import (
    CompositionRecord,
    InferenceDescriptor,
    InferenceProfiles,
)
from skulk.extensions.native_inference_hosts import InferenceHosts, InferenceListing
from skulk.extensions.native_inference_streams import NativeInferenceStream
from skulk.extensions.runtime_attachment import ServiceConnection
from skulk.extensions.runtime_files import write_private
from skulk.shared.types.common import NodeId


def listing(streaming: bool, change: str = "same") -> InferenceListing:
    """Admit real compiled descriptors independently of test dispatch effects."""
    descriptor = InferenceDescriptor.model_validate_json(
        json.dumps(
            {
                "contract_profile": "skulk.portable/2",
                "profile_digest": InferenceProfiles.qualified().portable,
                "id": "qualification.infer",
                "version": "1.0.0",
                "title": "Generation binding",
                "description": "Trusted selection qualification",
                "behavior": "read" if change == "behavior" else "inference",
                "processing_domain": "local" if change == "domain" else "external",
                "io_mode": "server_streaming" if streaming else "unary",
                "input_schema": {"type": "object"},
                "output_chunk_schema": {"type": "object"} if streaming else None,
                "annotations": {
                    "generation": change if change == "descriptor" else "a"
                },
            }
        )
    )
    return InferenceListing.model_validate_json(
        json.dumps(
            {
                "plugin_id": "managed.qualification",
                "runtime_digest": ("c" if change == "runtime" else "b") * 64,
                "incarnation": ("d" if change == "incarnation" else "a") * 32,
                "target": {
                    "node_id": "replacement" if change == "node" else "application",
                    "capability_id": descriptor.id,
                    "version": descriptor.version,
                    "descriptor_revision": descriptor.revision,
                    "behavior": descriptor.behavior,
                    "processing_domain": descriptor.processing_domain,
                },
                "descriptor": descriptor.model_dump(mode="json"),
                "inference_seconds": 10 if change == "ceiling" else None,
            }
        )
    )


class ReachedDispatchError(RuntimeError):
    """Signal actual composed-context admission before the test transport boundary."""


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "change",
    [
        "same",
        "runtime",
        "incarnation",
        "node",
        "descriptor",
        "behavior",
        "domain",
        "ceiling",
    ],
)
async def test_selection_refuses_replacement_even_for_enrolled_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    streaming: bool,
    change: str,
) -> None:
    """Fresh discovery cannot silently replace a selected owner-authorized target."""
    application(tmp_path)
    service = NativeInferenceCalls(
        ServiceConnection(manager_root=str(tmp_path), profile_id="a" * 32),
        "access-core",
    )
    selected, observed = listing(streaming), listing(streaming, change)
    caller = service.callers.owner(AuthenticatedOwner(NodeId("access-core")))
    policy = InferenceAccessPolicy(
        revision="f" * 64,
        enabled=True,
        callers=(
            PrincipalAccess(principal=caller.principal, grants=(observed.grant(),)),
        ),
    )
    write_private(
        tmp_path / "native-inference-policy.json", policy.model_dump_json().encode()
    )
    dispatched = False

    async def directory(
        _hosts: InferenceHosts, *, streaming: bool = False
    ) -> tuple[InferenceListing, ...]:
        """Inject only observations, preserving real compiled/context admission."""
        assert streaming == (observed.descriptor.io_mode != "unary")
        return (observed,)

    async def dispatch(
        _service: NativeInferenceCalls,
        admitted: InferenceListing,
        _context: CompositionRecord,
        _caller: AuthenticatedInferenceCaller,
        _payload: bytes,
        _expiry: float | None,
        _consume: Callable[[NativeInferenceStream], Awaitable[None]] | None = None,
        _caller_guard: Callable[[], None] | None = None,
    ) -> Received:
        """Observe whether exact admission reaches its independently tested transport."""
        nonlocal dispatched
        assert admitted == observed
        dispatched = True
        raise ReachedDispatchError

    async def consume(_stream: NativeInferenceStream) -> None:
        """This transport fixture must stop before provider streaming begins."""
        raise AssertionError("test transport unexpectedly ran")

    monkeypatch.setattr(InferenceHosts, "directory", directory)
    monkeypatch.setattr(NativeInferenceCalls, "_dispatch", dispatch)
    try:
        with pytest.raises(
            ReachedDispatchError if change == "same" else ValueError,
            match=None if change == "same" else "selected destination changed",
        ):
            if streaming:
                await service.invoke_stream(
                    caller,
                    "managed.qualification",
                    "qualification.infer",
                    "1.0.0",
                    b"{}",
                    consume,
                    expected_listing=selected,
                )
            else:
                await service.invoke(
                    caller,
                    "managed.qualification",
                    "qualification.infer",
                    "1.0.0",
                    b"{}",
                    expected_listing=selected,
                )
        assert dispatched == (change == "same")
        if change == "same":
            write_private(
                tmp_path / "native-inference-policy.json",
                InferenceAccessPolicy(revision="e" * 64, enabled=False, callers=())
                .model_dump_json()
                .encode(),
            )
            with pytest.raises(ValueError):
                if streaming:
                    await service.invoke_stream(
                        caller,
                        "managed.qualification",
                        "qualification.infer",
                        "1.0.0",
                        b"{}",
                        consume,
                        expected_listing=selected,
                    )
                else:
                    await service.invoke(
                        caller,
                        "managed.qualification",
                        "qualification.infer",
                        "1.0.0",
                        b"{}",
                        expected_listing=selected,
                    )
    finally:
        await service.close()
