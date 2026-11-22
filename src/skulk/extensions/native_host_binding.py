"""Normal manager provisioning of the private Core/owner role attachment."""

import secrets
from pathlib import Path
from typing import Literal, final

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import BaseModel, ConfigDict, Field

from skulk.extensions.native_host_protocol import HostBinding
from skulk.extensions.runtime_attachment import HostSettings
from skulk.extensions.runtime_files import RuntimeLock, read_private, write_private


@final
class CoreHostKey(BaseModel):
    """Profile-owned role signing key, never delivered to installed author code."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    protocol: Literal[1] = 1
    profile_id: str = Field(
        pattern=r"^[a-f0-9]{32}$", description="Owning manager profile."
    )
    private_key: str = Field(
        pattern=r"^[a-f0-9]{64}$", repr=False, description="Core role key seed."
    )

    def key(self) -> Ed25519PrivateKey:
        """Materialize the host-only signer; this is not caller authentication."""
        return Ed25519PrivateKey.from_private_bytes(bytes.fromhex(self.private_key))


def load_core_host_key(manager: Path, profile_id: str) -> CoreHostKey:
    """Read a provisioned Core role key and reject a stale service profile."""
    key = CoreHostKey.model_validate_json(
        read_private(manager / "native-host-key.json")
    )
    if key.profile_id != profile_id:
        raise ValueError("native host key profile differs")
    return key


def provision_native_host(
    installation: Path, runtime_digest: str, instance: str
) -> None:
    """Bind an owner lifetime while the normal launcher holds its exclusive fence.

    Older owners ignore the separate sidecar. Only a qualified calls/1 owner
    starts its new host listener; the sidecar alone advertises no native capacity.
    Unprofiled legacy launches retain their existing behavior.
    """
    if installation.parent.name != "installations":
        return
    manager = installation.parent.parent
    settings = HostSettings.model_validate_json(read_private(manager / "host.json"))
    if settings.profile_id is None:
        return
    lock = RuntimeLock(manager, "native-host-key.lock")
    try:
        try:
            key = load_core_host_key(manager, settings.profile_id)
        except FileNotFoundError:
            key = CoreHostKey(
                profile_id=settings.profile_id,
                private_key=Ed25519PrivateKey.generate().private_bytes_raw().hex(),
            )
            write_private(
                manager / "native-host-key.json", key.model_dump_json().encode()
            )
    finally:
        lock.close()
    binding = HostBinding(
        protocol=1,
        profile_id=settings.profile_id,
        plugin_id=installation.name,
        runtime_digest=runtime_digest,
        transport_node_id=settings.transport_node_id,
        service_instance=instance,
        core_public_key=key.key().public_key().public_bytes_raw().hex(),
        server_secret=secrets.token_hex(32),
    )
    body = binding.model_dump_json().encode()
    # Core reads its own manager copy, not author-writable discovery metadata.
    write_private(
        manager / "native-host-bindings" / (installation.name + ".json"), body
    )
    write_private(installation / "core-host.json", body)
