"""Protected local key provider for the single-gateway operator authority."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final, final
from uuid import uuid4

from skulk.shared.constants import SKULK_CONFIG_HOME

_LOCAL_KEY_ID: Final = "local-gateway-v1"
_DATA_KEY_BYTES: Final = 32


class AuthorityKeyUnavailableError(RuntimeError):
    """Raised when the designated gateway has no usable local authority key."""


@final
class LocalFileAuthorityKeyProvider:
    """Load one random authority data key from a protected local file.

    This provider is the explicit v1 availability tradeoff: the designated
    gateway owns its authority key and remote access is unavailable if that
    host is unavailable. POSIX ownership and mode ``0600`` protect the key at
    rest. Hardware-backed wrapping remains post-v1 hardening.
    """

    def __init__(self, path: Path | None = None) -> None:
        """Create a provider for the default or explicitly supplied key path.

        Args:
            path: Key file used by the designated gateway. Tests may inject a
                temporary path.
        """

        self._path = (
            path
            if path is not None
            else SKULK_CONFIG_HOME / "operator" / "authority-key-v1.bin"
        )

    @property
    def active_key_id(self) -> str:
        """Return the stable identifier for the v1 local gateway key."""

        return _LOCAL_KEY_ID

    @property
    def path(self) -> Path:
        """Return the authority key file path."""

        return self._path

    def ensure_data_key(self) -> bytes:
        """Load the local data key or create it atomically with mode ``0600``.

        Returns:
            The 32-byte authority data key.

        Side effects:
            Creates and fsyncs the protected operator directory and key file on
            first use.
        """

        self._ensure_parent()
        if self._path.exists():
            return self.load_data_key(_LOCAL_KEY_ID)

        key = os.urandom(_DATA_KEY_BYTES)
        temporary_path = self._path.with_name(f".{self._path.name}.{uuid4().hex}.tmp")
        try:
            descriptor = os.open(
                temporary_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            with os.fdopen(descriptor, "wb", closefd=True) as stream:
                stream.write(key)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary_path, self._path)
            except FileExistsError:
                return self.load_data_key(_LOCAL_KEY_ID)
            if os.name == "posix":
                self._path.chmod(0o600)
                parent_descriptor = os.open(self._path.parent, os.O_RDONLY)
                try:
                    os.fsync(parent_descriptor)
                finally:
                    os.close(parent_descriptor)
        finally:
            temporary_path.unlink(missing_ok=True)
        return key

    def load_data_key(self, key_id: str) -> bytes:
        """Load the active local authority data key.

        Args:
            key_id: Persisted key identifier. Only the active v1 identifier is
                accepted.

        Returns:
            The validated 32-byte authority data key.

        Raises:
            AuthorityKeyUnavailableError: The key is absent, has the wrong ID,
                has unsafe POSIX permissions, or has invalid length.
        """

        if key_id != _LOCAL_KEY_ID:
            raise AuthorityKeyUnavailableError("authority key version is unavailable")
        try:
            key = self._path.read_bytes()
        except FileNotFoundError as exc:
            raise AuthorityKeyUnavailableError(
                "operator gateway is not initialized; run `skulk operator pair`"
            ) from exc
        if len(key) != _DATA_KEY_BYTES:
            raise AuthorityKeyUnavailableError("operator authority key is malformed")
        if os.name == "posix" and self._path.stat().st_mode & 0o077:
            raise AuthorityKeyUnavailableError(
                "operator authority key permissions must not grant group or other access"
            )
        return key

    def _ensure_parent(self) -> None:
        """Create and harden the authority key directory."""

        self._path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if os.name == "posix":
            self._path.parent.chmod(0o700)
