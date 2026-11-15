"""Shared protected native broker custody for API application adapters."""

import asyncio
from pathlib import Path
from typing import final

from fastapi import HTTPException

from skulk.extensions.native_inference_calls import NativeInferenceCalls
from skulk.extensions.runtime_attachment import ServiceConnection
from skulk.extensions.runtime_files import read_private
from skulk.extensions.runtime_install import finish_runtime_work
from skulk.shared.constants import SKULK_CONFIG_HOME
from skulk.shared.types.common import NodeId


@final
class NativeInferenceService:
    """Share one broker across adapters without caching original-caller authority.

    The protected connection file selects host custody. Serializing replacement
    prevents concurrent discovery requests from creating independent peer pools
    or obtaining a broker while its previous generation is being withdrawn.
    Each adapter still authenticates and authorizes its actual request.
    """

    def __init__(
        self,
        node_id: NodeId,
        *,
        connection_path: Path = SKULK_CONFIG_HOME
        / "managed-service"
        / "connection.json",
    ) -> None:
        """Bind server-selected custody; no request may supply this file path."""
        self.node_id, self.connection_path = node_id, connection_path
        self._broker: NativeInferenceCalls | None = None
        self._lock = asyncio.Lock()
        self._closed = False

    async def host(self) -> NativeInferenceCalls:
        """Return the current broker only after draining any replaced generation.

        Raises:
            HTTPException: The service is stopped or protected host custody is
                unavailable. Cancellation propagates after owned cleanup drains.
        """
        async with self._lock:
            if self._closed:
                raise HTTPException(503, "Native inference service is stopped.")
            try:
                connection = ServiceConnection.model_validate_json(
                    read_private(self.connection_path)
                )
                if self._broker is not None and self._broker.connection != connection:
                    previous, self._broker = self._broker, None
                    await finish_runtime_work(asyncio.create_task(previous.close()))
                if self._broker is None:
                    self._broker = NativeInferenceCalls(connection, str(self.node_id))
                return self._broker
            except (OSError, ValueError):
                previous, self._broker = self._broker, None
                if previous is not None:
                    await finish_runtime_work(asyncio.create_task(previous.close()))
                raise HTTPException(
                    503, "Native inference host is unavailable."
                ) from None

    async def close(self) -> None:
        """Prevent reacquisition and drain all current calls and host connections."""
        async with self._lock:
            self._closed = True
            broker, self._broker = self._broker, None
            if broker is not None:
                await finish_runtime_work(asyncio.create_task(broker.close()))
