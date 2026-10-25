"""Operator-only prepared kit streaming from an owned descriptor lease."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Callable
from typing import final

from fastapi import HTTPException
from starlette.responses import Response, StreamingResponse
from starlette.types import Message, Receive, Scope, Send

from skulk.store.mirror_host import MirrorHost


def _range(value: str | None, size: int) -> tuple[int, int] | None:
    if value is None:
        return 0, size
    if len(value) > 160:
        return None
    match = re.fullmatch(r"bytes=([0-9]*)-([0-9]*)", value)
    if match is None:
        return None
    first, last = match.groups()
    if not first:
        if not last or int(last) == 0:
            return None
        return max(0, size - int(last)), size
    start = int(first)
    end = min(size, int(last) + 1) if last else size
    return (start, end) if start < end and start < size else None


@final
class MirrorKitDownload(Response):
    """Keep a validated descriptor owned through headers, ranges and disconnect.

    Opening occurs before response headers. Byte reads recheck the prepared file
    fingerprint and have bounded IO admission. No request performs a full-kit
    hash, grants model trust or resolves a caller-supplied filesystem path.
    """

    def __init__(
        self,
        owner: MirrorHost,
        job_id: str,
        byte_range: str | None,
        if_range: str | None,
        refusal: Callable[[Exception], HTTPException],
    ) -> None:
        """Select one known job and ordinary HTTP range metadata, without IO."""
        super().__init__(media_type="application/octet-stream")
        self.owner, self.job_id = owner, job_id
        self.byte_range, self.if_range, self.refusal = byte_range, if_range, refusal

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Serve only a leased unchanged artifact and release it on every exit."""
        started = False

        async def observe(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            async with self.owner.kit_file(self.job_id) as (descriptor, seal):
                length = seal.fingerprint.size_bytes
                etag = '"sha256-' + seal.sha256 + '"'
                headers = {
                    "Accept-Ranges": "bytes",
                    "Cache-Control": "private, no-store",
                    "ETag": etag,
                    "Content-Disposition": 'attachment; filename="sovereign-'
                    + seal.manifest_id[:12]
                    + '.foxkit"',
                }
                requested = self.byte_range if scope["method"] == "GET" else None
                if self.if_range is not None and self.if_range != etag:
                    requested = None
                selected = _range(requested, length)
                if selected is None:
                    await Response(
                        status_code=416,
                        headers={
                            "Content-Range": f"bytes */{length}",
                            "Cache-Control": "private, no-store",
                        },
                    )(scope, receive, observe)
                    return
                first, end = selected
                status = 206 if requested is not None else 200
                headers["Content-Length"] = str(end - first)
                if status == 206:
                    headers["Content-Range"] = f"bytes {first}-{end - 1}/{length}"

                async def chunks() -> AsyncIterator[bytes]:
                    offset = first
                    while offset < end:
                        chunk = await self.owner.kit_chunk(
                            descriptor, seal, offset, min(1024**2, end - offset)
                        )
                        offset += len(chunk)
                        yield chunk

                if scope["method"] == "HEAD":
                    await Response(
                        headers=headers, media_type="application/octet-stream"
                    )(scope, receive, observe)
                else:
                    await StreamingResponse(
                        chunks(),
                        status_code=status,
                        headers=headers,
                        media_type="application/octet-stream",
                    )(scope, receive, observe)
        except (OSError, ValueError) as error:
            if started:
                # A changed file interrupts the stream. It can never be turned
                # into a successful JSON body under previously sent kit headers.
                raise
            raise self.refusal(error) from error
