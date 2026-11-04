"""Bounded multipart originals shared by native separation and music editing."""

import time
from collections.abc import AsyncGenerator, Callable

import anyio
from fastapi import HTTPException, Request
from pydantic import ValidationError
from starlette.datastructures import FormData, UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser

from skulk.shared.types.audio_separation import MAX_SEPARATION_SOURCE_BYTES


class _ClosingMultipartParser(MultiPartParser):
    """Close every spool after disconnect, cancellation or parser failure."""

    async def parse(self) -> FormData:
        """Parse once; failed reads retain no temporary source or open spool."""
        try:
            return await super().parse()
        except BaseException:
            for uploaded in self._files_to_close_on_error:
                uploaded.close()
            raise


async def read_wav_upload[Parameters](
    request: Request,
    parse: Callable[[str], Parameters],
    *,
    metadata_bytes: int,
    deadline_seconds: float,
) -> tuple[Parameters, bytes]:
    """Read exactly request JSON and one WAV within declared and streamed limits.

    The caller owns admission and supplies its closed typed metadata parser.
    Complete encoded bytes are bounded here; finite samples and WAV geometry
    are independently checked before the accepting owner persists any intent.
    """
    if not 4096 <= metadata_bytes <= 256 * 1024 or not 0 < deadline_seconds <= 180:
        raise ValueError("Audio multipart policy exceeds its fixed limits")
    maximum = MAX_SEPARATION_SOURCE_BYTES + metadata_bytes + 28 * 1024
    length = request.headers.get("content-length")
    if length is not None:
        try:
            declared = int(length)
        except ValueError:
            raise HTTPException(400, "invalid content-length") from None
        if not 0 <= declared <= maximum:
            raise HTTPException(
                413, "audio upload exceeds 64 MiB plus bounded metadata"
            )
    if (
        not request.headers.get("content-type", "")
        .lower()
        .startswith("multipart/form-data;")
    ):
        raise HTTPException(415, "use multipart/form-data with request and file")
    overflow, timed_out = False, False

    async def bounded() -> AsyncGenerator[bytes, None]:
        nonlocal overflow, timed_out
        total = 0
        stream = request.stream()
        deadline = time.monotonic() + deadline_seconds
        while True:
            try:
                with anyio.fail_after(max(0.0, deadline - time.monotonic())):
                    data = await anext(stream)
            except StopAsyncIteration:
                return
            except TimeoutError:
                timed_out = True
                raise MultiPartException("audio upload deadline expired") from None
            total += len(data)
            if total > maximum:
                overflow = True
                raise MultiPartException("audio upload exceeds its byte limit")
            yield data

    try:
        form = await _ClosingMultipartParser(
            request.headers,
            bounded(),
            max_files=1,
            max_fields=1,
            max_part_size=metadata_bytes,
        ).parse()
    except MultiPartException:
        raise HTTPException(
            413 if overflow else 408 if timed_out else 400,
            "invalid or over-limit audio multipart upload",
        ) from None
    try:
        if sorted(form.keys()) != ["file", "request"] or len(form.multi_items()) != 2:
            raise HTTPException(400, "require exactly request JSON and one file")
        metadata, upload = form.get("request"), form.get("file")
        if not isinstance(metadata, str) or not isinstance(upload, UploadFile):
            raise HTTPException(400, "require request JSON and one file")
        try:
            parameters = parse(metadata)
        except ValidationError:
            raise HTTPException(422, "invalid audio request metadata") from None
        data = await upload.read(MAX_SEPARATION_SOURCE_BYTES + 1)
        if len(data) > MAX_SEPARATION_SOURCE_BYTES:
            raise HTTPException(413, "original WAV exceeds 64 MiB")
        return parameters, data
    finally:
        await form.close()
