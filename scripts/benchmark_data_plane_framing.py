"""Measure existing base64/JSON DATA framing against binary media framing."""

import argparse
import base64
import json
import statistics
import time
from collections.abc import Callable

from skulk.extensions.streams import (
    CapabilityStreamFrame,
    InlineMediaAttachment,
    decode_capability_stream_frame,
    encode_capability_stream_frame,
)
from skulk.routing.topics import DATA
from skulk.shared.models.model_cards import AudioResponseFormat, ModelId
from skulk.shared.types.chunks import AudioChunk, DataChunk
from skulk.shared.types.common import CommandId


def _measure(operation: Callable[[], object], iterations: int) -> float:
    samples: list[float] = []
    for _ in range(iterations):
        started_at = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - started_at)
    return statistics.median(samples)


def benchmark_size(size_bytes: int, iterations: int) -> dict[str, int | float]:
    """Benchmark both media framing paths for one deterministic payload size."""

    audio = bytes(index % 251 for index in range(size_bytes))

    def encode_json() -> bytes:
        return DATA.serialize(
            DataChunk(
                command_id=CommandId("framing-benchmark"),
                sequence=1,
                chunk=AudioChunk(
                    model=ModelId("benchmark/audio"),
                    data=base64.b64encode(audio).decode("ascii"),
                    chunk_index=0,
                    format=AudioResponseFormat.Wav,
                ),
            )
        )

    encoded_json = encode_json()

    def decode_json() -> bytes:
        frame = DATA.deserialize(encoded_json)
        assert isinstance(frame.chunk, AudioChunk)
        return base64.b64decode(frame.chunk.data)

    binary_frame = CapabilityStreamFrame(
        call_id="framing-benchmark",
        direction="provider_to_caller",
        sequence=1,
        kind="chunk",
        media=InlineMediaAttachment(
            data=audio,
            media_type="audio/pcm",
            codec="pcm_s16le",
            sample_rate=24000,
            channels=1,
        ),
    )

    def encode_binary() -> tuple[bytes, bytes | None]:
        return encode_capability_stream_frame(binary_frame)

    binary_header, binary_attachment = encode_binary()

    def decode_binary() -> CapabilityStreamFrame:
        return decode_capability_stream_frame(binary_header, binary_attachment)

    return {
        "payload_bytes": size_bytes,
        "json_wire_bytes": len(encoded_json),
        "binary_wire_bytes": len(binary_header) + len(binary_attachment or b""),
        "json_size_overhead_ratio": len(encoded_json) / size_bytes,
        "binary_size_overhead_ratio": (
            len(binary_header) + len(binary_attachment or b"")
        )
        / size_bytes,
        "json_encode_median_ms": _measure(encode_json, iterations) * 1000,
        "json_decode_median_ms": _measure(decode_json, iterations) * 1000,
        "binary_encode_median_ms": _measure(encode_binary, iterations) * 1000,
        "binary_decode_median_ms": _measure(decode_binary, iterations) * 1000,
    }


def main() -> None:
    """Run the framing benchmark and print machine-readable JSON results."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes",
        default="4096,65536,262144,1048576",
        help="Comma-separated media payload sizes in bytes.",
    )
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    sizes = [int(value) for value in args.sizes.split(",")]
    results = [benchmark_size(size, args.iterations) for size in sizes]
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
