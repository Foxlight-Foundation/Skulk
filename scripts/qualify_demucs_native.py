"""Exercise the real native CLI/supervisor with exact local weights and bounded WAV."""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import time
import wave
from pathlib import Path

from skulk.shared.types.audio_separation import (
    NativeSeparationJob,
    SeparationDevice,
    source_from_wav,
)
from skulk.worker.runner.demucs.process import run_native_separation


class Arguments(argparse.Namespace):
    """Explicit owned local test resources; this command never downloads models."""

    interpreter: Path
    weights: Path
    output: Path
    source: Path | None
    device: SeparationDevice


def main() -> None:
    """Prove native execution/geometry/custody without inferring musical usefulness."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interpreter", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument(
        "--device", choices=("cpu", "mps", "cuda", "rocm"), default="cpu"
    )
    arguments = Arguments()
    parser.parse_args(namespace=arguments)
    root = arguments.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    source = (
        arguments.source.resolve()
        if arguments.source is not None
        else root / "source.wav"
    )
    if arguments.source is None:
        # An odd native frame count exercises nearest-frame resampling and mono
        # duplication through the actual model, not only through schema tests.
        with wave.open(str(source), "wb") as wav:
            wav.setparams((1, 2, 8000, 8001, "NONE", "not compressed"))
            wav.writeframes(
                b"".join(
                    struct.pack(
                        "<h", int(3000 * math.sin(2 * math.pi * 220 * frame / 8000))
                    )
                    for frame in range(8001)
                )
            )
    with source.open("rb") as original:
        source_bytes = original.read(64 * 1024 * 1024 + 1)
    facts = source_from_wav(source_bytes)
    job = NativeSeparationJob(
        weights_path=str(arguments.weights.resolve()),
        source_path=str(source),
        output_dir=str(root / "stems"),
        source=facts,
        device=arguments.device,
        threads=4,
        segment_seconds=2.0,
    )
    supervised_started = time.monotonic()
    result = run_native_separation(
        # Resolving the executable's symlink would select base Python and lose
        # the provisioned virtual environment's native dependency set.
        arguments.interpreter.absolute(),
        job,
        progress=lambda record: print(
            record.model_dump_json(), file=sys.stderr, flush=True
        ),
        is_cancelled=lambda: False,
        timeout_seconds=600,
    )
    if result is None:
        raise ValueError("native separation was unexpectedly cancelled")
    supervised_seconds = time.monotonic() - supervised_started
    with source.open("rb") as original:
        if original.read(64 * 1024 * 1024 + 1) != source_bytes:
            raise ValueError("native execution changed the original source bytes")
    (root / "measurement.json").write_text(result.model_dump_json() + "\n")
    (root / "supervision.json").write_text(
        json.dumps({"supervised_wall_seconds": supervised_seconds}) + "\n"
    )
    print(result.model_dump_json())


if __name__ == "__main__":
    main()
