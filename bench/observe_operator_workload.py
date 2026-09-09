"""Explicit local-only aggregate observation of a generated operator fixture.

This does not attest installed app provenance, measure client-side delivery, or
qualify capacity. No existing cluster or cloud resource is ever selected.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import cast

from bench.operator_fixture_observer import (
    FixtureObservationError,
    FixtureObserver,
    ObservationFlow,
)
from bench.operator_fixture_recorder import RecorderSettings, aggregate_recorder
from bench.operator_workload_fixture import FixtureSettings, isolated_fixture

_FLOWS: frozenset[str] = frozenset(
    (
        "cold-launch",
        "foreground-refresh",
        "settled-foreground",
        "background-resume",
        "reconnect",
        "chat",
        "speech",
    )
)


@asynccontextmanager
async def _stdin() -> AsyncIterator[asyncio.StreamReader]:
    reader = asyncio.StreamReader(limit=128)
    transport, _ = await asyncio.get_running_loop().connect_read_pipe(
        lambda: asyncio.StreamReaderProtocol(reader),
        sys.stdin,
    )
    try:
        yield reader
    finally:
        transport.close()


async def capture_controls(
    reader: asyncio.StreamReader, observer: FixtureObserver
) -> None:
    """Apply bounded fixed-vocabulary flow commands, returning only on `finish`.

    `reader` contains local operator commands, not device traffic. `observer`
    enforces idle socket boundaries. Input lines are bounded to 128 bytes;
    EOF, unknown commands, and invalid flow transitions fail the capture.
    No input text is echoed. The caller validates aggregate completeness next.
    """
    while True:
        try:
            line = await reader.readuntil(b"\n")
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError):
            raise FixtureObservationError from None
        if len(line) > 128:
            raise FixtureObservationError
        if line == b"finish\n":
            if not observer.idle():
                raise FixtureObservationError
            return
        if line == b"end\n":
            observer.end()
        elif line.startswith(b"begin "):
            try:
                flow = line[6:-1].decode("ascii")
            except UnicodeDecodeError:
                raise FixtureObservationError from None
            if flow not in _FLOWS:
                raise FixtureObservationError
            observer.begin(cast(ObservationFlow, flow))
        else:
            raise FixtureObservationError
        print(
            '{"schema":"operator-observation-control.v1","accepted":true}', flush=True
        )


async def observe_fixture(
    settings: FixtureSettings, recorder_settings: RecorderSettings
) -> None:
    """Run local generated services and export only a complete aggregate to stdout.

    `settings` supplies the relay digest and whole-session lifetime.
    `recorder_settings` supplies the existing local recorder modules and digests.
    Readiness output exposes protected pairing artifact paths, never credentials.
    This gateway-boundary observation is explicitly unattested; actual installed
    application provenance and capacity acceptance remain separate requirements.
    Services, pipes, and ephemeral verified modules are closed on every exit.
    """
    async with aggregate_recorder(recorder_settings) as recorder:
        observer = FixtureObserver(recorder.record)
        async with isolated_fixture(settings, observer=observer) as fixture:
            print(
                json.dumps(
                    {
                        "schema": "operator-observation-ready.v1",
                        "pairingFile": str(fixture.directory / "pairing.txt"),
                        "pairingQr": str(fixture.directory / "pairing.png"),
                        "relayPort": fixture.relay_port,
                        "measurementBoundary": "gateway-inner-tls-tcp-and-asgi",
                        "synthetic": True,
                        "capacityQualified": False,
                    }
                ),
                flush=True,
            )
            async with _stdin() as reader:
                await capture_controls(reader, observer)
            result = await recorder.finish()
        # Do not expose a completed result until fixture teardown also succeeds.
        print(result.decode("utf-8"), end="", flush=True)


def main(arguments: Sequence[str] | None = None) -> None:
    """Parse explicit local artifact inputs and start the aggregate-only session.

    `arguments` defaults to command-line arguments. Raises on invalid inputs or
    capture failures; does not install dependencies or create paid resources.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relay-binary", type=Path, required=True)
    parser.add_argument("--relay-sha256", required=True)
    parser.add_argument("--lifetime-seconds", type=int, default=600)
    parser.add_argument("--node-binary", type=Path, required=True)
    parser.add_argument("--recorder-module", type=Path, required=True)
    parser.add_argument("--recorder-sha256", required=True)
    parser.add_argument("--cli-module", type=Path, required=True)
    parser.add_argument("--cli-sha256", required=True)
    options = vars(parser.parse_args(arguments))
    settings = FixtureSettings.model_validate(
        {
            key: options[key]
            for key in (
                "relay_binary",
                "relay_sha256",
                "lifetime_seconds",
            )
        }
    )
    recorder_settings = RecorderSettings.model_validate(
        {
            key: options[key]
            for key in (
                "node_binary",
                "recorder_module",
                "recorder_sha256",
                "cli_module",
                "cli_sha256",
            )
        }
    )
    asyncio.run(observe_fixture(settings, recorder_settings))


if __name__ == "__main__":
    main()
