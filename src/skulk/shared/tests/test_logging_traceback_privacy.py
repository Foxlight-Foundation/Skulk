"""A logged exception must not carry the values of local variables.

Loguru annotates exception tracebacks with the value of every variable on the
displayed source lines unless a sink is added with ``diagnose=False``. A failed
generation holds the request in its frames, so with annotation on, a failure
writes the start of a person's prompt into the node's log file and its captured
console stream. The JSON sink formats exceptions itself, without values, and is
checked so it stays that way.

Each test runs ``traceback_privacy_probe`` in a fresh interpreter (the property
depends on the real standard streams, loguru's import-time defaults, and a
spawned runner child) and then reads every sink.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path
from typing import cast

import pytest
import zstandard

import skulk
from skulk.shared.tests import traceback_privacy_probe as probe

_PROBE_CALL = "from skulk.shared.tests.traceback_privacy_probe import main\nmain()\n"
# Importing loguru before any Skulk module fixes its import-time defaults from
# the environment alone, the order an embedding host or a test runner produces.
_LOGURU_FIRST = "import loguru\n"
_ANSI_STYLE = re.compile(r"\x1b\[[0-9;]*m")


def _request_text() -> str:
    """Return request text that appears nowhere in any source file.

    A traceback prints source lines, so a literal would show up with or without
    variable annotation; a value built at run time can only reach a log through
    a frame's local variables.
    """
    return f"request-text-{uuid.uuid4().hex}"


def _run_probe(
    arguments: list[str],
    *,
    loguru_first: bool,
    loguru_diagnose: str | None,
) -> subprocess.CompletedProcess[str]:
    """Run the probe in a fresh interpreter and return its captured streams.

    Args:
        arguments: Command-line arguments for the probe.
        loguru_first: Import loguru before any Skulk module.
        loguru_diagnose: Value of ``LOGURU_DIAGNOSE`` in the child environment,
            or ``None`` to leave it unset.
    """
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("LOGURU_")
    }
    if loguru_diagnose is not None:
        environment["LOGURU_DIAGNOSE"] = loguru_diagnose
    # The external-shipper mode writes the JSON sink to stdout without starting
    # a log-shipper subprocess.
    environment["SKULK_LOGGING_EXTERNAL"] = "1"
    source_root = str(Path(skulk.__file__).resolve().parents[1])
    inherited_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_root
        if not inherited_path
        else os.pathsep.join((source_root, inherited_path))
    )
    code = (_LOGURU_FIRST if loguru_first else "") + _PROBE_CALL
    return subprocess.run(
        [sys.executable, "-c", code, *arguments],
        capture_output=True,
        text=True,
        env=environment,
        timeout=120,
        check=False,
    )


def _read_log_directory(directory: Path) -> str:
    """Return the text of every log generation, including rotated archives."""
    texts: list[str] = []
    for path in sorted(directory.iterdir()):
        raw = path.read_bytes()
        if path.suffix == ".zst":
            raw = zstandard.ZstdDecompressor().decompressobj().decompress(raw)
        texts.append(raw.decode("utf-8"))
    return "\n".join(texts)


def _json_entries(stdout: str) -> list[dict[str, object]]:
    """Parse the JSON sink's newline-delimited output."""
    entries: list[dict[str, object]] = []
    for line in stdout.splitlines():
        if line.strip():
            entries.append(cast(dict[str, object], json.loads(line)))
    return entries


def _assert_failure_reported(text: str, label: str, sink: str) -> None:
    """The failure stays diagnosable: its message, type and location survive."""
    assert label in text, f"{sink}: the failure was not logged"
    assert probe.FAILURE_TYPE_NAME in text, f"{sink}: exception type missing"
    assert probe.FAILURE_MESSAGE in text, f"{sink}: exception message missing"


@pytest.mark.parametrize(
    ("verbosity", "json_sink"),
    [(0, "startup"), (1, "runtime")],
)
def test_node_sinks_withhold_local_variable_values(
    tmp_path: Path, verbosity: int, json_sink: str
) -> None:
    """No sink a node configures prints a failing frame's local variables.

    Covers both console formats, the rotating log file, the JSON sink added at
    startup and the one added at runtime, and a runner child logging through
    the supervisor's logger and through its own process-global logger. Loguru
    is imported first with annotation requested by the environment, so nothing
    but the sinks' own configuration can be withholding the values.
    """
    request_text = _request_text()
    log_directory = tmp_path / "logs"
    log_directory.mkdir()

    result = _run_probe(
        [
            "--mode",
            "node",
            "--request-text",
            request_text,
            "--log-file",
            str(log_directory / "skulk.log"),
            "--verbosity",
            str(verbosity),
            "--json-sink",
            json_sink,
        ],
        loguru_first=True,
        loguru_diagnose="YES",
    )
    assert result.returncode == 0, result.stderr
    log_text = _read_log_directory(log_directory)

    for sink, text in (("console", result.stderr), ("log file", log_text)):
        for label in (probe.NODE_LABEL, probe.RUNNER_SUPERVISOR_LOGGER_LABEL):
            _assert_failure_reported(text, label, sink)
        # The frame holding the request text is in the traceback, so a missing
        # value below was withheld, not out of reach.
        assert probe.REQUEST_FRAME_SOURCE in text, f"{sink}: frame not shown"
    _assert_failure_reported(
        result.stderr, probe.RUNNER_PROCESS_LOGGER_LABEL, "runner console"
    )

    entries = _json_entries(result.stdout)
    for label in (probe.NODE_LABEL, probe.RUNNER_SUPERVISOR_LOGGER_LABEL):
        matching = [entry for entry in entries if entry.get("msg") == label]
        assert len(matching) == 1, f"JSON sink: expected one entry for {label!r}"
        exception_text = str(matching[0].get("exception"))
        assert probe.FAILURE_TYPE_NAME in exception_text, "JSON sink: type missing"
        assert probe.FAILURE_MESSAGE in exception_text, "JSON sink: message missing"

    for sink, text in (
        ("console", result.stderr),
        ("log file", log_text),
        ("JSON sink", result.stdout),
    ):
        assert request_text not in text, f"{sink} printed a local variable's value"


@pytest.mark.parametrize("loguru_diagnose", [None, "YES"])
def test_unconfigured_process_withholds_local_variable_values(
    loguru_diagnose: str | None,
) -> None:
    """A Skulk process that never configures logging withholds them too.

    Command-line tools, the plugin service and a spawned runner's own modules
    log through loguru's pre-installed handler. Importing Skulk settles that
    handler's behavior before loguru reads it, whatever the environment asked.
    """
    request_text = _request_text()

    result = _run_probe(
        ["--mode", "default", "--request-text", request_text],
        loguru_first=False,
        loguru_diagnose=loguru_diagnose,
    )

    assert result.returncode == 0, result.stderr
    # Loguru decides on its own whether to color this handler (it does under
    # some continuous-integration services even when the stream is a pipe), and
    # coloring splits a source line into separately styled tokens.
    plain_stderr = _ANSI_STYLE.sub("", result.stderr)
    _assert_failure_reported(plain_stderr, probe.DEFAULT_LABEL, "default handler")
    assert probe.REQUEST_FRAME_SOURCE in plain_stderr
    for text in (result.stderr, plain_stderr, result.stdout):
        assert request_text not in text
