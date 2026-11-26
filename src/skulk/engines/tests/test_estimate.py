"""Asking a really installed package's adapter what one load needs."""

import json
import time
from pathlib import Path
from typing import cast

import pytest

from skulk.engines.estimate import EngineNeed, ask_engine_need, stated_need
from skulk.engines.launch import engine_state_directory
from skulk.engines.protocol import EstimateResponse, LoadSettings, ModelIdentity
from skulk.engines.tests.package_fixtures import install_test_package

ESTIMATING_ADAPTER = rb"""
import json, os, struct, sys, time

digest = os.environ["SKULK_ENGINE_BUILD_DIGEST"]


def read_frame():
    head = sys.stdin.buffer.read(4)
    if not head:
        return None
    (length,) = struct.unpack(">I", head)
    return json.loads(sys.stdin.buffer.read(length))


def send(message):
    body = json.dumps(message, separators=(",", ":")).encode()
    sys.stdout.buffer.write(struct.pack(">I", len(body)) + body)
    sys.stdout.buffer.flush()


while True:
    request = read_frame()
    if request is None:
        break
    kind = request["type"]
    base = {"type": kind, "sequence": request["sequence"], "build_digest": digest}
    if kind == "hello":
        send({**base, "version": max(request["versions"])})
    elif kind == "estimate":
        directory = request["model"]["directory"]
        with open(os.path.join(directory, "mode.json")) as handle:
            mode = json.load(handle)["mode"]
        with open(os.path.join(directory, "asked.json"), "w") as handle:
            json.dump({"request": request, "state": os.environ["SKULK_ENGINE_STATE"]}, handle)
        if mode == "known":
            send({**base, "pool": "gpu", "load_bytes": 5000, "steady_bytes": 3000})
        elif mode == "unknown":
            send({**base, "pool": "host", "load_bytes": None, "steady_bytes": None})
        elif mode == "refuse":
            send({**base, "type": "error", "code": "unsupported_model",
                  "message": "no layout reader for /private/weights"})
        elif mode == "silent":
            time.sleep(600)
    elif kind == "stop":
        send(base)
        break
"""

SETTINGS = LoadSettings(
    context_tokens=32768, parallel_streams=1, cache_precision="default", drafting=False
)


def _model(tmp_path: Path, mode: str) -> ModelIdentity:
    directory = tmp_path / "model"
    directory.mkdir()
    (directory / "mode.json").write_text(json.dumps({"mode": mode}))
    return ModelIdentity(
        model_id="org/model", source_revision="d" * 40, directory=str(directory)
    )


def _estimate(
    load_bytes: int | None, steady_bytes: int | None
) -> EstimateResponse:
    return EstimateResponse(
        sequence=1,
        build_digest="a" * 64,
        pool="gpu",
        load_bytes=load_bytes,
        steady_bytes=steady_bytes,
    )


@pytest.mark.parametrize(
    ("load_bytes", "steady_bytes", "expected"),
    [
        (5000, 3000, 5000),
        (3000, 5000, 5000),
        (None, 3000, 3000),
        (5000, None, 5000),
        (0, None, 0),
    ],
)
def test_the_need_is_the_larger_figure_the_engine_knew(
    load_bytes: int | None, steady_bytes: int | None, expected: int
) -> None:
    """A load must hold its load-time peak and its steady state."""
    assert stated_need(_estimate(load_bytes, steady_bytes)) == EngineNeed(
        pool="gpu", bytes=expected
    )


def test_an_engine_that_knows_neither_figure_states_no_need() -> None:
    """Unknown is not zero: the caller keeps the admitted figure."""
    assert stated_need(_estimate(None, None)) is None


def test_the_adapter_is_asked_about_the_exact_load_and_leaves_nothing_behind(
    tmp_path: Path,
) -> None:
    """The engine sees the model and settings unchanged, in a session of its own."""
    installed = install_test_package(
        tmp_path, tmp_path / "packages", adapter=ESTIMATING_ADAPTER
    )
    model = _model(tmp_path, "known")
    state_root = tmp_path / "state"

    need = ask_engine_need(
        installed,
        instance_id="instance-a",
        model=model,
        settings=SETTINGS,
        state_root=state_root,
    )

    assert need == EngineNeed(pool="gpu", bytes=5000)
    asked = cast(
        "dict[str, object]",
        json.loads((Path(model.directory) / "asked.json").read_text()),
    )
    request = cast("dict[str, object]", asked["request"])
    assert request["instance_id"] == "instance-a"
    assert request["model"] == model.model_dump()
    assert request["settings"] == SETTINGS.model_dump()
    # The estimate ran in its own directory, apart from a loaded instance's,
    # and that directory is gone once the answer is in.
    estimates = engine_state_directory(installed.digest, root=state_root) / "estimates"
    assert asked["state"] == str(estimates / "instance-a")
    assert not (estimates / "instance-a").exists()


@pytest.mark.parametrize("mode", ["unknown", "refuse"])
def test_an_engine_that_does_not_say_leaves_the_admitted_figure(
    tmp_path: Path, mode: str
) -> None:
    """Not knowing and refusing to say both change nothing and fail nothing."""
    installed = install_test_package(
        tmp_path, tmp_path / "packages", adapter=ESTIMATING_ADAPTER
    )

    need = ask_engine_need(
        installed,
        instance_id="instance-a",
        model=_model(tmp_path, mode),
        settings=SETTINGS,
        state_root=tmp_path / "state",
    )

    assert need is None


def test_an_adapter_that_does_not_answer_is_given_up_within_the_deadline(
    tmp_path: Path,
) -> None:
    """The worker waits for this call, so a silent adapter must not hold it."""
    installed = install_test_package(
        tmp_path, tmp_path / "packages", adapter=ESTIMATING_ADAPTER
    )
    state_root = tmp_path / "state"
    started = time.monotonic()

    need = ask_engine_need(
        installed,
        instance_id="instance-a",
        model=_model(tmp_path, "silent"),
        settings=SETTINGS,
        state_root=state_root,
        deadline_seconds=1.5,
    )

    assert need is None
    assert time.monotonic() - started < 15
    assert not (
        engine_state_directory(installed.digest, root=state_root)
        / "estimates"
        / "instance-a"
    ).exists()


def test_an_adapter_that_cannot_start_states_no_need(tmp_path: Path) -> None:
    """A broken adapter is reported by the load itself, not by the estimate."""
    installed = install_test_package(
        tmp_path, tmp_path / "packages", adapter=b"import sys\nsys.exit(3)\n"
    )

    need = ask_engine_need(
        installed,
        instance_id="instance-a",
        model=_model(tmp_path, "known"),
        settings=SETTINGS,
        state_root=tmp_path / "state",
    )

    assert need is None
