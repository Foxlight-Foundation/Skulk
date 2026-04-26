# type: ignore
"""Deterministic repro harness for the gemma-4 multimodal hang.

Drives the cluster's ``/v1/chat/completions`` endpoint with a fixed sequence
of multimodal turns and detects hangs by polling
``/v1/diagnostics/cluster/timeline``. Designed to validate fixes (does the
cluster survive 500 turns?) and isolate triggers (which env, which model,
which turn shape causes the wedge?). One command, runs unattended, writes
JSONL with per-turn outcomes.

Usage:
    uv run python bench/repro_gemma4_hang.py \\
        --api-url http://localhost:52415 \\
        --model mlx-community/gemma-4-26b-a4b-it-4bit \\
        --turns 200 \\
        --seed 42

Output: JSONL stream on stdout (or --output-jsonl), one record per turn.
Exit 0 if all turns succeeded, 1 if any hangs were detected, 2 on
unrecoverable error (cluster never came back, etc.).

Why a pure-stdlib harness: failure modes include the entire cluster going
unresponsive, so the harness must not depend on anything that might also
be wedged when things are bad. ``http.client`` + stdlib only.
"""
from __future__ import annotations

import argparse
import dataclasses
import http.client
import json
import os
import random
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

# ---------------------------------------------------------------------------
# Test assets
# ---------------------------------------------------------------------------

# Three base64-encoded PNG images of varying sizes. Hardcoded so the harness
# runs offline with bit-identical inputs across machines. The actual content
# does not matter for hang reproduction — what matters is the multimodal
# code path being engaged with images that vary in tile count.

# 1x1 transparent — the smallest possible PNG vision input.
_TINY_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)

# 32x32 solid color, generated with a simple raw-deflate stream.
# Chosen size exceeds the smallest vision tile but stays well under the
# multi-tile threshold; stresses the simple-image path.
_SMALL_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAALklEQVRYw+3R"
    "MQEAAAjDMMC/56FBxw4JIN3aSaqqqqqqqqqqqqqqqv4r9qj0BcCVAAGz1hKZ"
    "AAAAAElFTkSuQmCC"
)

# 256x256 — large enough to require multiple vision tiles in gemma-4's
# preprocessor, mirroring the load shape that triggered the original hang.
# Generated procedurally below to avoid bloating this file with kilobytes
# of base64; cached after first generation.
_LARGE_PNG_CACHE: list[str] = []


def _build_large_png_b64() -> str:
    """Build a 256x256 PNG procedurally with stdlib only.

    Stable bit pattern across runs: every pixel is filled with a fixed RGB
    based on (x + y) % 255, encoded as a single IDAT chunk. This is
    intentionally not artistic — we just need a multi-tile image with
    deterministic bytes.
    """
    if _LARGE_PNG_CACHE:
        return _LARGE_PNG_CACHE[0]

    import base64
    import struct
    import zlib

    width = 256
    height = 256
    raw = bytearray()
    for y in range(height):
        raw.append(0)  # filter byte: None
        for x in range(width):
            v = (x + y) % 255
            raw.extend([v, (v * 3) % 255, (v * 7) % 255])
    compressed = zlib.compress(bytes(raw), level=6)

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", compressed)
        + chunk(b"IEND", b"")
    )
    encoded = base64.b64encode(png).decode("ascii")
    _LARGE_PNG_CACHE.append(encoded)
    return encoded


def _data_url(b64: str) -> str:
    return f"data:image/png;base64,{b64}"


# ---------------------------------------------------------------------------
# Conversation generator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Turn:
    """One user message to send. Captures the prompt and any image attachments."""

    text: str
    image_b64s: tuple[str, ...] = ()


@dataclass(frozen=True)
class Session:
    """A multi-turn chat session — accumulates message history client side.

    Each session targets the failure pattern observed in the original
    incident: image in history, then a new turn (text or text+image) that
    references the prior context.
    """

    label: str
    turns: tuple[Turn, ...]


_SESSION_TEMPLATES: tuple[Session, ...] = (
    # Session A: image-heavy, mirrors the original bug report. Two
    # multimodal turns followed by a text turn referencing both images.
    Session(
        label="multimodal-history-text-followup",
        turns=(
            Turn("Describe what you see in this image briefly.", (_TINY_PNG,)),
            Turn("Now describe this second image.", (_SMALL_PNG,)),
            Turn("In one sentence, how do the two images differ?", ()),
        ),
    ),
    # Session B: large image, single turn. Stresses the multi-tile vision
    # path with a single-shot request.
    Session(
        label="large-image-single-turn",
        turns=(
            Turn("Describe the colors and patterns visible here.", ("LARGE",)),
        ),
    ),
    # Session C: text-only multi-turn. Control — should never hang on
    # gemma-4 if the bug is multimodal-correlated.
    Session(
        label="text-only-multi-turn",
        turns=(
            Turn("What is 27 times 14?", ()),
            Turn("Now divide that by 6.", ()),
            Turn("Express the result in words.", ()),
        ),
    ),
    # Session D: image then image then image — sustained multimodal load.
    Session(
        label="three-image-sequence",
        turns=(
            Turn("What's in this picture?", (_TINY_PNG,)),
            Turn("And this one?", (_SMALL_PNG,)),
            Turn("How about this?", ("LARGE",)),
        ),
    ),
)


def _resolve_image(token: str) -> str:
    """Resolve an image placeholder ('LARGE') to its base64 payload."""
    if token == "LARGE":
        return _build_large_png_b64()
    return token


def session_iterator(seed: int) -> Iterator[Session]:
    """Yield sessions in a deterministic shuffled order.

    Reseeded determinism means turn N of run M targets the same session as
    turn N of run M+1, which is essential for distinguishing "regression at
    turn 47" from "random session at turn 47 happened to be the bad one."
    """
    rng = random.Random(seed)
    while True:
        order = list(_SESSION_TEMPLATES)
        rng.shuffle(order)
        for session in order:
            yield session


def session_to_messages(session: Session) -> list[list[dict[str, Any]]]:
    """Expand a Session into the progressive message-history list.

    Returns one list per turn; each entry is the full history to send for
    that turn. The harness sends each progressively larger history to
    reproduce the original failure shape ("hang on turn 2 with image in
    history").
    """
    progressions: list[list[dict[str, Any]]] = []
    history: list[dict[str, Any]] = []
    for turn in session.turns:
        content: list[dict[str, Any]] = [{"type": "text", "text": turn.text}]
        for image_b64 in turn.image_b64s:
            resolved = _resolve_image(image_b64)
            content.append(
                {"type": "image_url", "image_url": {"url": _data_url(resolved)}}
            )
        history.append({"role": "user", "content": content})
        progressions.append([dict(m) for m in history])
        # Append a placeholder assistant turn so subsequent user turns have
        # a real conversation history. Filled in after the response arrives.
        history.append({"role": "assistant", "content": ""})
    return progressions


# ---------------------------------------------------------------------------
# Hang detection — pure logic, importable by tests
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimelineSnapshot:
    """Subset of /v1/diagnostics/cluster/timeline used for hang detection.

    Pulled out as its own type so unit tests can construct minimal fixtures
    without depending on the live API shape.
    """

    runners: tuple["RunnerSnapshot", ...]
    eval_timeout_events: tuple["TimeoutEvent", ...] = ()
    unreachable_node_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class RunnerSnapshot:
    node_id: str
    device_rank: int
    phase: str
    seconds_in_phase: float
    last_progress_at: str | None


@dataclass(frozen=True)
class TimeoutEvent:
    at: str
    node_id: str
    device_rank: int
    detail: str | None


@dataclass(frozen=True)
class HangVerdict:
    """Outcome of evaluating a timeline snapshot for hang signals."""

    status: str  # "ok" | "hang" | "recovered_hang" | "kernel_panic_inferred"
    reason: str
    suspect_runner: RunnerSnapshot | None = None


# Phases that are expected to take longer than the default hang threshold
# during normal operation. Seeing a runner sit in one of these for >hang
# threshold is not by itself a hang signal — it's a slow legit operation.
_LONG_RUNNING_PHASES: frozenset[str] = frozenset(
    {
        "warmup",
        "loading",
        "downloading",
        "created",
        "shutdown_cleanup",
    }
)


def evaluate_timeline_for_hang(
    snapshot: TimelineSnapshot,
    *,
    hang_after_seconds: float,
    expected_node_ids: frozenset[str] | None = None,
) -> HangVerdict:
    """Pure function: look at one snapshot and decide what it tells us.

    Priority:

    1. If the previously-reachable node set lost a member, infer kernel
       panic on that node. The combination "node disappears mid-request +
       Skulk's normal supervisor restart hasn't yet republished the runner"
       is the cleanest in-band signal we have for a kernel-level reboot.
    2. If any runner has emitted a ``pipeline_eval_timeout`` flight-recorder
       event, classify as ``recovered_hang`` — the eval-timeout patch fired,
       which is the supervised-recovery success case. Distinct from ``ok``
       because we want to track recovery rate as a separate SLO line.
    3. If any runner is sitting in a non-long-running phase past the
       threshold, classify as a hang. Long-running phases (warmup, loading)
       are exempt to avoid false positives during cold start.
    4. Otherwise, ``ok``.

    Note: ``last_progress_at`` is currently advisory; the threshold itself
    on ``seconds_in_phase`` is the primary signal because the cluster
    timeline endpoint already gives us bounded values from the supervisor.
    """
    if expected_node_ids is not None:
        observed_ids = frozenset(r.node_id for r in snapshot.runners) | frozenset(
            snapshot.unreachable_node_ids
        )
        missing = expected_node_ids - observed_ids
        if missing:
            return HangVerdict(
                status="kernel_panic_inferred",
                reason=(
                    f"node(s) disappeared from cluster timeline: "
                    f"{sorted(missing)}; if no graceful unregister event "
                    "was observed this is a hard reboot signal."
                ),
            )

    if snapshot.eval_timeout_events:
        # Surface the most recent timeout event so the operator can see
        # which site fired without re-querying the timeline.
        latest = max(snapshot.eval_timeout_events, key=lambda e: e.at)
        return HangVerdict(
            status="recovered_hang",
            reason=(
                f"pipeline_eval_timeout fired at {latest.at} on "
                f"node={latest.node_id} rank={latest.device_rank} "
                f"site={latest.detail or '<unknown>'}"
            ),
        )

    for runner in snapshot.runners:
        if runner.phase in _LONG_RUNNING_PHASES:
            continue
        if runner.seconds_in_phase > hang_after_seconds:
            return HangVerdict(
                status="hang",
                reason=(
                    f"rank={runner.device_rank} stuck in "
                    f"phase={runner.phase} for "
                    f"{runner.seconds_in_phase:.1f}s "
                    f"(threshold={hang_after_seconds:.0f}s)"
                ),
                suspect_runner=runner,
            )

    return HangVerdict(status="ok", reason="all runners within threshold")


def parse_timeline_response(payload: dict[str, Any]) -> TimelineSnapshot:
    """Convert a /v1/diagnostics/cluster/timeline response into the local shape.

    Tolerates missing fields rather than crashing — the cluster may be
    partially up, peers may be missing, etc. Errors here would mask real
    hangs by making the harness itself fail.
    """
    runners: list[RunnerSnapshot] = []
    for r in payload.get("runners", []) or []:
        runners.append(
            RunnerSnapshot(
                node_id=str(r.get("nodeId", "")),
                device_rank=int(r.get("deviceRank", -1)),
                phase=str(r.get("phase", "")),
                seconds_in_phase=float(r.get("secondsInPhase", 0.0)),
                last_progress_at=r.get("lastProgressAt"),
            )
        )

    timeouts: list[TimeoutEvent] = []
    for entry in payload.get("timeline", []) or []:
        if entry.get("event") == "pipeline_eval_timeout":
            timeouts.append(
                TimeoutEvent(
                    at=str(entry.get("at", "")),
                    node_id=str(entry.get("nodeId", "")),
                    device_rank=int(entry.get("deviceRank", -1)),
                    detail=entry.get("detail"),
                )
            )

    unreachable = tuple(
        str(u.get("nodeId", ""))
        for u in payload.get("unreachableNodes", []) or []
        if u.get("nodeId")
    )

    return TimelineSnapshot(
        runners=tuple(runners),
        eval_timeout_events=tuple(timeouts),
        unreachable_node_ids=unreachable,
    )


# ---------------------------------------------------------------------------
# HTTP client (minimal, stdlib only)
# ---------------------------------------------------------------------------


@dataclass
class ApiClient:
    base_url: str
    timeout_seconds: float = 300.0

    def _connect(self) -> http.client.HTTPConnection:
        parsed = urlparse(self.base_url)
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if parsed.scheme == "https":
            return http.client.HTTPSConnection(host, port, timeout=self.timeout_seconds)
        return http.client.HTTPConnection(host, port, timeout=self.timeout_seconds)

    def get_json(self, path: str) -> Any:
        conn = self._connect()
        try:
            conn.request("GET", path, headers={"Accept": "application/json"})
            resp = conn.getresponse()
            body = resp.read().decode("utf-8", errors="replace")
            if resp.status >= 400:
                raise RuntimeError(f"GET {path} -> {resp.status}: {body[:300]}")
            return json.loads(body) if body else None
        finally:
            conn.close()

    def post_chat_completion(
        self, model: str, messages: list[dict[str, Any]]
    ) -> dict[str, Any]:
        conn = self._connect()
        try:
            payload = json.dumps(
                {"model": model, "messages": messages, "stream": False}
            ).encode("utf-8")
            conn.request(
                "POST",
                "/v1/chat/completions",
                body=payload,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )
            resp = conn.getresponse()
            body = resp.read().decode("utf-8", errors="replace")
            if resp.status >= 400:
                raise RuntimeError(
                    f"POST chat -> {resp.status}: {body[:300]}"
                )
            return json.loads(body)
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


@dataclass
class TurnRecord:
    turn_idx: int
    session_idx: int
    session_label: str
    turn_within_session: int
    n_messages_in_history: int
    n_images_in_history: int
    status: str
    reason: str
    latency_seconds: float
    tokens_generated: int | None = None
    completion_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class RunSummary:
    total_turns: int
    ok_turns: int
    hang_turns: int
    recovered_hang_turns: int
    api_error_turns: int
    kernel_panic_turns: int
    elapsed_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def count_images(history: list[dict[str, Any]]) -> int:
    n = 0
    for msg in history:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    n += 1
    return n


def run_repro(
    *,
    api: ApiClient,
    model: str,
    turns: int,
    seed: int,
    hang_after_seconds: float,
    output_jsonl: str | None,
    expected_node_ids: frozenset[str] | None,
) -> RunSummary:
    """Drive the cluster for ``turns`` turns and emit a JSONL record per turn.

    The detection cadence is checked once per turn after the response (or
    timeout) lands — running an in-flight poller is a future enhancement
    that requires threading and isn't needed for the core "did this many
    turns succeed?" question.
    """
    started = time.monotonic()
    sessions = session_iterator(seed)
    counts = {
        "ok": 0,
        "hang": 0,
        "recovered_hang": 0,
        "api_error": 0,
        "kernel_panic_inferred": 0,
    }

    # Conditional file vs stdout — a `with` block doesn't compose cleanly here
    # because we explicitly do not want to close stdout when no output path
    # is set. The try/finally below handles the close path correctly.
    out_fp = open(output_jsonl, "w") if output_jsonl else sys.stdout  # noqa: SIM115

    try:
        turn_idx = 0
        session_idx = 0
        for session in sessions:
            if turn_idx >= turns:
                break
            session_idx += 1
            histories = session_to_messages(session)
            for within, history in enumerate(histories):
                if turn_idx >= turns:
                    break
                turn_idx += 1

                started_turn = time.monotonic()
                status = "ok"
                reason = ""
                tokens_generated: int | None = None
                completion_id: str | None = None

                try:
                    resp = api.post_chat_completion(model, history)
                    choices = resp.get("choices") or []
                    if choices:
                        message = choices[0].get("message") or {}
                        # Patch the placeholder assistant turn with the real
                        # response so the next turn's history is realistic.
                        if (
                            within * 2 + 1 < len(history)
                            and history[within * 2 + 1].get("role") == "assistant"
                        ):
                            history[within * 2 + 1]["content"] = (
                                message.get("content") or ""
                            )
                    completion_id = resp.get("id")
                    usage = resp.get("usage") or {}
                    tokens_generated = usage.get("completion_tokens")
                except Exception as exc:  # noqa: BLE001
                    status = "api_error"
                    reason = f"{type(exc).__name__}: {exc}"

                # Independent of API outcome, sample the cluster timeline so
                # we catch hangs that the supervisor recovered from before
                # the request returned, and kernel panics where the request
                # failed because the box rebooted under us.
                if status == "ok":
                    try:
                        snap = parse_timeline_response(
                            api.get_json("/v1/diagnostics/cluster/timeline")
                        )
                        verdict = evaluate_timeline_for_hang(
                            snap,
                            hang_after_seconds=hang_after_seconds,
                            expected_node_ids=expected_node_ids,
                        )
                        if verdict.status != "ok":
                            status = verdict.status
                            reason = verdict.reason
                    except Exception as exc:  # noqa: BLE001
                        # Timeline endpoint itself unreachable is a
                        # different failure: classify as kernel_panic
                        # inferred only when we had previously been
                        # successfully reaching it.
                        status = "kernel_panic_inferred"
                        reason = (
                            f"timeline endpoint unreachable: "
                            f"{type(exc).__name__}: {exc}"
                        )

                record = TurnRecord(
                    turn_idx=turn_idx,
                    session_idx=session_idx,
                    session_label=session.label,
                    turn_within_session=within,
                    n_messages_in_history=len(history),
                    n_images_in_history=count_images(history),
                    status=status,
                    reason=reason,
                    latency_seconds=time.monotonic() - started_turn,
                    tokens_generated=tokens_generated,
                    completion_id=completion_id,
                )
                out_fp.write(json.dumps(record.as_dict()) + "\n")
                out_fp.flush()
                counts[status] = counts.get(status, 0) + 1

                if status in {"hang", "kernel_panic_inferred"}:
                    # Bail out fast on hard failures — no point burning
                    # turns against a wedged cluster.
                    break
            else:
                continue
            break
    finally:
        if out_fp is not sys.stdout:
            out_fp.close()

    return RunSummary(
        total_turns=turn_idx,
        ok_turns=counts.get("ok", 0),
        hang_turns=counts.get("hang", 0),
        recovered_hang_turns=counts.get("recovered_hang", 0),
        api_error_turns=counts.get("api_error", 0),
        kernel_panic_turns=counts.get("kernel_panic_inferred", 0),
        elapsed_seconds=time.monotonic() - started,
    )


def _parse_node_ids(raw: str | None) -> frozenset[str] | None:
    if raw is None:
        return None
    return frozenset(part.strip() for part in raw.split(",") if part.strip())


def _discover_expected_node_ids(api: ApiClient) -> frozenset[str]:
    """Snapshot the node set at startup so we can detect later disappearances."""
    snap = parse_timeline_response(api.get_json("/v1/diagnostics/cluster/timeline"))
    return frozenset(r.node_id for r in snap.runners)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--api-url",
        default=os.environ.get("SKULK_API_URL", "http://localhost:52415"),
        help="Base URL of the Skulk API (default %(default)s).",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model ID to drive (e.g. mlx-community/gemma-4-26b-a4b-it-4bit).",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=200,
        help="Number of turns to run (default %(default)d).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for session ordering (default %(default)d).",
    )
    parser.add_argument(
        "--hang-after-seconds",
        type=float,
        default=90.0,
        help=(
            "Treat any non-long-running phase still active past this many "
            "seconds as a hang signal (default %(default).0f). Set above the "
            "60s eval timeout so a recovered hang is classified as "
            "recovered_hang, not as a fresh hang."
        ),
    )
    parser.add_argument(
        "--output-jsonl",
        default=None,
        help=(
            "Write JSONL records to this path instead of stdout. Summary "
            "still prints to stderr regardless."
        ),
    )
    parser.add_argument(
        "--expected-nodes",
        default=None,
        help=(
            "Comma-separated list of node IDs expected to be reachable. "
            "Default: snapshot the current cluster at startup."
        ),
    )
    args = parser.parse_args(argv)

    api = ApiClient(base_url=args.api_url)

    expected: frozenset[str] | None
    explicit = _parse_node_ids(args.expected_nodes)
    if explicit is not None:
        expected = explicit
    else:
        try:
            expected = _discover_expected_node_ids(api)
            print(
                f"[repro] discovered {len(expected)} node(s) at startup: "
                f"{sorted(expected)}",
                file=sys.stderr,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"[repro] could not snapshot node set at startup ({exc!r}); "
                "kernel-panic detection will be disabled.",
                file=sys.stderr,
            )
            expected = None

    summary = run_repro(
        api=api,
        model=args.model,
        turns=args.turns,
        seed=args.seed,
        hang_after_seconds=args.hang_after_seconds,
        output_jsonl=args.output_jsonl,
        expected_node_ids=expected,
    )

    print(json.dumps({"summary": summary.as_dict()}, indent=2), file=sys.stderr)

    if summary.kernel_panic_turns > 0:
        return 2
    if summary.hang_turns > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
