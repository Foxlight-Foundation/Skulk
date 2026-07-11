"""Opt-in field telemetry: anonymous performance + reliability samples.

The collector lives on the API node and is inert unless the operator has
explicitly enabled telemetry (``telemetry.consent = "enabled"`` in
``skulk.yaml``, set through the dashboard consent flow; ``unasked`` and
``disabled`` collect nothing). ``SKULK_TELEMETRY_DISABLE=1`` is a node-local
hard kill switch that overrides fleet policy.

Content-free by construction: a sample carries model id, canonical hardware
classes, cluster shape, timing/token COUNTS, and error-class enums. Never
prompt or output text, node ids, hostnames, addresses, or operator strings.
The ingest service independently enforces the same allowlist.

Reliability signals ride the same channel (survivorship-bias guard): failed
generations carry an ``error_class``, and node deaths are peer-observed by
diffing the visible node set between flushes, because a crashed node cannot
report itself.

The collector must never affect inference: the queue is bounded
(drop-on-overflow, drops counted), the flush is fail-silent with the batch
retained for the next cycle, and the stream tap swallows its own errors.
"""

from __future__ import annotations

import os
import time
from collections import deque
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Iterable
from typing import TYPE_CHECKING, TypedDict, TypeVar, cast, final
from uuid import uuid4

import anyio
import httpx
from loguru import logger

from skulk.shared.types.profiling import SystemPerformanceProfile
from skulk.shared.version import get_skulk_version
from skulk.utils.pydantic_ext import FrozenModel

if TYPE_CHECKING:
    from skulk.store.config import TelemetryConfig

TELEMETRY_KILL_SWITCH = "SKULK_TELEMETRY_DISABLE"

#: Wire-contract error classes (mirrored by the ingest allowlist).
ErrorClass = str

_MAX_PENDING_SAMPLES = 1000
_FLUSH_INTERVAL_S = 60.0
_FLUSH_BATCH_LIMIT = 400  # ingest caps batches at 500; leave headroom

#: Standard memory tiers (GB) shared with the ledger taxonomy: raw readings
#: snap to the nearest tier, which doubles as an anonymity coarsener.
_MEMORY_TIERS_GB = (8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 512)


def _memory_tier_gb(ram_total_bytes: int | None) -> int | None:
    if ram_total_bytes is None or ram_total_bytes <= 0:
        return None
    gigabytes = ram_total_bytes / 2**30
    return min(_MEMORY_TIERS_GB, key=lambda tier: abs(tier - gigabytes))


def _slug(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "-" for ch in value.strip().lower())
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-")


def hardware_class(
    accelerator_vendor: str | None,
    accelerator_name: str | None,
    ram_total_bytes: int | None,
) -> str:
    """Canonical class for one node, e.g. ``apple-m4-24gb`` or ``amd-64gb``.

    Vendor plus optional chip-name slug plus memory tier. Never carries
    operator identity: every part is either a vendor enum, a marketing chip
    name, or a standard tier.
    """
    vendor = _slug(accelerator_vendor or "") or "unknown"
    parts = [vendor]
    name = _slug(accelerator_name or "")
    if name and name != "unknown":
        parts.append(name)
    tier = _memory_tier_gb(ram_total_bytes)
    if tier is not None:
        parts.append(f"{tier}gb")
    return "-".join(parts)


@final
class TelemetrySample(FrozenModel):
    """One wire-contract sample (snake_case fields match the ingest API).

    Attributes:
        kind: ``generation`` or ``node-death``. (The wire contract also
            admits ``runner-restart``; this collector does not emit it yet.)
        at: ISO-8601 timestamp of the observation.
        model_id: Model identifier (generation samples only).
        engine: Engine family when known.
        quantization: Quantization label when known.
        hardware: Canonical hardware classes of the nodes involved.
        node_count: Nodes participating in the placement / cluster.
        ttft_s: Time to first token, seconds.
        decode_tps: Steady-state decode tokens/second.
        prompt_tokens: Prompt token COUNT (never content).
        output_tokens: Output token COUNT (never content).
        mtp_accept_ratio: Speculative-decoding acceptance when active.
        error_class: Failure class enum; ``None`` for a clean sample.
    """

    kind: str
    at: str
    model_id: str | None = None
    engine: str | None = None
    quantization: str | None = None
    hardware: tuple[str, ...] = ()
    node_count: int | None = None
    ttft_s: float | None = None
    decode_tps: float | None = None
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    mtp_accept_ratio: float | None = None
    error_class: ErrorClass | None = None


def _sample_json(sample: TelemetrySample) -> dict[str, object]:
    """Wire/preview form of one sample (JSON types, None fields omitted)."""
    return cast(
        "dict[str, object]", sample.model_dump(mode="json", exclude_none=True)
    )


class TelemetryPreview(TypedDict):
    """What the dashboard preview pane renders: consent + the exact batch."""

    enabled: bool
    consent: str
    pending: list[dict[str, object]]
    dropped_since_start: int
    install_id: str
    ingest_url: str


def prepare_telemetry_config_update(
    config_data: dict[str, object],
    existing: dict[str, object] | None,
) -> None:
    """Normalize the telemetry section of a config save, in place.

    Applied by ``PUT /config``: preserves the existing section when a partial
    save omits it (consent must never be silently wiped or granted), stamps
    ``consented_version`` from the running Skulk ONLY once a consent decision
    exists (both fields ``unasked`` means no decision has been made), and
    backfills a server-generated ``install_id`` whenever either consent is
    enabled without one (consent must never be silently inert, and a browser
    without Web Crypto cannot generate an id).
    """
    if (
        "telemetry" not in config_data
        and existing is not None
        and "telemetry" in existing
    ):
        config_data["telemetry"] = existing["telemetry"]
    raw_section = config_data.get("telemetry")
    if not isinstance(raw_section, dict):
        return
    section = cast("dict[str, object]", raw_section)
    decided = (
        section.get("consent", "unasked") != "unasked"
        or section.get("diagnostics_consent", "unasked") != "unasked"
    )
    if decided and not section.get("consented_version"):
        section["consented_version"] = get_skulk_version()
    if not section.get("install_id") and (
        section.get("consent") == "enabled"
        or section.get("diagnostics_consent") == "enabled"
    ):
        section["install_id"] = str(uuid4())


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@final
class FieldTelemetryCollector:
    """Bounded, consent-gated, fail-silent sample collector for the API node.

    Args:
        config_provider: Returns the CURRENT telemetry config (re-read per
            check so dashboard changes apply without restart); ``None`` when
            the config has no telemetry section.
        hardware_provider: Returns the current per-node hardware snapshot
            (node id -> (system profile or None, ram_total_bytes or None)).
            Node ids are used ONLY for in-process death diffing and never
            leave the process.
        post: Injectable async HTTP effect (tests replace it); defaults to
            ``httpx.AsyncClient.post``.
    """

    def __init__(
        self,
        config_provider: Callable[[], "TelemetryConfig | None"],
        hardware_provider: Callable[
            [], dict[str, tuple[SystemPerformanceProfile | None, int | None]]
        ],
        post: Callable[[str, dict[str, object]], Awaitable[int]] | None = None,
    ) -> None:
        self._config_provider = config_provider
        self._hardware_provider = hardware_provider
        self._post = post if post is not None else self._default_post
        self._pending: deque[TelemetrySample] = deque(maxlen=_MAX_PENDING_SAMPLES)
        self._dropped = 0
        self._known_nodes: dict[str, str] = {}  # node id -> hardware class

    # ---- consent -----------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """True only with explicit consent, an install id, and no kill switch."""
        if os.environ.get(TELEMETRY_KILL_SWITCH, "").strip() == "1":
            return False
        config = self._config_provider()
        return (
            config is not None
            and config.consent == "enabled"
            and bool(config.install_id)
            and bool(config.ingest_url)
        )

    # ---- recording ---------------------------------------------------------

    def _append(self, sample: TelemetrySample) -> None:
        if len(self._pending) == _MAX_PENDING_SAMPLES:
            self._dropped += 1
        self._pending.append(sample)

    def record_generation(
        self,
        model_id: str,
        *,
        ttft_s: float | None,
        decode_tps: float | None,
        prompt_tokens: int | None,
        output_tokens: int | None,
        node_count: int | None,
        error_class: ErrorClass | None,
    ) -> None:
        """Record one completed (or failed) generation. No-op without consent."""
        if not self.enabled:
            return
        # One snapshot serves both the classes and the count, so they can
        # never disagree and the provider runs once per sample.
        snapshot = self._hardware_provider()
        self._append(
            TelemetrySample(
                kind="generation",
                at=_now_iso(),
                model_id=model_id,
                hardware=tuple(sorted(set(self._classes_of(snapshot)))),
                # Placement size when the caller knows it; otherwise the
                # visible cluster size (an honest upper bound, never zero).
                node_count=node_count if node_count is not None else (len(snapshot) or None),
                ttft_s=ttft_s,
                decode_tps=decode_tps,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                error_class=error_class,
            )
        )

    def _classes_of(
        self,
        snapshot: dict[str, tuple[SystemPerformanceProfile | None, int | None]],
    ) -> Iterable[str]:
        for profile, ram in snapshot.values():
            accelerator = profile.accelerator if profile is not None else None
            yield hardware_class(
                accelerator.vendor if accelerator is not None else None,
                accelerator.name if accelerator is not None else None,
                ram,
            )

    def observe_node_set(self) -> None:
        """Diff the visible node set and record peer-observed deaths.

        A crashed or wedged node cannot report itself; this node CAN see it
        vanish. Called from the flush loop, so death detection lags at most
        one flush interval. No-op without consent (the baseline still
        updates, so enabling later never emits stale deaths).
        """
        current: dict[str, str] = {}
        for node_id, (profile, ram) in self._hardware_provider().items():
            accelerator = profile.accelerator if profile is not None else None
            current[node_id] = hardware_class(
                accelerator.vendor if accelerator is not None else None,
                accelerator.name if accelerator is not None else None,
                ram,
            )
        vanished = [
            cls for node_id, cls in self._known_nodes.items() if node_id not in current
        ]
        if self.enabled:
            for cls in vanished:
                self._append(
                    TelemetrySample(kind="node-death", at=_now_iso(), hardware=(cls,))
                )
        self._known_nodes = current

    # ---- preview + flush ---------------------------------------------------

    def preview(self) -> "TelemetryPreview":
        """The exact next batch, for the dashboard's what-would-be-sent pane."""
        config = self._config_provider()
        return TelemetryPreview(
            enabled=self.enabled,
            consent=config.consent if config is not None else "unasked",
            # Snapshot before iterating: harmless today (asyncio, no await
            # inside), cheap insurance against future threaded callers.
            pending=[_sample_json(s) for s in tuple(self._pending)],
            dropped_since_start=self._dropped,
            install_id=config.install_id if config is not None else "",
            ingest_url=config.ingest_url if config is not None else "",
        )

    async def _default_post(self, url: str, payload: dict[str, object]) -> int:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(url, json=payload)
            return response.status_code

    async def flush_once(self) -> bool:
        """Send one batch. Returns True when samples were accepted.

        Fail-silent: any error keeps the batch (bounded) for the next cycle.
        """
        self.observe_node_set()
        if not self.enabled or not self._pending:
            return False
        config = self._config_provider()
        if config is None:
            return False
        batch = [self._pending.popleft() for _ in range(min(len(self._pending), _FLUSH_BATCH_LIMIT))]
        payload: dict[str, object] = {
            "install_id": config.install_id,
            "skulk_version": get_skulk_version(),
            "samples": [_sample_json(s) for s in batch],
        }
        try:
            status = await self._post(
                f"{config.ingest_url.rstrip('/')}/v1/telemetry", payload
            )
        except Exception as exc:  # noqa: BLE001 - telemetry must never propagate
            logger.debug(f"telemetry flush failed, retaining batch: {exc}")
            self._requeue(batch)
            return False
        if status >= 300:
            logger.debug(f"telemetry flush rejected ({status}); retaining batch")
            self._requeue(batch)
            return False
        return True

    def _requeue(self, batch: list[TelemetrySample]) -> None:
        for index, sample in enumerate(reversed(batch)):
            if len(self._pending) == _MAX_PENDING_SAMPLES:
                # Every sample that no longer fits is a drop, not just the
                # first one encountered.
                self._dropped += len(batch) - index
                break
            self._pending.appendleft(sample)

    async def flush_loop(self) -> None:
        """Periodic flush; lifetime task on the API node. Never raises."""
        while True:
            await anyio.sleep(_FLUSH_INTERVAL_S)
            try:
                await self.flush_once()
            except Exception as exc:  # noqa: BLE001 - lifetime loop must survive
                logger.debug(f"telemetry flush loop error suppressed: {exc}")


_TChunk = TypeVar("_TChunk")


async def tap_generation_stream(
    collector: FieldTelemetryCollector,
    model_id: str,
    node_count: int | None,
    stream: AsyncIterator[_TChunk],
) -> AsyncGenerator[_TChunk, None]:
    """Wrap a token-chunk stream, recording one sample when it completes.

    Duck-typed against the chunk protocol (``finish_reason``, optional
    ``stats`` carrying token counts and generation tps) so it composes with
    the extension tap. Telemetry work is guarded: a collector bug can slow
    nothing and break nothing.
    """
    started = time.monotonic()
    ttft_s: float | None = None
    decode_tps: float | None = None
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    error = False
    completed = False
    try:
        async for chunk in stream:
            if ttft_s is None and getattr(chunk, "text", None):
                ttft_s = time.monotonic() - started
            stats = cast("object | None", getattr(chunk, "stats", None))
            if stats is not None:
                tps = cast("float | None", getattr(stats, "generation_tps", None))
                if tps is not None:
                    decode_tps = float(tps)
                prompt = cast("int | None", getattr(stats, "prompt_tokens", None))
                if prompt is not None:
                    prompt_tokens = int(prompt)
                generated = cast(
                    "int | None", getattr(stats, "generation_tokens", None)
                )
                if generated is not None:
                    output_tokens = int(generated)
            finish = cast("str | None", getattr(chunk, "finish_reason", None))
            if finish is not None:
                completed = True
            if finish == "error":
                error = True
            yield chunk
    finally:
        try:
            # An aborted stream (client disconnect: no terminal chunk, no
            # error) is neither a clean sample nor a failure; recording it
            # would pollute both speed and reliability distributions.
            if completed or error:
                collector.record_generation(
                    model_id,
                    ttft_s=ttft_s,
                    decode_tps=decode_tps,
                    prompt_tokens=prompt_tokens,
                    output_tokens=output_tokens,
                    node_count=node_count,
                    error_class="generation-error" if error else None,
                )
        except Exception as exc:  # noqa: BLE001 - telemetry must never propagate
            logger.debug(f"telemetry generation record failed: {exc}")
