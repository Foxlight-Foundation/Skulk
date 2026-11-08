"""Sovereign sharing on Core's persistent mirror: caps, intake, announcements, advisories.

Every effect runs through the real owner and its known worker. Nothing here
starts sharing without an explicit decision, and a Core-started admission never
records a new grant.
"""

from __future__ import annotations

import asyncio
import inspect
import socket
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar, cast

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from foxlight_model_preservation.kits import export_kit
from foxlight_model_preservation.mirror_announce import mirror_key
from foxlight_model_preservation.mirror_cards import build_peers, issue_card
from foxlight_model_preservation.mirror_models import MirrorState, WithdrawalAdvisory
from foxlight_model_preservation.mirror_rendezvous import rendezvous_available
from foxlight_model_preservation.mirror_state import MirrorLedger, restart_allowed
from pydantic import ValidationError

from skulk.store.mirror_contracts import (
    MirrorAction,
    MirrorConfiguration,
    MirrorIntake,
    MirrorJob,
    MirrorRequest,
    MirrorUsage,
)
from skulk.store.mirror_host import (
    _usage as usage,  # pyright: ignore[reportPrivateUsage]
)
from skulk.store.mirror_host import check_address, peers_outcome
from skulk.store.mirror_worker import MirrorRetainSource
from skulk.store.preservation_contracts import PreservationCopy
from skulk.store.preservation_host import PreservationHost
from skulk.store.preservation_journal import PreservationHostError
from skulk.store.tests.test_mirror_worker import archive, configuration
from skulk.store.tests.test_preservation_sharing import empty_host, import_copy

LICENCE = "fixture-only"
T = TypeVar("T")


def synthetic_state(counters: dict[str, object]) -> MirrorState:
    """A minimal valid ledger state carrying the given usage counters."""
    return MirrorState.model_validate(
        {
            "identity": {".": (1, 1)},
            "name": "Synthetic",
            "salt": "a" * 64,
            "disk_bytes": 1024**3,
            "ingress_daily": 1024,
            "egress_daily": None,
            "egress_monthly": None,
            "clock_floor": 0,
            "schema_version": "foxlight.mirror-state.v6",
            **counters,
        }
    )


async def settled(host: PreservationHost, job_id: str) -> MirrorJob:
    """Wait through the owned worker's actual drain."""
    async with asyncio.timeout(30):
        while True:
            job = host.mirrors.job(job_id)
            if job.state not in {"queued", "running", "cancelling"}:
                return job
            await asyncio.sleep(0.025)


async def decide(host: PreservationHost, request: MirrorRequest) -> MirrorJob:
    """Submit one decision and return its terminal receipt."""
    return await settled(host, (await host.mirrors.start(request)).job_id)


async def configured(
    host: PreservationHost, settings: MirrorConfiguration | None = None
) -> str:
    """Configure offline and return the reviewed policy identity."""
    job = await decide(
        host,
        MirrorRequest(
            owner_id="library",
            operation_id="configure-" + str(time.monotonic_ns()),
            kind="configure",
            policy_id=(await host.mirrors.page()).policy_id,
            configuration=settings or configuration(),
        ),
    )
    assert job.state == "succeeded", job
    page = await host.mirrors.page()
    assert page.policy_id is not None
    return page.policy_id


async def intake(
    host: PreservationHost,
    policy: str,
    *,
    licences: tuple[str, ...] = (LICENCE,),
    share: str = "eligible",
    withheld: tuple[str, ...] = (),
    operation: str = "intake",
) -> MirrorJob:
    """Record a standing intake policy through an ordinary fenced decision."""
    return await decide(
        host,
        MirrorRequest.model_validate(
            {
                "owner_id": "library",
                "operation_id": operation,
                "kind": "intake",
                "policy_id": policy,
                "intake": {
                    "share": share,
                    "licenses": list(licences),
                    "definition": 1,
                    "withheld": list(withheld),
                },
            },
            strict=False,
        ),
    )


async def grant(host: PreservationHost, policy: str) -> int:
    """Record standing permission for the exact policy; returns its expiry."""
    expires = int(time.time()) + 3600
    job = await decide(
        host,
        MirrorRequest(
            owner_id="library",
            operation_id="grant-" + str(time.monotonic_ns()),
            kind="grant_resume",
            policy_id=policy,
            resume_expires=expires,
            confirm_resume=True,
        ),
    )
    assert job.state == "succeeded", job
    return expires


async def resume(host: PreservationHost, policy: str) -> MirrorJob:
    """Start the standing controller and wait until it owns its listener."""
    job = await host.mirrors.start(
        MirrorRequest(
            owner_id="library",
            operation_id="resume-" + str(time.monotonic_ns()),
            kind="run",
            policy_id=policy,
            mode="resume",
            public_participation=True,
        )
    )
    async with asyncio.timeout(30):
        while True:
            current = host.mirrors.job(job.job_id)
            assert current.state in {"queued", "running"}, current
            if current.activity and current.activity.running:
                return current
            await asyncio.sleep(0.025)


async def public_copy(
    host: PreservationHost, root: Path, name: str
) -> tuple[PreservationCopy, str]:
    """Import one genuine signed public closure as a kept public copy."""
    folder = root / name
    folder.mkdir()
    source, identity, trust = archive(folder, name)
    kit = folder / "complete.foxkit"
    export_kit(source, identity, trust, kit)
    return await import_copy(host, kit.read_bytes(), operation_id=name), identity


def test_configuration_contract_allows_no_cap_and_needs_an_address_to_announce() -> (
    None
):
    uncapped = configuration().model_copy(
        update={"egress_daily": None, "egress_monthly": None}
    )
    assert MirrorConfiguration.model_validate_json(uncapped.model_dump_json()) == (
        uncapped
    )
    with pytest.raises(ValidationError, match="public URL"):
        MirrorConfiguration.model_validate(
            configuration().model_dump() | {"announce_dht": True}
        )
    with pytest.raises(ValidationError, match="plain HTTP"):
        MirrorConfiguration.model_validate(
            configuration().model_dump()
            | {"public_url": "https://user:secret@mirror.example"}
        )
    # A saved address announces nothing until its disclosure is recorded.
    assert (
        not configuration()
        .model_copy(update={"public_url": "https://mirror.example"})
        .announces()
    )
    assert (
        configuration()
        .model_copy(
            update={"public_url": "https://mirror.example", "announce_disclosure": 1}
        )
        .announces()
    )


def test_intake_belongs_only_to_its_decision_and_matches_the_core_policy() -> None:
    with pytest.raises(ValidationError, match="intake policy belongs"):
        MirrorRequest.model_validate(
            {
                "owner_id": "library",
                "operation_id": "x",
                "kind": "run",
                "policy_id": "a" * 64,
                "public_participation": True,
                "intake": {"share": "eligible", "licenses": ["mit"], "definition": 1},
            },
            strict=False,
        )
    with pytest.raises(ValidationError, match="standing intake"):
        # The core stores normalized identifiers only; unknown or "other"
        # names no licence and can never be selected.
        MirrorIntake.model_validate(
            {"share": "eligible", "licenses": ["other"], "definition": 1},
            strict=False,
        )
    # Following catalog withdrawals is the operator's choice, off by default.
    assert not configuration().follow_withdrawals


@pytest.mark.asyncio
async def test_uncapped_uploads_and_announcement_wait_for_the_disclosure(
    tmp_path: Path,
) -> None:
    host = empty_host(tmp_path)
    try:
        settings = configuration().model_copy(
            update={
                "egress_daily": None,
                "egress_monthly": None,
                "public_url": "http://198.51.100.7:8743",
                "announce_dht": True,
                "publish_nostr": True,
            }
        )
        await configured(host, settings)
        ledger = MirrorLedger(host.root / "managed-mirror")
        state = ledger.read()
        assert state.egress_daily is None and state.egress_monthly is None
        assert state.schema_version == "foxlight.mirror-state.v6"
        assert state.announce is None  # Saved, but nothing is announced yet.
        page = await host.mirrors.page()
        assert page.configuration is not None
        assert page.configuration.egress_daily is None
        assert page.configuration.public_url == "http://198.51.100.7:8743"
        assert page.usage is not None and page.usage.uploaded_today_bytes == 0
        assert not page.dht_announcing and not page.nostr_publishing
        assert page.announce_reasons == ("announce_disclosure_required",)

        await configured(host, settings.model_copy(update={"announce_disclosure": 1}))
        state = ledger.read()
        assert state.announce is not None
        assert (state.announce.dht, state.announce.nostr) == (True, True)
        page = await host.mirrors.page()
        assert page.configuration is not None
        assert page.configuration.announce_disclosure == 1
        # Configured, but no controller runs: still nothing is announced.
        assert not page.dht_announcing and not page.nostr_publishing
        assert "not_serving" in page.announce_reasons

        # Configuring without the disclosure withdraws the announcement again.
        await configured(host, settings)
        assert ledger.read().announce is None
    finally:
        await host.close()


@pytest.mark.asyncio
async def test_intake_is_fenced_refused_beside_a_controller_and_clears_the_grant(
    tmp_path: Path,
) -> None:
    host = empty_host(tmp_path)
    try:
        policy = await configured(host)
        stale = await intake(host, "b" * 64, operation="stale")
        assert stale.state == "failed" and stale.error_code == "mirror_policy_changed"

        expires = await grant(host, policy)
        controller = await host.mirrors.start(
            MirrorRequest(
                owner_id="library",
                operation_id="finite",
                kind="run",
                policy_id=policy,
                public_participation=True,
                seconds=60,
            )
        )
        busy = await intake(host, policy, operation="busy")
        assert busy.state == "failed" and busy.error_code == "mirror_busy_stop_first"
        await host.mirrors.cancel(controller.job_id)

        policy = (await host.mirrors.page()).policy_id
        assert policy is not None
        expires = await grant(host, policy)
        saved = await intake(host, policy)
        assert saved.state == "succeeded" and saved.kind == "intake"
        page = await host.mirrors.page()
        assert page.intake == MirrorIntake(
            share="eligible", licenses=(LICENCE,), definition=1
        )
        # Adopting a policy is new policy: the grant and Core's approval clear.
        assert page.policy_id != policy and not page.restart_allowed
        assert host.journal.setting("mirror-approved-resume") is None
        history = host.journal.mirror_jobs("library", 0)
        assert any(job.kind == "intake" for job in history)

        # Narrowing only (withholding a copy) keeps a fresh grant.
        assert page.policy_id is not None
        expires = await grant(host, page.policy_id)
        narrowed = await intake(
            host, page.policy_id, withheld=("c" * 64,), operation="narrow"
        )
        assert narrowed.state == "succeeded", narrowed
        page = await host.mirrors.page()
        assert page.restart_expires == expires
        assert page.intake is not None and page.intake.withheld == ("c" * 64,)
    finally:
        await host.close()


@pytest.mark.asyncio
async def test_a_covered_public_copy_joins_a_running_mirror_without_new_consent(
    tmp_path: Path,
) -> None:
    host = empty_host(tmp_path)
    try:
        policy = await configured(host)
        assert (await intake(host, policy)).state == "succeeded"
        policy = (await host.mirrors.page()).policy_id
        assert policy is not None
        expires = await grant(host, policy)
        first = await resume(host, policy)

        _, identity = await public_copy(host, tmp_path, "covered")
        async with asyncio.timeout(60):
            while True:
                page = await host.mirrors.page()
                running = (
                    page.controller_job_id is not None
                    and page.controller_job_id != first.job_id
                    and page.activity is not None
                    and page.activity.running
                )
                if running and any(
                    model.manifest_id == identity and model.state == "retained"
                    for model in page.models
                ):
                    break
                await asyncio.sleep(0.05)
        # Same policy, same grant: no grant_resume was recorded for it.
        assert page.policy_id == policy and page.restart_expires == expires
        assert host.journal.setting("mirror-approved-resume") == {
            "policy_id": policy,
            "expires": expires,
        }
        history = host.journal.mirror_jobs("core-intake", 0)
        retained = [job for job in history if job.kind == "retain"]
        assert len(retained) == 1 and retained[0].admitted_by == "intake"
        assert retained[0].manifest_id == identity
        assert not any(job.kind == "grant_resume" for job in history)
        assert host.mirrors.job(first.job_id).state == "cancelled"
        state = MirrorLedger(host.root / "managed-mirror").read()
        assert restart_allowed(state)
    finally:
        await host.close()


@pytest.mark.asyncio
async def test_standing_intake_leaves_uncovered_withheld_and_stopped_copies_alone(
    tmp_path: Path,
) -> None:
    host = empty_host(tmp_path)
    try:
        policy = await configured(host)
        # Licence not on the list: left for the owner's own review.
        assert (await intake(host, policy, licences=("mit",))).state == "succeeded"
        policy = (await host.mirrors.page()).policy_id
        assert policy is not None
        await grant(host, policy)
        controller = await resume(host, policy)
        await public_copy(host, tmp_path, "unlisted")
        await asyncio.sleep(1)
        page = await host.mirrors.page()
        assert page.model_count == 0 and page.controller_job_id == controller.job_id
        await host.mirrors.cancel(controller.job_id)

        # Stopped (paused): never admitted, even when the licence is listed.
        policy = (await host.mirrors.page()).policy_id
        assert policy is not None
        assert (
            await intake(host, policy, licences=(LICENCE,), operation="listed")
        ).state == "succeeded"
        policy = (await host.mirrors.page()).policy_id
        assert policy is not None
        await grant(host, policy)
        await public_copy(host, tmp_path, "while-stopped")
        await asyncio.sleep(1)
        assert (await host.mirrors.page()).model_count == 0

        # Chosen share mode admits nothing on its own.
        assert (
            await intake(
                host, policy, licences=(LICENCE,), share="chosen", operation="chosen"
            )
        ).state == "succeeded"
        policy = (await host.mirrors.page()).policy_id
        assert policy is not None
        await grant(host, policy)
        controller = await resume(host, policy)
        await public_copy(host, tmp_path, "not-chosen")
        await asyncio.sleep(1)
        page = await host.mirrors.page()
        assert page.model_count == 0 and page.controller_job_id == controller.job_id
        assert not host.journal.mirror_jobs("core-intake", 0)
    finally:
        await host.close()


@pytest.mark.asyncio
async def test_withhold_release_and_withdrawal_advisories_never_take_down_alone(
    tmp_path: Path,
) -> None:
    host = empty_host(tmp_path)
    try:
        policy = await configured(host)
        assert (await intake(host, policy)).state == "succeeded"
        policy = (await host.mirrors.page()).policy_id
        assert policy is not None
        copy, identity = await public_copy(host, tmp_path, "kept")
        retained = await decide(
            host,
            MirrorRequest(
                owner_id="library",
                operation_id="retain",
                kind="retain",
                policy_id=policy,
                copy_id=copy.copy_id,
                manifest_id=identity,
                confirm_redistribution=True,
            ),
        )
        assert retained.state == "succeeded", retained
        page = await host.mirrors.page()
        assert page.policy_id == policy  # A covered copy keeps the policy.
        expires = await grant(host, policy)

        actions: tuple[tuple[MirrorAction, bool], ...] = (
            ("withhold", True),
            ("release", False),
        )
        for action, blocked in actions:
            job = await decide(
                host,
                MirrorRequest(
                    owner_id="library",
                    operation_id=action,
                    kind="model",
                    policy_id=policy,
                    manifest_id=identity,
                    action=action,
                ),
            )
            assert job.state == "succeeded", job
            page = await host.mirrors.page()
            assert page.models[0].blocked is blocked
            assert page.models[0].withheld is blocked
            # The exact lookup projects the same withheld state as the page.
            selected = await host.mirrors.selection(identity)
            assert selected.model is not None
            assert selected.model.withheld is blocked
            # Withholding only narrows: the policy and its grant stay.
            assert page.policy_id == policy and page.restart_expires == expires

        # No catalog has withdrawn it, so there is nothing to acknowledge.
        refused = await decide(
            host,
            MirrorRequest(
                owner_id="library",
                operation_id="acknowledge-none",
                kind="model",
                policy_id=policy,
                manifest_id=identity,
                action="acknowledge",
            ),
        )
        assert refused.state == "failed"
        ledger = MirrorLedger(host.root / "managed-mirror")
        advisory = WithdrawalAdvisory(
            source_id="d" * 64, catalog_id="e" * 64, observed_at=1
        )
        ledger.update(
            lambda state: state.model_copy(
                update={
                    "models": {
                        **state.models,
                        identity: state.models[identity].model_copy(
                            update={"withdrawal": advisory}
                        ),
                    }
                }
            )
        )
        page = await host.mirrors.page()
        assert page.withdrawal_advisory_count == 1 and not page.follow_withdrawals
        (shown,) = page.withdrawal_advisories
        # Advisory by default: still served, not blocked, nothing taken down.
        assert shown.manifest_id == identity and not shown.serving_withheld
        assert not shown.blocked and "you decide" in shown.advice

        await configured(
            host, configuration().model_copy(update={"follow_withdrawals": True})
        )
        page = await host.mirrors.page()
        assert (
            page.follow_withdrawals and page.withdrawal_advisories[0].serving_withheld
        )
        # Reported with the configuration, so an exact comparison sees it.
        assert page.configuration is not None
        assert page.configuration.follow_withdrawals
        # Following withdrawals is policy: it clears the grant.
        assert page.policy_id != policy and not page.restart_allowed
        assert host.journal.setting("mirror-approved-resume") is None
        assert page.policy_id is not None
        acknowledged = await decide(
            host,
            MirrorRequest(
                owner_id="library",
                operation_id="acknowledge",
                kind="model",
                policy_id=page.policy_id,
                manifest_id=identity,
                action="acknowledge",
            ),
        )
        assert acknowledged.state == "succeeded", acknowledged
        (shown,) = (await host.mirrors.page()).withdrawal_advisories
        assert shown.acknowledged and not shown.serving_withheld
    finally:
        await host.close()


@pytest.mark.asyncio
async def test_check_address_is_best_effort_from_this_host_only(tmp_path: Path) -> None:
    host = empty_host(tmp_path)
    try:
        with pytest.raises(ValueError, match="mirror_not_configured|address"):
            await host.mirrors.check_address()
        await configured(
            host,
            configuration().model_copy(update={"public_url": "http://127.0.0.1:9"}),
        )
        # Loopback is private: nothing is even contacted.
        checked = await host.mirrors.check_address()
        assert checked.outcome == "private_address" and checked.best_effort
        assert checked.address == "http://127.0.0.1:9"

        ledger = MirrorLedger(host.root / "managed-mirror")
        key = host.root / "managed-mirror" / "mirror-key.pem"
        assert peers_outcome(b"not json", key) == "not_a_mirror"
        assert peers_outcome(build_peers(None, ()), key) == "no_card"
        stranger = issue_card(
            Ed25519PrivateKey.generate(), ("https://other.example",), {"f" * 64}
        )
        assert peers_outcome(build_peers(stranger, ()), key) == "other_mirror"
        own = issue_card(mirror_key(ledger), ("https://mirror.example",), {"f" * 64})
        assert peers_outcome(build_peers(own, ()), key) == "this_mirror"
    finally:
        await host.close()


async def standing_mirror(host: PreservationHost) -> MirrorJob:
    """Configure an eligible intake policy, grant it and start the resume run."""
    policy = await configured(host)
    assert (await intake(host, policy)).state == "succeeded"
    policy = (await host.mirrors.page()).policy_id
    assert policy is not None
    await grant(host, policy)
    return await resume(host, policy)


async def admission_ended(host: PreservationHost) -> None:
    """Wait until no standing-intake admission owns the controller."""
    async with asyncio.timeout(60):
        while (
            host.mirrors._admission is not None  # pyright: ignore[reportPrivateUsage]
            or host.mirrors._offers  # pyright: ignore[reportPrivateUsage]
        ):
            await asyncio.sleep(0.025)


@pytest.mark.asyncio
@pytest.mark.parametrize("point", ["review", "after_stop", "retain", "before_resume"])
async def test_an_operator_stop_at_any_admission_step_is_never_undone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, point: str
) -> None:
    host = empty_host(tmp_path)
    try:
        first = await standing_mirror(host)
        reached = asyncio.Event()
        release = asyncio.Event()

        async def hold() -> None:
            reached.set()
            await release.wait()

        mirrors = host.mirrors
        if point == "review":
            original_source = mirrors.source

            @asynccontextmanager
            async def reviewed(
                copy_id: str, manifest_id: str
            ) -> AsyncIterator[MirrorRetainSource]:
                await hold()
                async with original_source(copy_id, manifest_id) as source:
                    yield source

            monkeypatch.setattr(mirrors, "source", reviewed)
        elif point == "retain":
            original_settled = mirrors._settled  # pyright: ignore[reportPrivateUsage]

            async def settling(job_id: str) -> MirrorJob:
                await hold()
                return await original_settled(job_id)

            monkeypatch.setattr(mirrors, "_settled", settling)
        else:
            # Hold the admission's own ledger read: the first one after its
            # internal stop, or the last one before it submits the resume.
            wanted = 1 if point == "after_stop" else 2
            original_run = mirrors.storage.run
            calls = 0

            def run(
                action: Callable[[], T],
                *,
                discard: Callable[[T], None] | None = None,
                _cleanup: bool = False,
            ) -> Awaitable[T]:
                nonlocal calls
                frame = inspect.currentframe()
                caller = frame.f_back if frame is not None else None
                if caller is not None and caller.f_code.co_name == "_admit":
                    calls += 1
                    if calls == wanted:

                        async def held() -> T:
                            await hold()
                            return await original_run(
                                action, discard=discard, _cleanup=_cleanup
                            )

                        return held()
                return original_run(action, discard=discard, _cleanup=_cleanup)

            monkeypatch.setattr(mirrors.storage, "run", run)
        await public_copy(host, tmp_path, "stopped-" + point)
        async with asyncio.timeout(30):
            await reached.wait()
        page = await mirrors.page()
        # The page names the job standing in for the controller, so the
        # Library's one Stop finds it at every step.
        assert page.controller_job_id is not None
        if point == "review":
            assert page.controller_job_id == first.job_id
        await mirrors.cancel(page.controller_job_id)
        release.set()
        await admission_ended(host)
        page = await mirrors.page()
        assert page.controller_job_id is None and page.paused
        history = host.journal.mirror_jobs("core-intake", 0)
        assert not any(job.kind == "run" for job in history)
        if point in {"review", "after_stop"}:
            assert not history  # Nothing was retained after the Stop either.
        assert host.journal.setting("mirror-running-intent") is None
        assert not MirrorLedger(host.root / "managed-mirror").running()
    finally:
        await host.close()


@pytest.mark.asyncio
async def test_concurrent_offers_all_join_one_running_mirror(tmp_path: Path) -> None:
    host = empty_host(tmp_path)
    try:
        first = await standing_mirror(host)
        expires = (await host.mirrors.page()).restart_expires
        copies = [await public_copy(host, tmp_path, f"copy-{n}") for n in range(3)]
        await admission_ended(host)
        page = await host.mirrors.page()
        retained = {model.manifest_id for model in page.models}
        assert {identity for _, identity in copies} <= retained
        assert all(model.state == "retained" for model in page.models)
        # Each admission handed over a running controller to the next one.
        assert page.controller_job_id not in {None, first.job_id}
        assert page.activity is not None and page.activity.running
        assert page.restart_expires == expires and page.restart_allowed
        history = host.journal.mirror_jobs("core-intake", 0)
        assert sum(job.kind == "retain" for job in history) == 3
        assert not any(job.kind == "grant_resume" for job in history)
    finally:
        await host.close()


@pytest.mark.asyncio
async def test_a_failed_automatic_retain_resumes_the_unchanged_sharing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = empty_host(tmp_path)
    try:
        first = await standing_mirror(host)
        page = await host.mirrors.page()
        approval = host.journal.setting("mirror-approved-resume")
        original_source = host.mirrors.source
        opened = 0

        @asynccontextmanager
        async def failing(
            copy_id: str, manifest_id: str
        ) -> AsyncIterator[MirrorRetainSource]:
            nonlocal opened
            opened += 1
            if opened == 2:  # The retain's own reopen, after the review.
                raise PreservationHostError("public_copy_required")
            async with original_source(copy_id, manifest_id) as source:
                yield source

        monkeypatch.setattr(host.mirrors, "source", failing)
        await public_copy(host, tmp_path, "unreadable")
        await admission_ended(host)
        history = host.journal.mirror_jobs("core-intake", 0)
        (retain,) = [job for job in history if job.kind == "retain"]
        assert retain.state == "failed" and retain.admitted_by == "intake"
        # The ledger's grant never changed, so Core's approval is restored and
        # the earlier sharing resumes under it.
        assert host.journal.setting("mirror-approved-resume") == approval
        after = await host.mirrors.page()
        assert after.policy_id == page.policy_id and after.model_count == 0
        assert after.controller_job_id not in {None, first.job_id}
        assert after.activity is not None and after.activity.running
    finally:
        await host.close()


def test_usage_projects_zero_for_expired_periods() -> None:
    stored: dict[str, object] = {
        "day": "2026-11-06",
        "month": "2026-11",
        "egress_day_used": 7,
        "egress_month_used": 70,
        "ingress_used": 5,
    }
    base = synthetic_state(stored)
    same_month = datetime(2026, 11, 7, 12, tzinfo=UTC)
    assert usage(base, same_month) == MirrorUsage(
        uploaded_today_bytes=0, uploaded_month_bytes=70, received_today_bytes=0
    )
    assert usage(base, datetime(2026, 11, 6, 23, tzinfo=UTC)) == MirrorUsage(
        uploaded_today_bytes=7, uploaded_month_bytes=70, received_today_bytes=5
    )
    assert usage(base, datetime(2026, 12, 1, tzinfo=UTC)) == MirrorUsage(
        uploaded_today_bytes=0, uploaded_month_bytes=0, received_today_bytes=0
    )


def test_check_address_keeps_one_deadline_across_every_address(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    public = ("203.0.113.1", "8.8.8.8", "1.1.1.1", "9.9.9.9")

    def resolve(*_: object, **__: object) -> list[tuple[object, ...]]:
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 80))
            for address in public
        ]

    def stall(self: socket.socket, _address: object) -> None:
        # A black-holed address: the attempt uses all the time it was given.
        time.sleep(self.gettimeout() or 0)
        raise TimeoutError("timed out")

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    monkeypatch.setattr(socket.socket, "connect", stall)
    started = time.monotonic()
    checked = check_address(
        "http://mirror.example", tmp_path / "no-key.pem", started + 0.5
    )
    assert checked.outcome == "too_slow"
    # Four addresses with ten-second attempts would take forty seconds.
    assert time.monotonic() - started < 2


@pytest.mark.asyncio
async def test_a_native_only_mirror_reports_no_announcement(tmp_path: Path) -> None:
    host = empty_host(tmp_path)
    try:
        await configured(
            host,
            configuration().model_copy(
                update={
                    "public_url": "http://198.51.100.7:8743",
                    "announce_dht": True,
                    "publish_nostr": True,
                    "announce_disclosure": 1,
                }
            ),
        )
        state = MirrorLedger(host.root / "managed-mirror").read()
        serving = state.model_copy(update={"paused": False})
        announcement = host.mirrors._announcement  # pyright: ignore[reportPrivateUsage]
        # The DHT also needs the core's rendezvous runtime on this host.
        dht = rendezvous_available()
        assert announcement(serving, live=True) == (
            dht,
            True,
            () if dht else ("dht_unavailable",),
        )
        # The core announces only beside its HTTP server: HTTP off is not serving.
        native_only = serving.model_copy(update={"http_enabled": False})
        reasons = ("not_serving",) if dht else ("not_serving", "dht_unavailable")
        assert announcement(native_only, live=True) == (False, False, reasons)
    finally:
        await host.close()


def test_check_address_ends_a_trickling_close_delimited_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = socket.create_server(("127.0.0.1", 0))
    port = cast(int, server.getsockname()[1])
    stop = threading.Event()

    def trickle() -> None:
        with server:
            peer = server.accept()[0]
            with peer:
                peer.recv(4096)
                # HTTP/1.0 without a length: http.client detaches the socket
                # from the connection and reads the body until close.
                peer.sendall(b"HTTP/1.0 200 OK\r\n\r\n")
                while not stop.is_set():
                    with suppress(OSError):
                        peer.sendall(b"{")
                    time.sleep(0.1)

    thread = threading.Thread(target=trickle, daemon=True)
    thread.start()
    original_connect = socket.socket.connect

    def resolve(*_: object, **__: object) -> list[tuple[object, ...]]:
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 80))]

    def local(self: socket.socket, _address: object) -> None:
        original_connect(self, ("127.0.0.1", port))

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    monkeypatch.setattr(socket.socket, "connect", local)
    try:
        started = time.monotonic()
        checked = check_address(
            "http://mirror.example", tmp_path / "no-key.pem", started + 1
        )
        assert checked.outcome == "too_slow"
        assert time.monotonic() - started < 3
    finally:
        stop.set()
        thread.join(5)
