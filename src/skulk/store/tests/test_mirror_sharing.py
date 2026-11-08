"""Sovereign sharing on Core's persistent mirror: caps, intake, announcements, advisories.

Every effect runs through the real owner and its known worker. Nothing here
starts sharing without an explicit decision, and a Core-started admission never
records a new grant.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from foxlight_model_preservation.kits import export_kit
from foxlight_model_preservation.mirror_announce import mirror_key
from foxlight_model_preservation.mirror_cards import build_peers, issue_card
from foxlight_model_preservation.mirror_models import WithdrawalAdvisory
from foxlight_model_preservation.mirror_state import MirrorLedger, restart_allowed
from pydantic import ValidationError

from skulk.store.mirror_contracts import (
    MirrorAction,
    MirrorConfiguration,
    MirrorIntake,
    MirrorJob,
    MirrorRequest,
)
from skulk.store.mirror_host import peers_outcome
from skulk.store.preservation_contracts import PreservationCopy
from skulk.store.preservation_host import PreservationHost
from skulk.store.tests.test_mirror_worker import archive, configuration
from skulk.store.tests.test_preservation_sharing import empty_host, import_copy

LICENCE = "fixture-only"


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
    source, identity, trust = archive(folder)
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


@pytest.mark.asyncio
async def test_an_operator_stop_during_an_admission_is_never_undone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = empty_host(tmp_path)
    try:
        policy = await configured(host)
        assert (await intake(host, policy)).state == "succeeded"
        policy = (await host.mirrors.page()).policy_id
        assert policy is not None
        await grant(host, policy)
        first = await resume(host, policy)
        reached = asyncio.Event()
        release = asyncio.Event()

        async def gated(job_id: str) -> MirrorJob:
            reached.set()
            await release.wait()
            return await settled(host, job_id)

        monkeypatch.setattr(host.mirrors, "_settled", gated)
        await public_copy(host, tmp_path, "stopped-midway")
        async with asyncio.timeout(30):
            await reached.wait()
        page = await host.mirrors.page()
        # The page names the step in flight, so the Library's one Stop finds it.
        assert page.controller_job_id is not None
        assert page.controller_job_id != first.job_id
        await host.mirrors.cancel(page.controller_job_id)
        release.set()
        async with asyncio.timeout(30):
            while host.mirrors._admission is not None:  # pyright: ignore[reportPrivateUsage]
                await asyncio.sleep(0.025)
        page = await host.mirrors.page()
        assert page.controller_job_id is None and page.paused
        assert not any(
            job.kind == "run" for job in host.journal.mirror_jobs("core-intake", 0)
        )
        assert host.journal.setting("mirror-running-intent") is None
    finally:
        await host.close()
