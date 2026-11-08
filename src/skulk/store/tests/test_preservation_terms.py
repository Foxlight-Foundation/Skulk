"""Hugging Face-style terms acceptance on receive, recorded before any download."""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from foxlight_model_preservation.licences import TermsAcceptance
from foxlight_model_preservation.models import Recipe

from skulk.api.main import API
from skulk.api.preservation import create_preservation_router
from skulk.store.preservation_contracts import (
    PlanMemberTerms,
    PreservationPlan,
    PreservationStart,
)
from skulk.store.preservation_host import PreservationHost
from skulk.store.preservation_journal import PreservationHostError, PreservationJournal
from skulk.store.preservation_terms import (
    TermsRefusedError,
    check_acceptance,
    plan_member_terms,
    required_terms,
)
from skulk.store.tests.test_preservation_sharing import empty_host


def receive_plan(*terms: tuple[str, str], kind: str = "receive") -> PreservationPlan:
    """A retained receive review naming each member's signed licence."""
    return PreservationPlan.model_validate(
        {
            "plan_id": ("a" if kind == "receive" else "b") * 64,
            "kind": kind,
            "model_id": "Synthetic received model",
            "location_id": "invitation",
            "size_bytes": 1,
            "member_count": len(terms) or 1,
            "expires_at": int(time.time()) + 600,
            "requires_owner_acceptance": kind == "receive",
            "member_terms": [
                {"member": f"member-{index}", "license_id": licence, "reference": ref}
                for index, (licence, ref) in enumerate(terms)
            ],
        },
        strict=False,
    )


def test_permissive_licences_need_no_tick_and_restricted_ones_do() -> None:
    assert required_terms(receive_plan(("apache-2.0", "LICENSE"))) == ()
    assert required_terms(
        receive_plan(("MIT", "LICENSE"), ("cc-by-nc-4.0", "LICENSE"))
    ) == ("cc-by-nc-4.0",)
    # An unidentified licence also needs explicit acceptance.
    assert required_terms(receive_plan(("other", "LICENSE"))) == ("other",)
    # Only receive plans carry a terms gate.
    assert required_terms(receive_plan(("other", "LICENSE"), kind="publish")) == ()


def test_acceptance_must_match_the_review_exactly() -> None:
    plan = receive_plan(("llama3.1", "LICENSE"), ("cc-by-nc-4.0", "NOTICE"))
    wanted = ("cc-by-nc-4.0", "llama3.1")
    for given in (None, (), ("llama3.1",), (*wanted, "mit")):
        with pytest.raises(TermsRefusedError, match="terms_acceptance_required"):
            check_acceptance(plan, given)
    assert check_acceptance(plan, ("llama3.1", "cc-by-nc-4.0")) == wanted
    # A plan without member terms cannot carry an acceptance.
    with pytest.raises(TermsRefusedError, match="terms_acceptance_unexpected"):
        check_acceptance(receive_plan(kind="publish"), ("mit",))
    assert check_acceptance(receive_plan(("mit", "LICENSE")), None) == ()


def test_member_terms_come_from_the_signed_members_once_each() -> None:
    members = [
        {
            "member_id": name,
            "role": "primary" if name == "model" else "tokenizer",
            "format": "fixture",
            "origin_repository": "fixture/shared",
            "origin_revision": "a" * 40,
            "artifact_identity": "fixture:" + name,
            "dependencies": ["tokenizer"] if name == "model" else [],
            "intake": {
                "audience": "public",
                "license_id": "gemma",
                "terms_reference": "LICENSE",
                "notice_paths": ["LICENSE"],
                "decision": "synthetic fixture",
            },
            "files": [{"path": "LICENSE"}],
        }
        for name in ("model", "tokenizer")
    ]
    recipe = Recipe.model_validate_json(
        json.dumps(
            {
                "schema_version": "foxlight.model-preservation.v1",
                "name": "Synthetic",
                "root_member": "model",
                "members": members,
                "workloads": ["fixture"],
                "runtime_requirements": [],
            }
        )
    )
    # Both members share licence, reference and origin: listed once.
    assert plan_member_terms(recipe.members) == (
        PlanMemberTerms(
            member="model",
            license_id="gemma",
            reference="LICENSE",
            repository="fixture/shared",
            revision="a" * 40,
        ),
    )


def test_start_refuses_without_acceptance_and_records_it_with_the_job(
    tmp_path: Path,
) -> None:
    journal = PreservationJournal(tmp_path / "owner")
    plan = receive_plan(("cc-by-nc-4.0", "LICENSE"))
    journal.add_plan(plan, {})
    start = PreservationStart(
        plan_id=plan.plan_id, operation_id="receive", accept_owner_capture=True
    )
    with pytest.raises(PreservationHostError, match="terms_acceptance_required"):
        journal.reserve(start)
    with pytest.raises(PreservationHostError, match="terms_acceptance_required"):
        journal.reserve(start.model_copy(update={"accepted_terms": ("mit",)}))
    accepted = start.model_copy(update={"accepted_terms": ("cc-by-nc-4.0",)})
    job, created = journal.reserve(accepted)
    assert created and job.accepted_terms == ("cc-by-nc-4.0",)
    record = TermsAcceptance.model_validate_json(
        json.dumps(journal.setting("terms-acceptance:" + job.job_id))
    )
    assert record.job_id == job.job_id and record.plan_id == plan.plan_id
    assert record.accepted == ("cc-by-nc-4.0",) and record.definition >= 1
    # The replay observes the original job; nothing is recorded twice.
    replay, again = journal.reserve(accepted)
    assert replay.job_id == job.job_id and not again


def test_older_starts_keep_their_replay_identity(tmp_path: Path) -> None:
    # No acceptance field on the wire means the same digest as before.
    start = PreservationStart(plan_id="a" * 64, operation_id="receive")
    assert "accepted_terms" not in start.model_dump_json()
    journal = PreservationJournal(tmp_path / "owner")
    plan = receive_plan(("apache-2.0", "LICENSE"))
    journal.add_plan(plan, {})
    job, _ = journal.reserve(start.model_copy(update={"accept_owner_capture": True}))
    assert job.accepted_terms == ()
    assert journal.setting("terms-acceptance:" + job.job_id) is None


@pytest.mark.asyncio
async def test_the_api_accepts_terms_as_the_library_sends_them(tmp_path: Path) -> None:
    host = empty_host(tmp_path)

    async def provider() -> PreservationHost:
        return host

    app = FastAPI()
    app.include_router(
        create_preservation_router(provider, API.require_preservation_operator)
    )
    plan = receive_plan(("cc-by-nc-4.0", "LICENSE"))
    host.journal.add_plan(plan, {})
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, client=("127.0.0.1", 1000)),
        base_url="http://localhost",
    ) as client:
        try:
            read = await client.get("/preservation/plans/" + plan.plan_id)
            assert read.status_code == 200
            assert json.loads(read.text)["member_terms"] == [
                {
                    "member": "member-0",
                    "license_id": "cc-by-nc-4.0",
                    "reference": "LICENSE",
                    "repository": None,
                    "revision": None,
                }
            ]
            body = {
                "plan_id": plan.plan_id,
                "operation_id": "receive",
                "accept_owner_capture": True,
            }
            refused = await client.post("/preservation/jobs", json=body)
            assert refused.status_code == 409
            assert json.loads(refused.text)["detail"] == "terms_acceptance_required"
            # JSON lists reach the strict start contract; the receive itself then
            # fails in the worker because this synthetic plan has no payload.
            accepted = await client.post(
                "/preservation/jobs", json=body | {"accepted_terms": ["cc-by-nc-4.0"]}
            )
            assert accepted.status_code == 202, accepted.text
            assert json.loads(accepted.text)["accepted_terms"] == ["cc-by-nc-4.0"]
        finally:
            await host.close()


def test_an_acceptance_too_large_for_the_job_receipt_refuses_at_review() -> None:
    # 100 distinct long identifiers fit the plan view but not a job receipt.
    plan = receive_plan(
        *((f"custom-{index:03d}-" + "x" * 180, "LICENSE") for index in range(100))
    )
    with pytest.raises(TermsRefusedError, match="terms_review_too_large"):
        required_terms(plan)
