"""Hugging Face-style terms acceptance before Core keeps a received public model.

A receive plan lists each member's signed licence (``member_terms``). Licences
with use restrictions or conditions to pass on, and any that can't be
identified, need an explicit "I accept these terms" for exactly those licences
before the download starts. The review and the acceptance record come from the
pinned core (``terms_review``, ``TermsAcceptance``), as Sovereign desktop does.
This describes licences; it is not legal advice.
"""

from __future__ import annotations

import time
from collections.abc import Iterable

from foxlight_model_preservation.licences import (
    LICENCE_DEFINITION_VERSION,
    MemberTerms,
    TermsAcceptance,
    member_terms,
    terms_review,
)
from foxlight_model_preservation.models import MemberBase

from skulk.store.preservation_contracts import PlanMemberTerms, PreservationPlan

# As many as an archive has members (v1 and PMF alike), as the core reviews.
MAX_MEMBER_TERMS = 256


class TermsRefusedError(ValueError):
    """A stable refusal code; callers surface it as a host refusal."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def plan_member_terms(members: Iterable[MemberBase]) -> tuple[PlanMemberTerms, ...]:
    """Each signed member's licence and terms reference for a receive review.

    Members that share an identical licence, reference and origin are listed
    once, which keeps a large closure within the plan's bounded view without
    hiding any licence. Nothing is ever omitted: a review that cannot fit
    refuses rather than drop terms someone would have to accept.
    """
    listed: dict[tuple[str, str, str | None, str | None], PlanMemberTerms] = {}
    for item in member_terms(members):
        key = (item.license_id, item.reference, item.repository, item.revision)
        if key not in listed:
            listed[key] = PlanMemberTerms(
                member=item.member,
                license_id=item.license_id,
                reference=item.reference,
                repository=item.repository,
                revision=item.revision,
            )
    if len(listed) > MAX_MEMBER_TERMS:
        raise TermsRefusedError("terms_review_too_large")
    return tuple(listed.values())


def required_terms(plan: PreservationPlan) -> tuple[str, ...]:
    """The licence identifiers a start of this plan must accept, sorted."""
    if plan.kind != "receive" or not plan.member_terms:
        return ()
    review = terms_review(
        MemberTerms(
            member=item.member,
            license_id=item.license_id,
            reference=item.reference,
            repository=item.repository,
            revision=item.revision,
        )
        for item in plan.member_terms
    )
    accept = review["accept"]
    if not isinstance(accept, list):
        raise TermsRefusedError("terms_review_unavailable")
    return tuple(sorted(str(item) for item in accept))


def check_acceptance(
    plan: PreservationPlan, accepted: tuple[str, ...] | None
) -> tuple[str, ...]:
    """Require exactly the reviewed acceptance; return the licences accepted.

    Refuses a start that omits a needed acceptance or accepts other licences,
    and an acceptance on a plan that carries no member terms.
    """
    if plan.kind != "receive" or not plan.member_terms:
        if accepted:
            raise TermsRefusedError("terms_acceptance_unexpected")
        return ()
    wanted = required_terms(plan)
    if tuple(sorted(accepted or ())) != wanted:
        raise TermsRefusedError("terms_acceptance_required")
    return wanted


def acceptance_record(
    plan: PreservationPlan, job_id: str, accepted: tuple[str, ...]
) -> TermsAcceptance:
    """The core's acceptance record for one job, written before any download."""
    return TermsAcceptance(
        plan_id=plan.plan_id,
        job_id=job_id,
        name=plan.model_id[:1000],
        definition=LICENCE_DEFINITION_VERSION,
        accepted=accepted,
        accepted_at=int(time.time()),
    )
