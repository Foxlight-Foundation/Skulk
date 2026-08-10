# pyright: reportPrivateUsage=false
"""Generation-selection invariants for cache-to-store reconciliation."""

from skulk.api.main import _select_reconciliation_generations


def _replica(
    *,
    model_id: str,
    verification: str,
    registry_card_id: str | None = None,
    owner_model_id: str | None = None,
    owner_card_id: str | None = None,
    artifact_role: str = "base",
) -> list[tuple[str, str, dict[str, object]]]:
    return [
        (
            "node-a",
            "http://node-a.invalid",
            {
                "modelId": model_id,
                "verificationState": verification,
                "registryCardId": registry_card_id,
                "ownerModelId": owner_model_id,
                "ownerCardId": owner_card_id,
                "artifactRole": artifact_role,
            },
        )
    ]


def test_reconciliation_selects_one_current_generation_per_artifact() -> None:
    current_id = "card_current"
    replicas = {
        ("aaa_stale", "1" * 64): _replica(
            model_id="org/base",
            verification="registry_verified",
            registry_card_id="card_stale",
        ),
        ("zzz_current", "2" * 64): _replica(
            model_id="org/base",
            verification="local_legacy",
            registry_card_id=current_id,
        ),
        ("aaa_old_companion", "3" * 64): _replica(
            model_id="org/sidecar",
            verification="registry_verified",
            owner_model_id="org/base",
            owner_card_id="card_stale",
            artifact_role="mtp_sidecar",
        ),
        ("zzz_current_companion", "4" * 64): _replica(
            model_id="org/sidecar",
            verification="local_legacy",
            owner_model_id="org/base",
            owner_card_id=current_id,
            artifact_role="mtp_sidecar",
        ),
    }

    selected = _select_reconciliation_generations(
        replicas,
        {"org/base": current_id},
    )

    assert set(selected) == {
        ("zzz_current", "2" * 64),
        ("zzz_current_companion", "4" * 64),
    }


def test_reconciliation_prefers_verified_generation_without_current_card() -> None:
    replicas = {
        ("aaa_local", "1" * 64): _replica(
            model_id="org/base",
            verification="local_legacy",
        ),
        ("zzz_verified", "2" * 64): _replica(
            model_id="org/base",
            verification="registry_verified",
        ),
    }

    selected = _select_reconciliation_generations(replicas, {})

    assert set(selected) == {("zzz_verified", "2" * 64)}
