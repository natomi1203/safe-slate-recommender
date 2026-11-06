from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
import json

import pytest

from safe_slate import (
    audit_ope_readiness,
    audit_ope_readiness_with_contract,
    classify_evaluation_evidence,
    create_ope_evaluation_contract,
    estimate_cross_fitted_policy_value,
    load_ope_evaluation_contract,
)
from safe_slate.ope import _audit_ope_readiness_with_contract
from safe_slate.estimation import _validate_evaluation_contract_binding


def _contract(**overrides: object):
    values: dict[str, object] = {
        "evaluation_id": "purchase-ope-final-v1",
        "scientific_question": "Does the target improve settled purchases per request?",
        "baseline_policy": "baseline-v1",
        "target_policy": "target-v2",
        "population_identity": "jp-home-feed-eligible-v3",
        "pilot_window_start": "2026-08-01T00:00:00+09:00",
        "pilot_window_end": "2026-08-08T00:00:00+09:00",
        "final_window_start": "2026-09-01T00:00:00+09:00",
        "final_window_end": "2026-09-08T00:00:00+09:00",
        "earliest_analysis_time": "2026-10-08T00:00:00+09:00",
        "label_settlement_rule": "30-day purchase window plus ingestion lag",
        "primary_metric": "settled purchases per request",
        "primary_contrast": "target policy value minus logging policy value",
        "propensity_mode": "known",
        "input_manifest_digest": "a" * 64,
        "code_revision": "09c7f11",
        "dependency_lock_digest": "b" * 64,
        "output_namespace": "evidence/purchase-ope-final-v1",
        "minimum_support_rate_for_full_estimand": 1.0,
        "minimum_support_rate_for_partial_estimand": 0.8,
        "minimum_user_effective_sample_size": 1.5,
        "minimum_purchase_events": 2,
        "maximum_missing_outcome_rate": 0.25,
        "maximum_p99_importance_weight": 5.0,
        "weight_clip": 10.0,
        "bootstrap_repetitions": 2_000,
        "bootstrap_seed": 16_630,
        "confidence_level": 0.95,
        "outcome_lower_bound": 0.0,
        "outcome_upper_bound": 1.0,
        "minimum_raw_requests": 4,
        "maximum_runtime_seconds": 3_600.0,
        "maximum_memory_mb": 4_096,
        "maximum_output_bytes": 10_000_000,
    }
    values.update(overrides)
    return create_ope_evaluation_contract(**values)  # type: ignore[arg-type]


def _row_evidence(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "request_ids": ["r0", "r1", "r2", "r3"],
        "user_ids": ["u0", "u0", "u1", "u1"],
        "guaranteed_target_support_mass": [1.0] * 4,
        "target_probability_of_logged_action": [0.5] * 4,
        "logging_propensities": [0.5] * 4,
        "purchase_events": [1, 0, 1, 0],
        "fold_ids": [0, 0, 1, 1],
        "candidate_sets_reconstructable": True,
        "labels_settled": True,
        "missingness_independent": True,
    }
    values.update(overrides)
    return values


def test_contract_is_frozen_and_has_stable_digest() -> None:
    first = _contract()
    second = _contract()

    assert first == second
    assert len(first.contract_digest) == 64
    assert first.verify_digest()
    assert "production_rollout" in first.explicitly_unauthorized
    assert first.stage_order[-1] == "independent_final_audit"
    with pytest.raises(FrozenInstanceError):
        first.bootstrap_seed = 1  # type: ignore[misc]


def test_contract_round_trip_rejects_tampering() -> None:
    contract = _contract()
    payload = contract.to_dict()

    assert load_ope_evaluation_contract(payload) == contract
    payload["weight_clip"] = 999.0
    with pytest.raises(ValueError, match="digest mismatch"):
        load_ope_evaluation_contract(payload)


def test_recomputed_digest_cannot_bypass_canonical_semantics() -> None:
    contract = _contract()
    payload = contract.to_dict()
    payload["analysis_unit"] = "request"
    unsigned = dict(payload)
    unsigned.pop("contract_digest")
    payload["contract_digest"] = sha256(
        json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    with pytest.raises(ValueError, match="canonical schema"):
        load_ope_evaluation_contract(payload)


@pytest.mark.parametrize(
    "overrides",
    [
        {"pilot_window_start": "2026-08-01T00:00:00"},
        {"pilot_window_end": "2026-09-02T00:00:00+09:00"},
        {"earliest_analysis_time": "2026-09-07T00:00:00+09:00"},
        {"baseline_policy": "target-v2"},
        {"input_manifest_digest": "not-a-digest"},
        {"minimum_support_rate_for_partial_estimand": 1.0},
        {"confidence_level": 1.0},
        {"outcome_lower_bound": 1.0, "outcome_upper_bound": 1.0},
        {"minimum_raw_requests": 0},
        {"maximum_runtime_seconds": 0.0},
        {"authorized_scope": ["same"], "explicitly_unauthorized": ["same"]},
        {"pilot_allowed_uses": ["same"], "pilot_forbidden_uses": ["same"]},
    ],
)
def test_invalid_contract_boundaries_are_rejected(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        _contract(**overrides)


def test_contract_bound_readiness_uses_frozen_thresholds_and_digest() -> None:
    contract = _contract()
    readiness = audit_ope_readiness_with_contract(
        contract,
        analysis_timestamp="2026-10-08T00:00:00+09:00",
        **_row_evidence(),
    )

    assert readiness.decision == "partial"
    assert not readiness.standard_ope_authorized
    assert "population_positivity_not_established" in readiness.reasons
    assert readiness.evaluation_contract_digest == contract.contract_digest
    assert readiness.minimum_raw_requests == contract.minimum_raw_requests
    assert readiness.weight_clip == contract.weight_clip
    assert readiness.minimum_user_effective_sample_size == (
        contract.minimum_user_effective_sample_size
    )


def test_contract_bound_readiness_rejects_early_or_overridden_analysis() -> None:
    contract = _contract()
    with pytest.raises(ValueError, match="labels_pending"):
        audit_ope_readiness_with_contract(
            contract,
            analysis_timestamp="2026-10-07T23:59:59+09:00",
            **_row_evidence(),
        )
    with pytest.raises(ValueError, match="may not be overridden"):
        audit_ope_readiness_with_contract(
            contract,
            analysis_timestamp="2026-10-08T00:00:00+09:00",
            **_row_evidence(weight_clip=1.0),
        )


def test_contract_minimum_raw_request_gate_fails_closed() -> None:
    contract = _contract(minimum_raw_requests=5)
    readiness = audit_ope_readiness_with_contract(
        contract,
        analysis_timestamp="2026-10-08T00:00:00+09:00",
        **_row_evidence(),
    )

    assert readiness.decision == "fail"
    assert "raw_request_count_below_gate" in readiness.reasons


def test_unbound_readiness_is_explicitly_unbound() -> None:
    readiness = audit_ope_readiness(
        **_row_evidence(),
        propensity_mode="known",
        minimum_user_effective_sample_size=1.5,
        minimum_purchase_events=2,
        maximum_missing_outcome_rate=0.25,
        maximum_allowed_p99_importance_weight=5.0,
        weight_clip=10.0,
    )

    assert readiness.evaluation_contract_digest is None
    assert readiness.minimum_raw_requests == 1


def test_estimate_inherits_contract_digest_and_frozen_inference() -> None:
    contract = _contract(bootstrap_repetitions=50)
    rows = _row_evidence()
    readiness = _audit_ope_readiness_with_contract(
        contract,
        analysis_timestamp="2026-10-08T00:00:00+09:00",
        _population_positivity_verified=True,
        **rows,
    )
    estimator_rows = {
        key: rows[key]
        for key in (
            "request_ids",
            "user_ids",
            "guaranteed_target_support_mass",
            "target_probability_of_logged_action",
            "logging_propensities",
            "purchase_events",
            "fold_ids",
        )
    }
    estimate = estimate_cross_fitted_policy_value(
        readiness,
        **estimator_rows,  # type: ignore[arg-type]
        outcome_prediction_logged_action=[0.4] * 4,
        outcome_prediction_target_policy=[0.6] * 4,
        nuisance_predictions_cross_fitted=True,
        logging_propensities_are_request_time_probabilities=True,
        outcomes_match_logged_actions=True,
        bootstrap_repetitions=contract.bootstrap_repetitions,
        bootstrap_seed=contract.bootstrap_seed,
        confidence_level=contract.confidence_level,
        evaluation_contract=contract,
    )

    assert estimate.evaluation_contract_digest == contract.contract_digest
    assert estimate.bootstrap_seed == contract.bootstrap_seed


def test_contract_bound_readiness_cannot_be_estimated_without_matching_contract() -> (
    None
):
    contract = _contract(bootstrap_repetitions=10)
    other_contract = _contract(
        evaluation_id="other-evaluation",
        output_namespace="evidence/other-evaluation",
        bootstrap_repetitions=10,
    )
    rows = _row_evidence()
    readiness = _audit_ope_readiness_with_contract(
        contract,
        analysis_timestamp="2026-10-08T00:00:00+09:00",
        _population_positivity_verified=True,
        **rows,
    )
    estimator_rows = {
        key: rows[key]
        for key in (
            "request_ids",
            "user_ids",
            "guaranteed_target_support_mass",
            "target_probability_of_logged_action",
            "logging_propensities",
            "purchase_events",
            "fold_ids",
        )
    }
    common = {
        **estimator_rows,
        "outcome_prediction_logged_action": [0.4] * 4,
        "outcome_prediction_target_policy": [0.6] * 4,
        "nuisance_predictions_cross_fitted": True,
        "logging_propensities_are_request_time_probabilities": True,
        "outcomes_match_logged_actions": True,
        "bootstrap_repetitions": 10,
        "bootstrap_seed": contract.bootstrap_seed,
    }

    with pytest.raises(ValueError, match="requires the matching"):
        estimate_cross_fitted_policy_value(readiness, **common)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="not bound to this"):
        estimate_cross_fitted_policy_value(
            readiness,
            **common,  # type: ignore[arg-type]
            evaluation_contract=other_contract,
        )
    with pytest.raises(ValueError, match="settings differ"):
        estimate_cross_fitted_policy_value(
            readiness,
            **{**common, "bootstrap_seed": 1},  # type: ignore[arg-type]
            evaluation_contract=contract,
        )


def test_estimation_contract_binding_rejects_tampering_and_bound_drift() -> None:
    contract = _contract(bootstrap_repetitions=10)
    readiness = _audit_ope_readiness_with_contract(
        contract,
        analysis_timestamp="2026-10-08T00:00:00+09:00",
        _population_positivity_verified=True,
        **_row_evidence(),
    )
    common = {
        "bootstrap_repetitions": contract.bootstrap_repetitions,
        "bootstrap_seed": contract.bootstrap_seed,
        "confidence_level": contract.confidence_level,
    }

    with pytest.raises(ValueError, match="tampered"):
        _validate_evaluation_contract_binding(
            replace(contract, contract_digest="0" * 64),
            readiness,
            **common,
        )
    with pytest.raises(ValueError, match="settings differ"):
        _validate_evaluation_contract_binding(
            contract,
            readiness,
            **common,
            outcome_lower_bound=contract.outcome_lower_bound,
        )
    with pytest.raises(ValueError, match="settings differ"):
        _validate_evaluation_contract_binding(
            contract,
            readiness,
            **common,
            outcome_lower_bound=-1.0,
            outcome_upper_bound=contract.outcome_upper_bound,
        )

    _validate_evaluation_contract_binding(
        contract,
        readiness,
        **common,
        outcome_lower_bound=contract.outcome_lower_bound,
        outcome_upper_bound=contract.outcome_upper_bound,
    )


def _evidence_status(**overrides: object):
    contract = overrides.pop("contract", _contract())
    values: dict[str, object] = {
        "analysis_timestamp": "2026-10-08T00:00:00+09:00",
        "collection_complete": True,
        "immutable_inputs_verified": True,
        "no_clobber_verified": True,
        "labels_settled": True,
        "label_snapshot_digest": "c" * 64,
        "readiness_decision": "pass",
        "readiness_contract_digest": getattr(contract, "contract_digest", None),
        "estimation_complete": True,
        "final_audit_passed": True,
    }
    values.update(overrides)
    return classify_evaluation_evidence(contract, **values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("overrides", "state", "reason"),
    [
        ({"collection_complete": False}, "partial_invalid", "collection_incomplete"),
        (
            {"immutable_inputs_verified": False},
            "evidence_rejected",
            "immutable_inputs_not_verified",
        ),
        (
            {"no_clobber_verified": False},
            "evidence_rejected",
            "no_clobber_not_verified",
        ),
        (
            {
                "analysis_timestamp": "2026-10-07T00:00:00+09:00",
                "labels_settled": False,
            },
            "labels_pending",
            "labels_not_yet_eligible_or_settled",
        ),
        (
            {"label_snapshot_digest": None},
            "evidence_rejected",
            "label_snapshot_digest_invalid",
        ),
        ({"readiness_decision": None}, "analysis_ready", "readiness_not_run"),
        (
            {"readiness_contract_digest": "d" * 64},
            "evidence_rejected",
            "readiness_contract_mismatch",
        ),
        (
            {"readiness_decision": "fail"},
            "insufficient_evidence",
            "statistical_or_support_gates_failed",
        ),
        (
            {"estimation_complete": False},
            "analysis_ready",
            "estimation_not_complete",
        ),
        (
            {"final_audit_passed": None},
            "artifact_complete",
            "independent_final_audit_pending",
        ),
        (
            {"final_audit_passed": False},
            "evidence_rejected",
            "independent_final_audit_failed",
        ),
    ],
)
def test_evidence_states_remain_separate(
    overrides: dict[str, object], state: str, reason: str
) -> None:
    status = _evidence_status(**overrides)

    assert status.overall_state == state
    assert reason in status.reasons
    assert not status.decision_eligible
    assert status.authorized_actions == ()


def test_decision_eligible_authorizes_only_contract_scope() -> None:
    status = _evidence_status()

    assert status.overall_state == "decision_eligible"
    assert status.collection_complete
    assert status.analysis_ready
    assert status.statistically_sufficient
    assert status.decision_eligible
    assert status.authorized_actions == ("offline_research_reporting",)
    assert "production_rollout" in status.explicitly_unauthorized
    assert status.to_dict()["overall_state"] == "decision_eligible"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"pilot_window_start": ""}, "non-empty ISO-8601"),
        ({"pilot_window_start": "not-a-time"}, "ISO-8601"),
        ({"evaluation_id": ""}, "non-empty string"),
        ({"maximum_missing_outcome_rate": True}, "finite and in"),
        ({"maximum_missing_outcome_rate": 1.1}, "finite and in"),
        ({"minimum_purchase_events": True}, "nonnegative integer"),
        ({"propensity_mode": "missing"}, "known or interval"),
        ({"authorized_scope": ()}, "non-empty and unique"),
    ],
)
def test_contract_rejects_remaining_malformed_values(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _contract(**overrides)


def test_contract_loader_rejects_type_and_schema_drift() -> None:
    with pytest.raises(ValueError, match="must be a mapping"):
        load_ope_evaluation_contract([])  # type: ignore[arg-type]
    payload = _contract().to_dict()
    payload.pop("evaluation_id")
    with pytest.raises(ValueError, match="fields mismatch"):
        load_ope_evaluation_contract(payload)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"contract": "bad"}, "untampered"),
        ({"collection_complete": 1}, "must be boolean"),
        ({"final_audit_passed": "yes"}, "boolean or None"),
        ({"readiness_decision": "unknown"}, "pass, partial, fail"),
    ],
)
def test_evidence_classifier_rejects_invalid_public_contract(
    overrides: dict[str, object], message: str
) -> None:
    contract = overrides.pop("contract", _contract())
    with pytest.raises(ValueError, match=message):
        _evidence_status(contract=contract, **overrides)
