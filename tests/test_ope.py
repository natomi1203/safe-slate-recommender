from __future__ import annotations

import math

import pytest

from safe_slate import audit_ope_readiness
from safe_slate.ope import (
    _audit_ope_readiness,
    _audit_ope_readiness_with_contract,
    _digest_value,
    _quantile,
)
from test_contract import _contract as _ope_contract, _row_evidence


def _base_audit(**overrides: object):
    values: dict[str, object] = {
        "request_ids": ["r0", "r1", "r2", "r3"],
        "user_ids": ["u0", "u0", "u1", "u1"],
        "guaranteed_target_support_mass": [1.0, 1.0, 1.0, 1.0],
        "target_probability_of_logged_action": [0.5, 0.5, 0.5, 0.5],
        "purchase_events": [1, 0, 1, 0],
        "fold_ids": [0, 0, 1, 1],
        "propensity_mode": "known",
        "logging_propensities": [0.5, 0.5, 0.5, 0.5],
        "candidate_sets_reconstructable": True,
        "labels_settled": True,
        "missingness_independent": True,
        "population_positivity_established": True,
        "minimum_user_effective_sample_size": 1.5,
        "minimum_purchase_events": 2,
        "maximum_missing_outcome_rate": 0.0,
        "maximum_allowed_p99_importance_weight": 5.0,
        "weight_clip": 10.0,
    }
    values.update(overrides)
    return _audit_ope_readiness(  # type: ignore[arg-type]
        **values,
        _population_positivity_verified=True,
    )


def test_known_propensity_full_support_authorizes_standard_ope() -> None:
    assessment = _base_audit()

    assert assessment.decision == "pass"
    assert assessment.authorized_estimand == "registered_full_target_population"
    assert assessment.standard_ope_authorized
    assert not assessment.supported_subpopulation_analysis_authorized
    assert not assessment.partial_identification_authorized
    assert not assessment.production_rollout_authorized
    assert assessment.user_effective_sample_size == pytest.approx(2.0)
    assert assessment.user_effective_sample_size_lower_bound == pytest.approx(2.0)
    assert assessment.effective_sample_size_kind == "exact_user_block"
    assert assessment.repeated_user_request_fraction == pytest.approx(0.5)
    assert assessment.maximum_requests_per_user == 2
    assert assessment.cross_fit_ready
    assert len(assessment.row_data_digest) == 64
    assert assessment.reasons == ()


def test_finite_log_full_support_does_not_establish_population_positivity() -> None:
    assessment = _base_audit(population_positivity_established=False)

    assert assessment.support_rate == 1.0
    assert not assessment.population_positivity_established
    assert assessment.decision == "partial"
    assert assessment.authorized_estimand == "observed_finite_log_population_only"
    assert not assessment.standard_ope_authorized
    assert "population_positivity_not_established" in assessment.reasons
    assert (
        "finite_log_support_does_not_imply_population_positivity"
        in assessment.reasons
    )


def test_bare_positivity_boolean_cannot_authorize_standard_ope() -> None:
    values = {
        "request_ids": ["r0", "r1"],
        "user_ids": ["u0", "u1"],
        "guaranteed_target_support_mass": [1.0, 1.0],
        "target_probability_of_logged_action": [1.0, 1.0],
        "purchase_events": [1, 0],
        "fold_ids": [0, 1],
        "propensity_mode": "known",
        "logging_propensities": [1.0, 1.0],
        "candidate_sets_reconstructable": True,
        "labels_settled": True,
        "missingness_independent": True,
        "population_positivity_established": True,
        "minimum_user_effective_sample_size": 1.0,
        "minimum_purchase_events": 1,
        "maximum_missing_outcome_rate": 0.0,
        "maximum_allowed_p99_importance_weight": 2.0,
        "weight_clip": 2.0,
    }

    assessment = audit_ope_readiness(**values)  # type: ignore[arg-type]

    assert assessment.decision == "partial"
    assert not assessment.standard_ope_authorized
    assert "population_positivity_evidence_not_manifest_bound" in assessment.reasons


def test_row_data_digest_is_stable_and_sensitive_to_audited_rows() -> None:
    first = _base_audit()
    second = _base_audit()
    changed = _base_audit(request_ids=["changed", "r1", "r2", "r3"])

    assert first.row_data_digest == second.row_data_digest
    assert first.row_data_digest != changed.row_data_digest


def test_known_propensity_partial_support_limits_the_estimand() -> None:
    assessment = _base_audit(guaranteed_target_support_mass=[0.9, 0.9, 0.9, 0.9])

    assert assessment.decision == "partial"
    assert assessment.authorized_estimand == "full_population_partial_support_bounds"
    assert not assessment.standard_ope_authorized
    assert not assessment.supported_subpopulation_analysis_authorized
    assert assessment.partial_identification_authorized
    assert "full_estimand_support_below_threshold" in assessment.reasons
    assert "partial_support_requires_partial_identification" in assessment.reasons


def test_ninety_nine_percent_support_does_not_identify_full_population() -> None:
    assessment = _base_audit(
        guaranteed_target_support_mass=[0.99, 0.99, 0.99, 0.99]
    )

    assert assessment.decision == "partial"
    assert assessment.authorized_estimand == "full_population_partial_support_bounds"
    assert not assessment.standard_ope_authorized
    assert assessment.partial_identification_authorized


def test_support_below_eighty_percent_allows_descriptive_replay_only() -> None:
    assessment = _base_audit(guaranteed_target_support_mass=[0.79, 0.79, 0.79, 0.79])

    assert assessment.decision == "fail"
    assert assessment.authorized_estimand == "descriptive_replay_only"
    assert not assessment.standard_ope_authorized
    assert "target_support_below_partial_gate" in assessment.reasons


def test_covering_propensity_intervals_only_authorize_partial_identification() -> None:
    assessment = _base_audit(
        propensity_mode="interval",
        logging_propensities=None,
        logging_propensity_lower=[0.4, 0.4, 0.4, 0.4],
        logging_propensity_upper=[0.6, 0.6, 0.6, 0.6],
        minimum_user_effective_sample_size=0.5,
    )

    assert assessment.decision == "partial"
    assert assessment.authorized_estimand == (
        "full_population_propensity_interval_bounds"
    )
    assert not assessment.standard_ope_authorized
    assert assessment.partial_identification_authorized
    assert assessment.user_effective_sample_size is None
    assert assessment.user_effective_sample_size_lower_bound == pytest.approx(8.0 / 9.0)
    assert assessment.effective_sample_size_kind == (
        "conservative_user_block_lower_bound"
    )
    assert "propensity_interval_requires_partial_identification" in (assessment.reasons)


def test_partial_support_and_propensity_intervals_require_joint_bounds() -> None:
    assessment = _base_audit(
        propensity_mode="interval",
        logging_propensities=None,
        logging_propensity_lower=[0.4, 0.4, 0.4, 0.4],
        logging_propensity_upper=[0.6, 0.6, 0.6, 0.6],
        guaranteed_target_support_mass=[0.9, 0.9, 0.9, 0.9],
        minimum_user_effective_sample_size=0.5,
    )

    assert assessment.decision == "partial"
    assert assessment.authorized_estimand == (
        "full_population_joint_propensity_and_support_bounds"
    )
    assert assessment.partial_identification_authorized


def test_interval_touching_zero_fails_when_target_probability_is_positive() -> None:
    assessment = _base_audit(
        propensity_mode="interval",
        logging_propensities=None,
        logging_propensity_lower=[0.0, 0.4, 0.4, 0.4],
        logging_propensity_upper=[0.6, 0.6, 0.6, 0.6],
        minimum_user_effective_sample_size=0.5,
    )

    assert assessment.decision == "fail"
    assert "propensity_interval_not_strictly_positive" in assessment.reasons
    assert "propensity_invalid" in assessment.reasons


def test_missing_propensities_fail_closed() -> None:
    assessment = _base_audit(
        propensity_mode="missing",
        logging_propensities=None,
    )

    assert assessment.decision == "fail"
    assert assessment.effective_sample_size_kind == "unavailable"
    assert "propensities_missing" in assessment.reasons
    assert "user_effective_sample_size_unavailable" in assessment.reasons


def test_user_crossing_cross_fit_folds_fails_closed() -> None:
    assessment = _base_audit(fold_ids=[0, 1, 1, 1])

    assert not assessment.cross_fit_ready
    assert assessment.decision == "fail"
    assert "user_crosses_cross_fit_folds" in assessment.reasons


def test_missing_outcomes_and_event_count_are_separate_gates() -> None:
    missing = _base_audit(
        purchase_events=[1, None, 1, 0],
        maximum_missing_outcome_rate=0.2,
    )
    too_few_events = _base_audit(
        purchase_events=[1, 0, 0, 0],
        minimum_purchase_events=2,
    )

    assert missing.missing_outcome_rate == pytest.approx(0.25)
    assert "missing_outcome_rate_above_gate" in missing.reasons
    assert "observation_propensities_missing" in missing.reasons
    assert "purchase_events_below_gate" in too_few_events.reasons
    assert missing.decision == too_few_events.decision == "fail"


def test_observation_propensities_enable_missing_outcome_readiness() -> None:
    assessment = _base_audit(
        purchase_events=[1, None, 1, 0],
        outcome_observation_propensities=[1.0, 0.5, 1.0, 1.0],
        maximum_missing_outcome_rate=0.25,
        minimum_user_effective_sample_size=1.5,
    )

    assert assessment.decision == "pass"
    assert assessment.observation_propensity_available
    assert assessment.minimum_observation_propensity == pytest.approx(0.5)
    assert assessment.maximum_observation_inverse_weight == pytest.approx(2.0)
    assert assessment.user_effective_sample_size == pytest.approx(1.8)


def test_unobserved_rows_still_enter_combined_weight_tail_gate() -> None:
    assessment = _base_audit(
        purchase_events=[1, None, 1, 0],
        outcome_observation_propensities=[1.0, 0.01, 1.0, 1.0],
        maximum_missing_outcome_rate=0.25,
        maximum_allowed_p99_importance_weight=5.0,
        minimum_user_effective_sample_size=1.5,
    )

    assert assessment.decision == "fail"
    assert assessment.p99_importance_weight_upper_bound is not None
    assert assessment.p99_importance_weight_upper_bound > 5.0
    assert "importance_weight_tail_above_gate" in assessment.reasons


def test_invalid_observation_propensity_fails_closed() -> None:
    assessment = _base_audit(
        purchase_events=[1, None, 1, 0],
        outcome_observation_propensities=[1.0, 0.0, 1.0, 1.0],
        maximum_missing_outcome_rate=0.25,
    )

    assert assessment.decision == "fail"
    assert "observation_propensity_invalid" in assessment.reasons
    assert "user_effective_sample_size_unavailable" in assessment.reasons


def test_weight_tail_and_user_ess_are_computed_after_user_blocking() -> None:
    heavy_tail = _base_audit(
        logging_propensities=[0.01, 0.5, 0.5, 0.5],
        maximum_allowed_p99_importance_weight=5.0,
    )
    low_ess = _base_audit(minimum_user_effective_sample_size=2.1)

    assert heavy_tail.p99_importance_weight_upper_bound is not None
    assert heavy_tail.p99_importance_weight_upper_bound > 5.0
    assert "importance_weight_tail_above_gate" in heavy_tail.reasons
    assert "user_effective_sample_size_below_gate" in low_ess.reasons
    assert heavy_tail.decision == low_ess.decision == "fail"


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        (
            {"candidate_sets_reconstructable": False},
            "candidate_sets_not_reconstructable",
        ),
        ({"labels_settled": None}, "settlement_unknown"),
        ({"missingness_independent": False}, "outcome_observation_not_independent"),
        (
            {"guaranteed_target_support_mass": [1.0, math.nan, 1.0, 1.0]},
            "target_support_mass_invalid",
        ),
        (
            {"target_probability_of_logged_action": [0.5, -0.1, 0.5, 0.5]},
            "target_probability_invalid",
        ),
        (
            {"guaranteed_target_support_mass": [0.4, 1.0, 1.0, 1.0]},
            "logged_action_probability_exceeds_guaranteed_support_mass",
        ),
        ({"purchase_events": [1, -1, 1, 0]}, "purchase_event_invalid"),
        ({"logging_propensities": [0.5, 0.0, 0.5, 0.5]}, "propensity_invalid"),
    ],
)
def test_runtime_evidence_errors_fail_closed(
    overrides: dict[str, object], reason: str
) -> None:
    assessment = _base_audit(**overrides)

    assert assessment.decision == "fail"
    assert reason in assessment.reasons


@pytest.mark.parametrize(
    "overrides",
    [
        {"request_ids": []},
        {"user_ids": ["u0"]},
        {"request_ids": ["r0", "r0", "r2", "r3"]},
        {"propensity_mode": "known", "logging_propensities": None},
        {
            "propensity_mode": "interval",
            "logging_propensities": None,
            "logging_propensity_lower": [0.4],
            "logging_propensity_upper": [0.6],
        },
    ],
)
def test_structural_contract_errors_raise(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _base_audit(**overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"full_estimand_support_threshold": 1.1},
        {
            "full_estimand_support_threshold": 0.7,
            "partial_estimand_support_threshold": 0.8,
        },
        {"minimum_user_effective_sample_size": 0.0},
        {"minimum_purchase_events": -1},
        {"maximum_missing_outcome_rate": math.nan},
        {"maximum_allowed_p99_importance_weight": math.inf},
        {"weight_clip": 0.0},
    ],
)
def test_invalid_frozen_thresholds_raise(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _base_audit(**overrides)


def test_assessment_serializes_without_authorizing_rollout() -> None:
    serialized = _base_audit().to_dict()

    assert serialized["decision"] == "pass"
    assert serialized["unique_users"] == 2
    assert serialized["user_effective_sample_size"] == pytest.approx(2.0)
    assert serialized["production_rollout_authorized"] is False
    assert serialized["reasons"] == []


def test_private_digest_and_quantile_helpers_cover_empty_nonfinite_and_unsupported() -> None:
    assert _quantile([], 0.5) is None
    assert _digest_value(math.nan) == "__nonfinite__:nan"
    assert _digest_value(math.inf) == "__nonfinite__:positive_infinity"
    assert _digest_value(-math.inf) == "__nonfinite__:negative_infinity"
    assert _digest_value(object()).startswith("__unsupported__:object")


@pytest.mark.parametrize(
    "overrides",
    [
        {"propensity_mode": "unknown"},
        {"population_positivity_established": 1},
        {"minimum_raw_requests": 0},
        {"_evaluation_contract_digest": "bad"},
        {"fold_ids": [0]},
        {"outcome_observation_propensities": [1.0]},
        {"request_ids": ["", "r1", "r2", "r3"]},
        {"user_ids": ["", "u0", "u1", "u1"]},
        {"logging_propensities": [0.5]},
        {
            "propensity_mode": "interval",
            "logging_propensities": [0.5] * 4,
            "logging_propensity_lower": [0.4] * 4,
            "logging_propensity_upper": [0.6] * 4,
        },
        {"propensity_mode": "missing", "logging_propensities": [0.5] * 4},
    ],
)
def test_remaining_structural_readiness_contracts_raise(
    overrides: dict[str, object]
) -> None:
    with pytest.raises(ValueError):
        _base_audit(**overrides)


def test_remaining_fail_closed_readiness_paths() -> None:
    no_outcomes = _base_audit(
        purchase_events=[None] * 4,
        outcome_observation_propensities=[1.0] * 4,
        maximum_missing_outcome_rate=1.0,
        minimum_purchase_events=0,
    )
    assert "no_analyzable_requests" in no_outcomes.reasons

    missing_folds = _base_audit(fold_ids=None)
    invalid_folds = _base_audit(fold_ids=[0, True, 1, 1])
    one_fold = _base_audit(fold_ids=[0, 0, 0, 0])
    assert "cross_fit_folds_missing" in missing_folds.reasons
    assert "cross_fit_fold_invalid" in invalid_folds.reasons
    assert "cross_fit_requires_two_folds" in one_fold.reasons

    invalid_interval = _base_audit(
        propensity_mode="interval",
        logging_propensities=None,
        logging_propensity_lower=[0.6] * 4,
        logging_propensity_upper=[0.4] * 4,
    )
    assert "propensity_invalid" in invalid_interval.reasons

    zero_mass = _base_audit(
        target_probability_of_logged_action=[0.0] * 4,
        minimum_user_effective_sample_size=0.1,
    )
    assert "zero_target_weight_mass" in zero_mass.reasons

    finite_interval = _base_audit(
        propensity_mode="interval",
        logging_propensities=None,
        logging_propensity_lower=[0.4] * 4,
        logging_propensity_upper=[0.6] * 4,
        minimum_user_effective_sample_size=0.5,
        population_positivity_established=False,
    )
    assert finite_interval.authorized_estimand == (
        "observed_finite_log_propensity_interval_bounds"
    )

    below_with_failure = _base_audit(
        guaranteed_target_support_mass=[0.5] * 4,
        candidate_sets_reconstructable=False,
    )
    assert "target_support_below_partial_gate" in below_with_failure.reasons


@pytest.mark.parametrize(
    ("contract", "timestamp", "message"),
    [
        ("bad", "2026-10-08T00:00:00+09:00", "untampered"),
        (None, "not-a-time", "ISO-8601"),
        (None, "2026-10-08T00:00:00", "timezone"),
    ],
)
def test_contract_bound_readiness_rejects_invalid_contract_and_timestamps(
    contract, timestamp: str, message: str
) -> None:
    value = contract if contract is not None else _ope_contract()
    with pytest.raises(ValueError, match=message):
        _audit_ope_readiness_with_contract(
            value,  # type: ignore[arg-type]
            analysis_timestamp=timestamp,
            _population_positivity_verified=False,
            **_row_evidence(),
        )


def test_contract_bound_readiness_accepts_z_earliest_timestamp() -> None:
    contract = _ope_contract(earliest_analysis_time="2026-10-07T15:00:00Z")
    assessment = _audit_ope_readiness_with_contract(
        contract,
        analysis_timestamp="2026-10-08T00:00:00+09:00",
        _population_positivity_verified=False,
        **_row_evidence(),
    )
    assert assessment.evaluation_contract_digest == contract.contract_digest
