from __future__ import annotations

import math

import pytest

from safe_slate import (
    estimate_cross_fitted_policy_value,
    estimate_joint_uncertainty_bounds,
    estimate_missingness_adjusted_policy_value,
    estimate_partial_identification_bounds,
    estimate_propensity_interval_bounds,
)
from safe_slate.ope import _audit_ope_readiness


def _row_data(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "request_ids": ["r0", "r1", "r2", "r3"],
        "user_ids": ["u0", "u0", "u1", "u1"],
        "guaranteed_target_support_mass": [1.0, 1.0, 1.0, 1.0],
        "target_probability_of_logged_action": [0.5, 0.5, 0.5, 0.5],
        "logging_propensities": [0.5, 0.5, 0.5, 0.5],
        "purchase_events": [1, 0, 1, 0],
        "fold_ids": [0, 0, 1, 1],
    }
    values.update(overrides)
    return values


def _readiness(row_data: dict[str, object], **overrides: object):
    values: dict[str, object] = {
        **row_data,
        "propensity_mode": "known",
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


def _estimate(
    readiness,
    row_data: dict[str, object],
    **overrides: object,
):
    values: dict[str, object] = {
        **row_data,
        "outcome_prediction_logged_action": [0.4, 0.4, 0.4, 0.4],
        "outcome_prediction_target_policy": [0.6, 0.6, 0.6, 0.6],
        "nuisance_predictions_cross_fitted": True,
        "logging_propensities_are_request_time_probabilities": True,
        "outcomes_match_logged_actions": True,
        "bootstrap_repetitions": 200,
        "bootstrap_seed": 16630,
    }
    values.update(overrides)
    return estimate_cross_fitted_policy_value(
        readiness,
        **values,  # type: ignore[arg-type]
    )


def _partial_rows(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {"guaranteed_target_support_mass": [0.9, 0.9, 0.9, 0.9]}
    values.update(overrides)
    return _row_data(**values)


def _partial_estimate(readiness, row_data: dict[str, object], **overrides: object):
    values: dict[str, object] = {
        **row_data,
        "outcome_prediction_logged_action": [0.4, 0.4, 0.4, 0.4],
        "outcome_prediction_supported_target_contribution": [0.54] * 4,
        "outcome_lower_bound": 0.0,
        "outcome_upper_bound": 1.0,
        "nuisance_predictions_cross_fitted": True,
        "logging_propensities_are_request_time_probabilities": True,
        "outcomes_match_logged_actions": True,
        "bootstrap_repetitions": 200,
        "bootstrap_seed": 16630,
    }
    values.update(overrides)
    return estimate_partial_identification_bounds(
        readiness,
        **values,  # type: ignore[arg-type]
    )


def _interval_rows(**overrides: object) -> dict[str, object]:
    values = _row_data()
    values.pop("logging_propensities")
    values.update(
        {
            "logging_propensity_lower": [0.4] * 4,
            "logging_propensity_upper": [0.6] * 4,
        }
    )
    values.update(overrides)
    return values


def _interval_readiness(row_data: dict[str, object], **overrides: object):
    values: dict[str, object] = {
        **row_data,
        "propensity_mode": "interval",
        "candidate_sets_reconstructable": True,
        "labels_settled": True,
        "missingness_independent": True,
        "population_positivity_established": True,
        "minimum_user_effective_sample_size": 0.5,
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


def _interval_estimate(readiness, row_data: dict[str, object], **overrides: object):
    support_mass = row_data["guaranteed_target_support_mass"]
    values: dict[str, object] = {
        **row_data,
        "outcome_prediction_logged_action": [0.4] * 4,
        "outcome_prediction_supported_target_contribution": [
            0.6 * float(value)
            for value in support_mass  # type: ignore[union-attr]
        ],
        "outcome_lower_bound": 0.0,
        "outcome_upper_bound": 1.0,
        "propensity_intervals_cover_true_probabilities": True,
        "nuisance_predictions_cross_fitted": True,
        "propensity_intervals_are_request_time": True,
        "outcomes_match_logged_actions": True,
        "bootstrap_repetitions": 200,
        "bootstrap_seed": 16630,
    }
    values.update(overrides)
    return estimate_propensity_interval_bounds(
        readiness,
        **values,  # type: ignore[arg-type]
    )


def _missing_rows(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "purchase_events": [1, None, 1, 0],
        "outcome_observation_propensities": [1.0, 0.5, 1.0, 1.0],
    }
    values.update(overrides)
    return _row_data(**values)


def _missing_readiness(row_data: dict[str, object], **overrides: object):
    values: dict[str, object] = {
        "maximum_missing_outcome_rate": 0.25,
        "minimum_user_effective_sample_size": 1.5,
    }
    values.update(overrides)
    return _readiness(row_data, **values)


def _missing_estimate(readiness, row_data: dict[str, object], **overrides: object):
    values: dict[str, object] = {
        **row_data,
        "outcome_prediction_logged_action": [0.4] * 4,
        "outcome_prediction_target_policy": [0.6] * 4,
        "nuisance_predictions_cross_fitted": True,
        "logging_propensities_are_request_time_probabilities": True,
        "observation_propensities_are_validated": True,
        "observation_propensities_are_request_time_probabilities": True,
        "outcomes_match_logged_actions": True,
        "conditional_missingness_independence": True,
        "bootstrap_repetitions": 200,
        "bootstrap_seed": 16630,
    }
    values.update(overrides)
    return estimate_missingness_adjusted_policy_value(
        readiness,
        **values,  # type: ignore[arg-type]
    )


def _joint_known_rows(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "guaranteed_target_support_mass": [0.9] * 4,
        "purchase_events": [1, None, 1, 0],
        "outcome_observation_propensities": [1.0, 0.5, 1.0, 1.0],
    }
    values.update(overrides)
    return _row_data(**values)


def _joint_interval_rows(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "guaranteed_target_support_mass": [0.9] * 4,
        "purchase_events": [1, None, 1, 0],
        "outcome_observation_propensities": [1.0, 0.5, 1.0, 1.0],
    }
    values.update(overrides)
    return _interval_rows(**values)


def _joint_readiness(row_data: dict[str, object], **overrides: object):
    values: dict[str, object] = {
        "maximum_missing_outcome_rate": 0.25,
        "minimum_user_effective_sample_size": 0.5,
    }
    values.update(overrides)
    if "logging_propensities" in row_data:
        return _readiness(row_data, **values)
    return _interval_readiness(row_data, **values)


def _joint_estimate(readiness, row_data: dict[str, object], **overrides: object):
    support_mass = row_data["guaranteed_target_support_mass"]
    values: dict[str, object] = {
        **row_data,
        "outcome_prediction_logged_action": [0.4] * 4,
        "outcome_prediction_supported_target_contribution": [
            0.6 * float(value)
            for value in support_mass  # type: ignore[union-attr]
        ],
        "outcome_lower_bound": 0.0,
        "outcome_upper_bound": 1.0,
        "nuisance_predictions_cross_fitted": True,
        "propensity_evidence_is_request_time": True,
        "observation_propensities_are_validated": True,
        "observation_propensities_are_request_time_probabilities": True,
        "outcomes_match_logged_actions": True,
        "conditional_missingness_independence": True,
        "bootstrap_repetitions": 200,
        "bootstrap_seed": 16630,
    }
    if readiness.propensity_mode == "interval":
        values["propensity_intervals_cover_true_probabilities"] = True
    if "outcome_observation_propensities" not in values:
        values["outcome_observation_propensities"] = [1.0] * 4
    values.update(overrides)
    return estimate_joint_uncertainty_bounds(
        readiness,
        **values,  # type: ignore[arg-type]
    )


def test_cross_fitted_dr_is_primary_and_ips_snips_are_diagnostics() -> None:
    rows = _row_data()
    readiness = _readiness(rows)
    estimate = _estimate(readiness, rows)

    assert estimate.logging_policy_value == pytest.approx(0.5)
    assert estimate.ips_target_value == pytest.approx(0.5)
    assert estimate.snips_target_value == pytest.approx(0.5)
    assert estimate.doubly_robust_target_value == pytest.approx(0.7)
    assert estimate.doubly_robust_lift == pytest.approx(0.2)
    assert estimate.doubly_robust_target_ci_low == pytest.approx(0.7)
    assert estimate.doubly_robust_target_ci_high == pytest.approx(0.7)
    assert estimate.doubly_robust_lift_ci_low == pytest.approx(0.2)
    assert estimate.doubly_robust_lift_ci_high == pytest.approx(0.2)
    assert estimate.bootstrap_unit == "pseudonymous_user"
    assert estimate.user_count == 2
    assert estimate.fold_count == 2
    assert estimate.conditional_causal_interpretation
    assert not estimate.production_rollout_authorized


def test_estimator_is_deterministic_for_a_frozen_seed() -> None:
    rows = _row_data(
        purchase_events=[1, 1, 0, 0],
        user_ids=["u0", "u0", "u1", "u1"],
    )
    readiness = _readiness(rows)

    first = _estimate(readiness, rows, bootstrap_seed=7)
    second = _estimate(readiness, rows, bootstrap_seed=7)

    assert first.to_dict() == second.to_dict()


def test_user_bootstrap_preserves_request_weighted_estimand() -> None:
    rows = _row_data(
        user_ids=["u0", "u1", "u1", "u1"],
        fold_ids=[0, 1, 1, 1],
        purchase_events=[1, 0, 0, 0],
    )
    readiness = _readiness(
        rows,
        minimum_purchase_events=1,
        minimum_user_effective_sample_size=1.5,
    )
    estimate = _estimate(
        readiness,
        rows,
        outcome_prediction_logged_action=[0.0] * 4,
        outcome_prediction_target_policy=[0.0] * 4,
        bootstrap_repetitions=1,
        bootstrap_seed=4,
    )

    # Seed 4 samples u0 and u1 once each. Restoring every request in the two
    # sampled clusters gives 1 purchase / 4 requests, not the 0.5 obtained by
    # incorrectly averaging the two user means with equal weight.
    assert estimate.doubly_robust_target_ci_low == pytest.approx(0.25)
    assert estimate.doubly_robust_target_ci_high == pytest.approx(0.25)


def test_clipping_is_reported_and_applied_to_all_three_estimators() -> None:
    rows = _row_data(
        target_probability_of_logged_action=[1.0, 0.5, 0.5, 0.5],
        logging_propensities=[0.1, 0.5, 0.5, 0.5],
    )
    readiness = _readiness(
        rows,
        weight_clip=2.0,
        maximum_allowed_p99_importance_weight=20.0,
        minimum_user_effective_sample_size=1.0,
    )
    estimate = _estimate(readiness, rows)

    assert estimate.maximum_raw_importance_weight == pytest.approx(10.0)
    assert estimate.weight_clip == pytest.approx(2.0)
    assert estimate.clipped_fraction == pytest.approx(0.25)
    assert estimate.ips_target_value == pytest.approx(0.75)


def test_row_digest_prevents_audit_estimation_data_mismatch() -> None:
    rows = _row_data()
    readiness = _readiness(rows)
    changed = _row_data(target_probability_of_logged_action=[0.4, 0.5, 0.5, 0.5])

    with pytest.raises(ValueError, match="digest"):
        _estimate(readiness, changed)


def test_partial_or_failed_readiness_cannot_run_point_ope() -> None:
    rows = _row_data()
    partial = _readiness(_row_data(guaranteed_target_support_mass=[0.9] * 4))
    failed = _readiness(rows, labels_settled=False)

    with pytest.raises(ValueError, match="does not authorize"):
        _estimate(partial, _row_data(guaranteed_target_support_mass=[0.9] * 4))
    with pytest.raises(ValueError, match="does not authorize"):
        _estimate(failed, rows)


def test_missing_outcomes_require_a_registered_adjusted_estimator() -> None:
    rows = _row_data(
        purchase_events=[1, None, 1, 0],
        outcome_observation_propensities=[1.0, 0.5, 1.0, 1.0],
    )
    readiness = _readiness(
        rows,
        maximum_missing_outcome_rate=0.25,
        minimum_user_effective_sample_size=1.5,
    )

    assert readiness.decision == "pass"
    with pytest.raises(ValueError, match="missingness-adjusted"):
        _estimate(readiness, rows)


@pytest.mark.parametrize(
    "attestation",
    [
        "nuisance_predictions_cross_fitted",
        "logging_propensities_are_request_time_probabilities",
        "outcomes_match_logged_actions",
    ],
)
def test_identification_attestations_are_required(attestation: str) -> None:
    rows = _row_data()
    readiness = _readiness(rows)

    with pytest.raises(ValueError, match="attestations failed"):
        _estimate(readiness, rows, **{attestation: False})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("outcome_prediction_logged_action", [0.4]),
        ("outcome_prediction_target_policy", [0.6, math.nan, 0.6, 0.6]),
        ("outcome_prediction_logged_action", [0.4, -0.1, 0.4, 0.4]),
    ],
)
def test_invalid_nuisance_predictions_are_rejected(field: str, value: object) -> None:
    rows = _row_data()
    readiness = _readiness(rows)

    with pytest.raises(ValueError):
        _estimate(readiness, rows, **{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("bootstrap_repetitions", 0),
        ("bootstrap_seed", -1),
        ("confidence_level", 0.0),
        ("confidence_level", 1.0),
        ("confidence_level", math.nan),
    ],
)
def test_invalid_inference_contract_is_rejected(field: str, value: object) -> None:
    rows = _row_data()
    readiness = _readiness(rows)

    with pytest.raises(ValueError):
        _estimate(readiness, rows, **{field: value})


def test_estimate_serializes_contract_and_claim_boundaries() -> None:
    rows = _row_data()
    readiness = _readiness(rows)
    serialized = _estimate(readiness, rows).to_dict()

    assert serialized["row_data_digest"] == readiness.row_data_digest
    assert serialized["bootstrap_seed"] == 16630
    assert serialized["conditional_causal_interpretation"] is True
    assert serialized["production_rollout_authorized"] is False


def test_partial_support_returns_full_population_value_and_lift_bounds() -> None:
    rows = _partial_rows()
    readiness = _readiness(rows)
    estimate = _partial_estimate(readiness, rows)

    assert readiness.authorized_estimand == "full_population_partial_support_bounds"
    assert estimate.supported_ips_contribution == pytest.approx(0.5)
    assert estimate.supported_doubly_robust_contribution == pytest.approx(0.64)
    assert estimate.mean_unsupported_target_mass == pytest.approx(0.1)
    assert estimate.target_value_lower == pytest.approx(0.64)
    assert estimate.target_value_upper == pytest.approx(0.74)
    assert estimate.lift_lower == pytest.approx(0.14)
    assert estimate.lift_upper == pytest.approx(0.24)
    assert estimate.identification_width == pytest.approx(0.1)
    assert estimate.target_value_confidence_lower == pytest.approx(0.64)
    assert estimate.target_value_confidence_upper == pytest.approx(0.74)
    assert estimate.conditional_partial_identification
    assert not estimate.production_rollout_authorized


def test_identification_width_tracks_unsupported_mass_and_outcome_range() -> None:
    rows = _partial_rows(guaranteed_target_support_mass=[0.8] * 4)
    readiness = _readiness(rows)
    estimate = _partial_estimate(
        readiness,
        rows,
        outcome_prediction_supported_target_contribution=[0.4] * 4,
        outcome_lower_bound=-1.0,
        outcome_upper_bound=1.0,
    )

    assert estimate.mean_unsupported_target_mass == pytest.approx(0.2)
    assert estimate.identification_width == pytest.approx(0.4)


def test_point_ope_and_partial_bounds_have_disjoint_authorization_gates() -> None:
    full_rows = _row_data()
    full_readiness = _readiness(full_rows)
    partial_rows = _partial_rows()
    partial_readiness = _readiness(partial_rows)

    with pytest.raises(ValueError, match="does not authorize"):
        _partial_estimate(full_readiness, full_rows)
    with pytest.raises(ValueError, match="does not authorize"):
        _estimate(partial_readiness, partial_rows)


def test_known_propensity_bound_estimator_rejects_propensity_intervals() -> None:
    rows = _partial_rows()
    values = dict(rows)
    values.pop("logging_propensities")
    readiness = _readiness(
        values,
        propensity_mode="interval",
        logging_propensity_lower=[0.4] * 4,
        logging_propensity_upper=[0.6] * 4,
        minimum_user_effective_sample_size=0.5,
    )

    assert readiness.partial_identification_authorized
    with pytest.raises(ValueError, match="does not authorize"):
        _partial_estimate(readiness, rows)


def test_partial_bounds_are_bound_to_the_audited_rows() -> None:
    rows = _partial_rows()
    readiness = _readiness(rows)
    changed = _partial_rows(target_probability_of_logged_action=[0.4, 0.5, 0.5, 0.5])

    with pytest.raises(ValueError, match="digest"):
        _partial_estimate(readiness, changed)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "outcome_prediction_supported_target_contribution",
            [1.0, 0.54, 0.54, 0.54],
            "support-scaled",
        ),
        (
            "outcome_prediction_logged_action",
            [1.1, 0.4, 0.4, 0.4],
            "logged-action predictions",
        ),
        ("outcome_lower_bound", 1.0, "strictly ordered"),
        ("outcome_upper_bound", math.inf, "strictly ordered"),
    ],
)
def test_partial_bound_contract_rejects_invalid_values(
    field: str, value: object, message: str
) -> None:
    rows = _partial_rows()
    readiness = _readiness(rows)

    with pytest.raises(ValueError, match=message):
        _partial_estimate(readiness, rows, **{field: value})


def test_observed_outcomes_must_lie_inside_registered_bounds() -> None:
    rows = _partial_rows(purchase_events=[2, 0, 1, 0])
    readiness = _readiness(rows)

    with pytest.raises(ValueError, match="purchase events"):
        _partial_estimate(readiness, rows, outcome_upper_bound=1.0)


def test_partial_bounds_reject_missing_outcomes() -> None:
    rows = _partial_rows(
        purchase_events=[1, None, 1, 0],
        outcome_observation_propensities=[1.0, 0.5, 1.0, 1.0],
    )
    readiness = _readiness(
        rows,
        maximum_missing_outcome_rate=0.25,
        minimum_user_effective_sample_size=1.5,
    )

    assert readiness.decision == "partial"
    with pytest.raises(ValueError, match="missingness-adjusted"):
        _partial_estimate(readiness, rows)


@pytest.mark.parametrize(
    "attestation",
    [
        "nuisance_predictions_cross_fitted",
        "logging_propensities_are_request_time_probabilities",
        "outcomes_match_logged_actions",
    ],
)
def test_partial_bounds_require_identification_attestations(attestation: str) -> None:
    rows = _partial_rows()
    readiness = _readiness(rows)

    with pytest.raises(ValueError, match="attestations failed"):
        _partial_estimate(readiness, rows, **{attestation: False})


def test_partial_estimate_serializes_claim_boundaries() -> None:
    rows = _partial_rows()
    readiness = _readiness(rows)
    serialized = _partial_estimate(readiness, rows).to_dict()

    assert serialized["row_data_digest"] == readiness.row_data_digest
    assert serialized["target_value_lower"] == pytest.approx(0.64)
    assert serialized["target_value_upper"] == pytest.approx(0.74)
    assert serialized["conditional_partial_identification"] is True
    assert serialized["production_rollout_authorized"] is False


def test_propensity_interval_bounds_are_sign_aware() -> None:
    rows = _interval_rows()
    readiness = _interval_readiness(rows)
    estimate = _interval_estimate(readiness, rows)

    assert readiness.authorized_estimand == (
        "full_population_propensity_interval_bounds"
    )
    assert estimate.supported_contribution_lower == pytest.approx(0.6)
    assert estimate.supported_contribution_upper == pytest.approx(97.0 / 120.0)
    assert estimate.target_value_lower == pytest.approx(0.6)
    assert estimate.target_value_upper == pytest.approx(97.0 / 120.0)
    assert estimate.lift_lower == pytest.approx(0.1)
    assert estimate.lift_upper == pytest.approx(37.0 / 120.0)
    assert estimate.propensity_uncertainty_width == pytest.approx(5.0 / 24.0)
    assert estimate.support_uncertainty_width == pytest.approx(0.0)
    assert estimate.identification_width == pytest.approx(5.0 / 24.0)
    assert estimate.target_value_confidence_lower == pytest.approx(0.6)
    assert estimate.target_value_confidence_upper == pytest.approx(97.0 / 120.0)


def test_propensity_and_support_uncertainty_widths_are_separated() -> None:
    rows = _interval_rows(guaranteed_target_support_mass=[0.9] * 4)
    readiness = _interval_readiness(rows)
    estimate = _interval_estimate(readiness, rows)

    assert readiness.authorized_estimand == (
        "full_population_joint_propensity_and_support_bounds"
    )
    assert estimate.mean_unsupported_target_mass == pytest.approx(0.1)
    assert estimate.propensity_uncertainty_width == pytest.approx(5.0 / 24.0)
    assert estimate.support_uncertainty_width == pytest.approx(0.1)
    assert estimate.identification_width == pytest.approx(37.0 / 120.0)


def test_propensity_interval_midpoint_is_never_substituted_for_bounds() -> None:
    rows = _interval_rows()
    readiness = _interval_readiness(rows)
    estimate = _interval_estimate(readiness, rows)

    midpoint_dr = 0.7
    assert estimate.target_value_lower < midpoint_dr
    assert estimate.target_value_upper > midpoint_dr
    assert estimate.target_value_lower != pytest.approx(midpoint_dr)
    assert estimate.target_value_upper != pytest.approx(midpoint_dr)


@pytest.mark.parametrize(
    "attestation",
    [
        "propensity_intervals_cover_true_probabilities",
        "nuisance_predictions_cross_fitted",
        "propensity_intervals_are_request_time",
        "outcomes_match_logged_actions",
    ],
)
def test_propensity_interval_bounds_require_attestations(attestation: str) -> None:
    rows = _interval_rows()
    readiness = _interval_readiness(rows)

    with pytest.raises(ValueError, match="attestations failed"):
        _interval_estimate(readiness, rows, **{attestation: False})


def test_propensity_interval_bounds_are_bound_to_audited_rows() -> None:
    rows = _interval_rows()
    readiness = _interval_readiness(rows)
    changed = _interval_rows(logging_propensity_lower=[0.3, 0.4, 0.4, 0.4])

    with pytest.raises(ValueError, match="digest"):
        _interval_estimate(readiness, changed)


def test_propensity_interval_estimator_rejects_known_propensity_readiness() -> None:
    rows = _partial_rows()
    readiness = _readiness(rows)
    interval_rows = _interval_rows(guaranteed_target_support_mass=[0.9] * 4)

    with pytest.raises(ValueError, match="does not authorize"):
        _interval_estimate(readiness, interval_rows)


def test_propensity_interval_estimate_serializes_claim_boundaries() -> None:
    rows = _interval_rows()
    readiness = _interval_readiness(rows)
    serialized = _interval_estimate(readiness, rows).to_dict()

    assert serialized["row_data_digest"] == readiness.row_data_digest
    assert serialized["propensity_interval_coverage_attested"] is True
    assert serialized["conditional_partial_identification"] is True
    assert serialized["production_rollout_authorized"] is False


def test_missingness_adjusted_dr_uses_observation_propensities() -> None:
    rows = _missing_rows()
    readiness = _missing_readiness(rows)
    estimate = _missing_estimate(readiness, rows)

    assert readiness.decision == "pass"
    assert estimate.observed_outcome_count == 3
    assert estimate.missing_outcome_rate == pytest.approx(0.25)
    assert estimate.minimum_observation_propensity == pytest.approx(0.5)
    assert estimate.ips_target_value == pytest.approx(0.5)
    assert estimate.snips_target_value == pytest.approx(2.0 / 3.0)
    assert estimate.doubly_robust_target_value == pytest.approx(0.8)
    assert estimate.logging_ipw_value == pytest.approx(0.5)
    assert estimate.logging_doubly_robust_value == pytest.approx(0.6)
    assert estimate.doubly_robust_lift == pytest.approx(0.2)
    assert estimate.doubly_robust_target_ci_low <= 0.8
    assert estimate.doubly_robust_target_ci_high >= 0.8
    assert estimate.user_effective_sample_size == pytest.approx(1.8)
    assert estimate.conditional_causal_interpretation
    assert not estimate.production_rollout_authorized


def test_observed_low_probability_label_receives_combined_weight() -> None:
    rows = _missing_rows(outcome_observation_propensities=[0.5, 0.5, 1.0, 1.0])
    readiness = _missing_readiness(rows)
    estimate = _missing_estimate(readiness, rows)

    assert estimate.maximum_raw_combined_weight == pytest.approx(2.0)
    assert estimate.doubly_robust_target_value == pytest.approx(0.95)
    assert estimate.logging_doubly_robust_value == pytest.approx(0.75)
    assert estimate.doubly_robust_lift == pytest.approx(0.2)


def test_missingness_adjusted_estimator_is_bound_to_observation_propensities() -> None:
    rows = _missing_rows()
    readiness = _missing_readiness(rows)
    changed = _missing_rows(outcome_observation_propensities=[0.9, 0.5, 1.0, 1.0])

    with pytest.raises(ValueError, match="digest"):
        _missing_estimate(readiness, changed)


def test_complete_case_readiness_does_not_authorize_missingness_estimator() -> None:
    rows = _row_data()
    readiness = _readiness(rows)
    missing_rows = _missing_rows()

    with pytest.raises(ValueError, match="does not authorize"):
        _missing_estimate(readiness, missing_rows)


@pytest.mark.parametrize(
    "attestation",
    [
        "nuisance_predictions_cross_fitted",
        "logging_propensities_are_request_time_probabilities",
        "observation_propensities_are_validated",
        "observation_propensities_are_request_time_probabilities",
        "outcomes_match_logged_actions",
        "conditional_missingness_independence",
    ],
)
def test_missingness_adjusted_estimator_requires_attestations(
    attestation: str,
) -> None:
    rows = _missing_rows()
    readiness = _missing_readiness(rows)

    with pytest.raises(ValueError, match="attestations failed"):
        _missing_estimate(readiness, rows, **{attestation: False})


def test_combined_weight_clipping_is_explicit() -> None:
    rows = _missing_rows(outcome_observation_propensities=[0.1, 0.5, 1.0, 1.0])
    readiness = _missing_readiness(
        rows,
        weight_clip=2.0,
        maximum_allowed_p99_importance_weight=20.0,
        minimum_user_effective_sample_size=1.0,
    )
    estimate = _missing_estimate(readiness, rows)

    assert estimate.maximum_raw_combined_weight == pytest.approx(10.0)
    assert estimate.combined_weight_clipped_fraction == pytest.approx(0.25)
    assert estimate.observation_weight_clipped_fraction == pytest.approx(0.25)


def test_missingness_adjusted_estimate_serializes_claim_boundaries() -> None:
    rows = _missing_rows()
    readiness = _missing_readiness(rows)
    serialized = _missing_estimate(readiness, rows).to_dict()

    assert serialized["row_data_digest"] == readiness.row_data_digest
    assert serialized["observation_propensities_validated_attested"] is True
    assert serialized["conditional_missingness_independence_attested"] is True
    assert serialized["production_rollout_authorized"] is False


def test_joint_known_propensity_bounds_combine_missingness_and_support() -> None:
    rows = _joint_known_rows()
    readiness = _joint_readiness(rows)
    estimate = _joint_estimate(readiness, rows)

    assert readiness.decision == "partial"
    assert estimate.propensity_mode == "known"
    assert estimate.observed_outcome_count == 3
    assert estimate.missing_outcome_rate == pytest.approx(0.25)
    assert estimate.supported_contribution_lower == pytest.approx(0.74)
    assert estimate.supported_contribution_upper == pytest.approx(0.74)
    assert estimate.logging_doubly_robust_value == pytest.approx(0.6)
    assert estimate.target_value_lower == pytest.approx(0.74)
    assert estimate.target_value_upper == pytest.approx(0.84)
    assert estimate.lift_lower == pytest.approx(0.14)
    assert estimate.lift_upper == pytest.approx(0.24)
    assert estimate.propensity_uncertainty_width == pytest.approx(0.0)
    assert estimate.support_uncertainty_width == pytest.approx(0.1)
    assert estimate.identification_width == pytest.approx(0.1)


def test_joint_interval_bounds_combine_all_three_uncertainty_sources() -> None:
    rows = _joint_interval_rows()
    readiness = _joint_readiness(rows)
    estimate = _joint_estimate(readiness, rows)

    assert readiness.authorized_estimand == (
        "full_population_joint_propensity_and_support_bounds"
    )
    assert estimate.propensity_mode == "interval"
    assert estimate.supported_contribution_lower == pytest.approx(0.665)
    assert estimate.supported_contribution_upper == pytest.approx(499.0 / 600.0)
    assert estimate.target_value_lower == pytest.approx(0.665)
    assert estimate.target_value_upper == pytest.approx(559.0 / 600.0)
    assert estimate.logging_doubly_robust_value == pytest.approx(0.6)
    assert estimate.propensity_uncertainty_width == pytest.approx(1.0 / 6.0)
    assert estimate.support_uncertainty_width == pytest.approx(0.1)
    assert estimate.identification_width == pytest.approx(4.0 / 15.0)
    assert estimate.propensity_interval_coverage_attested is True


def test_joint_interval_missingness_with_full_support_has_no_support_width() -> None:
    rows = _joint_interval_rows(guaranteed_target_support_mass=[1.0] * 4)
    readiness = _joint_readiness(rows)
    estimate = _joint_estimate(readiness, rows)

    assert readiness.authorized_estimand == (
        "full_population_propensity_interval_bounds"
    )
    assert estimate.mean_unsupported_target_mass == pytest.approx(0.0)
    assert estimate.support_uncertainty_width == pytest.approx(0.0)
    assert estimate.propensity_uncertainty_width == pytest.approx(1.0 / 6.0)
    assert estimate.identification_width == pytest.approx(1.0 / 6.0)


def test_joint_bounds_require_partial_readiness_and_missing_outcomes() -> None:
    full_support_rows = _missing_rows()
    full_support_readiness = _missing_readiness(full_support_rows)
    partial_complete_rows = _partial_rows()
    partial_complete_readiness = _readiness(partial_complete_rows)

    with pytest.raises(ValueError, match="does not authorize"):
        _joint_estimate(full_support_readiness, full_support_rows)
    with pytest.raises(ValueError, match="does not authorize"):
        _joint_estimate(partial_complete_readiness, partial_complete_rows)


@pytest.mark.parametrize(
    "attestation",
    [
        "nuisance_predictions_cross_fitted",
        "propensity_evidence_is_request_time",
        "observation_propensities_are_validated",
        "observation_propensities_are_request_time_probabilities",
        "outcomes_match_logged_actions",
        "conditional_missingness_independence",
    ],
)
def test_joint_bounds_require_common_attestations(attestation: str) -> None:
    rows = _joint_known_rows()
    readiness = _joint_readiness(rows)

    with pytest.raises(ValueError, match="attestations failed"):
        _joint_estimate(readiness, rows, **{attestation: False})


def test_joint_interval_bounds_require_interval_coverage_attestation() -> None:
    rows = _joint_interval_rows()
    readiness = _joint_readiness(rows)

    with pytest.raises(ValueError, match="attestations failed"):
        _joint_estimate(
            readiness,
            rows,
            propensity_intervals_cover_true_probabilities=False,
        )


def test_joint_bounds_are_bound_to_observation_propensities() -> None:
    rows = _joint_known_rows()
    readiness = _joint_readiness(rows)
    changed = _joint_known_rows(outcome_observation_propensities=[0.9, 0.5, 1.0, 1.0])

    with pytest.raises(ValueError, match="digest"):
        _joint_estimate(readiness, changed)


def test_joint_bounds_serialize_all_claim_boundaries() -> None:
    rows = _joint_interval_rows()
    readiness = _joint_readiness(rows)
    serialized = _joint_estimate(readiness, rows).to_dict()

    assert serialized["row_data_digest"] == readiness.row_data_digest
    assert serialized["conditional_partial_identification"] is True
    assert serialized["conditional_missingness_independence_attested"] is True
    assert serialized["production_rollout_authorized"] is False
