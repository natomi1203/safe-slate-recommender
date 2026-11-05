"""Cross-fitted off-policy estimators behind the aggregate readiness gate."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
from random import Random
from typing import Any, Sequence

from .contract import OpeEvaluationContract
from .ope import OpeReadinessAssessment, _row_data_digest


@dataclass(frozen=True, slots=True)
class CrossFittedOpeEstimate:
    """Point estimates and user-cluster bootstrap intervals."""

    row_data_digest: str
    evaluation_contract_digest: str | None
    row_count: int
    user_count: int
    fold_count: int
    logging_policy_value: float
    ips_target_value: float
    snips_target_value: float
    doubly_robust_target_value: float
    doubly_robust_target_ci_low: float
    doubly_robust_target_ci_high: float
    doubly_robust_lift: float
    doubly_robust_lift_ci_low: float
    doubly_robust_lift_ci_high: float
    confidence_level: float
    bootstrap_unit: str
    bootstrap_repetitions: int
    bootstrap_seed: int
    support_rate: float
    user_effective_sample_size: float
    weight_clip: float
    clipped_fraction: float
    maximum_raw_importance_weight: float
    nuisance_predictions_cross_fitted_attested: bool
    request_time_propensities_attested: bool
    logged_action_outcomes_attested: bool
    conditional_causal_interpretation: bool
    production_rollout_authorized: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_data_digest": self.row_data_digest,
            "evaluation_contract_digest": self.evaluation_contract_digest,
            "row_count": self.row_count,
            "user_count": self.user_count,
            "fold_count": self.fold_count,
            "logging_policy_value": self.logging_policy_value,
            "ips_target_value": self.ips_target_value,
            "snips_target_value": self.snips_target_value,
            "doubly_robust_target_value": self.doubly_robust_target_value,
            "doubly_robust_target_ci_low": self.doubly_robust_target_ci_low,
            "doubly_robust_target_ci_high": self.doubly_robust_target_ci_high,
            "doubly_robust_lift": self.doubly_robust_lift,
            "doubly_robust_lift_ci_low": self.doubly_robust_lift_ci_low,
            "doubly_robust_lift_ci_high": self.doubly_robust_lift_ci_high,
            "confidence_level": self.confidence_level,
            "bootstrap_unit": self.bootstrap_unit,
            "bootstrap_repetitions": self.bootstrap_repetitions,
            "bootstrap_seed": self.bootstrap_seed,
            "support_rate": self.support_rate,
            "user_effective_sample_size": self.user_effective_sample_size,
            "weight_clip": self.weight_clip,
            "clipped_fraction": self.clipped_fraction,
            "maximum_raw_importance_weight": (self.maximum_raw_importance_weight),
            "nuisance_predictions_cross_fitted_attested": (
                self.nuisance_predictions_cross_fitted_attested
            ),
            "request_time_propensities_attested": (
                self.request_time_propensities_attested
            ),
            "logged_action_outcomes_attested": (self.logged_action_outcomes_attested),
            "conditional_causal_interpretation": (
                self.conditional_causal_interpretation
            ),
            "production_rollout_authorized": self.production_rollout_authorized,
        }


@dataclass(frozen=True, slots=True)
class PartialIdentificationEstimate:
    """Estimated value and lift bounds under partial target-policy support."""

    row_data_digest: str
    evaluation_contract_digest: str | None
    row_count: int
    user_count: int
    fold_count: int
    support_rate: float
    mean_unsupported_target_mass: float
    outcome_lower_bound: float
    outcome_upper_bound: float
    supported_ips_contribution: float
    supported_doubly_robust_contribution: float
    logging_policy_value: float
    target_value_lower: float
    target_value_upper: float
    target_value_confidence_lower: float
    target_value_confidence_upper: float
    lift_lower: float
    lift_upper: float
    lift_confidence_lower: float
    lift_confidence_upper: float
    identification_width: float
    confidence_level: float
    bootstrap_unit: str
    bootstrap_repetitions: int
    bootstrap_seed: int
    weight_clip: float
    clipped_fraction: float
    nuisance_predictions_cross_fitted_attested: bool
    request_time_propensities_attested: bool
    logged_action_outcomes_attested: bool
    conditional_partial_identification: bool
    production_rollout_authorized: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_data_digest": self.row_data_digest,
            "evaluation_contract_digest": self.evaluation_contract_digest,
            "row_count": self.row_count,
            "user_count": self.user_count,
            "fold_count": self.fold_count,
            "support_rate": self.support_rate,
            "mean_unsupported_target_mass": self.mean_unsupported_target_mass,
            "outcome_lower_bound": self.outcome_lower_bound,
            "outcome_upper_bound": self.outcome_upper_bound,
            "supported_ips_contribution": self.supported_ips_contribution,
            "supported_doubly_robust_contribution": (
                self.supported_doubly_robust_contribution
            ),
            "logging_policy_value": self.logging_policy_value,
            "target_value_lower": self.target_value_lower,
            "target_value_upper": self.target_value_upper,
            "target_value_confidence_lower": (self.target_value_confidence_lower),
            "target_value_confidence_upper": (self.target_value_confidence_upper),
            "lift_lower": self.lift_lower,
            "lift_upper": self.lift_upper,
            "lift_confidence_lower": self.lift_confidence_lower,
            "lift_confidence_upper": self.lift_confidence_upper,
            "identification_width": self.identification_width,
            "confidence_level": self.confidence_level,
            "bootstrap_unit": self.bootstrap_unit,
            "bootstrap_repetitions": self.bootstrap_repetitions,
            "bootstrap_seed": self.bootstrap_seed,
            "weight_clip": self.weight_clip,
            "clipped_fraction": self.clipped_fraction,
            "nuisance_predictions_cross_fitted_attested": (
                self.nuisance_predictions_cross_fitted_attested
            ),
            "request_time_propensities_attested": (
                self.request_time_propensities_attested
            ),
            "logged_action_outcomes_attested": (self.logged_action_outcomes_attested),
            "conditional_partial_identification": (
                self.conditional_partial_identification
            ),
            "production_rollout_authorized": self.production_rollout_authorized,
        }


@dataclass(frozen=True, slots=True)
class PropensityIntervalEstimate:
    """Sign-aware value bounds under covering propensity intervals."""

    row_data_digest: str
    evaluation_contract_digest: str | None
    row_count: int
    user_count: int
    fold_count: int
    support_rate: float
    mean_unsupported_target_mass: float
    outcome_lower_bound: float
    outcome_upper_bound: float
    supported_contribution_lower: float
    supported_contribution_upper: float
    logging_policy_value: float
    target_value_lower: float
    target_value_upper: float
    target_value_confidence_lower: float
    target_value_confidence_upper: float
    lift_lower: float
    lift_upper: float
    lift_confidence_lower: float
    lift_confidence_upper: float
    propensity_uncertainty_width: float
    support_uncertainty_width: float
    identification_width: float
    confidence_level: float
    bootstrap_unit: str
    bootstrap_repetitions: int
    bootstrap_seed: int
    weight_clip: float
    clipped_fraction_upper_bound: float
    propensity_interval_coverage_attested: bool
    nuisance_predictions_cross_fitted_attested: bool
    request_time_propensity_intervals_attested: bool
    logged_action_outcomes_attested: bool
    conditional_partial_identification: bool
    production_rollout_authorized: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_data_digest": self.row_data_digest,
            "evaluation_contract_digest": self.evaluation_contract_digest,
            "row_count": self.row_count,
            "user_count": self.user_count,
            "fold_count": self.fold_count,
            "support_rate": self.support_rate,
            "mean_unsupported_target_mass": self.mean_unsupported_target_mass,
            "outcome_lower_bound": self.outcome_lower_bound,
            "outcome_upper_bound": self.outcome_upper_bound,
            "supported_contribution_lower": self.supported_contribution_lower,
            "supported_contribution_upper": self.supported_contribution_upper,
            "logging_policy_value": self.logging_policy_value,
            "target_value_lower": self.target_value_lower,
            "target_value_upper": self.target_value_upper,
            "target_value_confidence_lower": (self.target_value_confidence_lower),
            "target_value_confidence_upper": (self.target_value_confidence_upper),
            "lift_lower": self.lift_lower,
            "lift_upper": self.lift_upper,
            "lift_confidence_lower": self.lift_confidence_lower,
            "lift_confidence_upper": self.lift_confidence_upper,
            "propensity_uncertainty_width": self.propensity_uncertainty_width,
            "support_uncertainty_width": self.support_uncertainty_width,
            "identification_width": self.identification_width,
            "confidence_level": self.confidence_level,
            "bootstrap_unit": self.bootstrap_unit,
            "bootstrap_repetitions": self.bootstrap_repetitions,
            "bootstrap_seed": self.bootstrap_seed,
            "weight_clip": self.weight_clip,
            "clipped_fraction_upper_bound": self.clipped_fraction_upper_bound,
            "propensity_interval_coverage_attested": (
                self.propensity_interval_coverage_attested
            ),
            "nuisance_predictions_cross_fitted_attested": (
                self.nuisance_predictions_cross_fitted_attested
            ),
            "request_time_propensity_intervals_attested": (
                self.request_time_propensity_intervals_attested
            ),
            "logged_action_outcomes_attested": (self.logged_action_outcomes_attested),
            "conditional_partial_identification": (
                self.conditional_partial_identification
            ),
            "production_rollout_authorized": self.production_rollout_authorized,
        }


@dataclass(frozen=True, slots=True)
class MissingnessAdjustedOpeEstimate:
    """Observation-propensity-adjusted target value and lift estimates."""

    row_data_digest: str
    evaluation_contract_digest: str | None
    row_count: int
    observed_outcome_count: int
    user_count: int
    fold_count: int
    missing_outcome_rate: float
    minimum_observation_propensity: float
    logging_ipw_value: float
    logging_doubly_robust_value: float
    ips_target_value: float
    snips_target_value: float
    doubly_robust_target_value: float
    doubly_robust_target_ci_low: float
    doubly_robust_target_ci_high: float
    doubly_robust_lift: float
    doubly_robust_lift_ci_low: float
    doubly_robust_lift_ci_high: float
    confidence_level: float
    bootstrap_unit: str
    bootstrap_repetitions: int
    bootstrap_seed: int
    support_rate: float
    user_effective_sample_size: float
    weight_clip: float
    combined_weight_clipped_fraction: float
    observation_weight_clipped_fraction: float
    maximum_raw_combined_weight: float
    nuisance_predictions_cross_fitted_attested: bool
    request_time_propensities_attested: bool
    observation_propensities_validated_attested: bool
    observation_propensities_request_time_attested: bool
    logged_action_outcomes_attested: bool
    conditional_missingness_independence_attested: bool
    conditional_causal_interpretation: bool
    production_rollout_authorized: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_data_digest": self.row_data_digest,
            "evaluation_contract_digest": self.evaluation_contract_digest,
            "row_count": self.row_count,
            "observed_outcome_count": self.observed_outcome_count,
            "user_count": self.user_count,
            "fold_count": self.fold_count,
            "missing_outcome_rate": self.missing_outcome_rate,
            "minimum_observation_propensity": (self.minimum_observation_propensity),
            "logging_ipw_value": self.logging_ipw_value,
            "logging_doubly_robust_value": self.logging_doubly_robust_value,
            "ips_target_value": self.ips_target_value,
            "snips_target_value": self.snips_target_value,
            "doubly_robust_target_value": self.doubly_robust_target_value,
            "doubly_robust_target_ci_low": self.doubly_robust_target_ci_low,
            "doubly_robust_target_ci_high": self.doubly_robust_target_ci_high,
            "doubly_robust_lift": self.doubly_robust_lift,
            "doubly_robust_lift_ci_low": self.doubly_robust_lift_ci_low,
            "doubly_robust_lift_ci_high": self.doubly_robust_lift_ci_high,
            "confidence_level": self.confidence_level,
            "bootstrap_unit": self.bootstrap_unit,
            "bootstrap_repetitions": self.bootstrap_repetitions,
            "bootstrap_seed": self.bootstrap_seed,
            "support_rate": self.support_rate,
            "user_effective_sample_size": self.user_effective_sample_size,
            "weight_clip": self.weight_clip,
            "combined_weight_clipped_fraction": (self.combined_weight_clipped_fraction),
            "observation_weight_clipped_fraction": (
                self.observation_weight_clipped_fraction
            ),
            "maximum_raw_combined_weight": self.maximum_raw_combined_weight,
            "nuisance_predictions_cross_fitted_attested": (
                self.nuisance_predictions_cross_fitted_attested
            ),
            "request_time_propensities_attested": (
                self.request_time_propensities_attested
            ),
            "observation_propensities_validated_attested": (
                self.observation_propensities_validated_attested
            ),
            "observation_propensities_request_time_attested": (
                self.observation_propensities_request_time_attested
            ),
            "logged_action_outcomes_attested": (self.logged_action_outcomes_attested),
            "conditional_missingness_independence_attested": (
                self.conditional_missingness_independence_attested
            ),
            "conditional_causal_interpretation": (
                self.conditional_causal_interpretation
            ),
            "production_rollout_authorized": self.production_rollout_authorized,
        }


@dataclass(frozen=True, slots=True)
class JointUncertaintyBoundsEstimate:
    """Bounds combining partial support, propensity uncertainty, and missingness."""

    row_data_digest: str
    evaluation_contract_digest: str | None
    propensity_mode: str
    row_count: int
    observed_outcome_count: int
    user_count: int
    fold_count: int
    support_rate: float
    missing_outcome_rate: float
    minimum_observation_propensity: float
    mean_unsupported_target_mass: float
    outcome_lower_bound: float
    outcome_upper_bound: float
    logging_doubly_robust_value: float
    supported_contribution_lower: float
    supported_contribution_upper: float
    target_value_lower: float
    target_value_upper: float
    target_value_confidence_lower: float
    target_value_confidence_upper: float
    lift_lower: float
    lift_upper: float
    lift_confidence_lower: float
    lift_confidence_upper: float
    propensity_uncertainty_width: float
    support_uncertainty_width: float
    identification_width: float
    confidence_level: float
    bootstrap_unit: str
    bootstrap_repetitions: int
    bootstrap_seed: int
    weight_clip: float
    combined_weight_clipped_fraction_upper_bound: float
    observation_weight_clipped_fraction: float
    propensity_interval_coverage_attested: bool | None
    nuisance_predictions_cross_fitted_attested: bool
    request_time_propensity_evidence_attested: bool
    observation_propensities_validated_attested: bool
    observation_propensities_request_time_attested: bool
    logged_action_outcomes_attested: bool
    conditional_missingness_independence_attested: bool
    conditional_partial_identification: bool
    production_rollout_authorized: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_data_digest": self.row_data_digest,
            "evaluation_contract_digest": self.evaluation_contract_digest,
            "propensity_mode": self.propensity_mode,
            "row_count": self.row_count,
            "observed_outcome_count": self.observed_outcome_count,
            "user_count": self.user_count,
            "fold_count": self.fold_count,
            "support_rate": self.support_rate,
            "missing_outcome_rate": self.missing_outcome_rate,
            "minimum_observation_propensity": (self.minimum_observation_propensity),
            "mean_unsupported_target_mass": self.mean_unsupported_target_mass,
            "outcome_lower_bound": self.outcome_lower_bound,
            "outcome_upper_bound": self.outcome_upper_bound,
            "logging_doubly_robust_value": self.logging_doubly_robust_value,
            "supported_contribution_lower": self.supported_contribution_lower,
            "supported_contribution_upper": self.supported_contribution_upper,
            "target_value_lower": self.target_value_lower,
            "target_value_upper": self.target_value_upper,
            "target_value_confidence_lower": (self.target_value_confidence_lower),
            "target_value_confidence_upper": (self.target_value_confidence_upper),
            "lift_lower": self.lift_lower,
            "lift_upper": self.lift_upper,
            "lift_confidence_lower": self.lift_confidence_lower,
            "lift_confidence_upper": self.lift_confidence_upper,
            "propensity_uncertainty_width": self.propensity_uncertainty_width,
            "support_uncertainty_width": self.support_uncertainty_width,
            "identification_width": self.identification_width,
            "confidence_level": self.confidence_level,
            "bootstrap_unit": self.bootstrap_unit,
            "bootstrap_repetitions": self.bootstrap_repetitions,
            "bootstrap_seed": self.bootstrap_seed,
            "weight_clip": self.weight_clip,
            "combined_weight_clipped_fraction_upper_bound": (
                self.combined_weight_clipped_fraction_upper_bound
            ),
            "observation_weight_clipped_fraction": (
                self.observation_weight_clipped_fraction
            ),
            "propensity_interval_coverage_attested": (
                self.propensity_interval_coverage_attested
            ),
            "nuisance_predictions_cross_fitted_attested": (
                self.nuisance_predictions_cross_fitted_attested
            ),
            "request_time_propensity_evidence_attested": (
                self.request_time_propensity_evidence_attested
            ),
            "observation_propensities_validated_attested": (
                self.observation_propensities_validated_attested
            ),
            "observation_propensities_request_time_attested": (
                self.observation_propensities_request_time_attested
            ),
            "logged_action_outcomes_attested": (self.logged_action_outcomes_attested),
            "conditional_missingness_independence_attested": (
                self.conditional_missingness_independence_attested
            ),
            "conditional_partial_identification": (
                self.conditional_partial_identification
            ),
            "production_rollout_authorized": self.production_rollout_authorized,
        }


def _is_finite_real(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and isfinite(value)


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _validate_evaluation_contract_binding(
    contract: OpeEvaluationContract | None,
    readiness: OpeReadinessAssessment,
    *,
    bootstrap_repetitions: int,
    bootstrap_seed: int,
    confidence_level: float,
    outcome_lower_bound: float | None = None,
    outcome_upper_bound: float | None = None,
) -> None:
    if contract is None:
        if readiness.evaluation_contract_digest is not None:
            raise ValueError(
                "contract-bound readiness requires the matching evaluation contract"
            )
        return
    if not isinstance(contract, OpeEvaluationContract) or not contract.verify_digest():
        raise ValueError("evaluation contract is invalid or has been tampered with")
    if readiness.evaluation_contract_digest != contract.contract_digest:
        raise ValueError("readiness is not bound to this evaluation contract")
    frozen_pairs = {
        "propensity_mode": (readiness.propensity_mode, contract.propensity_mode),
        "minimum_raw_requests": (
            readiness.minimum_raw_requests,
            contract.minimum_raw_requests,
        ),
        "minimum_user_effective_sample_size": (
            readiness.minimum_user_effective_sample_size,
            contract.minimum_user_effective_sample_size,
        ),
        "minimum_purchase_events": (
            readiness.minimum_purchase_events,
            contract.minimum_purchase_events,
        ),
        "maximum_missing_outcome_rate": (
            readiness.maximum_missing_outcome_rate,
            contract.maximum_missing_outcome_rate,
        ),
        "maximum_p99_importance_weight": (
            readiness.maximum_allowed_p99_importance_weight,
            contract.maximum_p99_importance_weight,
        ),
        "weight_clip": (readiness.weight_clip, contract.weight_clip),
        "full_support_threshold": (
            readiness.full_estimand_support_threshold,
            contract.minimum_support_rate_for_full_estimand,
        ),
        "partial_support_threshold": (
            readiness.partial_estimand_support_threshold,
            contract.minimum_support_rate_for_partial_estimand,
        ),
        "bootstrap_repetitions": (
            int(bootstrap_repetitions),
            contract.bootstrap_repetitions,
        ),
        "bootstrap_seed": (int(bootstrap_seed), contract.bootstrap_seed),
        "confidence_level": (float(confidence_level), contract.confidence_level),
    }
    mismatched = [
        name for name, (actual, frozen) in frozen_pairs.items() if actual != frozen
    ]
    if outcome_lower_bound is not None and (
        float(outcome_lower_bound) != contract.outcome_lower_bound
        or float(outcome_upper_bound) != contract.outcome_upper_bound
    ):
        mismatched.append("outcome_bounds")
    if mismatched:
        raise ValueError(
            "estimation settings differ from the frozen contract: "
            + ", ".join(mismatched)
        )


def estimate_cross_fitted_policy_value(
    readiness: OpeReadinessAssessment,
    *,
    request_ids: Sequence[str],
    user_ids: Sequence[str],
    guaranteed_target_support_mass: Sequence[float],
    target_probability_of_logged_action: Sequence[float],
    logging_propensities: Sequence[float],
    purchase_events: Sequence[int | None],
    fold_ids: Sequence[int],
    outcome_prediction_logged_action: Sequence[float],
    outcome_prediction_target_policy: Sequence[float],
    nuisance_predictions_cross_fitted: bool,
    logging_propensities_are_request_time_probabilities: bool,
    outcomes_match_logged_actions: bool,
    bootstrap_repetitions: int,
    bootstrap_seed: int,
    confidence_level: float = 0.95,
    outcome_observation_propensities: Sequence[float] | None = None,
    evaluation_contract: OpeEvaluationContract | None = None,
) -> CrossFittedOpeEstimate:
    """Estimate a full-support target policy after an immutable readiness pass.

    Doubly robust estimation is the primary analysis. IPS and SNIPS are
    returned only as diagnostics. The percentile bootstrap resamples
    pseudonymous users as clusters and recomputes the request-weighted mean,
    preserving the settled-purchases-per-request estimand when users have
    different numbers of requests.

    This implementation deliberately rejects missing outcomes and propensity
    intervals. Those require a missingness-adjusted or partial-identification
    estimator rather than silently changing this estimand.
    """

    if not isinstance(readiness, OpeReadinessAssessment):
        raise ValueError("readiness must be an OpeReadinessAssessment")
    if (
        readiness.decision != "pass"
        or not readiness.standard_ope_authorized
        or readiness.authorized_estimand != "registered_full_target_population"
        or readiness.propensity_mode != "known"
    ):
        raise ValueError("readiness does not authorize full-population point OPE")
    if readiness.missing_outcome_rate != 0.0:
        raise ValueError(
            "this estimator requires complete outcomes; use a registered "
            "missingness-adjusted estimator instead"
        )
    _validate_evaluation_contract_binding(
        evaluation_contract,
        readiness,
        bootstrap_repetitions=bootstrap_repetitions,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
    )
    attestations = {
        "nuisance_predictions_cross_fitted": nuisance_predictions_cross_fitted,
        "logging_propensities_are_request_time_probabilities": (
            logging_propensities_are_request_time_probabilities
        ),
        "outcomes_match_logged_actions": outcomes_match_logged_actions,
    }
    failed_attestations = [
        name for name, value in attestations.items() if value is not True
    ]
    if failed_attestations:
        raise ValueError(
            "required identification attestations failed: "
            + ", ".join(failed_attestations)
        )
    if (
        not isinstance(bootstrap_repetitions, Integral)
        or isinstance(bootstrap_repetitions, bool)
        or bootstrap_repetitions <= 0
    ):
        raise ValueError("bootstrap_repetitions must be a positive integer")
    if (
        not isinstance(bootstrap_seed, Integral)
        or isinstance(bootstrap_seed, bool)
        or bootstrap_seed < 0
    ):
        raise ValueError("bootstrap_seed must be a nonnegative integer")
    if not _is_finite_real(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be finite and in (0, 1)")

    request_tuple = tuple(request_ids)
    user_tuple = tuple(user_ids)
    support_tuple = tuple(guaranteed_target_support_mass)
    target_tuple = tuple(target_probability_of_logged_action)
    propensity_tuple = tuple(logging_propensities)
    event_tuple = tuple(purchase_events)
    fold_tuple = tuple(fold_ids)
    q_logged_tuple = tuple(outcome_prediction_logged_action)
    q_target_tuple = tuple(outcome_prediction_target_policy)
    row_count = len(request_tuple)
    arrays = {
        "user_ids": user_tuple,
        "guaranteed_target_support_mass": support_tuple,
        "target_probability_of_logged_action": target_tuple,
        "logging_propensities": propensity_tuple,
        "purchase_events": event_tuple,
        "fold_ids": fold_tuple,
        "outcome_prediction_logged_action": q_logged_tuple,
        "outcome_prediction_target_policy": q_target_tuple,
    }
    mismatched = [name for name, values in arrays.items() if len(values) != row_count]
    if row_count == 0 or mismatched:
        detail = ", ".join(mismatched) if mismatched else "empty input"
        raise ValueError("OPE row arrays are inconsistent: " + detail)
    if row_count != readiness.raw_requests:
        raise ValueError("row count does not match readiness audit")

    digest = _row_data_digest(
        request_ids=request_tuple,
        user_ids=user_tuple,
        guaranteed_target_support_mass=support_tuple,
        target_probability_of_logged_action=target_tuple,
        purchase_events=event_tuple,
        fold_ids=fold_tuple,
        propensity_mode="known",
        logging_propensities=propensity_tuple,
        logging_propensity_lower=None,
        logging_propensity_upper=None,
        outcome_observation_propensities=outcome_observation_propensities,
    )
    if digest != readiness.row_data_digest:
        raise ValueError("OPE row data do not match the readiness audit digest")

    if any(value is None for value in event_tuple):
        raise ValueError("this estimator requires complete outcomes")
    if not all(
        isinstance(value, Integral) and not isinstance(value, bool) and value >= 0
        for value in event_tuple
    ):
        raise ValueError("purchase_events must be nonnegative integers")
    if not all(
        _is_finite_real(value) and value >= 0.0
        for value in (*q_logged_tuple, *q_target_tuple)
    ):
        raise ValueError("outcome predictions must be finite and nonnegative")

    folds_by_user: dict[str, set[int]] = defaultdict(set)
    for user_id, fold_id in zip(user_tuple, fold_tuple, strict=True):
        folds_by_user[user_id].add(int(fold_id))
    if any(len(folds) != 1 for folds in folds_by_user.values()):
        raise ValueError("a user block may not cross nuisance-model folds")
    fold_count = len(set(fold_tuple))
    if fold_count < 2:
        raise ValueError("cross-fitted OPE requires at least two folds")

    outcomes = [float(value) for value in event_tuple if value is not None]
    raw_weights = [
        float(target) / float(propensity)
        for target, propensity in zip(target_tuple, propensity_tuple, strict=True)
    ]
    weights = [min(weight, readiness.weight_clip) for weight in raw_weights]
    weighted_outcomes = [
        weight * outcome for weight, outcome in zip(weights, outcomes, strict=True)
    ]
    weight_sum = sum(weights)
    if weight_sum <= 0.0:
        raise ValueError("SNIPS denominator is zero")
    ips = sum(weighted_outcomes) / row_count
    snips = sum(weighted_outcomes) / weight_sum
    dr_rows = [
        float(q_target) + weight * (outcome - float(q_logged))
        for q_target, weight, outcome, q_logged in zip(
            q_target_tuple,
            weights,
            outcomes,
            q_logged_tuple,
            strict=True,
        )
    ]
    logging_rows = outcomes
    lift_rows = [
        target_value - logging_value
        for target_value, logging_value in zip(dr_rows, logging_rows, strict=True)
    ]
    logging_value = sum(logging_rows) / row_count
    dr_value = sum(dr_rows) / row_count
    dr_lift = sum(lift_rows) / row_count

    user_rows: dict[str, list[int]] = defaultdict(list)
    for index, user_id in enumerate(user_tuple):
        user_rows[user_id].append(index)
    users = tuple(sorted(user_rows))
    user_summaries = {
        user: (
            len(indices),
            sum(dr_rows[index] for index in indices),
            sum(lift_rows[index] for index in indices),
        )
        for user, indices in user_rows.items()
    }
    rng = Random(int(bootstrap_seed))
    bootstrap_target: list[float] = []
    bootstrap_lift: list[float] = []
    for _ in range(int(bootstrap_repetitions)):
        sampled = [users[rng.randrange(len(users))] for _ in users]
        sampled_rows = sum(user_summaries[user][0] for user in sampled)
        bootstrap_target.append(
            sum(user_summaries[user][1] for user in sampled) / sampled_rows
        )
        bootstrap_lift.append(
            sum(user_summaries[user][2] for user in sampled) / sampled_rows
        )
    alpha = (1.0 - float(confidence_level)) / 2.0

    support_rate = readiness.support_rate
    user_ess = readiness.user_effective_sample_size
    if support_rate is None or user_ess is None:
        raise ValueError("readiness audit lacks point-support diagnostics")
    return CrossFittedOpeEstimate(
        row_data_digest=digest,
        evaluation_contract_digest=readiness.evaluation_contract_digest,
        row_count=row_count,
        user_count=len(users),
        fold_count=fold_count,
        logging_policy_value=logging_value,
        ips_target_value=ips,
        snips_target_value=snips,
        doubly_robust_target_value=dr_value,
        doubly_robust_target_ci_low=_quantile(bootstrap_target, alpha),
        doubly_robust_target_ci_high=_quantile(bootstrap_target, 1.0 - alpha),
        doubly_robust_lift=dr_lift,
        doubly_robust_lift_ci_low=_quantile(bootstrap_lift, alpha),
        doubly_robust_lift_ci_high=_quantile(bootstrap_lift, 1.0 - alpha),
        confidence_level=float(confidence_level),
        bootstrap_unit="pseudonymous_user",
        bootstrap_repetitions=int(bootstrap_repetitions),
        bootstrap_seed=int(bootstrap_seed),
        support_rate=support_rate,
        user_effective_sample_size=user_ess,
        weight_clip=readiness.weight_clip,
        clipped_fraction=(
            sum(weight > readiness.weight_clip for weight in raw_weights) / row_count
        ),
        maximum_raw_importance_weight=max(raw_weights),
        nuisance_predictions_cross_fitted_attested=True,
        request_time_propensities_attested=True,
        logged_action_outcomes_attested=True,
        conditional_causal_interpretation=True,
        production_rollout_authorized=False,
    )


def estimate_partial_identification_bounds(
    readiness: OpeReadinessAssessment,
    *,
    request_ids: Sequence[str],
    user_ids: Sequence[str],
    guaranteed_target_support_mass: Sequence[float],
    target_probability_of_logged_action: Sequence[float],
    logging_propensities: Sequence[float],
    purchase_events: Sequence[int | None],
    fold_ids: Sequence[int],
    outcome_prediction_logged_action: Sequence[float],
    outcome_prediction_supported_target_contribution: Sequence[float],
    outcome_lower_bound: float,
    outcome_upper_bound: float,
    nuisance_predictions_cross_fitted: bool,
    logging_propensities_are_request_time_probabilities: bool,
    outcomes_match_logged_actions: bool,
    bootstrap_repetitions: int,
    bootstrap_seed: int,
    confidence_level: float = 0.95,
    outcome_observation_propensities: Sequence[float] | None = None,
    evaluation_contract: OpeEvaluationContract | None = None,
) -> PartialIdentificationEstimate:
    """Estimate full-population value bounds when target support is partial.

    The cross-fitted nuisance input for the target policy is the reward
    contribution integrated only over actions with guaranteed positive logging
    support. The remaining target-policy mass is bounded by the registered
    outcome range. No unsupported action is assigned a model-predicted outcome.
    """

    if not isinstance(readiness, OpeReadinessAssessment):
        raise ValueError("readiness must be an OpeReadinessAssessment")
    if (
        readiness.decision != "partial"
        or not readiness.partial_identification_authorized
        or readiness.authorized_estimand != "full_population_partial_support_bounds"
        or readiness.propensity_mode != "known"
    ):
        raise ValueError(
            "readiness does not authorize known-propensity partial-support bounds"
        )
    if readiness.missing_outcome_rate != 0.0:
        raise ValueError(
            "this estimator requires complete outcomes; use a registered "
            "missingness-adjusted bound instead"
        )
    if (
        not _is_finite_real(outcome_lower_bound)
        or not _is_finite_real(outcome_upper_bound)
        or outcome_lower_bound >= outcome_upper_bound
    ):
        raise ValueError("outcome bounds must be finite and strictly ordered")
    _validate_evaluation_contract_binding(
        evaluation_contract,
        readiness,
        bootstrap_repetitions=bootstrap_repetitions,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
        outcome_lower_bound=outcome_lower_bound,
        outcome_upper_bound=outcome_upper_bound,
    )
    attestations = {
        "nuisance_predictions_cross_fitted": nuisance_predictions_cross_fitted,
        "logging_propensities_are_request_time_probabilities": (
            logging_propensities_are_request_time_probabilities
        ),
        "outcomes_match_logged_actions": outcomes_match_logged_actions,
    }
    failed_attestations = [
        name for name, value in attestations.items() if value is not True
    ]
    if failed_attestations:
        raise ValueError(
            "required identification attestations failed: "
            + ", ".join(failed_attestations)
        )
    if (
        not isinstance(bootstrap_repetitions, Integral)
        or isinstance(bootstrap_repetitions, bool)
        or bootstrap_repetitions <= 0
    ):
        raise ValueError("bootstrap_repetitions must be a positive integer")
    if (
        not isinstance(bootstrap_seed, Integral)
        or isinstance(bootstrap_seed, bool)
        or bootstrap_seed < 0
    ):
        raise ValueError("bootstrap_seed must be a nonnegative integer")
    if not _is_finite_real(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be finite and in (0, 1)")

    request_tuple = tuple(request_ids)
    user_tuple = tuple(user_ids)
    support_tuple = tuple(guaranteed_target_support_mass)
    target_tuple = tuple(target_probability_of_logged_action)
    propensity_tuple = tuple(logging_propensities)
    event_tuple = tuple(purchase_events)
    fold_tuple = tuple(fold_ids)
    q_logged_tuple = tuple(outcome_prediction_logged_action)
    q_supported_tuple = tuple(outcome_prediction_supported_target_contribution)
    row_count = len(request_tuple)
    arrays = {
        "user_ids": user_tuple,
        "guaranteed_target_support_mass": support_tuple,
        "target_probability_of_logged_action": target_tuple,
        "logging_propensities": propensity_tuple,
        "purchase_events": event_tuple,
        "fold_ids": fold_tuple,
        "outcome_prediction_logged_action": q_logged_tuple,
        "outcome_prediction_supported_target_contribution": q_supported_tuple,
    }
    mismatched = [name for name, values in arrays.items() if len(values) != row_count]
    if row_count == 0 or mismatched:
        detail = ", ".join(mismatched) if mismatched else "empty input"
        raise ValueError(
            "partial-identification row arrays are inconsistent: " + detail
        )
    if row_count != readiness.raw_requests:
        raise ValueError("row count does not match readiness audit")

    digest = _row_data_digest(
        request_ids=request_tuple,
        user_ids=user_tuple,
        guaranteed_target_support_mass=support_tuple,
        target_probability_of_logged_action=target_tuple,
        purchase_events=event_tuple,
        fold_ids=fold_tuple,
        propensity_mode="known",
        logging_propensities=propensity_tuple,
        logging_propensity_lower=None,
        logging_propensity_upper=None,
        outcome_observation_propensities=outcome_observation_propensities,
    )
    if digest != readiness.row_data_digest:
        raise ValueError(
            "partial-identification row data do not match the readiness audit digest"
        )
    if any(value is None for value in event_tuple):
        raise ValueError("this estimator requires complete outcomes")
    if not all(
        isinstance(value, Integral)
        and not isinstance(value, bool)
        and value >= 0
        and outcome_lower_bound <= float(value) <= outcome_upper_bound
        for value in event_tuple
    ):
        raise ValueError(
            "purchase events must lie inside the registered outcome bounds"
        )
    if not all(
        _is_finite_real(value) and outcome_lower_bound <= value <= outcome_upper_bound
        for value in q_logged_tuple
    ):
        raise ValueError("logged-action predictions must lie inside outcome bounds")
    for support, contribution in zip(support_tuple, q_supported_tuple, strict=True):
        if not _is_finite_real(contribution):
            raise ValueError("supported-target contributions must be finite")
        contribution_lower = float(support) * float(outcome_lower_bound)
        contribution_upper = float(support) * float(outcome_upper_bound)
        if not contribution_lower - 1e-12 <= contribution <= contribution_upper + 1e-12:
            raise ValueError(
                "supported-target contribution lies outside support-scaled bounds"
            )

    folds_by_user: dict[str, set[int]] = defaultdict(set)
    for user_id, fold_id in zip(user_tuple, fold_tuple, strict=True):
        folds_by_user[user_id].add(int(fold_id))
    if any(len(folds) != 1 for folds in folds_by_user.values()):
        raise ValueError("a user block may not cross nuisance-model folds")
    fold_count = len(set(fold_tuple))
    if fold_count < 2:
        raise ValueError("cross-fitted bounds require at least two folds")

    outcomes = [float(value) for value in event_tuple if value is not None]
    raw_weights = [
        float(target) / float(propensity)
        for target, propensity in zip(target_tuple, propensity_tuple, strict=True)
    ]
    weights = [min(weight, readiness.weight_clip) for weight in raw_weights]
    supported_ips_rows = [
        weight * outcome for weight, outcome in zip(weights, outcomes, strict=True)
    ]
    supported_dr_rows = [
        float(q_supported) + weight * (outcome - float(q_logged))
        for q_supported, weight, outcome, q_logged in zip(
            q_supported_tuple,
            weights,
            outcomes,
            q_logged_tuple,
            strict=True,
        )
    ]
    unsupported_mass = [1.0 - float(support) for support in support_tuple]
    supported_dr = sum(supported_dr_rows) / row_count
    mean_unsupported = sum(unsupported_mass) / row_count
    raw_target_lower = supported_dr + mean_unsupported * float(outcome_lower_bound)
    raw_target_upper = supported_dr + mean_unsupported * float(outcome_upper_bound)
    target_lower = min(
        max(raw_target_lower, float(outcome_lower_bound)),
        float(outcome_upper_bound),
    )
    target_upper = min(
        max(raw_target_upper, float(outcome_lower_bound)),
        float(outcome_upper_bound),
    )
    logging_value = sum(outcomes) / row_count
    lift_lower = target_lower - logging_value
    lift_upper = target_upper - logging_value

    user_rows: dict[str, list[int]] = defaultdict(list)
    for index, user_id in enumerate(user_tuple):
        user_rows[user_id].append(index)
    users = tuple(sorted(user_rows))
    user_summaries = {
        user: (
            len(indices),
            sum(supported_dr_rows[index] for index in indices),
            sum(unsupported_mass[index] for index in indices),
            sum(outcomes[index] for index in indices),
        )
        for user, indices in user_rows.items()
    }
    rng = Random(int(bootstrap_seed))
    bootstrap_target_lower: list[float] = []
    bootstrap_target_upper: list[float] = []
    bootstrap_lift_lower: list[float] = []
    bootstrap_lift_upper: list[float] = []
    for _ in range(int(bootstrap_repetitions)):
        sampled = [users[rng.randrange(len(users))] for _ in users]
        sampled_rows = sum(user_summaries[user][0] for user in sampled)
        sampled_supported = sum(user_summaries[user][1] for user in sampled)
        sampled_unsupported = sum(user_summaries[user][2] for user in sampled)
        sampled_logging = (
            sum(user_summaries[user][3] for user in sampled) / sampled_rows
        )
        lower = (
            sampled_supported + sampled_unsupported * float(outcome_lower_bound)
        ) / sampled_rows
        upper = (
            sampled_supported + sampled_unsupported * float(outcome_upper_bound)
        ) / sampled_rows
        lower = min(max(lower, float(outcome_lower_bound)), float(outcome_upper_bound))
        upper = min(max(upper, float(outcome_lower_bound)), float(outcome_upper_bound))
        bootstrap_target_lower.append(lower)
        bootstrap_target_upper.append(upper)
        bootstrap_lift_lower.append(lower - sampled_logging)
        bootstrap_lift_upper.append(upper - sampled_logging)
    alpha = (1.0 - float(confidence_level)) / 2.0

    support_rate = readiness.support_rate
    if support_rate is None:
        raise ValueError("readiness audit lacks support diagnostics")
    return PartialIdentificationEstimate(
        row_data_digest=digest,
        evaluation_contract_digest=readiness.evaluation_contract_digest,
        row_count=row_count,
        user_count=len(users),
        fold_count=fold_count,
        support_rate=support_rate,
        mean_unsupported_target_mass=mean_unsupported,
        outcome_lower_bound=float(outcome_lower_bound),
        outcome_upper_bound=float(outcome_upper_bound),
        supported_ips_contribution=sum(supported_ips_rows) / row_count,
        supported_doubly_robust_contribution=supported_dr,
        logging_policy_value=logging_value,
        target_value_lower=target_lower,
        target_value_upper=target_upper,
        target_value_confidence_lower=_quantile(bootstrap_target_lower, alpha),
        target_value_confidence_upper=_quantile(bootstrap_target_upper, 1.0 - alpha),
        lift_lower=lift_lower,
        lift_upper=lift_upper,
        lift_confidence_lower=_quantile(bootstrap_lift_lower, alpha),
        lift_confidence_upper=_quantile(bootstrap_lift_upper, 1.0 - alpha),
        identification_width=target_upper - target_lower,
        confidence_level=float(confidence_level),
        bootstrap_unit="pseudonymous_user",
        bootstrap_repetitions=int(bootstrap_repetitions),
        bootstrap_seed=int(bootstrap_seed),
        weight_clip=readiness.weight_clip,
        clipped_fraction=(
            sum(weight > readiness.weight_clip for weight in raw_weights) / row_count
        ),
        nuisance_predictions_cross_fitted_attested=True,
        request_time_propensities_attested=True,
        logged_action_outcomes_attested=True,
        conditional_partial_identification=True,
        production_rollout_authorized=False,
    )


def estimate_propensity_interval_bounds(
    readiness: OpeReadinessAssessment,
    *,
    request_ids: Sequence[str],
    user_ids: Sequence[str],
    guaranteed_target_support_mass: Sequence[float],
    target_probability_of_logged_action: Sequence[float],
    logging_propensity_lower: Sequence[float],
    logging_propensity_upper: Sequence[float],
    purchase_events: Sequence[int | None],
    fold_ids: Sequence[int],
    outcome_prediction_logged_action: Sequence[float],
    outcome_prediction_supported_target_contribution: Sequence[float],
    outcome_lower_bound: float,
    outcome_upper_bound: float,
    propensity_intervals_cover_true_probabilities: bool,
    nuisance_predictions_cross_fitted: bool,
    propensity_intervals_are_request_time: bool,
    outcomes_match_logged_actions: bool,
    bootstrap_repetitions: int,
    bootstrap_seed: int,
    confidence_level: float = 0.95,
    outcome_observation_propensities: Sequence[float] | None = None,
    evaluation_contract: OpeEvaluationContract | None = None,
) -> PropensityIntervalEstimate:
    """Estimate sign-aware bounds under covering propensity intervals.

    For a positive residual, the lower bound uses the smallest importance
    weight and the upper bound the largest. For a negative residual, the order
    reverses. This preserves the worst-case direction instead of substituting
    an interval midpoint. Partial target support is handled simultaneously by
    assigning unsupported mass the registered outcome range.
    """

    if not isinstance(readiness, OpeReadinessAssessment):
        raise ValueError("readiness must be an OpeReadinessAssessment")
    if (
        readiness.decision != "partial"
        or not readiness.partial_identification_authorized
        or readiness.propensity_mode != "interval"
        or readiness.authorized_estimand
        not in {
            "full_population_propensity_interval_bounds",
            "full_population_joint_propensity_and_support_bounds",
        }
    ):
        raise ValueError("readiness does not authorize propensity-interval bounds")
    if readiness.missing_outcome_rate != 0.0:
        raise ValueError(
            "this estimator requires complete outcomes; use a registered "
            "missingness-adjusted bound instead"
        )
    if (
        not _is_finite_real(outcome_lower_bound)
        or not _is_finite_real(outcome_upper_bound)
        or outcome_lower_bound >= outcome_upper_bound
    ):
        raise ValueError("outcome bounds must be finite and strictly ordered")
    _validate_evaluation_contract_binding(
        evaluation_contract,
        readiness,
        bootstrap_repetitions=bootstrap_repetitions,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
        outcome_lower_bound=outcome_lower_bound,
        outcome_upper_bound=outcome_upper_bound,
    )
    attestations = {
        "propensity_intervals_cover_true_probabilities": (
            propensity_intervals_cover_true_probabilities
        ),
        "nuisance_predictions_cross_fitted": nuisance_predictions_cross_fitted,
        "propensity_intervals_are_request_time": (
            propensity_intervals_are_request_time
        ),
        "outcomes_match_logged_actions": outcomes_match_logged_actions,
    }
    failed_attestations = [
        name for name, value in attestations.items() if value is not True
    ]
    if failed_attestations:
        raise ValueError(
            "required identification attestations failed: "
            + ", ".join(failed_attestations)
        )
    if (
        not isinstance(bootstrap_repetitions, Integral)
        or isinstance(bootstrap_repetitions, bool)
        or bootstrap_repetitions <= 0
    ):
        raise ValueError("bootstrap_repetitions must be a positive integer")
    if (
        not isinstance(bootstrap_seed, Integral)
        or isinstance(bootstrap_seed, bool)
        or bootstrap_seed < 0
    ):
        raise ValueError("bootstrap_seed must be a nonnegative integer")
    if not _is_finite_real(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be finite and in (0, 1)")

    request_tuple = tuple(request_ids)
    user_tuple = tuple(user_ids)
    support_tuple = tuple(guaranteed_target_support_mass)
    target_tuple = tuple(target_probability_of_logged_action)
    propensity_lower_tuple = tuple(logging_propensity_lower)
    propensity_upper_tuple = tuple(logging_propensity_upper)
    event_tuple = tuple(purchase_events)
    fold_tuple = tuple(fold_ids)
    q_logged_tuple = tuple(outcome_prediction_logged_action)
    q_supported_tuple = tuple(outcome_prediction_supported_target_contribution)
    row_count = len(request_tuple)
    arrays = {
        "user_ids": user_tuple,
        "guaranteed_target_support_mass": support_tuple,
        "target_probability_of_logged_action": target_tuple,
        "logging_propensity_lower": propensity_lower_tuple,
        "logging_propensity_upper": propensity_upper_tuple,
        "purchase_events": event_tuple,
        "fold_ids": fold_tuple,
        "outcome_prediction_logged_action": q_logged_tuple,
        "outcome_prediction_supported_target_contribution": q_supported_tuple,
    }
    mismatched = [name for name, values in arrays.items() if len(values) != row_count]
    if row_count == 0 or mismatched:
        detail = ", ".join(mismatched) if mismatched else "empty input"
        raise ValueError("propensity-interval row arrays are inconsistent: " + detail)
    if row_count != readiness.raw_requests:
        raise ValueError("row count does not match readiness audit")

    digest = _row_data_digest(
        request_ids=request_tuple,
        user_ids=user_tuple,
        guaranteed_target_support_mass=support_tuple,
        target_probability_of_logged_action=target_tuple,
        purchase_events=event_tuple,
        fold_ids=fold_tuple,
        propensity_mode="interval",
        logging_propensities=None,
        logging_propensity_lower=propensity_lower_tuple,
        logging_propensity_upper=propensity_upper_tuple,
        outcome_observation_propensities=outcome_observation_propensities,
    )
    if digest != readiness.row_data_digest:
        raise ValueError(
            "propensity-interval row data do not match the readiness audit digest"
        )
    if any(value is None for value in event_tuple):
        raise ValueError("this estimator requires complete outcomes")
    if not all(
        isinstance(value, Integral)
        and not isinstance(value, bool)
        and value >= 0
        and outcome_lower_bound <= float(value) <= outcome_upper_bound
        for value in event_tuple
    ):
        raise ValueError(
            "purchase events must lie inside the registered outcome bounds"
        )
    if not all(
        _is_finite_real(value) and outcome_lower_bound <= value <= outcome_upper_bound
        for value in q_logged_tuple
    ):
        raise ValueError("logged-action predictions must lie inside outcome bounds")
    for support, contribution in zip(support_tuple, q_supported_tuple, strict=True):
        if not _is_finite_real(contribution):
            raise ValueError("supported-target contributions must be finite")
        contribution_lower = float(support) * float(outcome_lower_bound)
        contribution_upper = float(support) * float(outcome_upper_bound)
        if not contribution_lower - 1e-12 <= contribution <= contribution_upper + 1e-12:
            raise ValueError(
                "supported-target contribution lies outside support-scaled bounds"
            )

    folds_by_user: dict[str, set[int]] = defaultdict(set)
    for user_id, fold_id in zip(user_tuple, fold_tuple, strict=True):
        folds_by_user[user_id].add(int(fold_id))
    if any(len(folds) != 1 for folds in folds_by_user.values()):
        raise ValueError("a user block may not cross nuisance-model folds")
    fold_count = len(set(fold_tuple))
    if fold_count < 2:
        raise ValueError("cross-fitted bounds require at least two folds")

    outcomes = [float(value) for value in event_tuple if value is not None]
    weight_lower: list[float] = []
    weight_upper: list[float] = []
    for target, propensity_lower, propensity_upper in zip(
        target_tuple,
        propensity_lower_tuple,
        propensity_upper_tuple,
        strict=True,
    ):
        if (
            not _is_finite_real(propensity_lower)
            or not _is_finite_real(propensity_upper)
            or not 0.0 < propensity_lower <= propensity_upper <= 1.0
        ):
            raise ValueError("propensity intervals must be positive and ordered")
        lower = float(target) / float(propensity_upper)
        upper = float(target) / float(propensity_lower)
        weight_lower.append(min(lower, readiness.weight_clip))
        weight_upper.append(min(upper, readiness.weight_clip))

    supported_lower_rows: list[float] = []
    supported_upper_rows: list[float] = []
    for q_supported, q_logged, outcome, lower, upper in zip(
        q_supported_tuple,
        q_logged_tuple,
        outcomes,
        weight_lower,
        weight_upper,
        strict=True,
    ):
        residual = outcome - float(q_logged)
        if residual >= 0.0:
            correction_lower = lower * residual
            correction_upper = upper * residual
        else:
            correction_lower = upper * residual
            correction_upper = lower * residual
        supported_lower_rows.append(float(q_supported) + correction_lower)
        supported_upper_rows.append(float(q_supported) + correction_upper)

    unsupported_mass = [1.0 - float(support) for support in support_tuple]
    supported_lower = sum(supported_lower_rows) / row_count
    supported_upper = sum(supported_upper_rows) / row_count
    mean_unsupported = sum(unsupported_mass) / row_count
    raw_target_lower = supported_lower + mean_unsupported * float(outcome_lower_bound)
    raw_target_upper = supported_upper + mean_unsupported * float(outcome_upper_bound)
    target_lower = min(
        max(raw_target_lower, float(outcome_lower_bound)),
        float(outcome_upper_bound),
    )
    target_upper = min(
        max(raw_target_upper, float(outcome_lower_bound)),
        float(outcome_upper_bound),
    )
    logging_value = sum(outcomes) / row_count
    lift_lower = target_lower - logging_value
    lift_upper = target_upper - logging_value

    user_rows: dict[str, list[int]] = defaultdict(list)
    for index, user_id in enumerate(user_tuple):
        user_rows[user_id].append(index)
    users = tuple(sorted(user_rows))
    user_summaries = {
        user: (
            len(indices),
            sum(supported_lower_rows[index] for index in indices),
            sum(supported_upper_rows[index] for index in indices),
            sum(unsupported_mass[index] for index in indices),
            sum(outcomes[index] for index in indices),
        )
        for user, indices in user_rows.items()
    }
    rng = Random(int(bootstrap_seed))
    bootstrap_target_lower: list[float] = []
    bootstrap_target_upper: list[float] = []
    bootstrap_lift_lower: list[float] = []
    bootstrap_lift_upper: list[float] = []
    for _ in range(int(bootstrap_repetitions)):
        sampled = [users[rng.randrange(len(users))] for _ in users]
        sampled_rows = sum(user_summaries[user][0] for user in sampled)
        sampled_lower = sum(user_summaries[user][1] for user in sampled)
        sampled_upper = sum(user_summaries[user][2] for user in sampled)
        sampled_unsupported = sum(user_summaries[user][3] for user in sampled)
        sampled_logging = (
            sum(user_summaries[user][4] for user in sampled) / sampled_rows
        )
        lower = (
            sampled_lower + sampled_unsupported * float(outcome_lower_bound)
        ) / sampled_rows
        upper = (
            sampled_upper + sampled_unsupported * float(outcome_upper_bound)
        ) / sampled_rows
        lower = min(max(lower, float(outcome_lower_bound)), float(outcome_upper_bound))
        upper = min(max(upper, float(outcome_lower_bound)), float(outcome_upper_bound))
        bootstrap_target_lower.append(lower)
        bootstrap_target_upper.append(upper)
        bootstrap_lift_lower.append(lower - sampled_logging)
        bootstrap_lift_upper.append(upper - sampled_logging)
    alpha = (1.0 - float(confidence_level)) / 2.0

    support_rate = readiness.support_rate
    if support_rate is None:
        raise ValueError("readiness audit lacks support diagnostics")
    return PropensityIntervalEstimate(
        row_data_digest=digest,
        evaluation_contract_digest=readiness.evaluation_contract_digest,
        row_count=row_count,
        user_count=len(users),
        fold_count=fold_count,
        support_rate=support_rate,
        mean_unsupported_target_mass=mean_unsupported,
        outcome_lower_bound=float(outcome_lower_bound),
        outcome_upper_bound=float(outcome_upper_bound),
        supported_contribution_lower=supported_lower,
        supported_contribution_upper=supported_upper,
        logging_policy_value=logging_value,
        target_value_lower=target_lower,
        target_value_upper=target_upper,
        target_value_confidence_lower=_quantile(bootstrap_target_lower, alpha),
        target_value_confidence_upper=_quantile(bootstrap_target_upper, 1.0 - alpha),
        lift_lower=lift_lower,
        lift_upper=lift_upper,
        lift_confidence_lower=_quantile(bootstrap_lift_lower, alpha),
        lift_confidence_upper=_quantile(bootstrap_lift_upper, 1.0 - alpha),
        propensity_uncertainty_width=supported_upper - supported_lower,
        support_uncertainty_width=(
            mean_unsupported * (float(outcome_upper_bound) - float(outcome_lower_bound))
        ),
        identification_width=target_upper - target_lower,
        confidence_level=float(confidence_level),
        bootstrap_unit="pseudonymous_user",
        bootstrap_repetitions=int(bootstrap_repetitions),
        bootstrap_seed=int(bootstrap_seed),
        weight_clip=readiness.weight_clip,
        clipped_fraction_upper_bound=(
            sum(
                float(target) / float(lower) > readiness.weight_clip
                for target, lower in zip(
                    target_tuple, propensity_lower_tuple, strict=True
                )
            )
            / row_count
        ),
        propensity_interval_coverage_attested=True,
        nuisance_predictions_cross_fitted_attested=True,
        request_time_propensity_intervals_attested=True,
        logged_action_outcomes_attested=True,
        conditional_partial_identification=True,
        production_rollout_authorized=False,
    )


def estimate_missingness_adjusted_policy_value(
    readiness: OpeReadinessAssessment,
    *,
    request_ids: Sequence[str],
    user_ids: Sequence[str],
    guaranteed_target_support_mass: Sequence[float],
    target_probability_of_logged_action: Sequence[float],
    logging_propensities: Sequence[float],
    outcome_observation_propensities: Sequence[float],
    purchase_events: Sequence[int | None],
    fold_ids: Sequence[int],
    outcome_prediction_logged_action: Sequence[float],
    outcome_prediction_target_policy: Sequence[float],
    nuisance_predictions_cross_fitted: bool,
    logging_propensities_are_request_time_probabilities: bool,
    observation_propensities_are_validated: bool,
    observation_propensities_are_request_time_probabilities: bool,
    outcomes_match_logged_actions: bool,
    conditional_missingness_independence: bool,
    bootstrap_repetitions: int,
    bootstrap_seed: int,
    confidence_level: float = 0.95,
    evaluation_contract: OpeEvaluationContract | None = None,
) -> MissingnessAdjustedOpeEstimate:
    """Estimate target value with action and outcome-observation propensities.

    The target-policy residual correction uses the combined weight
    ``pi_target / (pi_log * rho_observe)``. The logging-policy comparator uses
    an observation-propensity doubly robust correction, so the reported lift
    retains a common request-level population despite missing outcomes.
    """

    if not isinstance(readiness, OpeReadinessAssessment):
        raise ValueError("readiness must be an OpeReadinessAssessment")
    if (
        readiness.decision != "pass"
        or not readiness.standard_ope_authorized
        or readiness.authorized_estimand != "registered_full_target_population"
        or readiness.propensity_mode != "known"
        or readiness.missing_outcome_rate <= 0.0
        or not readiness.observation_propensity_available
    ):
        raise ValueError("readiness does not authorize missingness-adjusted point OPE")
    _validate_evaluation_contract_binding(
        evaluation_contract,
        readiness,
        bootstrap_repetitions=bootstrap_repetitions,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
    )
    attestations = {
        "nuisance_predictions_cross_fitted": nuisance_predictions_cross_fitted,
        "logging_propensities_are_request_time_probabilities": (
            logging_propensities_are_request_time_probabilities
        ),
        "observation_propensities_are_validated": (
            observation_propensities_are_validated
        ),
        "observation_propensities_are_request_time_probabilities": (
            observation_propensities_are_request_time_probabilities
        ),
        "outcomes_match_logged_actions": outcomes_match_logged_actions,
        "conditional_missingness_independence": (conditional_missingness_independence),
    }
    failed_attestations = [
        name for name, value in attestations.items() if value is not True
    ]
    if failed_attestations:
        raise ValueError(
            "required identification attestations failed: "
            + ", ".join(failed_attestations)
        )
    if (
        not isinstance(bootstrap_repetitions, Integral)
        or isinstance(bootstrap_repetitions, bool)
        or bootstrap_repetitions <= 0
    ):
        raise ValueError("bootstrap_repetitions must be a positive integer")
    if (
        not isinstance(bootstrap_seed, Integral)
        or isinstance(bootstrap_seed, bool)
        or bootstrap_seed < 0
    ):
        raise ValueError("bootstrap_seed must be a nonnegative integer")
    if not _is_finite_real(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be finite and in (0, 1)")

    request_tuple = tuple(request_ids)
    user_tuple = tuple(user_ids)
    support_tuple = tuple(guaranteed_target_support_mass)
    target_tuple = tuple(target_probability_of_logged_action)
    propensity_tuple = tuple(logging_propensities)
    observation_propensity_tuple = tuple(outcome_observation_propensities)
    event_tuple = tuple(purchase_events)
    fold_tuple = tuple(fold_ids)
    q_logged_tuple = tuple(outcome_prediction_logged_action)
    q_target_tuple = tuple(outcome_prediction_target_policy)
    row_count = len(request_tuple)
    arrays = {
        "user_ids": user_tuple,
        "guaranteed_target_support_mass": support_tuple,
        "target_probability_of_logged_action": target_tuple,
        "logging_propensities": propensity_tuple,
        "outcome_observation_propensities": observation_propensity_tuple,
        "purchase_events": event_tuple,
        "fold_ids": fold_tuple,
        "outcome_prediction_logged_action": q_logged_tuple,
        "outcome_prediction_target_policy": q_target_tuple,
    }
    mismatched = [name for name, values in arrays.items() if len(values) != row_count]
    if row_count == 0 or mismatched:
        detail = ", ".join(mismatched) if mismatched else "empty input"
        raise ValueError("missingness-adjusted row arrays are inconsistent: " + detail)
    if row_count != readiness.raw_requests:
        raise ValueError("row count does not match readiness audit")

    digest = _row_data_digest(
        request_ids=request_tuple,
        user_ids=user_tuple,
        guaranteed_target_support_mass=support_tuple,
        target_probability_of_logged_action=target_tuple,
        purchase_events=event_tuple,
        fold_ids=fold_tuple,
        propensity_mode="known",
        logging_propensities=propensity_tuple,
        logging_propensity_lower=None,
        logging_propensity_upper=None,
        outcome_observation_propensities=observation_propensity_tuple,
    )
    if digest != readiness.row_data_digest:
        raise ValueError(
            "missingness-adjusted row data do not match the readiness audit digest"
        )
    if not all(
        value is None
        or (isinstance(value, Integral) and not isinstance(value, bool) and value >= 0)
        for value in event_tuple
    ):
        raise ValueError("purchase_events must be missing or nonnegative integers")
    observed = [value is not None for value in event_tuple]
    observed_count = sum(observed)
    if observed_count == 0 or observed_count == row_count:
        raise ValueError(
            "missingness-adjusted OPE requires both observed and missing outcomes"
        )
    if not all(
        _is_finite_real(value) and 0.0 < value <= 1.0
        for value in observation_propensity_tuple
    ):
        raise ValueError("observation propensities must be finite and in (0, 1]")
    if not all(
        _is_finite_real(value) and value >= 0.0
        for value in (*q_logged_tuple, *q_target_tuple)
    ):
        raise ValueError("outcome predictions must be finite and nonnegative")

    folds_by_user: dict[str, set[int]] = defaultdict(set)
    for user_id, fold_id in zip(user_tuple, fold_tuple, strict=True):
        folds_by_user[user_id].add(int(fold_id))
    if any(len(folds) != 1 for folds in folds_by_user.values()):
        raise ValueError("a user block may not cross nuisance-model folds")
    fold_count = len(set(fold_tuple))
    if fold_count < 2:
        raise ValueError("cross-fitted OPE requires at least two folds")

    policy_weights = [
        float(target) / float(propensity)
        for target, propensity in zip(target_tuple, propensity_tuple, strict=True)
    ]
    raw_combined_weights = [
        policy_weight / float(observation_propensity)
        for policy_weight, observation_propensity in zip(
            policy_weights,
            observation_propensity_tuple,
            strict=True,
        )
    ]
    raw_observation_weights = [
        1.0 / float(value) for value in observation_propensity_tuple
    ]
    combined_weights = [
        min(value, readiness.weight_clip) for value in raw_combined_weights
    ]
    observation_weights = [
        min(value, readiness.weight_clip) for value in raw_observation_weights
    ]

    ips_rows: list[float] = []
    snips_weights: list[float] = []
    target_dr_rows: list[float] = []
    logging_ipw_rows: list[float] = []
    logging_dr_rows: list[float] = []
    for (
        is_observed,
        event,
        combined_weight,
        observation_weight,
        q_logged,
        q_target,
    ) in zip(
        observed,
        event_tuple,
        combined_weights,
        observation_weights,
        q_logged_tuple,
        q_target_tuple,
        strict=True,
    ):
        if is_observed:
            outcome = float(event)
            ips_rows.append(combined_weight * outcome)
            snips_weights.append(combined_weight)
            target_dr_rows.append(
                float(q_target) + combined_weight * (outcome - float(q_logged))
            )
            logging_ipw_rows.append(observation_weight * outcome)
            logging_dr_rows.append(
                float(q_logged) + observation_weight * (outcome - float(q_logged))
            )
        else:
            ips_rows.append(0.0)
            snips_weights.append(0.0)
            target_dr_rows.append(float(q_target))
            logging_ipw_rows.append(0.0)
            logging_dr_rows.append(float(q_logged))
    snips_denominator = sum(snips_weights)
    if snips_denominator <= 0.0:
        raise ValueError("SNIPS denominator is zero")
    ips = sum(ips_rows) / row_count
    snips = sum(ips_rows) / snips_denominator
    target_value = sum(target_dr_rows) / row_count
    logging_ipw = sum(logging_ipw_rows) / row_count
    logging_dr = sum(logging_dr_rows) / row_count
    lift_rows = [
        target - logging
        for target, logging in zip(target_dr_rows, logging_dr_rows, strict=True)
    ]
    lift = sum(lift_rows) / row_count

    user_rows: dict[str, list[int]] = defaultdict(list)
    for index, user_id in enumerate(user_tuple):
        user_rows[user_id].append(index)
    users = tuple(sorted(user_rows))
    user_summaries = {
        user: (
            len(indices),
            sum(target_dr_rows[index] for index in indices),
            sum(lift_rows[index] for index in indices),
        )
        for user, indices in user_rows.items()
    }
    rng = Random(int(bootstrap_seed))
    bootstrap_target: list[float] = []
    bootstrap_lift: list[float] = []
    for _ in range(int(bootstrap_repetitions)):
        sampled = [users[rng.randrange(len(users))] for _ in users]
        sampled_rows = sum(user_summaries[user][0] for user in sampled)
        bootstrap_target.append(
            sum(user_summaries[user][1] for user in sampled) / sampled_rows
        )
        bootstrap_lift.append(
            sum(user_summaries[user][2] for user in sampled) / sampled_rows
        )
    alpha = (1.0 - float(confidence_level)) / 2.0

    support_rate = readiness.support_rate
    user_ess = readiness.user_effective_sample_size
    minimum_observation_propensity = readiness.minimum_observation_propensity
    if (
        support_rate is None
        or user_ess is None
        or minimum_observation_propensity is None
    ):
        raise ValueError("readiness audit lacks missingness diagnostics")
    return MissingnessAdjustedOpeEstimate(
        row_data_digest=digest,
        evaluation_contract_digest=readiness.evaluation_contract_digest,
        row_count=row_count,
        observed_outcome_count=observed_count,
        user_count=len(users),
        fold_count=fold_count,
        missing_outcome_rate=readiness.missing_outcome_rate,
        minimum_observation_propensity=minimum_observation_propensity,
        logging_ipw_value=logging_ipw,
        logging_doubly_robust_value=logging_dr,
        ips_target_value=ips,
        snips_target_value=snips,
        doubly_robust_target_value=target_value,
        doubly_robust_target_ci_low=_quantile(bootstrap_target, alpha),
        doubly_robust_target_ci_high=_quantile(bootstrap_target, 1.0 - alpha),
        doubly_robust_lift=lift,
        doubly_robust_lift_ci_low=_quantile(bootstrap_lift, alpha),
        doubly_robust_lift_ci_high=_quantile(bootstrap_lift, 1.0 - alpha),
        confidence_level=float(confidence_level),
        bootstrap_unit="pseudonymous_user",
        bootstrap_repetitions=int(bootstrap_repetitions),
        bootstrap_seed=int(bootstrap_seed),
        support_rate=support_rate,
        user_effective_sample_size=user_ess,
        weight_clip=readiness.weight_clip,
        combined_weight_clipped_fraction=(
            sum(value > readiness.weight_clip for value in raw_combined_weights)
            / row_count
        ),
        observation_weight_clipped_fraction=(
            sum(value > readiness.weight_clip for value in raw_observation_weights)
            / row_count
        ),
        maximum_raw_combined_weight=max(raw_combined_weights),
        nuisance_predictions_cross_fitted_attested=True,
        request_time_propensities_attested=True,
        observation_propensities_validated_attested=True,
        observation_propensities_request_time_attested=True,
        logged_action_outcomes_attested=True,
        conditional_missingness_independence_attested=True,
        conditional_causal_interpretation=True,
        production_rollout_authorized=False,
    )


def estimate_joint_uncertainty_bounds(
    readiness: OpeReadinessAssessment,
    *,
    request_ids: Sequence[str],
    user_ids: Sequence[str],
    guaranteed_target_support_mass: Sequence[float],
    target_probability_of_logged_action: Sequence[float],
    purchase_events: Sequence[int | None],
    fold_ids: Sequence[int],
    outcome_observation_propensities: Sequence[float],
    outcome_prediction_logged_action: Sequence[float],
    outcome_prediction_supported_target_contribution: Sequence[float],
    outcome_lower_bound: float,
    outcome_upper_bound: float,
    nuisance_predictions_cross_fitted: bool,
    propensity_evidence_is_request_time: bool,
    observation_propensities_are_validated: bool,
    observation_propensities_are_request_time_probabilities: bool,
    outcomes_match_logged_actions: bool,
    conditional_missingness_independence: bool,
    bootstrap_repetitions: int,
    bootstrap_seed: int,
    logging_propensities: Sequence[float] | None = None,
    logging_propensity_lower: Sequence[float] | None = None,
    logging_propensity_upper: Sequence[float] | None = None,
    propensity_intervals_cover_true_probabilities: bool | None = None,
    confidence_level: float = 0.95,
    evaluation_contract: OpeEvaluationContract | None = None,
) -> JointUncertaintyBoundsEstimate:
    """Combine support, propensity, and missing-outcome uncertainty.

    Known propensities yield zero propensity-identification width. Covering
    propensity intervals use sign-aware residual bounds. In both modes,
    observed residuals are divided by the outcome-observation propensity,
    missing rows retain their cross-fitted nuisance contribution, and
    unsupported target mass is bounded by the registered outcome range.
    """

    if not isinstance(readiness, OpeReadinessAssessment):
        raise ValueError("readiness must be an OpeReadinessAssessment")
    if (
        readiness.decision != "partial"
        or not readiness.partial_identification_authorized
        or readiness.propensity_mode not in {"known", "interval"}
        or readiness.missing_outcome_rate <= 0.0
        or not readiness.observation_propensity_available
    ):
        raise ValueError("readiness does not authorize joint uncertainty bounds")
    if (
        not _is_finite_real(outcome_lower_bound)
        or not _is_finite_real(outcome_upper_bound)
        or outcome_lower_bound >= outcome_upper_bound
    ):
        raise ValueError("outcome bounds must be finite and strictly ordered")
    _validate_evaluation_contract_binding(
        evaluation_contract,
        readiness,
        bootstrap_repetitions=bootstrap_repetitions,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
        outcome_lower_bound=outcome_lower_bound,
        outcome_upper_bound=outcome_upper_bound,
    )
    attestations = {
        "nuisance_predictions_cross_fitted": nuisance_predictions_cross_fitted,
        "propensity_evidence_is_request_time": propensity_evidence_is_request_time,
        "observation_propensities_are_validated": (
            observation_propensities_are_validated
        ),
        "observation_propensities_are_request_time_probabilities": (
            observation_propensities_are_request_time_probabilities
        ),
        "outcomes_match_logged_actions": outcomes_match_logged_actions,
        "conditional_missingness_independence": (conditional_missingness_independence),
    }
    if readiness.propensity_mode == "interval":
        attestations["propensity_intervals_cover_true_probabilities"] = (
            propensity_intervals_cover_true_probabilities
        )
    failed_attestations = [
        name for name, value in attestations.items() if value is not True
    ]
    if failed_attestations:
        raise ValueError(
            "required identification attestations failed: "
            + ", ".join(failed_attestations)
        )
    if (
        not isinstance(bootstrap_repetitions, Integral)
        or isinstance(bootstrap_repetitions, bool)
        or bootstrap_repetitions <= 0
    ):
        raise ValueError("bootstrap_repetitions must be a positive integer")
    if (
        not isinstance(bootstrap_seed, Integral)
        or isinstance(bootstrap_seed, bool)
        or bootstrap_seed < 0
    ):
        raise ValueError("bootstrap_seed must be a nonnegative integer")
    if not _is_finite_real(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be finite and in (0, 1)")

    request_tuple = tuple(request_ids)
    user_tuple = tuple(user_ids)
    support_tuple = tuple(guaranteed_target_support_mass)
    target_tuple = tuple(target_probability_of_logged_action)
    event_tuple = tuple(purchase_events)
    fold_tuple = tuple(fold_ids)
    observation_propensity_tuple = tuple(outcome_observation_propensities)
    q_logged_tuple = tuple(outcome_prediction_logged_action)
    q_supported_tuple = tuple(outcome_prediction_supported_target_contribution)
    known_tuple = (
        tuple(logging_propensities) if logging_propensities is not None else None
    )
    lower_tuple = (
        tuple(logging_propensity_lower)
        if logging_propensity_lower is not None
        else None
    )
    upper_tuple = (
        tuple(logging_propensity_upper)
        if logging_propensity_upper is not None
        else None
    )
    row_count = len(request_tuple)
    arrays: dict[str, Sequence[object]] = {
        "user_ids": user_tuple,
        "guaranteed_target_support_mass": support_tuple,
        "target_probability_of_logged_action": target_tuple,
        "purchase_events": event_tuple,
        "fold_ids": fold_tuple,
        "outcome_observation_propensities": observation_propensity_tuple,
        "outcome_prediction_logged_action": q_logged_tuple,
        "outcome_prediction_supported_target_contribution": q_supported_tuple,
    }
    if known_tuple is not None:
        arrays["logging_propensities"] = known_tuple
    if lower_tuple is not None:
        arrays["logging_propensity_lower"] = lower_tuple
    if upper_tuple is not None:
        arrays["logging_propensity_upper"] = upper_tuple
    mismatched = [name for name, values in arrays.items() if len(values) != row_count]
    if row_count == 0 or mismatched:
        detail = ", ".join(mismatched) if mismatched else "empty input"
        raise ValueError("joint-bound row arrays are inconsistent: " + detail)
    if row_count != readiness.raw_requests:
        raise ValueError("row count does not match readiness audit")
    if readiness.propensity_mode == "known":
        if known_tuple is None or lower_tuple is not None or upper_tuple is not None:
            raise ValueError("known mode requires only logging_propensities")
    elif known_tuple is not None or lower_tuple is None or upper_tuple is None:
        raise ValueError("interval mode requires only propensity interval arrays")

    digest = _row_data_digest(
        request_ids=request_tuple,
        user_ids=user_tuple,
        guaranteed_target_support_mass=support_tuple,
        target_probability_of_logged_action=target_tuple,
        purchase_events=event_tuple,
        fold_ids=fold_tuple,
        propensity_mode=readiness.propensity_mode,
        logging_propensities=known_tuple,
        logging_propensity_lower=lower_tuple,
        logging_propensity_upper=upper_tuple,
        outcome_observation_propensities=observation_propensity_tuple,
    )
    if digest != readiness.row_data_digest:
        raise ValueError("joint-bound row data do not match the readiness audit digest")
    if not all(
        value is None
        or (
            isinstance(value, Integral)
            and not isinstance(value, bool)
            and value >= 0
            and outcome_lower_bound <= float(value) <= outcome_upper_bound
        )
        for value in event_tuple
    ):
        raise ValueError(
            "purchase events must be missing or inside the registered bounds"
        )
    observed = [value is not None for value in event_tuple]
    observed_count = sum(observed)
    if observed_count == 0 or observed_count == row_count:
        raise ValueError("joint bounds require both observed and missing outcomes")
    if not all(
        _is_finite_real(value) and 0.0 < value <= 1.0
        for value in observation_propensity_tuple
    ):
        raise ValueError("observation propensities must be finite and in (0, 1]")
    if not all(
        _is_finite_real(value) and outcome_lower_bound <= value <= outcome_upper_bound
        for value in q_logged_tuple
    ):
        raise ValueError("logged-action predictions must lie inside outcome bounds")
    for support, contribution in zip(support_tuple, q_supported_tuple, strict=True):
        if not _is_finite_real(contribution):
            raise ValueError("supported-target contributions must be finite")
        contribution_lower = float(support) * float(outcome_lower_bound)
        contribution_upper = float(support) * float(outcome_upper_bound)
        if not contribution_lower - 1e-12 <= contribution <= contribution_upper + 1e-12:
            raise ValueError(
                "supported-target contribution lies outside support-scaled bounds"
            )

    folds_by_user: dict[str, set[int]] = defaultdict(set)
    for user_id, fold_id in zip(user_tuple, fold_tuple, strict=True):
        folds_by_user[user_id].add(int(fold_id))
    if any(len(folds) != 1 for folds in folds_by_user.values()):
        raise ValueError("a user block may not cross nuisance-model folds")
    fold_count = len(set(fold_tuple))
    if fold_count < 2:
        raise ValueError("cross-fitted bounds require at least two folds")

    weight_lower: list[float] = []
    weight_upper: list[float] = []
    if readiness.propensity_mode == "known":
        assert known_tuple is not None
        for target, propensity, observation_propensity in zip(
            target_tuple,
            known_tuple,
            observation_propensity_tuple,
            strict=True,
        ):
            if not _is_finite_real(propensity) or not 0.0 < propensity <= 1.0:
                raise ValueError("logging propensities must be finite and positive")
            weight = float(target) / (float(propensity) * float(observation_propensity))
            clipped = min(weight, readiness.weight_clip)
            weight_lower.append(clipped)
            weight_upper.append(clipped)
    else:
        assert lower_tuple is not None and upper_tuple is not None
        for target, propensity_lower, propensity_upper, observation_propensity in zip(
            target_tuple,
            lower_tuple,
            upper_tuple,
            observation_propensity_tuple,
            strict=True,
        ):
            if (
                not _is_finite_real(propensity_lower)
                or not _is_finite_real(propensity_upper)
                or not 0.0 < propensity_lower <= propensity_upper <= 1.0
            ):
                raise ValueError("propensity intervals must be positive and ordered")
            lower = float(target) / (
                float(propensity_upper) * float(observation_propensity)
            )
            upper = float(target) / (
                float(propensity_lower) * float(observation_propensity)
            )
            weight_lower.append(min(lower, readiness.weight_clip))
            weight_upper.append(min(upper, readiness.weight_clip))

    observation_weights = [
        min(1.0 / float(value), readiness.weight_clip)
        for value in observation_propensity_tuple
    ]
    supported_lower_rows: list[float] = []
    supported_upper_rows: list[float] = []
    logging_rows: list[float] = []
    for (
        is_observed,
        event,
        q_supported,
        q_logged,
        lower,
        upper,
        observation_weight,
    ) in zip(
        observed,
        event_tuple,
        q_supported_tuple,
        q_logged_tuple,
        weight_lower,
        weight_upper,
        observation_weights,
        strict=True,
    ):
        if is_observed:
            outcome = float(event)
            residual = outcome - float(q_logged)
            if residual >= 0.0:
                correction_lower = lower * residual
                correction_upper = upper * residual
            else:
                correction_lower = upper * residual
                correction_upper = lower * residual
            supported_lower_rows.append(float(q_supported) + correction_lower)
            supported_upper_rows.append(float(q_supported) + correction_upper)
            logging_rows.append(float(q_logged) + observation_weight * residual)
        else:
            supported_lower_rows.append(float(q_supported))
            supported_upper_rows.append(float(q_supported))
            logging_rows.append(float(q_logged))

    unsupported_mass = [1.0 - float(value) for value in support_tuple]
    supported_lower = sum(supported_lower_rows) / row_count
    supported_upper = sum(supported_upper_rows) / row_count
    mean_unsupported = sum(unsupported_mass) / row_count
    raw_target_lower = supported_lower + mean_unsupported * float(outcome_lower_bound)
    raw_target_upper = supported_upper + mean_unsupported * float(outcome_upper_bound)
    target_lower = min(
        max(raw_target_lower, float(outcome_lower_bound)),
        float(outcome_upper_bound),
    )
    target_upper = min(
        max(raw_target_upper, float(outcome_lower_bound)),
        float(outcome_upper_bound),
    )
    logging_value = sum(logging_rows) / row_count
    lift_lower = target_lower - logging_value
    lift_upper = target_upper - logging_value

    user_rows: dict[str, list[int]] = defaultdict(list)
    for index, user_id in enumerate(user_tuple):
        user_rows[user_id].append(index)
    users = tuple(sorted(user_rows))
    user_summaries = {
        user: (
            len(indices),
            sum(supported_lower_rows[index] for index in indices),
            sum(supported_upper_rows[index] for index in indices),
            sum(unsupported_mass[index] for index in indices),
            sum(logging_rows[index] for index in indices),
        )
        for user, indices in user_rows.items()
    }
    rng = Random(int(bootstrap_seed))
    bootstrap_target_lower: list[float] = []
    bootstrap_target_upper: list[float] = []
    bootstrap_lift_lower: list[float] = []
    bootstrap_lift_upper: list[float] = []
    for _ in range(int(bootstrap_repetitions)):
        sampled = [users[rng.randrange(len(users))] for _ in users]
        sampled_rows = sum(user_summaries[user][0] for user in sampled)
        sampled_lower = sum(user_summaries[user][1] for user in sampled)
        sampled_upper = sum(user_summaries[user][2] for user in sampled)
        sampled_unsupported = sum(user_summaries[user][3] for user in sampled)
        sampled_logging = (
            sum(user_summaries[user][4] for user in sampled) / sampled_rows
        )
        lower = (
            sampled_lower + sampled_unsupported * float(outcome_lower_bound)
        ) / sampled_rows
        upper = (
            sampled_upper + sampled_unsupported * float(outcome_upper_bound)
        ) / sampled_rows
        lower = min(max(lower, float(outcome_lower_bound)), float(outcome_upper_bound))
        upper = min(max(upper, float(outcome_lower_bound)), float(outcome_upper_bound))
        bootstrap_target_lower.append(lower)
        bootstrap_target_upper.append(upper)
        bootstrap_lift_lower.append(lower - sampled_logging)
        bootstrap_lift_upper.append(upper - sampled_logging)
    alpha = (1.0 - float(confidence_level)) / 2.0

    support_rate = readiness.support_rate
    minimum_observation_propensity = readiness.minimum_observation_propensity
    if support_rate is None or minimum_observation_propensity is None:
        raise ValueError("readiness audit lacks joint-uncertainty diagnostics")
    raw_upper_weights: list[float] = []
    if readiness.propensity_mode == "known":
        assert known_tuple is not None
        raw_upper_weights = [
            float(target) / (float(propensity) * float(observation_propensity))
            for target, propensity, observation_propensity in zip(
                target_tuple,
                known_tuple,
                observation_propensity_tuple,
                strict=True,
            )
        ]
    else:
        assert lower_tuple is not None
        raw_upper_weights = [
            float(target) / (float(lower) * float(observation_propensity))
            for target, lower, observation_propensity in zip(
                target_tuple,
                lower_tuple,
                observation_propensity_tuple,
                strict=True,
            )
        ]
    return JointUncertaintyBoundsEstimate(
        row_data_digest=digest,
        evaluation_contract_digest=readiness.evaluation_contract_digest,
        propensity_mode=readiness.propensity_mode,
        row_count=row_count,
        observed_outcome_count=observed_count,
        user_count=len(users),
        fold_count=fold_count,
        support_rate=support_rate,
        missing_outcome_rate=readiness.missing_outcome_rate,
        minimum_observation_propensity=minimum_observation_propensity,
        mean_unsupported_target_mass=mean_unsupported,
        outcome_lower_bound=float(outcome_lower_bound),
        outcome_upper_bound=float(outcome_upper_bound),
        logging_doubly_robust_value=logging_value,
        supported_contribution_lower=supported_lower,
        supported_contribution_upper=supported_upper,
        target_value_lower=target_lower,
        target_value_upper=target_upper,
        target_value_confidence_lower=_quantile(bootstrap_target_lower, alpha),
        target_value_confidence_upper=_quantile(bootstrap_target_upper, 1.0 - alpha),
        lift_lower=lift_lower,
        lift_upper=lift_upper,
        lift_confidence_lower=_quantile(bootstrap_lift_lower, alpha),
        lift_confidence_upper=_quantile(bootstrap_lift_upper, 1.0 - alpha),
        propensity_uncertainty_width=supported_upper - supported_lower,
        support_uncertainty_width=(
            mean_unsupported * (float(outcome_upper_bound) - float(outcome_lower_bound))
        ),
        identification_width=target_upper - target_lower,
        confidence_level=float(confidence_level),
        bootstrap_unit="pseudonymous_user",
        bootstrap_repetitions=int(bootstrap_repetitions),
        bootstrap_seed=int(bootstrap_seed),
        weight_clip=readiness.weight_clip,
        combined_weight_clipped_fraction_upper_bound=(
            sum(value > readiness.weight_clip for value in raw_upper_weights)
            / row_count
        ),
        observation_weight_clipped_fraction=(
            sum(
                1.0 / float(value) > readiness.weight_clip
                for value in observation_propensity_tuple
            )
            / row_count
        ),
        propensity_interval_coverage_attested=(
            True if readiness.propensity_mode == "interval" else None
        ),
        nuisance_predictions_cross_fitted_attested=True,
        request_time_propensity_evidence_attested=True,
        observation_propensities_validated_attested=True,
        observation_propensities_request_time_attested=True,
        logged_action_outcomes_attested=True,
        conditional_missingness_independence_attested=True,
        conditional_partial_identification=True,
        production_rollout_authorized=False,
    )
