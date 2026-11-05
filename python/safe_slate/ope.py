"""Aggregate, user-blocked readiness gates for off-policy evaluation."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime
from hashlib import sha256
import json
from math import isfinite
from numbers import Integral, Real
from typing import Any, Literal, Mapping, Sequence

from .contract import OpeEvaluationContract
from .input_manifest import (
    ImmutableInputManifest,
    OPE_REQUIRED_INPUT_ROLES,
    ObservedInputIdentity,
    ObservedInputVerification,
    audit_input_manifest_observations,
)
from .population_positivity import (
    PopulationPositivityEvidence,
    StatisticalPopulationPositivityEvidence,
    canonical_population_positivity_evidence_bytes,
)


PropensityMode = Literal["known", "interval", "missing"]


@dataclass(frozen=True, slots=True)
class OpeReadinessAssessment:
    """Fail-closed authorization boundary before running an OPE estimator."""

    propensity_mode: str
    evaluation_contract_digest: str | None
    row_data_digest: str
    decision: str
    authorized_estimand: str
    raw_requests: int
    minimum_raw_requests: int
    analyzable_requests: int
    unique_users: int
    repeated_user_request_fraction: float
    maximum_requests_per_user: int
    support_rate: float | None
    population_positivity_established: bool
    full_estimand_support_threshold: float
    partial_estimand_support_threshold: float
    missing_outcome_rate: float
    maximum_missing_outcome_rate: float
    observation_propensity_available: bool
    minimum_observation_propensity: float | None
    maximum_observation_inverse_weight: float | None
    purchase_events: int | None
    minimum_purchase_events: int
    cross_fit_fold_count: int
    cross_fit_ready: bool
    weight_clip: float
    p99_importance_weight_upper_bound: float | None
    maximum_importance_weight_upper_bound: float | None
    maximum_allowed_p99_importance_weight: float
    clipped_fraction_upper_bound: float | None
    user_effective_sample_size: float | None
    user_effective_sample_size_lower_bound: float | None
    effective_sample_size_kind: str
    minimum_user_effective_sample_size: float
    standard_ope_authorized: bool
    supported_subpopulation_analysis_authorized: bool
    partial_identification_authorized: bool
    production_rollout_authorized: bool
    reasons: tuple[str, ...]
    positivity_failure_probability: float | None = None
    ope_failure_probability: float | None = None
    joint_failure_probability_upper_bound: float | None = None
    joint_confidence_level_lower_bound: float | None = None
    joint_confidence_composition: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "propensity_mode": self.propensity_mode,
            "evaluation_contract_digest": self.evaluation_contract_digest,
            "row_data_digest": self.row_data_digest,
            "decision": self.decision,
            "authorized_estimand": self.authorized_estimand,
            "raw_requests": self.raw_requests,
            "minimum_raw_requests": self.minimum_raw_requests,
            "analyzable_requests": self.analyzable_requests,
            "unique_users": self.unique_users,
            "repeated_user_request_fraction": (self.repeated_user_request_fraction),
            "maximum_requests_per_user": self.maximum_requests_per_user,
            "support_rate": self.support_rate,
            "population_positivity_established": (
                self.population_positivity_established
            ),
            "full_estimand_support_threshold": (self.full_estimand_support_threshold),
            "partial_estimand_support_threshold": (
                self.partial_estimand_support_threshold
            ),
            "missing_outcome_rate": self.missing_outcome_rate,
            "maximum_missing_outcome_rate": self.maximum_missing_outcome_rate,
            "observation_propensity_available": (self.observation_propensity_available),
            "minimum_observation_propensity": self.minimum_observation_propensity,
            "maximum_observation_inverse_weight": (
                self.maximum_observation_inverse_weight
            ),
            "purchase_events": self.purchase_events,
            "minimum_purchase_events": self.minimum_purchase_events,
            "cross_fit_fold_count": self.cross_fit_fold_count,
            "cross_fit_ready": self.cross_fit_ready,
            "weight_clip": self.weight_clip,
            "p99_importance_weight_upper_bound": (
                self.p99_importance_weight_upper_bound
            ),
            "maximum_importance_weight_upper_bound": (
                self.maximum_importance_weight_upper_bound
            ),
            "maximum_allowed_p99_importance_weight": (
                self.maximum_allowed_p99_importance_weight
            ),
            "clipped_fraction_upper_bound": self.clipped_fraction_upper_bound,
            "user_effective_sample_size": self.user_effective_sample_size,
            "user_effective_sample_size_lower_bound": (
                self.user_effective_sample_size_lower_bound
            ),
            "effective_sample_size_kind": self.effective_sample_size_kind,
            "minimum_user_effective_sample_size": (
                self.minimum_user_effective_sample_size
            ),
            "standard_ope_authorized": self.standard_ope_authorized,
            "supported_subpopulation_analysis_authorized": (
                self.supported_subpopulation_analysis_authorized
            ),
            "partial_identification_authorized": (
                self.partial_identification_authorized
            ),
            "production_rollout_authorized": self.production_rollout_authorized,
            "reasons": list(self.reasons),
            "positivity_failure_probability": (
                self.positivity_failure_probability
            ),
            "ope_failure_probability": self.ope_failure_probability,
            "joint_failure_probability_upper_bound": (
                self.joint_failure_probability_upper_bound
            ),
            "joint_confidence_level_lower_bound": (
                self.joint_confidence_level_lower_bound
            ),
            "joint_confidence_composition": self.joint_confidence_composition,
        }


@dataclass(frozen=True, slots=True)
class ManifestBoundOpeReadinessAssessment:
    """Input-identity gate followed by the contract-bound OPE readiness gate."""

    evaluation_contract_digest: str
    input_manifest_digest: str
    immutable_inputs_verified: bool
    population_positivity_evidence_input_id: str | None
    population_positivity_evidence_id: str | None
    population_positivity_evidence_sha256: str | None
    population_positivity_schema_version: str | None
    population_positivity_guarantee_kind: str | None
    population_positivity_simultaneous_failure_probability: float | None
    population_positivity_validity_start: str | None
    population_positivity_validity_end: str | None
    ope_failure_probability: float
    joint_failure_probability_upper_bound: float | None
    joint_confidence_level_lower_bound: float | None
    joint_confidence_composition: str | None
    decision: str
    input_verification: ObservedInputVerification
    readiness: OpeReadinessAssessment | None
    production_rollout_authorized: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluation_contract_digest": self.evaluation_contract_digest,
            "input_manifest_digest": self.input_manifest_digest,
            "immutable_inputs_verified": self.immutable_inputs_verified,
            "population_positivity_evidence_input_id": (
                self.population_positivity_evidence_input_id
            ),
            "population_positivity_evidence_id": (
                self.population_positivity_evidence_id
            ),
            "population_positivity_evidence_sha256": (
                self.population_positivity_evidence_sha256
            ),
            "population_positivity_schema_version": (
                self.population_positivity_schema_version
            ),
            "population_positivity_guarantee_kind": (
                self.population_positivity_guarantee_kind
            ),
            "population_positivity_simultaneous_failure_probability": (
                self.population_positivity_simultaneous_failure_probability
            ),
            "population_positivity_validity_start": (
                self.population_positivity_validity_start
            ),
            "population_positivity_validity_end": (
                self.population_positivity_validity_end
            ),
            "ope_failure_probability": self.ope_failure_probability,
            "joint_failure_probability_upper_bound": (
                self.joint_failure_probability_upper_bound
            ),
            "joint_confidence_level_lower_bound": (
                self.joint_confidence_level_lower_bound
            ),
            "joint_confidence_composition": self.joint_confidence_composition,
            "decision": self.decision,
            "input_verification": self.input_verification.to_dict(),
            "readiness": self.readiness.to_dict() if self.readiness else None,
            "production_rollout_authorized": False,
            "reasons": list(self.reasons),
        }


def _is_finite_real(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and isfinite(value)


def _quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _digest_value(value: object) -> object:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        numeric = float(value)
        if isfinite(numeric):
            return numeric
        if numeric != numeric:
            return "__nonfinite__:nan"
        return (
            "__nonfinite__:positive_infinity"
            if numeric > 0
            else ("__nonfinite__:negative_infinity")
        )
    if isinstance(value, Sequence):
        return [_digest_value(item) for item in value]
    if isinstance(value, Mapping):
        return {
            str(key): _digest_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    return f"__unsupported__:{type(value).__qualname__}:{value!r}"


def _row_data_digest(
    *,
    request_ids: Sequence[str],
    user_ids: Sequence[str],
    guaranteed_target_support_mass: Sequence[float],
    target_probability_of_logged_action: Sequence[float],
    purchase_events: Sequence[int | None],
    fold_ids: Sequence[int] | None,
    propensity_mode: str,
    logging_propensities: Sequence[float] | None,
    logging_propensity_lower: Sequence[float] | None,
    logging_propensity_upper: Sequence[float] | None,
    outcome_observation_propensities: Sequence[float] | None,
) -> str:
    payload = {
        "digest_version": "safe-slate-ope-row-v1",
        "request_ids": request_ids,
        "user_ids": user_ids,
        "guaranteed_target_support_mass": guaranteed_target_support_mass,
        "target_probability_of_logged_action": (target_probability_of_logged_action),
        "purchase_events": purchase_events,
        "fold_ids": fold_ids,
        "propensity_mode": propensity_mode,
        "logging_propensities": logging_propensities,
        "logging_propensity_lower": logging_propensity_lower,
        "logging_propensity_upper": logging_propensity_upper,
        "outcome_observation_propensities": outcome_observation_propensities,
    }
    encoded = json.dumps(
        _digest_value(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _validate_policy_thresholds(
    *,
    full_estimand_support_threshold: float,
    partial_estimand_support_threshold: float,
    minimum_user_effective_sample_size: float,
    minimum_purchase_events: int,
    maximum_missing_outcome_rate: float,
    maximum_allowed_p99_importance_weight: float,
    weight_clip: float,
    minimum_raw_requests: int = 1,
) -> None:
    rate_values = {
        "full_estimand_support_threshold": full_estimand_support_threshold,
        "partial_estimand_support_threshold": partial_estimand_support_threshold,
        "maximum_missing_outcome_rate": maximum_missing_outcome_rate,
    }
    for name, value in rate_values.items():
        if not _is_finite_real(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1]")
    if partial_estimand_support_threshold > full_estimand_support_threshold:
        raise ValueError(
            "partial_estimand_support_threshold may not exceed the full threshold"
        )
    positive_values = {
        "minimum_user_effective_sample_size": minimum_user_effective_sample_size,
        "maximum_allowed_p99_importance_weight": (
            maximum_allowed_p99_importance_weight
        ),
        "weight_clip": weight_clip,
    }
    for name, value in positive_values.items():
        if not _is_finite_real(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if (
        not isinstance(minimum_purchase_events, Integral)
        or isinstance(minimum_purchase_events, bool)
        or minimum_purchase_events < 0
    ):
        raise ValueError("minimum_purchase_events must be a nonnegative integer")


def _audit_ope_readiness(
    *,
    request_ids: Sequence[str],
    user_ids: Sequence[str],
    guaranteed_target_support_mass: Sequence[float],
    target_probability_of_logged_action: Sequence[float],
    purchase_events: Sequence[int | None],
    fold_ids: Sequence[int] | None,
    propensity_mode: PropensityMode,
    candidate_sets_reconstructable: bool | None,
    labels_settled: bool | None,
    missingness_independent: bool | None,
    population_positivity_established: bool = False,
    minimum_user_effective_sample_size: float,
    minimum_purchase_events: int,
    maximum_missing_outcome_rate: float,
    maximum_allowed_p99_importance_weight: float,
    weight_clip: float,
    minimum_raw_requests: int = 1,
    logging_propensities: Sequence[float] | None = None,
    logging_propensity_lower: Sequence[float] | None = None,
    logging_propensity_upper: Sequence[float] | None = None,
    outcome_observation_propensities: Sequence[float] | None = None,
    full_estimand_support_threshold: float = 1.0,
    partial_estimand_support_threshold: float = 0.80,
    _evaluation_contract_digest: str | None = None,
    _population_positivity_verified: bool = False,
) -> OpeReadinessAssessment:
    """Audit aggregate overlap and sufficiency before OPE.

    Each row is one logged request. ``guaranteed_target_support_mass`` is the
    target-policy probability mass whose logging propensity is known positive
    for that request. Known propensities can authorize point OPE. Covering
    propensity intervals can authorize only partial-identification analysis;
    the function uses a conservative user-block ESS lower bound.

    ``population_positivity_established`` is a requested design-level premise,
    not evidence by itself. It becomes effective only inside the
    manifest-bound API after exact verification of the registered positivity
    artifact. Finite logged rows with 100% empirical support do not establish
    positivity on unseen target-population contexts. Passing this gate does not estimate policy value and never authorizes a
    rollout. Nuisance predictions must still be cross-fitted and the final
    estimator must preserve the authorized estimand returned here.
    """

    _validate_policy_thresholds(
        full_estimand_support_threshold=full_estimand_support_threshold,
        partial_estimand_support_threshold=partial_estimand_support_threshold,
        minimum_user_effective_sample_size=minimum_user_effective_sample_size,
        minimum_purchase_events=minimum_purchase_events,
        maximum_missing_outcome_rate=maximum_missing_outcome_rate,
        maximum_allowed_p99_importance_weight=(maximum_allowed_p99_importance_weight),
        weight_clip=weight_clip,
    )
    if propensity_mode not in {"known", "interval", "missing"}:
        raise ValueError("propensity_mode must be known, interval, or missing")
    if not isinstance(population_positivity_established, bool):
        raise ValueError("population_positivity_established must be boolean")
    positivity_requested_without_binding = (
        population_positivity_established
        and not _population_positivity_verified
    )
    population_positivity_established = bool(
        population_positivity_established
        and _population_positivity_verified
    )
    if (
        not isinstance(minimum_raw_requests, Integral)
        or isinstance(minimum_raw_requests, bool)
        or minimum_raw_requests <= 0
    ):
        raise ValueError("minimum_raw_requests must be a positive integer")
    if _evaluation_contract_digest is not None and (
        not isinstance(_evaluation_contract_digest, str)
        or len(_evaluation_contract_digest) != 64
        or any(
            character not in "0123456789abcdef"
            for character in _evaluation_contract_digest
        )
    ):
        raise ValueError("evaluation contract digest must be lowercase SHA-256")

    request_tuple = tuple(request_ids)
    user_tuple = tuple(user_ids)
    support_values = tuple(guaranteed_target_support_mass)
    target_values = tuple(target_probability_of_logged_action)
    event_values = tuple(purchase_events)
    row_count = len(request_tuple)
    if row_count == 0:
        raise ValueError("OPE readiness audit requires at least one request")
    lengths = {
        "user_ids": len(user_tuple),
        "guaranteed_target_support_mass": len(support_values),
        "target_probability_of_logged_action": len(target_values),
        "purchase_events": len(event_values),
    }
    mismatched = [name for name, length in lengths.items() if length != row_count]
    if mismatched:
        raise ValueError("row-array length mismatch: " + ", ".join(mismatched))
    if fold_ids is not None and len(fold_ids) != row_count:
        raise ValueError("row-array length mismatch: fold_ids")
    if (
        outcome_observation_propensities is not None
        and len(outcome_observation_propensities) != row_count
    ):
        raise ValueError("row-array length mismatch: outcome_observation_propensities")
    if any(not isinstance(value, str) or not value for value in request_tuple):
        raise ValueError("request_ids must be non-empty strings")
    if len(set(request_tuple)) != row_count:
        raise ValueError("request_ids must be unique")
    if any(not isinstance(value, str) or not value for value in user_tuple):
        raise ValueError("user_ids must be non-empty strings")

    supplied_known = logging_propensities is not None
    supplied_interval = (
        logging_propensity_lower is not None or logging_propensity_upper is not None
    )
    if propensity_mode == "known":
        if not supplied_known or supplied_interval:
            raise ValueError("known mode requires only logging_propensities")
        if len(logging_propensities) != row_count:
            raise ValueError("row-array length mismatch: logging_propensities")
    elif propensity_mode == "interval":
        if (
            supplied_known
            or logging_propensity_lower is None
            or logging_propensity_upper is None
        ):
            raise ValueError(
                "interval mode requires only lower and upper propensity arrays"
            )
        if (
            len(logging_propensity_lower) != row_count
            or len(logging_propensity_upper) != row_count
        ):
            raise ValueError("row-array length mismatch: propensity intervals")
    elif supplied_known or supplied_interval:
        raise ValueError("missing mode may not include propensity arrays")

    row_data_digest = _row_data_digest(
        request_ids=request_tuple,
        user_ids=user_tuple,
        guaranteed_target_support_mass=support_values,
        target_probability_of_logged_action=target_values,
        purchase_events=event_values,
        fold_ids=fold_ids,
        propensity_mode=propensity_mode,
        logging_propensities=logging_propensities,
        logging_propensity_lower=logging_propensity_lower,
        logging_propensity_upper=logging_propensity_upper,
        outcome_observation_propensities=outcome_observation_propensities,
    )

    reasons: list[str] = []
    hard_failures: list[str] = []
    if positivity_requested_without_binding:
        reasons.append("population_positivity_evidence_not_manifest_bound")

    def fail(reason: str) -> None:
        reasons.append(reason)
        hard_failures.append(reason)

    if row_count < minimum_raw_requests:
        fail("raw_request_count_below_gate")

    support_valid = all(
        _is_finite_real(value) and 0.0 <= value <= 1.0 for value in support_values
    )
    target_valid = all(
        _is_finite_real(value) and 0.0 <= value <= 1.0 for value in target_values
    )
    support_rate = (
        sum(float(value) for value in support_values) / row_count
        if support_valid
        else None
    )
    if not support_valid:
        fail("target_support_mass_invalid")
    if not target_valid:
        fail("target_probability_invalid")

    if candidate_sets_reconstructable is not True:
        fail(
            "candidate_sets_not_reconstructable"
            if candidate_sets_reconstructable is False
            else "candidate_reconstruction_unknown"
        )
    if labels_settled is not True:
        fail("labels_not_settled" if labels_settled is False else "settlement_unknown")
    if missingness_independent is not True:
        fail(
            "outcome_observation_not_independent"
            if missingness_independent is False
            else "outcome_observation_unknown"
        )

    event_valid = all(
        value is None
        or (isinstance(value, Integral) and not isinstance(value, bool) and value >= 0)
        for value in event_values
    )
    analyzable_indices = tuple(
        index for index, value in enumerate(event_values) if value is not None
    )
    missing_outcome_rate = 1.0 - len(analyzable_indices) / row_count
    event_count = (
        sum(int(event_values[index]) for index in analyzable_indices)
        if event_valid
        else None
    )
    if not event_valid:
        fail("purchase_event_invalid")
    if not analyzable_indices:
        fail("no_analyzable_requests")
    if missing_outcome_rate > maximum_missing_outcome_rate:
        fail("missing_outcome_rate_above_gate")
    if event_count is not None and event_count < minimum_purchase_events:
        fail("purchase_events_below_gate")

    observation_propensity_available = outcome_observation_propensities is not None
    if outcome_observation_propensities is None:
        observation_propensity_valid = missing_outcome_rate == 0.0
        observation_propensity_values = [1.0] * row_count
        if missing_outcome_rate > 0.0:
            fail("observation_propensities_missing")
    else:
        observation_propensity_values = list(outcome_observation_propensities)
        observation_propensity_valid = all(
            _is_finite_real(value) and 0.0 < value <= 1.0
            for value in observation_propensity_values
        )
        if not observation_propensity_valid:
            fail("observation_propensity_invalid")
    minimum_observation_propensity = (
        min(float(value) for value in observation_propensity_values)
        if observation_propensity_valid
        else None
    )
    maximum_observation_inverse_weight = (
        1.0 / minimum_observation_propensity
        if minimum_observation_propensity is not None
        else None
    )

    fold_tuple = tuple(fold_ids) if fold_ids is not None else ()
    fold_values_valid = fold_ids is not None and all(
        isinstance(value, Integral) and not isinstance(value, bool) and value >= 0
        for value in fold_tuple
    )
    fold_count = len(set(fold_tuple)) if fold_values_valid else 0
    user_fold_consistent = False
    if fold_ids is None:
        fail("cross_fit_folds_missing")
    elif not fold_values_valid:
        fail("cross_fit_fold_invalid")
    else:
        user_folds: dict[str, set[int]] = defaultdict(set)
        for user_id, fold_id in zip(user_tuple, fold_tuple, strict=True):
            user_folds[user_id].add(int(fold_id))
        user_fold_consistent = all(len(values) == 1 for values in user_folds.values())
        if not user_fold_consistent:
            fail("user_crosses_cross_fit_folds")
        if fold_count < 2:
            fail("cross_fit_requires_two_folds")
    cross_fit_ready = fold_values_valid and user_fold_consistent and fold_count >= 2

    all_lower_weights: list[float] = []
    all_upper_weights: list[float] = []
    propensity_valid = True
    if propensity_mode == "missing":
        propensity_valid = False
        fail("propensities_missing")
    elif target_valid:
        if propensity_mode == "known":
            assert logging_propensities is not None
            for index in range(row_count):
                propensity = logging_propensities[index]
                if not _is_finite_real(propensity) or not 0.0 < propensity <= 1.0:
                    propensity_valid = False
                    break
                weight = float(target_values[index]) / float(propensity)
                all_lower_weights.append(weight)
                all_upper_weights.append(weight)
        else:
            assert logging_propensity_lower is not None
            assert logging_propensity_upper is not None
            for index in range(row_count):
                lower = logging_propensity_lower[index]
                upper = logging_propensity_upper[index]
                target = float(target_values[index])
                if (
                    not _is_finite_real(lower)
                    or not _is_finite_real(upper)
                    or not 0.0 <= lower <= upper <= 1.0
                    or upper <= 0.0
                ):
                    propensity_valid = False
                    break
                if target > 0.0 and lower <= 0.0:
                    propensity_valid = False
                    reasons.append("propensity_interval_not_strictly_positive")
                    break
                all_lower_weights.append(
                    0.0 if target == 0.0 else target / float(upper)
                )
                all_upper_weights.append(
                    0.0 if target == 0.0 else target / float(lower)
                )
        if not propensity_valid:
            fail("propensity_invalid")
    else:
        propensity_valid = False

    if propensity_valid and support_valid and target_valid:
        if any(
            float(target_values[index]) > float(support_values[index]) + 1e-12
            for index in range(row_count)
        ):
            fail("logged_action_probability_exceeds_guaranteed_support_mass")

    if propensity_valid and observation_propensity_valid:
        all_lower_weights = [
            weight / float(observation_propensity)
            for weight, observation_propensity in zip(
                all_lower_weights,
                observation_propensity_values,
                strict=True,
            )
        ]
        all_upper_weights = [
            weight / float(observation_propensity)
            for weight, observation_propensity in zip(
                all_upper_weights,
                observation_propensity_values,
                strict=True,
            )
        ]

    lower_weights = (
        [all_lower_weights[index] for index in analyzable_indices]
        if propensity_valid and observation_propensity_valid
        else []
    )
    upper_weights = (
        [all_upper_weights[index] for index in analyzable_indices]
        if propensity_valid and observation_propensity_valid
        else []
    )

    p99_weight = (
        _quantile(all_upper_weights, 0.99)
        if propensity_valid and observation_propensity_valid
        else None
    )
    maximum_weight = (
        max(all_upper_weights, default=None)
        if propensity_valid and observation_propensity_valid
        else None
    )
    clipped_fraction = (
        sum(weight > weight_clip for weight in all_upper_weights)
        / len(all_upper_weights)
        if all_upper_weights and observation_propensity_valid
        else None
    )
    if p99_weight is not None and p99_weight > maximum_allowed_p99_importance_weight:
        fail("importance_weight_tail_above_gate")

    exact_ess: float | None = None
    ess_lower_bound: float | None = None
    ess_kind = "unavailable"
    if (
        propensity_valid
        and observation_propensity_valid
        and lower_weights
        and len(lower_weights) == len(analyzable_indices)
    ):
        user_lower: dict[str, float] = defaultdict(float)
        user_upper: dict[str, float] = defaultdict(float)
        for row_index, lower, upper in zip(
            analyzable_indices, lower_weights, upper_weights, strict=True
        ):
            user = user_tuple[row_index]
            user_lower[user] += min(lower, weight_clip)
            user_upper[user] += min(upper, weight_clip)
        lower_sum = sum(user_lower.values())
        upper_squared_sum = sum(value * value for value in user_upper.values())
        ess_lower_bound = (
            0.0
            if upper_squared_sum == 0.0
            else lower_sum * lower_sum / upper_squared_sum
        )
        if propensity_mode == "known":
            exact_ess = ess_lower_bound
            ess_kind = "exact_user_block"
        else:
            ess_kind = "conservative_user_block_lower_bound"
        if ess_lower_bound < minimum_user_effective_sample_size:
            fail("user_effective_sample_size_below_gate")
        if lower_sum <= 0.0:
            fail("zero_target_weight_mass")
    else:
        fail("user_effective_sample_size_unavailable")

    user_counts = Counter(user_tuple)
    unique_users = len(user_counts)
    repeated_fraction = 1.0 - unique_users / row_count
    max_requests_per_user = max(user_counts.values())

    decision = "fail"
    estimand = "descriptive_replay_only"
    standard_ope_authorized = False
    supported_analysis_authorized = False
    partial_identification_authorized = False
    if not hard_failures and support_rate is not None:
        if support_rate < partial_estimand_support_threshold:
            reasons.append("target_support_below_partial_gate")
        elif propensity_mode == "interval":
            decision = "partial"
            partial_identification_authorized = True
            if support_rate == 1.0 and not population_positivity_established:
                estimand = "observed_finite_log_propensity_interval_bounds"
                reasons.append("population_positivity_not_established")
                reasons.append(
                    "finite_log_support_does_not_imply_population_positivity"
                )
            else:
                estimand = (
                    "full_population_propensity_interval_bounds"
                    if support_rate == 1.0
                    else "full_population_joint_propensity_and_support_bounds"
                )
            reasons.append("propensity_interval_requires_partial_identification")
            if support_rate < 1.0:
                reasons.append("full_estimand_support_below_threshold")
        elif support_rate == 1.0:
            if population_positivity_established:
                decision = "pass"
                estimand = "registered_full_target_population"
                standard_ope_authorized = True
            else:
                decision = "partial"
                estimand = "observed_finite_log_population_only"
                reasons.append("population_positivity_not_established")
                reasons.append(
                    "finite_log_support_does_not_imply_population_positivity"
                )
        else:
            decision = "partial"
            estimand = "full_population_partial_support_bounds"
            partial_identification_authorized = True
            reasons.append("full_estimand_support_below_threshold")
            reasons.append("partial_support_requires_partial_identification")
    elif support_rate is not None and support_rate < partial_estimand_support_threshold:
        reasons.append("target_support_below_partial_gate")

    return OpeReadinessAssessment(
        propensity_mode=propensity_mode,
        evaluation_contract_digest=_evaluation_contract_digest,
        row_data_digest=row_data_digest,
        decision=decision,
        authorized_estimand=estimand,
        raw_requests=row_count,
        minimum_raw_requests=minimum_raw_requests,
        analyzable_requests=len(analyzable_indices),
        unique_users=unique_users,
        repeated_user_request_fraction=repeated_fraction,
        maximum_requests_per_user=max_requests_per_user,
        support_rate=support_rate,
        population_positivity_established=population_positivity_established,
        full_estimand_support_threshold=full_estimand_support_threshold,
        partial_estimand_support_threshold=partial_estimand_support_threshold,
        missing_outcome_rate=missing_outcome_rate,
        maximum_missing_outcome_rate=maximum_missing_outcome_rate,
        observation_propensity_available=observation_propensity_available,
        minimum_observation_propensity=minimum_observation_propensity,
        maximum_observation_inverse_weight=maximum_observation_inverse_weight,
        purchase_events=event_count,
        minimum_purchase_events=minimum_purchase_events,
        cross_fit_fold_count=fold_count,
        cross_fit_ready=cross_fit_ready,
        weight_clip=weight_clip,
        p99_importance_weight_upper_bound=p99_weight,
        maximum_importance_weight_upper_bound=maximum_weight,
        maximum_allowed_p99_importance_weight=(maximum_allowed_p99_importance_weight),
        clipped_fraction_upper_bound=clipped_fraction,
        user_effective_sample_size=exact_ess,
        user_effective_sample_size_lower_bound=ess_lower_bound,
        effective_sample_size_kind=ess_kind,
        minimum_user_effective_sample_size=(minimum_user_effective_sample_size),
        standard_ope_authorized=standard_ope_authorized,
        supported_subpopulation_analysis_authorized=(supported_analysis_authorized),
        partial_identification_authorized=partial_identification_authorized,
        production_rollout_authorized=False,
        reasons=tuple(dict.fromkeys(reasons)),
    )


def audit_ope_readiness(
    *,
    request_ids: Sequence[str],
    user_ids: Sequence[str],
    guaranteed_target_support_mass: Sequence[float],
    target_probability_of_logged_action: Sequence[float],
    purchase_events: Sequence[int | None],
    fold_ids: Sequence[int] | None,
    propensity_mode: PropensityMode,
    candidate_sets_reconstructable: bool | None,
    labels_settled: bool | None,
    missingness_independent: bool | None,
    population_positivity_established: bool = False,
    minimum_user_effective_sample_size: float,
    minimum_purchase_events: int,
    maximum_missing_outcome_rate: float,
    maximum_allowed_p99_importance_weight: float,
    weight_clip: float,
    minimum_raw_requests: int = 1,
    logging_propensities: Sequence[float] | None = None,
    logging_propensity_lower: Sequence[float] | None = None,
    logging_propensity_upper: Sequence[float] | None = None,
    outcome_observation_propensities: Sequence[float] | None = None,
    full_estimand_support_threshold: float = 1.0,
    partial_estimand_support_threshold: float = 0.80,
) -> OpeReadinessAssessment:
    """Audit rows without treating a caller-supplied positivity flag as evidence."""

    return _audit_ope_readiness(
        request_ids=request_ids,
        user_ids=user_ids,
        guaranteed_target_support_mass=guaranteed_target_support_mass,
        target_probability_of_logged_action=target_probability_of_logged_action,
        purchase_events=purchase_events,
        fold_ids=fold_ids,
        propensity_mode=propensity_mode,
        candidate_sets_reconstructable=candidate_sets_reconstructable,
        labels_settled=labels_settled,
        missingness_independent=missingness_independent,
        population_positivity_established=population_positivity_established,
        minimum_user_effective_sample_size=minimum_user_effective_sample_size,
        minimum_purchase_events=minimum_purchase_events,
        maximum_missing_outcome_rate=maximum_missing_outcome_rate,
        maximum_allowed_p99_importance_weight=(
            maximum_allowed_p99_importance_weight
        ),
        weight_clip=weight_clip,
        minimum_raw_requests=minimum_raw_requests,
        logging_propensities=logging_propensities,
        logging_propensity_lower=logging_propensity_lower,
        logging_propensity_upper=logging_propensity_upper,
        outcome_observation_propensities=outcome_observation_propensities,
        full_estimand_support_threshold=full_estimand_support_threshold,
        partial_estimand_support_threshold=partial_estimand_support_threshold,
        _population_positivity_verified=False,
    )


def _audit_ope_readiness_with_contract(
    contract: OpeEvaluationContract,
    *,
    analysis_timestamp: str,
    _population_positivity_verified: bool,
    **row_evidence: Any,
) -> OpeReadinessAssessment:
    """Internal contract gate; only the manifest-bound path verifies positivity."""

    if not isinstance(contract, OpeEvaluationContract) or not contract.verify_digest():
        raise ValueError("contract must be an untampered OpeEvaluationContract")
    normalized = (
        analysis_timestamp[:-1] + "+00:00"
        if isinstance(analysis_timestamp, str) and analysis_timestamp.endswith("Z")
        else analysis_timestamp
    )
    try:
        analysis_time = datetime.fromisoformat(normalized)
    except (TypeError, ValueError) as error:
        raise ValueError("analysis_timestamp must be ISO-8601") from error
    if analysis_time.tzinfo is None or analysis_time.utcoffset() is None:
        raise ValueError("analysis_timestamp must include a timezone")
    earliest_value = (
        contract.earliest_analysis_time[:-1] + "+00:00"
        if contract.earliest_analysis_time.endswith("Z")
        else contract.earliest_analysis_time
    )
    earliest = datetime.fromisoformat(earliest_value)
    if analysis_time < earliest:
        raise ValueError("labels_pending: earliest analysis time has not been reached")

    controlled = {
        "propensity_mode",
        "minimum_raw_requests",
        "minimum_user_effective_sample_size",
        "minimum_purchase_events",
        "maximum_missing_outcome_rate",
        "maximum_allowed_p99_importance_weight",
        "weight_clip",
        "full_estimand_support_threshold",
        "partial_estimand_support_threshold",
        "population_positivity_established",
        "_evaluation_contract_digest",
        "_population_positivity_verified",
    }
    supplied = controlled & set(row_evidence)
    if supplied:
        raise ValueError(
            "contract-controlled readiness fields may not be overridden: "
            + ", ".join(sorted(supplied))
        )
    return _audit_ope_readiness(
        **row_evidence,
        population_positivity_established=_population_positivity_verified,
        propensity_mode=contract.propensity_mode,
        minimum_raw_requests=contract.minimum_raw_requests,
        minimum_user_effective_sample_size=(
            contract.minimum_user_effective_sample_size
        ),
        minimum_purchase_events=contract.minimum_purchase_events,
        maximum_missing_outcome_rate=contract.maximum_missing_outcome_rate,
        maximum_allowed_p99_importance_weight=(contract.maximum_p99_importance_weight),
        weight_clip=contract.weight_clip,
        full_estimand_support_threshold=(
            contract.minimum_support_rate_for_full_estimand
        ),
        partial_estimand_support_threshold=(
            contract.minimum_support_rate_for_partial_estimand
        ),
        _evaluation_contract_digest=contract.contract_digest,
        _population_positivity_verified=_population_positivity_verified,
    )


def audit_ope_readiness_with_contract(
    contract: OpeEvaluationContract,
    *,
    analysis_timestamp: str,
    **row_evidence: Any,
) -> OpeReadinessAssessment:
    """Run contract thresholds without accepting positivity as a bare flag."""

    return _audit_ope_readiness_with_contract(
        contract,
        analysis_timestamp=analysis_timestamp,
        _population_positivity_verified=False,
        **row_evidence,
    )


def audit_manifest_bound_ope_readiness(
    contract: OpeEvaluationContract,
    input_manifest: ImmutableInputManifest,
    observed_inputs: Sequence[ObservedInputIdentity],
    *,
    analysis_timestamp: str,
    population_positivity_evidence: (
        PopulationPositivityEvidence
        | StatisticalPopulationPositivityEvidence
        | None
    ) = None,
    **row_evidence: Any,
) -> ManifestBoundOpeReadinessAssessment:
    """Run readiness only after exact observation of every immutable input.

    Storage adapters must observe URI, version/generation, size, and SHA-256
    immediately before reading. Missing, extra, duplicate, or changed inputs
    return ``evidence_rejected`` without inspecting OPE row evidence.
    """

    if not isinstance(contract, OpeEvaluationContract) or not contract.verify_digest():
        raise ValueError("contract must be an untampered OpeEvaluationContract")
    if (
        not isinstance(input_manifest, ImmutableInputManifest)
        or not input_manifest.verify_digest()
    ):
        raise ValueError("input_manifest must be an untampered manifest")

    verification = audit_input_manifest_observations(input_manifest, observed_inputs)
    binding_reasons: list[str] = []
    if input_manifest.manifest_digest != contract.input_manifest_digest:
        binding_reasons.append("input_manifest_contract_digest_mismatch")
    if input_manifest.population_identity != contract.population_identity:
        binding_reasons.append("input_manifest_population_mismatch")
    observed_roles = {value.semantic_role for value in input_manifest.inputs}
    declared_roles = set(input_manifest.required_roles)
    required_ope_roles = set(OPE_REQUIRED_INPUT_ROLES)
    for role in sorted(required_ope_roles - observed_roles):
        binding_reasons.append(f"required_input_role_missing:{role}")
    for role in sorted(required_ope_roles - declared_roles):
        binding_reasons.append(f"required_manifest_role_not_declared:{role}")

    def parse(value: str, field: str) -> datetime:
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        try:
            parsed = datetime.fromisoformat(normalized)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{field} must be ISO-8601") from error
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError(f"{field} must include a timezone")
        return parsed

    analysis_time = parse(analysis_timestamp, "analysis_timestamp")
    final_end = parse(contract.final_window_end, "final_window_end")
    earliest = parse(contract.earliest_analysis_time, "earliest_analysis_time")
    feature_cutoff = parse(input_manifest.feature_cutoff, "feature_cutoff")
    label_cutoff = parse(input_manifest.label_cutoff, "label_cutoff")
    manifest_created = parse(input_manifest.created_at, "manifest_created_at")
    positivity_entries = tuple(
        value
        for value in input_manifest.inputs
        if value.semantic_role == "population_positivity_evidence"
    )
    positivity_entry = positivity_entries[0] if len(positivity_entries) == 1 else None
    positivity_types = (
        PopulationPositivityEvidence,
        StatisticalPopulationPositivityEvidence,
    )

    def positivity_metadata() -> tuple[
        str | None,
        str | None,
        float | None,
        str | None,
        str | None,
    ]:
        evidence = population_positivity_evidence
        if not isinstance(evidence, positivity_types) or not evidence.verify_digest():
            return None, None, None, None, None
        if isinstance(evidence, StatisticalPopulationPositivityEvidence):
            return (
                evidence.schema_version,
                evidence.guarantee_kind,
                evidence.simultaneous_failure_probability,
                evidence.validity_start,
                evidence.validity_end,
            )
        return evidence.schema_version, "legacy_design_evidence", None, None, None

    (
        positivity_schema,
        positivity_guarantee,
        positivity_failure,
        positivity_validity_start,
        positivity_validity_end,
    ) = positivity_metadata()
    ope_failure = 1.0 - contract.confidence_level
    if positivity_failure is None:
        joint_failure = None
        joint_confidence = None
        joint_composition = None
    else:
        joint_failure = min(1.0, ope_failure + positivity_failure)
        joint_confidence = max(0.0, 1.0 - joint_failure)
        joint_composition = "union_bound_no_independence_assumed"
    if len(positivity_entries) != 1:
        binding_reasons.append("population_positivity_evidence_role_not_unique")
    if "population_positivity_established" in row_evidence:
        binding_reasons.append("population_positivity_boolean_must_be_evidence_derived")
    if population_positivity_evidence is None:
        binding_reasons.append("population_positivity_evidence_missing")
    elif (
        not isinstance(population_positivity_evidence, positivity_types)
        or not population_positivity_evidence.verify_digest()
    ):
        binding_reasons.append("population_positivity_evidence_invalid_or_tampered")
    elif positivity_entry is not None:
        evidence_bytes = canonical_population_positivity_evidence_bytes(
            population_positivity_evidence
        )
        if positivity_entry.schema_version != population_positivity_evidence.schema_version:
            binding_reasons.append("population_positivity_schema_mismatch")
        if positivity_entry.version != population_positivity_evidence.evidence_id:
            binding_reasons.append("population_positivity_evidence_identity_mismatch")
        if positivity_entry.byte_size != len(evidence_bytes):
            binding_reasons.append("population_positivity_byte_size_mismatch")
        if positivity_entry.sha256 != sha256(evidence_bytes).hexdigest():
            binding_reasons.append("population_positivity_digest_mismatch")
        if positivity_entry.snapshot_time != population_positivity_evidence.audited_at:
            binding_reasons.append("population_positivity_snapshot_mismatch")
        if not population_positivity_evidence.established:
            binding_reasons.append("population_positivity_not_established")
        if (
            population_positivity_evidence.population_identity
            != contract.population_identity
        ):
            binding_reasons.append("population_positivity_population_mismatch")
        if (
            population_positivity_evidence.logging_policy_identity
            != contract.baseline_policy
        ):
            binding_reasons.append("population_positivity_logger_mismatch")
        if (
            population_positivity_evidence.target_policy_identity
            != contract.target_policy
        ):
            binding_reasons.append("population_positivity_target_mismatch")
        if parse(
            population_positivity_evidence.audited_at,
            "population_positivity_audited_at",
        ) > manifest_created:
            binding_reasons.append("population_positivity_audited_after_manifest")
        if isinstance(
            population_positivity_evidence,
            StatisticalPopulationPositivityEvidence,
        ):
            if parse(
                population_positivity_evidence.validity_start,
                "population_positivity_validity_start",
            ) > parse(contract.final_window_start, "final_window_start"):
                binding_reasons.append(
                    "population_positivity_validity_starts_after_final_window"
                )
            if parse(
                population_positivity_evidence.validity_end,
                "population_positivity_validity_end",
            ) < final_end:
                binding_reasons.append(
                    "population_positivity_validity_ends_before_final_window"
                )
    if feature_cutoff != final_end:
        binding_reasons.append("feature_cutoff_not_equal_to_final_window_end")
    if label_cutoff < final_end or label_cutoff > earliest:
        binding_reasons.append("label_cutoff_outside_settlement_boundary")
    if manifest_created > analysis_time:
        binding_reasons.append("input_manifest_created_after_analysis")

    if not verification.valid or binding_reasons:
        return ManifestBoundOpeReadinessAssessment(
            evaluation_contract_digest=contract.contract_digest,
            input_manifest_digest=input_manifest.manifest_digest,
            immutable_inputs_verified=False,
            population_positivity_evidence_input_id=(
                positivity_entry.input_id if positivity_entry else None
            ),
            population_positivity_evidence_id=(
                population_positivity_evidence.evidence_id
                if isinstance(population_positivity_evidence, positivity_types)
                else None
            ),
            population_positivity_evidence_sha256=(
                positivity_entry.sha256 if positivity_entry else None
            ),
            population_positivity_schema_version=positivity_schema,
            population_positivity_guarantee_kind=positivity_guarantee,
            population_positivity_simultaneous_failure_probability=(
                positivity_failure
            ),
            population_positivity_validity_start=positivity_validity_start,
            population_positivity_validity_end=positivity_validity_end,
            ope_failure_probability=ope_failure,
            joint_failure_probability_upper_bound=joint_failure,
            joint_confidence_level_lower_bound=joint_confidence,
            joint_confidence_composition=joint_composition,
            decision="evidence_rejected",
            input_verification=verification,
            readiness=None,
            production_rollout_authorized=False,
            reasons=tuple((*verification.reasons, *binding_reasons)),
        )
    if analysis_time < earliest:
        return ManifestBoundOpeReadinessAssessment(
            evaluation_contract_digest=contract.contract_digest,
            input_manifest_digest=input_manifest.manifest_digest,
            immutable_inputs_verified=True,
            population_positivity_evidence_input_id=(
                positivity_entry.input_id if positivity_entry else None
            ),
            population_positivity_evidence_id=(
                population_positivity_evidence.evidence_id
                if isinstance(population_positivity_evidence, positivity_types)
                else None
            ),
            population_positivity_evidence_sha256=(
                positivity_entry.sha256 if positivity_entry else None
            ),
            population_positivity_schema_version=positivity_schema,
            population_positivity_guarantee_kind=positivity_guarantee,
            population_positivity_simultaneous_failure_probability=(
                positivity_failure
            ),
            population_positivity_validity_start=positivity_validity_start,
            population_positivity_validity_end=positivity_validity_end,
            ope_failure_probability=ope_failure,
            joint_failure_probability_upper_bound=joint_failure,
            joint_confidence_level_lower_bound=joint_confidence,
            joint_confidence_composition=joint_composition,
            decision="labels_pending",
            input_verification=verification,
            readiness=None,
            production_rollout_authorized=False,
            reasons=("earliest_analysis_time_not_reached",),
        )

    readiness = _audit_ope_readiness_with_contract(
        contract,
        analysis_timestamp=analysis_timestamp,
        _population_positivity_verified=True,
        **row_evidence,
    )
    readiness = replace(
        readiness,
        positivity_failure_probability=positivity_failure,
        ope_failure_probability=ope_failure,
        joint_failure_probability_upper_bound=joint_failure,
        joint_confidence_level_lower_bound=joint_confidence,
        joint_confidence_composition=joint_composition,
    )
    return ManifestBoundOpeReadinessAssessment(
        evaluation_contract_digest=contract.contract_digest,
        input_manifest_digest=input_manifest.manifest_digest,
        immutable_inputs_verified=True,
        population_positivity_evidence_input_id=(
            positivity_entry.input_id if positivity_entry else None
        ),
        population_positivity_evidence_id=(
            population_positivity_evidence.evidence_id
            if isinstance(population_positivity_evidence, positivity_types)
            else None
        ),
        population_positivity_evidence_sha256=(
            positivity_entry.sha256 if positivity_entry else None
        ),
        population_positivity_schema_version=positivity_schema,
        population_positivity_guarantee_kind=positivity_guarantee,
        population_positivity_simultaneous_failure_probability=(
            positivity_failure
        ),
        population_positivity_validity_start=positivity_validity_start,
        population_positivity_validity_end=positivity_validity_end,
        ope_failure_probability=ope_failure,
        joint_failure_probability_upper_bound=joint_failure,
        joint_confidence_level_lower_bound=joint_confidence,
        joint_confidence_composition=joint_composition,
        decision=readiness.decision,
        input_verification=verification,
        readiness=readiness,
        production_rollout_authorized=False,
        reasons=readiness.reasons,
    )
