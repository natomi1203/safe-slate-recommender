"""Feedback-coverage audits that prevent accepted-action-only lockout."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Any, Sequence

from .production import ProposalLog


@dataclass(frozen=True, slots=True)
class FeedbackCoverageAssessment:
    """Post-serving evidence gates for learning and OPE prechecks."""

    request_id: str
    serving_mode: str
    proposal_count: int
    selected_operation_indices: tuple[int, ...]
    direct_observed_operation_indices: tuple[int, ...]
    labels_settled: bool | None
    missingness_independent: bool | None
    direct_feedback_coverage_rate: float
    propensity_support_rate: float
    minimum_ope_support_rate: float
    maximum_inverse_propensity_weight: float | None
    maximum_allowed_inverse_propensity_weight: float
    accepted_only_feedback: bool
    direct_update_eligible: bool
    ope_precheck_eligible: bool
    descriptive_only: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "serving_mode": self.serving_mode,
            "proposal_count": self.proposal_count,
            "selected_operation_indices": list(self.selected_operation_indices),
            "direct_observed_operation_indices": list(
                self.direct_observed_operation_indices
            ),
            "labels_settled": self.labels_settled,
            "missingness_independent": self.missingness_independent,
            "direct_feedback_coverage_rate": self.direct_feedback_coverage_rate,
            "propensity_support_rate": self.propensity_support_rate,
            "minimum_ope_support_rate": self.minimum_ope_support_rate,
            "maximum_inverse_propensity_weight": (
                self.maximum_inverse_propensity_weight
            ),
            "maximum_allowed_inverse_propensity_weight": (
                self.maximum_allowed_inverse_propensity_weight
            ),
            "accepted_only_feedback": self.accepted_only_feedback,
            "direct_update_eligible": self.direct_update_eligible,
            "ope_precheck_eligible": self.ope_precheck_eligible,
            "descriptive_only": self.descriptive_only,
            "reasons": list(self.reasons),
        }


def assess_feedback_coverage(
    proposal_log: ProposalLog,
    *,
    direct_observed_operation_indices: Sequence[int] | None,
    propensities: Sequence[float | None] | None,
    labels_settled: bool | None,
    missingness_independent: bool | None,
    minimum_ope_support_rate: float = 0.99,
    maximum_allowed_inverse_propensity_weight: float = 100.0,
) -> FeedbackCoverageAssessment:
    """Audit one request before using its feedback for learning or OPE.

    ``direct_update_eligible`` requires settled, outcome-independent feedback
    for every proposed operation. ``ope_precheck_eligible`` instead requires
    positive propensity support at the configured rate and a bounded inverse
    propensity tail. It is only a request-level precheck; user-level ESS,
    repeated-user dependence, and final estimand identification remain external
    evaluation requirements.
    """

    if not isinstance(proposal_log, ProposalLog):
        raise ValueError("proposal_log must be a ProposalLog")
    if (
        not isinstance(minimum_ope_support_rate, Real)
        or isinstance(minimum_ope_support_rate, bool)
        or not isfinite(minimum_ope_support_rate)
        or not 0.0 <= minimum_ope_support_rate <= 1.0
    ):
        raise ValueError("minimum_ope_support_rate must be finite and in [0, 1]")
    if (
        not isinstance(maximum_allowed_inverse_propensity_weight, Real)
        or isinstance(maximum_allowed_inverse_propensity_weight, bool)
        or not isfinite(maximum_allowed_inverse_propensity_weight)
        or maximum_allowed_inverse_propensity_weight <= 0.0
    ):
        raise ValueError(
            "maximum_allowed_inverse_propensity_weight must be finite and positive"
        )

    proposal_count = len(proposal_log.add_item_ids)
    reasons: list[str] = []
    if direct_observed_operation_indices is None:
        observed_raw: tuple[int, ...] = ()
        reasons.append("direct_feedback_missing")
    else:
        observed_raw = tuple(direct_observed_operation_indices)
    observed_valid = all(
        isinstance(index, int)
        and not isinstance(index, bool)
        and 0 <= index < proposal_count
        for index in observed_raw
    )
    observed_unique = len(set(observed_raw)) == len(observed_raw)
    if not observed_valid:
        reasons.append("direct_feedback_index_invalid")
    if not observed_unique:
        reasons.append("direct_feedback_index_duplicate")
    observed = tuple(sorted(set(observed_raw))) if observed_valid else ()
    direct_feedback_coverage_rate = (
        len(observed) / proposal_count if proposal_count > 0 else 1.0
    )

    selected = tuple(proposal_log.selected_operation_indices)
    selected_set = set(selected)
    observed_set = set(observed)
    accepted_only_feedback = (
        proposal_count > len(selected_set)
        and bool(selected_set)
        and observed_set == selected_set
    )
    if accepted_only_feedback:
        reasons.append("accepted_only_feedback")
    if direct_feedback_coverage_rate < 1.0:
        reasons.append("direct_feedback_incomplete")

    propensity_valid = True
    positive_propensities: list[float] = []
    if propensities is None:
        propensity_valid = False
        propensity_support_rate = 0.0 if proposal_count > 0 else 1.0
        maximum_inverse_propensity_weight = None
        reasons.append("propensity_missing")
    elif len(propensities) != proposal_count:
        propensity_valid = False
        propensity_support_rate = 0.0 if proposal_count > 0 else 1.0
        maximum_inverse_propensity_weight = None
        reasons.append("propensity_length_mismatch")
    else:
        for propensity in propensities:
            if (
                propensity is None
                or not isinstance(propensity, Real)
                or isinstance(propensity, bool)
                or not isfinite(propensity)
                or not 0.0 <= propensity <= 1.0
            ):
                propensity_valid = False
                break
            if propensity > 0.0:
                positive_propensities.append(float(propensity))
        if not propensity_valid:
            propensity_support_rate = 0.0 if proposal_count > 0 else 1.0
            maximum_inverse_propensity_weight = None
            reasons.append("propensity_invalid")
        else:
            propensity_support_rate = (
                len(positive_propensities) / proposal_count
                if proposal_count > 0
                else 1.0
            )
            maximum_inverse_propensity_weight = (
                1.0 / min(positive_propensities)
                if positive_propensities
                else None
            )
            if propensity_support_rate < minimum_ope_support_rate:
                reasons.append("propensity_support_below_threshold")
            if (
                maximum_inverse_propensity_weight is not None
                and maximum_inverse_propensity_weight
                > maximum_allowed_inverse_propensity_weight
            ):
                reasons.append("propensity_weight_tail")

    if labels_settled is not True:
        reasons.append(
            "labels_unsettled" if labels_settled is False else "settlement_unknown"
        )
    if missingness_independent is not True:
        reasons.append(
            "missingness_not_independent"
            if missingness_independent is False
            else "missingness_unknown"
        )
    if not proposal_log.proposal_log_complete:
        reasons.append("proposal_log_incomplete")
    if not proposal_log.learning_update_eligible:
        reasons.append("serving_gate_not_learning_eligible")
    if proposal_log.serving_mode == "shadow":
        reasons.append("shadow_outcomes_not_candidate_feedback")

    common_ready = (
        labels_settled is True
        and missingness_independent is True
        and proposal_log.proposal_log_complete
        and proposal_log.learning_update_eligible
        and proposal_log.serving_mode == "serve"
    )
    direct_update_eligible = (
        common_ready
        and observed_valid
        and observed_unique
        and direct_feedback_coverage_rate == 1.0
    )
    weight_tail_ok = (
        maximum_inverse_propensity_weight is not None
        and maximum_inverse_propensity_weight
        <= maximum_allowed_inverse_propensity_weight
    ) or proposal_count == 0
    ope_precheck_eligible = (
        common_ready
        and propensity_valid
        and propensity_support_rate >= minimum_ope_support_rate
        and weight_tail_ok
    )
    descriptive_only = not direct_update_eligible and not ope_precheck_eligible

    return FeedbackCoverageAssessment(
        request_id=proposal_log.request_id,
        serving_mode=proposal_log.serving_mode,
        proposal_count=proposal_count,
        selected_operation_indices=selected,
        direct_observed_operation_indices=observed,
        labels_settled=labels_settled,
        missingness_independent=missingness_independent,
        direct_feedback_coverage_rate=direct_feedback_coverage_rate,
        propensity_support_rate=propensity_support_rate,
        minimum_ope_support_rate=minimum_ope_support_rate,
        maximum_inverse_propensity_weight=maximum_inverse_propensity_weight,
        maximum_allowed_inverse_propensity_weight=(
            maximum_allowed_inverse_propensity_weight
        ),
        accepted_only_feedback=accepted_only_feedback,
        direct_update_eligible=direct_update_eligible,
        ope_precheck_eligible=ope_precheck_eligible,
        descriptive_only=descriptive_only,
        reasons=tuple(dict.fromkeys(reasons)),
    )
