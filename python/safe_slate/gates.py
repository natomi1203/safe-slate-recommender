"""Fail-closed serving-gate assessment from runtime evidence."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any


@dataclass(frozen=True, slots=True)
class ServingGateAssessment:
    """Auditable inputs and decisions for guarded serving."""

    support_rate: float | None
    minimum_support_rate: float
    distribution_shift_score: float | None
    maximum_distribution_shift_score: float
    uncertainty_evidence_valid: bool | None
    expected_proposal_count: int | None
    logged_proposal_count: int | None
    support_ok: bool
    uncertainty_valid: bool
    distribution_shift_detected: bool
    proposal_log_complete: bool
    reasons: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return (
            self.support_ok
            and self.uncertainty_valid
            and not self.distribution_shift_detected
            and self.proposal_log_complete
        )

    def rerank_kwargs(self) -> dict[str, bool]:
        """Return the four inputs consumed by ``guarded_rerank``."""

        return {
            "support_ok": self.support_ok,
            "uncertainty_valid": self.uncertainty_valid,
            "distribution_shift_detected": self.distribution_shift_detected,
            "proposal_log_complete": self.proposal_log_complete,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "support_rate": self.support_rate,
            "minimum_support_rate": self.minimum_support_rate,
            "distribution_shift_score": self.distribution_shift_score,
            "maximum_distribution_shift_score": self.maximum_distribution_shift_score,
            "uncertainty_evidence_valid": self.uncertainty_evidence_valid,
            "expected_proposal_count": self.expected_proposal_count,
            "logged_proposal_count": self.logged_proposal_count,
            "support_ok": self.support_ok,
            "uncertainty_valid": self.uncertainty_valid,
            "distribution_shift_detected": self.distribution_shift_detected,
            "proposal_log_complete": self.proposal_log_complete,
            "passed": self.passed,
            "reasons": list(self.reasons),
        }


def assess_serving_gates(
    *,
    support_rate: float | None,
    distribution_shift_score: float | None,
    uncertainty_evidence_valid: bool | None,
    expected_proposal_count: int | None,
    logged_proposal_count: int | None,
    minimum_support_rate: float = 0.99,
    maximum_distribution_shift_score: float = 0.10,
) -> ServingGateAssessment:
    """Turn runtime evidence into fail-closed serving decisions.

    Missing, non-finite, or out-of-range runtime evidence fails its gate. The
    two configured thresholds are deployment contracts and therefore raise on
    invalid values instead of silently changing policy.
    """

    if (
        not isfinite(minimum_support_rate)
        or not 0.0 <= minimum_support_rate <= 1.0
    ):
        raise ValueError("minimum_support_rate must be finite and in [0, 1]")
    if (
        not isfinite(maximum_distribution_shift_score)
        or maximum_distribution_shift_score < 0.0
    ):
        raise ValueError(
            "maximum_distribution_shift_score must be finite and nonnegative"
        )

    reasons: list[str] = []
    if support_rate is None:
        support_ok = False
        reasons.append("support_missing")
    elif not isfinite(support_rate) or not 0.0 <= support_rate <= 1.0:
        support_ok = False
        reasons.append("support_invalid")
    elif support_rate < minimum_support_rate:
        support_ok = False
        reasons.append("support_below_threshold")
    else:
        support_ok = True

    if uncertainty_evidence_valid is True:
        uncertainty_valid = True
    else:
        uncertainty_valid = False
        reasons.append(
            "uncertainty_missing"
            if uncertainty_evidence_valid is None
            else "uncertainty_invalid"
        )

    if distribution_shift_score is None:
        distribution_shift_detected = True
        reasons.append("shift_score_missing")
    elif not isfinite(distribution_shift_score) or distribution_shift_score < 0.0:
        distribution_shift_detected = True
        reasons.append("shift_score_invalid")
    elif distribution_shift_score > maximum_distribution_shift_score:
        distribution_shift_detected = True
        reasons.append("distribution_shift")
    else:
        distribution_shift_detected = False

    counts_valid = (
        isinstance(expected_proposal_count, int)
        and not isinstance(expected_proposal_count, bool)
        and expected_proposal_count >= 0
        and isinstance(logged_proposal_count, int)
        and not isinstance(logged_proposal_count, bool)
        and logged_proposal_count >= 0
    )
    if not counts_valid:
        proposal_log_complete = False
        reasons.append("proposal_count_invalid")
    elif logged_proposal_count != expected_proposal_count:
        proposal_log_complete = False
        reasons.append("proposal_log_incomplete")
    else:
        proposal_log_complete = True

    return ServingGateAssessment(
        support_rate=support_rate,
        minimum_support_rate=minimum_support_rate,
        distribution_shift_score=distribution_shift_score,
        maximum_distribution_shift_score=maximum_distribution_shift_score,
        uncertainty_evidence_valid=uncertainty_evidence_valid,
        expected_proposal_count=expected_proposal_count,
        logged_proposal_count=logged_proposal_count,
        support_ok=support_ok,
        uncertainty_valid=uncertainty_valid,
        distribution_shift_detected=distribution_shift_detected,
        proposal_log_complete=proposal_log_complete,
        reasons=tuple(reasons),
    )
