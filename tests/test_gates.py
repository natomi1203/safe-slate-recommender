from __future__ import annotations

import math

import pytest

from safe_slate import assess_serving_gates, guarded_rerank


def test_complete_runtime_evidence_passes_all_gates() -> None:
    assessment = assess_serving_gates(
        support_rate=0.995,
        distribution_shift_score=0.03,
        uncertainty_evidence_valid=True,
        expected_proposal_count=200,
        logged_proposal_count=200,
    )

    assert assessment.passed
    assert assessment.reasons == ()
    assert assessment.rerank_kwargs() == {
        "support_ok": True,
        "uncertainty_valid": True,
        "distribution_shift_detected": False,
        "proposal_log_complete": True,
    }


def test_missing_and_invalid_runtime_evidence_fails_closed() -> None:
    assessment = assess_serving_gates(
        support_rate=math.nan,
        distribution_shift_score=None,
        uncertainty_evidence_valid=None,
        expected_proposal_count=200,
        logged_proposal_count=199,
    )

    assert not assessment.passed
    assert not assessment.support_ok
    assert not assessment.uncertainty_valid
    assert assessment.distribution_shift_detected
    assert not assessment.proposal_log_complete
    assert set(assessment.reasons) == {
        "support_invalid",
        "uncertainty_missing",
        "shift_score_missing",
        "proposal_log_incomplete",
    }


def test_invalid_gate_contract_thresholds_are_rejected() -> None:
    with pytest.raises(ValueError, match="minimum_support_rate"):
        assess_serving_gates(
            support_rate=1.0,
            distribution_shift_score=0.0,
            uncertainty_evidence_valid=True,
            expected_proposal_count=1,
            logged_proposal_count=1,
            minimum_support_rate=1.1,
        )


def test_gate_assessment_controls_rerank_and_is_logged() -> None:
    assessment = assess_serving_gates(
        support_rate=0.70,
        distribution_shift_score=0.02,
        uncertainty_evidence_valid=True,
        expected_proposal_count=1,
        logged_proposal_count=1,
    )
    result = guarded_rerank(
        request_id="gated",
        baseline_item_ids=[10, 11, 12],
        protected_count=2,
        remove_positions=[2],
        add_item_ids=[100],
        reward_deltas=[1.0],
        safety_mean_deltas=[0.1],
        safety_variances=[0.0],
        baseline_metric=0.2,
        degradation_tolerance=0.01,
        max_replacements=1,
        gate_assessment=assessment,
        # The structured assessment takes precedence over raw defaults.
        support_ok=True,
    )

    assert result.fallback_reason == "support_gate"
    assert result.slate == (10, 11, 12)
    assert result.proposal_log.schema_version == "1.3"
    assert result.proposal_log.gate_evidence is not None
    assert result.proposal_log.gate_evidence["support_rate"] == pytest.approx(0.70)
    assert result.proposal_log.gate_evidence["reasons"] == ["support_below_threshold"]
