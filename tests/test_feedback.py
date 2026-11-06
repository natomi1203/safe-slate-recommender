from __future__ import annotations

from dataclasses import replace
import math

import pytest

from safe_slate import assess_feedback_coverage, guarded_rerank


def _proposal_log(*, serving_mode: str = "serve", support_ok: bool = True):
    return guarded_rerank(
        request_id="feedback-request",
        event_timestamp="2026-10-01T12:00:00+09:00",
        baseline_item_ids=[10, 11, 12, 13],
        protected_count=2,
        remove_positions=[2, 2, 3],
        add_item_ids=[100, 102, 101],
        reward_deltas=[0.3, 0.2, 0.1],
        safety_mean_deltas=[0.04, 0.02, 0.03],
        safety_variances=[0.0, 0.0, 0.0],
        baseline_metric=0.5,
        degradation_tolerance=0.02,
        max_replacements=2,
        timeout_ms=10,
        serving_mode=serving_mode,
        support_ok=support_ok,
        uncertainty_valid=True,
        distribution_shift_detected=False,
        proposal_log_complete=True,
    ).proposal_log


def test_complete_feedback_passes_direct_and_ope_prechecks() -> None:
    assessment = assess_feedback_coverage(
        _proposal_log(),
        direct_observed_operation_indices=[0, 1, 2],
        propensities=[0.5, 0.25, 0.1],
        labels_settled=True,
        missingness_independent=True,
    )

    assert assessment.direct_update_eligible
    assert assessment.ope_precheck_eligible
    assert not assessment.descriptive_only
    assert not assessment.accepted_only_feedback
    assert assessment.direct_feedback_coverage_rate == 1.0
    assert assessment.propensity_support_rate == 1.0
    assert assessment.maximum_inverse_propensity_weight == pytest.approx(10.0)
    assert assessment.reasons == ()


def test_accepted_only_feedback_is_detected_and_fails_closed() -> None:
    log = _proposal_log()
    assessment = assess_feedback_coverage(
        log,
        direct_observed_operation_indices=log.selected_operation_indices,
        propensities=None,
        labels_settled=True,
        missingness_independent=True,
    )

    assert assessment.accepted_only_feedback
    assert not assessment.direct_update_eligible
    assert not assessment.ope_precheck_eligible
    assert assessment.descriptive_only
    assert "accepted_only_feedback" in assessment.reasons
    assert "direct_feedback_incomplete" in assessment.reasons
    assert "propensity_missing" in assessment.reasons


def test_propensity_support_can_pass_when_direct_feedback_is_partial() -> None:
    assessment = assess_feedback_coverage(
        _proposal_log(),
        direct_observed_operation_indices=[0],
        propensities=[0.3, 0.2, 0.1],
        labels_settled=True,
        missingness_independent=True,
    )

    assert not assessment.direct_update_eligible
    assert assessment.ope_precheck_eligible
    assert not assessment.descriptive_only
    assert "direct_feedback_incomplete" in assessment.reasons


def test_low_support_and_heavy_weight_tail_fail_ope_precheck() -> None:
    low_support = assess_feedback_coverage(
        _proposal_log(),
        direct_observed_operation_indices=[],
        propensities=[0.5, 0.0, 0.0],
        labels_settled=True,
        missingness_independent=True,
    )
    heavy_tail = assess_feedback_coverage(
        _proposal_log(),
        direct_observed_operation_indices=[],
        propensities=[0.5, 0.2, 0.001],
        labels_settled=True,
        missingness_independent=True,
    )

    assert not low_support.ope_precheck_eligible
    assert "propensity_support_below_threshold" in low_support.reasons
    assert not heavy_tail.ope_precheck_eligible
    assert "propensity_weight_tail" in heavy_tail.reasons


def test_shadow_outcomes_never_authorize_candidate_feedback() -> None:
    assessment = assess_feedback_coverage(
        _proposal_log(serving_mode="shadow"),
        direct_observed_operation_indices=[0, 1, 2],
        propensities=[0.5, 0.5, 0.5],
        labels_settled=True,
        missingness_independent=True,
    )

    assert not assessment.direct_update_eligible
    assert not assessment.ope_precheck_eligible
    assert assessment.descriptive_only
    assert "shadow_outcomes_not_candidate_feedback" in assessment.reasons


def test_failed_serving_gate_never_authorizes_learning_or_ope() -> None:
    assessment = assess_feedback_coverage(
        _proposal_log(support_ok=False),
        direct_observed_operation_indices=[0, 1, 2],
        propensities=[0.5, 0.5, 0.5],
        labels_settled=True,
        missingness_independent=True,
    )

    assert not assessment.direct_update_eligible
    assert not assessment.ope_precheck_eligible
    assert assessment.descriptive_only
    assert "serving_gate_not_learning_eligible" in assessment.reasons


@pytest.mark.parametrize(
    ("labels_settled", "missingness_independent", "reason"),
    [
        (False, True, "labels_unsettled"),
        (None, True, "settlement_unknown"),
        (True, False, "missingness_not_independent"),
        (True, None, "missingness_unknown"),
    ],
)
def test_settlement_and_missingness_gates_fail_closed(
    labels_settled: bool | None,
    missingness_independent: bool | None,
    reason: str,
) -> None:
    assessment = assess_feedback_coverage(
        _proposal_log(),
        direct_observed_operation_indices=[0, 1, 2],
        propensities=[0.5, 0.5, 0.5],
        labels_settled=labels_settled,
        missingness_independent=missingness_independent,
    )

    assert not assessment.direct_update_eligible
    assert not assessment.ope_precheck_eligible
    assert reason in assessment.reasons


@pytest.mark.parametrize(
    ("direct_indices", "propensities", "reason"),
    [
        (None, [0.5, 0.5, 0.5], "direct_feedback_missing"),
        ([0, 0], [0.5, 0.5, 0.5], "direct_feedback_index_duplicate"),
        ([3], [0.5, 0.5, 0.5], "direct_feedback_index_invalid"),
        ([], [0.5], "propensity_length_mismatch"),
        ([], [0.5, math.nan, 0.5], "propensity_invalid"),
        ([], [0.5, "invalid", 0.5], "propensity_invalid"),
    ],
)
def test_malformed_runtime_evidence_fails_closed(
    direct_indices: list[int] | None,
    propensities: list[object],
    reason: str,
) -> None:
    assessment = assess_feedback_coverage(
        _proposal_log(),
        direct_observed_operation_indices=direct_indices,
        propensities=propensities,  # type: ignore[arg-type]
        labels_settled=True,
        missingness_independent=True,
    )

    assert reason in assessment.reasons
    if reason.startswith("direct_feedback"):
        assert not assessment.direct_update_eligible
    if reason.startswith("propensity"):
        assert not assessment.ope_precheck_eligible


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("minimum_ope_support_rate", -0.1),
        ("minimum_ope_support_rate", 1.1),
        ("minimum_ope_support_rate", math.nan),
        ("maximum_allowed_inverse_propensity_weight", 0.0),
        ("maximum_allowed_inverse_propensity_weight", math.inf),
    ],
)
def test_invalid_policy_thresholds_raise(field: str, value: float) -> None:
    kwargs = {
        "direct_observed_operation_indices": [0, 1, 2],
        "propensities": [0.5, 0.5, 0.5],
        "labels_settled": True,
        "missingness_independent": True,
        field: value,
    }
    with pytest.raises(ValueError):
        assess_feedback_coverage(_proposal_log(), **kwargs)


def test_feedback_assessment_serializes_to_json_ready_dict() -> None:
    assessment = assess_feedback_coverage(
        _proposal_log(),
        direct_observed_operation_indices=[0, 1, 2],
        propensities=[0.5, 0.25, 0.1],
        labels_settled=True,
        missingness_independent=True,
    )

    serialized = assessment.to_dict()
    assert serialized["request_id"] == "feedback-request"
    assert serialized["direct_observed_operation_indices"] == [0, 1, 2]
    assert serialized["maximum_inverse_propensity_weight"] == pytest.approx(10.0)
    assert serialized["reasons"] == []


def test_feedback_rejects_invalid_log_type_and_incomplete_primary_log() -> None:
    with pytest.raises(ValueError, match="ProposalLog"):
        assess_feedback_coverage(
            "bad",  # type: ignore[arg-type]
            direct_observed_operation_indices=[],
            propensities=[],
            labels_settled=True,
            missingness_independent=True,
        )

    assessment = assess_feedback_coverage(
        replace(_proposal_log(), proposal_log_complete=False),
        direct_observed_operation_indices=[0, 1, 2],
        propensities=[0.5, 0.5, 0.5],
        labels_settled=True,
        missingness_independent=True,
    )
    assert "proposal_log_incomplete" in assessment.reasons
