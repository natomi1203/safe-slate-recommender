from __future__ import annotations

import pytest

from safe_slate import (
    build_candidate_pool_operations,
    guarded_candidate_pool_rerank,
)


def _request(**overrides: object) -> dict[str, object]:
    request: dict[str, object] = {
        "request_id": "feed-request-1",
        "uncertainty_snapshot_id": "ranker-v21/covariance/sha256:abc",
        "contrast_input_id": "feed-request-1/contrasts",
        "model_version": "ranker-v21",
        "baseline_item_ids": [10, 11, 12, 13],
        "baseline_reward_scores": [1.0, 0.9, 0.2, 0.1],
        "baseline_safety_means": [0.10, 0.09, 0.05, 0.05],
        "baseline_features": [[0.0, 0.0]] * 4,
        "candidate_item_ids": [100, 101],
        "candidate_reward_scores": [0.8, 0.7],
        "candidate_safety_means": [0.06, 0.06],
        "candidate_features": [[1.0, 0.0], [0.0, 1.0]],
        "parameter_covariance": [[0.0, 0.0], [0.0, 0.0]],
        "protected_count": 2,
        "baseline_metric": 0.5,
        "degradation_tolerance": 0.1,
        "max_replacements": 2,
        "support_ok": True,
        "uncertainty_valid": True,
        "distribution_shift_detected": False,
        "proposal_log_complete": True,
        "timeout_ms": 100,
    }
    request.update(overrides)
    return request


def test_operation_builder_forms_deterministic_add_minus_remove_contrasts() -> None:
    operations = build_candidate_pool_operations(
        baseline_item_ids=[10, 11, 12],
        baseline_reward_scores=[0.9, 0.5, 0.2],
        baseline_safety_means=[0.2, 0.1, 0.05],
        baseline_features=[[9.0, 9.0], [1.0, 2.0], [3.0, 4.0]],
        candidate_item_ids=[100, 101],
        candidate_reward_scores=[0.8, 0.6],
        candidate_safety_means=[0.15, 0.10],
        candidate_features=[[5.0, 7.0], [2.0, 6.0]],
        parameter_covariance=[[1.0, 0.0], [0.0, 1.0]],
        protected_count=1,
    )

    assert operations.remove_positions == (1, 1, 2, 2)
    assert operations.add_item_ids == (100, 101, 100, 101)
    assert operations.reward_deltas == pytest.approx((0.3, 0.1, 0.6, 0.4))
    assert operations.safety_mean_deltas == pytest.approx((0.05, 0.0, 0.1, 0.05))
    assert operations.contrast_features == (
        (4.0, 5.0),
        (1.0, 4.0),
        (2.0, 3.0),
        (-1.0, 2.0),
    )


def test_candidate_pool_facade_preserves_core_and_selects_maximum_safe_count() -> None:
    result = guarded_candidate_pool_rerank(**_request())

    assert result.slate[:2] == (10, 11)
    assert set(result.slate[2:]) == {100, 101}
    assert result.selected_count == 2
    assert result.k_lower == result.k_upper == 2
    assert result.oracle.rechecked
    assert len(result.proposal_log.remove_positions) == 4
    metadata = result.proposal_log.context["safe_slate_candidate_pool"]
    uncertainty = result.proposal_log.context["safe_slate_uncertainty"]
    assert metadata["baseline_count"] == 4
    assert metadata["replaceable_position_count"] == 2
    assert metadata["candidate_count"] == 2
    assert metadata["operation_count"] == 4
    assert metadata["requested_total_budget_ms"] == 100
    assert metadata["linear_guardrail_budget_ms"] <= 100
    assert metadata["matrix_digest_encoding"] == "f64le-v1"
    assert uncertainty["matrix_digest_encoding"] == "f64le-v1"


def test_shared_uncertainty_shrinks_request_specific_replacement_count() -> None:
    result = guarded_candidate_pool_rerank(
        **_request(
            baseline_safety_means=[0.10, 0.09, 0.05, 0.05],
            candidate_safety_means=[0.05, 0.05],
            baseline_features=[[0.0]] * 4,
            candidate_features=[[1.0], [1.0]],
            parameter_covariance=[[0.0004]],
        )
    )

    assert result.selected_count == 1
    assert result.k_lower == result.k_upper == 1
    assert result.proposal_log.selected_safety_lcb == pytest.approx(-0.0392)
    assert result.proposal_log.safety_threshold == pytest.approx(-0.05)


def test_default_runtime_evidence_fails_closed() -> None:
    result = guarded_candidate_pool_rerank(
        **_request(
            support_ok=False,
            uncertainty_valid=False,
            distribution_shift_detected=True,
            proposal_log_complete=False,
        )
    )

    assert result.slate == (10, 11, 12, 13)
    assert result.selected_count == 0
    assert result.fallback_reason == "support_gate"
    assert result.proposal_log.effective_max_replacements == 0


def test_default_max_replacements_uses_the_replaceable_suffix() -> None:
    result = guarded_candidate_pool_rerank(**_request(max_replacements=None))
    assert result.proposal_log.requested_max_replacements == 2


def test_candidate_conflicts_compile_to_native_operation_conflicts() -> None:
    operations = build_candidate_pool_operations(
        baseline_item_ids=[10, 11, 12, 13],
        baseline_reward_scores=[1.0, 0.9, 0.2, 0.1],
        baseline_safety_means=[0.1, 0.1, 0.1, 0.1],
        baseline_features=[[0.0]] * 4,
        candidate_item_ids=[100, 101],
        candidate_reward_scores=[0.8, 0.7],
        candidate_safety_means=[0.2, 0.2],
        candidate_features=[[0.0], [0.0]],
        parameter_covariance=[[0.0]],
        protected_count=2,
        candidate_conflicts=[(100, 101)],
    )

    assert operations.operation_conflicts == ((0, 1), (0, 3), (1, 2), (2, 3))

    result = guarded_candidate_pool_rerank(
        **_request(candidate_conflicts=[(100, 101)])
    )
    assert result.selected_count == 1


def test_shadow_candidate_pool_never_changes_served_slate() -> None:
    result = guarded_candidate_pool_rerank(
        **_request(
            serving_mode="shadow",
            event_timestamp="2026-11-03T12:00:00+09:00",
        )
    )

    assert result.slate == (10, 11, 12, 13)
    assert result.proposed_slate != result.slate
    assert result.shadow
    assert not result.proposal_log.learning_update_eligible


def test_candidate_expansion_and_invalid_overlap_fail_closed_before_search() -> None:
    with pytest.raises(ValueError, match="max_operations"):
        guarded_candidate_pool_rerank(**_request(max_operations=3))

    with pytest.raises(ValueError, match="must not already occur"):
        guarded_candidate_pool_rerank(
            **_request(candidate_item_ids=[12, 101])
        )

    with pytest.raises(ValueError, match="max_feature_cells"):
        guarded_candidate_pool_rerank(**_request(max_feature_cells=7))


def test_candidate_groups_are_expanded_per_operation() -> None:
    operations = build_candidate_pool_operations(
        baseline_item_ids=[10, 11, 12],
        baseline_reward_scores=[0.9, 0.5, 0.2],
        baseline_safety_means=[0.2, 0.1, 0.05],
        baseline_features=[[0.0], [0.0], [0.0]],
        candidate_item_ids=[100, 101],
        candidate_reward_scores=[0.8, 0.6],
        candidate_safety_means=[0.15, 0.10],
        candidate_features=[[0.0], [0.0]],
        parameter_covariance=[[0.0]],
        protected_count=1,
        baseline_group_ids=[9, 1, 2],
        remove_group_caps={1: 1, 2: 1},
        candidate_group_ids=[7, 8],
        add_group_caps={7: 1, 8: 1},
    )

    assert operations.remove_group_ids == (1, 1, 2, 2)
    assert operations.remove_group_caps == ((1, 1), (2, 1))
    assert operations.add_group_ids == (7, 8, 7, 8)
    assert operations.add_group_caps == ((7, 1), (8, 1))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"baseline_item_ids": []}, "must be nonempty"),
        ({"baseline_item_ids": [10, 10]}, "must be unique"),
        ({"candidate_item_ids": [100, -1]}, "nonnegative integers"),
        ({"protected_count": True}, "protected_count"),
        ({"max_operations": True}, "max_operations"),
        ({"max_feature_cells": -1}, "max_feature_cells"),
        ({"max_compiled_conflicts": -1}, "max_compiled_conflicts"),
        ({"parameter_covariance": []}, "must be nonempty"),
        ({"parameter_covariance": [[0.0], [0.0]]}, "row 0"),
        ({"baseline_features": [[0.0]]}, "baseline_features has 1 rows"),
        (
            {"candidate_features": [[float("inf"), 0.0], [0.0, 1.0]]},
            "finite real",
        ),
        ({"baseline_reward_scores": [1.0]}, "baseline_reward_scores has length"),
        ({"candidate_safety_means": [float("nan"), 0.1]}, "finite real"),
        ({"candidate_features": [[0.0], [0.0]]}, "candidate_features row"),
        ({"baseline_group_ids": [1, 2, 3, 4]}, "supplied together"),
        (
            {"baseline_group_ids": [1], "remove_group_caps": {1: 1}},
            "has length",
        ),
        (
            {
                "baseline_group_ids": [1, 1, 2, 2],
                "remove_group_caps": {1: 1},
            },
            "missing group ids",
        ),
        (
            {
                "candidate_group_ids": [7, 8],
                "add_group_caps": {7: -1, 8: 1},
            },
            "nonnegative integers",
        ),
        (
            {
                "candidate_group_ids": [True, 8],
                "add_group_caps": {7: 1, 8: 1},
            },
            "must contain integers",
        ),
        (
            {
                "candidate_group_ids": [7, 8],
                "add_group_caps": {True: 1, 8: 1},
            },
            "keys must be integers",
        ),
        ({"candidate_conflicts": [(100,)]}, "contain two item ids"),
        ({"candidate_conflicts": [(100, 100)]}, "self-conflicts"),
        ({"candidate_conflicts": [(100, 999)]}, "unknown candidate"),
        (
            {"candidate_conflicts": [(100, 101)], "max_compiled_conflicts": 0},
            "max_compiled_conflicts",
        ),
        ({"timeout_ms": True}, "timeout_ms"),
        ({"max_replacements": -1}, "max_replacements"),
        (
            {"max_replacements": None, "protected_count": True},
            "protected_count must be an integer",
        ),
        (
            {"context": {"safe_slate_candidate_pool": {}}},
            "reserved",
        ),
    ],
)
def test_candidate_pool_rejects_malformed_bounded_inputs(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        guarded_candidate_pool_rerank(**_request(**overrides))
