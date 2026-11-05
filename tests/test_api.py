from __future__ import annotations

import math

import pytest

from safe_slate import optimize_slate, project_linear_contrast_uncertainty


def test_linear_contrast_projection_retains_shared_covariance() -> None:
    projection = project_linear_contrast_uncertainty(
        safety_means=[0.3, 0.2],
        contrast_features=[[1.0, 1.0], [1.0, -1.0]],
        parameter_covariance=[[2.0, 0.5], [0.5, 1.0]],
        confidence_radius=2.0,
    )

    assert projection.feature_dimension == 2
    assert projection.safety_variances == pytest.approx([4.0, 2.0])
    assert projection.covariance == [[4.0, 1.0], [1.0, 2.0]]
    assert projection.individual_lcbs == pytest.approx(
        [0.3 - 4.0, 0.2 - 2.0 * math.sqrt(2.0)]
    )


def test_linear_contrast_projection_rejects_indefinite_parameter_covariance() -> None:
    with pytest.raises(ValueError, match="positive semidefinite"):
        project_linear_contrast_uncertainty(
            safety_means=[0.1],
            contrast_features=[[1.0, 0.0]],
            parameter_covariance=[[1.0, 2.0], [2.0, 1.0]],
        )


def test_factorized_linear_uncertainty_matches_full_covariance_search() -> None:
    common = {
        "rewards": [3.0, 2.0, 1.0],
        "safety_means": [0.04, 0.03, 0.02],
        "safety_variances": [0.0001, 0.000125, 0.0002],
        "max_items": 2,
        "safety_threshold": -0.01,
        "confidence_radius": 1.0,
        "beam_width": 16,
    }
    factorized = optimize_slate(
        **common,
        contrast_features=[[1.0, 0.0], [0.5, 1.0], [1.0, -1.0]],
        parameter_covariance=[[0.0001, 0.0], [0.0, 0.0001]],
    )
    full = optimize_slate(
        **common,
        covariance=[
            [0.0001, 0.00005, 0.0001],
            [0.00005, 0.000125, -0.00005],
            [0.0001, -0.00005, 0.0002],
        ],
    )

    assert factorized.selected_indices == full.selected_indices
    assert factorized.safety_variance == pytest.approx(full.safety_variance)
    assert factorized.safety_lcb == pytest.approx(full.safety_lcb)


def test_factorized_linear_uncertainty_rejects_wrong_marginals() -> None:
    with pytest.raises(ValueError, match="diagonal must match"):
        optimize_slate(
            rewards=[1.0],
            safety_means=[0.1],
            safety_variances=[999.0],
            max_items=1,
            contrast_features=[[1.0]],
            parameter_covariance=[[0.01]],
        )


def test_cardinality_first_and_native_constraints() -> None:
    result = optimize_slate(
        rewards=[10.0, 8.0, 7.0, 6.0],
        safety_means=[1.0, 1.0, 1.0, 1.0],
        safety_variances=[0.0, 0.0, 0.0, 0.0],
        max_items=3,
        group_ids=[0, 0, 1, 1],
        group_caps={0: 1, 1: 1},
        conflicts=[(0, 2)],
        beam_width=32,
    )

    assert result.exact
    assert result.selected_indices == [0, 3]
    assert result.k_lower == result.k_upper == 2
    assert result.rechecked


def test_nonmonotone_python_predicate_does_not_prune_descendants() -> None:
    def exactly_two(selected: list[int]) -> bool:
        return len(selected) == 2

    result = optimize_slate(
        rewards=[4.0, 3.0, 2.0],
        safety_means=[1.0, 1.0, 1.0],
        safety_variances=[0.0, 0.0, 0.0],
        max_items=2,
        custom_predicate=exactly_two,
        beam_width=16,
    )

    assert result.selected_indices == [0, 1]
    assert result.exact


def test_predicate_exception_fails_closed_to_baseline() -> None:
    calls = 0

    def unstable(_selected: list[int]) -> bool:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("business rule unavailable")
        return True

    result = optimize_slate(
        rewards=[2.0, 1.0],
        safety_means=[1.0, 1.0],
        safety_variances=[0.0, 0.0],
        max_items=2,
        custom_predicate=unstable,
    )

    assert result.selected_indices == []
    assert result.status == "baseline"
    assert result.predicate_errors > 0
    assert result.rechecked
    assert not result.exact
    assert result.k_upper == 2


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_input_is_rejected(bad: float) -> None:
    with pytest.raises(ValueError):
        optimize_slate(
            rewards=[bad],
            safety_means=[1.0],
            safety_variances=[0.0],
            max_items=1,
        )


def test_indefinite_covariance_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive semidefinite"):
        optimize_slate(
            rewards=[1.0, 1.0],
            safety_means=[1.0, 1.0],
            safety_variances=[1.0, 1.0],
            covariance=[[1.0, 2.0], [2.0, 1.0]],
            max_items=2,
        )


def test_beam_truncation_keeps_posterior_interval_open() -> None:
    result = optimize_slate(
        rewards=[5.0, 4.0, 3.0, 2.0],
        safety_means=[1.0, 1.0, 1.0, 1.0],
        safety_variances=[0.0, 0.0, 0.0, 0.0],
        max_items=3,
        beam_width=1,
    )

    assert result.rechecked
    assert not result.exact
    assert result.k_lower <= result.k_upper == 3


def test_unsafe_prefix_is_not_pruned_when_pair_is_safe() -> None:
    result = optimize_slate(
        rewards=[2.0, 1.0],
        safety_means=[1.0, 1.0],
        safety_variances=[1.0, 1.0],
        covariance=[[1.0, -0.9], [-0.9, 1.0]],
        confidence_radius=1.1,
        max_items=2,
        beam_width=8,
    )

    assert result.selected_indices == [0, 1]
    assert result.safety_lcb > 0.0
    assert result.exact


def test_batch_predicate_reduces_python_boundary_crossings() -> None:
    batch_sizes: list[int] = []

    def exactly_two_batch(candidates: list[list[int]]) -> list[bool]:
        batch_sizes.append(len(candidates))
        return [len(candidate) == 2 for candidate in candidates]

    result = optimize_slate(
        rewards=[5.0, 4.0, 3.0, 2.0, 1.0],
        safety_means=[1.0] * 5,
        safety_variances=[0.0] * 5,
        max_items=2,
        beam_width=32,
        custom_predicate_batch=exactly_two_batch,
    )

    assert result.exact
    assert result.selected_indices == [0, 1]
    assert any(size > 1 for size in batch_sizes)
    assert result.predicate_batches < result.predicate_calls


def test_invalid_batch_response_fails_closed() -> None:
    result = optimize_slate(
        rewards=[2.0, 1.0],
        safety_means=[1.0, 1.0],
        safety_variances=[0.0, 0.0],
        max_items=2,
        custom_predicate_batch=lambda _candidates: [],
    )

    assert result.status == "baseline"
    assert result.rechecked
    assert not result.exact
    assert result.predicate_errors > 0
