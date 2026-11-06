from __future__ import annotations

from collections.abc import Callable

import pytest

from safe_slate import (
    optimize_replacements,
    optimize_slate,
    prepare_linear_contrast_marginals_f64le,
    project_linear_contrast_marginals,
    project_linear_contrast_uncertainty,
)


def _replacement_kwargs() -> dict[str, object]:
    return {
        "remove_positions": [2, 3],
        "add_item_ids": [100, 101],
        "reward_deltas": [2.0, 1.0],
        "safety_means": [1.0, 1.0],
        "safety_variances": [0.0, 0.0],
        "baseline_size": 4,
        "protected_count": 2,
        "max_replacements": 2,
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(remove_positions=[2]), "remove_positions has length"),
        (lambda value: value.update(protected_count=5), "protected_count"),
        (
            lambda value: value.update(baseline_item_ids=[10, 11, 12]),
            "baseline_item_ids has length",
        ),
        (
            lambda value: value.update(baseline_item_ids=[10, 11, 11, 13]),
            "baseline_item_ids must be unique",
        ),
        (
            lambda value: value.update(remove_group_ids=[0]),
            "remove_group_ids and remove_group_caps",
        ),
        (
            lambda value: value.update(
                remove_group_ids=[0], remove_group_caps={0: 1}
            ),
            "remove_group_ids has length",
        ),
        (
            lambda value: value.update(add_group_caps={0: 1}),
            "add_group_ids and add_group_caps",
        ),
        (
            lambda value: value.update(add_group_ids=[0], add_group_caps={0: 1}),
            "add_group_ids has length",
        ),
        (
            lambda value: value.update(operation_conflicts=[(0, 0)]),
            "self-conflict",
        ),
    ],
)
def test_replacement_wrapper_rejects_malformed_contracts(
    mutation: Callable[[dict[str, object]], None], message: str
) -> None:
    kwargs = _replacement_kwargs()
    mutation(kwargs)
    with pytest.raises(ValueError, match=message):
        optimize_replacements(**kwargs)


def test_empty_replacement_set_is_an_exact_baseline() -> None:
    result = optimize_replacements(
        remove_positions=[],
        add_item_ids=[],
        reward_deltas=[],
        safety_means=[],
        safety_variances=[],
        baseline_size=2,
        protected_count=2,
        max_replacements=0,
        baseline_item_ids=[10, 11],
    )
    assert result.exact
    assert result.materialized_slate == [10, 11]
    assert "CertifiedReplacementResult" in repr(result)


def test_native_result_and_projection_representations_are_stable() -> None:
    result = optimize_slate(
        rewards=[1.0],
        safety_means=[1.0],
        safety_variances=[0.0],
        max_items=1,
    )
    projection = project_linear_contrast_uncertainty(
        safety_means=[0.1],
        contrast_features=[[1.0]],
        parameter_covariance=[[0.01]],
    )
    prepared = prepare_linear_contrast_marginals_f64le(
        contrast_features=[[1.0]], parameter_covariance=[[0.01]]
    )
    assert "CertifiedOracleResult" in repr(result)
    assert "ProjectedContrastUncertainty" in repr(projection)
    assert "PreparedLinearContrastMarginals" in repr(prepared)
    assert project_linear_contrast_marginals([[1.0]], [[0.01]]) == [0.01]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"contrast_features": [[1.0]]},
        {"parameter_covariance": [[1.0]]},
        {
            "covariance": [[0.0]],
            "contrast_features": [[1.0]],
            "parameter_covariance": [[0.0]],
        },
        {
            "contrast_features": [[1.0], [2.0]],
            "parameter_covariance": [[1.0]],
        },
    ],
)
def test_slate_wrapper_rejects_incomplete_factorized_uncertainty(
    kwargs: dict[str, object]
) -> None:
    with pytest.raises(ValueError):
        optimize_slate(
            rewards=[1.0],
            safety_means=[1.0],
            safety_variances=[0.0],
            max_items=1,
            **kwargs,
        )


def test_batch_predicate_exception_fails_closed() -> None:
    def unavailable(_candidates: list[list[int]]) -> list[bool]:
        raise RuntimeError("predicate service unavailable")

    result = optimize_slate(
        rewards=[2.0, 1.0],
        safety_means=[1.0, 1.0],
        safety_variances=[0.0, 0.0],
        max_items=2,
        custom_predicate_batch=unavailable,
    )
    assert result.status == "baseline"
    assert result.predicate_errors > 0
    assert result.rechecked


def test_replacement_batch_predicate_failures_are_fail_closed() -> None:
    for callback in (
        lambda _plans: [],
        lambda _plans: (_ for _ in ()).throw(RuntimeError("unavailable")),
    ):
        kwargs = _replacement_kwargs()
        kwargs["custom_predicate_batch"] = callback
        result = optimize_replacements(**kwargs)
        assert result.status == "baseline"
        assert result.predicate_errors > 0
        assert result.rechecked

    kwargs = _replacement_kwargs()
    kwargs["custom_predicate"] = lambda _plan: (_ for _ in ()).throw(
        RuntimeError("unavailable")
    )
    result = optimize_replacements(**kwargs)
    assert result.status == "baseline"
    assert result.predicate_errors > 0


def test_replacement_batch_skips_every_duplicate_materialization() -> None:
    result = optimize_replacements(
        remove_positions=[2],
        add_item_ids=[10],
        reward_deltas=[1.0],
        safety_means=[1.0],
        safety_variances=[0.0],
        baseline_size=4,
        protected_count=2,
        max_replacements=1,
        baseline_item_ids=[10, 11, 12, 13],
        custom_predicate_batch=lambda _plans: pytest.fail(
            "invalid materializations must not reach the callback"
        ),
    )
    assert result.status == "baseline"


def test_replacement_wrapper_rejects_bad_factorized_uncertainty() -> None:
    kwargs = _replacement_kwargs()
    kwargs.update(
        contrast_features=[[1.0], [1.0]],
        parameter_covariance=[[2.0]],
    )
    with pytest.raises(ValueError, match="diagonal must match"):
        optimize_replacements(**kwargs)


def test_native_tie_breaking_and_deep_psd_paths_are_exercised() -> None:
    tied = optimize_slate(
        rewards=[1.0, 1.0],
        safety_means=[1.0, 1.0],
        safety_variances=[0.0, 0.0],
        max_items=1,
    )
    assert tied.selected_indices == [0]

    projection = project_linear_contrast_uncertainty(
        safety_means=[1.0, 1.0, 1.0],
        contrast_features=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        parameter_covariance=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    )
    assert projection.safety_variances == [1.0, 1.0, 1.0]


def test_partition_identifier_overflow_fails_closed() -> None:
    kwargs = _replacement_kwargs()
    kwargs["add_item_ids"] = [2**63, 101]
    with pytest.raises(ValueError, match="partition identifier"):
        optimize_replacements(**kwargs)
