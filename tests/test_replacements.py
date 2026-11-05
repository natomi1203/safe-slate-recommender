from __future__ import annotations

import pytest

from safe_slate import optimize_replacements


def test_balanced_operations_enforce_remove_and_add_uniqueness() -> None:
    result = optimize_replacements(
        remove_positions=[2, 2, 3, 3],
        add_item_ids=[100, 101, 100, 102],
        reward_deltas=[10.0, 9.0, 8.0, 7.0],
        safety_means=[1.0, 1.0, 1.0, 1.0],
        safety_variances=[0.0, 0.0, 0.0, 0.0],
        baseline_size=4,
        protected_count=2,
        max_replacements=3,
        beam_width=32,
    )

    assert result.exact
    assert result.selected_operation_indices == [0, 3]
    assert result.remove_positions == [2, 3]
    assert result.add_item_ids == [100, 102]
    assert result.k_lower == result.k_upper == 2
    assert result.cardinality_gap == 0
    assert result.first_unresolved_depth is None


def test_protected_region_is_rejected_before_search() -> None:
    with pytest.raises(ValueError, match="protected"):
        optimize_replacements(
            remove_positions=[1],
            add_item_ids=[100],
            reward_deltas=[1.0],
            safety_means=[1.0],
            safety_variances=[0.0],
            baseline_size=4,
            protected_count=2,
            max_replacements=1,
        )


def test_left_and_right_group_caps_are_independent_partitions() -> None:
    result = optimize_replacements(
        remove_positions=[2, 3, 4],
        add_item_ids=[100, 101, 102],
        reward_deltas=[3.0, 2.0, 1.0],
        safety_means=[1.0, 1.0, 1.0],
        safety_variances=[0.0, 0.0, 0.0],
        baseline_size=5,
        protected_count=2,
        max_replacements=3,
        remove_group_ids=[0, 0, 1],
        remove_group_caps={0: 1, 1: 1},
        add_group_ids=[9, 9, 10],
        add_group_caps={9: 1, 10: 1},
        beam_width=16,
    )

    assert result.exact
    assert result.k_lower == result.k_upper == 2
    assert len(set(result.remove_positions)) == 2
    assert len(set(result.add_item_ids)) == 2


def test_replacement_predicate_receives_materialized_plan() -> None:
    seen: list[dict[str, object]] = []

    def exactly_two(plan: dict[str, object]) -> bool:
        seen.append(plan)
        operations = plan["operation_indices"]
        added = plan["add_item_ids"]
        materialized = plan["materialized_slate"]
        assert isinstance(operations, list)
        assert isinstance(added, list)
        assert isinstance(materialized, list)
        return (
            len(operations) == 2
            and 999 not in added
            and materialized[:2] == [10, 11]
        )

    result = optimize_replacements(
        remove_positions=[2, 3, 4],
        add_item_ids=[100, 101, 999],
        reward_deltas=[3.0, 2.0, 100.0],
        safety_means=[1.0, 1.0, 1.0],
        safety_variances=[0.0, 0.0, 0.0],
        baseline_size=5,
        protected_count=2,
        max_replacements=2,
        baseline_item_ids=[10, 11, 12, 13, 14],
        custom_predicate=exactly_two,
        beam_width=16,
    )

    assert seen
    assert result.selected_operation_indices == [0, 1]
    assert result.add_item_ids == [100, 101]
    assert result.rechecked


def test_budget_stop_returns_structural_upper_and_unresolved_depth() -> None:
    result = optimize_replacements(
        remove_positions=[2, 3, 4, 5],
        add_item_ids=[100, 101, 102, 103],
        reward_deltas=[4.0, 3.0, 2.0, 1.0],
        safety_means=[1.0, 1.0, 1.0, 1.0],
        safety_variances=[0.0, 0.0, 0.0, 0.0],
        baseline_size=6,
        protected_count=2,
        max_replacements=4,
        max_expansions=1,
    )

    assert result.rechecked
    assert not result.exact
    assert result.k_lower == 1
    assert result.k_upper == 4
    assert result.cardinality_gap == 3
    assert result.first_unresolved_depth == 2


def test_unresolved_baseline_collision_falls_back_exactly() -> None:
    result = optimize_replacements(
        remove_positions=[2],
        add_item_ids=[10],
        reward_deltas=[100.0],
        safety_means=[1.0],
        safety_variances=[0.0],
        baseline_size=3,
        protected_count=2,
        max_replacements=1,
        baseline_item_ids=[10, 11, 12],
    )

    assert result.status == "baseline"
    assert result.exact
    assert result.k_lower == result.k_upper == 0
    assert result.materialized_slate == [10, 11, 12]


def test_materialized_slate_preserves_protected_prefix_and_uniqueness() -> None:
    result = optimize_replacements(
        remove_positions=[2, 3],
        add_item_ids=[100, 101],
        reward_deltas=[2.0, 1.0],
        safety_means=[1.0, 1.0],
        safety_variances=[0.0, 0.0],
        baseline_size=4,
        protected_count=2,
        max_replacements=2,
        baseline_item_ids=[10, 11, 12, 13],
    )

    assert result.materialized_slate == [10, 11, 100, 101]
    assert result.materialized_slate[:2] == [10, 11]
    assert len(set(result.materialized_slate)) == 4


def test_collision_invalid_prefix_can_become_valid_after_second_replacement() -> None:
    result = optimize_replacements(
        remove_positions=[2, 3],
        add_item_ids=[13, 14],
        reward_deltas=[2.0, 1.0],
        safety_means=[1.0, 1.0],
        safety_variances=[0.0, 0.0],
        baseline_size=4,
        protected_count=2,
        max_replacements=2,
        baseline_item_ids=[10, 11, 12, 13],
        beam_width=8,
    )

    assert result.selected_operation_indices == [0, 1]
    assert result.materialized_slate == [10, 11, 13, 14]
    assert result.exact


def test_unresolved_group_caps_tighten_structural_upper() -> None:
    result = optimize_replacements(
        remove_positions=[2, 3, 4, 5],
        add_item_ids=[100, 101, 102, 103],
        reward_deltas=[4.0, 3.0, 2.0, 1.0],
        safety_means=[1.0, 1.0, 1.0, 1.0],
        safety_variances=[0.0, 0.0, 0.0, 0.0],
        baseline_size=6,
        protected_count=2,
        max_replacements=4,
        remove_group_ids=[0, 0, 1, 1],
        remove_group_caps={0: 1, 1: 1},
        max_expansions=1,
    )

    assert not result.exact
    assert result.k_lower == 1
    assert result.k_upper == 2
    assert result.first_unresolved_depth == 2


def test_pair_conflict_clique_cover_tightens_upper() -> None:
    result = optimize_replacements(
        remove_positions=[2, 3, 4, 5, 6, 7],
        add_item_ids=[100, 101, 102, 103, 104, 105],
        reward_deltas=[6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        safety_means=[1.0] * 6,
        safety_variances=[0.0] * 6,
        baseline_size=8,
        protected_count=2,
        max_replacements=6,
        operation_conflicts=[
            (0, 1),
            (0, 2),
            (1, 2),
            (3, 4),
            (3, 5),
            (4, 5),
        ],
        max_expansions=1,
    )

    assert not result.exact
    assert result.k_lower == 1
    assert result.k_upper == 2
    assert result.upper_bound_source == "remove_add_group_clique_cover"


def test_replacement_batch_predicate_receives_materialized_slates() -> None:
    observed_batch_sizes: list[int] = []

    def batch_rule(plans: list[dict[str, object]]) -> list[bool]:
        observed_batch_sizes.append(len(plans))
        decisions = []
        for plan in plans:
            materialized = plan["materialized_slate"]
            operations = plan["operation_indices"]
            assert isinstance(materialized, list)
            assert isinstance(operations, list)
            decisions.append(len(operations) == 2 and materialized[:2] == [10, 11])
        return decisions

    result = optimize_replacements(
        remove_positions=[2, 3, 4],
        add_item_ids=[100, 101, 102],
        reward_deltas=[3.0, 2.0, 1.0],
        safety_means=[1.0, 1.0, 1.0],
        safety_variances=[0.0, 0.0, 0.0],
        baseline_size=5,
        protected_count=2,
        max_replacements=2,
        baseline_item_ids=[10, 11, 12, 13, 14],
        custom_predicate_batch=batch_rule,
        beam_width=16,
    )

    assert result.selected_operation_indices == [0, 1]
    assert result.materialized_slate == [10, 11, 100, 101, 14]
    assert any(size > 1 for size in observed_batch_sizes)
    assert result.predicate_batches < result.predicate_calls
