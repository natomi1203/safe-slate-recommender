from __future__ import annotations

import json
from pathlib import Path
from random import Random

import pytest

import safe_slate
from safe_slate import build_candidate_pool_operations, guarded_rerank, optimize_slate


ROOT = Path(__file__).resolve().parents[1]


def _guarded_kwargs() -> dict[str, object]:
    return {
        "request_id": "contract-test",
        "baseline_item_ids": [10, 11, 12],
        "protected_count": 1,
        "remove_positions": [1, 2],
        "add_item_ids": [100, 101],
        "reward_deltas": [0.2, 0.1],
        "safety_mean_deltas": [0.02, 0.01],
        "safety_variances": [0.0, 0.0],
        "baseline_metric": 0.5,
        "degradation_tolerance": 0.01,
        "max_replacements": 2,
    }


def test_registered_public_api_is_importable() -> None:
    manifest = json.loads((ROOT / "library.json").read_text(encoding="utf-8"))
    for qualified_name in manifest["public_api"]:
        module_name, attribute = qualified_name.rsplit(".", 1)
        assert module_name == "safe_slate"
        assert callable(getattr(safe_slate, attribute))
        assert attribute in safe_slate.__all__


@pytest.mark.parametrize(
    "field",
    [
        "support_ok",
        "uncertainty_valid",
        "distribution_shift_detected",
        "proposal_log_complete",
    ],
)
def test_raw_serving_gates_require_actual_booleans(field: str) -> None:
    kwargs = _guarded_kwargs()
    kwargs[field] = 1
    with pytest.raises(ValueError, match=f"{field} must be a boolean"):
        guarded_rerank(**kwargs)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("baseline_item_ids", [10, 11.5, 12]),
        ("baseline_item_ids", [10, True, 12]),
        ("remove_positions", [1, "2"]),
        ("add_item_ids", [100, False]),
        ("reward_deltas", [0.2, "0.1"]),
        ("safety_mean_deltas", [0.02, True]),
        ("safety_variances", [0.0, -0.1]),
    ],
)
def test_guarded_rerank_rejects_coercible_or_unsafe_values(
    field: str, value: object
) -> None:
    kwargs = _guarded_kwargs()
    kwargs[field] = value
    with pytest.raises(ValueError):
        guarded_rerank(**kwargs)


def test_candidate_pool_rejects_coercible_numeric_strings() -> None:
    with pytest.raises(ValueError, match="finite real numbers"):
        build_candidate_pool_operations(
            baseline_item_ids=[10, 11],
            baseline_reward_scores=[0.9, "0.2"],
            baseline_safety_means=[0.1, 0.05],
            baseline_features=[[1.0], [0.0]],
            candidate_item_ids=[100],
            candidate_reward_scores=[0.8],
            candidate_safety_means=[0.08],
            candidate_features=[[0.5]],
            parameter_covariance=[[0.01]],
            protected_count=1,
        )


def test_zero_budget_returns_a_rechecked_fail_closed_result() -> None:
    result = guarded_rerank(**_guarded_kwargs(), timeout_ms=0)
    assert result.slate == (10, 11, 12)
    assert result.oracle.rechecked
    assert result.proposal_log.deadline_reached


def test_randomized_oracle_invariants_and_determinism() -> None:
    random = Random(20261106)
    for _ in range(80):
        item_count = random.randint(1, 10)
        max_items = random.randint(0, item_count)
        rewards = [random.uniform(-2.0, 2.0) for _ in range(item_count)]
        means = [random.uniform(-0.2, 0.8) for _ in range(item_count)]
        variances = [random.uniform(0.0, 0.05) for _ in range(item_count)]
        groups = [random.randrange(3) for _ in range(item_count)]
        caps = {group: random.randint(0, max_items) for group in range(3)}
        conflicts = [
            (left, right)
            for left in range(item_count)
            for right in range(left + 1, item_count)
            if random.random() < 0.08
        ]
        kwargs = {
            "rewards": rewards,
            "safety_means": means,
            "safety_variances": variances,
            "max_items": max_items,
            "group_ids": groups,
            "group_caps": caps,
            "conflicts": conflicts,
            "beam_width": 64,
            "max_expansions": 20_000,
            "timeout_ms": 1_000,
        }
        first = optimize_slate(**kwargs)
        second = optimize_slate(**kwargs)

        assert first.selected_indices == second.selected_indices
        assert first.k_lower == second.k_lower
        assert first.k_upper == second.k_upper
        assert first.status == second.status
        assert first.rechecked and second.rechecked
        assert len(first.selected_indices) <= max_items
        assert first.safety_lcb >= -1e-12
        for group, cap in caps.items():
            assert sum(groups[index] == group for index in first.selected_indices) <= cap
        selected = set(first.selected_indices)
        assert all(left not in selected or right not in selected for left, right in conflicts)
