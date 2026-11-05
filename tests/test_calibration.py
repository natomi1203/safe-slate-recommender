from __future__ import annotations

import math

import pytest

from safe_slate import finite_cell_lower_bounds


def test_finite_cell_hoeffding_lower_bound() -> None:
    result = finite_cell_lower_bounds(
        {"home": [0.2, 0.4, 0.6, 0.8], "search": [0.1, 0.3]},
        total_alpha=0.1,
        value_range=(0.0, 1.0),
    )

    expected_home = 0.5 - math.sqrt(math.log(20.0) / 8.0)
    assert result.lower_bounds["home"] == pytest.approx(expected_home)
    assert result.alpha_by_cell == {"home": 0.05, "search": 0.05}
    assert result.total_alpha == pytest.approx(0.1)
    assert result.allocated_alpha == pytest.approx(0.1)


def test_calibration_fails_closed_on_empty_or_out_of_range_cell() -> None:
    with pytest.raises(ValueError, match="empty"):
        finite_cell_lower_bounds(
            {"home": []}, total_alpha=0.1, value_range=(0.0, 1.0)
        )

    with pytest.raises(ValueError, match="outside"):
        finite_cell_lower_bounds(
            {"home": [1.1]}, total_alpha=0.1, value_range=(0.0, 1.0)
        )


def test_custom_allocation_may_leave_budget_unused() -> None:
    result = finite_cell_lower_bounds(
        {"home": [0.2], "search": [0.3]},
        total_alpha=0.1,
        alpha_by_cell={"home": 0.02, "search": 0.03},
        value_range={"home": (0.0, 1.0), "search": (0.0, 1.0)},
    )

    assert result.total_alpha == pytest.approx(0.1)
    assert result.allocated_alpha == pytest.approx(0.05)
