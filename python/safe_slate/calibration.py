"""Fail-closed finite-cell sample-split calibration utilities."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite, log, sqrt
from typing import Generic, TypeVar

Cell = TypeVar("Cell", bound=Hashable)


@dataclass(frozen=True)
class CalibrationResult(Generic[Cell]):
    """Simultaneous one-sided lower bounds for registered finite cells."""

    lower_bounds: dict[Cell, float]
    alpha_by_cell: dict[Cell, float]
    sample_sizes: dict[Cell, int]
    total_alpha: float
    allocated_alpha: float


def finite_cell_lower_bounds(
    calibration_contrasts: Mapping[Cell, Sequence[float]],
    *,
    total_alpha: float,
    value_range: tuple[float, float] | Mapping[Cell, tuple[float, float]],
    alpha_by_cell: Mapping[Cell, float] | None = None,
) -> CalibrationResult[Cell]:
    """Compute Proposition 3's finite-cell Hoeffding lower bounds.

    Calibration samples must be direct bounded contrasts from a split that is
    independent of model training. A scalar ``value_range`` is shared by all
    cells; a mapping supplies registered cell-specific ranges. If allocations
    are omitted, ``total_alpha`` is divided equally across cells.

    The predictor does not appear in this function because it cancels from the
    proposition's expression. Empty cells, non-finite values, invalid ranges,
    or an excessive error budget raise ``ValueError`` instead of producing an
    uncertified bound.
    """

    if not calibration_contrasts:
        raise ValueError("at least one registered calibration cell is required")
    if not isfinite(total_alpha) or not 0.0 < total_alpha < 1.0:
        raise ValueError("total_alpha must be finite and in (0, 1)")

    cells = list(calibration_contrasts)
    if alpha_by_cell is None:
        allocation = total_alpha / len(cells)
        allocations = {cell: allocation for cell in cells}
    else:
        if set(alpha_by_cell) != set(cells):
            raise ValueError("alpha_by_cell must contain exactly the registered cells")
        allocations = dict(alpha_by_cell)

    if any(not isfinite(alpha) or not 0.0 < alpha < 1.0 for alpha in allocations.values()):
        raise ValueError("every cell alpha must be finite and in (0, 1)")
    allocated_total = sum(allocations.values())
    if allocated_total > total_alpha + 1e-15:
        raise ValueError("cell alpha allocations exceed total_alpha")

    lowers: dict[Cell, float] = {}
    sizes: dict[Cell, int] = {}
    for cell in cells:
        samples = tuple(float(value) for value in calibration_contrasts[cell])
        if not samples:
            raise ValueError(f"calibration cell {cell!r} is empty")
        if any(not isfinite(value) for value in samples):
            raise ValueError(f"calibration cell {cell!r} contains NaN or infinity")

        if isinstance(value_range, Mapping):
            if cell not in value_range:
                raise ValueError(f"value_range is missing calibration cell {cell!r}")
            bounds = value_range[cell]
        else:
            bounds = value_range
        if len(bounds) != 2:
            raise ValueError(f"calibration cell {cell!r} needs a two-value range")
        lower, upper = float(bounds[0]), float(bounds[1])
        if not isfinite(lower) or not isfinite(upper) or lower >= upper:
            raise ValueError(f"calibration cell {cell!r} has an invalid value range")
        if any(value < lower or value > upper for value in samples):
            raise ValueError(f"calibration cell {cell!r} has a sample outside its range")

        count = len(samples)
        mean = sum(samples) / count
        radius = (upper - lower) * sqrt(log(1.0 / allocations[cell]) / (2.0 * count))
        lowers[cell] = mean - radius
        sizes[cell] = count

    return CalibrationResult(
        lower_bounds=lowers,
        alpha_by_cell=allocations,
        sample_sizes=sizes,
        total_alpha=total_alpha,
        allocated_alpha=allocated_total,
    )
