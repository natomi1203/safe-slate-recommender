"""High-level real-time guardrail for candidate-pool re-ranking.

This module keeps the Python-facing request shape item-oriented.  It expands
the replaceable baseline suffix and insertion candidate pool into atomic
replacement operations, then delegates uncertainty projection, certified
search, timeout handling, and final rechecks to the existing Rust-backed
serving path.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite
from numbers import Real
from time import perf_counter
from typing import Any, Literal, Mapping, Sequence

from .gates import ServingGateAssessment
from .production import GuardedRerankResult, guarded_linear_rerank
from .uncertainty_utils import MATRIX_DIGEST_F64LE_V1


@dataclass(frozen=True, slots=True)
class CandidatePoolOperations:
    """Deterministic operation expansion consumed by the Rust oracle."""

    remove_positions: tuple[int, ...]
    add_item_ids: tuple[int, ...]
    reward_deltas: tuple[float, ...]
    safety_mean_deltas: tuple[float, ...]
    contrast_features: tuple[tuple[float, ...], ...]
    remove_group_ids: tuple[int, ...] | None
    remove_group_caps: tuple[tuple[int, int], ...] | None
    add_group_ids: tuple[int, ...] | None
    add_group_caps: tuple[tuple[int, int], ...] | None
    operation_conflicts: tuple[tuple[int, int], ...]

    @property
    def operation_count(self) -> int:
        return len(self.remove_positions)


def _finite_values(name: str, values: Sequence[float], expected: int) -> tuple[float, ...]:
    if len(values) != expected:
        raise ValueError(f"{name} has length {len(values)}, expected {expected}")
    normalized: list[float] = []
    for value in values:
        if (
            not isinstance(value, Real)
            or isinstance(value, bool)
            or not isfinite(value)
        ):
            raise ValueError(f"{name} must contain finite real numbers")
        normalized.append(float(value))
    return tuple(normalized)


def _item_ids(name: str, values: Sequence[int]) -> tuple[int, ...]:
    normalized: list[int] = []
    for value in values:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must contain nonnegative integers")
        normalized.append(value)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must be unique")
    return tuple(normalized)


def _feature_rows(
    name: str,
    rows: Sequence[Sequence[float]],
    expected_rows: int,
    dimension: int,
) -> tuple[tuple[float, ...], ...]:
    if len(rows) != expected_rows:
        raise ValueError(f"{name} has {len(rows)} rows, expected {expected_rows}")
    normalized: list[tuple[float, ...]] = []
    for row_index, row in enumerate(rows):
        if len(row) != dimension:
            raise ValueError(
                f"{name} row {row_index} has length {len(row)}, expected {dimension}"
            )
        values: list[float] = []
        for value in row:
            if (
                not isinstance(value, Real)
                or isinstance(value, bool)
                or not isfinite(value)
            ):
                raise ValueError(f"{name} must contain finite real numbers")
            values.append(float(value))
        normalized.append(tuple(values))
    return tuple(normalized)


def _group_inputs(
    *,
    side: str,
    group_ids: Sequence[int] | None,
    group_caps: Mapping[int, int] | None,
    expected: int,
    required_indices: Sequence[int] | None = None,
) -> tuple[tuple[int, ...] | None, dict[int, int] | None]:
    if group_ids is None and group_caps is None:
        return None, None
    if group_ids is None or group_caps is None:
        raise ValueError(f"{side}_group_ids and {side}_group_caps must be supplied together")
    if len(group_ids) != expected:
        raise ValueError(
            f"{side}_group_ids has length {len(group_ids)}, expected {expected}"
        )
    normalized_ids: list[int] = []
    for value in group_ids:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{side}_group_ids must contain integers")
        normalized_ids.append(value)
    normalized_caps: dict[int, int] = {}
    for key, value in group_caps.items():
        if not isinstance(key, int) or isinstance(key, bool):
            raise ValueError(f"{side}_group_caps keys must be integers")
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{side}_group_caps values must be nonnegative integers")
        normalized_caps[key] = value
    required_ids = (
        normalized_ids
        if required_indices is None
        else [normalized_ids[index] for index in required_indices]
    )
    missing = sorted(set(required_ids) - set(normalized_caps))
    if missing:
        raise ValueError(f"{side}_group_caps is missing group ids {missing}")
    return tuple(normalized_ids), normalized_caps


def build_candidate_pool_operations(
    *,
    baseline_item_ids: Sequence[int],
    baseline_reward_scores: Sequence[float],
    baseline_safety_means: Sequence[float],
    baseline_features: Sequence[Sequence[float]],
    candidate_item_ids: Sequence[int],
    candidate_reward_scores: Sequence[float],
    candidate_safety_means: Sequence[float],
    candidate_features: Sequence[Sequence[float]],
    parameter_covariance: Sequence[Sequence[float]],
    protected_count: int,
    baseline_group_ids: Sequence[int] | None = None,
    remove_group_caps: Mapping[int, int] | None = None,
    candidate_group_ids: Sequence[int] | None = None,
    add_group_caps: Mapping[int, int] | None = None,
    candidate_conflicts: Sequence[tuple[int, int]] | None = None,
    max_operations: int = 100_000,
    max_feature_cells: int = 10_000_000,
    max_compiled_conflicts: int = 1_000_000,
) -> CandidatePoolOperations:
    """Compile item-level arrays into balanced add/remove operations.

    Candidate ids must be new insertions, not ids already present in the
    baseline. Reordering an unchanged item set is a different optimization
    problem and is deliberately rejected instead of being approximated by
    invalid duplicate-producing replacements.
    """

    baseline = _item_ids("baseline_item_ids", baseline_item_ids)
    candidates = _item_ids("candidate_item_ids", candidate_item_ids)
    if not baseline:
        raise ValueError("baseline_item_ids must be nonempty")
    overlap = sorted(set(baseline) & set(candidates))
    if overlap:
        raise ValueError(
            "candidate_item_ids must not already occur in baseline_item_ids; "
            f"overlap={overlap[:10]}"
        )
    if (
        not isinstance(protected_count, int)
        or isinstance(protected_count, bool)
        or not 0 <= protected_count <= len(baseline)
    ):
        raise ValueError("protected_count must be an integer in [0, baseline size]")
    if (
        not isinstance(max_operations, int)
        or isinstance(max_operations, bool)
        or max_operations < 0
    ):
        raise ValueError("max_operations must be a nonnegative integer")
    if (
        not isinstance(max_compiled_conflicts, int)
        or isinstance(max_compiled_conflicts, bool)
        or max_compiled_conflicts < 0
    ):
        raise ValueError("max_compiled_conflicts must be a nonnegative integer")
    if (
        not isinstance(max_feature_cells, int)
        or isinstance(max_feature_cells, bool)
        or max_feature_cells < 0
    ):
        raise ValueError("max_feature_cells must be a nonnegative integer")

    covariance_dimension = len(parameter_covariance)
    _feature_rows(
        "parameter_covariance",
        parameter_covariance,
        covariance_dimension,
        covariance_dimension,
    )
    if covariance_dimension == 0 and candidates:
        raise ValueError("parameter_covariance must be nonempty when candidates exist")

    baseline_rewards = _finite_values(
        "baseline_reward_scores", baseline_reward_scores, len(baseline)
    )
    baseline_safety = _finite_values(
        "baseline_safety_means", baseline_safety_means, len(baseline)
    )
    candidate_rewards = _finite_values(
        "candidate_reward_scores", candidate_reward_scores, len(candidates)
    )
    candidate_safety = _finite_values(
        "candidate_safety_means", candidate_safety_means, len(candidates)
    )
    baseline_feature_rows = _feature_rows(
        "baseline_features", baseline_features, len(baseline), covariance_dimension
    )
    candidate_feature_rows = _feature_rows(
        "candidate_features", candidate_features, len(candidates), covariance_dimension
    )
    baseline_groups, normalized_remove_caps = _group_inputs(
        side="remove",
        group_ids=baseline_group_ids,
        group_caps=remove_group_caps,
        expected=len(baseline),
        required_indices=range(protected_count, len(baseline)),
    )
    candidate_groups, normalized_add_caps = _group_inputs(
        side="add",
        group_ids=candidate_group_ids,
        group_caps=add_group_caps,
        expected=len(candidates),
    )

    replaceable_positions = tuple(range(protected_count, len(baseline)))
    operation_count = len(replaceable_positions) * len(candidates)
    if operation_count > max_operations:
        raise ValueError(
            f"candidate expansion creates {operation_count} operations, "
            f"exceeding max_operations={max_operations}"
        )
    feature_cells = operation_count * covariance_dimension
    if feature_cells > max_feature_cells:
        raise ValueError(
            f"candidate expansion creates {feature_cells} feature cells, "
            f"exceeding max_feature_cells={max_feature_cells}"
        )

    remove_positions: list[int] = []
    add_item_ids: list[int] = []
    reward_deltas: list[float] = []
    safety_mean_deltas: list[float] = []
    contrasts: list[tuple[float, ...]] = []
    operation_remove_groups: list[int] | None = [] if baseline_groups is not None else None
    operation_add_groups: list[int] | None = [] if candidate_groups is not None else None

    for position in replaceable_positions:
        for candidate_index, candidate_id in enumerate(candidates):
            remove_positions.append(position)
            add_item_ids.append(candidate_id)
            reward_deltas.append(
                candidate_rewards[candidate_index] - baseline_rewards[position]
            )
            safety_mean_deltas.append(
                candidate_safety[candidate_index] - baseline_safety[position]
            )
            contrasts.append(
                tuple(
                    candidate_feature_rows[candidate_index][feature]
                    - baseline_feature_rows[position][feature]
                    for feature in range(covariance_dimension)
                )
            )
            if operation_remove_groups is not None and baseline_groups is not None:
                operation_remove_groups.append(baseline_groups[position])
            if operation_add_groups is not None and candidate_groups is not None:
                operation_add_groups.append(candidate_groups[candidate_index])

    candidate_to_index = {item_id: index for index, item_id in enumerate(candidates)}
    compiled_conflicts: set[tuple[int, int]] = set()
    for pair in candidate_conflicts or ():
        if len(pair) != 2:
            raise ValueError("candidate_conflicts entries must contain two item ids")
        left_id, right_id = pair
        if left_id == right_id:
            raise ValueError("candidate_conflicts may not contain self-conflicts")
        if left_id not in candidate_to_index or right_id not in candidate_to_index:
            raise ValueError("candidate_conflicts references an unknown candidate item id")
        left_candidate = candidate_to_index[left_id]
        right_candidate = candidate_to_index[right_id]
        for left_position in range(len(replaceable_positions)):
            left_operation = left_position * len(candidates) + left_candidate
            for right_position in range(len(replaceable_positions)):
                right_operation = right_position * len(candidates) + right_candidate
                edge = (
                    min(left_operation, right_operation),
                    max(left_operation, right_operation),
                )
                # Self-conflicts are rejected above and candidate ids are unique,
                # so distinct pair members cannot compile to the same operation.
                if edge[0] != edge[1]:  # pragma: no branch - invariant defense
                    compiled_conflicts.add(edge)
                    if len(compiled_conflicts) > max_compiled_conflicts:
                        raise ValueError(
                            "candidate conflicts exceed "
                            f"max_compiled_conflicts={max_compiled_conflicts}"
                        )

    return CandidatePoolOperations(
        remove_positions=tuple(remove_positions),
        add_item_ids=tuple(add_item_ids),
        reward_deltas=tuple(reward_deltas),
        safety_mean_deltas=tuple(safety_mean_deltas),
        contrast_features=tuple(contrasts),
        remove_group_ids=(
            tuple(operation_remove_groups)
            if operation_remove_groups is not None
            else None
        ),
        remove_group_caps=(
            tuple(sorted(normalized_remove_caps.items()))
            if normalized_remove_caps is not None
            else None
        ),
        add_group_ids=(
            tuple(operation_add_groups) if operation_add_groups is not None else None
        ),
        add_group_caps=(
            tuple(sorted(normalized_add_caps.items()))
            if normalized_add_caps is not None
            else None
        ),
        operation_conflicts=tuple(sorted(compiled_conflicts)),
    )


def guarded_candidate_pool_rerank(
    *,
    request_id: str,
    uncertainty_snapshot_id: str,
    baseline_item_ids: Sequence[int],
    baseline_reward_scores: Sequence[float],
    baseline_safety_means: Sequence[float],
    baseline_features: Sequence[Sequence[float]],
    candidate_item_ids: Sequence[int],
    candidate_reward_scores: Sequence[float],
    candidate_safety_means: Sequence[float],
    candidate_features: Sequence[Sequence[float]],
    parameter_covariance: Sequence[Sequence[float]],
    protected_count: int,
    baseline_metric: float,
    degradation_tolerance: float,
    max_replacements: int | None = None,
    contrast_input_id: str | None = None,
    event_timestamp: str | None = None,
    gate_assessment: ServingGateAssessment | None = None,
    serving_mode: Literal["serve", "shadow"] = "serve",
    confidence_radius: float = 1.96,
    support_ok: bool = False,
    uncertainty_valid: bool = False,
    distribution_shift_detected: bool = True,
    proposal_log_complete: bool = False,
    model_version: str | None = None,
    context: Mapping[str, Any] | None = None,
    baseline_group_ids: Sequence[int] | None = None,
    remove_group_caps: Mapping[int, int] | None = None,
    candidate_group_ids: Sequence[int] | None = None,
    add_group_caps: Mapping[int, int] | None = None,
    candidate_conflicts: Sequence[tuple[int, int]] | None = None,
    beam_width: int = 128,
    max_expansions: int = 100_000,
    timeout_ms: int = 10,
    finalization_reserve_ms: int = 4,
    matrix_digest_encoding: Literal["json-v1", "f64le-v1"] = MATRIX_DIGEST_F64LE_V1,
    max_operations: int = 100_000,
    max_feature_cells: int = 10_000_000,
    max_compiled_conflicts: int = 1_000_000,
    custom_predicate: Any | None = None,
    custom_predicate_batch: Any | None = None,
) -> GuardedRerankResult:
    """Run one fail-closed candidate-pool re-ranking request.

    The protected prefix is immutable. Every replaceable suffix position is
    paired with every new candidate, and the Rust oracle chooses cardinality
    first and reward second. Shared parameter uncertainty is evaluated on the
    selected sum of add-minus-remove feature contrasts, so the returned
    ``selected_count`` is the request-specific certified replacement count.

    The default serving gates intentionally fail closed. Production callers
    should pass an evidence-derived ``gate_assessment`` or explicitly provide
    all four measured gate decisions.
    """

    if not isinstance(timeout_ms, int) or isinstance(timeout_ms, bool) or timeout_ms < 0:
        raise ValueError("timeout_ms must be a nonnegative integer")
    if max_replacements is None:
        if not isinstance(protected_count, int) or isinstance(protected_count, bool):
            raise ValueError("protected_count must be an integer")
        requested_max_replacements = max(0, len(baseline_item_ids) - protected_count)
    elif (
        not isinstance(max_replacements, int)
        or isinstance(max_replacements, bool)
        or max_replacements < 0
    ):
        raise ValueError("max_replacements must be a nonnegative integer or None")
    else:
        requested_max_replacements = max_replacements

    started = perf_counter()
    operations = build_candidate_pool_operations(
        baseline_item_ids=baseline_item_ids,
        baseline_reward_scores=baseline_reward_scores,
        baseline_safety_means=baseline_safety_means,
        baseline_features=baseline_features,
        candidate_item_ids=candidate_item_ids,
        candidate_reward_scores=candidate_reward_scores,
        candidate_safety_means=candidate_safety_means,
        candidate_features=candidate_features,
        parameter_covariance=parameter_covariance,
        protected_count=protected_count,
        baseline_group_ids=baseline_group_ids,
        remove_group_caps=remove_group_caps,
        candidate_group_ids=candidate_group_ids,
        add_group_caps=add_group_caps,
        candidate_conflicts=candidate_conflicts,
        max_operations=max_operations,
        max_feature_cells=max_feature_cells,
        max_compiled_conflicts=max_compiled_conflicts,
    )
    build_elapsed_ms = (perf_counter() - started) * 1_000.0
    linear_budget_ms = max(0, timeout_ms - ceil(build_elapsed_ms))

    merged_context = dict(context or {})
    metadata_key = "safe_slate_candidate_pool"
    if metadata_key in merged_context:
        raise ValueError(f"context key {metadata_key!r} is reserved")
    merged_context[metadata_key] = {
        "schema_version": "1.0",
        "baseline_count": len(baseline_item_ids),
        "protected_count": protected_count,
        "replaceable_position_count": len(baseline_item_ids) - protected_count,
        "candidate_count": len(candidate_item_ids),
        "operation_count": operations.operation_count,
        "compiled_conflict_count": len(operations.operation_conflicts),
        "requested_total_budget_ms": timeout_ms,
        "operation_build_elapsed_ms": build_elapsed_ms,
        "linear_guardrail_budget_ms": linear_budget_ms,
        "max_operations": max_operations,
        "max_feature_cells": max_feature_cells,
        "max_compiled_conflicts": max_compiled_conflicts,
        "matrix_digest_encoding": matrix_digest_encoding,
    }

    return guarded_linear_rerank(
        request_id=request_id,
        uncertainty_snapshot_id=uncertainty_snapshot_id,
        contrast_input_id=contrast_input_id,
        baseline_item_ids=baseline_item_ids,
        protected_count=protected_count,
        remove_positions=operations.remove_positions,
        add_item_ids=operations.add_item_ids,
        reward_deltas=operations.reward_deltas,
        safety_mean_deltas=operations.safety_mean_deltas,
        contrast_features=operations.contrast_features,
        parameter_covariance=parameter_covariance,
        baseline_metric=baseline_metric,
        degradation_tolerance=degradation_tolerance,
        max_replacements=requested_max_replacements,
        event_timestamp=event_timestamp,
        gate_assessment=gate_assessment,
        serving_mode=serving_mode,
        confidence_radius=confidence_radius,
        support_ok=support_ok,
        uncertainty_valid=uncertainty_valid,
        distribution_shift_detected=distribution_shift_detected,
        proposal_log_complete=proposal_log_complete,
        model_version=model_version,
        context=merged_context,
        remove_group_ids=operations.remove_group_ids,
        remove_group_caps=(
            dict(operations.remove_group_caps)
            if operations.remove_group_caps is not None
            else None
        ),
        add_group_ids=operations.add_group_ids,
        add_group_caps=(
            dict(operations.add_group_caps)
            if operations.add_group_caps is not None
            else None
        ),
        operation_conflicts=operations.operation_conflicts,
        beam_width=beam_width,
        max_expansions=max_expansions,
        timeout_ms=linear_budget_ms,
        finalization_reserve_ms=finalization_reserve_ms,
        matrix_digest_encoding=matrix_digest_encoding,
        custom_predicate=custom_predicate,
        custom_predicate_batch=custom_predicate_batch,
    )
