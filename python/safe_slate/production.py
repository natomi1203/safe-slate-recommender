"""Production-facing guarded re-ranking facade.

The search, safety evaluation, timeout handling, and final recheck stay in the
Rust core. This module turns serving gates and the proposal-dominating logging
contract into a small Python API.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime
from math import ceil, isfinite
from numbers import Integral, Real
from time import perf_counter
from typing import Any, Literal, Mapping, Sequence

from ._core import (
    CertifiedReplacementResult,
    optimize_replacements,
    prepare_linear_contrast_marginals_f64le,
    project_linear_contrast_marginals,
)
from .gates import ServingGateAssessment
from .uncertainty_utils import (
    MATRIX_DIGEST_F64LE_V1,
    MATRIX_DIGEST_JSON_V1,
    matrix_sha256_for_encoding,
)


@dataclass(frozen=True, slots=True)
class ServiceLatencyTrace:
    """Monotonic ingress-to-response and ingress-to-log-commit timing."""

    schema_version: str
    measurement_scope: str
    clock: str
    response_latency_ms: float
    log_commit_latency_ms: float
    candidate_generation_ms: float | None
    model_inference_ms: float | None
    queueing_ms: float | None
    rpc_ms: float | None
    guardrail_facade_ms: float | None
    serialization_ms: float | None
    proposal_log_write_ms: float | None
    complete: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ProposalLog:
    """Proposal-dominating record for one guarded re-ranking request.

    The record contains every considered operation, including operations that
    were not served. It intentionally contains no realized click or purchase;
    delayed outcomes should be joined later by ``request_id``.
    """

    schema_version: str
    request_id: str
    event_timestamp: str | None
    model_version: str | None
    serving_mode: str
    baseline_item_ids: tuple[int, ...]
    protected_count: int
    remove_positions: tuple[int, ...]
    add_item_ids: tuple[int, ...]
    reward_deltas: tuple[float, ...]
    safety_mean_deltas: tuple[float, ...]
    safety_variances: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...] | None
    selected_operation_indices: tuple[int, ...]
    selected_flags: tuple[bool, ...]
    proposed_slate: tuple[int, ...]
    served_slate: tuple[int, ...]
    baseline_metric: float
    degradation_tolerance: float
    safety_threshold: float
    confidence_radius: float
    requested_max_replacements: int
    effective_max_replacements: int
    timeout_ms: int
    support_ok: bool
    uncertainty_valid: bool
    distribution_shift_detected: bool
    proposal_log_complete: bool
    gate_evidence: Mapping[str, Any] | None
    learning_update_eligible: bool
    shadow_evaluation_eligible: bool
    gate_reason: str | None
    fallback_reason: str | None
    oracle_status: str
    oracle_stop_reason: str
    oracle_exact: bool
    k_lower: int
    k_upper: int
    cardinality_gap: int
    first_unresolved_depth: int | None
    upper_bound_source: str
    selected_reward_delta: float
    selected_safety_mean_delta: float
    selected_safety_variance: float
    selected_safety_lcb: float
    deadline_reached: bool
    rechecked: bool
    expansions: int
    incumbent_updates: int
    predicate_errors: int
    elapsed_seconds: float
    service_latency_trace: ServiceLatencyTrace | None
    context: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready dictionary without mutating the record."""

        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "event_timestamp": self.event_timestamp,
            "model_version": self.model_version,
            "serving_mode": self.serving_mode,
            "baseline_item_ids": list(self.baseline_item_ids),
            "protected_count": self.protected_count,
            "remove_positions": list(self.remove_positions),
            "add_item_ids": list(self.add_item_ids),
            "reward_deltas": list(self.reward_deltas),
            "safety_mean_deltas": list(self.safety_mean_deltas),
            "safety_variances": list(self.safety_variances),
            "covariance": (
                [list(row) for row in self.covariance]
                if self.covariance is not None
                else None
            ),
            "selected_operation_indices": list(self.selected_operation_indices),
            "selected_flags": list(self.selected_flags),
            "proposed_slate": list(self.proposed_slate),
            "served_slate": list(self.served_slate),
            "baseline_metric": self.baseline_metric,
            "degradation_tolerance": self.degradation_tolerance,
            "safety_threshold": self.safety_threshold,
            "confidence_radius": self.confidence_radius,
            "requested_max_replacements": self.requested_max_replacements,
            "effective_max_replacements": self.effective_max_replacements,
            "timeout_ms": self.timeout_ms,
            "support_ok": self.support_ok,
            "uncertainty_valid": self.uncertainty_valid,
            "distribution_shift_detected": self.distribution_shift_detected,
            "proposal_log_complete": self.proposal_log_complete,
            "gate_evidence": (
                dict(self.gate_evidence) if self.gate_evidence is not None else None
            ),
            "learning_update_eligible": self.learning_update_eligible,
            "shadow_evaluation_eligible": self.shadow_evaluation_eligible,
            "gate_reason": self.gate_reason,
            "fallback_reason": self.fallback_reason,
            "oracle_status": self.oracle_status,
            "oracle_stop_reason": self.oracle_stop_reason,
            "oracle_exact": self.oracle_exact,
            "k_lower": self.k_lower,
            "k_upper": self.k_upper,
            "cardinality_gap": self.cardinality_gap,
            "first_unresolved_depth": self.first_unresolved_depth,
            "upper_bound_source": self.upper_bound_source,
            "selected_reward_delta": self.selected_reward_delta,
            "selected_safety_mean_delta": self.selected_safety_mean_delta,
            "selected_safety_variance": self.selected_safety_variance,
            "selected_safety_lcb": self.selected_safety_lcb,
            "deadline_reached": self.deadline_reached,
            "rechecked": self.rechecked,
            "expansions": self.expansions,
            "incumbent_updates": self.incumbent_updates,
            "predicate_errors": self.predicate_errors,
            "elapsed_seconds": self.elapsed_seconds,
            "service_latency_trace": (
                self.service_latency_trace.to_dict()
                if self.service_latency_trace is not None
                else None
            ),
            "context": dict(self.context),
        }


def create_service_latency_trace(
    *,
    response_latency_ms: float,
    log_commit_latency_ms: float,
    candidate_generation_ms: float | None = None,
    model_inference_ms: float | None = None,
    queueing_ms: float | None = None,
    rpc_ms: float | None = None,
    guardrail_facade_ms: float | None = None,
    serialization_ms: float | None = None,
    proposal_log_write_ms: float | None = None,
) -> ServiceLatencyTrace:
    """Create a finite service trace without assuming stage additivity."""

    values = {
        "response_latency_ms": response_latency_ms,
        "log_commit_latency_ms": log_commit_latency_ms,
        "candidate_generation_ms": candidate_generation_ms,
        "model_inference_ms": model_inference_ms,
        "queueing_ms": queueing_ms,
        "rpc_ms": rpc_ms,
        "guardrail_facade_ms": guardrail_facade_ms,
        "serialization_ms": serialization_ms,
        "proposal_log_write_ms": proposal_log_write_ms,
    }
    normalized: dict[str, float | None] = {}
    for field, value in values.items():
        if value is None:
            normalized[field] = None
        elif (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not isfinite(value)
            or value < 0.0
        ):
            raise ValueError(f"{field} must be finite and nonnegative")
        else:
            normalized[field] = float(value)
    response = normalized["response_latency_ms"]
    log_commit = normalized["log_commit_latency_ms"]
    if response is None or log_commit is None:
        raise ValueError("response and log commit latencies are required")
    if log_commit < response:
        raise ValueError("log_commit_latency_ms may not precede response latency")
    for field, value in normalized.items():
        if field not in {"response_latency_ms", "log_commit_latency_ms"} and (
            value is not None and value > log_commit + 1e-9
        ):
            raise ValueError(f"{field} may not exceed log commit latency")
    stage_fields = (
        "candidate_generation_ms",
        "model_inference_ms",
        "queueing_ms",
        "rpc_ms",
        "guardrail_facade_ms",
        "serialization_ms",
        "proposal_log_write_ms",
    )
    return ServiceLatencyTrace(
        schema_version="1.0",
        measurement_scope="service_ingress_to_response_and_log_commit",
        clock="monotonic",
        response_latency_ms=response,
        log_commit_latency_ms=log_commit,
        candidate_generation_ms=normalized["candidate_generation_ms"],
        model_inference_ms=normalized["model_inference_ms"],
        queueing_ms=normalized["queueing_ms"],
        rpc_ms=normalized["rpc_ms"],
        guardrail_facade_ms=normalized["guardrail_facade_ms"],
        serialization_ms=normalized["serialization_ms"],
        proposal_log_write_ms=normalized["proposal_log_write_ms"],
        complete=all(normalized[field] is not None for field in stage_fields),
    )


def attach_service_latency(
    proposal_log: ProposalLog,
    trace: ServiceLatencyTrace,
) -> ProposalLog:
    """Attach one immutable post-response trace and cross-check oracle time."""

    if not isinstance(proposal_log, ProposalLog):
        raise ValueError("proposal_log must be a ProposalLog")
    if not isinstance(trace, ServiceLatencyTrace):
        raise ValueError("trace must be a ServiceLatencyTrace")
    if proposal_log.service_latency_trace is not None:
        raise ValueError("proposal log already has a service latency trace")
    oracle_ms = proposal_log.elapsed_seconds * 1_000.0
    if trace.response_latency_ms + 1e-9 < oracle_ms:
        raise ValueError("response latency may not be below Rust oracle latency")
    if (
        trace.guardrail_facade_ms is not None
        and trace.guardrail_facade_ms + 1e-9 < oracle_ms
    ):
        raise ValueError(
            "guardrail facade latency may not be below Rust oracle latency"
        )
    return replace(proposal_log, service_latency_trace=trace)


@dataclass(frozen=True, slots=True)
class GuardedRerankResult:
    """Served slate, certified frontier, and its audit log."""

    slate: tuple[int, ...]
    proposed_slate: tuple[int, ...]
    serving_mode: str
    shadow: bool
    selected_count: int
    k_lower: int
    k_upper: int
    status: str
    fallback: bool
    fallback_reason: str | None
    oracle: CertifiedReplacementResult
    proposal_log: ProposalLog


def _strict_nonnegative_int(value: object, field: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return int(value)


def _strict_bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _strict_real(value: object, field: str, *, nonnegative: bool = False) -> float:
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not isfinite(value)
        or (nonnegative and value < 0.0)
    ):
        qualifier = "finite and nonnegative" if nonnegative else "finite"
        raise ValueError(f"{field} must be {qualifier}")
    return float(value)


def _strict_int_sequence(
    values: Sequence[int],
    field: str,
    *,
    unique: bool = False,
) -> tuple[int, ...]:
    normalized = tuple(_strict_nonnegative_int(value, field) for value in values)
    if unique and len(set(normalized)) != len(normalized):
        raise ValueError(f"{field} must contain unique values")
    return normalized


def _strict_signed_int_sequence(
    values: Sequence[int],
    field: str,
) -> tuple[int, ...]:
    normalized: list[int] = []
    for value in values:
        if not isinstance(value, Integral) or isinstance(value, bool):
            raise ValueError(f"{field} must contain integers")
        normalized.append(int(value))
    return tuple(normalized)


def _strict_caps(values: Mapping[int, int], field: str) -> dict[int, int]:
    normalized: dict[int, int] = {}
    for key, value in values.items():
        if not isinstance(key, Integral) or isinstance(key, bool):
            raise ValueError(f"{field} keys must be integers")
        normalized[int(key)] = _strict_nonnegative_int(value, field)
    return normalized


def _strict_index_pairs(
    values: Sequence[tuple[int, int]],
    field: str,
) -> tuple[tuple[int, int], ...]:
    normalized: list[tuple[int, int]] = []
    for pair in values:
        if len(pair) != 2:
            raise ValueError(f"{field} entries must contain two indices")
        normalized.append(
            (
                _strict_nonnegative_int(pair[0], field),
                _strict_nonnegative_int(pair[1], field),
            )
        )
    return tuple(normalized)


def _strict_real_sequence(
    values: Sequence[float],
    field: str,
    *,
    nonnegative: bool = False,
) -> tuple[float, ...]:
    return tuple(
        _strict_real(value, field, nonnegative=nonnegative) for value in values
    )


def _strict_real_matrix(
    rows: Sequence[Sequence[float]],
    field: str,
) -> tuple[tuple[float, ...], ...]:
    return tuple(_strict_real_sequence(row, field) for row in rows)


def _gate_reason(
    *,
    support_ok: bool,
    uncertainty_valid: bool,
    distribution_shift_detected: bool,
    proposal_log_complete: bool,
) -> str | None:
    if not support_ok:
        return "support_gate"
    if distribution_shift_detected:
        return "distribution_shift_gate"
    if not uncertainty_valid:
        return "uncertainty_gate"
    if not proposal_log_complete:
        return "proposal_log_gate"
    return None


def guarded_rerank(
    *,
    request_id: str,
    event_timestamp: str | None = None,
    baseline_item_ids: Sequence[int],
    protected_count: int,
    remove_positions: Sequence[int],
    add_item_ids: Sequence[int],
    reward_deltas: Sequence[float],
    safety_mean_deltas: Sequence[float],
    safety_variances: Sequence[float],
    baseline_metric: float,
    degradation_tolerance: float,
    max_replacements: int,
    gate_assessment: ServingGateAssessment | None = None,
    serving_mode: Literal["serve", "shadow"] = "serve",
    confidence_radius: float = 1.96,
    covariance: Sequence[Sequence[float]] | None = None,
    _factorized_contrast_features: Sequence[Sequence[float]] | None = None,
    _factorized_parameter_covariance: Sequence[Sequence[float]] | None = None,
    support_ok: bool = False,
    uncertainty_valid: bool = False,
    distribution_shift_detected: bool = True,
    proposal_log_complete: bool = False,
    model_version: str | None = None,
    context: Mapping[str, Any] | None = None,
    remove_group_ids: Sequence[int] | None = None,
    remove_group_caps: Mapping[int, int] | None = None,
    add_group_ids: Sequence[int] | None = None,
    add_group_caps: Mapping[int, int] | None = None,
    operation_conflicts: Sequence[tuple[int, int]] | None = None,
    beam_width: int = 128,
    max_expansions: int = 100_000,
    timeout_ms: int = 10,
    custom_predicate: Any | None = None,
    custom_predicate_batch: Any | None = None,
) -> GuardedRerankResult:
    """Safely re-rank one baseline slate and emit a complete proposal log.

    ``safety_mean_deltas`` are operation-level expected metric changes relative
    to the unchanged baseline. The allowed degradation is translated into the
    Rust oracle threshold

    ``-degradation_tolerance * baseline_metric``.

    If ``gate_assessment`` is supplied, its evidence-derived decisions take
    precedence over the four raw gate booleans and are copied into the
    proposal log.

    In ``shadow`` mode the oracle runs normally, but ``slate`` remains the
    unchanged baseline and the counterfactual output is returned separately as
    ``proposed_slate``.

    Any failed serving gate sets the effective replacement cap to zero. The
    Rust core still materializes and rechecks the unchanged baseline, while the
    full proposal set remains available in ``proposal_log``.
    """

    if not isinstance(request_id, str) or not request_id:
        raise ValueError("request_id must be a non-empty string")
    if event_timestamp is not None:
        if not isinstance(event_timestamp, str) or not event_timestamp:
            raise ValueError("event_timestamp must be a non-empty ISO-8601 timestamp")
        normalized_timestamp = (
            event_timestamp[:-1] + "+00:00"
            if event_timestamp.endswith("Z")
            else event_timestamp
        )
        try:
            parsed_timestamp = datetime.fromisoformat(normalized_timestamp)
        except ValueError as error:
            raise ValueError("event_timestamp must be ISO-8601") from error
        if parsed_timestamp.tzinfo is None or parsed_timestamp.utcoffset() is None:
            raise ValueError("event_timestamp must include a timezone")
    max_replacements = _strict_nonnegative_int(max_replacements, "max_replacements")
    timeout_ms = _strict_nonnegative_int(timeout_ms, "timeout_ms")
    protected_count = _strict_nonnegative_int(protected_count, "protected_count")
    beam_width = _strict_nonnegative_int(beam_width, "beam_width")
    max_expansions = _strict_nonnegative_int(max_expansions, "max_expansions")
    if beam_width == 0 or max_expansions == 0:
        raise ValueError("beam_width and max_expansions must be positive integers")
    if serving_mode not in {"serve", "shadow"}:
        raise ValueError("serving_mode must be 'serve' or 'shadow'")
    if serving_mode == "shadow" and event_timestamp is None:
        raise ValueError("shadow requests require event_timestamp")
    if gate_assessment is not None and not isinstance(
        gate_assessment, ServingGateAssessment
    ):
        raise ValueError("gate_assessment must be a ServingGateAssessment")
    baseline_metric = _strict_real(baseline_metric, "baseline_metric")
    degradation_tolerance = _strict_real(
        degradation_tolerance, "degradation_tolerance"
    )
    confidence_radius = _strict_real(
        confidence_radius, "confidence_radius", nonnegative=True
    )
    if not 0.0 <= baseline_metric <= 1.0:
        raise ValueError("baseline_metric must be finite and in [0, 1]")
    if not 0.0 <= degradation_tolerance < 1.0:
        raise ValueError("degradation_tolerance must be finite and in [0, 1)")
    if model_version is not None and (
        not isinstance(model_version, str) or not model_version
    ):
        raise ValueError("model_version must be a non-empty string or None")
    if context is not None and not isinstance(context, Mapping):
        raise ValueError("context must be a mapping or None")

    baseline = _strict_int_sequence(
        baseline_item_ids, "baseline_item_ids", unique=True
    )
    removes = _strict_int_sequence(remove_positions, "remove_positions")
    additions = _strict_int_sequence(add_item_ids, "add_item_ids")
    rewards = _strict_real_sequence(reward_deltas, "reward_deltas")
    means = _strict_real_sequence(safety_mean_deltas, "safety_mean_deltas")
    variances = _strict_real_sequence(
        safety_variances, "safety_variances", nonnegative=True
    )
    covariance_rows = (
        _strict_real_matrix(covariance, "covariance")
        if covariance is not None
        else None
    )
    if protected_count > len(baseline):
        raise ValueError("protected_count cannot exceed the baseline size")
    if gate_assessment is None:
        support_ok = _strict_bool(support_ok, "support_ok")
        uncertainty_valid = _strict_bool(uncertainty_valid, "uncertainty_valid")
        distribution_shift_detected = _strict_bool(
            distribution_shift_detected, "distribution_shift_detected"
        )
        proposal_log_complete = _strict_bool(
            proposal_log_complete, "proposal_log_complete"
        )
    normalized_remove_group_ids = (
        _strict_signed_int_sequence(remove_group_ids, "remove_group_ids")
        if remove_group_ids is not None
        else None
    )
    normalized_add_group_ids = (
        _strict_signed_int_sequence(add_group_ids, "add_group_ids")
        if add_group_ids is not None
        else None
    )
    normalized_remove_group_caps = (
        _strict_caps(remove_group_caps, "remove_group_caps")
        if remove_group_caps is not None
        else None
    )
    normalized_add_group_caps = (
        _strict_caps(add_group_caps, "add_group_caps")
        if add_group_caps is not None
        else None
    )
    normalized_operation_conflicts = (
        _strict_index_pairs(operation_conflicts, "operation_conflicts")
        if operation_conflicts is not None
        else None
    )
    if gate_assessment is not None:
        support_ok = gate_assessment.support_ok
        uncertainty_valid = gate_assessment.uncertainty_valid
        distribution_shift_detected = gate_assessment.distribution_shift_detected
        proposal_log_complete = gate_assessment.proposal_log_complete
        gate_evidence: Mapping[str, Any] | None = gate_assessment.to_dict()
    else:
        gate_evidence = None
    gate_reason = _gate_reason(
        support_ok=support_ok,
        uncertainty_valid=uncertainty_valid,
        distribution_shift_detected=distribution_shift_detected,
        proposal_log_complete=proposal_log_complete,
    )
    effective_max_replacements = 0 if gate_reason is not None else max_replacements
    safety_threshold = -degradation_tolerance * baseline_metric

    oracle = optimize_replacements(
        remove_positions=list(removes),
        add_item_ids=list(additions),
        reward_deltas=list(rewards),
        safety_means=list(means),
        safety_variances=list(variances),
        baseline_size=len(baseline),
        protected_count=protected_count,
        max_replacements=effective_max_replacements,
        safety_threshold=safety_threshold,
        confidence_radius=confidence_radius,
        covariance=(
            [list(row) for row in covariance_rows]
            if covariance_rows is not None
            else None
        ),
        contrast_features=(
            [list(row) for row in _factorized_contrast_features]
            if _factorized_contrast_features is not None
            else None
        ),
        parameter_covariance=(
            [list(row) for row in _factorized_parameter_covariance]
            if _factorized_parameter_covariance is not None
            else None
        ),
        baseline_item_ids=list(baseline),
        remove_group_ids=(
            list(normalized_remove_group_ids)
            if normalized_remove_group_ids is not None
            else None
        ),
        remove_group_caps=(
            normalized_remove_group_caps
            if normalized_remove_group_caps is not None
            else None
        ),
        add_group_ids=(
            list(normalized_add_group_ids)
            if normalized_add_group_ids is not None
            else None
        ),
        add_group_caps=(
            normalized_add_group_caps if normalized_add_group_caps is not None else None
        ),
        operation_conflicts=(
            list(normalized_operation_conflicts)
            if normalized_operation_conflicts is not None
            else None
        ),
        beam_width=beam_width,
        max_expansions=max_expansions,
        timeout_ms=timeout_ms,
        custom_predicate=custom_predicate,
        custom_predicate_batch=custom_predicate_batch,
    )

    proposed_slate = tuple(oracle.materialized_slate or baseline)
    served_slate = baseline if serving_mode == "shadow" else proposed_slate
    selected = tuple(oracle.selected_operation_indices)
    selected_set = set(selected)
    selected_flags = tuple(index in selected_set for index in range(len(rewards)))
    fallback = len(selected) == 0
    if gate_reason is not None:
        fallback_reason = gate_reason
    elif fallback:
        fallback_reason = (
            oracle.stop_reason
            if oracle.stop_reason != "search_exhausted"
            else "no_certified_replacement"
        )
    else:
        fallback_reason = None

    learning_update_eligible = (
        serving_mode == "serve"
        and proposal_log_complete
        and uncertainty_valid
        and not distribution_shift_detected
        and support_ok
    )
    shadow_evaluation_eligible = (
        serving_mode == "shadow" and proposal_log_complete and oracle.rechecked
    )
    proposal_log = ProposalLog(
        schema_version="1.3",
        request_id=request_id,
        event_timestamp=event_timestamp,
        model_version=model_version,
        serving_mode=serving_mode,
        baseline_item_ids=baseline,
        protected_count=protected_count,
        remove_positions=removes,
        add_item_ids=additions,
        reward_deltas=rewards,
        safety_mean_deltas=means,
        safety_variances=variances,
        covariance=covariance_rows,
        selected_operation_indices=selected,
        selected_flags=selected_flags,
        proposed_slate=proposed_slate,
        served_slate=served_slate,
        baseline_metric=baseline_metric,
        degradation_tolerance=degradation_tolerance,
        safety_threshold=safety_threshold,
        confidence_radius=confidence_radius,
        requested_max_replacements=max_replacements,
        effective_max_replacements=effective_max_replacements,
        timeout_ms=timeout_ms,
        support_ok=support_ok,
        uncertainty_valid=uncertainty_valid,
        distribution_shift_detected=distribution_shift_detected,
        proposal_log_complete=proposal_log_complete,
        gate_evidence=gate_evidence,
        learning_update_eligible=learning_update_eligible,
        shadow_evaluation_eligible=shadow_evaluation_eligible,
        gate_reason=gate_reason,
        fallback_reason=fallback_reason,
        oracle_status=oracle.status,
        oracle_stop_reason=oracle.stop_reason,
        oracle_exact=oracle.exact,
        k_lower=oracle.k_lower,
        k_upper=oracle.k_upper,
        cardinality_gap=oracle.cardinality_gap,
        first_unresolved_depth=oracle.first_unresolved_depth,
        upper_bound_source=oracle.upper_bound_source,
        selected_reward_delta=oracle.reward,
        selected_safety_mean_delta=oracle.safety_mean,
        selected_safety_variance=oracle.safety_variance,
        selected_safety_lcb=oracle.safety_lcb,
        deadline_reached=oracle.deadline_reached,
        rechecked=oracle.rechecked,
        expansions=oracle.expansions,
        incumbent_updates=oracle.incumbent_updates,
        predicate_errors=oracle.predicate_errors,
        elapsed_seconds=oracle.elapsed_seconds,
        service_latency_trace=None,
        context=dict(context or {}),
    )
    return GuardedRerankResult(
        slate=served_slate,
        proposed_slate=proposed_slate,
        serving_mode=serving_mode,
        shadow=serving_mode == "shadow",
        selected_count=len(selected),
        k_lower=oracle.k_lower,
        k_upper=oracle.k_upper,
        status=oracle.status,
        fallback=fallback,
        fallback_reason=fallback_reason,
        oracle=oracle,
        proposal_log=proposal_log,
    )


def guarded_linear_rerank(
    *,
    request_id: str,
    uncertainty_snapshot_id: str,
    contrast_input_id: str | None = None,
    baseline_item_ids: Sequence[int],
    protected_count: int,
    remove_positions: Sequence[int],
    add_item_ids: Sequence[int],
    reward_deltas: Sequence[float],
    safety_mean_deltas: Sequence[float],
    contrast_features: Sequence[Sequence[float]],
    parameter_covariance: Sequence[Sequence[float]],
    baseline_metric: float,
    degradation_tolerance: float,
    max_replacements: int,
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
    remove_group_ids: Sequence[int] | None = None,
    remove_group_caps: Mapping[int, int] | None = None,
    add_group_ids: Sequence[int] | None = None,
    add_group_caps: Mapping[int, int] | None = None,
    operation_conflicts: Sequence[tuple[int, int]] | None = None,
    beam_width: int = 128,
    max_expansions: int = 100_000,
    timeout_ms: int = 10,
    finalization_reserve_ms: int = 2,
    matrix_digest_encoding: Literal["json-v1", "f64le-v1"] = MATRIX_DIGEST_JSON_V1,
    custom_predicate: Any | None = None,
    custom_predicate_batch: Any | None = None,
) -> GuardedRerankResult:
    """Project shared linear uncertainty and run the guarded re-ranker.

    Each row of ``contrast_features`` is the feature contrast for one atomic
    replacement. Rust computes ``Z parameter_covariance Z^T`` so correlations
    induced by a shared fitted model are retained when a slate is certified.
    ``uncertainty_snapshot_id`` must identify the immutable model/covariance
    snapshot used for this request; its matrices are digested into the
    proposal log context without copying the full parameter covariance there.
    ``contrast_input_id`` identifies the immutable external payload containing
    the request contrast matrix. Serving may omit it, but confirmatory shadow
    evidence rejects an unbound factorized log.
    ``matrix_digest_encoding`` names the canonical byte representation used by
    both logged matrix digests. ``json-v1`` preserves the original external
    contract; ``f64le-v1`` is a faster shape-bound IEEE-754 encoding for
    latency-sensitive paths. Immutable manifests must hash the same encoding.

    The projection does not establish statistical coverage by itself. Callers
    must set ``uncertainty_valid`` (or supply ``gate_assessment``) only after
    the confidence method and snapshot have passed their external validity
    checks. Otherwise the regular serving gate returns the unchanged baseline.
    ``timeout_ms`` is a cooperative budget for projection, oracle search, and
    Python result materialization. The remaining whole milliseconds after
    projection, input digesting, and ``finalization_reserve_ms`` are passed to
    Rust. Final rechecks are mandatory and may finish after that budget, so
    end-to-end service deadlines still need an external timeout and the
    measured service-latency trace.
    """

    if not isinstance(uncertainty_snapshot_id, str) or not uncertainty_snapshot_id:
        raise ValueError("uncertainty_snapshot_id must be a non-empty string")
    if contrast_input_id is not None and (
        not isinstance(contrast_input_id, str) or not contrast_input_id
    ):
        raise ValueError("contrast_input_id must be a non-empty string or None")
    if not isinstance(timeout_ms, int) or isinstance(timeout_ms, bool) or timeout_ms < 0:
        raise ValueError("timeout_ms must be a nonnegative integer")
    if (
        not isinstance(finalization_reserve_ms, int)
        or isinstance(finalization_reserve_ms, bool)
        or finalization_reserve_ms < 0
    ):
        raise ValueError("finalization_reserve_ms must be a nonnegative integer")
    guardrail_started = perf_counter()
    if matrix_digest_encoding == MATRIX_DIGEST_F64LE_V1:
        prepared = prepare_linear_contrast_marginals_f64le(
            contrast_features=contrast_features,
            parameter_covariance=parameter_covariance,
        )
        safety_variances = prepared.safety_variances
        contrast_digest = prepared.contrast_features_sha256
        parameter_covariance_digest = prepared.parameter_covariance_sha256
    else:
        safety_variances = project_linear_contrast_marginals(
            contrast_features=contrast_features,
            parameter_covariance=parameter_covariance,
        )
        contrast_digest = matrix_sha256_for_encoding(
            contrast_features, matrix_digest_encoding
        )
        parameter_covariance_digest = matrix_sha256_for_encoding(
            parameter_covariance, matrix_digest_encoding
        )
    merged_context = dict(context or {})
    metadata_key = "safe_slate_uncertainty"
    if metadata_key in merged_context:
        raise ValueError(f"context key {metadata_key!r} is reserved")
    pre_oracle_elapsed_ms = (perf_counter() - guardrail_started) * 1_000.0
    oracle_budget_ms = max(
        0,
        timeout_ms - ceil(pre_oracle_elapsed_ms) - finalization_reserve_ms,
    )
    merged_context[metadata_key] = {
        "method": "linear_contrast_covariance",
        "snapshot_id": uncertainty_snapshot_id,
        "contrast_input_id": contrast_input_id,
        "feature_dimension": len(parameter_covariance),
        "operation_count": len(contrast_features),
        "representation": "factorized",
        "matrix_digest_encoding": matrix_digest_encoding,
        "contrast_features_sha256": contrast_digest,
        "parameter_covariance_sha256": parameter_covariance_digest,
        "requested_guardrail_budget_ms": timeout_ms,
        "finalization_reserve_ms": finalization_reserve_ms,
        "pre_oracle_elapsed_ms": pre_oracle_elapsed_ms,
        "oracle_budget_ms": oracle_budget_ms,
    }
    return guarded_rerank(
        request_id=request_id,
        event_timestamp=event_timestamp,
        baseline_item_ids=baseline_item_ids,
        protected_count=protected_count,
        remove_positions=remove_positions,
        add_item_ids=add_item_ids,
        reward_deltas=reward_deltas,
        safety_mean_deltas=safety_mean_deltas,
        safety_variances=safety_variances,
        covariance=None,
        _factorized_contrast_features=contrast_features,
        _factorized_parameter_covariance=parameter_covariance,
        baseline_metric=baseline_metric,
        degradation_tolerance=degradation_tolerance,
        max_replacements=max_replacements,
        gate_assessment=gate_assessment,
        serving_mode=serving_mode,
        confidence_radius=confidence_radius,
        support_ok=support_ok,
        uncertainty_valid=uncertainty_valid,
        distribution_shift_detected=distribution_shift_detected,
        proposal_log_complete=proposal_log_complete,
        model_version=model_version,
        context=merged_context,
        remove_group_ids=remove_group_ids,
        remove_group_caps=remove_group_caps,
        add_group_ids=add_group_ids,
        add_group_caps=add_group_caps,
        operation_conflicts=operation_conflicts,
        beam_width=beam_width,
        max_expansions=max_expansions,
        timeout_ms=oracle_budget_ms,
        custom_predicate=custom_predicate,
        custom_predicate_batch=custom_predicate_batch,
    )
