"""Aggregate operability checks for read-only shadow re-ranking runs."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import json
from math import isfinite
from numbers import Integral, Real
from typing import Any, Sequence

from .production import ProposalLog, ServiceLatencyTrace, create_service_latency_trace


@dataclass(frozen=True, slots=True)
class ShadowRunSummary:
    """Operational evidence from shadow logs, without causal claims."""

    request_count: int
    duplicate_request_id_count: int
    log_data_digest: str
    window_start: str
    window_end: str
    event_time_min: str | None
    event_time_max: str | None
    oracle_latency_ms_p50: float | None
    oracle_latency_ms_p95: float | None
    oracle_latency_ms_p99: float | None
    maximum_oracle_p95_ms: float
    service_response_latency_ms_p50: float | None
    service_response_latency_ms_p95: float | None
    service_response_latency_ms_p99: float | None
    log_commit_latency_ms_p50: float | None
    log_commit_latency_ms_p95: float | None
    log_commit_latency_ms_p99: float | None
    require_service_latency_trace: bool
    maximum_service_response_p95_ms: float | None
    maximum_log_commit_p95_ms: float | None
    service_latency_trace_rate: float
    complete_service_latency_trace_rate: float
    fallback_rate: float
    maximum_fallback_rate: float
    gate_fallback_rate: float
    deadline_rate: float
    exact_rate: float
    recheck_failure_rate: float
    maximum_recheck_failure_rate: float
    invalid_output_rate: float
    maximum_invalid_output_rate: float
    certificate_violation_rate: float
    predicate_error_request_rate: float
    maximum_predicate_error_request_rate: float
    mean_selected_count: float
    mean_cardinality_gap: float
    safety_margin_p05: float | None
    minimum_safety_margin: float | None
    status_counts: tuple[tuple[str, int], ...]
    stop_reason_counts: tuple[tuple[str, int], ...]
    fallback_reason_counts: tuple[tuple[str, int], ...]
    invalid_reason_counts: tuple[tuple[str, int], ...]
    operability_pass: bool
    shadow_service_latency_evidence_valid: bool
    production_latency_claim_authorized: bool
    causal_lift_claim_authorized: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_count": self.request_count,
            "duplicate_request_id_count": self.duplicate_request_id_count,
            "log_data_digest": self.log_data_digest,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "event_time_min": self.event_time_min,
            "event_time_max": self.event_time_max,
            "oracle_latency_ms_p50": self.oracle_latency_ms_p50,
            "oracle_latency_ms_p95": self.oracle_latency_ms_p95,
            "oracle_latency_ms_p99": self.oracle_latency_ms_p99,
            "maximum_oracle_p95_ms": self.maximum_oracle_p95_ms,
            "service_response_latency_ms_p50": self.service_response_latency_ms_p50,
            "service_response_latency_ms_p95": self.service_response_latency_ms_p95,
            "service_response_latency_ms_p99": self.service_response_latency_ms_p99,
            "log_commit_latency_ms_p50": self.log_commit_latency_ms_p50,
            "log_commit_latency_ms_p95": self.log_commit_latency_ms_p95,
            "log_commit_latency_ms_p99": self.log_commit_latency_ms_p99,
            "require_service_latency_trace": self.require_service_latency_trace,
            "maximum_service_response_p95_ms": (self.maximum_service_response_p95_ms),
            "maximum_log_commit_p95_ms": self.maximum_log_commit_p95_ms,
            "service_latency_trace_rate": self.service_latency_trace_rate,
            "complete_service_latency_trace_rate": (
                self.complete_service_latency_trace_rate
            ),
            "fallback_rate": self.fallback_rate,
            "maximum_fallback_rate": self.maximum_fallback_rate,
            "gate_fallback_rate": self.gate_fallback_rate,
            "deadline_rate": self.deadline_rate,
            "exact_rate": self.exact_rate,
            "recheck_failure_rate": self.recheck_failure_rate,
            "maximum_recheck_failure_rate": self.maximum_recheck_failure_rate,
            "invalid_output_rate": self.invalid_output_rate,
            "maximum_invalid_output_rate": self.maximum_invalid_output_rate,
            "certificate_violation_rate": self.certificate_violation_rate,
            "predicate_error_request_rate": self.predicate_error_request_rate,
            "maximum_predicate_error_request_rate": (
                self.maximum_predicate_error_request_rate
            ),
            "mean_selected_count": self.mean_selected_count,
            "mean_cardinality_gap": self.mean_cardinality_gap,
            "safety_margin_p05": self.safety_margin_p05,
            "minimum_safety_margin": self.minimum_safety_margin,
            "status_counts": dict(self.status_counts),
            "stop_reason_counts": dict(self.stop_reason_counts),
            "fallback_reason_counts": dict(self.fallback_reason_counts),
            "invalid_reason_counts": dict(self.invalid_reason_counts),
            "operability_pass": self.operability_pass,
            "shadow_service_latency_evidence_valid": (
                self.shadow_service_latency_evidence_valid
            ),
            "production_latency_claim_authorized": (
                self.production_latency_claim_authorized
            ),
            "causal_lift_claim_authorized": self.causal_lift_claim_authorized,
            "reasons": list(self.reasons),
        }


def _is_finite_real(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and isfinite(value)


def _json_safe(value: object) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, Real):
        return _is_finite_real(value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _json_safe(item) for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return all(_json_safe(item) for item in value)
    return False


def _digest_value(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Real):
        numeric = float(value)
        if isfinite(numeric):
            return numeric
        if numeric != numeric:
            return "__nonfinite__:nan"
        return (
            "__nonfinite__:positive_infinity"
            if numeric > 0
            else "__nonfinite__:negative_infinity"
        )
    if isinstance(value, dict):
        return {
            str(key): _digest_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_digest_value(item) for item in value]
    return f"__unsupported__:{type(value).__qualname__}"


def _log_data_digest(records: Sequence[ProposalLog]) -> str:
    encoded = json.dumps(
        _digest_value([record.to_dict() for record in records]),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _parse_event_time(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed


def _service_trace_valid(log: ProposalLog, *, require_complete: bool) -> bool:
    trace = log.service_latency_trace
    if trace is None or not isinstance(trace, ServiceLatencyTrace):
        return False
    try:
        recreated = create_service_latency_trace(
            response_latency_ms=trace.response_latency_ms,
            log_commit_latency_ms=trace.log_commit_latency_ms,
            candidate_generation_ms=trace.candidate_generation_ms,
            model_inference_ms=trace.model_inference_ms,
            queueing_ms=trace.queueing_ms,
            rpc_ms=trace.rpc_ms,
            guardrail_facade_ms=trace.guardrail_facade_ms,
            serialization_ms=trace.serialization_ms,
            proposal_log_write_ms=trace.proposal_log_write_ms,
        )
    except ValueError:
        return False
    oracle_ms = log.elapsed_seconds * 1_000.0
    return (
        recreated == trace
        and (not require_complete or trace.complete)
        and trace.response_latency_ms + 1e-9 >= oracle_ms
        and (
            trace.guardrail_facade_ms is None
            or trace.guardrail_facade_ms + 1e-9 >= oracle_ms
        )
    )


def _record_invalid_reasons(
    log: ProposalLog,
    *,
    window_start: datetime,
    window_end: datetime,
    require_service_latency_trace: bool,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if log.schema_version != "1.3":
        reasons.append("unsupported_schema")
    event_time = _parse_event_time(log.event_timestamp)
    if event_time is None:
        reasons.append("event_timestamp_missing_or_invalid")
    elif not window_start <= event_time < window_end:
        reasons.append("event_outside_frozen_window")
    if not _json_safe(log.to_dict()):
        reasons.append("proposal_log_not_canonical_json")
    if log.service_latency_trace is None:
        if require_service_latency_trace:
            reasons.append("service_latency_trace_missing")
    elif not _service_trace_valid(log, require_complete=require_service_latency_trace):
        reasons.append("service_latency_trace_invalid")
    if log.serving_mode != "shadow":
        reasons.append("non_shadow_record")
    if log.served_slate != log.baseline_item_ids:
        reasons.append("shadow_served_slate_changed")
    if not log.proposal_log_complete:
        reasons.append("proposal_log_incomplete")
    if not log.shadow_evaluation_eligible:
        reasons.append("shadow_evaluation_not_eligible")
    if not log.rechecked:
        reasons.append("oracle_recheck_failure")
    if (
        not isinstance(log.protected_count, Integral)
        or isinstance(log.protected_count, bool)
        or not 0 <= log.protected_count <= len(log.baseline_item_ids)
    ):
        reasons.append("protected_count_invalid")
    selected_indices_valid = all(
        isinstance(index, Integral)
        and not isinstance(index, bool)
        and 0 <= index < len(log.add_item_ids)
        for index in log.selected_operation_indices
    )
    if not selected_indices_valid or len(set(log.selected_operation_indices)) != len(
        log.selected_operation_indices
    ):
        reasons.append("selected_operation_indices_invalid")
    if len(log.selected_operation_indices) != log.k_lower:
        reasons.append("selected_count_k_lower_mismatch")
    if log.k_upper < log.k_lower or log.cardinality_gap != log.k_upper - log.k_lower:
        reasons.append("cardinality_frontier_inconsistent")
    if len(log.selected_flags) != len(log.add_item_ids):
        reasons.append("selected_flag_length_mismatch")
    else:
        selected = set(log.selected_operation_indices)
        expected = tuple(index in selected for index in range(len(log.add_item_ids)))
        if log.selected_flags != expected:
            reasons.append("selected_flags_inconsistent")
    if len(log.proposed_slate) != len(log.baseline_item_ids):
        reasons.append("proposed_slate_length_mismatch")
    elif (
        log.proposed_slate[: log.protected_count]
        != log.baseline_item_ids[: log.protected_count]
    ):
        reasons.append("protected_core_changed")
    if len(set(log.proposed_slate)) != len(log.proposed_slate):
        reasons.append("proposed_slate_contains_duplicates")
    if not _is_finite_real(log.elapsed_seconds) or log.elapsed_seconds < 0.0:
        reasons.append("oracle_latency_invalid")
    if not _is_finite_real(log.selected_safety_lcb) or not _is_finite_real(
        log.safety_threshold
    ):
        reasons.append("certificate_value_invalid")
    elif log.selected_safety_lcb < log.safety_threshold - 1e-12:
        reasons.append("certificate_violation")
    return tuple(dict.fromkeys(reasons))


def summarize_shadow_run(
    logs: Sequence[ProposalLog],
    *,
    window_start: str,
    window_end: str,
    minimum_requests: int,
    maximum_oracle_p95_ms: float,
    maximum_fallback_rate: float,
    maximum_invalid_output_rate: float = 0.0,
    maximum_recheck_failure_rate: float = 0.0,
    maximum_predicate_error_request_rate: float = 0.0,
    require_service_latency_trace: bool = False,
    maximum_service_response_p95_ms: float | None = None,
    maximum_log_commit_p95_ms: float | None = None,
) -> ShadowRunSummary:
    """Summarize a frozen read-only shadow run and enforce operability gates.

    Latency is the Rust oracle's process-local elapsed time recorded in each
    proposal log. It excludes network, model inference, serialization, and
    queueing, so even a passing summary does not establish production latency.
    The summary never authorizes a causal purchase-lift claim.
    """

    if (
        not isinstance(minimum_requests, Integral)
        or isinstance(minimum_requests, bool)
        or minimum_requests <= 0
    ):
        raise ValueError("minimum_requests must be a positive integer")
    window_start_time = _parse_event_time(window_start)
    window_end_time = _parse_event_time(window_end)
    if window_start_time is None or window_end_time is None:
        raise ValueError("shadow window boundaries must be timezone-aware ISO-8601")
    if window_start_time >= window_end_time:
        raise ValueError("shadow window must be non-empty and ordered")
    if not _is_finite_real(maximum_oracle_p95_ms) or maximum_oracle_p95_ms <= 0:
        raise ValueError("maximum_oracle_p95_ms must be finite and positive")
    if not isinstance(require_service_latency_trace, bool):
        raise ValueError("require_service_latency_trace must be boolean")
    service_thresholds = {
        "maximum_service_response_p95_ms": maximum_service_response_p95_ms,
        "maximum_log_commit_p95_ms": maximum_log_commit_p95_ms,
    }
    if require_service_latency_trace and any(
        value is None for value in service_thresholds.values()
    ):
        raise ValueError("required service latency traces need both p95 thresholds")
    if not require_service_latency_trace and any(
        value is not None for value in service_thresholds.values()
    ):
        raise ValueError("service p95 thresholds require service latency traces")
    for name, value in service_thresholds.items():
        if value is not None and (not _is_finite_real(value) or value <= 0.0):
            raise ValueError(f"{name} must be finite and positive")
    rate_thresholds = {
        "maximum_fallback_rate": maximum_fallback_rate,
        "maximum_invalid_output_rate": maximum_invalid_output_rate,
        "maximum_recheck_failure_rate": maximum_recheck_failure_rate,
        "maximum_predicate_error_request_rate": (maximum_predicate_error_request_rate),
    }
    for name, value in rate_thresholds.items():
        if not _is_finite_real(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1]")

    records = tuple(logs)
    if not records:
        raise ValueError("shadow summary requires at least one proposal log")
    if any(not isinstance(log, ProposalLog) for log in records):
        raise ValueError("logs must contain only ProposalLog records")

    request_count = len(records)
    reasons: list[str] = []
    request_ids = [log.request_id for log in records]
    duplicate_request_id_count = request_count - len(set(request_ids))
    if duplicate_request_id_count:
        reasons.append("duplicate_request_id")

    invalid_reason_counts: Counter[str] = Counter()
    invalid_records = 0
    certificate_violation_count = 0
    latencies_ms: list[float] = []
    safety_margins: list[float] = []
    event_times: list[datetime] = []
    response_latencies_ms: list[float] = []
    log_commit_latencies_ms: list[float] = []
    trace_count = 0
    complete_trace_count = 0
    for log in records:
        record_reasons = _record_invalid_reasons(
            log,
            window_start=window_start_time,
            window_end=window_end_time,
            require_service_latency_trace=require_service_latency_trace,
        )
        if record_reasons:
            invalid_records += 1
            invalid_reason_counts.update(record_reasons)
        if "certificate_violation" in record_reasons:
            certificate_violation_count += 1
        if _is_finite_real(log.elapsed_seconds) and log.elapsed_seconds >= 0.0:
            latencies_ms.append(float(log.elapsed_seconds) * 1_000.0)
        if _is_finite_real(log.selected_safety_lcb) and _is_finite_real(
            log.safety_threshold
        ):
            safety_margins.append(
                float(log.selected_safety_lcb) - float(log.safety_threshold)
            )
        event_time = _parse_event_time(log.event_timestamp)
        if event_time is not None:
            event_times.append(event_time)
        trace = log.service_latency_trace
        if trace is not None:
            trace_count += 1
            if trace.complete:
                complete_trace_count += 1
            if _service_trace_valid(
                log, require_complete=require_service_latency_trace
            ):
                response_latencies_ms.append(trace.response_latency_ms)
                log_commit_latencies_ms.append(trace.log_commit_latency_ms)

    fallback_count = sum(not log.selected_operation_indices for log in records)
    gate_fallback_count = sum(log.gate_reason is not None for log in records)
    deadline_count = sum(log.deadline_reached for log in records)
    exact_count = sum(log.oracle_exact for log in records)
    recheck_failure_count = sum(not log.rechecked for log in records)
    predicate_error_count = sum(log.predicate_errors > 0 for log in records)

    fallback_rate = fallback_count / request_count
    gate_fallback_rate = gate_fallback_count / request_count
    deadline_rate = deadline_count / request_count
    exact_rate = exact_count / request_count
    recheck_failure_rate = recheck_failure_count / request_count
    invalid_output_rate = invalid_records / request_count
    certificate_violation_rate = certificate_violation_count / request_count
    predicate_error_rate = predicate_error_count / request_count
    p50 = _quantile(latencies_ms, 0.50)
    p95 = _quantile(latencies_ms, 0.95)
    p99 = _quantile(latencies_ms, 0.99)
    service_response_p50 = _quantile(response_latencies_ms, 0.50)
    service_response_p95 = _quantile(response_latencies_ms, 0.95)
    service_response_p99 = _quantile(response_latencies_ms, 0.99)
    log_commit_p50 = _quantile(log_commit_latencies_ms, 0.50)
    log_commit_p95 = _quantile(log_commit_latencies_ms, 0.95)
    log_commit_p99 = _quantile(log_commit_latencies_ms, 0.99)

    if request_count < minimum_requests:
        reasons.append("request_count_below_gate")
    if p95 is None or p95 > maximum_oracle_p95_ms:
        reasons.append("oracle_p95_latency_above_gate")
    if fallback_rate > maximum_fallback_rate:
        reasons.append("fallback_rate_above_gate")
    if invalid_output_rate > maximum_invalid_output_rate:
        reasons.append("invalid_output_rate_above_gate")
    if recheck_failure_rate > maximum_recheck_failure_rate:
        reasons.append("recheck_failure_rate_above_gate")
    if certificate_violation_rate > 0.0:
        reasons.append("certificate_violation_observed")
    if predicate_error_rate > maximum_predicate_error_request_rate:
        reasons.append("predicate_error_rate_above_gate")
    if require_service_latency_trace and (
        service_response_p95 is None
        or service_response_p95 > float(maximum_service_response_p95_ms)
    ):
        reasons.append("service_response_p95_latency_above_gate")
    if require_service_latency_trace and (
        log_commit_p95 is None or log_commit_p95 > float(maximum_log_commit_p95_ms)
    ):
        reasons.append("log_commit_p95_latency_above_gate")

    status_counts = Counter(log.oracle_status for log in records)
    stop_reason_counts = Counter(log.oracle_stop_reason for log in records)
    fallback_reason_counts = Counter(
        log.fallback_reason for log in records if log.fallback_reason is not None
    )
    return ShadowRunSummary(
        request_count=request_count,
        duplicate_request_id_count=duplicate_request_id_count,
        log_data_digest=_log_data_digest(records),
        window_start=window_start_time.isoformat(),
        window_end=window_end_time.isoformat(),
        event_time_min=(min(event_times).isoformat() if event_times else None),
        event_time_max=(max(event_times).isoformat() if event_times else None),
        oracle_latency_ms_p50=p50,
        oracle_latency_ms_p95=p95,
        oracle_latency_ms_p99=p99,
        maximum_oracle_p95_ms=float(maximum_oracle_p95_ms),
        service_response_latency_ms_p50=service_response_p50,
        service_response_latency_ms_p95=service_response_p95,
        service_response_latency_ms_p99=service_response_p99,
        log_commit_latency_ms_p50=log_commit_p50,
        log_commit_latency_ms_p95=log_commit_p95,
        log_commit_latency_ms_p99=log_commit_p99,
        require_service_latency_trace=require_service_latency_trace,
        maximum_service_response_p95_ms=(
            float(maximum_service_response_p95_ms)
            if maximum_service_response_p95_ms is not None
            else None
        ),
        maximum_log_commit_p95_ms=(
            float(maximum_log_commit_p95_ms)
            if maximum_log_commit_p95_ms is not None
            else None
        ),
        service_latency_trace_rate=trace_count / request_count,
        complete_service_latency_trace_rate=complete_trace_count / request_count,
        fallback_rate=fallback_rate,
        maximum_fallback_rate=float(maximum_fallback_rate),
        gate_fallback_rate=gate_fallback_rate,
        deadline_rate=deadline_rate,
        exact_rate=exact_rate,
        recheck_failure_rate=recheck_failure_rate,
        maximum_recheck_failure_rate=float(maximum_recheck_failure_rate),
        invalid_output_rate=invalid_output_rate,
        maximum_invalid_output_rate=float(maximum_invalid_output_rate),
        certificate_violation_rate=certificate_violation_rate,
        predicate_error_request_rate=predicate_error_rate,
        maximum_predicate_error_request_rate=float(
            maximum_predicate_error_request_rate
        ),
        mean_selected_count=(
            sum(len(log.selected_operation_indices) for log in records) / request_count
        ),
        mean_cardinality_gap=(
            sum(log.cardinality_gap for log in records) / request_count
        ),
        safety_margin_p05=_quantile(safety_margins, 0.05),
        minimum_safety_margin=min(safety_margins, default=None),
        status_counts=tuple(sorted(status_counts.items())),
        stop_reason_counts=tuple(sorted(stop_reason_counts.items())),
        fallback_reason_counts=tuple(sorted(fallback_reason_counts.items())),
        invalid_reason_counts=tuple(sorted(invalid_reason_counts.items())),
        operability_pass=not reasons,
        shadow_service_latency_evidence_valid=(
            require_service_latency_trace
            and invalid_output_rate == 0.0
            and service_response_p95 is not None
            and service_response_p95 <= float(maximum_service_response_p95_ms)
            and log_commit_p95 is not None
            and log_commit_p95 <= float(maximum_log_commit_p95_ms)
        ),
        production_latency_claim_authorized=False,
        causal_lift_claim_authorized=False,
        reasons=tuple(dict.fromkeys(reasons)),
    )
