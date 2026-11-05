"""Immutable post-commit service traces bound to proposal-log bytes."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from hashlib import sha256
import json
from math import ceil, isfinite
from numbers import Integral
from pathlib import Path
import re
from time import perf_counter
from typing import Any, Callable, Iterator, Mapping, Sequence

from .production import (
    ProposalLog,
    ServiceLatencyTrace,
    attach_service_latency,
    create_service_latency_trace,
)


TRACE_SCHEMA_VERSION = "safe-slate-service-trace-completion-1.0"
REGISTERED_SERVICE_STAGES = (
    "candidate_generation",
    "model_inference",
    "queueing",
    "rpc",
    "guardrail_facade",
    "serialization",
    "proposal_log_write",
)
TRACE_KEYS = {
    "schema_version",
    "trace_id",
    "request_id",
    "event_timestamp",
    "completed_at",
    "trace_source",
    "proposal_log_sha256",
    "service_latency_trace",
}


@dataclass(frozen=True, slots=True)
class ServiceTraceCompletion:
    """Post-commit sidecar for one already durable proposal record.

    ``log_commit_latency_ms`` measures the primary proposal-log commit. This
    completion record is written afterwards, avoiding the impossible cycle of
    placing a log's own future commit time inside the same immutable record.
    """

    schema_version: str
    trace_id: str
    request_id: str
    event_timestamp: str
    completed_at: str
    trace_source: str
    proposal_log_sha256: str
    service_latency_trace: ServiceLatencyTrace

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["service_latency_trace"] = self.service_latency_trace.to_dict()
        return value


@dataclass(frozen=True, slots=True)
class ProductionServiceTraceSummary:
    """Observed-window service trace audit, distinct from a universal claim."""

    schema_version: str
    serving_mode: str
    window_start: str
    window_end: str
    request_count: int
    trace_source_counts: tuple[tuple[str, int], ...]
    invalid_record_count: int
    invalid_record_rate: float
    invalid_reason_counts: tuple[tuple[str, int], ...]
    fallback_rate: float
    deadline_rate: float
    recheck_failure_rate: float
    certificate_violation_rate: float
    latency_ms: Mapping[str, Mapping[str, float]]
    observed_window_latency_evidence_valid: bool
    production_latency_claim_authorized: bool
    production_safety_claim_authorized: bool
    causal_lift_claim_authorized: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["trace_source_counts"] = dict(self.trace_source_counts)
        value["invalid_reason_counts"] = dict(self.invalid_reason_counts)
        value["latency_ms"] = {
            stage: dict(summary) for stage, summary in self.latency_ms.items()
        }
        value["reasons"] = list(self.reasons)
        return value


class ServiceTraceRecorder:
    """Per-request monotonic recorder for a real service integration.

    The recorder is intentionally mutable and request-scoped. It is not
    thread-safe and must not be shared across requests. Remote or already
    measured spans can be supplied with ``record_stage_ms``; local work can be
    wrapped by ``measure``. Finalization is impossible until every registered
    stage, response completion, and primary proposal-log commit are observed.
    """

    def __init__(
        self,
        *,
        trace_id: str,
        event_timestamp: str,
        trace_source: str = "opentelemetry",
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        if not callable(clock):
            raise ValueError("clock must be callable")
        self.trace_id = _trace_id(trace_id)
        self.event_timestamp = _aware_timestamp(
            event_timestamp, "event_timestamp"
        )[0]
        self.trace_source = _nonempty(trace_source, "trace_source")
        self._clock = clock
        started = clock()
        if not isinstance(started, (int, float)) or not isfinite(started):
            raise ValueError("clock must return finite monotonic seconds")
        self._started = float(started)
        self._last_clock = self._started
        self._stages: dict[str, float] = {}
        self._active_stages: dict[str, float] = {}
        self._response_latency_ms: float | None = None
        self._log_commit_latency_ms: float | None = None
        self._completed = False

    def _now(self) -> float:
        value = self._clock()
        if not isinstance(value, (int, float)) or not isfinite(value):
            raise ValueError("clock must return finite monotonic seconds")
        result = float(value)
        if result < self._last_clock:
            raise ValueError("clock moved backwards")
        self._last_clock = result
        return result

    @staticmethod
    def _stage_name(stage: object) -> str:
        if stage not in REGISTERED_SERVICE_STAGES:
            raise ValueError(f"unknown service stage: {stage}")
        return str(stage)

    def record_stage_ms(self, stage: str, duration_ms: float) -> None:
        """Record one externally measured finite stage duration exactly once."""

        name = self._stage_name(stage)
        if self._completed:
            raise ValueError("service trace recorder is already completed")
        if name in self._stages or name in self._active_stages:
            raise ValueError(f"service stage already recorded: {name}")
        if (
            not isinstance(duration_ms, (int, float))
            or isinstance(duration_ms, bool)
            or not isfinite(duration_ms)
            or duration_ms < 0.0
        ):
            raise ValueError("stage duration must be finite and nonnegative")
        self._stages[name] = float(duration_ms)

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        """Measure one local stage with the recorder's monotonic clock."""

        name = self._stage_name(stage)
        if self._completed:
            raise ValueError("service trace recorder is already completed")
        if name in self._stages or name in self._active_stages:
            raise ValueError(f"service stage already recorded: {name}")
        started = self._now()
        self._active_stages[name] = started
        try:
            yield
        finally:
            finished = self._now()
            self._active_stages.pop(name, None)
            self._stages[name] = (finished - started) * 1_000.0

    def mark_response(self) -> float:
        """Record ingress-to-response duration after response serialization."""

        if self._completed:
            raise ValueError("service trace recorder is already completed")
        if self._response_latency_ms is not None:
            raise ValueError("response was already marked")
        if self._active_stages:
            raise ValueError("response may not be marked with active stages")
        self._response_latency_ms = (self._now() - self._started) * 1_000.0
        return self._response_latency_ms

    def mark_primary_log_commit(self) -> float:
        """Record ingress-to-durable-primary-log commit duration."""

        if self._completed:
            raise ValueError("service trace recorder is already completed")
        if self._response_latency_ms is None:
            raise ValueError("response must be marked before primary log commit")
        if self._log_commit_latency_ms is not None:
            raise ValueError("primary log commit was already marked")
        if self._active_stages:
            raise ValueError("log commit may not be marked with active stages")
        if "proposal_log_write" not in self._stages:
            raise ValueError("proposal_log_write must be recorded before commit")
        self._log_commit_latency_ms = (self._now() - self._started) * 1_000.0
        if self._log_commit_latency_ms < self._response_latency_ms:
            raise ValueError("primary log commit may not precede response")
        return self._log_commit_latency_ms

    def complete(
        self,
        proposal_log: ProposalLog,
        *,
        completed_at: str,
    ) -> ServiceTraceCompletion:
        """Build the post-commit sidecar after every required observation."""

        if self._completed:
            raise ValueError("service trace recorder is already completed")
        if self._active_stages:
            raise ValueError("service trace contains active stages")
        missing = sorted(set(REGISTERED_SERVICE_STAGES) - set(self._stages))
        if missing:
            raise ValueError(f"service trace stages are missing: {missing}")
        if self._response_latency_ms is None:
            raise ValueError("response has not been marked")
        if self._log_commit_latency_ms is None:
            raise ValueError("primary log commit has not been marked")
        if proposal_log.event_timestamp != self.event_timestamp:
            raise ValueError("proposal event timestamp differs from recorder")
        trace = create_service_latency_trace(
            response_latency_ms=self._response_latency_ms,
            log_commit_latency_ms=self._log_commit_latency_ms,
            candidate_generation_ms=self._stages["candidate_generation"],
            model_inference_ms=self._stages["model_inference"],
            queueing_ms=self._stages["queueing"],
            rpc_ms=self._stages["rpc"],
            guardrail_facade_ms=self._stages["guardrail_facade"],
            serialization_ms=self._stages["serialization"],
            proposal_log_write_ms=self._stages["proposal_log_write"],
        )
        completion = create_service_trace_completion(
            proposal_log,
            trace_id=self.trace_id,
            completed_at=completed_at,
            trace_source=self.trace_source,
            service_latency_trace=trace,
        )
        self._completed = True
        return completion


def _aware_timestamp(value: object, field: str) -> tuple[str, datetime]:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty ISO-8601 timestamp")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError(f"{field} must be ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone")
    return value, parsed


def _trace_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[0-9a-f]{32}", value) is None
        or value == "0" * 32
    ):
        raise ValueError("trace_id must be a nonzero lowercase W3C trace id")
    return value


def _nonempty(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def canonical_proposal_log_bytes(proposal_log: ProposalLog) -> bytes:
    """Return canonical bytes for an unattached primary proposal record."""

    if not isinstance(proposal_log, ProposalLog):
        raise ValueError("proposal_log must be a ProposalLog")
    if proposal_log.service_latency_trace is not None:
        raise ValueError("primary proposal log must not already contain a service trace")
    return json.dumps(
        proposal_log.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def proposal_log_sha256(proposal_log: ProposalLog) -> str:
    """Digest the exact canonical primary record referenced by a sidecar."""

    return sha256(canonical_proposal_log_bytes(proposal_log)).hexdigest()


def create_service_trace_completion(
    proposal_log: ProposalLog,
    *,
    trace_id: str,
    completed_at: str,
    trace_source: str,
    service_latency_trace: ServiceLatencyTrace,
) -> ServiceTraceCompletion:
    """Create one complete sidecar after the primary log commit succeeds."""

    if not isinstance(service_latency_trace, ServiceLatencyTrace):
        raise ValueError("service_latency_trace must be a ServiceLatencyTrace")
    if not service_latency_trace.complete:
        raise ValueError("service trace completion requires every registered stage")
    if proposal_log.event_timestamp is None:
        raise ValueError("proposal log requires an event_timestamp for service tracing")
    event_timestamp, event_time = _aware_timestamp(
        proposal_log.event_timestamp, "event_timestamp"
    )
    completed_value, completion_time = _aware_timestamp(completed_at, "completed_at")
    if completion_time < event_time:
        raise ValueError("completed_at may not precede event_timestamp")
    # Reuse the serving-path cross-check that response/facade durations cover
    # the measured Rust oracle duration.
    attach_service_latency(proposal_log, service_latency_trace)
    return ServiceTraceCompletion(
        schema_version=TRACE_SCHEMA_VERSION,
        trace_id=_trace_id(trace_id),
        request_id=_nonempty(proposal_log.request_id, "request_id"),
        event_timestamp=event_timestamp,
        completed_at=completed_value,
        trace_source=_nonempty(trace_source, "trace_source"),
        proposal_log_sha256=proposal_log_sha256(proposal_log),
        service_latency_trace=service_latency_trace,
    )


def _load_latency_trace(value: object) -> ServiceLatencyTrace:
    if not isinstance(value, Mapping) or set(value) != set(
        ServiceLatencyTrace.__dataclass_fields__
    ):
        raise ValueError("service_latency_trace fields do not match canonical schema")
    trace = create_service_latency_trace(
        response_latency_ms=value["response_latency_ms"],
        log_commit_latency_ms=value["log_commit_latency_ms"],
        candidate_generation_ms=value["candidate_generation_ms"],
        model_inference_ms=value["model_inference_ms"],
        queueing_ms=value["queueing_ms"],
        rpc_ms=value["rpc_ms"],
        guardrail_facade_ms=value["guardrail_facade_ms"],
        serialization_ms=value["serialization_ms"],
        proposal_log_write_ms=value["proposal_log_write_ms"],
    )
    if trace.to_dict() != dict(value) or not trace.complete:
        raise ValueError("service trace completion must be complete and canonical")
    return trace


def load_service_trace_completion(
    payload: Mapping[str, Any],
) -> ServiceTraceCompletion:
    """Strictly reconstruct one completion sidecar."""

    if not isinstance(payload, Mapping) or set(payload) != TRACE_KEYS:
        raise ValueError("service trace completion fields do not match schema")
    if payload["schema_version"] != TRACE_SCHEMA_VERSION:
        raise ValueError("unsupported service trace completion schema")
    event_value, event_time = _aware_timestamp(
        payload["event_timestamp"], "event_timestamp"
    )
    completed_value, completion_time = _aware_timestamp(
        payload["completed_at"], "completed_at"
    )
    if completion_time < event_time:
        raise ValueError("completed_at may not precede event_timestamp")
    return ServiceTraceCompletion(
        schema_version=TRACE_SCHEMA_VERSION,
        trace_id=_trace_id(payload["trace_id"]),
        request_id=_nonempty(payload["request_id"], "request_id"),
        event_timestamp=event_value,
        completed_at=completed_value,
        trace_source=_nonempty(payload["trace_source"], "trace_source"),
        proposal_log_sha256=_sha256(
            payload["proposal_log_sha256"], "proposal_log_sha256"
        ),
        service_latency_trace=_load_latency_trace(payload["service_latency_trace"]),
    )


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value is forbidden: {value}")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_service_trace_completions_jsonl(
    path: str | Path,
    *,
    maximum_records: int,
    maximum_bytes: int,
) -> tuple[ServiceTraceCompletion, ...]:
    """Read a bounded sidecar stream without changing its source."""

    for name, value in (
        ("maximum_records", maximum_records),
        ("maximum_bytes", maximum_bytes),
    ):
        if not isinstance(value, Integral) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    source = Path(path).resolve(strict=True)
    if not source.is_file():
        raise ValueError("service trace source must be a regular file")
    if source.stat().st_size > maximum_bytes:
        raise ValueError("service trace source exceeds maximum_bytes")
    records: list[ServiceTraceCompletion] = []
    try:
        with source.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    raise ValueError(f"blank service trace line: {line_number}")
                if len(records) >= maximum_records:
                    raise ValueError("service trace source exceeds maximum_records")
                try:
                    payload = json.loads(
                        line,
                        object_pairs_hook=_reject_duplicate_pairs,
                        parse_constant=_reject_constant,
                    )
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"invalid service trace JSON at line {line_number}"
                    ) from error
                records.append(load_service_trace_completion(payload))
    except UnicodeDecodeError as error:
        raise ValueError("service trace source must be UTF-8") from error
    if not records:
        raise ValueError("service trace source is empty")
    return tuple(records)


def join_service_trace_completions(
    proposal_logs: tuple[ProposalLog, ...] | list[ProposalLog],
    completions: tuple[ServiceTraceCompletion, ...] | list[ServiceTraceCompletion],
) -> tuple[ProposalLog, ...]:
    """Require an exact one-to-one digest-bound join and attach all traces."""

    logs = tuple(proposal_logs)
    sidecars = tuple(completions)
    if not logs:
        raise ValueError("proposal_logs must be nonempty")
    if any(not isinstance(log, ProposalLog) for log in logs):
        raise ValueError("proposal_logs contains an invalid record")
    if any(not isinstance(value, ServiceTraceCompletion) for value in sidecars):
        raise ValueError("completions contains an invalid record")
    log_ids = [log.request_id for log in logs]
    completion_ids = [value.request_id for value in sidecars]
    trace_ids = [value.trace_id for value in sidecars]
    if len(log_ids) != len(set(log_ids)):
        raise ValueError("proposal logs contain duplicate request ids")
    if len(completion_ids) != len(set(completion_ids)):
        raise ValueError("service traces contain duplicate request ids")
    if len(trace_ids) != len(set(trace_ids)):
        raise ValueError("service traces contain duplicate trace ids")
    if set(log_ids) != set(completion_ids):
        missing = sorted(set(log_ids) - set(completion_ids))
        extra = sorted(set(completion_ids) - set(log_ids))
        raise ValueError(
            f"service trace join is incomplete; missing={missing}, extra={extra}"
        )
    by_request = {value.request_id: value for value in sidecars}
    attached: list[ProposalLog] = []
    for log in logs:
        completion = by_request[log.request_id]
        if log.event_timestamp != completion.event_timestamp:
            raise ValueError("service trace event timestamp mismatch")
        if proposal_log_sha256(log) != completion.proposal_log_sha256:
            raise ValueError("service trace proposal-log digest mismatch")
        attached.append(attach_service_latency(log, completion.service_latency_trace))
    return tuple(attached)


def _percentiles(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("latency summary requires at least one value")
    ordered = sorted(values)

    def at(probability: float) -> float:
        return ordered[max(0, ceil(probability * len(ordered)) - 1)]

    return {
        "p50": at(0.50),
        "p95": at(0.95),
        "p99": at(0.99),
        "max": ordered[-1],
    }


def _finite_rate(value: object, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not isfinite(value)
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return float(value)


def summarize_production_service_run(
    proposal_logs: Sequence[ProposalLog],
    completions: Sequence[ServiceTraceCompletion],
    *,
    window_start: str,
    window_end: str,
    minimum_requests: int,
    maximum_response_p95_ms: float,
    maximum_log_commit_p95_ms: float,
    maximum_fallback_rate: float,
    maximum_invalid_record_rate: float = 0.0,
    maximum_recheck_failure_rate: float = 0.0,
    serving_mode: str = "serve",
) -> ProductionServiceTraceSummary:
    """Audit one frozen service window with complete post-commit traces.

    A passing result authorizes only a descriptive statement about this exact
    observed window and environment. It does not establish counterfactual
    business lift, future safety, or a universal production-latency guarantee.
    """

    if serving_mode not in {"serve", "shadow"}:
        raise ValueError("serving_mode must be serve or shadow")
    start_value, start_time = _aware_timestamp(window_start, "window_start")
    end_value, end_time = _aware_timestamp(window_end, "window_end")
    if start_time >= end_time:
        raise ValueError("service trace window must be nonempty and ordered")
    if (
        not isinstance(minimum_requests, Integral)
        or isinstance(minimum_requests, bool)
        or minimum_requests <= 0
    ):
        raise ValueError("minimum_requests must be a positive integer")
    for name, value in (
        ("maximum_response_p95_ms", maximum_response_p95_ms),
        ("maximum_log_commit_p95_ms", maximum_log_commit_p95_ms),
    ):
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not isfinite(value)
            or value <= 0.0
        ):
            raise ValueError(f"{name} must be finite and positive")
    fallback_gate = _finite_rate(maximum_fallback_rate, "maximum_fallback_rate")
    invalid_gate = _finite_rate(
        maximum_invalid_record_rate, "maximum_invalid_record_rate"
    )
    recheck_gate = _finite_rate(
        maximum_recheck_failure_rate, "maximum_recheck_failure_rate"
    )

    logs = tuple(proposal_logs)
    sidecars = tuple(completions)
    attached = join_service_trace_completions(logs, sidecars)
    by_request = {value.request_id: value for value in sidecars}
    invalid_counts: Counter[str] = Counter()
    trace_sources: Counter[str] = Counter()
    stage_values: dict[str, list[float]] = {
        "response": [],
        "log_commit": [],
        "candidate_generation": [],
        "model_inference": [],
        "queueing": [],
        "rpc": [],
        "guardrail_facade": [],
        "serialization": [],
        "proposal_log_write": [],
    }
    invalid_records = 0
    certificate_violations = 0
    for log in attached:
        record_reasons: list[str] = []
        completion = by_request[log.request_id]
        trace_sources[completion.trace_source] += 1
        _, event_time = _aware_timestamp(log.event_timestamp, "event_timestamp")
        if not start_time <= event_time < end_time:
            record_reasons.append("event_outside_frozen_window")
        if log.serving_mode != serving_mode:
            record_reasons.append("serving_mode_mismatch")
        if tuple(log.served_slate[: log.protected_count]) != tuple(
            log.baseline_item_ids[: log.protected_count]
        ):
            record_reasons.append("protected_core_changed")
        if serving_mode == "serve" and log.served_slate != log.proposed_slate:
            record_reasons.append("served_slate_differs_from_proposal")
        if serving_mode == "shadow" and log.served_slate != log.baseline_item_ids:
            record_reasons.append("shadow_changed_served_slate")
        if not log.rechecked:
            record_reasons.append("final_recheck_failed")
        if log.selected_safety_lcb + 1e-12 < log.safety_threshold:
            record_reasons.append("certificate_violation")
            certificate_violations += 1
        if log.predicate_errors:
            record_reasons.append("predicate_error")
        trace = log.service_latency_trace
        if trace is None or not trace.complete:
            record_reasons.append("complete_service_trace_missing")
        else:
            stage_values["response"].append(trace.response_latency_ms)
            stage_values["log_commit"].append(trace.log_commit_latency_ms)
            stage_values["candidate_generation"].append(
                float(trace.candidate_generation_ms)
            )
            stage_values["model_inference"].append(float(trace.model_inference_ms))
            stage_values["queueing"].append(float(trace.queueing_ms))
            stage_values["rpc"].append(float(trace.rpc_ms))
            stage_values["guardrail_facade"].append(float(trace.guardrail_facade_ms))
            stage_values["serialization"].append(float(trace.serialization_ms))
            stage_values["proposal_log_write"].append(
                float(trace.proposal_log_write_ms)
            )
        if record_reasons:
            invalid_records += 1
            invalid_counts.update(record_reasons)

    request_count = len(attached)
    fallback_rate = (
        sum(not log.selected_operation_indices for log in attached) / request_count
    )
    deadline_rate = sum(log.deadline_reached for log in attached) / request_count
    recheck_failure_rate = sum(not log.rechecked for log in attached) / request_count
    certificate_violation_rate = certificate_violations / request_count
    invalid_rate = invalid_records / request_count
    latency_summary = {
        stage: _percentiles(values) for stage, values in stage_values.items()
    }
    reasons: list[str] = []
    if request_count < minimum_requests:
        reasons.append("request_count_below_gate")
    if invalid_rate > invalid_gate:
        reasons.append("invalid_record_rate_above_gate")
    if fallback_rate > fallback_gate:
        reasons.append("fallback_rate_above_gate")
    if recheck_failure_rate > recheck_gate:
        reasons.append("recheck_failure_rate_above_gate")
    if certificate_violation_rate > 0.0:
        reasons.append("certificate_violation_observed")
    if latency_summary["response"]["p95"] > maximum_response_p95_ms:
        reasons.append("response_p95_above_gate")
    if latency_summary["log_commit"]["p95"] > maximum_log_commit_p95_ms:
        reasons.append("log_commit_p95_above_gate")
    unique_reasons = tuple(dict.fromkeys(reasons))
    return ProductionServiceTraceSummary(
        schema_version="1.0",
        serving_mode=serving_mode,
        window_start=start_value,
        window_end=end_value,
        request_count=request_count,
        trace_source_counts=tuple(sorted(trace_sources.items())),
        invalid_record_count=invalid_records,
        invalid_record_rate=invalid_rate,
        invalid_reason_counts=tuple(sorted(invalid_counts.items())),
        fallback_rate=fallback_rate,
        deadline_rate=deadline_rate,
        recheck_failure_rate=recheck_failure_rate,
        certificate_violation_rate=certificate_violation_rate,
        latency_ms=latency_summary,
        observed_window_latency_evidence_valid=not unique_reasons,
        production_latency_claim_authorized=False,
        production_safety_claim_authorized=False,
        causal_lift_claim_authorized=False,
        reasons=unique_reasons,
    )
