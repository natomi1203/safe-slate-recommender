"""Bounded, strict loading for persisted proposal-dominating logs."""

from __future__ import annotations

from datetime import datetime
import json
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping, Sequence

from .production import (
    ProposalLog,
    ServiceLatencyTrace,
    create_service_latency_trace,
)


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _optional_string(value: object, field: str) -> str | None:
    if value is None:
        return None
    return _string(value, field)


def _integer(value: object, field: str, *, nonnegative: bool = True) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    result = int(value)
    if nonnegative and result < 0:
        raise ValueError(f"{field} must be nonnegative")
    return result


def _optional_integer(value: object, field: str) -> int | None:
    if value is None:
        return None
    return _integer(value, field)


def _real(value: object, field: str, *, nonnegative: bool = False) -> float:
    if not isinstance(value, Real) or isinstance(value, bool) or not isfinite(value):
        raise ValueError(f"{field} must be finite numeric")
    result = float(value)
    if nonnegative and result < 0.0:
        raise ValueError(f"{field} must be nonnegative")
    return result


def _integers(value: object, field: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be an integer array")
    return tuple(_integer(item, field, nonnegative=False) for item in value)


def _reals(value: object, field: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be a numeric array")
    return tuple(_real(item, field) for item in value)


def _booleans(value: object, field: str) -> tuple[bool, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be a boolean array")
    if any(not isinstance(item, bool) for item in value):
        raise ValueError(f"{field} must contain only booleans")
    return tuple(value)


def _json_safe(value: object) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, Real):
        return not isinstance(value, bool) and isfinite(value)
    if isinstance(value, Mapping):
        return all(
            isinstance(key, str) and _json_safe(item) for key, item in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return all(_json_safe(item) for item in value)
    return False


def _mapping(value: object, field: str, *, optional: bool = False):
    if value is None and optional:
        return None
    if not isinstance(value, Mapping) or not _json_safe(value):
        raise ValueError(f"{field} must be a finite JSON object")
    return dict(value)


def _event_timestamp(value: object) -> str | None:
    if value is None:
        return None
    timestamp = _string(value, "event_timestamp")
    normalized = timestamp[:-1] + "+00:00" if timestamp.endswith("Z") else timestamp
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError("event_timestamp must be ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("event_timestamp must include a timezone")
    return timestamp


def _service_latency_trace(value: object) -> ServiceLatencyTrace | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != set(
        ServiceLatencyTrace.__dataclass_fields__
    ):
        raise ValueError("service_latency_trace fields do not match canonical schema")
    if value.get("schema_version") != "1.0":
        raise ValueError("unsupported service latency trace schema")
    if (
        value.get("measurement_scope") != ("service_ingress_to_response_and_log_commit")
        or value.get("clock") != "monotonic"
    ):
        raise ValueError("service latency trace scope or clock is invalid")
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
    if trace.to_dict() != dict(value):
        raise ValueError("service latency trace completeness is noncanonical")
    return trace


def load_proposal_log(payload: Mapping[str, Any]) -> ProposalLog:
    """Strictly reconstruct one canonical proposal-log schema 1.3 record."""

    if not isinstance(payload, Mapping):
        raise ValueError("proposal log payload must be a mapping")
    if set(payload) != set(ProposalLog.__dataclass_fields__):
        raise ValueError("proposal log fields do not match canonical schema")
    schema_version = _string(payload["schema_version"], "schema_version")
    if schema_version != "1.3":
        raise ValueError("unsupported proposal log schema")
    serving_mode = _string(payload["serving_mode"], "serving_mode")
    if serving_mode not in {"serve", "shadow"}:
        raise ValueError("serving_mode must be serve or shadow")

    covariance_payload = payload["covariance"]
    covariance = None
    if covariance_payload is not None:
        if not isinstance(covariance_payload, Sequence) or isinstance(
            covariance_payload, (str, bytes)
        ):
            raise ValueError("covariance must be a matrix or None")
        covariance = tuple(_reals(row, "covariance") for row in covariance_payload)

    boolean_fields = (
        "support_ok",
        "uncertainty_valid",
        "distribution_shift_detected",
        "proposal_log_complete",
        "learning_update_eligible",
        "shadow_evaluation_eligible",
        "oracle_exact",
        "deadline_reached",
        "rechecked",
    )
    booleans: dict[str, bool] = {}
    for field in boolean_fields:
        value = payload[field]
        if not isinstance(value, bool):
            raise ValueError(f"{field} must be boolean")
        booleans[field] = value

    record = ProposalLog(
        schema_version=schema_version,
        request_id=_string(payload["request_id"], "request_id"),
        event_timestamp=_event_timestamp(payload["event_timestamp"]),
        model_version=_optional_string(payload["model_version"], "model_version"),
        serving_mode=serving_mode,
        baseline_item_ids=_integers(payload["baseline_item_ids"], "baseline_item_ids"),
        protected_count=_integer(payload["protected_count"], "protected_count"),
        remove_positions=_integers(payload["remove_positions"], "remove_positions"),
        add_item_ids=_integers(payload["add_item_ids"], "add_item_ids"),
        reward_deltas=_reals(payload["reward_deltas"], "reward_deltas"),
        safety_mean_deltas=_reals(payload["safety_mean_deltas"], "safety_mean_deltas"),
        safety_variances=_reals(payload["safety_variances"], "safety_variances"),
        covariance=covariance,
        selected_operation_indices=_integers(
            payload["selected_operation_indices"], "selected_operation_indices"
        ),
        selected_flags=_booleans(payload["selected_flags"], "selected_flags"),
        proposed_slate=_integers(payload["proposed_slate"], "proposed_slate"),
        served_slate=_integers(payload["served_slate"], "served_slate"),
        baseline_metric=_real(payload["baseline_metric"], "baseline_metric"),
        degradation_tolerance=_real(
            payload["degradation_tolerance"], "degradation_tolerance"
        ),
        safety_threshold=_real(payload["safety_threshold"], "safety_threshold"),
        confidence_radius=_real(
            payload["confidence_radius"], "confidence_radius", nonnegative=True
        ),
        requested_max_replacements=_integer(
            payload["requested_max_replacements"], "requested_max_replacements"
        ),
        effective_max_replacements=_integer(
            payload["effective_max_replacements"], "effective_max_replacements"
        ),
        timeout_ms=_integer(payload["timeout_ms"], "timeout_ms"),
        gate_evidence=_mapping(
            payload["gate_evidence"], "gate_evidence", optional=True
        ),
        gate_reason=_optional_string(payload["gate_reason"], "gate_reason"),
        fallback_reason=_optional_string(payload["fallback_reason"], "fallback_reason"),
        oracle_status=_string(payload["oracle_status"], "oracle_status"),
        oracle_stop_reason=_string(payload["oracle_stop_reason"], "oracle_stop_reason"),
        k_lower=_integer(payload["k_lower"], "k_lower"),
        k_upper=_integer(payload["k_upper"], "k_upper"),
        cardinality_gap=_integer(payload["cardinality_gap"], "cardinality_gap"),
        first_unresolved_depth=_optional_integer(
            payload["first_unresolved_depth"], "first_unresolved_depth"
        ),
        upper_bound_source=_string(payload["upper_bound_source"], "upper_bound_source"),
        selected_reward_delta=_real(
            payload["selected_reward_delta"], "selected_reward_delta"
        ),
        selected_safety_mean_delta=_real(
            payload["selected_safety_mean_delta"], "selected_safety_mean_delta"
        ),
        selected_safety_variance=_real(
            payload["selected_safety_variance"],
            "selected_safety_variance",
            nonnegative=True,
        ),
        selected_safety_lcb=_real(
            payload["selected_safety_lcb"], "selected_safety_lcb"
        ),
        expansions=_integer(payload["expansions"], "expansions"),
        incumbent_updates=_integer(payload["incumbent_updates"], "incumbent_updates"),
        predicate_errors=_integer(payload["predicate_errors"], "predicate_errors"),
        elapsed_seconds=_real(
            payload["elapsed_seconds"], "elapsed_seconds", nonnegative=True
        ),
        service_latency_trace=_service_latency_trace(payload["service_latency_trace"]),
        context=_mapping(payload["context"], "context"),
        **booleans,
    )
    operation_count = len(record.add_item_ids)
    operation_arrays = {
        "remove_positions": record.remove_positions,
        "reward_deltas": record.reward_deltas,
        "safety_mean_deltas": record.safety_mean_deltas,
        "safety_variances": record.safety_variances,
        "selected_flags": record.selected_flags,
    }
    if any(len(values) != operation_count for values in operation_arrays.values()):
        raise ValueError("proposal operation arrays have inconsistent lengths")
    if record.covariance is not None and (
        len(record.covariance) != operation_count
        or any(len(row) != operation_count for row in record.covariance)
    ):
        raise ValueError("covariance shape does not match proposal operations")
    if len(record.proposed_slate) != len(record.baseline_item_ids) or len(
        record.served_slate
    ) != len(record.baseline_item_ids):
        raise ValueError("slate lengths do not match the baseline")
    if any(
        index < 0 or index >= operation_count
        for index in record.selected_operation_indices
    ) or len(set(record.selected_operation_indices)) != len(
        record.selected_operation_indices
    ):
        raise ValueError("selected operation indices are invalid")
    selected = set(record.selected_operation_indices)
    if record.selected_flags != tuple(
        index in selected for index in range(operation_count)
    ):
        raise ValueError("selected flags disagree with selected operation indices")
    if not 0.0 <= record.baseline_metric <= 1.0:
        raise ValueError("baseline_metric must be in [0, 1]")
    if not 0.0 <= record.degradation_tolerance < 1.0:
        raise ValueError("degradation_tolerance must be in [0, 1)")
    if any(value < 0.0 for value in record.safety_variances):
        raise ValueError("safety variances must be nonnegative")
    if serving_mode == "shadow" and record.event_timestamp is None:
        raise ValueError("shadow proposal logs require event_timestamp")
    return record


def load_proposal_logs_jsonl(
    path: str | Path,
    *,
    maximum_records: int,
    maximum_bytes: int,
) -> tuple[ProposalLog, ...]:
    """Read a bounded UTF-8 JSONL file without modifying the source."""

    if (
        not isinstance(maximum_records, Integral)
        or isinstance(maximum_records, bool)
        or maximum_records <= 0
    ):
        raise ValueError("maximum_records must be a positive integer")
    if (
        not isinstance(maximum_bytes, Integral)
        or isinstance(maximum_bytes, bool)
        or maximum_bytes <= 0
    ):
        raise ValueError("maximum_bytes must be a positive integer")
    source = Path(path).resolve(strict=True)
    if not source.is_file():
        raise ValueError("proposal log source must be a regular file")
    if source.stat().st_size > maximum_bytes:
        raise ValueError("proposal log source exceeds maximum_bytes")
    records: list[ProposalLog] = []
    try:
        with source.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    raise ValueError(f"blank proposal log line: {line_number}")
                if len(records) >= maximum_records:
                    raise ValueError("proposal log source exceeds maximum_records")
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"invalid proposal log JSON at line {line_number}"
                    ) from error
                records.append(load_proposal_log(payload))
    except UnicodeDecodeError as error:
        raise ValueError("proposal log source must be UTF-8") from error
    if not records:
        raise ValueError("proposal log source is empty")
    return tuple(records)
