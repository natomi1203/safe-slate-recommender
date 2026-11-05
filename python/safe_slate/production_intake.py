"""Strict intake for the production gRPC server-envelope sidecar.

The serving service emits this record only after gRPC reports an outgoing
payload.  It closes the process-local RPC envelope, but it deliberately does
not manufacture upstream queueing, vendor-internal stages, pure serialization,
or a downstream warehouse-commit acknowledgement.
"""

from __future__ import annotations

import calendar
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Mapping


PRODUCTION_GRPC_ENVELOPE_SCHEMA_VERSION = (
    "safe-slate-production-post-response-1.1"
)
PRODUCTION_GRPC_ENVELOPE_SCOPE = "grpc_begin_to_payload_sent"
LIST_PREDICTED_PRODUCTS_METHOD = (
    "/x_asia.api.gb.product_recommendation.v1."
    "ProductRecommendationService/ListPredictedProducts"
)

_TRACE_ID = re.compile(r"[0-9a-f]{32}\Z")
_UTC_TIMESTAMP = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2})T"
    r"(?P<time>\d{2}:\d{2}:\d{2})"
    r"(?:\.(?P<fraction>\d{1,9}))?Z\Z"
)

MISSING_FULL_SERVICE_STAGES = (
    "upstream_queueing_before_grpc_begin",
    "vendor_aggregate_rpc",
    "vendor_internal_candidate_generation",
    "vendor_internal_model_inference",
    "safe_slate_guardrail",
    "pure_response_serialization",
    "proposal_broker_publish",
    "durable_proposal_log_commit",
)


@dataclass(frozen=True, slots=True)
class ObservedGRPCEnvelope:
    """Canonical process-local timing emitted by the production Go service."""

    schema_version: str
    request_id: str
    trace_id: str
    full_method_name: str
    measurement_scope: str
    rpc_begin_at: str
    handler_started_at: str
    response_ready_at: str
    payload_sent_at: str
    rpc_begin_to_handler_started_nanoseconds: int
    handler_started_to_response_ready_nanoseconds: int
    response_ready_to_payload_sent_nanoseconds: int
    rpc_begin_to_payload_sent_nanoseconds: int
    payload_length_bytes: int
    compressed_payload_length_bytes: int
    wire_payload_length_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ProductionTraceCoverage:
    """What the accepted sidecars do and do not support scientifically."""

    record_count: int
    request_ids_unique: bool
    trace_ids_unique: bool
    process_local_server_envelope_complete: bool
    full_service_trace_complete: bool
    missing_full_service_stages: tuple[str, ...]
    evidence_state: str

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["missing_full_service_stages"] = list(
            self.missing_full_service_stages
        )
        return result


def _nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _nonnegative_integer(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def _utc_timestamp_nanoseconds(value: object, field: str) -> int:
    text = _nonempty_string(value, field)
    match = _UTC_TIMESTAMP.fullmatch(text)
    if match is None:
        raise ValueError(f"{field} must be canonical RFC3339 UTC with Z")
    fraction = match.group("fraction")
    if fraction is not None and fraction.endswith("0"):
        raise ValueError(f"{field} has a noncanonical trailing fractional zero")
    try:
        whole = datetime.strptime(
            f"{match.group('date')}T{match.group('time')}",
            "%Y-%m-%dT%H:%M:%S",
        ).replace(tzinfo=timezone.utc)
    except ValueError as error:
        raise ValueError(f"{field} is not a valid UTC timestamp") from error
    seconds = calendar.timegm(whole.utctimetuple())
    fractional_nanoseconds = int((fraction or "0").ljust(9, "0"))
    return seconds * 1_000_000_000 + fractional_nanoseconds


def load_observed_grpc_envelope(
    payload: Mapping[str, Any],
) -> ObservedGRPCEnvelope:
    """Validate one exact Go schema 1.1 record without broadening its claim."""

    if not isinstance(payload, Mapping):
        raise ValueError("production gRPC envelope must be a mapping")
    expected = set(ObservedGRPCEnvelope.__dataclass_fields__)
    if set(payload) != expected:
        raise ValueError("production gRPC envelope fields do not match schema")
    if payload["schema_version"] != PRODUCTION_GRPC_ENVELOPE_SCHEMA_VERSION:
        raise ValueError("unsupported production gRPC envelope schema")
    if payload["measurement_scope"] != PRODUCTION_GRPC_ENVELOPE_SCOPE:
        raise ValueError("production gRPC envelope measurement scope is invalid")
    if payload["full_method_name"] != LIST_PREDICTED_PRODUCTS_METHOD:
        raise ValueError("production gRPC envelope method is invalid")

    request_id = _nonempty_string(payload["request_id"], "request_id")
    trace_id = _nonempty_string(payload["trace_id"], "trace_id")
    if _TRACE_ID.fullmatch(trace_id) is None or trace_id == "0" * 32:
        raise ValueError("trace_id must be a nonzero lowercase W3C trace ID")

    timestamp_fields = (
        "rpc_begin_at",
        "handler_started_at",
        "response_ready_at",
        "payload_sent_at",
    )
    timestamps = {
        field: _utc_timestamp_nanoseconds(payload[field], field)
        for field in timestamp_fields
    }
    if list(timestamps.values()) != sorted(timestamps.values()):
        raise ValueError("production gRPC envelope timestamps are not monotonic")

    duration_fields = (
        "rpc_begin_to_handler_started_nanoseconds",
        "handler_started_to_response_ready_nanoseconds",
        "response_ready_to_payload_sent_nanoseconds",
        "rpc_begin_to_payload_sent_nanoseconds",
    )
    durations = {
        field: _nonnegative_integer(payload[field], field)
        for field in duration_fields
    }
    expected_durations = {
        "rpc_begin_to_handler_started_nanoseconds": (
            timestamps["handler_started_at"] - timestamps["rpc_begin_at"]
        ),
        "handler_started_to_response_ready_nanoseconds": (
            timestamps["response_ready_at"] - timestamps["handler_started_at"]
        ),
        "response_ready_to_payload_sent_nanoseconds": (
            timestamps["payload_sent_at"] - timestamps["response_ready_at"]
        ),
        "rpc_begin_to_payload_sent_nanoseconds": (
            timestamps["payload_sent_at"] - timestamps["rpc_begin_at"]
        ),
    }
    if durations != expected_durations:
        raise ValueError("production gRPC envelope durations do not match timestamps")

    length_fields = (
        "payload_length_bytes",
        "compressed_payload_length_bytes",
        "wire_payload_length_bytes",
    )
    lengths = {
        field: _nonnegative_integer(payload[field], field)
        for field in length_fields
    }
    if lengths["wire_payload_length_bytes"] < lengths[
        "compressed_payload_length_bytes"
    ]:
        raise ValueError("wire payload length must include compressed payload")

    return ObservedGRPCEnvelope(
        schema_version=PRODUCTION_GRPC_ENVELOPE_SCHEMA_VERSION,
        request_id=request_id,
        trace_id=trace_id,
        full_method_name=LIST_PREDICTED_PRODUCTS_METHOD,
        measurement_scope=PRODUCTION_GRPC_ENVELOPE_SCOPE,
        rpc_begin_at=payload["rpc_begin_at"],
        handler_started_at=payload["handler_started_at"],
        response_ready_at=payload["response_ready_at"],
        payload_sent_at=payload["payload_sent_at"],
        **durations,
        **lengths,
    )


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def load_observed_grpc_envelopes_jsonl(
    path: str | Path,
    *,
    maximum_records: int,
    maximum_bytes: int,
) -> tuple[ObservedGRPCEnvelope, ...]:
    """Load a bounded immutable export; never infer missing production rows."""

    source = Path(path)
    if maximum_records <= 0 or maximum_bytes <= 0:
        raise ValueError("maximum_records and maximum_bytes must be positive")
    size = source.stat().st_size
    if size > maximum_bytes:
        raise ValueError("production gRPC envelope input exceeds maximum_bytes")
    records: list[ObservedGRPCEnvelope] = []
    with source.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL record at line {line_number}")
            try:
                payload = json.loads(
                    line,
                    object_pairs_hook=_reject_duplicate_pairs,
                    parse_constant=_reject_nonfinite_constant,
                )
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(
                    f"invalid production gRPC envelope JSON at line {line_number}"
                ) from error
            records.append(load_observed_grpc_envelope(payload))
            if len(records) > maximum_records:
                raise ValueError(
                    "production gRPC envelope input exceeds maximum_records"
                )
    return tuple(records)


def audit_observed_grpc_envelopes(
    records: tuple[ObservedGRPCEnvelope, ...] | list[ObservedGRPCEnvelope],
) -> ProductionTraceCoverage:
    """Classify coverage without promoting a partial envelope to full trace."""

    if any(not isinstance(record, ObservedGRPCEnvelope) for record in records):
        raise ValueError("records must contain only ObservedGRPCEnvelope values")
    request_ids = [record.request_id for record in records]
    trace_ids = [record.trace_id for record in records]
    request_ids_unique = len(request_ids) == len(set(request_ids))
    trace_ids_unique = len(trace_ids) == len(set(trace_ids))
    if not request_ids_unique or not trace_ids_unique:
        raise ValueError("production gRPC envelope identities must be one-to-one")
    return ProductionTraceCoverage(
        record_count=len(records),
        request_ids_unique=request_ids_unique,
        trace_ids_unique=trace_ids_unique,
        process_local_server_envelope_complete=bool(records),
        full_service_trace_complete=False,
        missing_full_service_stages=MISSING_FULL_SERVICE_STAGES,
        evidence_state=("partial_observability" if records else "insufficient_evidence"),
    )
