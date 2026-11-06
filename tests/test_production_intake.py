from __future__ import annotations

import json
from pathlib import Path

import pytest

from safe_slate import (
    MISSING_FULL_SERVICE_STAGES,
    audit_observed_grpc_envelopes,
    load_observed_grpc_envelope,
    load_observed_grpc_envelopes_jsonl,
)


PRODUCTION_ENVELOPE_SCHEMA = (
    Path(__file__).parents[1]
    / "schemas"
    / "production_grpc_envelope.schema.json"
)


def _payload(request_id: str = "request-1", trace_id: str = "1" * 32):
    return {
        "schema_version": "safe-slate-production-post-response-1.1",
        "request_id": request_id,
        "trace_id": trace_id,
        "full_method_name": (
            "/x_asia.api.gb.product_recommendation.v1."
            "ProductRecommendationService/ListPredictedProducts"
        ),
        "measurement_scope": "grpc_begin_to_payload_sent",
        "rpc_begin_at": "2026-11-03T03:00:00Z",
        "handler_started_at": "2026-11-03T03:00:00.002Z",
        "response_ready_at": "2026-11-03T03:00:00.007Z",
        "payload_sent_at": "2026-11-03T03:00:00.01Z",
        "rpc_begin_to_handler_started_nanoseconds": 2_000_000,
        "handler_started_to_response_ready_nanoseconds": 5_000_000,
        "response_ready_to_payload_sent_nanoseconds": 3_000_000,
        "rpc_begin_to_payload_sent_nanoseconds": 10_000_000,
        "payload_length_bytes": 100,
        "compressed_payload_length_bytes": 80,
        "wire_payload_length_bytes": 85,
    }


def test_go_production_envelope_round_trips_without_broadening_claim() -> None:
    envelope = load_observed_grpc_envelope(_payload())
    assert envelope.to_dict() == _payload()
    coverage = audit_observed_grpc_envelopes([envelope])
    assert coverage.process_local_server_envelope_complete
    assert not coverage.full_service_trace_complete
    assert coverage.evidence_state == "partial_observability"
    assert coverage.missing_full_service_stages == MISSING_FULL_SERVICE_STAGES
    assert "upstream_queueing_before_grpc_begin" in MISSING_FULL_SERVICE_STAGES
    assert "vendor_aggregate_rpc" in MISSING_FULL_SERVICE_STAGES
    assert "pure_response_serialization" in MISSING_FULL_SERVICE_STAGES
    assert "proposal_broker_publish" in MISSING_FULL_SERVICE_STAGES
    assert "durable_proposal_log_commit" in MISSING_FULL_SERVICE_STAGES
    assert coverage.to_dict()["missing_full_service_stages"] == list(
        MISSING_FULL_SERVICE_STAGES
    )


def test_published_go_envelope_schema_matches_strict_loader_fields() -> None:
    schema = json.loads(PRODUCTION_ENVELOPE_SCHEMA.read_text(encoding="utf-8"))
    payload = _payload()
    assert set(schema["required"]) == set(payload)
    assert set(schema["properties"]) == set(payload)
    assert schema["properties"]["schema_version"]["const"] == payload[
        "schema_version"
    ]
    assert schema["properties"]["measurement_scope"]["const"] == payload[
        "measurement_scope"
    ]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", "1.0", "unsupported"),
        ("measurement_scope", "serialization", "scope"),
        ("trace_id", "0" * 32, "nonzero"),
        ("rpc_begin_at", "2026-11-03T12:00:00+09:00", "canonical"),
        ("handler_started_at", "2026-11-03T02:59:59Z", "not monotonic"),
        (
            "rpc_begin_to_payload_sent_nanoseconds",
            9_000_000,
            "do not match",
        ),
        ("wire_payload_length_bytes", 79, "wire payload"),
    ],
)
def test_go_production_envelope_rejects_invalid_evidence(
    field: str, value: object, message: str
) -> None:
    payload = _payload()
    payload[field] = value
    with pytest.raises(ValueError, match=message):
        load_observed_grpc_envelope(payload)


def test_go_production_envelope_rejects_schema_drift() -> None:
    missing = _payload()
    missing.pop("payload_sent_at")
    with pytest.raises(ValueError, match="fields"):
        load_observed_grpc_envelope(missing)
    extra = _payload()
    extra["serialization_ms"] = 1.0
    with pytest.raises(ValueError, match="fields"):
        load_observed_grpc_envelope(extra)


def test_bounded_jsonl_loader_and_identity_audit_fail_closed(tmp_path) -> None:
    path = tmp_path / "post-response.jsonl"
    first = _payload("request-1", "1" * 32)
    second = _payload("request-2", "2" * 32)
    path.write_text(
        json.dumps(first) + "\n" + json.dumps(second) + "\n",
        encoding="utf-8",
    )
    records = load_observed_grpc_envelopes_jsonl(
        path,
        maximum_records=2,
        maximum_bytes=path.stat().st_size,
    )
    assert len(records) == 2
    assert audit_observed_grpc_envelopes(records).record_count == 2
    with pytest.raises(ValueError, match="maximum_records"):
        load_observed_grpc_envelopes_jsonl(
            path,
            maximum_records=1,
            maximum_bytes=path.stat().st_size,
        )
    with pytest.raises(ValueError, match="maximum_bytes"):
        load_observed_grpc_envelopes_jsonl(
            path,
            maximum_records=2,
            maximum_bytes=path.stat().st_size - 1,
        )
    with pytest.raises(ValueError, match="one-to-one"):
        audit_observed_grpc_envelopes([records[0], records[0]])


def test_jsonl_loader_rejects_duplicate_keys_and_empty_is_insufficient(tmp_path) -> None:
    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text(
        '{"schema_version":"x","schema_version":"y"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid production"):
        load_observed_grpc_envelopes_jsonl(
            duplicate,
            maximum_records=1,
            maximum_bytes=duplicate.stat().st_size,
        )
    empty = audit_observed_grpc_envelopes([])
    assert empty.evidence_state == "insufficient_evidence"
    assert not empty.process_local_server_envelope_complete


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("request_id", "", "non-empty string"),
        ("payload_length_bytes", True, "nonnegative integer"),
        ("rpc_begin_at", "2026-11-03T03:00:00.010Z", "trailing fractional zero"),
        ("rpc_begin_at", "2026-02-30T03:00:00Z", "not a valid UTC"),
        ("full_method_name", "wrong", "method is invalid"),
    ],
)
def test_production_envelope_rejects_remaining_field_contracts(
    field: str, value: object, message: str
) -> None:
    payload = _payload()
    payload[field] = value
    with pytest.raises(ValueError, match=message):
        load_observed_grpc_envelope(payload)


def test_production_envelope_rejects_non_mapping_payload() -> None:
    with pytest.raises(ValueError, match="must be a mapping"):
        load_observed_grpc_envelope([])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("\n", "blank JSONL"),
        ('{"value":NaN}\n', "invalid production"),
        ("{bad}\n", "invalid production"),
    ],
)
def test_production_jsonl_rejects_blank_nonfinite_and_invalid_rows(
    tmp_path, contents: str, message: str
) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_text(contents)
    with pytest.raises(ValueError, match=message):
        load_observed_grpc_envelopes_jsonl(
            path, maximum_records=1, maximum_bytes=1_000
        )


def test_production_jsonl_rejects_nonpositive_caps_and_invalid_audit_records(
    tmp_path,
) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    with pytest.raises(ValueError, match="must be positive"):
        load_observed_grpc_envelopes_jsonl(
            path, maximum_records=0, maximum_bytes=1
        )
    with pytest.raises(ValueError, match="ObservedGRPCEnvelope"):
        audit_observed_grpc_envelopes(["bad"])  # type: ignore[list-item]
