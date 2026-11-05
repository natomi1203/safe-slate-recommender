from __future__ import annotations

from dataclasses import replace

import pytest

from safe_slate import (
    create_shadow_evaluation_contract,
    load_shadow_evaluation_contract,
)


def _contract(**overrides: object):
    values: dict[str, object] = {
        "evaluation_id": "shadow-operability-v1",
        "scientific_question": "Does the guarded reranker operate safely in shadow?",
        "population_identity": "jp-home-feed-eligible-v3",
        "window_start": "2026-10-01T00:00:00+09:00",
        "window_end": "2026-10-08T00:00:00+09:00",
        "input_manifest_digest": "a" * 64,
        "code_revision": "987d88b",
        "dependency_lock_digest": "b" * 64,
        "output_namespace": "evidence/shadow-operability-v1",
        "minimum_requests": 10_000,
        "maximum_oracle_p95_ms": 10.0,
        "maximum_fallback_rate": 0.01,
        "maximum_invalid_output_rate": 0.0,
        "maximum_recheck_failure_rate": 0.0,
        "maximum_predicate_error_request_rate": 0.0,
        "maximum_runtime_seconds": 300.0,
        "maximum_memory_mb": 1_024,
        "maximum_output_bytes": 1_000_000,
    }
    values.update(overrides)
    return create_shadow_evaluation_contract(**values)  # type: ignore[arg-type]


def test_shadow_contract_is_stable_and_round_trips() -> None:
    first = _contract()
    second = _contract()

    assert first.contract_digest == second.contract_digest
    assert len(first.contract_digest) == 64
    assert first.verify_digest()
    assert load_shadow_evaluation_contract(first.to_dict()) == first
    assert first.authorized_scope == ("offline_shadow_operability_reporting",)
    assert "production_rollout" in first.explicitly_unauthorized
    assert "production_latency_claim" in first.explicitly_unauthorized
    assert "causal_purchase_lift_claim" in first.explicitly_unauthorized


def test_shadow_contract_can_require_complete_service_latency_traces() -> None:
    contract = _contract(
        require_service_latency_trace=True,
        maximum_service_response_p95_ms=30.0,
        maximum_log_commit_p95_ms=40.0,
    )

    assert contract.expected_log_schema == "1.3"
    assert contract.require_service_latency_trace
    assert contract.maximum_service_response_p95_ms == 30.0
    assert contract.maximum_log_commit_p95_ms == 40.0
    assert load_shadow_evaluation_contract(contract.to_dict()) == contract


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("window_start", "2026-10-01T00:00:00"),
        ("window_end", "2026-10-01T00:00:00+09:00"),
        ("input_manifest_digest", "invalid"),
        ("minimum_requests", 0),
        ("maximum_oracle_p95_ms", 0.0),
        ("maximum_fallback_rate", 1.1),
        ("maximum_invalid_output_rate", -0.1),
        ("maximum_runtime_seconds", float("inf")),
        ("maximum_memory_mb", 0),
        ("maximum_output_bytes", 0),
        ("expected_log_schema", "1.2"),
        ("maximum_service_response_p95_ms", 10.0),
        ("require_service_latency_trace", True),
    ],
)
def test_invalid_shadow_contract_fields_are_rejected(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        _contract(**{field: value})


def test_shadow_contract_rejects_overlapping_permissions() -> None:
    with pytest.raises(ValueError, match="overlap"):
        _contract(
            authorized_scope=("production_rollout",),
            explicitly_unauthorized=("production_rollout",),
        )


def test_serialized_shadow_contract_tampering_is_rejected() -> None:
    contract = _contract()
    payload = contract.to_dict()
    payload["maximum_fallback_rate"] = 0.9

    with pytest.raises(ValueError, match="digest mismatch"):
        load_shadow_evaluation_contract(payload)

    tampered = replace(contract, maximum_fallback_rate=0.9)
    assert not tampered.verify_digest()
