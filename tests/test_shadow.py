from __future__ import annotations

from dataclasses import replace
import math

import pytest

from safe_slate import (
    attach_service_latency,
    create_service_latency_trace,
    guarded_rerank,
    summarize_shadow_run,
)
from safe_slate.shadow import _service_trace_valid


def _shadow_log(
    request_id: str,
    *,
    with_service_latency: bool = False,
    **overrides: object,
):
    values: dict[str, object] = {
        "request_id": request_id,
        "event_timestamp": "2026-10-01T12:00:00+09:00",
        "baseline_item_ids": [10, 11, 12, 13],
        "protected_count": 2,
        "remove_positions": [2, 3],
        "add_item_ids": [100, 101],
        "reward_deltas": [0.2, 0.1],
        "safety_mean_deltas": [0.04, 0.03],
        "safety_variances": [0.0, 0.0],
        "baseline_metric": 0.5,
        "degradation_tolerance": 0.02,
        "max_replacements": 2,
        "timeout_ms": 10,
        "serving_mode": "shadow",
        "support_ok": True,
        "uncertainty_valid": True,
        "distribution_shift_detected": False,
        "proposal_log_complete": True,
    }
    values.update(overrides)
    log = guarded_rerank(**values).proposal_log  # type: ignore[arg-type]
    if not with_service_latency:
        return log
    return attach_service_latency(
        log,
        create_service_latency_trace(
            response_latency_ms=20.0,
            log_commit_latency_ms=25.0,
            candidate_generation_ms=3.0,
            model_inference_ms=5.0,
            queueing_ms=1.0,
            rpc_ms=2.0,
            guardrail_facade_ms=10.0,
            serialization_ms=1.0,
            proposal_log_write_ms=2.0,
        ),
    )


def _summarize(logs, **overrides: object):
    values: dict[str, object] = {
        "window_start": "2026-10-01T00:00:00+09:00",
        "window_end": "2026-10-02T00:00:00+09:00",
        "minimum_requests": 2,
        "maximum_oracle_p95_ms": 100.0,
        "maximum_fallback_rate": 0.0,
        "maximum_invalid_output_rate": 0.0,
        "maximum_recheck_failure_rate": 0.0,
        "maximum_predicate_error_request_rate": 0.0,
    }
    values.update(overrides)
    return summarize_shadow_run(logs, **values)  # type: ignore[arg-type]


def test_valid_shadow_run_passes_operability_without_authorizing_claims() -> None:
    summary = _summarize([_shadow_log("r0"), _shadow_log("r1")])

    assert summary.operability_pass
    assert summary.request_count == 2
    assert summary.duplicate_request_id_count == 0
    assert len(summary.log_data_digest) == 64
    assert summary.window_start == "2026-10-01T00:00:00+09:00"
    assert summary.window_end == "2026-10-02T00:00:00+09:00"
    assert summary.event_time_min == "2026-10-01T12:00:00+09:00"
    assert summary.event_time_max == "2026-10-01T12:00:00+09:00"
    assert summary.fallback_rate == 0.0
    assert summary.invalid_output_rate == 0.0
    assert summary.recheck_failure_rate == 0.0
    assert summary.certificate_violation_rate == 0.0
    assert summary.mean_selected_count == pytest.approx(2.0)
    assert summary.minimum_safety_margin is not None
    assert summary.minimum_safety_margin >= 0.0
    assert not summary.production_latency_claim_authorized
    assert not summary.causal_lift_claim_authorized
    assert summary.reasons == ()


def test_failed_serving_gate_is_reported_as_shadow_fallback() -> None:
    summary = _summarize(
        [_shadow_log("r0"), _shadow_log("r1", support_ok=False)],
        maximum_fallback_rate=0.4,
    )

    assert summary.fallback_rate == pytest.approx(0.5)
    assert summary.gate_fallback_rate == pytest.approx(0.5)
    assert not summary.operability_pass
    assert "fallback_rate_above_gate" in summary.reasons
    assert dict(summary.fallback_reason_counts)["support_gate"] == 1


def test_non_shadow_and_changed_served_slate_are_invalid() -> None:
    serve_log = _shadow_log("r0", serving_mode="serve")
    changed_log = replace(
        _shadow_log("r1"),
        served_slate=(10, 11, 100, 101),
    )
    summary = _summarize([serve_log, changed_log])

    assert summary.invalid_output_rate == 1.0
    assert not summary.operability_pass
    invalid = dict(summary.invalid_reason_counts)
    assert invalid["non_shadow_record"] == 1
    assert invalid["shadow_served_slate_changed"] == 2


def test_missing_event_timestamp_is_invalid_shadow_evidence() -> None:
    missing_time = replace(_shadow_log("r0"), event_timestamp=None)
    summary = _summarize([missing_time, _shadow_log("r1")])

    assert not summary.operability_pass
    assert (
        dict(summary.invalid_reason_counts)["event_timestamp_missing_or_invalid"] == 1
    )


def test_shadow_window_uses_absolute_time_not_timestamp_text_order() -> None:
    later = replace(
        _shadow_log("r0"),
        event_timestamp="2026-10-01T23:00:00+09:00",
    )
    earlier = replace(
        _shadow_log("r1"),
        event_timestamp="2026-10-01T13:30:00+00:00",
    )

    summary = _summarize([later, earlier])

    assert summary.event_time_min == "2026-10-01T13:30:00+00:00"
    assert summary.event_time_max == "2026-10-01T23:00:00+09:00"


def test_shadow_records_outside_frozen_window_are_invalid() -> None:
    before = replace(
        _shadow_log("r0"),
        event_timestamp="2026-09-30T23:59:59+09:00",
    )
    at_end = replace(
        _shadow_log("r1"),
        event_timestamp="2026-10-02T00:00:00+09:00",
    )

    summary = _summarize([before, at_end])

    assert summary.invalid_output_rate == 1.0
    assert dict(summary.invalid_reason_counts)["event_outside_frozen_window"] == 2
    assert not summary.operability_pass


def test_recheck_failure_and_certificate_violation_are_separate() -> None:
    failed_recheck = replace(_shadow_log("r0"), rechecked=False)
    unsafe = replace(
        _shadow_log("r1"),
        selected_safety_lcb=-1.0,
    )
    summary = _summarize([failed_recheck, unsafe])

    assert summary.recheck_failure_rate == pytest.approx(0.5)
    assert summary.certificate_violation_rate == pytest.approx(0.5)
    assert "recheck_failure_rate_above_gate" in summary.reasons
    assert "certificate_violation_observed" in summary.reasons


def test_latency_gate_uses_oracle_elapsed_time_only() -> None:
    slow = replace(_shadow_log("r1"), elapsed_seconds=0.200)
    summary = _summarize(
        [_shadow_log("r0"), slow],
        maximum_oracle_p95_ms=100.0,
    )

    assert summary.oracle_latency_ms_p95 is not None
    assert summary.oracle_latency_ms_p95 > 100.0
    assert "oracle_p95_latency_above_gate" in summary.reasons
    assert not summary.production_latency_claim_authorized


def test_complete_service_latency_traces_support_shadow_service_evidence() -> None:
    summary = _summarize(
        [
            _shadow_log("r0", with_service_latency=True),
            _shadow_log("r1", with_service_latency=True),
        ],
        require_service_latency_trace=True,
        maximum_service_response_p95_ms=30.0,
        maximum_log_commit_p95_ms=30.0,
    )

    assert summary.service_latency_trace_rate == 1.0
    assert summary.complete_service_latency_trace_rate == 1.0
    assert summary.service_response_latency_ms_p95 == pytest.approx(20.0)
    assert summary.log_commit_latency_ms_p95 == pytest.approx(25.0)
    assert summary.shadow_service_latency_evidence_valid
    assert summary.operability_pass
    assert not summary.production_latency_claim_authorized


def test_missing_or_slow_service_latency_trace_fails_closed() -> None:
    missing = _summarize(
        [_shadow_log("r0"), _shadow_log("r1", with_service_latency=True)],
        require_service_latency_trace=True,
        maximum_service_response_p95_ms=30.0,
        maximum_log_commit_p95_ms=30.0,
    )
    slow_log = attach_service_latency(
        _shadow_log("r0"),
        create_service_latency_trace(
            response_latency_ms=40.0,
            log_commit_latency_ms=45.0,
            candidate_generation_ms=3.0,
            model_inference_ms=5.0,
            queueing_ms=1.0,
            rpc_ms=2.0,
            guardrail_facade_ms=10.0,
            serialization_ms=1.0,
            proposal_log_write_ms=2.0,
        ),
    )
    slow = _summarize(
        [slow_log, _shadow_log("r1", with_service_latency=True)],
        require_service_latency_trace=True,
        maximum_service_response_p95_ms=30.0,
        maximum_log_commit_p95_ms=30.0,
    )

    assert not missing.operability_pass
    assert dict(missing.invalid_reason_counts)["service_latency_trace_missing"] == 1
    assert not missing.shadow_service_latency_evidence_valid
    assert "service_response_p95_latency_above_gate" in slow.reasons
    assert "log_commit_p95_latency_above_gate" in slow.reasons
    assert not slow.shadow_service_latency_evidence_valid


def test_duplicate_requests_and_predicate_errors_fail_configured_gates() -> None:
    first = _shadow_log("same")
    second = replace(_shadow_log("same"), predicate_errors=1)
    summary = _summarize([first, second])

    assert summary.predicate_error_request_rate == pytest.approx(0.5)
    assert summary.duplicate_request_id_count == 1
    assert "duplicate_request_id" in summary.reasons
    assert "predicate_error_rate_above_gate" in summary.reasons


def test_request_count_gate_does_not_relabel_small_run_as_evidence() -> None:
    summary = _summarize([_shadow_log("r0")], minimum_requests=2)

    assert not summary.operability_pass
    assert "request_count_below_gate" in summary.reasons


def test_non_json_proposal_context_is_rejected_but_digestible() -> None:
    invalid = replace(_shadow_log("r0"), context={"bad": object()})

    summary = _summarize([invalid, _shadow_log("r1")])

    assert not summary.operability_pass
    assert dict(summary.invalid_reason_counts)["proposal_log_not_canonical_json"] == 1
    assert len(summary.log_data_digest) == 64


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("minimum_requests", 0),
        ("window_start", "2026-10-01T00:00:00"),
        ("window_end", "2026-10-01T00:00:00+09:00"),
        ("maximum_oracle_p95_ms", 0.0),
        ("maximum_oracle_p95_ms", math.inf),
        ("maximum_fallback_rate", -0.1),
        ("maximum_invalid_output_rate", 1.1),
        ("maximum_recheck_failure_rate", math.nan),
        ("maximum_predicate_error_request_rate", -1.0),
        ("require_service_latency_trace", True),
        ("maximum_service_response_p95_ms", 10.0),
    ],
)
def test_invalid_shadow_contract_thresholds_raise(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        _summarize([_shadow_log("r0"), _shadow_log("r1")], **{field: value})


def test_empty_or_wrong_record_types_raise() -> None:
    with pytest.raises(ValueError):
        _summarize([])
    with pytest.raises(ValueError):
        _summarize([_shadow_log("r0"), object()])


def test_shadow_summary_serializes_count_maps() -> None:
    serialized = _summarize([_shadow_log("r0"), _shadow_log("r1")]).to_dict()

    assert serialized["request_count"] == 2
    assert sum(serialized["status_counts"].values()) == 2
    assert serialized["invalid_reason_counts"] == {}
    assert serialized["operability_pass"] is True
    assert serialized["causal_lift_claim_authorized"] is False


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_context_values_remain_digestible_but_invalid(value: float) -> None:
    log = replace(_shadow_log("r0"), context={"value": value})
    summary = _summarize([log], minimum_requests=1, maximum_invalid_output_rate=1.0)
    assert "proposal_log_not_canonical_json" in dict(summary.invalid_reason_counts)
    assert len(summary.log_data_digest) == 64


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"schema_version": "1.2"}, "unsupported_schema"),
        ({"event_timestamp": "not-a-time"}, "event_timestamp_missing_or_invalid"),
        ({"proposal_log_complete": False}, "proposal_log_incomplete"),
        ({"protected_count": True}, "protected_count_invalid"),
        ({"selected_operation_indices": (True,)}, "selected_operation_indices_invalid"),
        ({"k_lower": 1}, "selected_count_k_lower_mismatch"),
        ({"k_upper": 1}, "cardinality_frontier_inconsistent"),
        ({"cardinality_gap": 1}, "cardinality_frontier_inconsistent"),
        ({"selected_flags": (True,)}, "selected_flag_length_mismatch"),
        ({"selected_flags": (False, False)}, "selected_flags_inconsistent"),
        ({"proposed_slate": (10,)}, "proposed_slate_length_mismatch"),
        ({"proposed_slate": (99, 11, 100, 101)}, "protected_core_changed"),
        ({"proposed_slate": (10, 11, 100, 100)}, "proposed_slate_contains_duplicates"),
        ({"elapsed_seconds": math.nan}, "oracle_latency_invalid"),
        ({"selected_safety_lcb": math.nan}, "certificate_value_invalid"),
    ],
)
def test_shadow_summary_records_every_invalid_proposal_contract(
    changes: dict[str, object], reason: str
) -> None:
    log = replace(_shadow_log("r0"), **changes)
    summary = _summarize([log], minimum_requests=1, maximum_invalid_output_rate=1.0)
    assert reason in dict(summary.invalid_reason_counts)


def test_shadow_service_trace_validation_rejects_noncanonical_variants() -> None:
    log = _shadow_log("r0", with_service_latency=True)
    invalid_trace = replace(log.service_latency_trace, response_latency_ms=-1.0)
    invalid = replace(log, service_latency_trace=invalid_trace)
    incomplete = replace(
        log,
        service_latency_trace=create_service_latency_trace(
            response_latency_ms=20.0, log_commit_latency_ms=25.0
        ),
    )
    for candidate in (invalid, incomplete):
        summary = _summarize(
            [candidate],
            minimum_requests=1,
            maximum_invalid_output_rate=1.0,
            require_service_latency_trace=True,
            maximum_service_response_p95_ms=30.0,
            maximum_log_commit_p95_ms=30.0,
        )
        assert "service_latency_trace_invalid" in dict(summary.invalid_reason_counts)

    non_trace = replace(log, service_latency_trace=object())
    assert not _service_trace_valid(non_trace, require_complete=True)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"require_service_latency_trace": 1}, "must be boolean"),
        (
            {
                "require_service_latency_trace": True,
                "maximum_service_response_p95_ms": -1.0,
                "maximum_log_commit_p95_ms": 30.0,
            },
            "finite and positive",
        ),
    ],
)
def test_shadow_summary_rejects_remaining_invalid_thresholds(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _summarize([_shadow_log("r0")], **overrides)
