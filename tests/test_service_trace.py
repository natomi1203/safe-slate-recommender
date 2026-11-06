from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from safe_slate import (
    attach_service_latency,
    canonical_proposal_log_bytes,
    create_service_latency_trace,
    create_service_trace_completion,
    guarded_rerank,
    join_service_trace_completions,
    load_service_trace_completion,
    load_service_trace_completions_jsonl,
    proposal_log_sha256,
    REGISTERED_SERVICE_STAGES,
    ServiceTraceRecorder,
    summarize_production_service_run,
)
from safe_slate.service_trace import _percentiles


SERVICE_TRACE_SCHEMA = (
    Path(__file__).parents[1]
    / "schemas"
    / "service_trace_completion.schema.json"
)


def _proposal(request_id: str):
    return guarded_rerank(
        request_id=request_id,
        event_timestamp="2026-11-03T12:00:00+09:00",
        baseline_item_ids=[10, 11, 12, 13],
        protected_count=2,
        remove_positions=[2, 3],
        add_item_ids=[100, 101],
        reward_deltas=[0.2, 0.1],
        safety_mean_deltas=[0.04, 0.03],
        safety_variances=[0.0, 0.0],
        baseline_metric=0.5,
        degradation_tolerance=0.02,
        max_replacements=2,
        timeout_ms=10,
        serving_mode="serve",
        support_ok=True,
        uncertainty_valid=True,
        distribution_shift_detected=False,
        proposal_log_complete=True,
    ).proposal_log


def _trace():
    return create_service_latency_trace(
        response_latency_ms=20.0,
        log_commit_latency_ms=25.0,
        candidate_generation_ms=3.0,
        model_inference_ms=5.0,
        queueing_ms=1.0,
        rpc_ms=2.0,
        guardrail_facade_ms=10.0,
        serialization_ms=1.0,
        proposal_log_write_ms=2.0,
    )


class _FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance_ms(self, value: float) -> None:
        self.value += value / 1_000.0


def _completion(
    request_id: str,
    trace_id: str = "1" * 32,
    *,
    proposal=None,
    trace=None,
):
    return create_service_trace_completion(
        proposal or _proposal(request_id),
        trace_id=trace_id,
        completed_at="2026-11-03T12:00:01+09:00",
        trace_source="opentelemetry",
        service_latency_trace=trace or _trace(),
    )


def test_post_commit_sidecar_round_trips_and_binds_primary_bytes() -> None:
    proposal = _proposal("r0")
    completion = create_service_trace_completion(
        proposal,
        trace_id="0123456789abcdef0123456789abcdef",
        completed_at="2026-11-03T12:00:01+09:00",
        trace_source="opentelemetry",
        service_latency_trace=_trace(),
    )

    assert completion.proposal_log_sha256 == proposal_log_sha256(proposal)
    assert len(canonical_proposal_log_bytes(proposal)) > 0
    assert load_service_trace_completion(completion.to_dict()) == completion
    assert completion.service_latency_trace.complete


def test_published_service_trace_schema_matches_canonical_python_fields() -> None:
    schema = json.loads(SERVICE_TRACE_SCHEMA.read_text(encoding="utf-8"))
    completion = _completion("r0").to_dict()
    trace = completion["service_latency_trace"]

    assert set(schema["required"]) == set(completion)
    assert set(schema["properties"]) == set(completion)
    assert schema["properties"]["schema_version"]["const"] == completion[
        "schema_version"
    ]
    trace_schema = schema["$defs"]["service_latency_trace"]
    assert set(trace_schema["required"]) == set(trace)
    assert set(trace_schema["properties"]) == set(trace)
    assert trace_schema["properties"]["schema_version"]["const"] == trace[
        "schema_version"
    ]
    assert trace_schema["properties"]["measurement_scope"]["const"] == trace[
        "measurement_scope"
    ]
    assert trace_schema["properties"]["clock"]["const"] == trace["clock"]
    assert trace_schema["properties"]["complete"]["const"] is True


def test_exact_join_attaches_trace_without_mutating_primary_record() -> None:
    first = _proposal("r0")
    second = _proposal("r1")
    completions = [
        _completion("r1", "2" * 32, proposal=second),
        _completion("r0", "1" * 32, proposal=first),
    ]

    joined = join_service_trace_completions([first, second], completions)

    assert first.service_latency_trace is None
    assert second.service_latency_trace is None
    assert all(log.service_latency_trace == _trace() for log in joined)
    assert [log.request_id for log in joined] == ["r0", "r1"]


def test_digest_tampering_and_incomplete_join_fail_closed() -> None:
    proposal = _proposal("r0")
    completion = _completion("r0")
    tampered = replace(proposal, context={"tampered": True})

    with pytest.raises(ValueError, match="digest mismatch"):
        join_service_trace_completions([tampered], [completion])
    with pytest.raises(ValueError, match="incomplete"):
        join_service_trace_completions([proposal, _proposal("r1")], [completion])
    with pytest.raises(ValueError, match="incomplete"):
        join_service_trace_completions(
            [proposal], [completion, _completion("r1", "2" * 32)]
        )


def test_duplicate_request_or_trace_identity_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate request"):
        join_service_trace_completions(
            [_proposal("r0"), _proposal("r0")],
            [_completion("r0")],
        )
    with pytest.raises(ValueError, match="duplicate trace"):
        join_service_trace_completions(
            [_proposal("r0"), _proposal("r1")],
            [_completion("r0", "1" * 32), _completion("r1", "1" * 32)],
        )


def test_completion_requires_full_trace_and_valid_chronology() -> None:
    incomplete = create_service_latency_trace(
        response_latency_ms=20.0,
        log_commit_latency_ms=25.0,
    )
    with pytest.raises(ValueError, match="every registered stage"):
        create_service_trace_completion(
            _proposal("r0"),
            trace_id="1" * 32,
            completed_at="2026-11-03T12:00:01+09:00",
            trace_source="opentelemetry",
            service_latency_trace=incomplete,
        )
    with pytest.raises(ValueError, match="precede"):
        create_service_trace_completion(
            _proposal("r0"),
            trace_id="1" * 32,
            completed_at="2026-11-03T11:59:59+09:00",
            trace_source="opentelemetry",
            service_latency_trace=_trace(),
        )
    with pytest.raises(ValueError, match="trace_id"):
        create_service_trace_completion(
            _proposal("r0"),
            trace_id="0" * 32,
            completed_at="2026-11-03T12:00:01+09:00",
            trace_source="opentelemetry",
            service_latency_trace=_trace(),
        )


def test_primary_record_with_embedded_trace_cannot_be_rebound() -> None:
    attached = attach_service_latency(_proposal("r0"), _trace())
    with pytest.raises(ValueError, match="must not already"):
        canonical_proposal_log_bytes(attached)


def test_bounded_sidecar_jsonl_loader_preserves_order(tmp_path: Path) -> None:
    completions = [_completion("r0", "1" * 32), _completion("r1", "2" * 32)]
    source = tmp_path / "service-traces.jsonl"
    source.write_text(
        "".join(json.dumps(value.to_dict()) + "\n" for value in completions),
        encoding="utf-8",
    )

    loaded = load_service_trace_completions_jsonl(
        source,
        maximum_records=2,
        maximum_bytes=source.stat().st_size,
    )
    assert loaded == tuple(completions)
    with pytest.raises(ValueError, match="maximum_records"):
        load_service_trace_completions_jsonl(
            source,
            maximum_records=1,
            maximum_bytes=source.stat().st_size,
        )
    with pytest.raises(ValueError, match="maximum_bytes"):
        load_service_trace_completions_jsonl(
            source,
            maximum_records=2,
            maximum_bytes=source.stat().st_size - 1,
        )


def test_sidecar_loader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    source = tmp_path / "duplicate.jsonl"
    source.write_text(
        '{"schema_version":"x","schema_version":"y"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_service_trace_completions_jsonl(
            source,
            maximum_records=1,
            maximum_bytes=source.stat().st_size,
        )


def _summary(logs, completions, **overrides):
    values = {
        "window_start": "2026-11-03T00:00:00+09:00",
        "window_end": "2026-11-04T00:00:00+09:00",
        "minimum_requests": 2,
        "maximum_response_p95_ms": 30.0,
        "maximum_log_commit_p95_ms": 30.0,
        "maximum_fallback_rate": 0.0,
    }
    values.update(overrides)
    return summarize_production_service_run(logs, completions, **values)


def test_production_service_summary_reports_every_stage_without_broad_claims() -> None:
    first = _proposal("r0")
    second = _proposal("r1")
    summary = _summary(
        [first, second],
        [
            _completion("r0", "1" * 32, proposal=first),
            _completion("r1", "2" * 32, proposal=second),
        ],
    )

    assert summary.observed_window_latency_evidence_valid
    assert summary.request_count == 2
    assert summary.invalid_record_rate == 0.0
    assert summary.fallback_rate == 0.0
    assert summary.latency_ms["response"]["p95"] == pytest.approx(20.0)
    assert summary.latency_ms["log_commit"]["p95"] == pytest.approx(25.0)
    assert summary.latency_ms["candidate_generation"]["p95"] == pytest.approx(3.0)
    assert summary.latency_ms["model_inference"]["p95"] == pytest.approx(5.0)
    assert not summary.production_latency_claim_authorized
    assert not summary.production_safety_claim_authorized
    assert not summary.causal_lift_claim_authorized
    assert summary.to_dict()["trace_source_counts"] == {"opentelemetry": 2}


def test_production_service_summary_fails_slow_or_invalid_window() -> None:
    first = _proposal("r0")
    second = _proposal("r1")
    slow_trace = create_service_latency_trace(
        response_latency_ms=40.0,
        log_commit_latency_ms=45.0,
        candidate_generation_ms=3.0,
        model_inference_ms=5.0,
        queueing_ms=1.0,
        rpc_ms=2.0,
        guardrail_facade_ms=10.0,
        serialization_ms=1.0,
        proposal_log_write_ms=2.0,
    )
    summary = _summary(
        [first, second],
        [
            _completion("r0", "1" * 32, proposal=first, trace=slow_trace),
            _completion("r1", "2" * 32, proposal=second),
        ],
    )

    assert not summary.observed_window_latency_evidence_valid
    assert "response_p95_above_gate" in summary.reasons
    assert "log_commit_p95_above_gate" in summary.reasons

    outside = replace(
        first,
        event_timestamp="2026-11-04T00:00:00+09:00",
    )
    outside_summary = _summary(
        [outside, second],
        [
            create_service_trace_completion(
                outside,
                trace_id="1" * 32,
                completed_at="2026-11-04T00:00:01+09:00",
                trace_source="opentelemetry",
                service_latency_trace=_trace(),
            ),
            _completion("r1", "2" * 32, proposal=second),
        ],
    )
    assert not outside_summary.observed_window_latency_evidence_valid
    assert outside_summary.invalid_record_rate == pytest.approx(0.5)
    assert "invalid_record_rate_above_gate" in outside_summary.reasons


def test_production_summary_preserves_insufficient_sample_state() -> None:
    proposal = _proposal("r0")
    summary = _summary(
        [proposal],
        [_completion("r0", proposal=proposal)],
        minimum_requests=2,
    )
    assert not summary.observed_window_latency_evidence_valid
    assert summary.reasons == ("request_count_below_gate",)


def test_request_recorder_measures_real_stages_and_completes_sidecar() -> None:
    clock = _FakeClock()
    recorder = ServiceTraceRecorder(
        trace_id="1" * 32,
        event_timestamp="2026-11-03T12:00:00+09:00",
        trace_source="opentelemetry",
        clock=clock,
    )
    recorder.record_stage_ms("queueing", 1.0)
    recorder.record_stage_ms("rpc", 2.0)
    for stage, duration in (
        ("candidate_generation", 3.0),
        ("model_inference", 5.0),
        ("guardrail_facade", 10.0),
        ("serialization", 1.0),
    ):
        with recorder.measure(stage):
            clock.advance_ms(duration)
    response = recorder.mark_response()
    with recorder.measure("proposal_log_write"):
        clock.advance_ms(2.0)
    commit = recorder.mark_primary_log_commit()
    completion = recorder.complete(
        _proposal("r0"),
        completed_at="2026-11-03T12:00:01+09:00",
    )

    assert response == pytest.approx(19.0)
    assert commit == pytest.approx(21.0)
    trace = completion.service_latency_trace
    assert trace.candidate_generation_ms == pytest.approx(3.0)
    assert trace.model_inference_ms == pytest.approx(5.0)
    assert trace.queueing_ms == pytest.approx(1.0)
    assert trace.rpc_ms == pytest.approx(2.0)
    assert trace.guardrail_facade_ms == pytest.approx(10.0)
    assert trace.serialization_ms == pytest.approx(1.0)
    assert trace.proposal_log_write_ms == pytest.approx(2.0)
    assert set(REGISTERED_SERVICE_STAGES) == {
        "candidate_generation",
        "model_inference",
        "queueing",
        "rpc",
        "guardrail_facade",
        "serialization",
        "proposal_log_write",
    }


def test_request_recorder_rejects_missing_duplicate_or_out_of_order_events() -> None:
    clock = _FakeClock()
    recorder = ServiceTraceRecorder(
        trace_id="1" * 32,
        event_timestamp="2026-11-03T12:00:00+09:00",
        clock=clock,
    )
    recorder.record_stage_ms("queueing", 1.0)
    with pytest.raises(ValueError, match="already recorded"):
        recorder.record_stage_ms("queueing", 2.0)
    with pytest.raises(ValueError, match="response must be marked"):
        recorder.mark_primary_log_commit()
    recorder.mark_response()
    with pytest.raises(ValueError, match="proposal_log_write"):
        recorder.mark_primary_log_commit()
    with pytest.raises(ValueError, match="stages are missing"):
        recorder.complete(
            _proposal("r0"),
            completed_at="2026-11-03T12:00:01+09:00",
        )


def test_request_recorder_detects_backward_clock() -> None:
    clock = _FakeClock()
    recorder = ServiceTraceRecorder(
        trace_id="1" * 32,
        event_timestamp="2026-11-03T12:00:00+09:00",
        clock=clock,
    )
    clock.value = -1.0
    with pytest.raises(ValueError, match="backwards"):
        recorder.mark_response()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.pop("trace_source"), "fields"),
        (lambda payload: payload.update(extra=True), "fields"),
        (lambda payload: payload.update(schema_version="2.0"), "unsupported"),
        (
            lambda payload: payload.update(completed_at="2026-11-03T11:59:59+09:00"),
            "precede",
        ),
        (lambda payload: payload.update(trace_id="0" * 32), "trace_id"),
        (lambda payload: payload.update(trace_id="A" * 32), "trace_id"),
        (lambda payload: payload.update(request_id=""), "nonempty"),
        (lambda payload: payload.update(trace_source="  "), "nonempty"),
        (lambda payload: payload.update(proposal_log_sha256="bad"), "SHA-256"),
        (
            lambda payload: payload["service_latency_trace"].pop("clock"),
            "canonical schema",
        ),
        (
            lambda payload: payload["service_latency_trace"].update(complete=False),
            "complete and canonical",
        ),
    ],
)
def test_sidecar_loader_rejects_malformed_contract(mutation, message: str) -> None:
    payload = _completion("r0").to_dict()
    mutation(payload)
    with pytest.raises(ValueError, match=message):
        load_service_trace_completion(payload)


def test_service_trace_creation_validates_types_and_timestamps() -> None:
    proposal = _proposal("r0")
    with pytest.raises(ValueError, match="ServiceLatencyTrace"):
        create_service_trace_completion(
            proposal,
            trace_id="1" * 32,
            completed_at="2026-11-03T12:00:01+09:00",
            trace_source="opentelemetry",
            service_latency_trace={},  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="event_timestamp"):
        create_service_trace_completion(
            replace(proposal, event_timestamp=None),
            trace_id="1" * 32,
            completed_at="2026-11-03T12:00:01+09:00",
            trace_source="opentelemetry",
            service_latency_trace=_trace(),
        )
    with pytest.raises(ValueError, match="ISO-8601"):
        create_service_trace_completion(
            proposal,
            trace_id="1" * 32,
            completed_at="not-a-time",
            trace_source="opentelemetry",
            service_latency_trace=_trace(),
        )
    with pytest.raises(ValueError, match="ProposalLog"):
        canonical_proposal_log_bytes([])  # type: ignore[arg-type]


def test_sidecar_jsonl_loader_rejects_resource_and_encoding_errors(
    tmp_path: Path,
) -> None:
    valid = tmp_path / "valid.jsonl"
    valid.write_text(json.dumps(_completion("r0").to_dict()) + "\n", encoding="utf-8")
    for field, value in [("maximum_records", True), ("maximum_bytes", 0)]:
        kwargs = {
            "maximum_records": 1,
            "maximum_bytes": valid.stat().st_size,
            field: value,
        }
        with pytest.raises(ValueError, match="positive integer"):
            load_service_trace_completions_jsonl(valid, **kwargs)

    with pytest.raises(ValueError, match="regular file"):
        load_service_trace_completions_jsonl(
            tmp_path, maximum_records=1, maximum_bytes=1_000
        )

    for name, contents, message in [
        ("empty", b"", "empty"),
        ("blank", b"\n", "blank"),
        ("invalid", b"{bad}\n", "invalid service trace JSON"),
        ("nonfinite", b'{"value":NaN}\n', "non-finite JSON"),
        ("utf8", b"\xff\xfe\n", "UTF-8"),
    ]:
        source = tmp_path / f"{name}.jsonl"
        source.write_bytes(contents)
        with pytest.raises(ValueError, match=message):
            load_service_trace_completions_jsonl(
                source, maximum_records=1, maximum_bytes=1_000
            )


def test_service_trace_join_validates_collection_identity_and_timestamps() -> None:
    proposal = _proposal("r0")
    completion = _completion("r0", proposal=proposal)
    with pytest.raises(ValueError, match="nonempty"):
        join_service_trace_completions([], [])
    with pytest.raises(ValueError, match="invalid record"):
        join_service_trace_completions(["bad"], [])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="invalid record"):
        join_service_trace_completions([proposal], ["bad"])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="duplicate request"):
        join_service_trace_completions([proposal], [completion, completion])
    with pytest.raises(ValueError, match="timestamp mismatch"):
        join_service_trace_completions(
            [proposal],
            [replace(completion, event_timestamp="2026-11-03T12:00:01+09:00")],
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"serving_mode": "preview"}, "serve or shadow"),
        ({"window_end": "2026-11-03T00:00:00+09:00"}, "ordered"),
        ({"minimum_requests": True}, "positive integer"),
        ({"maximum_response_p95_ms": 0.0}, "finite and positive"),
        ({"maximum_log_commit_p95_ms": float("inf")}, "finite and positive"),
        ({"maximum_fallback_rate": True}, "finite and in"),
        ({"maximum_invalid_record_rate": 1.1}, "finite and in"),
        ({"maximum_recheck_failure_rate": -0.1}, "finite and in"),
    ],
)
def test_production_service_summary_rejects_invalid_gates(
    overrides: dict[str, object], message: str
) -> None:
    proposal = _proposal("r0")
    with pytest.raises(ValueError, match=message):
        _summary(
            [proposal],
            [_completion("r0", proposal=proposal)],
            **overrides,
        )


def test_recorder_rejects_invalid_clocks_stages_and_durations() -> None:
    with pytest.raises(ValueError, match="clock must be callable"):
        ServiceTraceRecorder(
            trace_id="1" * 32,
            event_timestamp="2026-11-03T12:00:00+09:00",
            clock=1,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="finite monotonic"):
        ServiceTraceRecorder(
            trace_id="1" * 32,
            event_timestamp="2026-11-03T12:00:00+09:00",
            clock=lambda: float("nan"),
        )

    clock = _FakeClock()
    recorder = ServiceTraceRecorder(
        trace_id="1" * 32,
        event_timestamp="2026-11-03T12:00:00+09:00",
        clock=clock,
    )
    with pytest.raises(ValueError, match="unknown service stage"):
        recorder.record_stage_ms("unknown", 1.0)
    for value in (True, -1.0, float("inf")):
        with pytest.raises(ValueError, match="finite and nonnegative"):
            recorder.record_stage_ms("queueing", value)  # type: ignore[arg-type]
    recorder._clock = lambda: float("nan")
    with pytest.raises(ValueError, match="finite monotonic"):
        recorder.mark_response()


def test_recorder_active_duplicate_and_completion_state_guards() -> None:
    clock = _FakeClock()
    recorder = ServiceTraceRecorder(
        trace_id="1" * 32,
        event_timestamp="2026-11-03T12:00:00+09:00",
        clock=clock,
    )
    with recorder.measure("queueing"):
        with pytest.raises(ValueError, match="already recorded"):
            with recorder.measure("queueing"):
                pass
        with pytest.raises(ValueError, match="active stages"):
            recorder.mark_response()
        recorder._response_latency_ms = 0.0
        with pytest.raises(ValueError, match="active stages"):
            recorder.mark_primary_log_commit()
        recorder._response_latency_ms = None
        with pytest.raises(ValueError, match="active stages"):
            recorder.complete(
                _proposal("r0"), completed_at="2026-11-03T12:00:01+09:00"
            )

    recorder.mark_response()
    with pytest.raises(ValueError, match="already marked"):
        recorder.mark_response()
    recorder.record_stage_ms("proposal_log_write", 1.0)
    recorder.mark_primary_log_commit()
    with pytest.raises(ValueError, match="already marked"):
        recorder.mark_primary_log_commit()


def test_recorder_completion_validates_missing_events_timestamp_and_reuse() -> None:
    clock = _FakeClock()
    recorder = ServiceTraceRecorder(
        trace_id="1" * 32,
        event_timestamp="2026-11-03T12:00:00+09:00",
        clock=clock,
    )
    for stage in REGISTERED_SERVICE_STAGES:
        recorder.record_stage_ms(stage, 1.0)
    with pytest.raises(ValueError, match="response has not been marked"):
        recorder.complete(_proposal("r0"), completed_at="2026-11-03T12:00:01+09:00")
    clock.advance_ms(2.0)
    recorder.mark_response()
    with pytest.raises(ValueError, match="log commit has not been marked"):
        recorder.complete(_proposal("r0"), completed_at="2026-11-03T12:00:01+09:00")
    clock.advance_ms(1.0)
    recorder.mark_primary_log_commit()
    with pytest.raises(ValueError, match="timestamp differs"):
        recorder.complete(
            replace(_proposal("r0"), event_timestamp="2026-11-03T12:00:01+09:00"),
            completed_at="2026-11-03T12:00:01+09:00",
        )
    completion = recorder.complete(
        _proposal("r0"), completed_at="2026-11-03T12:00:01+09:00"
    )
    assert completion.request_id == "r0"
    for operation in (
        lambda: recorder.record_stage_ms("rpc", 1.0),
        lambda: recorder.mark_response(),
        lambda: recorder.mark_primary_log_commit(),
        lambda: recorder.complete(
            _proposal("r0"), completed_at="2026-11-03T12:00:01+09:00"
        ),
    ):
        with pytest.raises(ValueError, match="already completed"):
            operation()
    with pytest.raises(ValueError, match="already completed"):
        with recorder.measure("rpc"):
            pass


def test_recorder_rejects_commit_that_precedes_recorded_response() -> None:
    clock = _FakeClock()
    recorder = ServiceTraceRecorder(
        trace_id="1" * 32,
        event_timestamp="2026-11-03T12:00:00+09:00",
        clock=clock,
    )
    recorder.record_stage_ms("proposal_log_write", 0.0)
    recorder._response_latency_ms = 1.0
    with pytest.raises(ValueError, match="may not precede response"):
        recorder.mark_primary_log_commit()


def test_timestamp_and_percentile_helpers_reject_empty_or_naive_values() -> None:
    payload = _completion("r0").to_dict()
    payload["event_timestamp"] = ""
    with pytest.raises(ValueError, match="nonempty ISO-8601"):
        load_service_trace_completion(payload)
    payload = _completion("r0").to_dict()
    payload["event_timestamp"] = "2026-11-03T12:00:00"
    with pytest.raises(ValueError, match="timezone"):
        load_service_trace_completion(payload)
    with pytest.raises(ValueError, match="at least one"):
        _percentiles([])


@pytest.mark.parametrize(
    ("proposal_changes", "summary_changes", "reason"),
    [
        ({"serving_mode": "shadow"}, {"serving_mode": "serve"}, "serving_mode_mismatch"),
        ({"served_slate": (99, 11, 100, 101)}, {}, "protected_core_changed"),
        ({"served_slate": (10, 11, 12, 13)}, {}, "served_slate_differs_from_proposal"),
        (
            {"serving_mode": "shadow", "served_slate": (10, 11, 100, 101)},
            {"serving_mode": "shadow"},
            "shadow_changed_served_slate",
        ),
        ({"rechecked": False}, {"maximum_recheck_failure_rate": 0.0}, "final_recheck_failed"),
        ({"selected_safety_lcb": -1.0}, {}, "certificate_violation"),
        ({"predicate_errors": 1}, {}, "predicate_error"),
        ({"selected_operation_indices": ()}, {"maximum_fallback_rate": 0.0}, "fallback_rate_above_gate"),
    ],
)
def test_production_summary_records_each_invalid_runtime_condition(
    proposal_changes: dict[str, object],
    summary_changes: dict[str, object],
    reason: str,
) -> None:
    proposal = replace(_proposal("r0"), **proposal_changes)
    summary = _summary(
        [proposal],
        [_completion("r0", proposal=proposal)],
        minimum_requests=1,
        maximum_invalid_record_rate=1.0,
        **summary_changes,
    )
    assert reason in dict(summary.invalid_reason_counts) or reason in summary.reasons


def test_production_summary_rejects_incomplete_attached_trace() -> None:
    proposal = _proposal("r0")
    incomplete = create_service_latency_trace(
        response_latency_ms=20.0, log_commit_latency_ms=25.0
    )
    completion = replace(
        _completion("r0", proposal=proposal), service_latency_trace=incomplete
    )
    with pytest.raises(ValueError, match="at least one value"):
        _summary(
            [proposal],
            [completion],
            minimum_requests=1,
            maximum_invalid_record_rate=1.0,
        )
