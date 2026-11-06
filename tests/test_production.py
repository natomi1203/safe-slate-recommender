from __future__ import annotations

import pytest

from safe_slate import (
    attach_service_latency,
    create_service_latency_trace,
    guarded_linear_rerank,
    guarded_rerank,
)


def _base_request(**overrides: object) -> dict[str, object]:
    request: dict[str, object] = {
        "request_id": "request-123",
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
        "support_ok": True,
        "uncertainty_valid": True,
        "distribution_shift_detected": False,
        "proposal_log_complete": True,
    }
    request.update(overrides)
    return request


def test_guarded_rerank_serves_maximum_certified_cardinality() -> None:
    result = guarded_rerank(**_base_request())

    assert result.slate == (10, 11, 100, 101)
    assert result.slate[:2] == (10, 11)
    assert result.selected_count == 2
    assert result.k_lower == result.k_upper == 2
    assert not result.fallback
    assert result.oracle.rechecked


def test_guarded_rerank_missing_runtime_evidence_fails_closed() -> None:
    request = _base_request()
    for gate in (
        "support_ok",
        "uncertainty_valid",
        "distribution_shift_detected",
        "proposal_log_complete",
    ):
        del request[gate]

    result = guarded_rerank(**request)

    assert result.slate == (10, 11, 12, 13)
    assert result.fallback
    assert result.fallback_reason == "support_gate"
    assert result.proposal_log.gate_evidence is None


def test_guarded_linear_rerank_projects_and_logs_shared_uncertainty() -> None:
    result = guarded_linear_rerank(
        request_id="linear-request",
        uncertainty_snapshot_id="covariance-snapshot-v17",
        contrast_input_id="contrast-linear-request",
        baseline_item_ids=[10, 11, 12, 13],
        protected_count=2,
        remove_positions=[2, 3],
        add_item_ids=[100, 101],
        reward_deltas=[0.2, 0.1],
        safety_mean_deltas=[0.04, 0.03],
        contrast_features=[[1.0, 0.0], [0.5, 1.0]],
        parameter_covariance=[[0.0001, 0.0], [0.0, 0.0001]],
        baseline_metric=0.5,
        degradation_tolerance=0.02,
        max_replacements=2,
        confidence_radius=1.0,
        support_ok=True,
        uncertainty_valid=True,
        distribution_shift_detected=False,
        proposal_log_complete=True,
        model_version="ranker-v17",
        context={"surface": "home"},
    )

    assert result.slate[:2] == (10, 11)
    assert result.k_lower == 2
    assert result.proposal_log.safety_variances == pytest.approx(
        (0.0001, 0.000125)
    )
    assert result.proposal_log.covariance is None
    assert result.proposal_log.selected_safety_variance == pytest.approx(0.000325)
    metadata = result.proposal_log.context["safe_slate_uncertainty"]
    assert metadata["method"] == "linear_contrast_covariance"
    assert metadata["snapshot_id"] == "covariance-snapshot-v17"
    assert metadata["contrast_input_id"] == "contrast-linear-request"
    assert metadata["feature_dimension"] == 2
    assert metadata["operation_count"] == 2
    assert metadata["representation"] == "factorized"
    assert metadata["matrix_digest_encoding"] == "json-v1"
    assert len(metadata["contrast_features_sha256"]) == 64
    assert len(metadata["parameter_covariance_sha256"]) == 64
    assert metadata["requested_guardrail_budget_ms"] == 10
    assert metadata["finalization_reserve_ms"] == 2
    assert 0 <= metadata["oracle_budget_ms"] <= 10
    assert metadata["pre_oracle_elapsed_ms"] >= 0.0
    assert result.proposal_log.timeout_ms == metadata["oracle_budget_ms"]


def test_guarded_linear_rerank_uncertainty_gate_fails_closed() -> None:
    result = guarded_linear_rerank(
        request_id="linear-gated",
        uncertainty_snapshot_id="covariance-snapshot-v17",
        baseline_item_ids=[10, 11, 12],
        protected_count=2,
        remove_positions=[2],
        add_item_ids=[100],
        reward_deltas=[1.0],
        safety_mean_deltas=[1.0],
        contrast_features=[[1.0]],
        parameter_covariance=[[0.01]],
        baseline_metric=0.5,
        degradation_tolerance=0.02,
        max_replacements=1,
        support_ok=True,
        uncertainty_valid=False,
        distribution_shift_detected=False,
        proposal_log_complete=True,
    )

    assert result.slate == (10, 11, 12)
    assert result.fallback_reason == "uncertainty_gate"
    assert result.k_lower == result.k_upper == 0


def test_guarded_linear_rerank_missing_runtime_evidence_fails_closed() -> None:
    result = guarded_linear_rerank(
        request_id="linear-default-gates",
        uncertainty_snapshot_id="covariance-snapshot-v17",
        baseline_item_ids=[10, 11, 12],
        protected_count=2,
        remove_positions=[2],
        add_item_ids=[100],
        reward_deltas=[1.0],
        safety_mean_deltas=[1.0],
        contrast_features=[[1.0]],
        parameter_covariance=[[0.01]],
        baseline_metric=0.5,
        degradation_tolerance=0.02,
        max_replacements=1,
    )

    assert result.slate == (10, 11, 12)
    assert result.fallback_reason == "support_gate"


def test_degradation_tolerance_is_translated_to_relative_threshold() -> None:
    result = guarded_rerank(
        **_base_request(
            remove_positions=[2],
            add_item_ids=[100],
            reward_deltas=[1.0],
            safety_mean_deltas=[-0.04],
            safety_variances=[0.0],
            degradation_tolerance=0.1,
            max_replacements=1,
        )
    )

    assert result.selected_count == 1
    assert result.proposal_log.safety_threshold == pytest.approx(-0.05)
    assert result.oracle.safety_lcb == pytest.approx(-0.04)
    assert result.proposal_log.selected_safety_lcb == pytest.approx(-0.04)


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"support_ok": False}, "support_gate"),
        ({"uncertainty_valid": False}, "uncertainty_gate"),
        ({"distribution_shift_detected": True}, "distribution_shift_gate"),
        ({"proposal_log_complete": False}, "proposal_log_gate"),
    ],
)
def test_failed_serving_gate_falls_back_to_unchanged_baseline(
    overrides: dict[str, object], reason: str
) -> None:
    result = guarded_rerank(**_base_request(**overrides))

    assert result.slate == (10, 11, 12, 13)
    assert result.selected_count == 0
    assert result.k_lower == result.k_upper == 0
    assert result.fallback
    assert result.fallback_reason == reason
    assert result.proposal_log.gate_reason == reason
    assert result.proposal_log.fallback_reason == reason
    assert result.proposal_log.requested_max_replacements == 2
    assert result.proposal_log.effective_max_replacements == 0
    assert result.oracle.rechecked


def test_proposal_log_retains_selected_and_unselected_candidates() -> None:
    result = guarded_rerank(
        **_base_request(
            remove_positions=[2, 2, 3],
            add_item_ids=[100, 102, 101],
            reward_deltas=[0.3, 0.2, 0.1],
            safety_mean_deltas=[0.04, 0.02, 0.03],
            safety_variances=[0.0, 0.0, 0.0],
            model_version="ranker-v7",
            context={"surface": "home", "country": "JP"},
        )
    )

    log = result.proposal_log
    assert len(log.selected_flags) == 3
    assert any(log.selected_flags)
    assert not all(log.selected_flags)
    assert log.add_item_ids == (100, 102, 101)
    assert log.safety_mean_deltas == (0.04, 0.02, 0.03)
    assert log.learning_update_eligible
    serialized = log.to_dict()
    assert serialized["request_id"] == "request-123"
    assert serialized["model_version"] == "ranker-v7"
    assert serialized["context"] == {"surface": "home", "country": "JP"}
    assert serialized["selected_safety_lcb"] == pytest.approx(result.oracle.safety_lcb)
    assert serialized["upper_bound_source"] == result.oracle.upper_bound_source
    assert "outcome" not in serialized


def test_shadow_mode_computes_plan_but_serves_unchanged_baseline() -> None:
    result = guarded_rerank(
        **_base_request(
            serving_mode="shadow",
            event_timestamp="2026-10-01T12:00:00+09:00",
        )
    )

    assert result.shadow
    assert result.serving_mode == "shadow"
    assert result.slate == (10, 11, 12, 13)
    assert result.proposed_slate == (10, 11, 100, 101)
    assert result.selected_count == 2
    assert not result.fallback
    assert result.proposal_log.served_slate == (10, 11, 12, 13)
    assert result.proposal_log.proposed_slate == (10, 11, 100, 101)
    assert not result.proposal_log.learning_update_eligible
    assert result.proposal_log.shadow_evaluation_eligible
    serialized = result.proposal_log.to_dict()
    assert serialized["schema_version"] == "1.3"
    assert serialized["serving_mode"] == "shadow"
    assert serialized["event_timestamp"] == "2026-10-01T12:00:00+09:00"


def test_shadow_mode_requires_timezone_aware_event_timestamp() -> None:
    with pytest.raises(ValueError, match="event_timestamp"):
        guarded_rerank(**_base_request(serving_mode="shadow"))
    with pytest.raises(ValueError, match="timezone"):
        guarded_rerank(
            **_base_request(
                serving_mode="shadow",
                event_timestamp="2026-10-01T12:00:00",
            )
        )


def test_service_latency_trace_is_attached_after_response_without_overwrite() -> None:
    result = guarded_rerank(**_base_request())
    trace = create_service_latency_trace(
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

    traced = attach_service_latency(result.proposal_log, trace)

    assert traced.service_latency_trace == trace
    assert trace.complete
    assert traced.to_dict()["service_latency_trace"] == trace.to_dict()
    assert result.proposal_log.service_latency_trace is None
    with pytest.raises(ValueError, match="already"):
        attach_service_latency(traced, trace)


def test_service_latency_trace_rejects_impossible_timing() -> None:
    with pytest.raises(ValueError, match="precede"):
        create_service_latency_trace(
            response_latency_ms=20.0,
            log_commit_latency_ms=19.0,
        )
    with pytest.raises(ValueError, match="may not exceed"):
        create_service_latency_trace(
            response_latency_ms=20.0,
            log_commit_latency_ms=25.0,
            model_inference_ms=30.0,
        )

    result = guarded_rerank(**_base_request())
    too_short = create_service_latency_trace(
        response_latency_ms=0.0,
        log_commit_latency_ms=0.0,
    )
    with pytest.raises(ValueError, match="Rust oracle"):
        attach_service_latency(result.proposal_log, too_short)


def test_incomplete_search_returns_rechecked_incumbent_and_open_frontier() -> None:
    result = guarded_rerank(
        **_base_request(
            remove_positions=[2, 2, 3, 3],
            add_item_ids=[100, 102, 101, 103],
            reward_deltas=[0.4, 0.3, 0.2, 0.1],
            safety_mean_deltas=[0.04, 0.03, 0.02, 0.01],
            safety_variances=[0.0, 0.0, 0.0, 0.0],
            max_expansions=1,
        )
    )

    assert result.oracle.rechecked
    assert not result.oracle.exact
    assert result.k_lower == 1
    assert result.k_upper == 2
    assert result.slate[:2] == (10, 11)
    assert result.proposal_log.first_unresolved_depth == 2
    assert result.proposal_log.expansions == 1


def test_600_item_feed_shape_preserves_560_item_protected_core() -> None:
    baseline = list(range(600))
    result = guarded_rerank(
        request_id="feed-shape",
        baseline_item_ids=baseline,
        protected_count=560,
        remove_positions=list(range(560, 600)),
        add_item_ids=list(range(10_000, 10_040)),
        reward_deltas=[float(40 - index) for index in range(40)],
        safety_mean_deltas=[0.01] * 40,
        safety_variances=[0.0] * 40,
        baseline_metric=0.1,
        degradation_tolerance=0.01,
        max_replacements=40,
        max_expansions=64,
        timeout_ms=10,
        support_ok=True,
        uncertainty_valid=True,
        distribution_shift_detected=False,
        proposal_log_complete=True,
    )

    assert result.slate[:560] == tuple(baseline[:560])
    assert result.selected_count == 40
    assert result.oracle.rechecked
    assert len(result.proposal_log.selected_flags) == 40
    assert result.proposal_log.timeout_ms == 10


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("baseline_metric", -0.1),
        ("baseline_metric", 1.1),
        ("degradation_tolerance", -0.1),
        ("degradation_tolerance", 1.0),
        ("timeout_ms", -1),
        ("serving_mode", "invalid"),
    ],
)
def test_serving_contract_rejects_invalid_values(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        guarded_rerank(**_base_request(**{field: value}))


def test_service_latency_and_attachment_reject_remaining_invalid_contracts() -> None:
    with pytest.raises(ValueError, match="latencies are required"):
        create_service_latency_trace(
            response_latency_ms=None,  # type: ignore[arg-type]
            log_commit_latency_ms=1.0,
        )
    with pytest.raises(ValueError, match="ProposalLog"):
        attach_service_latency("bad", create_service_latency_trace(response_latency_ms=1, log_commit_latency_ms=1))  # type: ignore[arg-type]

    result = guarded_rerank(**_base_request())
    with pytest.raises(ValueError, match="ServiceLatencyTrace"):
        attach_service_latency(result.proposal_log, "bad")  # type: ignore[arg-type]
    facade_too_short = create_service_latency_trace(
        response_latency_ms=20.0,
        log_commit_latency_ms=25.0,
        guardrail_facade_ms=0.0,
    )
    with pytest.raises(ValueError, match="facade latency"):
        attach_service_latency(result.proposal_log, facade_too_short)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"baseline_item_ids": [10, 10]}, "unique values"),
        ({"remove_group_ids": [True, 1]}, "must contain integers"),
        ({"remove_group_caps": {True: 1}}, "keys must be integers"),
        ({"operation_conflicts": [(0,)]}, "contain two indices"),
        ({"covariance": [[float("nan")]]}, "must be finite"),
        ({"request_id": ""}, "non-empty string"),
        ({"event_timestamp": ""}, "non-empty ISO-8601"),
        ({"event_timestamp": "not-a-time"}, "must be ISO-8601"),
        ({"beam_width": 0}, "positive integers"),
        ({"gate_assessment": "bad"}, "ServingGateAssessment"),
        ({"support_ok": 1}, "must be a boolean"),
        ({"model_version": ""}, "non-empty string or None"),
        ({"context": []}, "mapping or None"),
        ({"protected_count": 5}, "cannot exceed"),
    ],
)
def test_guarded_rerank_rejects_remaining_malformed_inputs(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        guarded_rerank(**_base_request(**overrides))


def test_no_available_replacement_uses_certified_fallback_reason() -> None:
    result = guarded_rerank(
        **_base_request(
            remove_positions=[],
            add_item_ids=[],
            reward_deltas=[],
            safety_mean_deltas=[],
            safety_variances=[],
        )
    )
    assert result.fallback
    assert result.fallback_reason == "no_certified_replacement"


def test_group_caps_and_operation_conflicts_reach_native_oracle() -> None:
    result = guarded_rerank(
        **_base_request(
            remove_group_ids=[1, 2],
            remove_group_caps={1: 1, 2: 1},
            add_group_ids=[7, 8],
            add_group_caps={7: 1, 8: 1},
            operation_conflicts=[(0, 1)],
        )
    )
    assert result.oracle.rechecked


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"uncertainty_snapshot_id": ""}, "non-empty string"),
        ({"contrast_input_id": ""}, "non-empty string or None"),
        ({"timeout_ms": True}, "nonnegative integer"),
        ({"finalization_reserve_ms": -1}, "nonnegative integer"),
        (
            {"context": {"safe_slate_uncertainty": {}}},
            "reserved",
        ),
    ],
)
def test_guarded_linear_rerank_rejects_remaining_wrapper_contracts(
    overrides: dict[str, object], message: str
) -> None:
    values = {
        "request_id": "linear-invalid",
        "uncertainty_snapshot_id": "snapshot",
        "contrast_input_id": "contrasts",
        "baseline_item_ids": [10, 11, 12],
        "protected_count": 2,
        "remove_positions": [2],
        "add_item_ids": [100],
        "reward_deltas": [1.0],
        "safety_mean_deltas": [1.0],
        "contrast_features": [[1.0]],
        "parameter_covariance": [[0.01]],
        "baseline_metric": 0.5,
        "degradation_tolerance": 0.02,
        "max_replacements": 1,
    }
    values.update(overrides)
    with pytest.raises(ValueError, match=message):
        guarded_linear_rerank(**values)  # type: ignore[arg-type]
