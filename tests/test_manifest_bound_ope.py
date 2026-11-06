from __future__ import annotations

from dataclasses import replace
from hashlib import sha256

import pytest

from safe_slate import (
    audit_manifest_bound_ope_readiness,
    canonical_population_positivity_evidence_bytes,
    create_immutable_input,
    create_input_manifest,
    create_observed_input_identity,
    create_ope_evaluation_contract,
    create_population_positivity_evidence,
    create_statistical_population_positivity_evidence,
)


def _digest(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def _positivity_evidence(**overrides):
    values = {
        "evidence_id": "positivity-v1",
        "decision": "established",
        "population_identity": "eligible-population-v1",
        "logging_policy_identity": "baseline-v1",
        "target_policy_identity": "target-v2",
        "audited_at": "2026-10-01T00:00:00+09:00",
        "auditor_identity": "independent-design-auditor",
        "method": "registered logger lower-envelope audit",
        "minimum_logging_probability_lower_bound": 0.01,
    }
    values.update(overrides)
    return create_population_positivity_evidence(**values)


def _statistical_positivity_evidence(**overrides):
    values = {
        "evidence_id": "positivity-v2",
        "decision": "established",
        "population_identity": "eligible-population-v1",
        "logging_policy_identity": "baseline-v1",
        "target_policy_identity": "target-v2",
        "audited_at": "2026-08-31T00:00:00+09:00",
        "auditor_identity": "independent-design-auditor",
        "method": "epochwise metric-net exact lower audit",
        "minimum_logging_probability_lower_bound": 0.01,
        "guarantee_kind": "finite_sample",
        "simultaneous_failure_probability": 0.025,
        "validity_start": "2026-09-01T00:00:00+09:00",
        "validity_end": "2026-09-02T00:00:00+09:00",
        "registered_epoch_count": 2,
        "audit_design_identity": "metric-net-v2",
        "source_certificate_sha256": "c" * 64,
    }
    values.update(overrides)
    return create_statistical_population_positivity_evidence(**values)


def _manifest(
    *,
    manifest_id: str = "manifest-v1",
    population_identity: str = "eligible-population-v1",
    created_at: str = "2026-10-02T00:00:00+09:00",
    label_cutoff: str = "2026-10-02T00:00:00+09:00",
    complete_ope_roles: bool = True,
    positivity_evidence=None,
):
    feature = create_immutable_input(
        input_id="features",
        semantic_role="candidate_action_log",
        canonical_uri="gs://research-inputs/features.parquet",
        version="v1",
        generation="101",
        metageneration="1",
        byte_size=100,
        sha256_digest=_digest("features"),
        snapshot_time="2026-09-02T00:00:00+09:00",
        temporal_scope="feature",
        schema_version="1.0",
    )
    labels = create_immutable_input(
        input_id="labels",
        semantic_role="settled_purchase_labels",
        canonical_uri="gs://research-inputs/labels.parquet",
        version="v1",
        generation="102",
        metageneration="1",
        byte_size=100,
        sha256_digest=_digest("labels"),
        snapshot_time=label_cutoff,
        temporal_scope="label",
        schema_version="1.0",
    )
    inputs = [feature, labels]
    required_roles = ["candidate_action_log", "settled_purchase_labels"]
    if complete_ope_roles:
        propensities = create_immutable_input(
            input_id="propensities",
            semantic_role="propensity_or_interval",
            canonical_uri="gs://research-inputs/propensities.parquet",
            version="v1",
            generation="103",
            metageneration="1",
            byte_size=100,
            sha256_digest=_digest("propensities"),
            snapshot_time="2026-09-02T00:00:00+09:00",
            temporal_scope="feature",
            schema_version="1.0",
        )
        user_blocks = create_immutable_input(
            input_id="user-blocks",
            semantic_role="pseudonymous_user_blocks",
            canonical_uri="gs://research-inputs/user-blocks.parquet",
            version="v1",
            generation="104",
            metageneration="1",
            byte_size=100,
            sha256_digest=_digest("user-blocks"),
            snapshot_time="2026-09-02T00:00:00+09:00",
            temporal_scope="feature",
            schema_version="1.0",
        )
        positivity = positivity_evidence or _positivity_evidence(
            population_identity=population_identity
        )
        positivity_bytes = canonical_population_positivity_evidence_bytes(positivity)
        positivity_input = create_immutable_input(
            input_id="population-positivity",
            semantic_role="population_positivity_evidence",
            canonical_uri="gs://research-inputs/population-positivity.json",
            version=positivity.evidence_id,
            generation="105",
            metageneration="1",
            byte_size=len(positivity_bytes),
            sha256_digest=sha256(positivity_bytes).hexdigest(),
            snapshot_time=positivity.audited_at,
            temporal_scope="atemporal",
            schema_version=positivity.schema_version,
        )
        inputs.extend((propensities, user_blocks, positivity_input))
        required_roles.extend(
            (
                "propensity_or_interval",
                "pseudonymous_user_blocks",
                "population_positivity_evidence",
            )
        )
    return create_input_manifest(
        manifest_id=manifest_id,
        population_identity=population_identity,
        created_at=created_at,
        feature_cutoff="2026-09-02T00:00:00+09:00",
        label_cutoff=label_cutoff,
        required_roles=required_roles,
        inputs=inputs,
    )


def _observations(manifest):
    return [
        create_observed_input_identity(
            input_id=value.input_id,
            canonical_uri=value.canonical_uri,
            version=value.version,
            generation=value.generation,
            metageneration=value.metageneration,
            byte_size=value.byte_size,
            sha256_digest=value.sha256,
        )
        for value in manifest.inputs
    ]


def _contract(
    manifest,
    *,
    population_identity: str = "eligible-population-v1",
    propensity_mode: str = "known",
    maximum_missing_outcome_rate: float = 0.0,
    minimum_user_effective_sample_size: float = 1.5,
    final_window_end: str = "2026-09-02T00:00:00+09:00",
    earliest_analysis_time: str = "2026-10-02T00:00:00+09:00",
):
    return create_ope_evaluation_contract(
        evaluation_id="manifest-bound-ope-v1",
        scientific_question="Does the target improve settled purchase value?",
        baseline_policy="baseline-v1",
        target_policy="target-v2",
        population_identity=population_identity,
        pilot_window_start="2026-08-01T00:00:00+09:00",
        pilot_window_end="2026-08-02T00:00:00+09:00",
        final_window_start="2026-09-01T00:00:00+09:00",
        final_window_end=final_window_end,
        earliest_analysis_time=earliest_analysis_time,
        label_settlement_rule="30-day settled purchase window",
        primary_metric="settled purchases per request",
        primary_contrast="target minus logging policy value",
        propensity_mode=propensity_mode,  # type: ignore[arg-type]
        input_manifest_digest=manifest.manifest_digest,
        code_revision="d564f87",
        dependency_lock_digest="b" * 64,
        output_namespace="evidence/manifest-bound-ope-v1",
        minimum_support_rate_for_full_estimand=1.0,
        minimum_support_rate_for_partial_estimand=0.8,
        minimum_user_effective_sample_size=minimum_user_effective_sample_size,
        minimum_purchase_events=2,
        maximum_missing_outcome_rate=maximum_missing_outcome_rate,
        maximum_p99_importance_weight=5.0,
        weight_clip=10.0,
        bootstrap_repetitions=10,
        bootstrap_seed=7,
        confidence_level=0.95,
        outcome_lower_bound=0.0,
        outcome_upper_bound=1.0,
        minimum_raw_requests=4,
        maximum_runtime_seconds=60.0,
        maximum_memory_mb=512,
        maximum_output_bytes=1_000_000,
    )


def _rows():
    return {
        "request_ids": ["r0", "r1", "r2", "r3"],
        "user_ids": ["u0", "u0", "u1", "u1"],
        "guaranteed_target_support_mass": [1.0] * 4,
        "target_probability_of_logged_action": [0.5] * 4,
        "logging_propensities": [0.5] * 4,
        "purchase_events": [1, 0, 1, 0],
        "fold_ids": [0, 0, 1, 1],
        "candidate_sets_reconstructable": True,
        "labels_settled": True,
        "missingness_independent": True,
    }


def test_exact_observations_unlock_contract_bound_readiness() -> None:
    manifest = _manifest()
    contract = _contract(manifest)

    result = audit_manifest_bound_ope_readiness(
        contract,
        manifest,
        _observations(manifest),
        analysis_timestamp="2026-10-02T00:00:00+09:00",
        population_positivity_evidence=_positivity_evidence(),
        **_rows(),
    )

    assert result.immutable_inputs_verified
    assert result.decision == "pass"
    assert result.readiness is not None
    assert result.readiness.evaluation_contract_digest == contract.contract_digest
    assert result.readiness.standard_ope_authorized
    assert result.population_positivity_evidence_input_id == "population-positivity"
    assert result.population_positivity_evidence_id == "positivity-v1"
    assert result.population_positivity_evidence_sha256 is not None
    assert not result.production_rollout_authorized


def test_statistical_positivity_evidence_preserves_alpha_and_validity() -> None:
    evidence = _statistical_positivity_evidence()
    manifest = _manifest(positivity_evidence=evidence)
    contract = _contract(manifest)

    result = audit_manifest_bound_ope_readiness(
        contract,
        manifest,
        _observations(manifest),
        analysis_timestamp="2026-10-02T00:00:00+09:00",
        population_positivity_evidence=evidence,
        **_rows(),
    )

    assert result.decision == "pass"
    assert result.population_positivity_schema_version == evidence.schema_version
    assert result.population_positivity_guarantee_kind == "finite_sample"
    assert result.population_positivity_simultaneous_failure_probability == pytest.approx(
        0.025
    )
    assert result.population_positivity_validity_start == evidence.validity_start
    assert result.population_positivity_validity_end == evidence.validity_end
    assert result.ope_failure_probability == pytest.approx(0.05)
    assert result.joint_failure_probability_upper_bound == pytest.approx(0.075)
    assert result.joint_confidence_level_lower_bound == pytest.approx(0.925)
    assert result.joint_confidence_composition == "union_bound_no_independence_assumed"
    assert result.readiness is not None
    assert result.readiness.positivity_failure_probability == pytest.approx(0.025)
    assert result.readiness.ope_failure_probability == pytest.approx(0.05)
    assert result.readiness.joint_failure_probability_upper_bound == pytest.approx(
        0.075
    )
    assert result.readiness.joint_confidence_level_lower_bound == pytest.approx(0.925)
    serialized = result.to_dict()
    assert serialized["joint_confidence_level_lower_bound"] == pytest.approx(0.925)
    assert serialized["readiness"]["joint_confidence_level_lower_bound"] == pytest.approx(0.925)


def test_statistical_positivity_must_cover_full_final_window() -> None:
    evidence = _statistical_positivity_evidence(
        validity_start="2026-09-01T12:00:00+09:00"
    )
    manifest = _manifest(positivity_evidence=evidence)
    contract = _contract(manifest)
    result = audit_manifest_bound_ope_readiness(
        contract,
        manifest,
        _observations(manifest),
        analysis_timestamp="2026-10-02T00:00:00+09:00",
        population_positivity_evidence=evidence,
        **_rows(),
    )
    assert result.decision == "evidence_rejected"
    assert "population_positivity_validity_starts_after_final_window" in result.reasons


def test_deterministic_v2_adds_zero_to_ope_failure_probability() -> None:
    evidence = _statistical_positivity_evidence(
        guarantee_kind="deterministic",
        simultaneous_failure_probability=0.0,
    )
    manifest = _manifest(positivity_evidence=evidence)
    contract = _contract(manifest)
    result = audit_manifest_bound_ope_readiness(
        contract,
        manifest,
        _observations(manifest),
        analysis_timestamp="2026-10-02T00:00:00+09:00",
        population_positivity_evidence=evidence,
        **_rows(),
    )
    assert result.decision == "pass"
    assert result.joint_failure_probability_upper_bound == pytest.approx(0.05)
    assert result.joint_confidence_level_lower_bound == pytest.approx(0.95)


def test_changed_input_rejects_evidence_before_row_audit() -> None:
    manifest = _manifest()
    contract = _contract(manifest)
    observations = _observations(manifest)
    first = observations[0]
    observations[0] = create_observed_input_identity(
        input_id=first.input_id,
        canonical_uri=first.canonical_uri,
        version=first.version,
        generation="changed",
        metageneration=first.metageneration,
        byte_size=first.byte_size,
        sha256_digest=first.sha256,
    )

    result = audit_manifest_bound_ope_readiness(
        contract,
        manifest,
        observations,
        analysis_timestamp="2026-10-02T00:00:00+09:00",
        population_positivity_evidence=_positivity_evidence(),
    )

    assert not result.immutable_inputs_verified
    assert result.decision == "evidence_rejected"
    assert result.readiness is None
    assert any(":generation" in value for value in result.reasons)


def test_manifest_contract_population_mismatch_rejects_evidence() -> None:
    manifest = _manifest(population_identity="different-population")
    contract = _contract(manifest, population_identity="eligible-population-v1")

    result = audit_manifest_bound_ope_readiness(
        contract,
        manifest,
        _observations(manifest),
        analysis_timestamp="2026-10-02T00:00:00+09:00",
        population_positivity_evidence=_positivity_evidence(),
    )

    assert result.decision == "evidence_rejected"
    assert "input_manifest_population_mismatch" in result.reasons


def test_missing_required_ope_input_role_rejects_evidence() -> None:
    manifest = _manifest(complete_ope_roles=False)
    contract = _contract(manifest)

    result = audit_manifest_bound_ope_readiness(
        contract,
        manifest,
        _observations(manifest),
        analysis_timestamp="2026-10-02T00:00:00+09:00",
        population_positivity_evidence=_positivity_evidence(),
    )

    assert result.decision == "evidence_rejected"
    assert "required_input_role_missing:propensity_or_interval" in result.reasons
    assert "required_input_role_missing:pseudonymous_user_blocks" in result.reasons
    assert "required_input_role_missing:population_positivity_evidence" in result.reasons


def test_labels_pending_does_not_run_readiness() -> None:
    positivity = _positivity_evidence(audited_at="2026-09-29T00:00:00+09:00")
    manifest = _manifest(
        created_at="2026-09-30T00:00:00+09:00",
        label_cutoff="2026-09-30T00:00:00+09:00",
        positivity_evidence=positivity,
    )
    contract = _contract(manifest)

    result = audit_manifest_bound_ope_readiness(
        contract,
        manifest,
        _observations(manifest),
        analysis_timestamp="2026-10-01T00:00:00+09:00",
        population_positivity_evidence=positivity,
    )

    assert result.immutable_inputs_verified
    assert result.decision == "labels_pending"
    assert result.readiness is None


def test_valid_inputs_preserve_statistical_gate_failure() -> None:
    manifest = _manifest()
    contract = _contract(manifest)
    rows = _rows()
    rows["guaranteed_target_support_mass"] = [0.5] * 4

    result = audit_manifest_bound_ope_readiness(
        contract,
        manifest,
        _observations(manifest),
        analysis_timestamp="2026-10-02T00:00:00+09:00",
        population_positivity_evidence=_positivity_evidence(),
        **rows,
    )

    assert result.immutable_inputs_verified
    assert result.decision == "fail"
    assert result.readiness is not None
    assert "target_support_below_partial_gate" in result.reasons


def test_boolean_without_positivity_evidence_is_rejected_before_row_audit() -> None:
    manifest = _manifest(complete_ope_roles=False)
    contract = _contract(manifest)
    rows = _rows()
    rows["population_positivity_established"] = True

    result = audit_manifest_bound_ope_readiness(
        contract,
        manifest,
        _observations(manifest),
        analysis_timestamp="2026-10-02T00:00:00+09:00",
        **rows,
    )

    assert result.decision == "evidence_rejected"
    assert result.readiness is None
    assert "population_positivity_evidence_missing" in result.reasons
    assert (
        "population_positivity_boolean_must_be_evidence_derived" in result.reasons
    )


def test_missing_positivity_object_rejects_even_when_manifest_role_exists() -> None:
    manifest = _manifest()
    contract = _contract(manifest)

    result = audit_manifest_bound_ope_readiness(
        contract,
        manifest,
        _observations(manifest),
        analysis_timestamp="2026-10-02T00:00:00+09:00",
        **_rows(),
    )

    assert result.decision == "evidence_rejected"
    assert "population_positivity_evidence_missing" in result.reasons


def test_positivity_identity_digest_and_decision_are_contract_bound() -> None:
    evidence = _positivity_evidence()
    manifest = _manifest(positivity_evidence=evidence)
    contract = _contract(manifest)

    cases = (
        (
            _positivity_evidence(evidence_id="different-evidence"),
            "population_positivity_evidence_identity_mismatch",
        ),
        (
            _positivity_evidence(population_identity="different-population"),
            "population_positivity_digest_mismatch",
        ),
        (
            _positivity_evidence(logging_policy_identity="different-logger"),
            "population_positivity_logger_mismatch",
        ),
        (
            _positivity_evidence(target_policy_identity="different-target"),
            "population_positivity_target_mismatch",
        ),
        (
            _positivity_evidence(
                decision="not_established",
                minimum_logging_probability_lower_bound=None,
            ),
            "population_positivity_not_established",
        ),
        (
            replace(evidence, evidence_digest="0" * 64),
            "population_positivity_evidence_invalid_or_tampered",
        ),
    )
    for supplied, reason in cases:
        result = audit_manifest_bound_ope_readiness(
            contract,
            manifest,
            _observations(manifest),
            analysis_timestamp="2026-10-02T00:00:00+09:00",
            population_positivity_evidence=supplied,
            **_rows(),
        )
        assert result.decision == "evidence_rejected"
        assert result.readiness is None
        assert reason in result.reasons


def test_manifest_bound_readiness_rejects_invalid_public_arguments_and_times() -> None:
    manifest = _manifest()
    contract = _contract(manifest)
    with pytest.raises(ValueError, match="untampered OpeEvaluationContract"):
        audit_manifest_bound_ope_readiness(
            "bad",  # type: ignore[arg-type]
            manifest,
            _observations(manifest),
            analysis_timestamp="2026-10-02T00:00:00+09:00",
        )
    with pytest.raises(ValueError, match="untampered manifest"):
        audit_manifest_bound_ope_readiness(
            contract,
            "bad",  # type: ignore[arg-type]
            (),
            analysis_timestamp="2026-10-02T00:00:00+09:00",
        )
    for timestamp, message in [
        ("not-a-time", "ISO-8601"),
        ("2026-10-02T00:00:00", "timezone"),
    ]:
        with pytest.raises(ValueError, match=message):
            audit_manifest_bound_ope_readiness(
                contract,
                manifest,
                _observations(manifest),
                analysis_timestamp=timestamp,
                population_positivity_evidence=_positivity_evidence(),
                **_rows(),
            )


def test_manifest_bound_readiness_reports_digest_and_temporal_mismatches() -> None:
    manifest = _manifest()
    other = _manifest(manifest_id="other")
    digest_mismatch = audit_manifest_bound_ope_readiness(
        _contract(other),
        manifest,
        _observations(manifest),
        analysis_timestamp="2026-10-02T00:00:00+09:00",
        population_positivity_evidence=_positivity_evidence(),
        **_rows(),
    )
    assert "input_manifest_contract_digest_mismatch" in digest_mismatch.reasons

    feature_mismatch = audit_manifest_bound_ope_readiness(
        _contract(manifest, final_window_end="2026-09-03T00:00:00+09:00"),
        manifest,
        _observations(manifest),
        analysis_timestamp="2026-10-02T00:00:00+09:00",
        population_positivity_evidence=_positivity_evidence(),
        **_rows(),
    )
    assert "feature_cutoff_not_equal_to_final_window_end" in feature_mismatch.reasons

    label_mismatch = audit_manifest_bound_ope_readiness(
        _contract(manifest, earliest_analysis_time="2026-10-01T00:00:00+09:00"),
        manifest,
        _observations(manifest),
        analysis_timestamp="2026-10-02T00:00:00+09:00",
        population_positivity_evidence=_positivity_evidence(),
        **_rows(),
    )
    assert "label_cutoff_outside_settlement_boundary" in label_mismatch.reasons

    created_after = audit_manifest_bound_ope_readiness(
        _contract(manifest, earliest_analysis_time="2026-10-01T00:00:00+09:00"),
        manifest,
        _observations(manifest),
        analysis_timestamp="2026-10-01T00:00:00+09:00",
        population_positivity_evidence=_positivity_evidence(),
        **_rows(),
    )
    assert "input_manifest_created_after_analysis" in created_after.reasons


def test_manifest_bound_readiness_checks_positivity_schema_snapshot_and_audit_time() -> None:
    evidence = _positivity_evidence()
    base = _manifest(positivity_evidence=evidence)

    def changed_manifest(**changes):
        inputs = [
            replace(value, **changes)
            if value.semantic_role == "population_positivity_evidence"
            else value
            for value in base.inputs
        ]
        return create_input_manifest(
            manifest_id="changed-positivity-input",
            population_identity=base.population_identity,
            created_at=base.created_at,
            feature_cutoff=base.feature_cutoff,
            label_cutoff=base.label_cutoff,
            required_roles=base.required_roles,
            inputs=inputs,
        )

    for manifest, supplied, reason in [
        (
            changed_manifest(schema_version="wrong"),
            evidence,
            "population_positivity_schema_mismatch",
        ),
        (
            changed_manifest(snapshot_time="2026-09-30T00:00:00+09:00"),
            evidence,
            "population_positivity_snapshot_mismatch",
        ),
    ]:
        result = audit_manifest_bound_ope_readiness(
            _contract(manifest),
            manifest,
            _observations(manifest),
            analysis_timestamp="2026-10-02T00:00:00+09:00",
            population_positivity_evidence=supplied,
            **_rows(),
        )
        assert reason in result.reasons

    late_evidence = _positivity_evidence(audited_at="2026-10-03T00:00:00+09:00")
    late_manifest = _manifest(positivity_evidence=late_evidence)
    late = audit_manifest_bound_ope_readiness(
        _contract(late_manifest),
        late_manifest,
        _observations(late_manifest),
        analysis_timestamp="2026-10-04T00:00:00+09:00",
        population_positivity_evidence=late_evidence,
        **_rows(),
    )
    assert "population_positivity_audited_after_manifest" in late.reasons

    short_evidence = _statistical_positivity_evidence(
        validity_end="2026-09-01T12:00:00+09:00"
    )
    short_manifest = _manifest(positivity_evidence=short_evidence)
    short = audit_manifest_bound_ope_readiness(
        _contract(short_manifest),
        short_manifest,
        _observations(short_manifest),
        analysis_timestamp="2026-10-02T00:00:00+09:00",
        population_positivity_evidence=short_evidence,
        **_rows(),
    )
    assert "population_positivity_validity_ends_before_final_window" in short.reasons
