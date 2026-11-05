from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

from safe_slate import (
    audit_input_manifest_observations,
    classify_evaluation_evidence,
    create_immutable_input,
    create_input_manifest,
    create_observed_input_identity,
    create_ope_evaluation_contract,
    estimate_cross_fitted_policy_value,
    verify_evidence_bundle,
    write_evidence_bundle,
)
from safe_slate.ope import _audit_ope_readiness_with_contract


REQUIRED_INPUT_ROLES = (
    "candidate_action_logs",
    "propensity_evidence",
    "settled_labels",
    "pseudonymous_user_blocks",
    "constraint_traces",
    "shadow_records",
)


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _input_manifest(
    *,
    population_identity: str = "eligible-population-v1",
    manifest_id: str = "bundle-evaluation-inputs-v1",
):
    inputs = []
    for input_id, role in (
        ("candidate-log", "candidate_action_logs"),
        ("propensities", "propensity_evidence"),
        ("user-blocks", "pseudonymous_user_blocks"),
        ("constraints", "constraint_traces"),
        ("shadow", "shadow_records"),
    ):
        inputs.append(
            create_immutable_input(
                input_id=input_id,
                semantic_role=role,
                canonical_uri=f"gs://research-inputs/{input_id}.parquet",
                version="v1",
                generation="123",
                metageneration="1",
                byte_size=100,
                sha256_digest=sha256(input_id.encode()).hexdigest(),
                snapshot_time="2026-09-02T00:00:00+09:00",
                temporal_scope="feature",
                schema_version="1.0",
            )
        )
    inputs.append(
        create_immutable_input(
            input_id="labels",
            semantic_role="settled_labels",
            canonical_uri="gs://research-inputs/labels.parquet",
            version="v1",
            generation="124",
            metageneration="1",
            byte_size=100,
            sha256_digest=sha256(b"labels").hexdigest(),
            snapshot_time="2026-10-01T00:00:00+09:00",
            temporal_scope="label",
            schema_version="1.0",
        )
    )
    return create_input_manifest(
        manifest_id=manifest_id,
        population_identity=population_identity,
        created_at="2026-10-02T00:00:00+09:00",
        feature_cutoff="2026-09-02T00:00:00+09:00",
        label_cutoff="2026-10-02T00:00:00+09:00",
        required_roles=REQUIRED_INPUT_ROLES,
        inputs=inputs,
    )


def _input_verification(input_manifest):
    observations = [
        create_observed_input_identity(
            input_id=value.input_id,
            canonical_uri=value.canonical_uri,
            version=value.version,
            generation=value.generation,
            metageneration=value.metageneration,
            byte_size=value.byte_size,
            sha256_digest=value.sha256,
        )
        for value in input_manifest.inputs
    ]
    return audit_input_manifest_observations(input_manifest, observations)


def _components(**contract_overrides: object):
    input_manifest = _input_manifest(
        population_identity=str(
            contract_overrides.get("population_identity", "eligible-population-v1")
        )
    )
    input_verification = _input_verification(input_manifest)
    contract_values: dict[str, object] = {
        "evaluation_id": "bundle-evaluation-v1",
        "scientific_question": "Does the target improve purchase value?",
        "baseline_policy": "baseline-v1",
        "target_policy": "target-v2",
        "population_identity": "eligible-population-v1",
        "pilot_window_start": "2026-08-01T00:00:00+09:00",
        "pilot_window_end": "2026-08-02T00:00:00+09:00",
        "final_window_start": "2026-09-01T00:00:00+09:00",
        "final_window_end": "2026-09-02T00:00:00+09:00",
        "earliest_analysis_time": "2026-10-02T00:00:00+09:00",
        "label_settlement_rule": "30-day settled purchase window",
        "primary_metric": "settled purchases per request",
        "primary_contrast": "target minus logging policy value",
        "propensity_mode": "known",
        "input_manifest_digest": input_manifest.manifest_digest,
        "code_revision": "02efa0d",
        "dependency_lock_digest": "b" * 64,
        "output_namespace": "evidence/bundle-evaluation-v1",
        "minimum_support_rate_for_full_estimand": 1.0,
        "minimum_support_rate_for_partial_estimand": 0.8,
        "minimum_user_effective_sample_size": 1.5,
        "minimum_purchase_events": 2,
        "maximum_missing_outcome_rate": 0.0,
        "maximum_p99_importance_weight": 5.0,
        "weight_clip": 10.0,
        "bootstrap_repetitions": 10,
        "bootstrap_seed": 7,
        "confidence_level": 0.95,
        "outcome_lower_bound": 0.0,
        "outcome_upper_bound": 1.0,
        "minimum_raw_requests": 4,
        "maximum_runtime_seconds": 60.0,
        "maximum_memory_mb": 512,
        "maximum_output_bytes": 1_000_000,
    }
    contract_values.update(contract_overrides)
    contract = create_ope_evaluation_contract(**contract_values)  # type: ignore[arg-type]
    rows = {
        "request_ids": ["r0", "r1", "r2", "r3"],
        "user_ids": ["u0", "u0", "u1", "u1"],
        "guaranteed_target_support_mass": [1.0] * 4,
        "target_probability_of_logged_action": [0.5] * 4,
        "logging_propensities": [0.5] * 4,
        "purchase_events": [1, 0, 1, 0],
        "fold_ids": [0, 0, 1, 1],
    }
    readiness = _audit_ope_readiness_with_contract(
        contract,
        analysis_timestamp="2026-10-02T00:00:00+09:00",
        _population_positivity_verified=True,
        **rows,
        candidate_sets_reconstructable=True,
        labels_settled=True,
        missingness_independent=True,
    )
    estimate = estimate_cross_fitted_policy_value(
        readiness,
        **rows,
        outcome_prediction_logged_action=[0.4] * 4,
        outcome_prediction_target_policy=[0.6] * 4,
        nuisance_predictions_cross_fitted=True,
        logging_propensities_are_request_time_probabilities=True,
        outcomes_match_logged_actions=True,
        bootstrap_repetitions=contract.bootstrap_repetitions,
        bootstrap_seed=contract.bootstrap_seed,
        confidence_level=contract.confidence_level,
        evaluation_contract=contract,
    )
    status = classify_evaluation_evidence(
        contract,
        analysis_timestamp="2026-10-02T00:00:00+09:00",
        collection_complete=True,
        immutable_inputs_verified=True,
        no_clobber_verified=True,
        labels_settled=True,
        label_snapshot_digest="c" * 64,
        readiness_decision=readiness.decision,
        readiness_contract_digest=readiness.evaluation_contract_digest,
        estimation_complete=True,
        final_audit_passed=True,
    )
    return input_manifest, input_verification, contract, readiness, estimate, status


def _write(path: Path, **overrides: object):
    input_manifest, input_verification, contract, readiness, estimate, status = (
        _components()
    )
    values: dict[str, object] = {
        "output_identity": contract.output_namespace,
        "contract": contract,
        "input_manifest": input_manifest,
        "input_verification": input_verification,
        "readiness": readiness,
        "estimate": estimate,
        "evidence_status": status,
        "created_at": "2026-10-02T00:00:01+09:00",
        "successful_execution_ids": ["local-evaluation-001"],
        "actual_runtime_seconds": 2.5,
        "peak_memory_mb": 128,
        "no_matching_execution_verified": True,
    }
    values.update(overrides)
    manifest = write_evidence_bundle(path, **values)  # type: ignore[arg-type]
    return (
        manifest,
        input_manifest,
        input_verification,
        contract,
        readiness,
        estimate,
        status,
    )


def _rewrite_artifact_and_markers(
    target: Path, filename: str, payload: dict[str, object]
) -> None:
    artifact_bytes = _canonical(payload)
    (target / filename).write_bytes(artifact_bytes)
    manifest_path = target / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    for artifact in manifest["artifacts"]:
        if artifact["filename"] == filename:
            artifact["byte_size"] = len(artifact_bytes)
            artifact["sha256"] = sha256(artifact_bytes).hexdigest()
    manifest_bytes = _canonical(manifest)
    manifest_path.write_bytes(manifest_bytes)
    completion_path = target / "complete.json"
    completion = json.loads(completion_path.read_bytes())
    completion["manifest_byte_size"] = len(manifest_bytes)
    completion["manifest_sha256"] = sha256(manifest_bytes).hexdigest()
    completion_path.write_bytes(_canonical(completion))


def test_no_clobber_bundle_round_trip_and_completion_marker(tmp_path: Path) -> None:
    target = tmp_path / "evidence-v1"
    (
        manifest,
        input_manifest,
        input_verification,
        contract,
        readiness,
        estimate,
        status,
    ) = _write(target)
    verification = verify_evidence_bundle(target)

    assert manifest.contract_digest == contract.contract_digest
    assert manifest.input_manifest_digest == input_manifest.manifest_digest
    assert input_verification.valid
    assert manifest.row_data_digest == readiness.row_data_digest
    assert manifest.estimate_type == type(estimate).__name__
    assert manifest.evidence_state == status.overall_state
    assert (target / "complete.json").is_file()
    assert verification.valid
    assert verification.evidence_state == "decision_eligible"
    assert verification.authorized_actions == ("offline_research_reporting",)
    assert "production_rollout" in verification.explicitly_unauthorized


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    target = tmp_path / "evidence-v1"
    _write(target)
    before = (target / "complete.json").read_bytes()

    with pytest.raises(FileExistsError):
        _write(target)
    assert (target / "complete.json").read_bytes() == before


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("output_identity", "wrong-output", "output identity"),
        ("no_matching_execution_verified", False, "execution identity"),
        ("production_source_writes", True, "safety flags"),
        ("serving_changes", True, "safety flags"),
        ("credentials_or_personal_records_emitted", True, "safety flags"),
        ("actual_runtime_seconds", 61.0, "runtime"),
        ("peak_memory_mb", 513, "memory"),
        ("created_at", "2026-10-01T00:00:00+09:00", "earliest"),
        (
            "successful_execution_ids",
            ["duplicate", "duplicate"],
            "unique",
        ),
    ],
)
def test_writer_rejects_contract_and_safety_violations(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    target = tmp_path / field
    with pytest.raises(ValueError, match=message):
        _write(target, **{field: value})
    assert not target.exists()


def test_writer_rejects_digest_mismatches_before_writing(tmp_path: Path) -> None:
    input_manifest, input_verification, contract, readiness, estimate, status = (
        _components()
    )
    invalid_estimate = replace(estimate, evaluation_contract_digest="d" * 64)
    target = tmp_path / "invalid-digest"

    with pytest.raises(ValueError, match="estimate contract digest"):
        write_evidence_bundle(
            target,
            output_identity=contract.output_namespace,
            contract=contract,
            input_manifest=input_manifest,
            input_verification=input_verification,
            readiness=readiness,
            estimate=invalid_estimate,
            evidence_status=status,
            created_at="2026-10-02T00:00:01+09:00",
            successful_execution_ids=["run-1"],
            actual_runtime_seconds=1.0,
            peak_memory_mb=100,
            no_matching_execution_verified=True,
        )
    assert not target.exists()


def test_writer_rejects_input_manifest_not_bound_to_contract(tmp_path: Path) -> None:
    _, input_verification, contract, readiness, estimate, status = _components()
    other_manifest = _input_manifest(manifest_id="different-input-snapshot")
    target = tmp_path / "wrong-input-manifest"

    with pytest.raises(ValueError, match="input manifest digest"):
        write_evidence_bundle(
            target,
            output_identity=contract.output_namespace,
            contract=contract,
            input_manifest=other_manifest,
            input_verification=input_verification,
            readiness=readiness,
            estimate=estimate,
            evidence_status=status,
            created_at="2026-10-02T00:00:01+09:00",
            successful_execution_ids=["run-1"],
            actual_runtime_seconds=1.0,
            peak_memory_mb=100,
            no_matching_execution_verified=True,
        )
    assert not target.exists()


def test_writer_rejects_incomplete_input_observation_evidence(tmp_path: Path) -> None:
    input_manifest, input_verification, _, _, _, _ = _components()
    invalid = replace(
        input_verification,
        valid=False,
        verified_input_ids=input_verification.verified_input_ids[:-1],
        missing_input_ids=(input_manifest.inputs[-1].input_id,),
        reasons=(f"missing_input_observation:{input_manifest.inputs[-1].input_id}",),
    )
    target = tmp_path / "incomplete-input-verification"

    with pytest.raises(ValueError, match="do not fully verify"):
        _write(target, input_verification=invalid)
    assert not target.exists()


def test_output_byte_cap_cleans_partial_temporary_state(tmp_path: Path) -> None:
    (
        input_manifest,
        input_verification,
        contract,
        readiness,
        estimate,
        status,
    ) = _components(maximum_output_bytes=100)
    target = tmp_path / "too-large"

    with pytest.raises(ValueError, match="byte cap"):
        write_evidence_bundle(
            target,
            output_identity=contract.output_namespace,
            contract=contract,
            input_manifest=input_manifest,
            input_verification=input_verification,
            readiness=readiness,
            estimate=estimate,
            evidence_status=status,
            created_at="2026-10-02T00:00:01+09:00",
            successful_execution_ids=["run-1"],
            actual_runtime_seconds=1.0,
            peak_memory_mb=100,
            no_matching_execution_verified=True,
        )
    assert not target.exists()
    assert not list(tmp_path.glob(".too-large.tmp-*"))


def test_verifier_rejects_tampering_missing_markers_and_extra_files(
    tmp_path: Path,
) -> None:
    tampered = tmp_path / "tampered"
    _write(tampered)
    estimate_payload = json.loads((tampered / "estimate.json").read_bytes())
    estimate_payload["doubly_robust_lift"] = 999.0
    (tampered / "estimate.json").write_bytes(_canonical(estimate_payload))
    assert not verify_evidence_bundle(tampered).valid

    missing = tmp_path / "missing-marker"
    _write(missing)
    (missing / "complete.json").unlink()
    assert not verify_evidence_bundle(missing).valid

    extra = tmp_path / "extra-file"
    _write(extra)
    (extra / "unexpected.json").write_text("{}\n", encoding="utf-8")
    assert not verify_evidence_bundle(extra).valid


def test_verifier_requires_digest_coverage_for_every_evidence_file(
    tmp_path: Path,
) -> None:
    target = tmp_path / "missing-artifact-record"
    _write(target)
    manifest_path = target / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["artifacts"] = [
        artifact
        for artifact in manifest["artifacts"]
        if artifact["filename"] != "input_manifest.json"
    ]
    manifest_bytes = _canonical(manifest)
    manifest_path.write_bytes(manifest_bytes)
    completion_path = target / "complete.json"
    completion = json.loads(completion_path.read_bytes())
    completion["artifact_count"] = len(manifest["artifacts"])
    completion["manifest_byte_size"] = len(manifest_bytes)
    completion["manifest_sha256"] = sha256(manifest_bytes).hexdigest()
    completion_path.write_bytes(_canonical(completion))

    verification = verify_evidence_bundle(target)
    assert not verification.valid
    assert any(
        "does not cover every evidence file" in reason
        for reason in verification.reasons
    )


def test_independent_verifier_rejects_rehashed_authorization_expansion(
    tmp_path: Path,
) -> None:
    target = tmp_path / "authorization-tamper"
    _write(target)
    status_path = target / "evidence_status.json"
    status = json.loads(status_path.read_bytes())
    status["authorized_actions"] = ["production_rollout"]
    _rewrite_artifact_and_markers(target, "evidence_status.json", status)

    verification = verify_evidence_bundle(target)
    assert not verification.valid
    assert "authorized action boundary mismatch" in verification.reasons


def test_independent_verifier_rejects_rehashed_inference_change(
    tmp_path: Path,
) -> None:
    target = tmp_path / "inference-tamper"
    _write(target)
    estimate = json.loads((target / "estimate.json").read_bytes())
    estimate["bootstrap_seed"] = 999
    _rewrite_artifact_and_markers(target, "estimate.json", estimate)

    verification = verify_evidence_bundle(target)
    assert not verification.valid
    assert "estimation settings differ from contract" in verification.reasons


def test_independent_verifier_rejects_rehashed_input_manifest_change(
    tmp_path: Path,
) -> None:
    target = tmp_path / "input-manifest-tamper"
    _write(target)
    input_manifest = json.loads((target / "input_manifest.json").read_bytes())
    input_manifest["inputs"][0]["byte_size"] += 1
    _rewrite_artifact_and_markers(target, "input_manifest.json", input_manifest)

    verification = verify_evidence_bundle(target)
    assert not verification.valid
    assert "input manifest digest mismatch" in verification.reasons


def test_independent_verifier_rejects_rehashed_input_verification_claim(
    tmp_path: Path,
) -> None:
    target = tmp_path / "input-verification-tamper"
    _write(target)
    verification_payload = json.loads((target / "input_verification.json").read_bytes())
    verification_payload["valid"] = False
    verification_payload["reasons"] = ["storage_observation_failed"]
    _rewrite_artifact_and_markers(
        target, "input_verification.json", verification_payload
    )

    verification = verify_evidence_bundle(target)
    assert not verification.valid
    assert "persisted input verification is incomplete or invalid" in (
        verification.reasons
    )


def test_noneligible_bundle_preserves_empty_authorized_actions(tmp_path: Path) -> None:
    input_manifest, input_verification, contract, readiness, estimate, _ = _components()
    status = classify_evaluation_evidence(
        contract,
        analysis_timestamp="2026-10-02T00:00:00+09:00",
        collection_complete=True,
        immutable_inputs_verified=True,
        no_clobber_verified=True,
        labels_settled=True,
        label_snapshot_digest="c" * 64,
        readiness_decision=readiness.decision,
        readiness_contract_digest=readiness.evaluation_contract_digest,
        estimation_complete=True,
        final_audit_passed=None,
    )
    target = tmp_path / "audit-pending"
    write_evidence_bundle(
        target,
        output_identity=contract.output_namespace,
        contract=contract,
        input_manifest=input_manifest,
        input_verification=input_verification,
        readiness=readiness,
        estimate=estimate,
        evidence_status=status,
        created_at="2026-10-02T00:00:01+09:00",
        successful_execution_ids=["run-1"],
        actual_runtime_seconds=1.0,
        peak_memory_mb=100,
        no_matching_execution_verified=True,
    )

    verification = verify_evidence_bundle(target)
    assert verification.valid
    assert verification.evidence_state == "artifact_complete"
    assert verification.authorized_actions == ()
