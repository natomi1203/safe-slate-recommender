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


def _rewrite_manifest_and_marker(target: Path, manifest: dict[str, object]) -> None:
    manifest_bytes = _canonical(manifest)
    (target / "manifest.json").write_bytes(manifest_bytes)
    completion_path = target / "complete.json"
    completion = json.loads(completion_path.read_bytes())
    completion["manifest_byte_size"] = len(manifest_bytes)
    completion["manifest_sha256"] = sha256(manifest_bytes).hexdigest()
    completion_path.write_bytes(_canonical(completion))


def _rewrite_raw_artifact_and_markers(
    target: Path, filename: str, data: bytes
) -> None:
    (target / filename).write_bytes(data)
    manifest = json.loads((target / "manifest.json").read_bytes())
    for artifact in manifest["artifacts"]:
        if artifact["filename"] == filename:
            artifact["byte_size"] = len(data)
            artifact["sha256"] = sha256(data).hexdigest()
    _rewrite_manifest_and_marker(target, manifest)


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
        ("actual_runtime_seconds", -0.1, "runtime"),
        ("actual_runtime_seconds", True, "runtime"),
        ("peak_memory_mb", 513, "memory"),
        ("peak_memory_mb", -1, "memory"),
        ("peak_memory_mb", True, "memory"),
        ("created_at", "2026-10-01T00:00:00+09:00", "earliest"),
        ("created_at", "not-a-time", "ISO-8601"),
        ("created_at", "2026-10-02T00:00:00", "timezone"),
        ("successful_execution_ids", [], "at least one"),
        ("successful_execution_ids", [""], "non-empty strings"),
        (
            "successful_execution_ids",
            ["duplicate", "duplicate"],
            "unique",
        ),
        (
            "failed_execution_ids",
            ["local-evaluation-001"],
            "may not overlap",
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


def test_writer_rejects_empty_timestamp_and_manifest_overhead_cap(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-empty ISO-8601"):
        _write(tmp_path / "empty-time", created_at="")

    components = _components(maximum_output_bytes=9_500)
    with pytest.raises(ValueError, match="byte cap"):
        _write(
            tmp_path / "manifest-overhead",
            input_manifest=components[0],
            input_verification=components[1],
            contract=components[2],
            readiness=components[3],
            estimate=components[4],
            evidence_status=components[5],
        )


def test_writer_rejects_invalid_component_bindings(tmp_path: Path) -> None:
    input_manifest, input_verification, contract, readiness, estimate, status = (
        _components()
    )

    class EstimateWithoutSerialization:
        evaluation_contract_digest = contract.contract_digest
        row_data_digest = readiness.row_data_digest

    cases = [
        ({"contract": "bad"}, "contract is invalid"),
        ({"input_manifest": "bad"}, "input manifest is invalid"),
        ({"input_verification": "bad"}, "input_verification"),
        ({"readiness": "bad"}, "readiness"),
        ({"evidence_status": "bad"}, "evidence_status"),
        (
            {
                "readiness": replace(
                    readiness, evaluation_contract_digest="0" * 64
                )
            },
            "readiness contract digest",
        ),
        (
            {"estimate": replace(estimate, row_data_digest="0" * 64)},
            "estimate row digest",
        ),
        (
            {"evidence_status": replace(status, contract_digest="0" * 64)},
            "evidence status contract digest",
        ),
        ({"estimate": EstimateWithoutSerialization()}, "expose to_dict"),
        (
            {"estimate": None},
            "requires a completed estimate",
        ),
        (
            {
                "evidence_status": replace(
                    status,
                    overall_state="insufficient_evidence",
                    decision_eligible=False,
                    authorized_actions=(),
                )
            },
            "may not contain an estimate",
        ),
    ]
    for index, (overrides, message) in enumerate(cases):
        target = tmp_path / f"invalid-component-{index}"
        with pytest.raises(ValueError, match=message):
            _write(target, **overrides)
        assert not target.exists()

    missing_parent = tmp_path / "missing" / "bundle"
    with pytest.raises(ValueError, match="parent directory"):
        _write(missing_parent)


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
    assert verification.to_dict()["valid"] is True


def test_writer_rejects_population_and_temporal_binding_mismatches(
    tmp_path: Path,
) -> None:
    manifest = _input_manifest()
    verification = _input_verification(manifest)
    _, _, population_contract, readiness, estimate, status = _components(
        population_identity="different-population",
        input_manifest_digest=manifest.manifest_digest,
    )
    with pytest.raises(ValueError, match="population"):
        write_evidence_bundle(
            tmp_path / "population",
            output_identity=population_contract.output_namespace,
            contract=population_contract,
            input_manifest=manifest,
            input_verification=verification,
            readiness=readiness,
            estimate=estimate,
            evidence_status=status,
            created_at="2026-10-02T00:00:01+09:00",
            successful_execution_ids=["run"],
            actual_runtime_seconds=1.0,
            peak_memory_mb=1,
            no_matching_execution_verified=True,
        )

    components = _components(final_window_end="2026-09-03T00:00:00+09:00")
    with pytest.raises(ValueError, match="feature cutoff"):
        _write(
            tmp_path / "feature-cutoff",
            input_manifest=components[0],
            input_verification=components[1],
            contract=components[2],
            readiness=components[3],
            estimate=components[4],
            evidence_status=components[5],
        )

    components = _components(earliest_analysis_time="2026-10-01T00:00:00+09:00")
    with pytest.raises(ValueError, match="label cutoff"):
        _write(
            tmp_path / "label-cutoff",
            input_manifest=components[0],
            input_verification=components[1],
            contract=components[2],
            readiness=components[3],
            estimate=components[4],
            evidence_status=components[5],
        )


def test_writer_rejects_late_manifest_specific_path_and_nonfinite_payload(
    tmp_path: Path,
) -> None:
    late_manifest = create_input_manifest(
        manifest_id="late-manifest",
        population_identity="eligible-population-v1",
        created_at="2026-10-03T00:00:00+09:00",
        feature_cutoff="2026-09-02T00:00:00+09:00",
        label_cutoff="2026-10-02T00:00:00+09:00",
        required_roles=REQUIRED_INPUT_ROLES,
        inputs=_input_manifest().inputs,
    )
    late_verification = _input_verification(late_manifest)
    components = _components(input_manifest_digest=late_manifest.manifest_digest)
    with pytest.raises(ValueError, match="created after"):
        _write(
            tmp_path / "late-manifest",
            input_manifest=late_manifest,
            input_verification=late_verification,
            contract=components[2],
            readiness=components[3],
            estimate=components[4],
            evidence_status=components[5],
        )

    with pytest.raises(ValueError, match="specific directory"):
        _write(Path("."))

    _, _, _, readiness, _, _ = _components()
    with pytest.raises(ValueError, match="non-finite"):
        _write(tmp_path / "nonfinite", readiness=replace(readiness, support_rate=float("nan")))


def test_verifier_rejects_marker_manifest_and_artifact_integrity_variants(
    tmp_path: Path,
) -> None:
    assert not verify_evidence_bundle(tmp_path / "missing").valid

    target = tmp_path / "marker-state"
    _write(target)
    completion = json.loads((target / "complete.json").read_bytes())
    completion["state"] = "partial"
    (target / "complete.json").write_bytes(_canonical(completion))
    assert "completion marker is not complete" in verify_evidence_bundle(target).reasons

    target = tmp_path / "manifest-digest"
    _write(target)
    (target / "manifest.json").write_bytes((target / "manifest.json").read_bytes() + b" ")
    assert "manifest digest mismatch" in verify_evidence_bundle(target).reasons

    target = tmp_path / "manifest-size"
    _write(target)
    completion = json.loads((target / "complete.json").read_bytes())
    completion["manifest_byte_size"] += 1
    (target / "complete.json").write_bytes(_canonical(completion))
    assert "manifest size mismatch" in verify_evidence_bundle(target).reasons

    target = tmp_path / "manifest-schema"
    _write(target)
    manifest = json.loads((target / "manifest.json").read_bytes())
    manifest["schema_version"] = "2.0"
    _rewrite_manifest_and_marker(target, manifest)
    assert "unsupported manifest schema" in verify_evidence_bundle(target).reasons

    target = tmp_path / "artifact-count"
    _write(target)
    completion = json.loads((target / "complete.json").read_bytes())
    completion["artifact_count"] += 1
    (target / "complete.json").write_bytes(_canonical(completion))
    assert "artifact count mismatch" in verify_evidence_bundle(target).reasons

    for field, value, message in [
        ("filename", "manifest.json", "artifact filename"),
        ("role", "unknown", "artifact role"),
        ("sha256", "0" * 64, "artifact digest mismatch"),
    ]:
        target = tmp_path / f"artifact-{field}"
        _write(target)
        manifest = json.loads((target / "manifest.json").read_bytes())
        manifest["artifacts"][0][field] = value
        _rewrite_manifest_and_marker(target, manifest)
        assert any(message in reason for reason in verify_evidence_bundle(target).reasons)


def test_verifier_rejects_nonfinite_and_oversized_rehashed_payloads(
    tmp_path: Path,
) -> None:
    target = tmp_path / "nonfinite"
    _write(target)
    readiness = json.loads((target / "readiness.json").read_bytes())
    readiness["support_rate"] = float("nan")
    data = (json.dumps(readiness, sort_keys=True, separators=(",", ":")) + "\n").encode()
    _rewrite_raw_artifact_and_markers(target, "readiness.json", data)
    assert any(
        "non-finite numeric evidence" in reason
        for reason in verify_evidence_bundle(target).reasons
    )

    target = tmp_path / "oversized"
    _write(target)
    readiness_data = (target / "readiness.json").read_bytes() + b" " * 1_000_000
    _rewrite_raw_artifact_and_markers(target, "readiness.json", readiness_data)
    assert "bundle exceeds output byte cap" in verify_evidence_bundle(target).reasons


def test_bundle_without_estimate_round_trips_analysis_ready_state(tmp_path: Path) -> None:
    input_manifest, input_verification, contract, readiness, _, _ = _components()
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
        estimation_complete=False,
        final_audit_passed=None,
    )
    target = tmp_path / "analysis-ready"
    write_evidence_bundle(
        target,
        output_identity=contract.output_namespace,
        contract=contract,
        input_manifest=input_manifest,
        input_verification=input_verification,
        readiness=readiness,
        estimate=None,
        evidence_status=status,
        created_at="2026-10-02T00:00:01+09:00",
        successful_execution_ids=["run"],
        actual_runtime_seconds=1.0,
        peak_memory_mb=1,
        no_matching_execution_verified=True,
    )
    verification = verify_evidence_bundle(target)
    assert verification.valid
    assert verification.evidence_state == "analysis_ready"

    status_payload = json.loads((target / "evidence_status.json").read_bytes())
    status_payload["overall_state"] = "decision_eligible"
    _rewrite_artifact_and_markers(target, "evidence_status.json", status_payload)
    manifest_payload = json.loads((target / "manifest.json").read_bytes())
    manifest_payload["evidence_state"] = "decision_eligible"
    _rewrite_manifest_and_marker(target, manifest_payload)
    assert any(
        "requires a persisted estimate" in reason
        for reason in verify_evidence_bundle(target).reasons
    )


def test_verifier_rejects_rehashed_cross_artifact_binding_changes(
    tmp_path: Path,
) -> None:
    manifest_cases = [
        ("contract_digest", "0" * 64, "manifest contract digest mismatch"),
        ("input_manifest_digest", "0" * 64, "bundle input manifest digest mismatch"),
        ("output_identity", "wrong", "output identity mismatch"),
        ("row_data_digest", "0" * 64, "manifest row digest mismatch"),
        ("evidence_state", "wrong", "evidence state mismatch"),
        ("estimate_type", None, "estimate presence disagrees"),
        ("estimate_type", "", "estimate type is empty"),
        ("created_at", "2026-10-01T00:00:00+09:00", "predates"),
        ("actual_runtime_seconds", True, "runtime cap exceeded"),
        ("peak_memory_mb", -1, "memory cap exceeded"),
        ("safety_flags", [], "offline safety flags"),
        ("successful_execution_ids", [], "execution identities"),
    ]
    for index, (field, value, message) in enumerate(manifest_cases):
        target = tmp_path / f"manifest-binding-{index}"
        _write(target)
        manifest = json.loads((target / "manifest.json").read_bytes())
        manifest[field] = value
        _rewrite_manifest_and_marker(target, manifest)
        assert any(message in reason for reason in verify_evidence_bundle(target).reasons)

    artifact_cases = [
        ("readiness.json", "evaluation_contract_digest", "0" * 64, "readiness contract digest"),
        ("estimate.json", "evaluation_contract_digest", "0" * 64, "estimate contract digest"),
        ("evidence_status.json", "contract_digest", "0" * 64, "status contract digest"),
        ("estimate.json", "row_data_digest", "0" * 64, "estimate row digest"),
        ("readiness.json", "minimum_raw_requests", 999, "readiness settings differ"),
        ("estimate.json", "outcome_lower_bound", -1.0, "outcome bounds differ"),
        ("evidence_status.json", "explicitly_unauthorized", [], "unauthorized action boundary"),
        ("evidence_status.json", "authorized_actions", [], "authorized action boundary"),
    ]
    for index, (filename, field, value, message) in enumerate(artifact_cases):
        target = tmp_path / f"artifact-binding-{index}"
        _write(target)
        payload = json.loads((target / filename).read_bytes())
        payload[field] = value
        _rewrite_artifact_and_markers(target, filename, payload)
        assert any(message in reason for reason in verify_evidence_bundle(target).reasons)

    target = tmp_path / "different-valid-input-manifest"
    _write(target)
    other = _input_manifest(manifest_id="other-manifest")
    _rewrite_artifact_and_markers(target, "input_manifest.json", other.to_dict())
    assert "input manifest digest does not match the contract" in verify_evidence_bundle(target).reasons


def test_verifier_rejects_rehashed_state_machine_contradictions(tmp_path: Path) -> None:
    def rewrite_status_and_state(
        target: Path, changes: dict[str, object], state: str | None = None
    ) -> None:
        status = json.loads((target / "evidence_status.json").read_bytes())
        status.update(changes)
        _rewrite_artifact_and_markers(target, "evidence_status.json", status)
        if state is not None:
            manifest = json.loads((target / "manifest.json").read_bytes())
            manifest["evidence_state"] = state
            _rewrite_manifest_and_marker(target, manifest)

    cases = [
        (
            {"overall_state": "unknown"},
            "unknown",
            "outside contract vocabulary",
        ),
        (
            {
                "overall_state": "decision_eligible",
                "collection_complete": False,
            },
            "decision_eligible",
            "internally inconsistent",
        ),
        (
            {
                "overall_state": "artifact_complete",
                "decision_eligible": True,
                "authorized_actions": ["offline_research_reporting"],
            },
            "artifact_complete",
            "boolean contradicts state",
        ),
        (
            {
                "overall_state": "analysis_ready",
                "decision_eligible": False,
                "authorized_actions": [],
            },
            "analysis_ready",
            "failed readiness was promoted",
        ),
        (
            {
                "overall_state": "artifact_complete",
                "decision_eligible": False,
                "authorized_actions": [],
                "statistically_sufficient": False,
            },
            "artifact_complete",
            "readiness sufficiency contradicts",
        ),
    ]
    for index, (changes, state, message) in enumerate(cases):
        target = tmp_path / f"state-{index}"
        _write(target)
        if message == "failed readiness was promoted":
            readiness = json.loads((target / "readiness.json").read_bytes())
            readiness["decision"] = "fail"
            _rewrite_artifact_and_markers(target, "readiness.json", readiness)
        rewrite_status_and_state(target, changes, state)
        assert any(message in reason for reason in verify_evidence_bundle(target).reasons)

    target = tmp_path / "insufficient-with-estimate"
    _write(target)
    rewrite_status_and_state(
        target,
        {"overall_state": "insufficient_evidence", "decision_eligible": False, "authorized_actions": []},
        "insufficient_evidence",
    )
    assert "insufficient evidence may not contain an estimate" in verify_evidence_bundle(target).reasons


def _replace_bundle_components(
    target: Path,
    input_manifest,
    input_verification,
    contract,
    readiness,
    estimate,
    status,
) -> None:
    for filename, value in (
        ("contract.json", contract),
        ("input_manifest.json", input_manifest),
        ("input_verification.json", input_verification),
        ("readiness.json", readiness),
        ("estimate.json", estimate),
        ("evidence_status.json", status),
    ):
        _rewrite_artifact_and_markers(target, filename, value.to_dict())
    manifest = json.loads((target / "manifest.json").read_bytes())
    manifest.update(
        contract_digest=contract.contract_digest,
        input_manifest_digest=input_manifest.manifest_digest,
        row_data_digest=readiness.row_data_digest,
        evidence_state=status.overall_state,
        estimate_type=type(estimate).__name__,
        output_identity=contract.output_namespace,
    )
    _rewrite_manifest_and_marker(target, manifest)


def test_verifier_rejects_coherently_rehashed_population_and_time_changes(
    tmp_path: Path,
) -> None:
    population_manifest = _input_manifest()
    population_verification = _input_verification(population_manifest)
    population_components = _components(
        population_identity="different-population",
        input_manifest_digest=population_manifest.manifest_digest,
    )
    target = tmp_path / "population-mismatch"
    _write(target)
    _replace_bundle_components(
        target,
        population_manifest,
        population_verification,
        *population_components[2:],
    )
    assert any(
        "input manifest population does not match" in reason
        for reason in verify_evidence_bundle(target).reasons
    )

    feature_components = _components(final_window_end="2026-09-03T00:00:00+09:00")
    target = tmp_path / "feature-mismatch"
    _write(target)
    _replace_bundle_components(target, *feature_components)
    assert any(
        "feature cutoff must equal" in reason
        for reason in verify_evidence_bundle(target).reasons
    )

    label_components = _components(earliest_analysis_time="2026-10-01T00:00:00+09:00")
    target = tmp_path / "label-mismatch"
    _write(target)
    _replace_bundle_components(target, *label_components)
    assert any(
        "label cutoff is outside" in reason
        for reason in verify_evidence_bundle(target).reasons
    )

    late_manifest = create_input_manifest(
        manifest_id="late-verifier-manifest",
        population_identity="eligible-population-v1",
        created_at="2026-10-03T00:00:00+09:00",
        feature_cutoff="2026-09-02T00:00:00+09:00",
        label_cutoff="2026-10-02T00:00:00+09:00",
        required_roles=REQUIRED_INPUT_ROLES,
        inputs=_input_manifest().inputs,
    )
    late_components = _components(input_manifest_digest=late_manifest.manifest_digest)
    target = tmp_path / "late-manifest-verifier"
    _write(target)
    _replace_bundle_components(
        target,
        late_manifest,
        _input_verification(late_manifest),
        *late_components[2:],
    )
    assert any(
        "postdates the evidence bundle" in reason
        for reason in verify_evidence_bundle(target).reasons
    )
