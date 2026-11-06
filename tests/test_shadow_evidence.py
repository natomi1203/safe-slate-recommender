from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

from safe_slate import (
    LINEAR_PARAMETER_COVARIANCE_ROLE,
    REQUEST_CONTRAST_FEATURES_ROLE,
    canonical_matrix_bytes,
    create_immutable_input,
    create_input_manifest,
    create_observed_input_identity,
    guarded_linear_rerank,
    matrix_sha256,
    run_no_clobber_shadow_evaluation,
    verify_shadow_evidence_bundle,
    audit_input_manifest_observations,
)
from safe_slate.shadow_evidence import (
    ShadowEvidenceVerification,
    _summarize,
    _validate_contract_binding,
    _write_bundle,
)
from test_shadow import _shadow_log
from test_shadow_contract import _contract


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


def _manifest():
    shadow_records = create_immutable_input(
        input_id="shadow-records",
        semantic_role="shadow_records",
        canonical_uri="gs://research-inputs/shadow-records.jsonl",
        version="2026-10-shadow-v1",
        generation="1001",
        metageneration="1",
        byte_size=1_000,
        sha256_digest=sha256(b"shadow-records").hexdigest(),
        snapshot_time="2026-10-08T00:00:00+09:00",
        temporal_scope="feature",
        schema_version="proposal-log-1.3",
    )
    return create_input_manifest(
        manifest_id="shadow-inputs-v1",
        population_identity="jp-home-feed-eligible-v3",
        created_at="2026-10-08T00:00:01+09:00",
        feature_cutoff="2026-10-08T00:00:00+09:00",
        label_cutoff="2026-10-08T00:00:00+09:00",
        required_roles=("shadow_records",),
        inputs=(shadow_records,),
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


def _linear_shadow_log(request_id: str):
    return guarded_linear_rerank(
        request_id=request_id,
        event_timestamp="2026-10-01T12:00:00+09:00",
        serving_mode="shadow",
        uncertainty_snapshot_id="covariance-snapshot-v17",
        contrast_input_id=f"contrast-{request_id}",
        model_version="ranker-v17",
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
        support_ok=True,
        uncertainty_valid=True,
        distribution_shift_detected=False,
        proposal_log_complete=True,
    ).proposal_log


def _factorized_manifest():
    base = _manifest()
    covariance = [[0.0001, 0.0], [0.0, 0.0001]]
    encoded = canonical_matrix_bytes(covariance)
    snapshot = create_immutable_input(
        input_id="covariance-snapshot-v17",
        semantic_role=LINEAR_PARAMETER_COVARIANCE_ROLE,
        canonical_uri="gs://research-inputs/covariance-v17.json",
        version="ranker-v17",
        byte_size=len(encoded),
        sha256_digest=matrix_sha256(covariance),
        snapshot_time="2026-10-01T00:00:00+09:00",
        temporal_scope="feature",
        schema_version="safe-slate-linear-covariance-1.0",
    )
    contrasts = [[1.0, 0.0], [0.5, 1.0]]
    contrast_payload = canonical_matrix_bytes(contrasts)
    contrast_entries = tuple(
        create_immutable_input(
            input_id=f"contrast-r{index}",
            semantic_role=REQUEST_CONTRAST_FEATURES_ROLE,
            canonical_uri=f"gs://research-inputs/contrasts/r{index}.json",
            version="ranker-v17",
            byte_size=len(contrast_payload),
            sha256_digest=matrix_sha256(contrasts),
            snapshot_time="2026-10-01T11:59:00+09:00",
            temporal_scope="feature",
            schema_version="safe-slate-request-contrasts-1.0",
        )
        for index in range(2)
    )
    return create_input_manifest(
        manifest_id="shadow-factorized-inputs-v1",
        population_identity=base.population_identity,
        created_at=base.created_at,
        feature_cutoff=base.feature_cutoff,
        label_cutoff=base.label_cutoff,
        required_roles=(
            "shadow_records",
            LINEAR_PARAMETER_COVARIANCE_ROLE,
            REQUEST_CONTRAST_FEATURES_ROLE,
        ),
        inputs=(*base.inputs, snapshot, *contrast_entries),
    )


def _run(
    target: Path,
    *,
    contract=None,
    manifest=None,
    observations=None,
    logs=None,
    **overrides,
):
    manifest = manifest or _manifest()
    contract = contract or _contract(
        input_manifest_digest=manifest.manifest_digest,
        minimum_requests=2,
        maximum_oracle_p95_ms=100.0,
        maximum_fallback_rate=0.0,
    )
    logs = logs or [_shadow_log("r0"), _shadow_log("r1")]
    values = {
        "output_identity": contract.output_namespace,
        "contract": contract,
        "input_manifest": manifest,
        "observed_inputs": observations or _observations(manifest),
        "logs": logs,
        "created_at": "2026-10-08T00:00:02+09:00",
        "successful_execution_ids": ("shadow-job-001",),
        "peak_memory_mb": 128,
        "no_matching_execution_verified": True,
    }
    values.update(overrides)
    return run_no_clobber_shadow_evaluation(target, **values)


def _rewrite_artifact(target: Path, filename: str, payload: dict[str, object]) -> None:
    data = _canonical(payload)
    (target / filename).write_bytes(data)
    manifest_path = target / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    for artifact in manifest["artifacts"]:
        if artifact["filename"] == filename:
            artifact["byte_size"] = len(data)
            artifact["sha256"] = sha256(data).hexdigest()
    manifest_bytes = _canonical(manifest)
    manifest_path.write_bytes(manifest_bytes)
    completion_path = target / "complete.json"
    completion = json.loads(completion_path.read_bytes())
    completion["manifest_byte_size"] = len(manifest_bytes)
    completion["manifest_sha256"] = sha256(manifest_bytes).hexdigest()
    completion_path.write_bytes(_canonical(completion))


def _rewrite_shadow_manifest(target: Path, manifest: dict[str, object]) -> None:
    manifest_bytes = _canonical(manifest)
    (target / "manifest.json").write_bytes(manifest_bytes)
    completion_path = target / "complete.json"
    completion = json.loads(completion_path.read_bytes())
    completion["manifest_byte_size"] = len(manifest_bytes)
    completion["manifest_sha256"] = sha256(manifest_bytes).hexdigest()
    completion_path.write_bytes(_canonical(completion))


def _shadow_components(**contract_overrides):
    manifest = _manifest()
    contract_values = {
        "input_manifest_digest": manifest.manifest_digest,
        "minimum_requests": 2,
        "maximum_oracle_p95_ms": 100.0,
        "maximum_fallback_rate": 0.0,
    }
    contract_values.update(contract_overrides)
    contract = _contract(
        **contract_values,
    )
    verification = audit_input_manifest_observations(manifest, _observations(manifest))
    summary = _summarize(contract, [_shadow_log("r0"), _shadow_log("r1")])
    return manifest, contract, verification, summary


def test_shadow_runner_publishes_and_recomputes_source_summary(tmp_path: Path) -> None:
    target = tmp_path / "shadow-evidence"
    logs = [_shadow_log("r0"), _shadow_log("r1")]

    result = _run(target, logs=logs)

    assert result.state == "evidence_valid"
    assert result.summary is not None
    assert result.summary.operability_pass
    assert result.bundle_manifest is not None
    assert result.bundle_verification is not None
    assert result.bundle_verification.valid
    assert result.bundle_verification.source_recomputed
    assert result.bundle_verification.evidence_state == "evidence_valid"
    assert result.authorized_actions == ("offline_shadow_operability_reporting",)
    assert "production_rollout" in result.explicitly_unauthorized
    assert not result.production_latency_claim_authorized
    assert not result.causal_lift_claim_authorized
    assert (target / "complete.json").is_file()
    assert result.to_dict()["state"] == "evidence_valid"


def test_shadow_runner_binds_factorized_uncertainty_snapshot(tmp_path: Path) -> None:
    manifest = _factorized_manifest()
    result = _run(
        tmp_path / "factorized-shadow-evidence",
        manifest=manifest,
        observations=_observations(manifest),
        logs=[_linear_shadow_log("r0"), _linear_shadow_log("r1")],
    )

    assert result.state == "evidence_valid"
    assert result.bundle_verification is not None
    assert result.bundle_verification.valid


def test_shadow_runner_rejects_unbound_factorized_uncertainty(tmp_path: Path) -> None:
    manifest = _manifest()
    result = _run(
        tmp_path / "unbound-factorized-shadow-evidence",
        manifest=manifest,
        observations=_observations(manifest),
        logs=[_linear_shadow_log("r0"), _linear_shadow_log("r1")],
    )

    assert result.state == "evidence_rejected"
    assert result.output_directory is None
    assert any(
        "uncertainty_snapshot_manifest_missing" in reason
        for reason in result.reasons
    )


def test_verifier_requires_source_logs_and_exact_ordered_content(
    tmp_path: Path,
) -> None:
    target = tmp_path / "shadow-evidence"
    logs = [_shadow_log("r0"), _shadow_log("r1")]
    _run(target, logs=logs)

    without_source = verify_shadow_evidence_bundle(target)
    reversed_source = verify_shadow_evidence_bundle(target, logs=list(reversed(logs)))

    assert not without_source.valid
    assert not without_source.source_recomputed
    assert "source logs are required" in without_source.reasons[0]
    assert not reversed_source.valid
    assert reversed_source.source_recomputed
    assert "differs from source recomputation" in reversed_source.reasons[0]


def test_rehashed_summary_tampering_is_rejected_by_source_recomputation(
    tmp_path: Path,
) -> None:
    target = tmp_path / "tampered-summary"
    logs = [_shadow_log("r0"), _shadow_log("r1")]
    _run(target, logs=logs)
    summary = json.loads((target / "summary.json").read_bytes())
    summary["fallback_rate"] = 0.9
    _rewrite_artifact(target, "summary.json", summary)

    verification = verify_shadow_evidence_bundle(target, logs=logs)

    assert not verification.valid
    assert "differs from source recomputation" in verification.reasons[0]


def test_statistically_small_shadow_run_is_preserved_as_insufficient(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    contract = _contract(
        input_manifest_digest=manifest.manifest_digest,
        minimum_requests=3,
        maximum_oracle_p95_ms=100.0,
        maximum_fallback_rate=0.0,
    )

    result = _run(
        tmp_path / "insufficient",
        manifest=manifest,
        contract=contract,
    )

    assert result.state == "insufficient_evidence"
    assert result.bundle_verification is not None
    assert result.bundle_verification.valid
    assert "request_count_below_gate" in result.reasons


def test_operability_gate_failure_remains_valid_negative_evidence(
    tmp_path: Path,
) -> None:
    logs = [_shadow_log("r0"), _shadow_log("r1", support_ok=False)]

    result = _run(tmp_path / "negative", logs=logs)

    assert result.state == "evidence_valid"
    assert result.summary is not None
    assert not result.summary.operability_pass
    assert "fallback_rate_above_gate" in result.reasons
    assert result.bundle_verification is not None
    assert result.bundle_verification.valid


def test_shadow_evidence_can_require_complete_service_latency_traces(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    contract = _contract(
        input_manifest_digest=manifest.manifest_digest,
        minimum_requests=2,
        maximum_oracle_p95_ms=100.0,
        maximum_fallback_rate=0.0,
        require_service_latency_trace=True,
        maximum_service_response_p95_ms=30.0,
        maximum_log_commit_p95_ms=30.0,
    )
    logs = [
        _shadow_log("r0", with_service_latency=True),
        _shadow_log("r1", with_service_latency=True),
    ]

    result = _run(
        tmp_path / "service-latency",
        manifest=manifest,
        contract=contract,
        logs=logs,
    )

    assert result.state == "evidence_valid"
    assert result.summary is not None
    assert result.summary.shadow_service_latency_evidence_valid
    assert not result.production_latency_claim_authorized


def test_required_service_latency_trace_missing_is_integrity_rejection(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    contract = _contract(
        input_manifest_digest=manifest.manifest_digest,
        minimum_requests=2,
        maximum_oracle_p95_ms=100.0,
        maximum_fallback_rate=0.0,
        require_service_latency_trace=True,
        maximum_service_response_p95_ms=30.0,
        maximum_log_commit_p95_ms=30.0,
    )

    result = _run(
        tmp_path / "missing-service-latency",
        manifest=manifest,
        contract=contract,
    )

    assert result.state == "evidence_rejected"
    assert result.summary is not None
    assert not result.summary.shadow_service_latency_evidence_valid
    assert result.authorized_actions == ()


def test_integrity_failure_is_preserved_but_never_authorized(tmp_path: Path) -> None:
    logs = [
        replace(
            _shadow_log("r0"),
            event_timestamp="2026-09-30T23:59:59+09:00",
        ),
        _shadow_log("r1"),
    ]

    result = _run(tmp_path / "rejected", logs=logs)

    assert result.state == "evidence_rejected"
    assert result.bundle_verification is not None
    assert result.bundle_verification.valid
    assert result.authorized_actions == ()


def test_input_identity_failure_stops_before_publication(tmp_path: Path) -> None:
    manifest = _manifest()
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
    target = tmp_path / "input-rejected"

    result = _run(target, manifest=manifest, observations=observations)

    assert result.state == "evidence_rejected"
    assert result.summary is None
    assert result.bundle_manifest is None
    assert not target.exists()


def test_shadow_runner_refuses_existing_or_unverified_identity(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        _run(existing)

    manifest = _manifest()
    contract = _contract(
        input_manifest_digest=manifest.manifest_digest,
        minimum_requests=2,
    )
    with pytest.raises(ValueError, match="execution identity"):
        run_no_clobber_shadow_evaluation(
            tmp_path / "unchecked",
            output_identity=contract.output_namespace,
            contract=contract,
            input_manifest=manifest,
            observed_inputs=_observations(manifest),
            logs=[_shadow_log("r0"), _shadow_log("r1")],
            created_at="2026-10-08T00:00:02+09:00",
            successful_execution_ids=("shadow-job-001",),
            peak_memory_mb=128,
            no_matching_execution_verified=False,
        )


def test_shadow_output_cap_removes_only_temporary_bundle(tmp_path: Path) -> None:
    manifest = _manifest()
    contract = _contract(
        input_manifest_digest=manifest.manifest_digest,
        minimum_requests=2,
        maximum_oracle_p95_ms=100.0,
        maximum_fallback_rate=0.0,
        maximum_output_bytes=100,
    )
    target = tmp_path / "too-large"

    with pytest.raises(ValueError, match="byte cap"):
        _run(target, manifest=manifest, contract=contract)

    assert not target.exists()
    assert not list(tmp_path.glob(".too-large.tmp-*"))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"output_identity": "wrong"}, "output identity"),
        ({"created_at": "not-a-time"}, "ISO-8601"),
        ({"created_at": "2026-10-08T00:00:02"}, "timezone"),
        ({"created_at": "2026-10-07T23:59:59+09:00"}, "predates"),
        ({"peak_memory_mb": True}, "must be an integer"),
        ({"peak_memory_mb": -1}, "memory exceeds"),
        ({"peak_memory_mb": 1_025}, "memory exceeds"),
        ({"successful_execution_ids": ()}, "execution identities"),
        ({"successful_execution_ids": ("",)}, "non-empty strings"),
        (
            {"successful_execution_ids": ("duplicate", "duplicate")},
            "must be unique",
        ),
        (
            {
                "successful_execution_ids": ("same",),
                "failed_execution_ids": ("same",),
            },
            "execution identities",
        ),
    ],
)
def test_shadow_bundle_rejects_malformed_runtime_contract(
    tmp_path: Path, overrides: dict[str, object], message: str
) -> None:
    target = tmp_path / message.replace(" ", "-")
    with pytest.raises(ValueError, match=message):
        _run(target, **overrides)
    assert not target.exists()


def test_shadow_runner_requires_existing_output_parent(tmp_path: Path) -> None:
    target = tmp_path / "missing" / "bundle"
    with pytest.raises(ValueError, match="parent directory"):
        _run(target)


def test_shadow_contract_binding_rejects_population_cutoff_and_roles(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    population_mismatch = _contract(
        input_manifest_digest=manifest.manifest_digest,
        population_identity="other-population",
        minimum_requests=2,
    )
    with pytest.raises(ValueError, match="population"):
        _run(tmp_path / "population", manifest=manifest, contract=population_mismatch)

    cutoff_mismatch = _contract(
        input_manifest_digest=manifest.manifest_digest,
        window_end="2026-10-07T00:00:00+09:00",
        minimum_requests=2,
    )
    with pytest.raises(ValueError, match="cutoff"):
        _run(tmp_path / "cutoff", manifest=manifest, contract=cutoff_mismatch)

    unrelated = create_immutable_input(
        input_id="other",
        semantic_role="other_role",
        canonical_uri="gs://research-inputs/other.jsonl",
        version="v1",
        byte_size=1,
        sha256_digest="a" * 64,
        snapshot_time="2026-10-08T00:00:00+09:00",
        temporal_scope="feature",
        schema_version="1.0",
    )
    no_shadow = create_input_manifest(
        manifest_id="no-shadow",
        population_identity=manifest.population_identity,
        created_at=manifest.created_at,
        feature_cutoff=manifest.feature_cutoff,
        label_cutoff=manifest.label_cutoff,
        required_roles=("other_role",),
        inputs=(unrelated,),
    )
    no_shadow_contract = _contract(
        input_manifest_digest=no_shadow.manifest_digest,
        minimum_requests=2,
    )
    with pytest.raises(ValueError, match="lacks shadow_records"):
        _run(
            tmp_path / "no-shadow",
            manifest=no_shadow,
            contract=no_shadow_contract,
        )


def test_shadow_writer_and_binding_reject_remaining_invalid_contracts(
    tmp_path: Path,
) -> None:
    manifest, contract, verification, summary = _shadow_components()
    with pytest.raises(ValueError, match="contract is invalid"):
        _validate_contract_binding("bad", manifest)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="manifest is invalid"):
        _validate_contract_binding(contract, "bad")  # type: ignore[arg-type]
    mismatched = _contract(input_manifest_digest="0" * 64)
    with pytest.raises(ValueError, match="digest does not match"):
        _validate_contract_binding(mismatched, manifest)

    values = {
        "output_identity": contract.output_namespace,
        "contract": contract,
        "input_manifest": manifest,
        "input_verification": verification,
        "summary": summary,
        "evidence_state": "artifact_complete",
        "created_at": "2026-10-08T00:00:02+09:00",
        "successful_execution_ids": ("run",),
        "failed_execution_ids": (),
        "actual_runtime_seconds": 1.0,
        "peak_memory_mb": 1,
    }
    cases = [
        ({"created_at": "2026-10-08T00:00:00+09:00"}, "created after"),
        (
            {"input_verification": replace(verification, valid=False)},
            "do not fully verify",
        ),
        ({"actual_runtime_seconds": 301.0}, "runtime exceeds"),
        ({"peak_memory_mb": True}, "must be an integer"),
        ({"peak_memory_mb": 2_000}, "memory exceeds"),
        ({"summary": replace(summary, fallback_rate=float("nan"))}, "non-finite"),
    ]
    for index, (changes, message) in enumerate(cases):
        with pytest.raises(ValueError, match=message):
            _write_bundle(tmp_path / f"invalid-writer-{index}", **{**values, **changes})

    existing = tmp_path / "existing-writer"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        _write_bundle(existing, **values)
    with pytest.raises(ValueError, match="parent directory"):
        _write_bundle(tmp_path / "missing" / "bundle", **values)

    capped_manifest, capped_contract, capped_verification, capped_summary = (
        _shadow_components(maximum_output_bytes=4_500)
    )
    with pytest.raises(ValueError, match="byte cap"):
        _write_bundle(
            tmp_path / "manifest-overhead-cap",
            **{
                **values,
                "contract": capped_contract,
                "input_manifest": capped_manifest,
                "input_verification": capped_verification,
                "summary": capped_summary,
            },
        )


def test_shadow_verifier_rejects_marker_manifest_and_artifact_variants(
    tmp_path: Path,
) -> None:
    assert not verify_shadow_evidence_bundle(tmp_path / "missing", logs=[]).valid

    target = tmp_path / "completion-fields"
    _run(target)
    completion = json.loads((target / "complete.json").read_bytes())
    completion["extra"] = True
    (target / "complete.json").write_bytes(_canonical(completion))
    assert "completion fields" in verify_shadow_evidence_bundle(target, logs=[]).reasons[0]

    target = tmp_path / "completion-state"
    _run(target)
    completion = json.loads((target / "complete.json").read_bytes())
    completion["state"] = "partial"
    (target / "complete.json").write_bytes(_canonical(completion))
    assert "marker is not complete" in verify_shadow_evidence_bundle(target, logs=[]).reasons[0]

    target = tmp_path / "marker-mismatch"
    _run(target)
    completion = json.loads((target / "complete.json").read_bytes())
    completion["manifest_byte_size"] += 1
    (target / "complete.json").write_bytes(_canonical(completion))
    assert "manifest marker mismatch" in verify_shadow_evidence_bundle(target, logs=[]).reasons[0]

    target = tmp_path / "manifest-fields"
    _run(target)
    manifest = json.loads((target / "manifest.json").read_bytes())
    manifest["extra"] = True
    _rewrite_shadow_manifest(target, manifest)
    assert "manifest fields" in verify_shadow_evidence_bundle(target, logs=[]).reasons[0]

    target = tmp_path / "artifact-count"
    _run(target)
    completion = json.loads((target / "complete.json").read_bytes())
    completion["artifact_count"] += 1
    (target / "complete.json").write_bytes(_canonical(completion))
    assert "artifact count" in verify_shadow_evidence_bundle(target, logs=[]).reasons[0]

    for field, value, message in [
        ("filename", "manifest.json", "artifact name"),
        ("role", "unknown", "artifact role"),
        ("sha256", "0" * 64, "artifact integrity"),
    ]:
        target = tmp_path / f"artifact-{field}"
        _run(target)
        manifest = json.loads((target / "manifest.json").read_bytes())
        manifest["artifacts"][0][field] = value
        _rewrite_shadow_manifest(target, manifest)
        assert message in verify_shadow_evidence_bundle(target, logs=[]).reasons[0]

    target = tmp_path / "artifact-incomplete"
    _run(target)
    manifest = json.loads((target / "manifest.json").read_bytes())
    manifest["artifacts"] = manifest["artifacts"][:-1]
    completion = json.loads((target / "complete.json").read_bytes())
    completion["artifact_count"] = len(manifest["artifacts"])
    _rewrite_shadow_manifest(target, manifest)
    (target / "complete.json").write_bytes(_canonical({
        **json.loads((target / "complete.json").read_bytes()),
        "artifact_count": len(manifest["artifacts"]),
    }))
    assert "artifact manifest is incomplete" in verify_shadow_evidence_bundle(target, logs=[]).reasons[0]

    target = tmp_path / "extra-file"
    _run(target)
    (target / "extra.json").write_text("{}")
    assert "file set" in verify_shadow_evidence_bundle(target, logs=[]).reasons[0]


def test_shadow_verifier_rejects_input_and_summary_payload_drift(tmp_path: Path) -> None:
    target = tmp_path / "input-verification"
    _run(target)
    verification = json.loads((target / "input_verification.json").read_bytes())
    verification["valid"] = False
    _rewrite_artifact(target, "input_verification.json", verification)
    assert "input verification is incomplete" in verify_shadow_evidence_bundle(target, logs=[]).reasons[0]

    target = tmp_path / "summary-fields"
    _run(target)
    summary = json.loads((target / "summary.json").read_bytes())
    summary.pop("request_count")
    _rewrite_artifact(target, "summary.json", summary)
    assert "summary fields" in verify_shadow_evidence_bundle(target, logs=[]).reasons[0]

    target = tmp_path / "summary-nonfinite"
    _run(target)
    summary = json.loads((target / "summary.json").read_bytes())
    summary["fallback_rate"] = float("nan")
    data = (json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n").encode()
    (target / "summary.json").write_bytes(data)
    manifest = json.loads((target / "manifest.json").read_bytes())
    for artifact in manifest["artifacts"]:
        if artifact["filename"] == "summary.json":
            artifact["byte_size"] = len(data)
            artifact["sha256"] = sha256(data).hexdigest()
    _rewrite_shadow_manifest(target, manifest)
    assert "invalid numeric" in verify_shadow_evidence_bundle(target, logs=[]).reasons[0]

    target = tmp_path / "factorized-binding"
    _run(target)
    assert "factorized uncertainty binding" in verify_shadow_evidence_bundle(
        target, logs=[_linear_shadow_log("r0"), _linear_shadow_log("r1")]
    ).reasons[0]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", "2.0", "unsupported shadow evidence schema"),
        ("evaluation_id", "wrong", "evaluation identity mismatch"),
        ("contract_digest", "0" * 64, "manifest contract digest mismatch"),
        ("input_manifest_digest", "0" * 64, "manifest input digest mismatch"),
        ("log_data_digest", "0" * 64, "manifest log digest mismatch"),
        ("output_identity", "wrong", "output identity mismatch"),
        ("evidence_state", "wrong", "evidence state mismatch"),
        ("operability_pass", False, "operability decision mismatch"),
        ("created_at", "2026-10-07T00:00:00+09:00", "predates"),
        ("actual_runtime_seconds", True, "runtime cap exceeded"),
        ("peak_memory_mb", -1, "memory cap exceeded"),
        ("safety_flags", [], "safety flags are invalid"),
        ("successful_execution_ids", [], "execution identities are invalid"),
    ],
)
def test_shadow_verifier_rejects_manifest_semantic_drift(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    target = tmp_path / field
    logs = [_shadow_log("r0"), _shadow_log("r1")]
    _run(target, logs=logs)
    manifest = json.loads((target / "manifest.json").read_bytes())
    manifest[field] = value
    _rewrite_shadow_manifest(target, manifest)
    assert message in verify_shadow_evidence_bundle(target, logs=logs).reasons[0]


def test_shadow_verifier_rejects_oversized_and_postdated_coherent_bundle(
    tmp_path: Path,
) -> None:
    logs = [_shadow_log("r0"), _shadow_log("r1")]
    target = tmp_path / "oversized"
    _run(target, logs=logs)
    summary_data = (target / "summary.json").read_bytes() + b" " * 1_000_000
    (target / "summary.json").write_bytes(summary_data)
    manifest_payload = json.loads((target / "manifest.json").read_bytes())
    for artifact in manifest_payload["artifacts"]:
        if artifact["filename"] == "summary.json":
            artifact["byte_size"] = len(summary_data)
            artifact["sha256"] = sha256(summary_data).hexdigest()
    _rewrite_shadow_manifest(target, manifest_payload)
    assert "exceeds output byte cap" in verify_shadow_evidence_bundle(target, logs=logs).reasons[0]

    base = _manifest()
    late_manifest = create_input_manifest(
        manifest_id="late-shadow-inputs",
        population_identity=base.population_identity,
        created_at="2026-10-09T00:00:00+09:00",
        feature_cutoff=base.feature_cutoff,
        label_cutoff=base.label_cutoff,
        required_roles=base.required_roles,
        inputs=base.inputs,
    )
    contract = _contract(
        input_manifest_digest=late_manifest.manifest_digest,
        minimum_requests=2,
        maximum_oracle_p95_ms=100.0,
        maximum_fallback_rate=0.0,
    )
    verification = audit_input_manifest_observations(
        late_manifest, _observations(late_manifest)
    )
    summary = _summarize(contract, logs)
    target = tmp_path / "postdated"
    _run(target, logs=logs)
    for filename, payload in (
        ("contract.json", contract.to_dict()),
        ("input_manifest.json", late_manifest.to_dict()),
        ("input_verification.json", verification.to_dict()),
        ("summary.json", summary.to_dict()),
    ):
        _rewrite_artifact(target, filename, payload)
    manifest_payload = json.loads((target / "manifest.json").read_bytes())
    manifest_payload.update(
        contract_digest=contract.contract_digest,
        input_manifest_digest=late_manifest.manifest_digest,
        log_data_digest=summary.log_data_digest,
    )
    _rewrite_shadow_manifest(target, manifest_payload)
    assert "postdates the bundle" in verify_shadow_evidence_bundle(target, logs=logs).reasons[0]


def test_shadow_runner_preserves_verifier_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rejected = ShadowEvidenceVerification(
        valid=False,
        source_recomputed=False,
        evidence_state="evidence_rejected",
        contract_digest=None,
        manifest_digest=None,
        authorized_actions=(),
        explicitly_unauthorized=(),
        reasons=("forced verifier failure",),
    )
    monkeypatch.setattr(
        "safe_slate.shadow_evidence.verify_shadow_evidence_bundle",
        lambda *args, **kwargs: rejected,
    )
    result = _run(tmp_path / "forced-rejection")
    assert result.state == "evidence_rejected"
    assert result.reasons == ("forced verifier failure",)
