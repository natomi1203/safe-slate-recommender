from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
import runpy
import struct
import sys

import pytest

from safe_slate import (
    EXTERNAL_INPUT_ROLES,
    OPE_REQUIRED_INPUT_ROLES,
    POSITIVITY_EVIDENCE_SCHEMA_VERSION,
    canonical_population_positivity_evidence_bytes,
    canonical_matrix_bytes,
    canonical_matrix_f64le_bytes,
    create_population_positivity_evidence,
    create_statistical_population_positivity_evidence,
    create_service_latency_trace,
    create_service_trace_completion,
    freeze_external_input_manifest,
    guarded_rerank,
    load_input_manifest,
    write_external_input_manifest,
)
from safe_slate.external_intake import (
    F64LE_MAGIC,
    _validate_matrix_file,
    load_intake_spec,
    main as external_intake_main,
)


TEMPLATE = Path(__file__).resolve().parents[1] / "external-inputs.template.json"


def _spec(tmp_path: Path) -> dict[str, object]:
    proposal = guarded_rerank(
        request_id="production-r0",
        event_timestamp="2026-10-01T00:00:00+09:00",
        baseline_item_ids=[10, 11, 12],
        protected_count=2,
        remove_positions=[2],
        add_item_ids=[100],
        reward_deltas=[0.1],
        safety_mean_deltas=[0.01],
        safety_variances=[0.0],
        baseline_metric=0.5,
        degradation_tolerance=0.02,
        max_replacements=1,
        serving_mode="serve",
        support_ok=True,
        uncertainty_valid=True,
        distribution_shift_detected=False,
        proposal_log_complete=True,
    ).proposal_log
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
    completion = create_service_trace_completion(
        proposal,
        trace_id="1" * 32,
        completed_at="2026-10-01T00:00:01+09:00",
        trace_source="opentelemetry",
        service_latency_trace=trace,
    )
    inputs: list[dict[str, object]] = []
    for role in EXTERNAL_INPUT_ROLES:
        path = tmp_path / f"{role}.bin"
        if role == "candidate_action_log":
            path.write_text(json.dumps(proposal.to_dict()) + "\n", encoding="utf-8")
            schema_version = "safe-slate-proposal-log-jsonl-1.3"
        elif role == "service_trace_completion":
            path.write_text(json.dumps(completion.to_dict()) + "\n", encoding="utf-8")
            schema_version = "safe-slate-service-trace-completion-jsonl-1.0"
        elif role in {"linear_parameter_covariance", "request_contrast_features"}:
            path.write_bytes(canonical_matrix_f64le_bytes([[0.1, 0.0], [0.0, 0.2]]))
            schema_version = "safe-slate-matrix-f64le-v1"
        elif role == "population_positivity_evidence":
            evidence = create_population_positivity_evidence(
                evidence_id="positivity-audit-v1",
                decision="established",
                population_identity="jp-home-feed-eligible-v3",
                logging_policy_identity="baseline-v1",
                target_policy_identity="target-v2",
                audited_at="2026-10-08T00:00:00+09:00",
                auditor_identity="independent-design-auditor",
                method="registered logger lower-envelope audit",
                minimum_logging_probability_lower_bound=0.01,
            )
            path.write_bytes(canonical_population_positivity_evidence_bytes(evidence))
            schema_version = POSITIVITY_EVIDENCE_SCHEMA_VERSION
        else:
            path.write_bytes(f"privacy-reviewed:{role}\n".encode())
            schema_version = f"{role}-1.0"
        is_label = role == "settled_purchase_labels"
        is_positivity = role == "population_positivity_evidence"
        inputs.append(
            {
                "input_id": f"external-{role}",
                "semantic_role": role,
                "path": str(path),
                "version": (
                    evidence.evidence_id if is_positivity else "frozen-v1"
                ),
                "snapshot_time": (
                    "2026-10-08T00:00:00+09:00"
                    if is_label or is_positivity
                    else "2026-10-01T00:00:00+09:00"
                ),
                "temporal_scope": (
                    "label" if is_label else "atemporal" if is_positivity else "feature"
                ),
                "schema_version": schema_version,
            }
        )
    return {
        "schema_version": "safe-slate-external-intake-1.0",
        "manifest_id": "private-production-inputs-v1",
        "population_identity": "jp-home-feed-eligible-v3",
        "created_at": "2026-10-09T00:00:00+09:00",
        "feature_cutoff": "2026-10-01T00:00:00+09:00",
        "label_cutoff": "2026-10-08T00:00:00+09:00",
        "privacy_reviewed": True,
        "synthetic_inputs": False,
        "maximum_input_bytes": 1_000_000,
        "maximum_log_records": 100,
        "inputs": inputs,
    }


def test_external_intake_freezes_exactly_ten_private_input_identities(
    tmp_path: Path,
) -> None:
    manifest = freeze_external_input_manifest(_spec(tmp_path))

    assert manifest.verify_digest()
    assert manifest.schema_version == "safe-slate-external-input-manifest-1.0"
    assert set(manifest.required_roles) == set(EXTERNAL_INPUT_ROLES)
    assert set(OPE_REQUIRED_INPUT_ROLES) <= set(manifest.required_roles)
    assert len(manifest.inputs) == 10
    assert all(entry.canonical_uri.startswith("file://") for entry in manifest.inputs)
    assert all(not entry.contains_credentials for entry in manifest.inputs)
    assert all(not entry.contains_direct_identifiers for entry in manifest.inputs)
    covariance = next(
        entry
        for entry in manifest.inputs
        if entry.semantic_role == "linear_parameter_covariance"
    )
    assert covariance.schema_version == "safe-slate-matrix-f64le-v1"


def test_external_intake_publishes_no_clobber_marker_last(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    spec_path = tmp_path / "intake.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    output_dir = tmp_path / "frozen"
    output_dir.mkdir()
    output = output_dir / "manifest.json"
    marker = output_dir / "manifest.complete.json"

    manifest = write_external_input_manifest(
        spec_path,
        output=output,
        completion_marker=marker,
    )
    output_bytes = output.read_bytes()
    completion = json.loads(marker.read_text(encoding="utf-8"))
    assert load_input_manifest(json.loads(output_bytes)) == manifest
    assert completion["manifest_digest"] == manifest.manifest_digest
    assert completion["sha256"] == sha256(output_bytes).hexdigest()
    assert completion["input_count"] == 10
    assert completion["production_rollout_authorized"] is False

    before = output.read_bytes(), marker.read_bytes()
    with pytest.raises(FileExistsError):
        write_external_input_manifest(
            spec_path,
            output=output,
            completion_marker=marker,
        )
    assert (output.read_bytes(), marker.read_bytes()) == before


def test_external_intake_cli_reports_ten_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec = _spec(tmp_path)
    spec_path = tmp_path / "intake.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    output_dir = tmp_path / "frozen"
    output_dir.mkdir()
    monkeypatch.setattr(
        "sys.argv",
        [
            "safe-slate-freeze-inputs",
            "--spec",
            str(spec_path),
            "--output",
            str(output_dir / "manifest.json"),
            "--completion-marker",
            str(output_dir / "manifest.complete.json"),
        ],
    )
    external_intake_main()
    assert "Frozen ten private input identities" in capsys.readouterr().out


def test_public_external_intake_template_is_complete_and_fail_closed() -> None:
    template = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    assert template["privacy_reviewed"] is False
    assert template["synthetic_inputs"] is False
    assert {entry["semantic_role"] for entry in template["inputs"]} == set(
        EXTERNAL_INPUT_ROLES
    )
    assert len(template["inputs"]) == 10
    assert all(
        entry["path"].startswith("/private/replace-me/")
        for entry in template["inputs"]
    )
    with pytest.raises(ValueError, match="privacy_reviewed"):
        freeze_external_input_manifest(template)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("privacy_reviewed", False, "privacy_reviewed"),
        ("synthetic_inputs", True, "synthetic_inputs"),
    ],
)
def test_external_intake_rejects_unaudited_or_synthetic_inputs(
    tmp_path: Path, field: str, value: object, match: str
) -> None:
    spec = _spec(tmp_path)
    spec[field] = value
    with pytest.raises(ValueError, match=match):
        freeze_external_input_manifest(spec)


def test_external_intake_requires_every_role_once(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    spec["inputs"] = spec["inputs"][:-1]
    with pytest.raises(ValueError, match="roles differ"):
        freeze_external_input_manifest(spec)


def test_external_intake_binds_positivity_population_and_canonical_bytes(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    positivity = next(
        value
        for value in spec["inputs"]
        if value["semantic_role"] == "population_positivity_evidence"
    )
    positivity["version"] = "different-evidence-id"
    with pytest.raises(ValueError, match="version must equal evidence_id"):
        freeze_external_input_manifest(spec)

    spec = _spec(tmp_path)
    positivity = next(
        value
        for value in spec["inputs"]
        if value["semantic_role"] == "population_positivity_evidence"
    )
    evidence = create_population_positivity_evidence(
        evidence_id="positivity-audit-v1",
        decision="established",
        population_identity="different-population",
        logging_policy_identity="baseline-v1",
        target_policy_identity="target-v2",
        audited_at="2026-10-08T00:00:00+09:00",
        auditor_identity="independent-design-auditor",
        method="registered logger lower-envelope audit",
        minimum_logging_probability_lower_bound=0.01,
    )
    Path(positivity["path"]).write_bytes(
        canonical_population_positivity_evidence_bytes(evidence)
    )
    with pytest.raises(ValueError, match="population identity mismatch"):
        freeze_external_input_manifest(spec)

    spec = _spec(tmp_path)
    positivity = next(
        value
        for value in spec["inputs"]
        if value["semantic_role"] == "population_positivity_evidence"
    )
    path = Path(positivity["path"])
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="not canonical"):
        freeze_external_input_manifest(spec)


def test_external_intake_accepts_canonical_statistical_positivity_v2(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    positivity = next(
        value
        for value in spec["inputs"]
        if value["semantic_role"] == "population_positivity_evidence"
    )
    evidence = create_statistical_population_positivity_evidence(
        evidence_id="positivity-audit-v2",
        decision="established",
        population_identity="jp-home-feed-eligible-v3",
        logging_policy_identity="baseline-v1",
        target_policy_identity="target-v2",
        audited_at="2026-09-30T00:00:00+09:00",
        auditor_identity="independent-design-auditor",
        method="epochwise metric-net exact lower audit",
        minimum_logging_probability_lower_bound=0.01,
        guarantee_kind="finite_sample",
        simultaneous_failure_probability=0.025,
        validity_start="2026-10-01T00:00:00+09:00",
        validity_end="2026-10-08T00:00:00+09:00",
        registered_epoch_count=2,
        audit_design_identity="metric-net-v2",
        source_certificate_sha256="c" * 64,
    )
    Path(positivity["path"]).write_bytes(
        canonical_population_positivity_evidence_bytes(evidence)
    )
    positivity["schema_version"] = evidence.schema_version
    positivity["version"] = evidence.evidence_id
    positivity["snapshot_time"] = evidence.audited_at

    manifest = freeze_external_input_manifest(spec)
    entry = next(
        value
        for value in manifest.inputs
        if value.semantic_role == "population_positivity_evidence"
    )
    assert entry.schema_version == evidence.schema_version
    assert entry.version == evidence.evidence_id


def test_external_intake_rejects_mislabeled_matrix_bytes(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    covariance = next(
        value
        for value in spec["inputs"]
        if value["semantic_role"] == "linear_parameter_covariance"
    )
    Path(covariance["path"]).write_bytes(b"not-a-canonical-matrix")
    with pytest.raises(ValueError, match="header"):
        freeze_external_input_manifest(spec)


def test_external_intake_rejects_trace_bound_to_different_primary_bytes(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    candidate = next(
        value
        for value in spec["inputs"]
        if value["semantic_role"] == "candidate_action_log"
    )
    path = Path(candidate["path"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["context"] = {"tampered_after_primary_commit": True}
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="digest mismatch"):
        freeze_external_input_manifest(spec)


def test_external_intake_spec_loader_rejects_noncanonical_json(tmp_path: Path) -> None:
    cases = [
        ("[]", "one JSON object"),
        ('{"value": 1, "value": 2}', "duplicate JSON key"),
        ('{"value": NaN}', "non-finite JSON"),
    ]
    for index, (contents, message) in enumerate(cases):
        path = tmp_path / f"bad-spec-{index}.json"
        path.write_text(contents, encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            load_intake_spec(path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda spec: spec.pop("manifest_id"), "keys differ"),
        (
            lambda spec: spec.update(schema_version="2.0"),
            "unsupported intake schema_version",
        ),
        (lambda spec: spec.update(maximum_input_bytes=True), "positive integer"),
        (lambda spec: spec.update(maximum_log_records=0), "positive integer"),
        (lambda spec: spec.update(inputs={}), "inputs must be a list"),
        (lambda spec: spec["inputs"].__setitem__(0, []), "must be an object"),
        (lambda spec: spec["inputs"][0].pop("input_id"), "keys differ"),
        (
            lambda spec: spec["inputs"][0].update(semantic_role="unknown"),
            "unknown semantic_role",
        ),
        (lambda spec: spec.update(maximum_input_bytes=1), "maximum_input_bytes"),
    ],
)
def test_external_intake_rejects_malformed_spec_contract(
    tmp_path: Path, mutation, message: str
) -> None:
    spec = _spec(tmp_path)
    mutation(spec)
    with pytest.raises(ValueError, match=message):
        freeze_external_input_manifest(spec)


@pytest.mark.parametrize(
    ("role", "field", "value", "message"),
    [
        (
            "candidate_action_log",
            "schema_version",
            "wrong",
            "candidate_action_log schema_version",
        ),
        (
            "service_trace_completion",
            "schema_version",
            "wrong",
            "service_trace_completion schema_version",
        ),
        (
            "linear_parameter_covariance",
            "schema_version",
            "wrong",
            "explicitly versioned matrix",
        ),
        (
            "population_positivity_evidence",
            "schema_version",
            "wrong",
            "positivity_evidence schema_version",
        ),
        (
            "population_positivity_evidence",
            "snapshot_time",
            "2026-10-07T00:00:00+09:00",
            "snapshot_time must equal audited_at",
        ),
        (
            "population_positivity_evidence",
            "temporal_scope",
            "feature",
            "must use atemporal scope",
        ),
    ],
)
def test_external_intake_rejects_role_specific_schema_mismatches(
    tmp_path: Path, role: str, field: str, value: str, message: str
) -> None:
    spec = _spec(tmp_path)
    entry = next(item for item in spec["inputs"] if item["semantic_role"] == role)
    entry[field] = value
    with pytest.raises(ValueError, match=message):
        freeze_external_input_manifest(spec)


def test_matrix_intake_rejects_noncanonical_and_truncated_encodings(
    tmp_path: Path,
) -> None:
    json_path = tmp_path / "matrix.json"
    json_path.write_bytes(canonical_matrix_bytes([[1.0]]) + b" ")
    with pytest.raises(ValueError, match="not in canonical"):
        _validate_matrix_file(json_path, "safe-slate-matrix-json-v1")

    cases = [
        (b"bad-header", "invalid header"),
        (F64LE_MAGIC + b"short", "missing its shape"),
        (F64LE_MAGIC + struct.pack("<QQ", 1, 1), "byte size"),
        (
            F64LE_MAGIC + struct.pack("<QQ", 1, 1) + struct.pack("<d", math.nan),
            "non-finite",
        ),
    ]
    for index, (contents, message) in enumerate(cases):
        path = tmp_path / f"matrix-{index}.bin"
        path.write_bytes(contents)
        with pytest.raises(ValueError, match=message):
            _validate_matrix_file(path, "safe-slate-matrix-f64le-v1")

    with pytest.raises(ValueError, match="unsupported matrix"):
        _validate_matrix_file(json_path, "unknown")


def test_external_intake_writer_requires_shared_existing_parent(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    spec_path = tmp_path / "intake.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    output_parent = tmp_path / "output"
    output_parent.mkdir()
    other_parent = tmp_path / "other"
    other_parent.mkdir()

    with pytest.raises(ValueError, match="contain both outputs"):
        write_external_input_manifest(
            spec_path,
            output=output_parent / "manifest.json",
            completion_marker=other_parent / "complete.json",
        )


def test_external_intake_accepts_canonical_json_matrices(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    for entry in spec["inputs"]:
        if entry["semantic_role"] in {
            "linear_parameter_covariance",
            "request_contrast_features",
        }:
            Path(entry["path"]).write_bytes(canonical_matrix_bytes([[1.0, 0.0]]))
            entry["schema_version"] = "safe-slate-matrix-json-v1"
    assert freeze_external_input_manifest(spec).verify_digest()


def test_external_intake_rejects_non_mapping_and_embedded_primary_trace(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="spec must be a mapping"):
        freeze_external_input_manifest([])  # type: ignore[arg-type]

    spec = _spec(tmp_path)
    candidate = next(
        entry
        for entry in spec["inputs"]
        if entry["semantic_role"] == "candidate_action_log"
    )
    completion = next(
        entry
        for entry in spec["inputs"]
        if entry["semantic_role"] == "service_trace_completion"
    )
    candidate_payload = json.loads(Path(candidate["path"]).read_text())
    completion_payload = json.loads(Path(completion["path"]).read_text())
    candidate_payload["service_latency_trace"] = completion_payload[
        "service_latency_trace"
    ]
    Path(candidate["path"]).write_text(json.dumps(candidate_payload) + "\n")
    with pytest.raises(ValueError, match="must not embed"):
        freeze_external_input_manifest(spec)


def test_external_intake_rejects_declared_positivity_schema_mismatch(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    positivity = next(
        entry
        for entry in spec["inputs"]
        if entry["semantic_role"] == "population_positivity_evidence"
    )
    positivity["schema_version"] = "safe-slate-population-positivity-evidence-2.0"
    with pytest.raises(ValueError, match="declared schema does not match bytes"):
        freeze_external_input_manifest(spec)


def test_external_intake_module_entrypoint(tmp_path: Path, monkeypatch) -> None:
    spec = _spec(tmp_path)
    spec_path = tmp_path / "module-intake.json"
    spec_path.write_text(json.dumps(spec))
    output_dir = tmp_path / "module-output"
    output_dir.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "safe_slate.external_intake",
            "--spec",
            str(spec_path),
            "--output",
            str(output_dir / "manifest.json"),
            "--completion-marker",
            str(output_dir / "complete.json"),
        ],
    )
    with pytest.warns(RuntimeWarning, match="found in sys.modules"):
        runpy.run_module("safe_slate.external_intake", run_name="__main__")
    assert (output_dir / "complete.json").is_file()
