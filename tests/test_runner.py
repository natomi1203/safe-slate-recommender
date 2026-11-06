from __future__ import annotations

from pathlib import Path

import pytest

from safe_slate import (
    CrossFittedOpeEstimate,
    JointUncertaintyBoundsEstimate,
    MissingnessAdjustedOpeEstimate,
    PartialIdentificationEstimate,
    PropensityIntervalEstimate,
    create_observed_input_identity,
    audit_manifest_bound_ope_readiness,
    freeze_external_input_manifest,
    load_population_positivity_evidence_file,
    observe_manifest_inputs,
    run_no_clobber_ope,
)
from test_external_intake import _spec
from test_manifest_bound_ope import (
    _contract,
    _manifest,
    _observations,
    _positivity_evidence,
    _rows,
)
from safe_slate.runner import _attestation, _require


def _estimation_evidence(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "outcome_prediction_logged_action": [0.4] * 4,
        "outcome_prediction_target_policy": [0.6] * 4,
        "outcome_prediction_supported_target_contribution": [0.5] * 4,
        "nuisance_predictions_cross_fitted": True,
        "propensity_evidence_is_request_time": True,
        "outcomes_match_logged_actions": True,
        "propensity_intervals_cover_true_probabilities": True,
        "observation_propensities_are_validated": True,
        "observation_propensities_are_request_time_probabilities": True,
        "conditional_missingness_independence": True,
    }
    values.update(overrides)
    return values


def _run(
    target: Path,
    *,
    contract=None,
    manifest=None,
    observations=None,
    rows=None,
    estimation=None,
    collection_complete: bool = True,
    label_snapshot_digest: str | None = "c" * 64,
    **overrides,
):
    manifest = manifest or _manifest()
    contract = contract or _contract(manifest)
    values = {
        "output_identity": getattr(contract, "output_namespace", "invalid"),
        "contract": contract,
        "input_manifest": manifest,
        "observed_inputs": observations or _observations(manifest),
        "population_positivity_evidence": _positivity_evidence(),
        "analysis_timestamp": "2026-10-02T00:00:00+09:00",
        "created_at": "2026-10-02T00:00:01+09:00",
        "row_evidence": rows or _rows(),
        "estimation_evidence": estimation or _estimation_evidence(),
        "collection_complete": collection_complete,
        "label_snapshot_digest": label_snapshot_digest,
        "successful_execution_ids": ("local-run-001",),
        "peak_memory_mb": 128,
        "no_matching_execution_verified": True,
    }
    values.update(overrides)
    return run_no_clobber_ope(target, **values)


def test_runner_publishes_full_support_point_ope(tmp_path: Path) -> None:
    target = tmp_path / "point-ope"

    result = _run(target)

    assert result.state == "artifact_complete"
    assert isinstance(result.estimate, CrossFittedOpeEstimate)
    assert result.bundle_manifest is not None
    assert result.bundle_verification is not None
    assert result.bundle_verification.valid
    assert result.output_directory == str(target)
    assert not result.production_rollout_authorized
    assert (target / "complete.json").is_file()
    assert result.to_dict()["state"] == "artifact_complete"


def test_runner_rejects_missing_fields_and_invalid_public_arguments(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="required evaluation field"):
        _require({}, "missing")
    with pytest.raises(ValueError, match="attestation must be boolean"):
        _attestation({"flag": 1}, "flag")

    manifest = _manifest()
    contract = _contract(manifest)
    cases = [
        ({"contract": "bad"}, "untampered"),
        ({"output_identity": "wrong"}, "output identity"),
        ({"collection_complete": "yes"}, "must be boolean"),
    ]
    for index, (overrides, message) in enumerate(cases):
        with pytest.raises(ValueError, match=message):
            _run(tmp_path / f"invalid-{index}", **overrides)

    with pytest.raises(ValueError, match="parent directory"):
        _run(tmp_path / "missing" / "output")


def test_freeze_cli_manifest_flows_directly_to_ope_preflight(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    manifest = freeze_external_input_manifest(spec)
    positivity_input = next(
        value
        for value in spec["inputs"]
        if value["semantic_role"] == "population_positivity_evidence"
    )
    positivity = load_population_positivity_evidence_file(
        positivity_input["path"]  # type: ignore[arg-type]
    )
    observations = observe_manifest_inputs(manifest)
    assert observations.valid
    contract = _contract(
        manifest,
        population_identity="jp-home-feed-eligible-v3",
        final_window_end="2026-10-01T00:00:00+09:00",
        earliest_analysis_time="2026-10-08T00:00:00+09:00",
    )

    result = audit_manifest_bound_ope_readiness(
        contract,
        manifest,
        observations.observations,
        analysis_timestamp="2026-10-09T00:00:00+09:00",
        population_positivity_evidence=positivity,
        **_rows(),
    )

    assert result.immutable_inputs_verified
    assert result.decision == "pass"
    assert result.readiness is not None
    assert result.readiness.standard_ope_authorized
    assert not any("required_input_role_missing" in value for value in result.reasons)


def test_runner_dispatches_known_partial_support_bounds(tmp_path: Path) -> None:
    rows = _rows()
    rows["guaranteed_target_support_mass"] = [0.9] * 4

    result = _run(tmp_path / "partial", rows=rows)

    assert result.state == "artifact_complete"
    assert isinstance(result.estimate, PartialIdentificationEstimate)
    assert result.bundle_verification is not None
    assert result.bundle_verification.valid


def test_runner_dispatches_propensity_interval_bounds(tmp_path: Path) -> None:
    manifest = _manifest()
    contract = _contract(
        manifest,
        propensity_mode="interval",
        minimum_user_effective_sample_size=0.5,
    )
    rows = _rows()
    rows.pop("logging_propensities")
    rows["logging_propensity_lower"] = [0.4] * 4
    rows["logging_propensity_upper"] = [0.6] * 4

    result = _run(
        tmp_path / "interval",
        manifest=manifest,
        contract=contract,
        rows=rows,
    )

    assert result.state == "artifact_complete"
    assert isinstance(result.estimate, PropensityIntervalEstimate)
    assert result.bundle_verification is not None
    assert result.bundle_verification.valid


def test_runner_dispatches_missingness_adjusted_point_ope(tmp_path: Path) -> None:
    manifest = _manifest()
    contract = _contract(manifest, maximum_missing_outcome_rate=0.25)
    rows = _rows()
    rows["purchase_events"] = [1, None, 1, 0]
    rows["outcome_observation_propensities"] = [0.8] * 4

    result = _run(
        tmp_path / "missing",
        manifest=manifest,
        contract=contract,
        rows=rows,
    )

    assert result.state == "artifact_complete"
    assert isinstance(result.estimate, MissingnessAdjustedOpeEstimate)
    assert result.bundle_verification is not None
    assert result.bundle_verification.valid


def test_runner_dispatches_joint_support_and_missingness_bounds(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    contract = _contract(manifest, maximum_missing_outcome_rate=0.25)
    rows = _rows()
    rows["guaranteed_target_support_mass"] = [0.9] * 4
    rows["purchase_events"] = [1, None, 1, 0]
    rows["outcome_observation_propensities"] = [0.8] * 4

    result = _run(
        tmp_path / "joint",
        manifest=manifest,
        contract=contract,
        rows=rows,
    )

    assert result.state == "artifact_complete"
    assert isinstance(result.estimate, JointUncertaintyBoundsEstimate)
    assert result.bundle_verification is not None
    assert result.bundle_verification.valid


def test_runner_stops_before_estimation_on_input_mismatch(tmp_path: Path) -> None:
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
    target = tmp_path / "rejected"

    result = _run(target, manifest=manifest, observations=observations)

    assert result.state == "evidence_rejected"
    assert result.estimate is None
    assert result.bundle_manifest is None
    assert result.output_directory is None
    assert not target.exists()


def test_runner_preserves_insufficient_evidence_without_estimating(
    tmp_path: Path,
) -> None:
    rows = _rows()
    rows["guaranteed_target_support_mass"] = [0.5] * 4
    target = tmp_path / "insufficient"

    result = _run(target, rows=rows)

    assert result.state == "insufficient_evidence"
    assert result.readiness is not None
    assert result.readiness.decision == "fail"
    assert result.estimate is None
    assert result.bundle_manifest is not None
    assert result.bundle_manifest.estimate_type is None
    assert result.bundle_verification is not None
    assert result.bundle_verification.valid
    assert target.is_dir()
    assert not (target / "estimate.json").exists()


def test_runner_refuses_existing_or_unverified_execution_identity(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        _run(existing)

    manifest = _manifest()
    contract = _contract(manifest)
    with pytest.raises(ValueError, match="execution identity"):
        run_no_clobber_ope(
            tmp_path / "unchecked",
            output_identity=contract.output_namespace,
            contract=contract,
            input_manifest=manifest,
            observed_inputs=_observations(manifest),
            population_positivity_evidence=_positivity_evidence(),
            analysis_timestamp="2026-10-02T00:00:00+09:00",
            created_at="2026-10-02T00:00:01+09:00",
            row_evidence=_rows(),
            estimation_evidence=_estimation_evidence(),
            collection_complete=True,
            label_snapshot_digest="c" * 64,
            successful_execution_ids=("run",),
            peak_memory_mb=128,
            no_matching_execution_verified=False,
        )


@pytest.mark.parametrize(
    ("overrides", "expected_state"),
    [
        ({"collection_complete": False}, "partial_invalid"),
        ({"label_snapshot_digest": "invalid"}, "evidence_rejected"),
    ],
)
def test_runner_does_not_estimate_when_evidence_is_not_analysis_ready(
    tmp_path: Path,
    overrides: dict[str, object],
    expected_state: str,
) -> None:
    target = tmp_path / expected_state

    result = _run(target, **overrides)

    assert result.state == expected_state
    assert result.estimate is None
    assert result.bundle_manifest is None
    assert not target.exists()


def test_runner_does_not_publish_unsettled_labels(tmp_path: Path) -> None:
    rows = _rows()
    rows["labels_settled"] = False
    target = tmp_path / "labels-pending"

    result = _run(target, rows=rows)

    assert result.state == "labels_pending"
    assert result.estimate is None
    assert result.bundle_manifest is None
    assert not target.exists()
