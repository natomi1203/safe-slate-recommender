from __future__ import annotations

from dataclasses import replace

import pytest

from safe_slate import (
    LINEAR_PARAMETER_COVARIANCE_ROLE,
    REQUEST_CONTRAST_FEATURES_ROLE,
    audit_input_manifest_observations,
    audit_linear_uncertainty_snapshot_binding,
    canonical_matrix_bytes,
    canonical_matrix_f64le_bytes,
    create_immutable_input,
    create_input_manifest,
    create_observed_input_identity,
    guarded_linear_rerank,
    matrix_f64le_sha256,
    matrix_sha256,
)


COVARIANCE = [[0.0001, 0.0], [0.0, 0.0001]]
CONTRASTS = [[1.0, 0.0], [0.5, 1.0]]
SNAPSHOT_ID = "covariance-snapshot-v17"
CONTRAST_INPUT_ID = "contrast-binding-request"
MODEL_VERSION = "ranker-v17"


def _proposal_log(*, matrix_digest_encoding: str = "json-v1"):
    return guarded_linear_rerank(
        request_id="binding-request",
        event_timestamp="2026-10-02T12:00:00+00:00",
        serving_mode="shadow",
        uncertainty_snapshot_id=SNAPSHOT_ID,
        contrast_input_id=CONTRAST_INPUT_ID,
        model_version=MODEL_VERSION,
        baseline_item_ids=[10, 11, 12, 13],
        protected_count=2,
        remove_positions=[2, 3],
        add_item_ids=[100, 101],
        reward_deltas=[0.2, 0.1],
        safety_mean_deltas=[0.04, 0.03],
        contrast_features=CONTRASTS,
        parameter_covariance=COVARIANCE,
        baseline_metric=0.5,
        degradation_tolerance=0.02,
        max_replacements=2,
        support_ok=True,
        uncertainty_valid=True,
        distribution_shift_detected=False,
        proposal_log_complete=True,
        matrix_digest_encoding=matrix_digest_encoding,
    ).proposal_log


def _manifest(
    *,
    digest: str | None = None,
    version: str = MODEL_VERSION,
    role: str = LINEAR_PARAMETER_COVARIANCE_ROLE,
    snapshot_time: str = "2026-10-01T00:00:00+00:00",
    contrast_digest: str | None = None,
    contrast_role: str = REQUEST_CONTRAST_FEATURES_ROLE,
    contrast_time: str = "2026-10-02T11:59:00+00:00",
    matrix_digest_encoding: str = "json-v1",
    temporal_scope: str = "feature",
    contrast_temporal_scope: str = "feature",
    schema_version: str | None = None,
    contrast_schema_version: str | None = None,
    contrast_version: str = MODEL_VERSION,
):
    encode = (
        canonical_matrix_f64le_bytes
        if matrix_digest_encoding == "f64le-v1"
        else canonical_matrix_bytes
    )
    digest_matrix = (
        matrix_f64le_sha256
        if matrix_digest_encoding == "f64le-v1"
        else matrix_sha256
    )
    payload = encode(COVARIANCE)
    entry = create_immutable_input(
        input_id=SNAPSHOT_ID,
        semantic_role=role,
        canonical_uri="gs://model-registry/covariance-v17.json",
        version=version,
        byte_size=len(payload),
        sha256_digest=digest or digest_matrix(COVARIANCE),
        snapshot_time=snapshot_time,
        temporal_scope=temporal_scope,
        schema_version=schema_version
        or (
            "safe-slate-matrix-f64le-v1"
            if matrix_digest_encoding == "f64le-v1"
            else "safe-slate-linear-covariance-1.0"
        ),
    )
    contrast_payload = encode(CONTRASTS)
    contrast_entry = create_immutable_input(
        input_id=CONTRAST_INPUT_ID,
        semantic_role=contrast_role,
        canonical_uri="gs://proposal-features/binding-request.json",
        version=contrast_version,
        byte_size=len(contrast_payload),
        sha256_digest=contrast_digest or digest_matrix(CONTRASTS),
        snapshot_time=contrast_time,
        temporal_scope=contrast_temporal_scope,
        schema_version=contrast_schema_version
        or (
            "safe-slate-matrix-f64le-v1"
            if matrix_digest_encoding == "f64le-v1"
            else "safe-slate-request-contrasts-1.0"
        ),
    )
    manifest = create_input_manifest(
        manifest_id="linear-binding-v1",
        population_identity="home-feed-jp-v1",
        created_at="2026-10-04T00:00:00+00:00",
        feature_cutoff="2026-10-03T00:00:00+00:00",
        label_cutoff="2026-10-03T12:00:00+00:00",
        required_roles=[role, contrast_role],
        inputs=[entry, contrast_entry],
    )
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
        for value in manifest.inputs
    ]
    return manifest, audit_input_manifest_observations(manifest, observations)


def test_uncertainty_snapshot_binding_accepts_exact_observed_input() -> None:
    manifest, observations = _manifest()
    assessment = audit_linear_uncertainty_snapshot_binding(
        _proposal_log(),
        manifest=manifest,
        observed_inputs=observations,
    )

    assert assessment.valid
    assert assessment.snapshot_manifest_bound
    assert assessment.snapshot_observed
    assert assessment.contrast_manifest_bound
    assert assessment.contrast_observed
    assert assessment.model_version_bound
    assert assessment.contrast_digest_recorded
    assert assessment.reasons == ()


def test_uncertainty_binding_accepts_fast_f64le_input_identity() -> None:
    manifest, observations = _manifest(matrix_digest_encoding="f64le-v1")
    assessment = audit_linear_uncertainty_snapshot_binding(
        _proposal_log(matrix_digest_encoding="f64le-v1"),
        manifest=manifest,
        observed_inputs=observations,
    )

    assert assessment.valid
    assert assessment.matrix_digest_encoding == "f64le-v1"


def test_uncertainty_binding_rejects_unregistered_digest_encoding() -> None:
    manifest, observations = _manifest()
    log = _proposal_log()
    context = dict(log.context)
    metadata = dict(context["safe_slate_uncertainty"])
    metadata["matrix_digest_encoding"] = "unknown"
    context["safe_slate_uncertainty"] = metadata
    assessment = audit_linear_uncertainty_snapshot_binding(
        replace(log, context=context),
        manifest=manifest,
        observed_inputs=observations,
    )

    assert not assessment.valid
    assert "linear_uncertainty_digest_encoding_invalid" in assessment.reasons


@pytest.mark.parametrize(
    ("manifest_kwargs", "reason"),
    [
        ({"contrast_digest": "0" * 64}, "contrast_input_digest_mismatch"),
        ({"contrast_role": "generic_features"}, "contrast_input_role_mismatch"),
        (
            {"contrast_time": "2026-10-02T13:00:00+00:00"},
            "contrast_input_after_request",
        ),
    ],
)
def test_uncertainty_binding_rejects_mismatched_contrast_payload(
    manifest_kwargs: dict[str, str],
    reason: str,
) -> None:
    manifest, observations = _manifest(**manifest_kwargs)
    assessment = audit_linear_uncertainty_snapshot_binding(
        _proposal_log(),
        manifest=manifest,
        observed_inputs=observations,
    )
    assert not assessment.valid
    assert reason in assessment.reasons


@pytest.mark.parametrize(
    ("manifest_kwargs", "reason"),
    [
        ({"digest": "0" * 64}, "uncertainty_snapshot_digest_mismatch"),
        ({"version": "ranker-v18"}, "uncertainty_snapshot_model_version_mismatch"),
        ({"role": "generic_model"}, "uncertainty_snapshot_role_mismatch"),
        (
            {"snapshot_time": "2026-10-03T00:00:00+00:00"},
            "uncertainty_snapshot_after_request",
        ),
    ],
)
def test_uncertainty_snapshot_binding_rejects_mismatched_manifest(
    manifest_kwargs: dict[str, str],
    reason: str,
) -> None:
    manifest, observations = _manifest(**manifest_kwargs)
    assessment = audit_linear_uncertainty_snapshot_binding(
        _proposal_log(),
        manifest=manifest,
        observed_inputs=observations,
    )

    assert not assessment.valid
    assert reason in assessment.reasons


def test_uncertainty_snapshot_binding_requires_preflight_observation() -> None:
    manifest, _ = _manifest()
    missing = audit_input_manifest_observations(manifest, [])
    assessment = audit_linear_uncertainty_snapshot_binding(
        _proposal_log(),
        manifest=manifest,
        observed_inputs=missing,
    )

    assert not assessment.valid
    assert "input_observation_invalid" in assessment.reasons
    assert not assessment.snapshot_observed


def test_uncertainty_snapshot_binding_rejects_tampered_request_metadata() -> None:
    manifest, observations = _manifest()
    log = _proposal_log()
    context = dict(log.context)
    metadata = dict(context["safe_slate_uncertainty"])
    metadata["operation_count"] = 999
    context["safe_slate_uncertainty"] = metadata
    tampered = replace(log, context=context)

    assessment = audit_linear_uncertainty_snapshot_binding(
        tampered,
        manifest=manifest,
        observed_inputs=observations,
    )
    assert not assessment.valid
    assert "linear_uncertainty_operation_count_mismatch" in assessment.reasons


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (lambda metadata: metadata.update(method="other"), "linear_uncertainty_method_invalid"),
        (
            lambda metadata: metadata.update(representation="dense"),
            "linear_uncertainty_representation_invalid",
        ),
        (
            lambda metadata: metadata.update(feature_dimension=True),
            "linear_uncertainty_feature_dimension_invalid",
        ),
        (
            lambda metadata: metadata.update(feature_dimension=-1),
            "linear_uncertainty_feature_dimension_invalid",
        ),
        (lambda metadata: metadata.update(snapshot_id=""), "uncertainty_snapshot_id_missing"),
        (lambda metadata: metadata.update(contrast_input_id=7), "contrast_input_id_missing"),
        (
            lambda metadata: metadata.update(parameter_covariance_sha256="xyz"),
            "parameter_covariance_digest_invalid",
        ),
        (
            lambda metadata: metadata.update(contrast_features_sha256="A" * 64),
            "contrast_features_digest_invalid",
        ),
        (
            lambda metadata: metadata.update(snapshot_id="missing"),
            "uncertainty_snapshot_manifest_missing",
        ),
        (
            lambda metadata: metadata.update(contrast_input_id="missing"),
            "contrast_input_manifest_missing",
        ),
    ],
)
def test_uncertainty_binding_rejects_malformed_metadata(mutation, reason: str) -> None:
    manifest, observations = _manifest()
    log = _proposal_log()
    context = dict(log.context)
    metadata = dict(context["safe_slate_uncertainty"])
    mutation(metadata)
    context["safe_slate_uncertainty"] = metadata

    assessment = audit_linear_uncertainty_snapshot_binding(
        replace(log, context=context),
        manifest=manifest,
        observed_inputs=observations,
    )

    assert not assessment.valid
    assert reason in assessment.reasons
    assert assessment.to_dict()["valid"] is False


def test_uncertainty_binding_requires_mapping_metadata_and_aware_event_time() -> None:
    manifest, observations = _manifest()
    log = _proposal_log()
    for context, event_timestamp, reason in [
        ({}, log.event_timestamp, "linear_uncertainty_metadata_missing"),
        (
            {"safe_slate_uncertainty": []},
            log.event_timestamp,
            "linear_uncertainty_metadata_missing",
        ),
        (log.context, "not-a-time", "proposal_event_timestamp_missing_or_invalid"),
        (log.context, "2026-10-02T12:00:00", "proposal_event_timestamp_missing_or_invalid"),
    ]:
        assessment = audit_linear_uncertainty_snapshot_binding(
            replace(log, context=context, event_timestamp=event_timestamp),
            manifest=manifest,
            observed_inputs=observations,
        )
        assert not assessment.valid
        assert reason in assessment.reasons


@pytest.mark.parametrize(
    ("manifest_kwargs", "reason"),
    [
        (
            {
                "temporal_scope": "label",
                "snapshot_time": "2026-10-03T01:00:00+00:00",
            },
            "uncertainty_snapshot_temporal_scope_invalid",
        ),
        (
            {
                "contrast_temporal_scope": "label",
                "contrast_time": "2026-10-03T01:00:00+00:00",
            },
            "contrast_input_temporal_scope_invalid",
        ),
        (
            {
                "matrix_digest_encoding": "f64le-v1",
                "schema_version": "safe-slate-linear-covariance-1.0",
            },
            "uncertainty_snapshot_digest_encoding_mismatch",
        ),
        (
            {
                "matrix_digest_encoding": "f64le-v1",
                "contrast_schema_version": "safe-slate-request-contrasts-1.0",
            },
            "contrast_input_digest_encoding_mismatch",
        ),
        (
            {"contrast_version": "ranker-v18"},
            "contrast_input_model_version_mismatch",
        ),
    ],
)
def test_uncertainty_binding_rejects_manifest_contract_mismatches(
    manifest_kwargs: dict[str, str], reason: str
) -> None:
    manifest, observations = _manifest(**manifest_kwargs)
    encoding = manifest_kwargs.get("matrix_digest_encoding", "json-v1")
    assessment = audit_linear_uncertainty_snapshot_binding(
        _proposal_log(matrix_digest_encoding=encoding),
        manifest=manifest,
        observed_inputs=observations,
    )
    assert not assessment.valid
    assert reason in assessment.reasons


def test_uncertainty_binding_rejects_mismatched_or_incomplete_observation() -> None:
    manifest, observations = _manifest()
    mismatched = replace(observations, manifest_digest="0" * 64)
    missing_contrast = replace(
        observations,
        verified_input_ids=(SNAPSHOT_ID,),
    )

    mismatch = audit_linear_uncertainty_snapshot_binding(
        _proposal_log(), manifest=manifest, observed_inputs=mismatched
    )
    incomplete = audit_linear_uncertainty_snapshot_binding(
        _proposal_log(), manifest=manifest, observed_inputs=missing_contrast
    )

    assert "input_observation_manifest_mismatch" in mismatch.reasons
    assert "contrast_input_not_observed" in incomplete.reasons


def test_uncertainty_binding_validates_public_argument_types() -> None:
    manifest, observations = _manifest()
    log = _proposal_log()
    invalid_calls = [
        (("bad", manifest, observations, LINEAR_PARAMETER_COVARIANCE_ROLE, REQUEST_CONTRAST_FEATURES_ROLE), "proposal_log"),
        ((log, "bad", observations, LINEAR_PARAMETER_COVARIANCE_ROLE, REQUEST_CONTRAST_FEATURES_ROLE), "manifest"),
        ((log, manifest, "bad", LINEAR_PARAMETER_COVARIANCE_ROLE, REQUEST_CONTRAST_FEATURES_ROLE), "observed_inputs"),
        ((log, manifest, observations, "", REQUEST_CONTRAST_FEATURES_ROLE), "expected_semantic_role"),
        ((log, manifest, observations, LINEAR_PARAMETER_COVARIANCE_ROLE, ""), "expected_contrast_role"),
    ]
    for (proposal, input_manifest, observed, role, contrast_role), message in invalid_calls:
        with pytest.raises(ValueError, match=message):
            audit_linear_uncertainty_snapshot_binding(
                proposal,  # type: ignore[arg-type]
                manifest=input_manifest,  # type: ignore[arg-type]
                observed_inputs=observed,  # type: ignore[arg-type]
                expected_semantic_role=role,
                expected_contrast_role=contrast_role,
            )


def test_uncertainty_binding_rejects_non_string_time_tampered_manifest_and_missing_model() -> None:
    manifest, observations = _manifest()
    log = _proposal_log()
    for timestamp in (None, ""):
        assessment = audit_linear_uncertainty_snapshot_binding(
            replace(log, event_timestamp=timestamp),
            manifest=manifest,
            observed_inputs=observations,
        )
        assert "proposal_event_timestamp_missing_or_invalid" in assessment.reasons

    tampered_manifest = replace(manifest, manifest_digest="0" * 64)
    tampered = audit_linear_uncertainty_snapshot_binding(
        log,
        manifest=tampered_manifest,
        observed_inputs=observations,
    )
    assert "input_manifest_invalid_or_tampered" in tampered.reasons

    missing_model = audit_linear_uncertainty_snapshot_binding(
        replace(log, model_version=None),
        manifest=manifest,
        observed_inputs=observations,
    )
    assert "proposal_model_version_missing" in missing_model.reasons
