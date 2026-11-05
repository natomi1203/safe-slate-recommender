from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest

from safe_slate import (
    audit_input_manifest_observations,
    create_immutable_input,
    create_input_manifest,
    create_observed_input_identity,
    immutable_input_from_local_file,
    load_input_manifest,
    observe_manifest_inputs,
    observed_input_from_local_file,
    verify_local_manifest_inputs,
)


REQUIRED_ROLES = (
    "candidate_action_logs",
    "propensity_evidence",
    "settled_labels",
    "pseudonymous_user_blocks",
    "constraint_traces",
    "shadow_records",
)


def _digest(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def _input(
    input_id: str,
    role: str,
    *,
    temporal_scope: str = "feature",
    snapshot_time: str = "2026-09-02T00:00:00+09:00",
    canonical_uri: str | None = None,
    version: str = "v1",
):
    return create_immutable_input(
        input_id=input_id,
        semantic_role=role,
        canonical_uri=canonical_uri or f"gs://research-inputs/{input_id}.parquet",
        version=version,
        generation="123",
        metageneration="1",
        byte_size=100,
        sha256_digest=_digest(input_id),
        snapshot_time=snapshot_time,
        temporal_scope=temporal_scope,  # type: ignore[arg-type]
        schema_version="1.0",
    )


def _inputs():
    return [
        _input("candidate-log", "candidate_action_logs"),
        _input("propensities", "propensity_evidence"),
        _input(
            "labels",
            "settled_labels",
            temporal_scope="label",
            snapshot_time="2026-10-01T00:00:00+09:00",
        ),
        _input("user-blocks", "pseudonymous_user_blocks"),
        _input("constraints", "constraint_traces"),
        _input("shadow", "shadow_records"),
        _input(
            "model",
            "target_model",
            temporal_scope="atemporal",
            snapshot_time="2026-08-31T00:00:00+09:00",
            canonical_uri="model://registry/target-v2",
        ),
        _input(
            "query",
            "frozen_query",
            temporal_scope="atemporal",
            snapshot_time="2026-08-31T00:00:00+09:00",
            canonical_uri="query://warehouse/purchase-ope-v1",
        ),
        _input(
            "code",
            "evaluator_code",
            temporal_scope="atemporal",
            snapshot_time="2026-08-31T00:00:00+09:00",
            canonical_uri="git://repository/02efa0d",
        ),
    ]


def _manifest(**overrides: object):
    values: dict[str, object] = {
        "manifest_id": "purchase-ope-inputs-v1",
        "population_identity": "eligible-population-v1",
        "created_at": "2026-10-02T00:00:00+09:00",
        "feature_cutoff": "2026-09-02T00:00:00+09:00",
        "label_cutoff": "2026-10-02T00:00:00+09:00",
        "required_roles": REQUIRED_ROLES,
        "inputs": _inputs(),
    }
    values.update(overrides)
    return create_input_manifest(**values)  # type: ignore[arg-type]


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


def _observe_entry(value):
    return create_observed_input_identity(
        input_id=value.input_id,
        canonical_uri=value.canonical_uri,
        version=value.version,
        generation=value.generation,
        metageneration=value.metageneration,
        byte_size=value.byte_size,
        sha256_digest=value.sha256,
    )


def test_manifest_is_sorted_stable_and_round_trips() -> None:
    first = _manifest()
    second = _manifest(inputs=list(reversed(_inputs())))

    assert first.manifest_digest == second.manifest_digest
    assert [entry.input_id for entry in first.inputs] == sorted(
        entry.input_id for entry in first.inputs
    )
    assert len(first.manifest_digest) == 64
    assert first.verify_digest()
    assert load_input_manifest(first.to_dict()) == first


def test_manifest_rejects_missing_roles_and_duplicate_identities() -> None:
    with pytest.raises(ValueError, match="required input roles are missing"):
        _manifest(inputs=_inputs()[:-4])
    duplicated_id = _inputs()
    duplicated_id.append(
        _input("candidate-log", "another_role", canonical_uri="gs://other/path")
    )
    with pytest.raises(ValueError, match="input_id"):
        _manifest(inputs=duplicated_id)
    duplicated_resource = _inputs()
    duplicated_resource.append(
        _input(
            "other-id",
            "another_role",
            canonical_uri=duplicated_resource[0].canonical_uri,
        )
    )
    with pytest.raises(ValueError, match="resource identities"):
        _manifest(inputs=duplicated_resource)


def test_feature_and_label_boundaries_are_enforced() -> None:
    late_feature = _inputs()
    late_feature[0] = _input(
        "candidate-log",
        "candidate_action_logs",
        snapshot_time="2026-09-03T00:00:00+09:00",
    )
    with pytest.raises(ValueError, match="feature input crosses"):
        _manifest(inputs=late_feature)

    early_label = _inputs()
    early_label[2] = _input(
        "labels",
        "settled_labels",
        temporal_scope="label",
        snapshot_time="2026-09-01T00:00:00+09:00",
    )
    with pytest.raises(ValueError, match="label input crosses"):
        _manifest(inputs=early_label)

    late_label = _inputs()
    late_label[2] = _input(
        "labels",
        "settled_labels",
        temporal_scope="label",
        snapshot_time="2026-10-03T00:00:00+09:00",
    )
    with pytest.raises(ValueError, match="label input crosses"):
        _manifest(inputs=late_label)


@pytest.mark.parametrize(
    "uri",
    [
        "/relative/or/absolute/path",
        "https://user:password@example.com/data",
        "https://example.com/data?signature=secret",
        "https://example.com/data#token",
        "file://relative/path",
    ],
)
def test_manifest_uri_cannot_embed_ambient_or_secret_identity(uri: str) -> None:
    with pytest.raises(ValueError):
        _input("unsafe", "unsafe_role", canonical_uri=uri)


@pytest.mark.parametrize(
    "overrides",
    [
        {"contains_credentials": True},
        {"contains_direct_identifiers": True},
        {"byte_size": -1},
        {"sha256_digest": "invalid"},
        {"version": ""},
        {"snapshot_time": "2026-01-01T00:00:00"},
        {"temporal_scope": "invalid"},
    ],
)
def test_invalid_input_identity_is_rejected(overrides: dict[str, object]) -> None:
    values: dict[str, object] = {
        "input_id": "input",
        "semantic_role": "role",
        "canonical_uri": "gs://bucket/object",
        "version": "v1",
        "byte_size": 1,
        "sha256_digest": _digest("input"),
        "snapshot_time": "2026-01-01T00:00:00+09:00",
        "temporal_scope": "atemporal",
        "schema_version": "1.0",
    }
    values.update(overrides)
    with pytest.raises(ValueError):
        create_immutable_input(**values)  # type: ignore[arg-type]


def test_serialized_manifest_tampering_is_rejected() -> None:
    manifest = _manifest()
    payload = manifest.to_dict()
    payload["inputs"][0]["byte_size"] = 999

    with pytest.raises(ValueError, match="digest mismatch"):
        load_input_manifest(payload)


def test_local_file_helper_and_reverification_detect_mutation(tmp_path: Path) -> None:
    local = tmp_path / "query.sql"
    local.write_text("SELECT 1\n", encoding="utf-8")
    entry = immutable_input_from_local_file(
        local,
        input_id="query",
        semantic_role="frozen_query",
        version="git:abc",
        snapshot_time="2026-08-31T00:00:00+09:00",
        temporal_scope="atemporal",
        schema_version="sql-v1",
    )
    inputs = [value for value in _inputs() if value.input_id != "query"] + [entry]
    manifest = _manifest(inputs=inputs)

    verified = verify_local_manifest_inputs(manifest)
    assert verified.valid
    assert verified.local_inputs_verified == 1
    assert verified.external_inputs_declared == len(inputs) - 1

    local_observation = observed_input_from_local_file(
        local,
        input_id="query",
        version="git:abc",
    )
    observed = [
        local_observation
        if value.input_id == "query"
        else create_observed_input_identity(
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
    assert audit_input_manifest_observations(manifest, observed).valid
    batch = observe_manifest_inputs(
        manifest,
        observers={scheme: _observe_entry for scheme in ("gs", "model", "git")},
    )
    assert batch.valid

    local.write_text("SELECT 2\n", encoding="utf-8")
    changed = verify_local_manifest_inputs(manifest)
    assert not changed.valid
    assert "local_input_digest_mismatch:query" in changed.reasons
    changed_observation = observed_input_from_local_file(
        local,
        input_id="query",
        version="git:abc",
    )
    observed = [
        changed_observation if value.input_id == "query" else value
        for value in observed
    ]
    changed_audit = audit_input_manifest_observations(manifest, observed)
    assert not changed_audit.valid
    assert "query" in changed_audit.mismatched_input_ids
    changed_batch = observe_manifest_inputs(
        manifest,
        observers={scheme: _observe_entry for scheme in ("gs", "model", "git")},
    )
    assert not changed_batch.valid
    assert "query" in changed_batch.verification.mismatched_input_ids


def test_manifest_time_order_and_timezone_are_required() -> None:
    with pytest.raises(ValueError):
        _manifest(created_at="2026-10-01T00:00:00+09:00")
    with pytest.raises(ValueError):
        _manifest(feature_cutoff="2026-09-02T00:00:00")


def test_observed_input_audit_requires_exact_complete_identity_set() -> None:
    manifest = _manifest()
    observations = _observations(manifest)

    verification = audit_input_manifest_observations(manifest, observations)

    assert verification.valid
    assert verification.manifest_digest == manifest.manifest_digest
    assert verification.expected_input_count == len(manifest.inputs)
    assert verification.observed_input_count == len(manifest.inputs)
    assert verification.verified_input_ids == tuple(
        value.input_id for value in manifest.inputs
    )
    assert verification.reasons == ()


def test_backend_independent_observer_registry_audits_all_schemes() -> None:
    manifest = _manifest()

    batch = observe_manifest_inputs(
        manifest,
        observers={
            scheme: _observe_entry for scheme in ("gs", "model", "query", "git")
        },
    )

    assert batch.valid
    assert batch.verification.valid
    assert len(batch.observations) == len(manifest.inputs)
    assert batch.reasons == ()


def test_observer_registry_fails_closed_on_missing_or_failed_adapter() -> None:
    manifest = _manifest()
    missing = observe_manifest_inputs(manifest)

    assert not missing.valid
    assert missing.observations == ()
    assert any(
        reason.startswith("input_observer_missing") for reason in missing.reasons
    )

    def failed_observer(_):
        raise RuntimeError("credential=must-not-appear")

    failed = observe_manifest_inputs(
        manifest,
        observers={
            "gs": failed_observer,
            "model": _observe_entry,
            "query": _observe_entry,
            "git": _observe_entry,
        },
    )
    assert not failed.valid
    assert any(
        reason == "input_observer_failed:candidate-log:RuntimeError"
        for reason in failed.reasons
    )
    assert all("must-not-appear" not in reason for reason in failed.reasons)


def test_observed_input_audit_reports_missing_extra_duplicate_and_changed() -> None:
    manifest = _manifest()
    observations = _observations(manifest)
    changed_source = observations[1]
    changed = create_observed_input_identity(
        input_id=changed_source.input_id,
        canonical_uri=changed_source.canonical_uri,
        version=changed_source.version,
        generation="different-generation",
        metageneration=changed_source.metageneration,
        byte_size=changed_source.byte_size + 1,
        sha256_digest=_digest("changed"),
    )
    extra = create_observed_input_identity(
        input_id="extra",
        canonical_uri="gs://research-inputs/extra.parquet",
        version="v1",
        generation="1",
        metageneration="1",
        byte_size=1,
        sha256_digest=_digest("extra"),
    )
    supplied = observations[1:]
    supplied[0] = changed
    supplied.extend([extra, observations[2]])

    verification = audit_input_manifest_observations(manifest, supplied)

    assert not verification.valid
    assert manifest.inputs[0].input_id in verification.missing_input_ids
    assert "extra" in verification.unexpected_input_ids
    assert changed.input_id in verification.mismatched_input_ids
    assert any("duplicate_input_observation" in value for value in verification.reasons)
    assert any(":generation" in value for value in verification.reasons)
    assert any(":byte_size" in value for value in verification.reasons)
    assert any(":sha256" in value for value in verification.reasons)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("canonical_uri", "https://user:secret@example.com/input"),
        ("version", ""),
        ("byte_size", -1),
        ("sha256_digest", "not-a-digest"),
    ],
)
def test_invalid_observed_identity_is_rejected(field: str, value: object) -> None:
    values: dict[str, object] = {
        "input_id": "input",
        "canonical_uri": "gs://bucket/input",
        "version": "v1",
        "generation": "1",
        "metageneration": "1",
        "byte_size": 1,
        "sha256_digest": _digest("input"),
    }
    values[field] = value
    with pytest.raises(ValueError):
        create_observed_input_identity(**values)  # type: ignore[arg-type]
