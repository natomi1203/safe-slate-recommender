from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

from safe_slate import (
    POSITIVITY_EVIDENCE_SCHEMA_VERSION,
    POSITIVITY_SCOPE,
    STATISTICAL_POSITIVITY_EVIDENCE_SCHEMA_VERSION,
    PopulationPositivityEvidence,
    StatisticalPopulationPositivityEvidence,
    canonical_population_positivity_evidence_bytes,
    create_population_positivity_evidence,
    create_statistical_population_positivity_evidence,
    load_population_positivity_evidence,
    load_population_positivity_evidence_file,
)


def _evidence():
    return create_population_positivity_evidence(
        evidence_id="positivity-v1",
        decision="established",
        population_identity="population-v1",
        logging_policy_identity="logger-v1",
        target_policy_identity="target-v1",
        audited_at="2026-10-01T00:00:00+09:00",
        auditor_identity="independent-auditor",
        method="registered logger lower-envelope audit",
        minimum_logging_probability_lower_bound=0.01,
    )


def _statistical_evidence(**overrides):
    values = {
        "evidence_id": "positivity-statistical-v2",
        "decision": "established",
        "population_identity": "population-v1",
        "logging_policy_identity": "logger-v1",
        "target_policy_identity": "target-v1",
        "audited_at": "2026-09-30T00:00:00+09:00",
        "auditor_identity": "independent-auditor",
        "method": "epochwise metric-net exact lower audit",
        "minimum_logging_probability_lower_bound": 0.01,
        "guarantee_kind": "finite_sample",
        "simultaneous_failure_probability": 0.025,
        "validity_start": "2026-10-01T00:00:00+09:00",
        "validity_end": "2026-10-31T23:59:59+09:00",
        "registered_epoch_count": 3,
        "audit_design_identity": "metric-net-design-v4",
        "source_certificate_sha256": "c" * 64,
    }
    values.update(overrides)
    return create_statistical_population_positivity_evidence(**values)


def test_population_positivity_round_trip_is_canonical(tmp_path: Path) -> None:
    evidence = _evidence()
    path = tmp_path / "positivity.json"
    path.write_bytes(canonical_population_positivity_evidence_bytes(evidence))

    assert load_population_positivity_evidence(evidence.to_dict()) == evidence
    assert load_population_positivity_evidence_file(path) == evidence


def test_legacy_v1_canonical_identity_remains_immutable() -> None:
    raw = canonical_population_positivity_evidence_bytes(_evidence())
    assert len(raw) == 569
    assert sha256(raw).hexdigest() == (
        "e49bcccadbda59e7fed1f4587ef0b981483b89163b141de494240d99ae6ef194"
    )


def test_statistical_positivity_round_trip_retains_failure_and_validity(
    tmp_path: Path,
) -> None:
    evidence = _statistical_evidence()
    path = tmp_path / "statistical-positivity.json"
    path.write_bytes(canonical_population_positivity_evidence_bytes(evidence))

    loaded = load_population_positivity_evidence_file(path)
    assert isinstance(loaded, StatisticalPopulationPositivityEvidence)
    assert loaded.schema_version == STATISTICAL_POSITIVITY_EVIDENCE_SCHEMA_VERSION
    assert loaded.simultaneous_failure_probability == pytest.approx(0.025)
    assert loaded.validity_start == "2026-10-01T00:00:00+09:00"
    assert loaded.validity_end == "2026-10-31T23:59:59+09:00"
    assert loaded.registered_epoch_count == 3
    assert loaded.source_certificate_sha256 == "c" * 64


@pytest.mark.parametrize(
    "changes",
    (
        {"guarantee_kind": "finite_sample", "simultaneous_failure_probability": 0.0},
        {"guarantee_kind": "deterministic", "simultaneous_failure_probability": 0.01},
        {"validity_start": "2026-09-29T00:00:00+09:00"},
        {"validity_end": "2026-09-30T12:00:00+09:00"},
        {"registered_epoch_count": 0},
        {"source_certificate_sha256": "C" * 64},
    ),
)
def test_statistical_positivity_rejects_invalid_confidence_contract(
    changes: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        _statistical_evidence(**changes)


def test_population_positivity_rejects_tampering_and_noncanonical_json(
    tmp_path: Path,
) -> None:
    evidence = _evidence()
    with pytest.raises(ValueError, match="invalid or tampered"):
        canonical_population_positivity_evidence_bytes(
            replace(evidence, target_policy_identity="other-target")
        )

    path = tmp_path / "pretty.json"
    path.write_text(json.dumps(evidence.to_dict(), indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="not canonical"):
        load_population_positivity_evidence_file(path)


@pytest.mark.parametrize(
    "decision,lower_bound",
    (("established", None), ("not_established", 0.01)),
)
def test_population_positivity_decision_and_lower_bound_are_consistent(
    decision: str,
    lower_bound: float | None,
) -> None:
    with pytest.raises(ValueError):
        create_population_positivity_evidence(
            evidence_id="positivity-v1",
            decision=decision,  # type: ignore[arg-type]
            population_identity="population-v1",
            logging_policy_identity="logger-v1",
            target_policy_identity="target-v1",
            audited_at="2026-10-01T00:00:00+09:00",
            auditor_identity="independent-auditor",
            method="registered audit",
            minimum_logging_probability_lower_bound=lower_bound,
        )


@pytest.mark.parametrize(
    "changes",
    (
        {"schema_version": "unknown-schema"},
        {"support_scope": "finite_logged_rows_only"},
        {"decision": "unknown"},
        {
            "decision": "established",
            "minimum_logging_probability_lower_bound": None,
        },
        {
            "decision": "not_established",
            "minimum_logging_probability_lower_bound": 0.01,
        },
    ),
)
def test_direct_typed_constructor_rejects_semantically_invalid_rehashed_objects(
    changes: dict[str, object],
) -> None:
    payload = {
        "schema_version": POSITIVITY_EVIDENCE_SCHEMA_VERSION,
        "evidence_id": "positivity-v1",
        "decision": "established",
        "population_identity": "population-v1",
        "logging_policy_identity": "logger-v1",
        "target_policy_identity": "target-v1",
        "support_scope": POSITIVITY_SCOPE,
        "audited_at": "2026-10-01T00:00:00+09:00",
        "auditor_identity": "independent-auditor",
        "method": "registered audit",
        "minimum_logging_probability_lower_bound": 0.01,
    }
    payload.update(changes)
    evidence_digest = sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()

    with pytest.raises(ValueError):
        PopulationPositivityEvidence(
            **payload,  # type: ignore[arg-type]
            evidence_digest=evidence_digest,
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"evidence_id": ""}, "non-empty string"),
        ({"audited_at": "not-a-time"}, "ISO-8601"),
        ({"audited_at": "2026-10-01T00:00:00"}, "timezone"),
        (
            {"minimum_logging_probability_lower_bound": True},
            "must be in",
        ),
        (
            {"minimum_logging_probability_lower_bound": 0.0},
            "must be in",
        ),
    ],
)
def test_legacy_positivity_rejects_remaining_malformed_values(
    changes: dict[str, object], message: str
) -> None:
    values = {
        "evidence_id": "positivity-v1",
        "decision": "established",
        "population_identity": "population-v1",
        "logging_policy_identity": "logger-v1",
        "target_policy_identity": "target-v1",
        "audited_at": "2026-10-01T00:00:00+09:00",
        "auditor_identity": "auditor",
        "method": "audit",
        "minimum_logging_probability_lower_bound": 0.01,
    }
    values.update(changes)
    with pytest.raises(ValueError, match=message):
        create_population_positivity_evidence(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"schema_version": "wrong"}, "unsupported statistical"),
        ({"guarantee_kind": "asymptotic"}, "guarantee_kind"),
        ({"simultaneous_failure_probability": True}, "must lie"),
        ({"simultaneous_failure_probability": 1.0}, "must lie"),
    ],
)
def test_statistical_positivity_rejects_remaining_malformed_values(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _statistical_evidence(**changes)


def test_positivity_loaders_reject_type_fields_and_digest_tampering() -> None:
    with pytest.raises(ValueError, match="must be an object"):
        load_population_positivity_evidence([])  # type: ignore[arg-type]

    legacy = _evidence()
    missing_legacy = legacy.to_dict()
    missing_legacy.pop("evidence_id")
    with pytest.raises(ValueError, match="fields differ"):
        load_population_positivity_evidence(missing_legacy)
    bad_legacy_digest = legacy.to_dict()
    bad_legacy_digest["evidence_digest"] = "0" * 64
    with pytest.raises(ValueError, match="digest mismatch"):
        load_population_positivity_evidence(bad_legacy_digest)

    statistical = _statistical_evidence()
    missing_statistical = statistical.to_dict()
    missing_statistical.pop("guarantee_kind")
    with pytest.raises(ValueError, match="fields differ"):
        load_population_positivity_evidence(missing_statistical)
    bad_statistical_digest = statistical.to_dict()
    bad_statistical_digest["evidence_digest"] = "0" * 64
    with pytest.raises(ValueError, match="digest mismatch"):
        load_population_positivity_evidence(bad_statistical_digest)
    assert not replace(statistical, evidence_digest="0" * 64).verify_digest()


def test_positivity_file_loader_rejects_duplicate_and_nonfinite_json(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":"x","schema_version":"y"}')
    with pytest.raises(ValueError, match="duplicate population positivity key"):
        load_population_positivity_evidence_file(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value":NaN}')
    with pytest.raises(ValueError, match="non-finite population positivity"):
        load_population_positivity_evidence_file(nonfinite)


def test_not_established_properties_are_false_for_both_schemas() -> None:
    legacy = create_population_positivity_evidence(
        evidence_id="negative-v1",
        decision="not_established",
        population_identity="population-v1",
        logging_policy_identity="logger-v1",
        target_policy_identity="target-v1",
        audited_at="2026-09-30T00:00:00+09:00",
        auditor_identity="auditor",
        method="audit",
        minimum_logging_probability_lower_bound=None,
    )
    statistical = _statistical_evidence(
        decision="not_established",
        minimum_logging_probability_lower_bound=None,
    )
    assert not legacy.established
    assert not statistical.established
