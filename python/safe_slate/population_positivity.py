"""Digest-bound design evidence for population-level logging positivity."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from hashlib import sha256
import json
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Any, Literal, Mapping


POSITIVITY_EVIDENCE_SCHEMA_VERSION = (
    "safe-slate-population-positivity-evidence-1.0"
)
STATISTICAL_POSITIVITY_EVIDENCE_SCHEMA_VERSION = (
    "safe-slate-population-positivity-evidence-2.0"
)
POSITIVITY_EVIDENCE_SCHEMA_VERSIONS = (
    POSITIVITY_EVIDENCE_SCHEMA_VERSION,
    STATISTICAL_POSITIVITY_EVIDENCE_SCHEMA_VERSION,
)
POSITIVITY_SCOPE = "target_support_almost_sure_in_registered_population"
PositivityDecision = Literal["established", "not_established"]
PositivityGuaranteeKind = Literal["deterministic", "finite_sample"]


def _nonempty(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _timestamp(value: object, field: str) -> str:
    text = _nonempty(value, field)
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError(f"{field} must be ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone")
    return text


def _parsed_timestamp(value: object, field: str) -> tuple[str, datetime]:
    text = _timestamp(value, field)
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    return text, datetime.fromisoformat(normalized)


def _sha256_hex(value: object, field: str) -> str:
    text = _nonempty(value, field)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field} must be 64 lowercase hexadecimal characters")
    return text


def _digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _validated_payload(
    *,
    schema_version: object,
    evidence_id: object,
    decision: object,
    population_identity: object,
    logging_policy_identity: object,
    target_policy_identity: object,
    support_scope: object,
    audited_at: object,
    auditor_identity: object,
    method: object,
    minimum_logging_probability_lower_bound: object,
) -> dict[str, Any]:
    """Return the canonical semantic payload or reject it fail-closed."""

    if schema_version != POSITIVITY_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("unsupported population positivity evidence schema")
    if decision not in {"established", "not_established"}:
        raise ValueError("positivity decision must be established or not_established")
    if support_scope != POSITIVITY_SCOPE:
        raise ValueError("population positivity support_scope is not canonical")
    lower_bound: float | None
    if minimum_logging_probability_lower_bound is None:
        lower_bound = None
    elif (
        not isinstance(minimum_logging_probability_lower_bound, Real)
        or isinstance(minimum_logging_probability_lower_bound, bool)
        or not isfinite(minimum_logging_probability_lower_bound)
        or not 0.0 < minimum_logging_probability_lower_bound <= 1.0
    ):
        raise ValueError(
            "minimum_logging_probability_lower_bound must be in (0,1] or null"
        )
    else:
        lower_bound = float(minimum_logging_probability_lower_bound)
    if decision == "established" and lower_bound is None:
        raise ValueError("established positivity requires a positive lower bound")
    if decision == "not_established" and lower_bound is not None:
        raise ValueError("not_established positivity must not declare a lower bound")
    return {
        "schema_version": schema_version,
        "evidence_id": _nonempty(evidence_id, "evidence_id"),
        "decision": decision,
        "population_identity": _nonempty(
            population_identity, "population_identity"
        ),
        "logging_policy_identity": _nonempty(
            logging_policy_identity, "logging_policy_identity"
        ),
        "target_policy_identity": _nonempty(
            target_policy_identity, "target_policy_identity"
        ),
        "support_scope": support_scope,
        "audited_at": _timestamp(audited_at, "audited_at"),
        "auditor_identity": _nonempty(auditor_identity, "auditor_identity"),
        "method": _nonempty(method, "method"),
        "minimum_logging_probability_lower_bound": lower_bound,
    }


@dataclass(frozen=True, slots=True)
class PopulationPositivityEvidence:
    schema_version: str
    evidence_id: str
    decision: str
    population_identity: str
    logging_policy_identity: str
    target_policy_identity: str
    support_scope: str
    audited_at: str
    auditor_identity: str
    method: str
    minimum_logging_probability_lower_bound: float | None
    evidence_digest: str

    def __post_init__(self) -> None:
        _validated_payload(**self.payload_dict())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def payload_dict(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload.pop("evidence_digest")
        return payload

    def verify_digest(self) -> bool:
        try:
            return load_population_positivity_evidence(self.to_dict()) == self
        except (TypeError, ValueError):
            return False

    @property
    def established(self) -> bool:
        return self.decision == "established"


def _validated_statistical_payload(
    *,
    schema_version: object,
    evidence_id: object,
    decision: object,
    population_identity: object,
    logging_policy_identity: object,
    target_policy_identity: object,
    support_scope: object,
    audited_at: object,
    auditor_identity: object,
    method: object,
    minimum_logging_probability_lower_bound: object,
    guarantee_kind: object,
    simultaneous_failure_probability: object,
    validity_start: object,
    validity_end: object,
    registered_epoch_count: object,
    audit_design_identity: object,
    source_certificate_sha256: object,
) -> dict[str, Any]:
    """Validate statistical/deterministic v2 evidence without weakening v1."""

    if schema_version != STATISTICAL_POSITIVITY_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("unsupported statistical positivity evidence schema")
    if guarantee_kind not in {"deterministic", "finite_sample"}:
        raise ValueError("guarantee_kind must be deterministic or finite_sample")
    if (
        not isinstance(simultaneous_failure_probability, Real)
        or isinstance(simultaneous_failure_probability, bool)
        or not isfinite(simultaneous_failure_probability)
        or not 0.0 <= simultaneous_failure_probability < 1.0
    ):
        raise ValueError("simultaneous_failure_probability must lie in [0,1)")
    failure = float(simultaneous_failure_probability)
    if guarantee_kind == "deterministic" and failure != 0.0:
        raise ValueError("deterministic evidence must have zero failure probability")
    if guarantee_kind == "finite_sample" and failure <= 0.0:
        raise ValueError("finite_sample evidence must have positive failure probability")
    if (
        isinstance(registered_epoch_count, bool)
        or not isinstance(registered_epoch_count, int)
        or registered_epoch_count < 1
    ):
        raise ValueError("registered_epoch_count must be a positive integer")
    common = _validated_payload(
        schema_version=POSITIVITY_EVIDENCE_SCHEMA_VERSION,
        evidence_id=evidence_id,
        decision=decision,
        population_identity=population_identity,
        logging_policy_identity=logging_policy_identity,
        target_policy_identity=target_policy_identity,
        support_scope=support_scope,
        audited_at=audited_at,
        auditor_identity=auditor_identity,
        method=method,
        minimum_logging_probability_lower_bound=(
            minimum_logging_probability_lower_bound
        ),
    )
    common["schema_version"] = STATISTICAL_POSITIVITY_EVIDENCE_SCHEMA_VERSION
    audited_text, audited_time = _parsed_timestamp(audited_at, "audited_at")
    start_text, start_time = _parsed_timestamp(validity_start, "validity_start")
    end_text, end_time = _parsed_timestamp(validity_end, "validity_end")
    if audited_time > start_time:
        raise ValueError("positivity audit must complete no later than validity_start")
    if start_time > end_time:
        raise ValueError("validity_start must not exceed validity_end")
    common.update(
        {
            "audited_at": audited_text,
            "guarantee_kind": guarantee_kind,
            "simultaneous_failure_probability": failure,
            "validity_start": start_text,
            "validity_end": end_text,
            "registered_epoch_count": registered_epoch_count,
            "audit_design_identity": _nonempty(
                audit_design_identity, "audit_design_identity"
            ),
            "source_certificate_sha256": _sha256_hex(
                source_certificate_sha256, "source_certificate_sha256"
            ),
        }
    )
    return common


@dataclass(frozen=True, slots=True)
class StatisticalPopulationPositivityEvidence:
    schema_version: str
    evidence_id: str
    decision: str
    population_identity: str
    logging_policy_identity: str
    target_policy_identity: str
    support_scope: str
    audited_at: str
    auditor_identity: str
    method: str
    minimum_logging_probability_lower_bound: float | None
    guarantee_kind: str
    simultaneous_failure_probability: float
    validity_start: str
    validity_end: str
    registered_epoch_count: int
    audit_design_identity: str
    source_certificate_sha256: str
    evidence_digest: str

    def __post_init__(self) -> None:
        _validated_statistical_payload(**self.payload_dict())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def payload_dict(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload.pop("evidence_digest")
        return payload

    def verify_digest(self) -> bool:
        try:
            return load_population_positivity_evidence(self.to_dict()) == self
        except (TypeError, ValueError):
            return False

    @property
    def established(self) -> bool:
        return self.decision == "established"


def create_population_positivity_evidence(
    *,
    evidence_id: str,
    decision: PositivityDecision,
    population_identity: str,
    logging_policy_identity: str,
    target_policy_identity: str,
    audited_at: str,
    auditor_identity: str,
    method: str,
    minimum_logging_probability_lower_bound: float | None,
    schema_version: str = POSITIVITY_EVIDENCE_SCHEMA_VERSION,
    support_scope: str = POSITIVITY_SCOPE,
) -> PopulationPositivityEvidence:
    """Validate and digest one external positivity audit decision."""

    payload = _validated_payload(
        schema_version=schema_version,
        evidence_id=evidence_id,
        decision=decision,
        population_identity=population_identity,
        logging_policy_identity=logging_policy_identity,
        target_policy_identity=target_policy_identity,
        support_scope=support_scope,
        audited_at=audited_at,
        auditor_identity=auditor_identity,
        method=method,
        minimum_logging_probability_lower_bound=(
            minimum_logging_probability_lower_bound
        ),
    )
    return PopulationPositivityEvidence(
        **payload,
        evidence_digest=_digest(payload),
    )


def create_statistical_population_positivity_evidence(
    *,
    evidence_id: str,
    decision: PositivityDecision,
    population_identity: str,
    logging_policy_identity: str,
    target_policy_identity: str,
    audited_at: str,
    auditor_identity: str,
    method: str,
    minimum_logging_probability_lower_bound: float | None,
    guarantee_kind: PositivityGuaranteeKind,
    simultaneous_failure_probability: float,
    validity_start: str,
    validity_end: str,
    registered_epoch_count: int,
    audit_design_identity: str,
    source_certificate_sha256: str,
    schema_version: str = STATISTICAL_POSITIVITY_EVIDENCE_SCHEMA_VERSION,
    support_scope: str = POSITIVITY_SCOPE,
) -> StatisticalPopulationPositivityEvidence:
    """Create digest-bound v2 evidence retaining confidence and validity."""

    payload = _validated_statistical_payload(
        schema_version=schema_version,
        evidence_id=evidence_id,
        decision=decision,
        population_identity=population_identity,
        logging_policy_identity=logging_policy_identity,
        target_policy_identity=target_policy_identity,
        support_scope=support_scope,
        audited_at=audited_at,
        auditor_identity=auditor_identity,
        method=method,
        minimum_logging_probability_lower_bound=(
            minimum_logging_probability_lower_bound
        ),
        guarantee_kind=guarantee_kind,
        simultaneous_failure_probability=simultaneous_failure_probability,
        validity_start=validity_start,
        validity_end=validity_end,
        registered_epoch_count=registered_epoch_count,
        audit_design_identity=audit_design_identity,
        source_certificate_sha256=source_certificate_sha256,
    )
    return StatisticalPopulationPositivityEvidence(
        **payload,
        evidence_digest=_digest(payload),
    )


def load_population_positivity_evidence(
    payload: Mapping[str, Any],
) -> PopulationPositivityEvidence | StatisticalPopulationPositivityEvidence:
    """Load and semantically revalidate serialized positivity evidence."""

    if not isinstance(payload, Mapping):
        raise ValueError("population positivity evidence must be an object")
    if payload.get("schema_version") == STATISTICAL_POSITIVITY_EVIDENCE_SCHEMA_VERSION:
        expected = set(StatisticalPopulationPositivityEvidence.__dataclass_fields__)
        if set(payload) != expected:
            raise ValueError("statistical population positivity evidence fields differ")
        recreated = create_statistical_population_positivity_evidence(
            schema_version=payload["schema_version"],
            evidence_id=payload["evidence_id"],
            decision=payload["decision"],
            population_identity=payload["population_identity"],
            logging_policy_identity=payload["logging_policy_identity"],
            target_policy_identity=payload["target_policy_identity"],
            support_scope=payload["support_scope"],
            audited_at=payload["audited_at"],
            auditor_identity=payload["auditor_identity"],
            method=payload["method"],
            minimum_logging_probability_lower_bound=(
                payload["minimum_logging_probability_lower_bound"]
            ),
            guarantee_kind=payload["guarantee_kind"],
            simultaneous_failure_probability=(
                payload["simultaneous_failure_probability"]
            ),
            validity_start=payload["validity_start"],
            validity_end=payload["validity_end"],
            registered_epoch_count=payload["registered_epoch_count"],
            audit_design_identity=payload["audit_design_identity"],
            source_certificate_sha256=payload["source_certificate_sha256"],
        )
        if recreated.evidence_digest != payload["evidence_digest"]:
            raise ValueError("population positivity evidence digest mismatch")
        return recreated
    expected = set(PopulationPositivityEvidence.__dataclass_fields__)
    if set(payload) != expected:
        raise ValueError("population positivity evidence fields differ")
    recreated = create_population_positivity_evidence(
        schema_version=payload["schema_version"],
        evidence_id=payload["evidence_id"],
        decision=payload["decision"],
        population_identity=payload["population_identity"],
        logging_policy_identity=payload["logging_policy_identity"],
        target_policy_identity=payload["target_policy_identity"],
        support_scope=payload["support_scope"],
        audited_at=payload["audited_at"],
        auditor_identity=payload["auditor_identity"],
        method=payload["method"],
        minimum_logging_probability_lower_bound=(
            payload["minimum_logging_probability_lower_bound"]
        ),
    )
    if recreated.evidence_digest != payload["evidence_digest"]:
        raise ValueError("population positivity evidence digest mismatch")
    return recreated


def canonical_population_positivity_evidence_bytes(
    evidence: PopulationPositivityEvidence | StatisticalPopulationPositivityEvidence,
) -> bytes:
    """Return the only accepted on-disk encoding for positivity evidence."""

    if (
        not isinstance(
            evidence,
            (PopulationPositivityEvidence, StatisticalPopulationPositivityEvidence),
        )
        or not evidence.verify_digest()
    ):
        raise ValueError("population positivity evidence is invalid or tampered")
    return (
        json.dumps(
            evidence.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def load_population_positivity_evidence_file(
    path: str | Path,
) -> PopulationPositivityEvidence | StatisticalPopulationPositivityEvidence:
    """Load evidence and reject noncanonical or duplicate-key JSON bytes."""

    raw = Path(path).read_bytes()

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate population positivity key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite population positivity value: {value}")

    payload = json.loads(
        raw,
        object_pairs_hook=reject_duplicates,
        parse_constant=reject_constant,
    )
    evidence = load_population_positivity_evidence(payload)
    if raw != canonical_population_positivity_evidence_bytes(evidence):
        raise ValueError("population positivity evidence encoding is not canonical")
    return evidence
