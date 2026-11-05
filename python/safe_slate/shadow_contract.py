"""Immutable contracts for read-only shadow operability evaluations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from hashlib import sha256
import json
from math import isfinite
from numbers import Integral, Real
from typing import Any, Mapping, Sequence


def _nonempty(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _aware_time(value: str, field: str) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed


def _positive_real(value: object, field: str) -> float:
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not isfinite(value)
        or value <= 0.0
    ):
        raise ValueError(f"{field} must be finite and positive")
    return float(value)


def _rate(value: object, field: str) -> float:
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError(f"{field} must be finite and in [0, 1]")
    return float(value)


def _positive_integer(value: object, field: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return int(value)


def _sha256(value: str, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class ShadowEvaluationContract:
    contract_version: str
    evaluation_id: str
    scientific_question: str
    population_identity: str
    window_start: str
    window_end: str
    expected_log_schema: str
    input_manifest_digest: str
    code_revision: str
    dependency_lock_digest: str
    output_namespace: str
    minimum_requests: int
    maximum_oracle_p95_ms: float
    maximum_fallback_rate: float
    maximum_invalid_output_rate: float
    maximum_recheck_failure_rate: float
    maximum_predicate_error_request_rate: float
    require_service_latency_trace: bool
    maximum_service_response_p95_ms: float | None
    maximum_log_commit_p95_ms: float | None
    maximum_runtime_seconds: float
    maximum_memory_mb: int
    maximum_output_bytes: int
    authorized_scope: tuple[str, ...]
    explicitly_unauthorized: tuple[str, ...]
    stage_order: tuple[str, ...]
    decision_vocabulary: tuple[str, ...]
    contract_digest: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def payload_dict(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload.pop("contract_digest")
        return payload

    def verify_digest(self) -> bool:
        return self.contract_digest == _digest(self.payload_dict())


def create_shadow_evaluation_contract(
    *,
    evaluation_id: str,
    scientific_question: str,
    population_identity: str,
    window_start: str,
    window_end: str,
    input_manifest_digest: str,
    code_revision: str,
    dependency_lock_digest: str,
    output_namespace: str,
    minimum_requests: int,
    maximum_oracle_p95_ms: float,
    maximum_fallback_rate: float,
    maximum_invalid_output_rate: float,
    maximum_recheck_failure_rate: float,
    maximum_predicate_error_request_rate: float,
    require_service_latency_trace: bool = False,
    maximum_service_response_p95_ms: float | None = None,
    maximum_log_commit_p95_ms: float | None = None,
    maximum_runtime_seconds: float,
    maximum_memory_mb: int,
    maximum_output_bytes: int,
    expected_log_schema: str = "1.3",
    authorized_scope: Sequence[str] = ("offline_shadow_operability_reporting",),
    explicitly_unauthorized: Sequence[str] = (
        "production_rollout",
        "production_latency_claim",
        "causal_purchase_lift_claim",
        "serving_change",
        "online_experiment",
        "overwrite_existing_output",
    ),
    contract_version: str = "1.0.0",
) -> ShadowEvaluationContract:
    """Validate and freeze a shadow-only operability contract."""

    start = _aware_time(window_start, "window_start")
    end = _aware_time(window_end, "window_end")
    if start >= end:
        raise ValueError("shadow window must be non-empty and ordered")
    if expected_log_schema != "1.3":
        raise ValueError("expected_log_schema must match proposal-log schema 1.3")
    if not isinstance(require_service_latency_trace, bool):
        raise ValueError("require_service_latency_trace must be boolean")
    if require_service_latency_trace and (
        maximum_service_response_p95_ms is None or maximum_log_commit_p95_ms is None
    ):
        raise ValueError("required service traces need response and log p95 gates")
    if not require_service_latency_trace and (
        maximum_service_response_p95_ms is not None
        or maximum_log_commit_p95_ms is not None
    ):
        raise ValueError("service p95 gates require service latency traces")
    authorized = tuple(
        _nonempty(value, "authorized_scope") for value in authorized_scope
    )
    unauthorized = tuple(
        _nonempty(value, "explicitly_unauthorized") for value in explicitly_unauthorized
    )
    if not authorized or len(set(authorized)) != len(authorized):
        raise ValueError("authorized_scope must be non-empty and unique")
    if not unauthorized or len(set(unauthorized)) != len(unauthorized):
        raise ValueError("explicitly_unauthorized must be non-empty and unique")
    if set(authorized) & set(unauthorized):
        raise ValueError("authorized and unauthorized scopes may not overlap")

    payload: dict[str, Any] = {
        "contract_version": _nonempty(contract_version, "contract_version"),
        "evaluation_id": _nonempty(evaluation_id, "evaluation_id"),
        "scientific_question": _nonempty(scientific_question, "scientific_question"),
        "population_identity": _nonempty(population_identity, "population_identity"),
        "window_start": window_start,
        "window_end": window_end,
        "expected_log_schema": _nonempty(expected_log_schema, "expected_log_schema"),
        "input_manifest_digest": _sha256(
            input_manifest_digest, "input_manifest_digest"
        ),
        "code_revision": _nonempty(code_revision, "code_revision"),
        "dependency_lock_digest": _sha256(
            dependency_lock_digest, "dependency_lock_digest"
        ),
        "output_namespace": _nonempty(output_namespace, "output_namespace"),
        "minimum_requests": _positive_integer(minimum_requests, "minimum_requests"),
        "maximum_oracle_p95_ms": _positive_real(
            maximum_oracle_p95_ms, "maximum_oracle_p95_ms"
        ),
        "maximum_fallback_rate": _rate(maximum_fallback_rate, "maximum_fallback_rate"),
        "maximum_invalid_output_rate": _rate(
            maximum_invalid_output_rate, "maximum_invalid_output_rate"
        ),
        "maximum_recheck_failure_rate": _rate(
            maximum_recheck_failure_rate, "maximum_recheck_failure_rate"
        ),
        "maximum_predicate_error_request_rate": _rate(
            maximum_predicate_error_request_rate,
            "maximum_predicate_error_request_rate",
        ),
        "require_service_latency_trace": require_service_latency_trace,
        "maximum_service_response_p95_ms": (
            _positive_real(
                maximum_service_response_p95_ms,
                "maximum_service_response_p95_ms",
            )
            if maximum_service_response_p95_ms is not None
            else None
        ),
        "maximum_log_commit_p95_ms": (
            _positive_real(
                maximum_log_commit_p95_ms,
                "maximum_log_commit_p95_ms",
            )
            if maximum_log_commit_p95_ms is not None
            else None
        ),
        "maximum_runtime_seconds": _positive_real(
            maximum_runtime_seconds, "maximum_runtime_seconds"
        ),
        "maximum_memory_mb": _positive_integer(maximum_memory_mb, "maximum_memory_mb"),
        "maximum_output_bytes": _positive_integer(
            maximum_output_bytes, "maximum_output_bytes"
        ),
        "authorized_scope": authorized,
        "explicitly_unauthorized": unauthorized,
        "stage_order": (
            "input_observation",
            "shadow_log_audit",
            "operability_summary",
            "atomic_publication",
            "independent_final_audit",
        ),
        "decision_vocabulary": (
            "insufficient_evidence",
            "evidence_rejected",
            "artifact_complete",
            "evidence_valid",
        ),
    }
    return ShadowEvaluationContract(**payload, contract_digest=_digest(payload))


def load_shadow_evaluation_contract(
    payload: Mapping[str, Any],
) -> ShadowEvaluationContract:
    """Load and semantically revalidate a serialized shadow contract."""

    if not isinstance(payload, Mapping):
        raise ValueError("shadow contract payload must be a mapping")
    if set(payload) != set(ShadowEvaluationContract.__dataclass_fields__):
        raise ValueError("shadow contract fields do not match canonical schema")
    contract = create_shadow_evaluation_contract(
        evaluation_id=payload["evaluation_id"],
        scientific_question=payload["scientific_question"],
        population_identity=payload["population_identity"],
        window_start=payload["window_start"],
        window_end=payload["window_end"],
        expected_log_schema=payload["expected_log_schema"],
        input_manifest_digest=payload["input_manifest_digest"],
        code_revision=payload["code_revision"],
        dependency_lock_digest=payload["dependency_lock_digest"],
        output_namespace=payload["output_namespace"],
        minimum_requests=payload["minimum_requests"],
        maximum_oracle_p95_ms=payload["maximum_oracle_p95_ms"],
        maximum_fallback_rate=payload["maximum_fallback_rate"],
        maximum_invalid_output_rate=payload["maximum_invalid_output_rate"],
        maximum_recheck_failure_rate=payload["maximum_recheck_failure_rate"],
        maximum_predicate_error_request_rate=payload[
            "maximum_predicate_error_request_rate"
        ],
        require_service_latency_trace=payload["require_service_latency_trace"],
        maximum_service_response_p95_ms=payload["maximum_service_response_p95_ms"],
        maximum_log_commit_p95_ms=payload["maximum_log_commit_p95_ms"],
        maximum_runtime_seconds=payload["maximum_runtime_seconds"],
        maximum_memory_mb=payload["maximum_memory_mb"],
        maximum_output_bytes=payload["maximum_output_bytes"],
        authorized_scope=tuple(payload["authorized_scope"]),
        explicitly_unauthorized=tuple(payload["explicitly_unauthorized"]),
        contract_version=payload["contract_version"],
    )
    if contract.contract_digest != payload["contract_digest"]:
        raise ValueError("shadow contract digest mismatch")
    if (
        tuple(payload["stage_order"]) != contract.stage_order
        or tuple(payload["decision_vocabulary"]) != contract.decision_vocabulary
    ):
        raise ValueError("shadow contract protocol fields are noncanonical")
    return contract
