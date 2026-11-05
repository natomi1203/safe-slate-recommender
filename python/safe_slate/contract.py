"""Immutable contracts for consequential offline OPE evaluations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from hashlib import sha256
import json
from math import isfinite
from numbers import Integral, Real
from typing import Any, Literal, Mapping, Sequence


PropensityMode = Literal["known", "interval"]


def _parse_aware_timestamp(value: str, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty ISO-8601 timestamp")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed


def _nonempty(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _finite_rate(value: object, field: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool) or not isfinite(value):
        raise ValueError(f"{field} must be finite and in [0, 1]")
    normalized = float(value)
    if not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{field} must be finite and in [0, 1]")
    return normalized


def _finite_positive(value: object, field: str) -> float:
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not isfinite(value)
        or value <= 0.0
    ):
        raise ValueError(f"{field} must be finite and positive")
    return float(value)


def _nonnegative_integer(value: object, field: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return int(value)


def _positive_integer(value: object, field: str) -> int:
    result = _nonnegative_integer(value, field)
    if result == 0:
        raise ValueError(f"{field} must be positive")
    return result


@dataclass(frozen=True, slots=True)
class OpeEvaluationContract:
    """Frozen scientific, data, inference, execution, and permission boundary."""

    contract_version: str
    evaluation_id: str
    scientific_question: str
    baseline_policy: str
    target_policy: str
    population_identity: str
    pilot_window_start: str
    pilot_window_end: str
    final_window_start: str
    final_window_end: str
    earliest_analysis_time: str
    label_settlement_rule: str
    primary_metric: str
    primary_contrast: str
    analysis_unit: str
    propensity_mode: Literal["known", "interval", "missing"]
    input_manifest_digest: str
    code_revision: str
    dependency_lock_digest: str
    output_namespace: str
    minimum_support_rate_for_full_estimand: float
    minimum_support_rate_for_partial_estimand: float
    minimum_user_effective_sample_size: float
    minimum_purchase_events: int
    maximum_missing_outcome_rate: float
    maximum_p99_importance_weight: float
    weight_clip: float
    bootstrap_repetitions: int
    bootstrap_seed: int
    confidence_level: float
    outcome_lower_bound: float
    outcome_upper_bound: float
    minimum_raw_requests: int
    maximum_runtime_seconds: float
    maximum_memory_mb: int
    maximum_output_bytes: int
    authorized_scope: tuple[str, ...]
    explicitly_unauthorized: tuple[str, ...]
    pilot_allowed_uses: tuple[str, ...]
    pilot_forbidden_uses: tuple[str, ...]
    stage_order: tuple[str, ...]
    decision_vocabulary: tuple[str, ...]
    null_negative_and_missing_policy: str
    retry_policy: str
    contract_digest: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def payload_dict(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload.pop("contract_digest")
        return payload

    def verify_digest(self) -> bool:
        return self.contract_digest == _contract_digest(self.payload_dict())


@dataclass(frozen=True, slots=True)
class EvaluationEvidenceStatus:
    """Independent collection, label, sufficiency, and decision dimensions."""

    contract_digest: str
    collection_state: str
    label_state: str
    statistical_state: str
    decision_state: str
    overall_state: str
    collection_complete: bool
    analysis_ready: bool
    statistically_sufficient: bool
    decision_eligible: bool
    authorized_actions: tuple[str, ...]
    explicitly_unauthorized: tuple[str, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _contract_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def create_ope_evaluation_contract(
    *,
    evaluation_id: str,
    scientific_question: str,
    baseline_policy: str,
    target_policy: str,
    population_identity: str,
    pilot_window_start: str,
    pilot_window_end: str,
    final_window_start: str,
    final_window_end: str,
    earliest_analysis_time: str,
    label_settlement_rule: str,
    primary_metric: str,
    primary_contrast: str,
    propensity_mode: PropensityMode,
    input_manifest_digest: str,
    code_revision: str,
    dependency_lock_digest: str,
    output_namespace: str,
    minimum_support_rate_for_full_estimand: float,
    minimum_support_rate_for_partial_estimand: float,
    minimum_user_effective_sample_size: float,
    minimum_purchase_events: int,
    maximum_missing_outcome_rate: float,
    maximum_p99_importance_weight: float,
    weight_clip: float,
    bootstrap_repetitions: int,
    bootstrap_seed: int,
    confidence_level: float,
    outcome_lower_bound: float,
    outcome_upper_bound: float,
    minimum_raw_requests: int,
    maximum_runtime_seconds: float,
    maximum_memory_mb: int,
    maximum_output_bytes: int,
    authorized_scope: Sequence[str] = ("offline_research_reporting",),
    explicitly_unauthorized: Sequence[str] = (
        "production_rollout",
        "serving_change",
        "online_experiment",
        "post_label_threshold_change",
        "overwrite_existing_output",
    ),
    pilot_allowed_uses: Sequence[str] = (
        "support_and_propensity_diagnostics",
        "variance_and_event_rate_estimation",
        "weight_clipping_selection",
        "final_sample_size_planning",
    ),
    pilot_forbidden_uses: Sequence[str] = (
        "select_final_winner",
        "change_primary_metric",
        "select_favorable_subgroup",
        "report_pilot_lift_as_confirmatory",
    ),
    contract_version: str = "1.0.0",
) -> OpeEvaluationContract:
    """Validate and freeze a no-clobber OPE evaluation contract."""

    strings = {
        "contract_version": contract_version,
        "evaluation_id": evaluation_id,
        "scientific_question": scientific_question,
        "baseline_policy": baseline_policy,
        "target_policy": target_policy,
        "population_identity": population_identity,
        "label_settlement_rule": label_settlement_rule,
        "primary_metric": primary_metric,
        "primary_contrast": primary_contrast,
        "input_manifest_digest": input_manifest_digest,
        "code_revision": code_revision,
        "dependency_lock_digest": dependency_lock_digest,
        "output_namespace": output_namespace,
    }
    normalized_strings = {
        field: _nonempty(value, field) for field, value in strings.items()
    }
    if baseline_policy == target_policy:
        raise ValueError("baseline_policy and target_policy must differ")
    if propensity_mode not in {"known", "interval"}:
        raise ValueError("propensity_mode must be known or interval")
    for field in ("input_manifest_digest", "dependency_lock_digest"):
        value = normalized_strings[field]
        if len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(f"{field} must be a lowercase SHA-256 digest")

    pilot_start = _parse_aware_timestamp(pilot_window_start, "pilot_window_start")
    pilot_end = _parse_aware_timestamp(pilot_window_end, "pilot_window_end")
    final_start = _parse_aware_timestamp(final_window_start, "final_window_start")
    final_end = _parse_aware_timestamp(final_window_end, "final_window_end")
    analysis_time = _parse_aware_timestamp(
        earliest_analysis_time, "earliest_analysis_time"
    )
    if not pilot_start < pilot_end <= final_start < final_end:
        raise ValueError("pilot and final windows must be ordered and non-overlapping")
    if analysis_time < final_end:
        raise ValueError("earliest_analysis_time may not precede final_window_end")

    full_support = _finite_rate(
        minimum_support_rate_for_full_estimand,
        "minimum_support_rate_for_full_estimand",
    )
    partial_support = _finite_rate(
        minimum_support_rate_for_partial_estimand,
        "minimum_support_rate_for_partial_estimand",
    )
    if partial_support >= full_support:
        raise ValueError("partial support threshold must be below full threshold")
    missing_rate = _finite_rate(
        maximum_missing_outcome_rate, "maximum_missing_outcome_rate"
    )
    confidence = _finite_rate(confidence_level, "confidence_level")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    lower = float(outcome_lower_bound)
    upper = float(outcome_upper_bound)
    if not isfinite(lower) or not isfinite(upper) or lower >= upper:
        raise ValueError("outcome bounds must be finite and strictly ordered")

    def normalized_items(values: Sequence[str], field: str) -> tuple[str, ...]:
        result = tuple(_nonempty(value, field) for value in values)
        if not result or len(set(result)) != len(result):
            raise ValueError(f"{field} must be non-empty and unique")
        return result

    authorized = normalized_items(authorized_scope, "authorized_scope")
    unauthorized = normalized_items(explicitly_unauthorized, "explicitly_unauthorized")
    if set(authorized) & set(unauthorized):
        raise ValueError("authorized and unauthorized scopes may not overlap")
    pilot_allowed = normalized_items(pilot_allowed_uses, "pilot_allowed_uses")
    pilot_forbidden = normalized_items(pilot_forbidden_uses, "pilot_forbidden_uses")
    if set(pilot_allowed) & set(pilot_forbidden):
        raise ValueError("pilot allowed and forbidden uses may not overlap")

    payload: dict[str, Any] = {
        **normalized_strings,
        "pilot_window_start": pilot_window_start,
        "pilot_window_end": pilot_window_end,
        "final_window_start": final_window_start,
        "final_window_end": final_window_end,
        "earliest_analysis_time": earliest_analysis_time,
        "analysis_unit": "pseudonymous_user",
        "propensity_mode": propensity_mode,
        "minimum_support_rate_for_full_estimand": full_support,
        "minimum_support_rate_for_partial_estimand": partial_support,
        "minimum_user_effective_sample_size": _finite_positive(
            minimum_user_effective_sample_size,
            "minimum_user_effective_sample_size",
        ),
        "minimum_purchase_events": _nonnegative_integer(
            minimum_purchase_events, "minimum_purchase_events"
        ),
        "maximum_missing_outcome_rate": missing_rate,
        "maximum_p99_importance_weight": _finite_positive(
            maximum_p99_importance_weight,
            "maximum_p99_importance_weight",
        ),
        "weight_clip": _finite_positive(weight_clip, "weight_clip"),
        "bootstrap_repetitions": _positive_integer(
            bootstrap_repetitions, "bootstrap_repetitions"
        ),
        "bootstrap_seed": _nonnegative_integer(bootstrap_seed, "bootstrap_seed"),
        "confidence_level": confidence,
        "outcome_lower_bound": lower,
        "outcome_upper_bound": upper,
        "minimum_raw_requests": _positive_integer(
            minimum_raw_requests, "minimum_raw_requests"
        ),
        "maximum_runtime_seconds": _finite_positive(
            maximum_runtime_seconds, "maximum_runtime_seconds"
        ),
        "maximum_memory_mb": _positive_integer(maximum_memory_mb, "maximum_memory_mb"),
        "maximum_output_bytes": _positive_integer(
            maximum_output_bytes, "maximum_output_bytes"
        ),
        "authorized_scope": authorized,
        "explicitly_unauthorized": unauthorized,
        "pilot_allowed_uses": pilot_allowed,
        "pilot_forbidden_uses": pilot_forbidden,
        "stage_order": (
            "capture",
            "lock",
            "label_capture",
            "readiness_audit",
            "estimation",
            "independent_final_audit",
        ),
        "decision_vocabulary": (
            "not_started",
            "operational_non_run",
            "failed_non_evidence",
            "partial_invalid",
            "artifact_complete",
            "labels_pending",
            "analysis_ready",
            "insufficient_evidence",
            "evidence_rejected",
            "evidence_valid",
            "decision_eligible",
            "complete",
        ),
        "null_negative_and_missing_policy": (
            "Preserve null, negative, unsupported, insufficient, and missing "
            "results without substituting another estimand."
        ),
        "retry_policy": (
            "No overwrite or retry under the same evaluation and output identity."
        ),
    }
    digest = _contract_digest(payload)
    return OpeEvaluationContract(**payload, contract_digest=digest)


def load_ope_evaluation_contract(payload: Mapping[str, Any]) -> OpeEvaluationContract:
    """Load a serialized contract and reject digest or schema tampering."""

    if not isinstance(payload, Mapping):
        raise ValueError("contract payload must be a mapping")
    expected_fields = set(OpeEvaluationContract.__dataclass_fields__)
    missing = expected_fields - set(payload)
    extra = set(payload) - expected_fields
    if missing or extra:
        raise ValueError(
            "contract fields mismatch; missing="
            + ",".join(sorted(missing))
            + "; extra="
            + ",".join(sorted(extra))
        )
    normalized = dict(payload)
    for field in (
        "authorized_scope",
        "explicitly_unauthorized",
        "pilot_allowed_uses",
        "pilot_forbidden_uses",
        "stage_order",
        "decision_vocabulary",
    ):
        normalized[field] = tuple(normalized[field])
    contract = OpeEvaluationContract(**normalized)
    if not contract.verify_digest():
        raise ValueError("contract digest mismatch")
    recreated = create_ope_evaluation_contract(
        evaluation_id=contract.evaluation_id,
        scientific_question=contract.scientific_question,
        baseline_policy=contract.baseline_policy,
        target_policy=contract.target_policy,
        population_identity=contract.population_identity,
        pilot_window_start=contract.pilot_window_start,
        pilot_window_end=contract.pilot_window_end,
        final_window_start=contract.final_window_start,
        final_window_end=contract.final_window_end,
        earliest_analysis_time=contract.earliest_analysis_time,
        label_settlement_rule=contract.label_settlement_rule,
        primary_metric=contract.primary_metric,
        primary_contrast=contract.primary_contrast,
        propensity_mode=contract.propensity_mode,  # type: ignore[arg-type]
        input_manifest_digest=contract.input_manifest_digest,
        code_revision=contract.code_revision,
        dependency_lock_digest=contract.dependency_lock_digest,
        output_namespace=contract.output_namespace,
        minimum_support_rate_for_full_estimand=(
            contract.minimum_support_rate_for_full_estimand
        ),
        minimum_support_rate_for_partial_estimand=(
            contract.minimum_support_rate_for_partial_estimand
        ),
        minimum_user_effective_sample_size=(
            contract.minimum_user_effective_sample_size
        ),
        minimum_purchase_events=contract.minimum_purchase_events,
        maximum_missing_outcome_rate=contract.maximum_missing_outcome_rate,
        maximum_p99_importance_weight=contract.maximum_p99_importance_weight,
        weight_clip=contract.weight_clip,
        bootstrap_repetitions=contract.bootstrap_repetitions,
        bootstrap_seed=contract.bootstrap_seed,
        confidence_level=contract.confidence_level,
        outcome_lower_bound=contract.outcome_lower_bound,
        outcome_upper_bound=contract.outcome_upper_bound,
        minimum_raw_requests=contract.minimum_raw_requests,
        maximum_runtime_seconds=contract.maximum_runtime_seconds,
        maximum_memory_mb=contract.maximum_memory_mb,
        maximum_output_bytes=contract.maximum_output_bytes,
        authorized_scope=contract.authorized_scope,
        explicitly_unauthorized=contract.explicitly_unauthorized,
        pilot_allowed_uses=contract.pilot_allowed_uses,
        pilot_forbidden_uses=contract.pilot_forbidden_uses,
        contract_version=contract.contract_version,
    )
    if recreated != contract:
        raise ValueError("contract semantic fields do not match the canonical schema")
    return contract


def classify_evaluation_evidence(
    contract: OpeEvaluationContract,
    *,
    analysis_timestamp: str,
    collection_complete: bool,
    immutable_inputs_verified: bool,
    no_clobber_verified: bool,
    labels_settled: bool,
    label_snapshot_digest: str | None,
    readiness_decision: str | None,
    readiness_contract_digest: str | None,
    estimation_complete: bool,
    final_audit_passed: bool | None,
) -> EvaluationEvidenceStatus:
    """Classify evidence without collapsing operational and scientific gates."""

    if not isinstance(contract, OpeEvaluationContract) or not contract.verify_digest():
        raise ValueError("contract must be an untampered OpeEvaluationContract")
    analysis_time = _parse_aware_timestamp(analysis_timestamp, "analysis_timestamp")
    earliest = _parse_aware_timestamp(
        contract.earliest_analysis_time, "earliest_analysis_time"
    )
    for field, value in {
        "collection_complete": collection_complete,
        "immutable_inputs_verified": immutable_inputs_verified,
        "no_clobber_verified": no_clobber_verified,
        "labels_settled": labels_settled,
        "estimation_complete": estimation_complete,
    }.items():
        if not isinstance(value, bool):
            raise ValueError(f"{field} must be boolean")
    if final_audit_passed is not None and not isinstance(final_audit_passed, bool):
        raise ValueError("final_audit_passed must be boolean or None")
    if readiness_decision not in {None, "pass", "partial", "fail"}:
        raise ValueError("readiness_decision must be pass, partial, fail, or None")

    reasons: list[str] = []
    collection_state = "complete" if collection_complete else "incomplete"
    label_state = "settled" if labels_settled else "pending"
    statistical_state = "not_assessed"
    decision_state = "not_eligible"
    overall_state = "partial_invalid"

    if not collection_complete:
        reasons.append("collection_incomplete")
    elif not immutable_inputs_verified:
        collection_state = "rejected"
        overall_state = "evidence_rejected"
        reasons.append("immutable_inputs_not_verified")
    elif not no_clobber_verified:
        collection_state = "rejected"
        overall_state = "evidence_rejected"
        reasons.append("no_clobber_not_verified")
    elif analysis_time < earliest or not labels_settled:
        overall_state = "labels_pending"
        label_state = "pending"
        reasons.append("labels_not_yet_eligible_or_settled")
    elif (
        not isinstance(label_snapshot_digest, str)
        or len(label_snapshot_digest) != 64
        or any(
            character not in "0123456789abcdef" for character in label_snapshot_digest
        )
    ):
        label_state = "rejected"
        overall_state = "evidence_rejected"
        reasons.append("label_snapshot_digest_invalid")
    elif readiness_decision is None:
        overall_state = "analysis_ready"
        reasons.append("readiness_not_run")
    elif readiness_contract_digest != contract.contract_digest:
        overall_state = "evidence_rejected"
        reasons.append("readiness_contract_mismatch")
    elif readiness_decision == "fail":
        statistical_state = "insufficient"
        overall_state = "insufficient_evidence"
        reasons.append("statistical_or_support_gates_failed")
    else:
        statistical_state = "sufficient"
        if not estimation_complete:
            overall_state = "analysis_ready"
            reasons.append("estimation_not_complete")
        elif final_audit_passed is None:
            overall_state = "artifact_complete"
            reasons.append("independent_final_audit_pending")
        elif not final_audit_passed:
            overall_state = "evidence_rejected"
            reasons.append("independent_final_audit_failed")
        else:
            overall_state = "decision_eligible"
            decision_state = "eligible"

    analysis_ready = (
        collection_state == "complete"
        and label_state == "settled"
        and overall_state
        not in {"partial_invalid", "labels_pending", "evidence_rejected"}
    )
    statistically_sufficient = statistical_state == "sufficient"
    decision_eligible = decision_state == "eligible"
    return EvaluationEvidenceStatus(
        contract_digest=contract.contract_digest,
        collection_state=collection_state,
        label_state=label_state,
        statistical_state=statistical_state,
        decision_state=decision_state,
        overall_state=overall_state,
        collection_complete=collection_state == "complete",
        analysis_ready=analysis_ready,
        statistically_sufficient=statistically_sufficient,
        decision_eligible=decision_eligible,
        authorized_actions=(contract.authorized_scope if decision_eligible else ()),
        explicitly_unauthorized=contract.explicitly_unauthorized,
        reasons=tuple(reasons),
    )
