"""No-clobber evidence bundles and independent offline verification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

from .contract import (
    EvaluationEvidenceStatus,
    OpeEvaluationContract,
    load_ope_evaluation_contract,
)
from .input_manifest import (
    ImmutableInputManifest,
    ObservedInputVerification,
    load_input_manifest,
)
from .ope import OpeReadinessAssessment


@dataclass(frozen=True, slots=True)
class EvidenceArtifact:
    role: str
    filename: str
    byte_size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class EvidenceBundleManifest:
    schema_version: str
    evaluation_id: str
    output_identity: str
    created_at: str
    contract_digest: str
    input_manifest_digest: str
    row_data_digest: str
    estimate_type: str | None
    evidence_state: str
    successful_execution_ids: tuple[str, ...]
    failed_execution_ids: tuple[str, ...]
    actual_runtime_seconds: float
    peak_memory_mb: int
    safety_flags: tuple[tuple[str, bool], ...]
    artifacts: tuple[EvidenceArtifact, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvidenceBundleVerification:
    valid: bool
    evidence_state: str
    contract_digest: str | None
    manifest_digest: str | None
    authorized_actions: tuple[str, ...]
    explicitly_unauthorized: tuple[str, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _digest_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def _parse_aware_timestamp(value: str, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty ISO-8601 timestamp")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError(f"{field} must be ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed


def _finite_json(value: Any) -> bool:
    if isinstance(value, float):
        return value == value and value not in {float("inf"), float("-inf")}
    if isinstance(value, Mapping):
        return all(
            _finite_json(key) and _finite_json(item) for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return all(_finite_json(item) for item in value)
    return True


def _normalized_ids(values: Sequence[str], field: str) -> tuple[str, ...]:
    result = tuple(values)
    if any(not isinstance(value, str) or not value for value in result):
        raise ValueError(f"{field} must contain non-empty strings")
    if len(set(result)) != len(result):
        raise ValueError(f"{field} must be unique")
    return result


def write_evidence_bundle(
    output_directory: str | Path,
    *,
    output_identity: str,
    contract: OpeEvaluationContract,
    input_manifest: ImmutableInputManifest,
    input_verification: ObservedInputVerification,
    readiness: OpeReadinessAssessment,
    estimate: Any,
    evidence_status: EvaluationEvidenceStatus,
    created_at: str,
    successful_execution_ids: Sequence[str],
    failed_execution_ids: Sequence[str] = (),
    actual_runtime_seconds: float,
    peak_memory_mb: int,
    no_matching_execution_verified: bool,
    production_source_writes: bool = False,
    serving_changes: bool = False,
    credentials_or_personal_records_emitted: bool = False,
) -> EvidenceBundleManifest:
    """Atomically publish a new evidence directory and completion marker."""

    if not isinstance(contract, OpeEvaluationContract) or not contract.verify_digest():
        raise ValueError("contract is invalid or has been tampered with")
    if (
        not isinstance(input_manifest, ImmutableInputManifest)
        or not input_manifest.verify_digest()
    ):
        raise ValueError("input manifest is invalid or has been tampered with")
    if input_manifest.manifest_digest != contract.input_manifest_digest:
        raise ValueError("input manifest digest does not match the contract")
    if input_manifest.population_identity != contract.population_identity:
        raise ValueError("input manifest population does not match the contract")
    if not isinstance(input_verification, ObservedInputVerification):
        raise ValueError("input_verification must be an ObservedInputVerification")
    expected_input_ids = tuple(value.input_id for value in input_manifest.inputs)
    if (
        not input_verification.valid
        or input_verification.manifest_digest != input_manifest.manifest_digest
        or input_verification.expected_input_count != len(input_manifest.inputs)
        or input_verification.observed_input_count != len(input_manifest.inputs)
        or input_verification.verified_input_ids != expected_input_ids
        or input_verification.missing_input_ids
        or input_verification.unexpected_input_ids
        or input_verification.mismatched_input_ids
        or input_verification.reasons
    ):
        raise ValueError("input observations do not fully verify the manifest")
    if not isinstance(readiness, OpeReadinessAssessment):
        raise ValueError("readiness must be an OpeReadinessAssessment")
    if not isinstance(evidence_status, EvaluationEvidenceStatus):
        raise ValueError("evidence_status must be an EvaluationEvidenceStatus")
    if output_identity != contract.output_namespace:
        raise ValueError("output identity does not match the contract")
    if no_matching_execution_verified is not True:
        raise ValueError("matching execution identity was not proven absent")
    if any(
        value is not False
        for value in (
            production_source_writes,
            serving_changes,
            credentials_or_personal_records_emitted,
        )
    ):
        raise ValueError("offline evidence safety flags must all remain false")
    if readiness.evaluation_contract_digest != contract.contract_digest:
        raise ValueError("readiness contract digest mismatch")
    if estimate is not None:
        estimate_contract_digest = getattr(estimate, "evaluation_contract_digest", None)
        estimate_row_digest = getattr(estimate, "row_data_digest", None)
        if estimate_contract_digest != contract.contract_digest:
            raise ValueError("estimate contract digest mismatch")
        if estimate_row_digest != readiness.row_data_digest:
            raise ValueError("estimate row digest mismatch")
    if evidence_status.contract_digest != contract.contract_digest:
        raise ValueError("evidence status contract digest mismatch")
    if estimate is not None and not hasattr(estimate, "to_dict"):
        raise ValueError("estimate must expose to_dict")
    if estimate is None and evidence_status.overall_state not in {
        "analysis_ready",
        "insufficient_evidence",
    }:
        raise ValueError("this evidence state requires a completed estimate")
    if (
        estimate is not None
        and evidence_status.overall_state == "insufficient_evidence"
    ):
        raise ValueError("insufficient evidence may not contain an estimate")

    created = _parse_aware_timestamp(created_at, "created_at")
    earliest = _parse_aware_timestamp(
        contract.earliest_analysis_time, "earliest_analysis_time"
    )
    if created < earliest:
        raise ValueError("evidence may not be created before earliest_analysis_time")
    manifest_created = _parse_aware_timestamp(
        input_manifest.created_at, "input_manifest.created_at"
    )
    feature_cutoff = _parse_aware_timestamp(
        input_manifest.feature_cutoff, "input_manifest.feature_cutoff"
    )
    label_cutoff = _parse_aware_timestamp(
        input_manifest.label_cutoff, "input_manifest.label_cutoff"
    )
    final_window_end = _parse_aware_timestamp(
        contract.final_window_end, "final_window_end"
    )
    if feature_cutoff != final_window_end:
        raise ValueError("input manifest feature cutoff must equal final window end")
    if label_cutoff < final_window_end or label_cutoff > earliest:
        raise ValueError("input manifest label cutoff is outside settlement boundary")
    if manifest_created > created:
        raise ValueError("input manifest was created after the evidence bundle")
    if (
        not isinstance(actual_runtime_seconds, (int, float))
        or isinstance(actual_runtime_seconds, bool)
        or not 0.0 <= actual_runtime_seconds <= contract.maximum_runtime_seconds
    ):
        raise ValueError("actual runtime exceeds or violates the contract")
    if (
        not isinstance(peak_memory_mb, int)
        or isinstance(peak_memory_mb, bool)
        or not 0 <= peak_memory_mb <= contract.maximum_memory_mb
    ):
        raise ValueError("peak memory exceeds or violates the contract")
    successful = _normalized_ids(successful_execution_ids, "successful_execution_ids")
    failed = _normalized_ids(failed_execution_ids, "failed_execution_ids")
    if not successful:
        raise ValueError("at least one successful execution identity is required")
    if set(successful) & set(failed):
        raise ValueError("successful and failed execution identities may not overlap")

    target = Path(output_directory)
    if target.name in {"", ".", ".."}:
        raise ValueError("output directory must name a new, specific directory")
    parent = target.parent
    if not parent.exists() or not parent.is_dir():
        raise ValueError("output parent directory must already exist")
    if target.exists():
        raise FileExistsError(f"refusing to overwrite evidence directory: {target}")

    contract_dict = contract.to_dict()
    input_manifest_dict = input_manifest.to_dict()
    input_verification_dict = input_verification.to_dict()
    readiness_dict = readiness.to_dict()
    estimate_dict = estimate.to_dict() if estimate is not None else None
    status_dict = evidence_status.to_dict()
    payloads = {
        "contract": ("contract.json", contract_dict),
        "input_manifest": ("input_manifest.json", input_manifest_dict),
        "input_verification": (
            "input_verification.json",
            input_verification_dict,
        ),
        "readiness": ("readiness.json", readiness_dict),
        "evidence_status": ("evidence_status.json", status_dict),
    }
    if estimate_dict is not None:
        payloads["estimate"] = ("estimate.json", estimate_dict)
    if not all(_finite_json(payload) for _, payload in payloads.values()):
        raise ValueError("evidence payload contains non-finite values")

    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=parent))
    try:
        artifacts: list[EvidenceArtifact] = []
        total_bytes = 0
        for role, (filename, payload) in payloads.items():
            encoded = _canonical_json(payload)
            total_bytes += len(encoded)
            if total_bytes > contract.maximum_output_bytes:
                raise ValueError("evidence output exceeds contract byte cap")
            (temporary / filename).write_bytes(encoded)
            artifacts.append(
                EvidenceArtifact(
                    role=role,
                    filename=filename,
                    byte_size=len(encoded),
                    sha256=_digest_bytes(encoded),
                )
            )
        manifest = EvidenceBundleManifest(
            schema_version="1.0.0",
            evaluation_id=contract.evaluation_id,
            output_identity=output_identity,
            created_at=created_at,
            contract_digest=contract.contract_digest,
            input_manifest_digest=input_manifest.manifest_digest,
            row_data_digest=readiness.row_data_digest,
            estimate_type=type(estimate).__name__ if estimate is not None else None,
            evidence_state=evidence_status.overall_state,
            successful_execution_ids=successful,
            failed_execution_ids=failed,
            actual_runtime_seconds=float(actual_runtime_seconds),
            peak_memory_mb=peak_memory_mb,
            safety_flags=(
                ("credentials_or_personal_records_emitted", False),
                ("production_source_writes", False),
                ("serving_changes", False),
            ),
            artifacts=tuple(artifacts),
        )
        manifest_bytes = _canonical_json(manifest.to_dict())
        total_bytes += len(manifest_bytes)
        completion = {
            "schema_version": "1.0.0",
            "state": "complete",
            "manifest_filename": "manifest.json",
            "manifest_byte_size": len(manifest_bytes),
            "manifest_sha256": _digest_bytes(manifest_bytes),
            "artifact_count": len(artifacts),
        }
        completion_bytes = _canonical_json(completion)
        total_bytes += len(completion_bytes)
        if total_bytes > contract.maximum_output_bytes:
            raise ValueError("evidence output exceeds contract byte cap")
        (temporary / "manifest.json").write_bytes(manifest_bytes)
        # Completion is deliberately the last file written.
        (temporary / "complete.json").write_bytes(completion_bytes)
        os.replace(temporary, target)
        return manifest
    except BaseException:
        # Every fallible operation before atomic replace leaves this exact
        # temporary directory in place; successful replace returns normally.
        if temporary.exists() and temporary.parent == parent:  # pragma: no branch
            shutil.rmtree(temporary)
        raise


def verify_evidence_bundle(
    output_directory: str | Path,
) -> EvidenceBundleVerification:
    """Independently verify a persisted bundle from bytes and digests alone."""

    target = Path(output_directory)
    reasons: list[str] = []
    contract_digest: str | None = None
    manifest_digest: str | None = None
    evidence_state = "evidence_rejected"
    authorized_actions: tuple[str, ...] = ()
    explicitly_unauthorized: tuple[str, ...] = ()
    try:
        base_files = {
            "contract.json",
            "input_manifest.json",
            "input_verification.json",
            "readiness.json",
            "evidence_status.json",
            "manifest.json",
            "complete.json",
        }
        files_with_estimate = base_files | {"estimate.json"}
        if not target.is_dir():
            raise ValueError("evidence directory is missing")
        observed_files = {path.name for path in target.iterdir() if path.is_file()}
        if observed_files == files_with_estimate:
            estimate_present = True
            expected_files = files_with_estimate
        elif observed_files == base_files:
            estimate_present = False
            expected_files = base_files
        else:
            raise ValueError("evidence file set is incomplete or contains extras")

        completion_bytes = (target / "complete.json").read_bytes()
        completion = json.loads(completion_bytes)
        if completion.get("state") != "complete":
            raise ValueError("completion marker is not complete")
        manifest_bytes = (target / "manifest.json").read_bytes()
        manifest_digest = _digest_bytes(manifest_bytes)
        if manifest_digest != completion.get("manifest_sha256"):
            raise ValueError("manifest digest mismatch")
        if len(manifest_bytes) != completion.get("manifest_byte_size"):
            raise ValueError("manifest size mismatch")
        manifest = json.loads(manifest_bytes)
        if manifest.get("schema_version") != "1.0.0":
            raise ValueError("unsupported manifest schema")
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, list) or len(artifacts) != completion.get(
            "artifact_count"
        ):
            raise ValueError("artifact count mismatch")
        artifact_names: set[str] = set()
        artifact_roles: set[str] = set()
        for artifact in artifacts:
            filename = artifact.get("filename")
            role = artifact.get("role")
            if filename in artifact_names or filename not in expected_files - {
                "manifest.json",
                "complete.json",
            }:
                raise ValueError("artifact filename is invalid or duplicated")
            allowed_roles = {
                "contract",
                "input_manifest",
                "input_verification",
                "readiness",
                "evidence_status",
            }
            if estimate_present:
                allowed_roles.add("estimate")
            if role in artifact_roles or role not in allowed_roles:
                raise ValueError("artifact role is invalid or duplicated")
            artifact_names.add(filename)
            artifact_roles.add(role)
            data = (target / filename).read_bytes()
            if len(data) != artifact.get("byte_size"):
                raise ValueError(f"artifact size mismatch: {filename}")
            if _digest_bytes(data) != artifact.get("sha256"):
                raise ValueError(f"artifact digest mismatch: {filename}")
        expected_artifact_names = expected_files - {"manifest.json", "complete.json"}
        expected_artifact_roles = {
            "contract",
            "input_manifest",
            "input_verification",
            "readiness",
            "evidence_status",
        }
        if estimate_present:
            expected_artifact_roles.add("estimate")
        if (
            artifact_names != expected_artifact_names
            or artifact_roles != expected_artifact_roles
        ):
            raise ValueError("artifact manifest does not cover every evidence file")

        contract_payload = json.loads((target / "contract.json").read_bytes())
        input_manifest_payload = json.loads(
            (target / "input_manifest.json").read_bytes()
        )
        input_verification = json.loads(
            (target / "input_verification.json").read_bytes()
        )
        readiness = json.loads((target / "readiness.json").read_bytes())
        estimate = (
            json.loads((target / "estimate.json").read_bytes())
            if estimate_present
            else None
        )
        status = json.loads((target / "evidence_status.json").read_bytes())
        if not all(
            _finite_json(payload)
            for payload in (
                contract_payload,
                input_manifest_payload,
                input_verification,
                readiness,
                status,
                manifest,
                *([estimate] if estimate is not None else []),
            )
        ):
            raise ValueError("bundle contains non-finite numeric evidence")
        contract = load_ope_evaluation_contract(contract_payload)
        input_manifest = load_input_manifest(input_manifest_payload)
        contract_digest = contract.contract_digest
        if sum(path.stat().st_size for path in target.iterdir() if path.is_file()) > (
            contract.maximum_output_bytes
        ):
            raise ValueError("bundle exceeds output byte cap")
        if manifest.get("contract_digest") != contract.contract_digest:
            raise ValueError("manifest contract digest mismatch")
        if input_manifest.manifest_digest != contract.input_manifest_digest:
            raise ValueError("input manifest digest does not match the contract")
        if manifest.get("input_manifest_digest") != input_manifest.manifest_digest:
            raise ValueError("bundle input manifest digest mismatch")
        if input_manifest.population_identity != contract.population_identity:
            raise ValueError("input manifest population does not match the contract")
        expected_input_ids = [value.input_id for value in input_manifest.inputs]
        expected_verification_fields = set(
            ObservedInputVerification.__dataclass_fields__
        )
        if (
            not isinstance(input_verification, dict)
            or set(input_verification) != expected_verification_fields
            or input_verification.get("valid") is not True
            or input_verification.get("manifest_digest")
            != input_manifest.manifest_digest
            or input_verification.get("expected_input_count")
            != len(input_manifest.inputs)
            or input_verification.get("observed_input_count")
            != len(input_manifest.inputs)
            or input_verification.get("verified_input_ids") != expected_input_ids
            or input_verification.get("missing_input_ids") != []
            or input_verification.get("unexpected_input_ids") != []
            or input_verification.get("mismatched_input_ids") != []
            or input_verification.get("reasons") != []
        ):
            raise ValueError("persisted input verification is incomplete or invalid")
        if manifest.get("output_identity") != contract.output_namespace:
            raise ValueError("output identity mismatch")
        if readiness.get("evaluation_contract_digest") != contract.contract_digest:
            raise ValueError("readiness contract digest mismatch")
        if estimate is not None and (
            estimate.get("evaluation_contract_digest") != contract.contract_digest
        ):
            raise ValueError("estimate contract digest mismatch")
        if status.get("contract_digest") != contract.contract_digest:
            raise ValueError("status contract digest mismatch")
        row_digest = readiness.get("row_data_digest")
        if estimate is not None and estimate.get("row_data_digest") != row_digest:
            raise ValueError("estimate row digest mismatch")
        if manifest.get("row_data_digest") != row_digest:
            raise ValueError("manifest row digest mismatch")
        if manifest.get("evidence_state") != status.get("overall_state"):
            raise ValueError("evidence state mismatch")
        estimate_type = manifest.get("estimate_type")
        if estimate_present != isinstance(estimate_type, str):
            raise ValueError("estimate presence disagrees with bundle manifest")
        if estimate_present and not estimate_type:
            raise ValueError("estimate type is empty")
        if not estimate_present and status.get("overall_state") not in {
            "analysis_ready",
            "insufficient_evidence",
        }:
            raise ValueError("evidence state requires a persisted estimate")
        if estimate_present and status.get("overall_state") == "insufficient_evidence":
            raise ValueError("insufficient evidence may not contain an estimate")
        frozen_readiness = {
            "minimum_raw_requests": contract.minimum_raw_requests,
            "minimum_user_effective_sample_size": (
                contract.minimum_user_effective_sample_size
            ),
            "minimum_purchase_events": contract.minimum_purchase_events,
            "maximum_missing_outcome_rate": contract.maximum_missing_outcome_rate,
            "maximum_allowed_p99_importance_weight": (
                contract.maximum_p99_importance_weight
            ),
            "weight_clip": contract.weight_clip,
            "full_estimand_support_threshold": (
                contract.minimum_support_rate_for_full_estimand
            ),
            "partial_estimand_support_threshold": (
                contract.minimum_support_rate_for_partial_estimand
            ),
            "propensity_mode": contract.propensity_mode,
        }
        if any(readiness.get(key) != value for key, value in frozen_readiness.items()):
            raise ValueError("readiness settings differ from contract")
        frozen_estimation = {
            "bootstrap_repetitions": contract.bootstrap_repetitions,
            "bootstrap_seed": contract.bootstrap_seed,
            "confidence_level": contract.confidence_level,
            "weight_clip": contract.weight_clip,
        }
        if estimate is not None and any(
            estimate.get(key) != value for key, value in frozen_estimation.items()
        ):
            raise ValueError("estimation settings differ from contract")
        if (
            estimate is not None
            and "outcome_lower_bound" in estimate
            and (
                estimate.get("outcome_lower_bound") != contract.outcome_lower_bound
                or estimate.get("outcome_upper_bound") != contract.outcome_upper_bound
            )
        ):
            raise ValueError("estimate outcome bounds differ from contract")
        if status.get("overall_state") not in contract.decision_vocabulary:
            raise ValueError("evidence state is outside contract vocabulary")
        created = _parse_aware_timestamp(manifest.get("created_at"), "created_at")
        earliest = _parse_aware_timestamp(
            contract.earliest_analysis_time, "earliest_analysis_time"
        )
        if created < earliest:
            raise ValueError("bundle predates earliest analysis time")
        manifest_created = _parse_aware_timestamp(
            input_manifest.created_at, "input_manifest.created_at"
        )
        feature_cutoff = _parse_aware_timestamp(
            input_manifest.feature_cutoff, "input_manifest.feature_cutoff"
        )
        label_cutoff = _parse_aware_timestamp(
            input_manifest.label_cutoff, "input_manifest.label_cutoff"
        )
        final_window_end = _parse_aware_timestamp(
            contract.final_window_end, "final_window_end"
        )
        if feature_cutoff != final_window_end:
            raise ValueError(
                "input manifest feature cutoff must equal final window end"
            )
        if label_cutoff < final_window_end or label_cutoff > earliest:
            raise ValueError(
                "input manifest label cutoff is outside settlement boundary"
            )
        if manifest_created > created:
            raise ValueError("input manifest postdates the evidence bundle")
        runtime = manifest.get("actual_runtime_seconds")
        if (
            not isinstance(runtime, (int, float))
            or isinstance(runtime, bool)
            or not 0.0 <= runtime <= contract.maximum_runtime_seconds
        ):
            raise ValueError("runtime cap exceeded")
        memory = manifest.get("peak_memory_mb")
        if (
            not isinstance(memory, int)
            or isinstance(memory, bool)
            or not 0 <= memory <= contract.maximum_memory_mb
        ):
            raise ValueError("memory cap exceeded")
        safety_flags = dict(manifest.get("safety_flags", []))
        if safety_flags != {
            "credentials_or_personal_records_emitted": False,
            "production_source_writes": False,
            "serving_changes": False,
        }:
            raise ValueError("offline safety flags are invalid")
        successful_ids = tuple(manifest.get("successful_execution_ids", ()))
        failed_ids = tuple(manifest.get("failed_execution_ids", ()))
        if (
            not successful_ids
            or len(set(successful_ids)) != len(successful_ids)
            or len(set(failed_ids)) != len(failed_ids)
            or set(successful_ids) & set(failed_ids)
        ):
            raise ValueError("execution identities are invalid")
        if tuple(status.get("explicitly_unauthorized", ())) != (
            contract.explicitly_unauthorized
        ):
            raise ValueError("unauthorized action boundary mismatch")
        expected_authorized = (
            contract.authorized_scope if status.get("decision_eligible") is True else ()
        )
        if tuple(status.get("authorized_actions", ())) != expected_authorized:
            raise ValueError("authorized action boundary mismatch")
        # Contract construction rejects overlapping scopes, and
        # expected_authorized is either its authorized scope or empty.
        if set(expected_authorized) & set(  # pragma: no cover - invariant defense
            contract.explicitly_unauthorized
        ):
            raise ValueError("authorized and unauthorized actions overlap")
        overall_state = status.get("overall_state")
        if overall_state == "decision_eligible" and not all(
            status.get(field) is True
            for field in (
                "collection_complete",
                "analysis_ready",
                "statistically_sufficient",
                "decision_eligible",
            )
        ):
            raise ValueError("decision-eligible state is internally inconsistent")
        if (
            overall_state != "decision_eligible"
            and status.get("decision_eligible") is True
        ):
            raise ValueError("decision eligibility boolean contradicts state")
        readiness_decision = readiness.get("decision")
        if readiness_decision == "fail" and overall_state not in {
            "insufficient_evidence",
            "evidence_rejected",
        }:
            raise ValueError("failed readiness was promoted to valid evidence")
        if (
            readiness_decision in {"pass", "partial"}
            and status.get("statistically_sufficient") is not True
        ):
            raise ValueError("readiness sufficiency contradicts evidence status")
        evidence_state = str(status.get("overall_state"))
        authorized_actions = tuple(status.get("authorized_actions", ()))
        explicitly_unauthorized = contract.explicitly_unauthorized
    except (OSError, ValueError, TypeError, json.JSONDecodeError, KeyError) as error:
        reasons.append(str(error))
    return EvidenceBundleVerification(
        valid=not reasons,
        evidence_state=evidence_state,
        contract_digest=contract_digest,
        manifest_digest=manifest_digest,
        authorized_actions=authorized_actions if not reasons else (),
        explicitly_unauthorized=explicitly_unauthorized,
        reasons=tuple(reasons),
    )
