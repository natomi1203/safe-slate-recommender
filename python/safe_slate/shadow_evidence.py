"""No-clobber shadow operability evidence and source-recomputed verification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
import shutil
import tempfile
from time import perf_counter
from typing import Any, Mapping, Sequence

from .input_manifest import (
    ImmutableInputManifest,
    ObservedInputIdentity,
    ObservedInputVerification,
    audit_input_manifest_observations,
    load_input_manifest,
)
from .production import ProposalLog
from .shadow import ShadowRunSummary, summarize_shadow_run
from .shadow_contract import (
    ShadowEvaluationContract,
    load_shadow_evaluation_contract,
)
from .uncertainty_binding import audit_linear_uncertainty_input_binding


@dataclass(frozen=True, slots=True)
class ShadowEvidenceArtifact:
    role: str
    filename: str
    byte_size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ShadowEvidenceManifest:
    schema_version: str
    evaluation_id: str
    output_identity: str
    created_at: str
    contract_digest: str
    input_manifest_digest: str
    log_data_digest: str
    evidence_state: str
    operability_pass: bool
    successful_execution_ids: tuple[str, ...]
    failed_execution_ids: tuple[str, ...]
    actual_runtime_seconds: float
    peak_memory_mb: int
    safety_flags: tuple[tuple[str, bool], ...]
    artifacts: tuple[ShadowEvidenceArtifact, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ShadowEvidenceVerification:
    valid: bool
    source_recomputed: bool
    evidence_state: str
    contract_digest: str | None
    manifest_digest: str | None
    authorized_actions: tuple[str, ...]
    explicitly_unauthorized: tuple[str, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class NoClobberShadowRunResult:
    state: str
    summary: ShadowRunSummary | None
    input_verification: ObservedInputVerification
    bundle_manifest: ShadowEvidenceManifest | None
    bundle_verification: ShadowEvidenceVerification | None
    output_directory: str | None
    authorized_actions: tuple[str, ...]
    explicitly_unauthorized: tuple[str, ...]
    production_rollout_authorized: bool
    production_latency_claim_authorized: bool
    causal_lift_claim_authorized: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "summary": self.summary.to_dict() if self.summary else None,
            "input_verification": self.input_verification.to_dict(),
            "bundle_manifest": (
                self.bundle_manifest.to_dict() if self.bundle_manifest else None
            ),
            "bundle_verification": (
                self.bundle_verification.to_dict() if self.bundle_verification else None
            ),
            "output_directory": self.output_directory,
            "authorized_actions": list(self.authorized_actions),
            "explicitly_unauthorized": list(self.explicitly_unauthorized),
            "production_rollout_authorized": False,
            "production_latency_claim_authorized": False,
            "causal_lift_claim_authorized": False,
            "reasons": list(self.reasons),
        }


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


def _digest(value: bytes) -> str:
    return sha256(value).hexdigest()


def _aware_time(value: str, field: str) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed


def _finite_json(value: Any) -> bool:
    if isinstance(value, float):
        return isfinite(value)
    if isinstance(value, Mapping):
        return all(
            isinstance(key, str) and _finite_json(item) for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return all(_finite_json(item) for item in value)
    return value is None or isinstance(value, (str, bool, int))


def _normalized_ids(values: Sequence[str], field: str) -> tuple[str, ...]:
    normalized = tuple(values)
    if any(not isinstance(value, str) or not value for value in normalized):
        raise ValueError(f"{field} must contain non-empty strings")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field} must be unique")
    return normalized


def _input_verification_complete(
    input_manifest: ImmutableInputManifest,
    verification: ObservedInputVerification,
) -> bool:
    expected_ids = tuple(value.input_id for value in input_manifest.inputs)
    return (
        verification.valid
        and verification.manifest_digest == input_manifest.manifest_digest
        and verification.expected_input_count == len(input_manifest.inputs)
        and verification.observed_input_count == len(input_manifest.inputs)
        and verification.verified_input_ids == expected_ids
        and not verification.missing_input_ids
        and not verification.unexpected_input_ids
        and not verification.mismatched_input_ids
        and not verification.reasons
    )


def _summary_state(summary: ShadowRunSummary) -> str:
    if summary.duplicate_request_id_count > 0 or summary.invalid_output_rate > 0.0:
        return "evidence_rejected"
    if "request_count_below_gate" in summary.reasons:
        return "insufficient_evidence"
    return "artifact_complete"


def _factorized_uncertainty_binding_reasons(
    logs: Sequence[ProposalLog],
    *,
    input_manifest: ImmutableInputManifest,
    input_verification: ObservedInputVerification,
) -> tuple[str, ...]:
    reasons: list[str] = []
    for log in logs:
        if "safe_slate_uncertainty" not in log.context:
            continue
        assessment = audit_linear_uncertainty_input_binding(
            log,
            manifest=input_manifest,
            observed_inputs=input_verification,
        )
        reasons.extend(
            f"linear_uncertainty_binding:{log.request_id}:{reason}"
            for reason in assessment.reasons
        )
    return tuple(dict.fromkeys(reasons))


def _validate_contract_binding(
    contract: ShadowEvaluationContract,
    input_manifest: ImmutableInputManifest,
) -> None:
    if (
        not isinstance(contract, ShadowEvaluationContract)
        or not contract.verify_digest()
    ):
        raise ValueError("shadow contract is invalid or tampered")
    if (
        not isinstance(input_manifest, ImmutableInputManifest)
        or not input_manifest.verify_digest()
    ):
        raise ValueError("input manifest is invalid or tampered")
    if input_manifest.manifest_digest != contract.input_manifest_digest:
        raise ValueError("input manifest digest does not match shadow contract")
    if input_manifest.population_identity != contract.population_identity:
        raise ValueError("input manifest population does not match shadow contract")
    if "shadow_records" not in {value.semantic_role for value in input_manifest.inputs}:
        raise ValueError("input manifest lacks shadow_records")
    if _aware_time(input_manifest.feature_cutoff, "feature_cutoff") != _aware_time(
        contract.window_end, "window_end"
    ):
        raise ValueError("input manifest cutoff does not equal shadow window end")


def _summarize(
    contract: ShadowEvaluationContract,
    logs: Sequence[ProposalLog],
) -> ShadowRunSummary:
    return summarize_shadow_run(
        logs,
        window_start=contract.window_start,
        window_end=contract.window_end,
        minimum_requests=contract.minimum_requests,
        maximum_oracle_p95_ms=contract.maximum_oracle_p95_ms,
        maximum_fallback_rate=contract.maximum_fallback_rate,
        maximum_invalid_output_rate=contract.maximum_invalid_output_rate,
        maximum_recheck_failure_rate=contract.maximum_recheck_failure_rate,
        maximum_predicate_error_request_rate=(
            contract.maximum_predicate_error_request_rate
        ),
        require_service_latency_trace=contract.require_service_latency_trace,
        maximum_service_response_p95_ms=(contract.maximum_service_response_p95_ms),
        maximum_log_commit_p95_ms=contract.maximum_log_commit_p95_ms,
    )


def _write_bundle(
    target: Path,
    *,
    output_identity: str,
    contract: ShadowEvaluationContract,
    input_manifest: ImmutableInputManifest,
    input_verification: ObservedInputVerification,
    summary: ShadowRunSummary,
    evidence_state: str,
    created_at: str,
    successful_execution_ids: Sequence[str],
    failed_execution_ids: Sequence[str],
    actual_runtime_seconds: float,
    peak_memory_mb: int,
) -> ShadowEvidenceManifest:
    created = _aware_time(created_at, "created_at")
    if created < _aware_time(contract.window_end, "window_end"):
        raise ValueError("shadow evidence predates the end of its frozen window")
    if _aware_time(input_manifest.created_at, "input_manifest.created_at") > created:
        raise ValueError("shadow input manifest was created after the evidence")
    if output_identity != contract.output_namespace:
        raise ValueError("output identity does not match shadow contract")
    if not _input_verification_complete(input_manifest, input_verification):
        raise ValueError("input observations do not fully verify shadow inputs")
    if not 0.0 <= actual_runtime_seconds <= contract.maximum_runtime_seconds:
        raise ValueError("shadow runtime exceeds its contract")
    if not isinstance(peak_memory_mb, int) or isinstance(peak_memory_mb, bool):
        raise ValueError("peak_memory_mb must be an integer")
    if not 0 <= peak_memory_mb <= contract.maximum_memory_mb:
        raise ValueError("shadow memory exceeds its contract")
    successful = _normalized_ids(successful_execution_ids, "successful_execution_ids")
    failed = _normalized_ids(failed_execution_ids, "failed_execution_ids")
    if not successful or set(successful) & set(failed):
        raise ValueError("shadow execution identities are invalid")
    if target.exists():
        raise FileExistsError(f"refusing to overwrite shadow evidence: {target}")
    if not target.parent.is_dir():
        raise ValueError("shadow evidence parent directory must exist")

    payloads = {
        "contract": ("contract.json", contract.to_dict()),
        "input_manifest": ("input_manifest.json", input_manifest.to_dict()),
        "input_verification": (
            "input_verification.json",
            input_verification.to_dict(),
        ),
        "summary": ("summary.json", summary.to_dict()),
    }
    if not all(_finite_json(payload) for _, payload in payloads.values()):
        raise ValueError("shadow evidence contains non-finite or non-JSON values")
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    try:
        artifacts: list[ShadowEvidenceArtifact] = []
        total_bytes = 0
        for role, (filename, payload) in payloads.items():
            encoded = _canonical_json(payload)
            total_bytes += len(encoded)
            if total_bytes > contract.maximum_output_bytes:
                raise ValueError("shadow evidence exceeds output byte cap")
            (temporary / filename).write_bytes(encoded)
            artifacts.append(
                ShadowEvidenceArtifact(
                    role=role,
                    filename=filename,
                    byte_size=len(encoded),
                    sha256=_digest(encoded),
                )
            )
        manifest = ShadowEvidenceManifest(
            schema_version="1.0.0",
            evaluation_id=contract.evaluation_id,
            output_identity=output_identity,
            created_at=created_at,
            contract_digest=contract.contract_digest,
            input_manifest_digest=input_manifest.manifest_digest,
            log_data_digest=summary.log_data_digest,
            evidence_state=evidence_state,
            operability_pass=summary.operability_pass,
            successful_execution_ids=successful,
            failed_execution_ids=failed,
            actual_runtime_seconds=float(actual_runtime_seconds),
            peak_memory_mb=peak_memory_mb,
            safety_flags=(
                ("causal_lift_claim_emitted", False),
                ("production_latency_claim_emitted", False),
                ("production_source_writes", False),
                ("serving_changes", False),
            ),
            artifacts=tuple(artifacts),
        )
        manifest_bytes = _canonical_json(manifest.to_dict())
        completion = {
            "schema_version": "1.0.0",
            "state": "complete",
            "manifest_filename": "manifest.json",
            "manifest_byte_size": len(manifest_bytes),
            "manifest_sha256": _digest(manifest_bytes),
            "artifact_count": len(artifacts),
        }
        completion_bytes = _canonical_json(completion)
        total_bytes += len(manifest_bytes) + len(completion_bytes)
        if total_bytes > contract.maximum_output_bytes:
            raise ValueError("shadow evidence exceeds output byte cap")
        (temporary / "manifest.json").write_bytes(manifest_bytes)
        (temporary / "complete.json").write_bytes(completion_bytes)
        os.replace(temporary, target)
        return manifest
    except BaseException:
        # Every fallible operation before atomic replace leaves this exact
        # temporary directory in place; successful replace returns normally.
        if (  # pragma: no branch - atomic publication invariant
            temporary.exists() and temporary.parent == target.parent
        ):
            shutil.rmtree(temporary)
        raise


def verify_shadow_evidence_bundle(
    output_directory: str | Path,
    *,
    logs: Sequence[ProposalLog] | None = None,
) -> ShadowEvidenceVerification:
    """Verify persisted bytes and recompute the summary from immutable logs."""

    target = Path(output_directory)
    reasons: list[str] = []
    contract_digest: str | None = None
    manifest_digest: str | None = None
    evidence_state = "evidence_rejected"
    authorized: tuple[str, ...] = ()
    unauthorized: tuple[str, ...] = ()
    source_recomputed = False
    try:
        expected_files = {
            "contract.json",
            "input_manifest.json",
            "input_verification.json",
            "summary.json",
            "manifest.json",
            "complete.json",
        }
        if not target.is_dir():
            raise ValueError("shadow evidence directory is missing")
        if {path.name for path in target.iterdir() if path.is_file()} != expected_files:
            raise ValueError(
                "shadow evidence file set is incomplete or contains extras"
            )
        completion = json.loads((target / "complete.json").read_bytes())
        manifest_bytes = (target / "manifest.json").read_bytes()
        manifest_digest = _digest(manifest_bytes)
        if set(completion) != {
            "schema_version",
            "state",
            "manifest_filename",
            "manifest_byte_size",
            "manifest_sha256",
            "artifact_count",
        }:
            raise ValueError("shadow completion fields are noncanonical")
        if (
            completion.get("schema_version") != "1.0.0"
            or completion.get("manifest_filename") != "manifest.json"
            or completion.get("state") != "complete"
        ):
            raise ValueError("shadow completion marker is not complete")
        if completion.get("manifest_sha256") != manifest_digest or completion.get(
            "manifest_byte_size"
        ) != len(manifest_bytes):
            raise ValueError("shadow manifest marker mismatch")
        manifest = json.loads(manifest_bytes)
        if set(manifest) != set(ShadowEvidenceManifest.__dataclass_fields__):
            raise ValueError("shadow manifest fields do not match canonical schema")
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, list) or len(artifacts) != completion.get(
            "artifact_count"
        ):
            raise ValueError("shadow artifact count mismatch")
        expected_artifacts = expected_files - {"manifest.json", "complete.json"}
        names: set[str] = set()
        roles: set[str] = set()
        for artifact in artifacts:
            name = artifact.get("filename")
            role = artifact.get("role")
            if name in names or name not in expected_artifacts:
                raise ValueError("shadow artifact name is invalid or duplicated")
            if role in roles or role not in {
                "contract",
                "input_manifest",
                "input_verification",
                "summary",
            }:
                raise ValueError("shadow artifact role is invalid or duplicated")
            data = (target / name).read_bytes()
            if len(data) != artifact.get("byte_size") or _digest(data) != artifact.get(
                "sha256"
            ):
                raise ValueError(f"shadow artifact integrity mismatch: {name}")
            names.add(name)
            roles.add(role)
        if names != expected_artifacts or roles != {
            "contract",
            "input_manifest",
            "input_verification",
            "summary",
        }:
            raise ValueError("shadow artifact manifest is incomplete")

        contract = load_shadow_evaluation_contract(
            json.loads((target / "contract.json").read_bytes())
        )
        contract_digest = contract.contract_digest
        input_manifest = load_input_manifest(
            json.loads((target / "input_manifest.json").read_bytes())
        )
        _validate_contract_binding(contract, input_manifest)
        input_verification = json.loads(
            (target / "input_verification.json").read_bytes()
        )
        expected_input_ids = [value.input_id for value in input_manifest.inputs]
        if (
            set(input_verification)
            != set(ObservedInputVerification.__dataclass_fields__)
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
            raise ValueError("shadow input verification is incomplete")
        summary_payload = json.loads((target / "summary.json").read_bytes())
        if set(summary_payload) != set(ShadowRunSummary.__dataclass_fields__):
            raise ValueError("shadow summary fields do not match canonical schema")
        if not _finite_json(summary_payload):
            raise ValueError("shadow summary contains invalid numeric values")
        if logs is None:
            raise ValueError("source logs are required for independent verification")
        persisted_input_verification = ObservedInputVerification(
            valid=input_verification["valid"],
            manifest_digest=input_verification["manifest_digest"],
            expected_input_count=input_verification["expected_input_count"],
            observed_input_count=input_verification["observed_input_count"],
            verified_input_ids=tuple(input_verification["verified_input_ids"]),
            missing_input_ids=tuple(input_verification["missing_input_ids"]),
            unexpected_input_ids=tuple(input_verification["unexpected_input_ids"]),
            mismatched_input_ids=tuple(input_verification["mismatched_input_ids"]),
            reasons=tuple(input_verification["reasons"]),
        )
        uncertainty_reasons = _factorized_uncertainty_binding_reasons(
            logs,
            input_manifest=input_manifest,
            input_verification=persisted_input_verification,
        )
        if uncertainty_reasons:
            raise ValueError("factorized uncertainty binding is invalid")
        recomputed = _summarize(contract, logs)
        source_recomputed = True
        if recomputed.to_dict() != summary_payload:
            raise ValueError(
                "persisted shadow summary differs from source recomputation"
            )
        expected_state = _summary_state(recomputed)
        if manifest.get("schema_version") != "1.0.0":
            raise ValueError("unsupported shadow evidence schema")
        if manifest.get("evaluation_id") != contract.evaluation_id:
            raise ValueError("shadow evaluation identity mismatch")
        if manifest.get("contract_digest") != contract.contract_digest:
            raise ValueError("shadow manifest contract digest mismatch")
        if manifest.get("input_manifest_digest") != input_manifest.manifest_digest:
            raise ValueError("shadow manifest input digest mismatch")
        if manifest.get("log_data_digest") != recomputed.log_data_digest:
            raise ValueError("shadow manifest log digest mismatch")
        if manifest.get("output_identity") != contract.output_namespace:
            raise ValueError("shadow output identity mismatch")
        if manifest.get("evidence_state") != expected_state:
            raise ValueError("shadow evidence state mismatch")
        if manifest.get("operability_pass") is not recomputed.operability_pass:
            raise ValueError("shadow operability decision mismatch")
        created = _aware_time(manifest.get("created_at"), "created_at")
        if created < _aware_time(contract.window_end, "window_end"):
            raise ValueError("shadow bundle predates its frozen window end")
        if (
            _aware_time(input_manifest.created_at, "input_manifest.created_at")
            > created
        ):
            raise ValueError("shadow input manifest postdates the bundle")
        if sum(path.stat().st_size for path in target.iterdir() if path.is_file()) > (
            contract.maximum_output_bytes
        ):
            raise ValueError("shadow bundle exceeds output byte cap")
        runtime = manifest.get("actual_runtime_seconds")
        memory = manifest.get("peak_memory_mb")
        if (
            not isinstance(runtime, (int, float))
            or isinstance(runtime, bool)
            or not 0.0 <= runtime <= contract.maximum_runtime_seconds
        ):
            raise ValueError("shadow runtime cap exceeded")
        if (
            not isinstance(memory, int)
            or isinstance(memory, bool)
            or not 0 <= memory <= contract.maximum_memory_mb
        ):
            raise ValueError("shadow memory cap exceeded")
        if dict(manifest.get("safety_flags", ())) != {
            "causal_lift_claim_emitted": False,
            "production_latency_claim_emitted": False,
            "production_source_writes": False,
            "serving_changes": False,
        }:
            raise ValueError("shadow safety flags are invalid")
        successful_ids = tuple(manifest.get("successful_execution_ids", ()))
        failed_ids = tuple(manifest.get("failed_execution_ids", ()))
        if (
            not successful_ids
            or len(set(successful_ids)) != len(successful_ids)
            or len(set(failed_ids)) != len(failed_ids)
            or set(successful_ids) & set(failed_ids)
        ):
            raise ValueError("shadow execution identities are invalid")
        evidence_state = (
            "evidence_valid"
            if expected_state == "artifact_complete"
            else expected_state
        )
        unauthorized = contract.explicitly_unauthorized
        if evidence_state != "evidence_rejected":
            authorized = contract.authorized_scope
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        reasons.append(str(error))
    return ShadowEvidenceVerification(
        valid=not reasons,
        source_recomputed=source_recomputed,
        evidence_state=evidence_state,
        contract_digest=contract_digest,
        manifest_digest=manifest_digest,
        authorized_actions=authorized if not reasons else (),
        explicitly_unauthorized=unauthorized,
        reasons=tuple(reasons),
    )


def run_no_clobber_shadow_evaluation(
    output_directory: str | Path,
    *,
    output_identity: str,
    contract: ShadowEvaluationContract,
    input_manifest: ImmutableInputManifest,
    observed_inputs: Sequence[ObservedInputIdentity],
    logs: Sequence[ProposalLog],
    created_at: str,
    successful_execution_ids: Sequence[str],
    failed_execution_ids: Sequence[str] = (),
    peak_memory_mb: int,
    no_matching_execution_verified: bool,
) -> NoClobberShadowRunResult:
    """Audit and atomically publish one immutable shadow evaluation."""

    _validate_contract_binding(contract, input_manifest)
    if no_matching_execution_verified is not True:
        raise ValueError("matching shadow execution identity was not proven absent")
    target = Path(output_directory)
    if target.exists():
        raise FileExistsError(f"refusing to reuse shadow output: {target}")
    if not target.parent.is_dir():
        raise ValueError("shadow output parent directory must exist")
    input_verification = audit_input_manifest_observations(
        input_manifest, observed_inputs
    )
    if not _input_verification_complete(input_manifest, input_verification):
        return NoClobberShadowRunResult(
            state="evidence_rejected",
            summary=None,
            input_verification=input_verification,
            bundle_manifest=None,
            bundle_verification=None,
            output_directory=None,
            authorized_actions=(),
            explicitly_unauthorized=contract.explicitly_unauthorized,
            production_rollout_authorized=False,
            production_latency_claim_authorized=False,
            causal_lift_claim_authorized=False,
            reasons=input_verification.reasons,
        )
    uncertainty_reasons = _factorized_uncertainty_binding_reasons(
        logs,
        input_manifest=input_manifest,
        input_verification=input_verification,
    )
    if uncertainty_reasons:
        return NoClobberShadowRunResult(
            state="evidence_rejected",
            summary=None,
            input_verification=input_verification,
            bundle_manifest=None,
            bundle_verification=None,
            output_directory=None,
            authorized_actions=(),
            explicitly_unauthorized=contract.explicitly_unauthorized,
            production_rollout_authorized=False,
            production_latency_claim_authorized=False,
            causal_lift_claim_authorized=False,
            reasons=uncertainty_reasons,
        )
    started = perf_counter()
    summary = _summarize(contract, logs)
    state = _summary_state(summary)
    elapsed = perf_counter() - started
    manifest = _write_bundle(
        target,
        output_identity=output_identity,
        contract=contract,
        input_manifest=input_manifest,
        input_verification=input_verification,
        summary=summary,
        evidence_state=state,
        created_at=created_at,
        successful_execution_ids=successful_execution_ids,
        failed_execution_ids=failed_execution_ids,
        actual_runtime_seconds=elapsed,
        peak_memory_mb=peak_memory_mb,
    )
    verification = verify_shadow_evidence_bundle(target, logs=logs)
    if not verification.valid:
        final_state = "evidence_rejected"
        reasons = verification.reasons
        authorized: tuple[str, ...] = ()
    elif state == "artifact_complete":
        final_state = "evidence_valid"
        reasons = summary.reasons
        authorized = contract.authorized_scope
    else:
        final_state = state
        reasons = summary.reasons
        authorized = (
            contract.authorized_scope if state == "insufficient_evidence" else ()
        )
    return NoClobberShadowRunResult(
        state=final_state,
        summary=summary,
        input_verification=input_verification,
        bundle_manifest=manifest,
        bundle_verification=verification,
        output_directory=str(target),
        authorized_actions=authorized,
        explicitly_unauthorized=contract.explicitly_unauthorized,
        production_rollout_authorized=False,
        production_latency_claim_authorized=False,
        causal_lift_claim_authorized=False,
        reasons=tuple(reasons),
    )
