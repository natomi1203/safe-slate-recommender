"""No-clobber orchestration for manifest-bound offline OPE."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

from .contract import (
    EvaluationEvidenceStatus,
    OpeEvaluationContract,
    classify_evaluation_evidence,
)
from .estimation import (
    CrossFittedOpeEstimate,
    JointUncertaintyBoundsEstimate,
    MissingnessAdjustedOpeEstimate,
    PartialIdentificationEstimate,
    PropensityIntervalEstimate,
    estimate_cross_fitted_policy_value,
    estimate_joint_uncertainty_bounds,
    estimate_missingness_adjusted_policy_value,
    estimate_partial_identification_bounds,
    estimate_propensity_interval_bounds,
)
from .evidence import (
    EvidenceBundleManifest,
    EvidenceBundleVerification,
    verify_evidence_bundle,
    write_evidence_bundle,
)
from .input_manifest import (
    ImmutableInputManifest,
    ObservedInputIdentity,
    ObservedInputVerification,
)
from .ope import (
    ManifestBoundOpeReadinessAssessment,
    OpeReadinessAssessment,
    audit_manifest_bound_ope_readiness,
)
from .population_positivity import (
    PopulationPositivityEvidence,
    StatisticalPopulationPositivityEvidence,
)


OpeEstimate = (
    CrossFittedOpeEstimate
    | PartialIdentificationEstimate
    | PropensityIntervalEstimate
    | MissingnessAdjustedOpeEstimate
    | JointUncertaintyBoundsEstimate
)


@dataclass(frozen=True, slots=True)
class NoClobberOpeRunResult:
    """Terminal local orchestration state without production authorization."""

    state: str
    evaluation_contract_digest: str
    input_manifest_digest: str
    input_verification: ObservedInputVerification
    readiness: OpeReadinessAssessment | None
    estimate: OpeEstimate | None
    evidence_status: EvaluationEvidenceStatus
    bundle_manifest: EvidenceBundleManifest | None
    bundle_verification: EvidenceBundleVerification | None
    output_directory: str | None
    evaluation_runtime_seconds: float
    production_rollout_authorized: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "evaluation_contract_digest": self.evaluation_contract_digest,
            "input_manifest_digest": self.input_manifest_digest,
            "input_verification": self.input_verification.to_dict(),
            "readiness": self.readiness.to_dict() if self.readiness else None,
            "estimate": self.estimate.to_dict() if self.estimate else None,
            "evidence_status": self.evidence_status.to_dict(),
            "bundle_manifest": (
                self.bundle_manifest.to_dict() if self.bundle_manifest else None
            ),
            "bundle_verification": (
                self.bundle_verification.to_dict() if self.bundle_verification else None
            ),
            "output_directory": self.output_directory,
            "evaluation_runtime_seconds": self.evaluation_runtime_seconds,
            "production_rollout_authorized": False,
            "reasons": list(self.reasons),
        }


def _require(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"required evaluation field is missing: {key}")
    return mapping[key]


def _attestation(mapping: Mapping[str, Any], key: str) -> bool:
    value = _require(mapping, key)
    if not isinstance(value, bool):
        raise ValueError(f"identification attestation must be boolean: {key}")
    return value


def _common_rows(row_evidence: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: _require(row_evidence, key)
        for key in (
            "request_ids",
            "user_ids",
            "guaranteed_target_support_mass",
            "target_probability_of_logged_action",
            "purchase_events",
            "fold_ids",
        )
    }


def _estimate(
    contract: OpeEvaluationContract,
    readiness: OpeReadinessAssessment,
    row_evidence: Mapping[str, Any],
    estimation_evidence: Mapping[str, Any],
) -> OpeEstimate:
    common = _common_rows(row_evidence)
    q_logged = _require(estimation_evidence, "outcome_prediction_logged_action")
    nuisance_ok = _attestation(estimation_evidence, "nuisance_predictions_cross_fitted")
    propensity_time_ok = _attestation(
        estimation_evidence, "propensity_evidence_is_request_time"
    )
    outcomes_match = _attestation(estimation_evidence, "outcomes_match_logged_actions")
    bootstrap: dict[str, Any] = {
        "bootstrap_repetitions": contract.bootstrap_repetitions,
        "bootstrap_seed": contract.bootstrap_seed,
        "confidence_level": contract.confidence_level,
        "evaluation_contract": contract,
    }
    missing = readiness.missing_outcome_rate > 0.0
    partial = readiness.decision == "partial"

    if missing:
        observation_propensities = _require(
            row_evidence, "outcome_observation_propensities"
        )
        observation_valid = _attestation(
            estimation_evidence, "observation_propensities_are_validated"
        )
        observation_time = _attestation(
            estimation_evidence,
            "observation_propensities_are_request_time_probabilities",
        )
        missingness_ok = _attestation(
            estimation_evidence, "conditional_missingness_independence"
        )
        if not partial:
            return estimate_missingness_adjusted_policy_value(
                readiness,
                **common,
                logging_propensities=_require(row_evidence, "logging_propensities"),
                outcome_observation_propensities=observation_propensities,
                outcome_prediction_logged_action=q_logged,
                outcome_prediction_target_policy=_require(
                    estimation_evidence, "outcome_prediction_target_policy"
                ),
                nuisance_predictions_cross_fitted=nuisance_ok,
                logging_propensities_are_request_time_probabilities=(
                    propensity_time_ok
                ),
                observation_propensities_are_validated=observation_valid,
                observation_propensities_are_request_time_probabilities=(
                    observation_time
                ),
                outcomes_match_logged_actions=outcomes_match,
                conditional_missingness_independence=missingness_ok,
                **bootstrap,
            )
        return estimate_joint_uncertainty_bounds(
            readiness,
            **common,
            outcome_observation_propensities=observation_propensities,
            outcome_prediction_logged_action=q_logged,
            outcome_prediction_supported_target_contribution=_require(
                estimation_evidence,
                "outcome_prediction_supported_target_contribution",
            ),
            outcome_lower_bound=contract.outcome_lower_bound,
            outcome_upper_bound=contract.outcome_upper_bound,
            nuisance_predictions_cross_fitted=nuisance_ok,
            propensity_evidence_is_request_time=propensity_time_ok,
            observation_propensities_are_validated=observation_valid,
            observation_propensities_are_request_time_probabilities=(observation_time),
            outcomes_match_logged_actions=outcomes_match,
            conditional_missingness_independence=missingness_ok,
            logging_propensities=row_evidence.get("logging_propensities"),
            logging_propensity_lower=row_evidence.get("logging_propensity_lower"),
            logging_propensity_upper=row_evidence.get("logging_propensity_upper"),
            propensity_intervals_cover_true_probabilities=(
                _attestation(
                    estimation_evidence,
                    "propensity_intervals_cover_true_probabilities",
                )
                if readiness.propensity_mode == "interval"
                else None
            ),
            **bootstrap,
        )

    if readiness.propensity_mode == "interval":
        return estimate_propensity_interval_bounds(
            readiness,
            **common,
            logging_propensity_lower=_require(row_evidence, "logging_propensity_lower"),
            logging_propensity_upper=_require(row_evidence, "logging_propensity_upper"),
            outcome_prediction_logged_action=q_logged,
            outcome_prediction_supported_target_contribution=_require(
                estimation_evidence,
                "outcome_prediction_supported_target_contribution",
            ),
            outcome_lower_bound=contract.outcome_lower_bound,
            outcome_upper_bound=contract.outcome_upper_bound,
            propensity_intervals_cover_true_probabilities=_attestation(
                estimation_evidence,
                "propensity_intervals_cover_true_probabilities",
            ),
            nuisance_predictions_cross_fitted=nuisance_ok,
            propensity_intervals_are_request_time=propensity_time_ok,
            outcomes_match_logged_actions=outcomes_match,
            outcome_observation_propensities=row_evidence.get(
                "outcome_observation_propensities"
            ),
            **bootstrap,
        )

    known_propensities = _require(row_evidence, "logging_propensities")
    if partial:
        return estimate_partial_identification_bounds(
            readiness,
            **common,
            logging_propensities=known_propensities,
            outcome_prediction_logged_action=q_logged,
            outcome_prediction_supported_target_contribution=_require(
                estimation_evidence,
                "outcome_prediction_supported_target_contribution",
            ),
            outcome_lower_bound=contract.outcome_lower_bound,
            outcome_upper_bound=contract.outcome_upper_bound,
            nuisance_predictions_cross_fitted=nuisance_ok,
            logging_propensities_are_request_time_probabilities=(propensity_time_ok),
            outcomes_match_logged_actions=outcomes_match,
            outcome_observation_propensities=row_evidence.get(
                "outcome_observation_propensities"
            ),
            **bootstrap,
        )
    return estimate_cross_fitted_policy_value(
        readiness,
        **common,
        logging_propensities=known_propensities,
        outcome_prediction_logged_action=q_logged,
        outcome_prediction_target_policy=_require(
            estimation_evidence, "outcome_prediction_target_policy"
        ),
        nuisance_predictions_cross_fitted=nuisance_ok,
        logging_propensities_are_request_time_probabilities=propensity_time_ok,
        outcomes_match_logged_actions=outcomes_match,
        outcome_observation_propensities=row_evidence.get(
            "outcome_observation_propensities"
        ),
        **bootstrap,
    )


def run_no_clobber_ope(
    output_directory: str | Path,
    *,
    output_identity: str,
    contract: OpeEvaluationContract,
    input_manifest: ImmutableInputManifest,
    observed_inputs: Sequence[ObservedInputIdentity],
    population_positivity_evidence: (
        PopulationPositivityEvidence | StatisticalPopulationPositivityEvidence
    ),
    analysis_timestamp: str,
    created_at: str,
    row_evidence: Mapping[str, Any],
    estimation_evidence: Mapping[str, Any],
    collection_complete: bool,
    label_snapshot_digest: str | None,
    successful_execution_ids: Sequence[str],
    failed_execution_ids: Sequence[str] = (),
    peak_memory_mb: int,
    no_matching_execution_verified: bool,
) -> NoClobberOpeRunResult:
    """Execute immutable preflight, readiness, estimation, and publication.

    Failed input or settlement gates stop before estimation and do not create
    an output directory. A statistically insufficient readiness result is
    persisted without an estimate. A successful estimator publishes one new
    evidence directory, which remains ``artifact_complete`` until a separate
    final auditor evaluates it.
    """

    if not isinstance(contract, OpeEvaluationContract) or not contract.verify_digest():
        raise ValueError("contract must be an untampered OpeEvaluationContract")
    if output_identity != contract.output_namespace:
        raise ValueError("output identity does not match the contract")
    if not isinstance(collection_complete, bool):
        raise ValueError("collection_complete must be boolean")
    if no_matching_execution_verified is not True:
        raise ValueError("matching execution identity was not proven absent")
    target = Path(output_directory)
    if target.exists():
        raise FileExistsError(f"refusing to reuse evaluation output: {target}")
    if not target.parent.is_dir():
        raise ValueError("output parent directory must already exist")

    started = perf_counter()
    preflight: ManifestBoundOpeReadinessAssessment = audit_manifest_bound_ope_readiness(
        contract,
        input_manifest,
        observed_inputs,
        analysis_timestamp=analysis_timestamp,
        population_positivity_evidence=population_positivity_evidence,
        **dict(row_evidence),
    )
    labels_settled = row_evidence.get("labels_settled") is True
    readiness = preflight.readiness
    readiness_decision = readiness.decision if readiness else None
    status = classify_evaluation_evidence(
        contract,
        analysis_timestamp=analysis_timestamp,
        collection_complete=collection_complete,
        immutable_inputs_verified=preflight.immutable_inputs_verified,
        no_clobber_verified=True,
        labels_settled=labels_settled,
        label_snapshot_digest=label_snapshot_digest,
        readiness_decision=readiness_decision,
        readiness_contract_digest=(
            readiness.evaluation_contract_digest if readiness else None
        ),
        estimation_complete=False,
        final_audit_passed=None,
    )
    if readiness is None:
        elapsed = perf_counter() - started
        return NoClobberOpeRunResult(
            state=status.overall_state,
            evaluation_contract_digest=contract.contract_digest,
            input_manifest_digest=input_manifest.manifest_digest,
            input_verification=preflight.input_verification,
            readiness=readiness,
            estimate=None,
            evidence_status=status,
            bundle_manifest=None,
            bundle_verification=None,
            output_directory=None,
            evaluation_runtime_seconds=elapsed,
            production_rollout_authorized=False,
            reasons=tuple((*preflight.reasons, *status.reasons)),
        )
    if status.overall_state not in {"analysis_ready", "insufficient_evidence"}:
        elapsed = perf_counter() - started
        return NoClobberOpeRunResult(
            state=status.overall_state,
            evaluation_contract_digest=contract.contract_digest,
            input_manifest_digest=input_manifest.manifest_digest,
            input_verification=preflight.input_verification,
            readiness=readiness,
            estimate=None,
            evidence_status=status,
            bundle_manifest=None,
            bundle_verification=None,
            output_directory=None,
            evaluation_runtime_seconds=elapsed,
            production_rollout_authorized=False,
            reasons=tuple((*preflight.reasons, *status.reasons)),
        )
    if readiness.decision == "fail":
        elapsed = perf_counter() - started
        bundle_manifest = write_evidence_bundle(
            target,
            output_identity=output_identity,
            contract=contract,
            input_manifest=input_manifest,
            input_verification=preflight.input_verification,
            readiness=readiness,
            estimate=None,
            evidence_status=status,
            created_at=created_at,
            successful_execution_ids=successful_execution_ids,
            failed_execution_ids=failed_execution_ids,
            actual_runtime_seconds=elapsed,
            peak_memory_mb=peak_memory_mb,
            no_matching_execution_verified=True,
        )
        verification = verify_evidence_bundle(target)
        state = status.overall_state if verification.valid else "evidence_rejected"
        reasons = status.reasons if verification.valid else verification.reasons
        return NoClobberOpeRunResult(
            state=state,
            evaluation_contract_digest=contract.contract_digest,
            input_manifest_digest=input_manifest.manifest_digest,
            input_verification=preflight.input_verification,
            readiness=readiness,
            estimate=None,
            evidence_status=status,
            bundle_manifest=bundle_manifest,
            bundle_verification=verification,
            output_directory=str(target),
            evaluation_runtime_seconds=elapsed,
            production_rollout_authorized=False,
            reasons=tuple(reasons),
        )

    estimate = _estimate(contract, readiness, row_evidence, estimation_evidence)
    status = classify_evaluation_evidence(
        contract,
        analysis_timestamp=analysis_timestamp,
        collection_complete=collection_complete,
        immutable_inputs_verified=True,
        no_clobber_verified=True,
        labels_settled=labels_settled,
        label_snapshot_digest=label_snapshot_digest,
        readiness_decision=readiness.decision,
        readiness_contract_digest=readiness.evaluation_contract_digest,
        estimation_complete=True,
        final_audit_passed=None,
    )
    elapsed = perf_counter() - started
    bundle_manifest = write_evidence_bundle(
        target,
        output_identity=output_identity,
        contract=contract,
        input_manifest=input_manifest,
        input_verification=preflight.input_verification,
        readiness=readiness,
        estimate=estimate,
        evidence_status=status,
        created_at=created_at,
        successful_execution_ids=successful_execution_ids,
        failed_execution_ids=failed_execution_ids,
        actual_runtime_seconds=elapsed,
        peak_memory_mb=peak_memory_mb,
        no_matching_execution_verified=True,
    )
    verification = verify_evidence_bundle(target)
    final_state = status.overall_state if verification.valid else "evidence_rejected"
    reasons = status.reasons if verification.valid else tuple(verification.reasons)
    return NoClobberOpeRunResult(
        state=final_state,
        evaluation_contract_digest=contract.contract_digest,
        input_manifest_digest=input_manifest.manifest_digest,
        input_verification=preflight.input_verification,
        readiness=readiness,
        estimate=estimate,
        evidence_status=status,
        bundle_manifest=bundle_manifest,
        bundle_verification=verification,
        output_directory=str(target),
        evaluation_runtime_seconds=elapsed,
        production_rollout_authorized=False,
        reasons=reasons,
    )
