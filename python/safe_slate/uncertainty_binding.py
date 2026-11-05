"""Fail-closed binding of serving-time uncertainty to immutable inputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Mapping

from .input_manifest import ImmutableInputManifest, ObservedInputVerification
from .production import ProposalLog
from .uncertainty_utils import MATRIX_DIGEST_F64LE_V1, MATRIX_DIGEST_JSON_V1


LINEAR_PARAMETER_COVARIANCE_ROLE = "linear_parameter_covariance"
REQUEST_CONTRAST_FEATURES_ROLE = "request_contrast_features"


def _aware_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed


def _sha256(value: object) -> str | None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        return None
    return value


@dataclass(frozen=True, slots=True)
class LinearUncertaintySnapshotBinding:
    """Evidence that one proposal used observed covariance and contrast inputs."""

    request_id: str
    valid: bool
    manifest_digest: str | None
    snapshot_input_id: str | None
    contrast_input_id: str | None
    parameter_covariance_sha256: str | None
    contrast_features_sha256: str | None
    matrix_digest_encoding: str | None
    snapshot_manifest_bound: bool
    snapshot_observed: bool
    contrast_manifest_bound: bool
    contrast_observed: bool
    model_version_bound: bool
    contrast_digest_recorded: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit_linear_uncertainty_snapshot_binding(
    proposal_log: ProposalLog,
    *,
    manifest: ImmutableInputManifest,
    observed_inputs: ObservedInputVerification,
    expected_semantic_role: str = LINEAR_PARAMETER_COVARIANCE_ROLE,
    expected_contrast_role: str = REQUEST_CONTRAST_FEATURES_ROLE,
) -> LinearUncertaintySnapshotBinding:
    """Bind factorized serving metadata to frozen, observed matrix inputs.

    A valid result requires an untampered manifest, an exact pre-evaluation
    storage observation, matching ids/digests/model version, and input times no
    later than the request event. Missing evidence is never inferred.
    """

    if not isinstance(proposal_log, ProposalLog):
        raise ValueError("proposal_log must be a ProposalLog")
    if not isinstance(manifest, ImmutableInputManifest):
        raise ValueError("manifest must be an ImmutableInputManifest")
    if not isinstance(observed_inputs, ObservedInputVerification):
        raise ValueError("observed_inputs must be an ObservedInputVerification")
    if not isinstance(expected_semantic_role, str) or not expected_semantic_role:
        raise ValueError("expected_semantic_role must be a non-empty string")
    if not isinstance(expected_contrast_role, str) or not expected_contrast_role:
        raise ValueError("expected_contrast_role must be a non-empty string")

    reasons: list[str] = []
    metadata_raw = proposal_log.context.get("safe_slate_uncertainty")
    metadata: Mapping[str, Any]
    if not isinstance(metadata_raw, Mapping):
        metadata = {}
        reasons.append("linear_uncertainty_metadata_missing")
    else:
        metadata = metadata_raw
    if metadata.get("method") != "linear_contrast_covariance":
        reasons.append("linear_uncertainty_method_invalid")
    if metadata.get("representation") != "factorized":
        reasons.append("linear_uncertainty_representation_invalid")
    digest_encoding_raw = metadata.get(
        "matrix_digest_encoding", MATRIX_DIGEST_JSON_V1
    )
    digest_encoding = (
        digest_encoding_raw
        if digest_encoding_raw in {MATRIX_DIGEST_JSON_V1, MATRIX_DIGEST_F64LE_V1}
        else None
    )
    if digest_encoding is None:
        reasons.append("linear_uncertainty_digest_encoding_invalid")
    feature_dimension = metadata.get("feature_dimension")
    if (
        not isinstance(feature_dimension, int)
        or isinstance(feature_dimension, bool)
        or feature_dimension < 0
    ):
        reasons.append("linear_uncertainty_feature_dimension_invalid")
    operation_count = metadata.get("operation_count")
    if (
        not isinstance(operation_count, int)
        or isinstance(operation_count, bool)
        or operation_count != len(proposal_log.add_item_ids)
    ):
        reasons.append("linear_uncertainty_operation_count_mismatch")

    snapshot_value = metadata.get("snapshot_id")
    snapshot_input_id = (
        snapshot_value if isinstance(snapshot_value, str) and snapshot_value else None
    )
    if snapshot_input_id is None:
        reasons.append("uncertainty_snapshot_id_missing")
    contrast_value = metadata.get("contrast_input_id")
    contrast_input_id = (
        contrast_value if isinstance(contrast_value, str) and contrast_value else None
    )
    if contrast_input_id is None:
        reasons.append("contrast_input_id_missing")
    parameter_digest = _sha256(metadata.get("parameter_covariance_sha256"))
    if parameter_digest is None:
        reasons.append("parameter_covariance_digest_invalid")
    contrast_digest = _sha256(metadata.get("contrast_features_sha256"))
    contrast_digest_recorded = contrast_digest is not None
    if not contrast_digest_recorded:
        reasons.append("contrast_features_digest_invalid")

    manifest_valid = manifest.verify_digest()
    if not manifest_valid:
        reasons.append("input_manifest_invalid_or_tampered")
    manifest_digest = manifest.manifest_digest if manifest_valid else None
    event_time = _aware_timestamp(proposal_log.event_timestamp)
    if event_time is None:
        reasons.append("proposal_event_timestamp_missing_or_invalid")
    matching = (
        [entry for entry in manifest.inputs if entry.input_id == snapshot_input_id]
        if manifest_valid and snapshot_input_id is not None
        else []
    )
    snapshot_manifest_bound = False
    model_version_bound = False
    if len(matching) != 1:
        reasons.append(
            "uncertainty_snapshot_manifest_missing"
            if not matching
            else "uncertainty_snapshot_manifest_ambiguous"
        )
    else:
        entry = matching[0]
        if entry.semantic_role != expected_semantic_role:
            reasons.append("uncertainty_snapshot_role_mismatch")
        if (
            digest_encoding == MATRIX_DIGEST_F64LE_V1
            and entry.schema_version != "safe-slate-matrix-f64le-v1"
        ):
            reasons.append("uncertainty_snapshot_digest_encoding_mismatch")
        if parameter_digest is None or entry.sha256 != parameter_digest:
            reasons.append("uncertainty_snapshot_digest_mismatch")
        if entry.temporal_scope not in {"feature", "atemporal"}:
            reasons.append("uncertainty_snapshot_temporal_scope_invalid")
        snapshot_time = _aware_timestamp(entry.snapshot_time)
        if event_time is not None and (
            snapshot_time is None or snapshot_time > event_time
        ):
            reasons.append("uncertainty_snapshot_after_request")
        if proposal_log.model_version is None:
            reasons.append("proposal_model_version_missing")
        elif entry.version != proposal_log.model_version:
            reasons.append("uncertainty_snapshot_model_version_mismatch")
        else:
            model_version_bound = True
        snapshot_manifest_bound = not any(
            reason
            in {
                "uncertainty_snapshot_role_mismatch",
                "uncertainty_snapshot_digest_encoding_mismatch",
                "uncertainty_snapshot_digest_mismatch",
                "uncertainty_snapshot_temporal_scope_invalid",
                "uncertainty_snapshot_after_request",
            }
            for reason in reasons
        )

    contrast_matching = (
        [entry for entry in manifest.inputs if entry.input_id == contrast_input_id]
        if manifest_valid and contrast_input_id is not None
        else []
    )
    contrast_manifest_bound = False
    if len(contrast_matching) != 1:
        reasons.append(
            "contrast_input_manifest_missing"
            if not contrast_matching
            else "contrast_input_manifest_ambiguous"
        )
    else:
        contrast_entry = contrast_matching[0]
        if contrast_entry.semantic_role != expected_contrast_role:
            reasons.append("contrast_input_role_mismatch")
        if (
            digest_encoding == MATRIX_DIGEST_F64LE_V1
            and contrast_entry.schema_version != "safe-slate-matrix-f64le-v1"
        ):
            reasons.append("contrast_input_digest_encoding_mismatch")
        if contrast_digest is None or contrast_entry.sha256 != contrast_digest:
            reasons.append("contrast_input_digest_mismatch")
        if contrast_entry.temporal_scope not in {"feature", "atemporal"}:
            reasons.append("contrast_input_temporal_scope_invalid")
        contrast_time = _aware_timestamp(contrast_entry.snapshot_time)
        if event_time is not None and (
            contrast_time is None or contrast_time > event_time
        ):
            reasons.append("contrast_input_after_request")
        if (
            proposal_log.model_version is not None
            and contrast_entry.version != proposal_log.model_version
        ):
            reasons.append("contrast_input_model_version_mismatch")
        contrast_manifest_bound = not any(
            reason
            in {
                "contrast_input_role_mismatch",
                "contrast_input_digest_encoding_mismatch",
                "contrast_input_digest_mismatch",
                "contrast_input_temporal_scope_invalid",
                "contrast_input_after_request",
                "contrast_input_model_version_mismatch",
            }
            for reason in reasons
        )

    snapshot_observed = False
    contrast_observed = False
    if not observed_inputs.valid:
        reasons.append("input_observation_invalid")
    elif observed_inputs.manifest_digest != manifest_digest:
        reasons.append("input_observation_manifest_mismatch")
    else:
        if snapshot_input_id not in observed_inputs.verified_input_ids:
            reasons.append("uncertainty_snapshot_not_observed")
        else:
            snapshot_observed = True
        if contrast_input_id not in observed_inputs.verified_input_ids:
            reasons.append("contrast_input_not_observed")
        else:
            contrast_observed = True

    unique_reasons = tuple(dict.fromkeys(reasons))
    return LinearUncertaintySnapshotBinding(
        request_id=proposal_log.request_id,
        valid=(
            not unique_reasons
            and snapshot_manifest_bound
            and snapshot_observed
            and contrast_manifest_bound
            and contrast_observed
            and model_version_bound
            and contrast_digest_recorded
        ),
        manifest_digest=manifest_digest,
        snapshot_input_id=snapshot_input_id,
        contrast_input_id=contrast_input_id,
        parameter_covariance_sha256=parameter_digest,
        contrast_features_sha256=contrast_digest,
        matrix_digest_encoding=digest_encoding,
        snapshot_manifest_bound=snapshot_manifest_bound,
        snapshot_observed=snapshot_observed,
        contrast_manifest_bound=contrast_manifest_bound,
        contrast_observed=contrast_observed,
        model_version_bound=model_version_bound,
        contrast_digest_recorded=contrast_digest_recorded,
        reasons=unique_reasons,
    )


# Clearer names for the expanded two-input contract. Keep the original names
# as compatibility aliases for the covariance-only API introduced in 0.2.x.
LinearUncertaintyInputBinding = LinearUncertaintySnapshotBinding
audit_linear_uncertainty_input_binding = audit_linear_uncertainty_snapshot_binding
