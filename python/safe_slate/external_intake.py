"""No-clobber intake of privacy-reviewed external evaluation inputs."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import struct
from typing import Any, Mapping

from .input_manifest import (
    ImmutableInputManifest,
    OPE_REQUIRED_INPUT_ROLES,
    create_input_manifest,
    immutable_input_from_local_file,
)
from .log_io import load_proposal_logs_jsonl
from .population_positivity import (
    POSITIVITY_EVIDENCE_SCHEMA_VERSIONS,
    load_population_positivity_evidence_file,
)
from .service_trace import (
    join_service_trace_completions,
    load_service_trace_completions_jsonl,
)
from .uncertainty_utils import canonical_matrix_bytes


EXTERNAL_INPUT_ROLES = (
    *OPE_REQUIRED_INPUT_ROLES,
    "production_constraint_trace",
    "shadow_record",
    "service_trace_completion",
    "linear_parameter_covariance",
    "request_contrast_features",
)
SPEC_KEYS = {
    "schema_version",
    "manifest_id",
    "population_identity",
    "created_at",
    "feature_cutoff",
    "label_cutoff",
    "privacy_reviewed",
    "synthetic_inputs",
    "maximum_input_bytes",
    "maximum_log_records",
    "inputs",
}
INPUT_KEYS = {
    "input_id",
    "semantic_role",
    "path",
    "version",
    "snapshot_time",
    "temporal_scope",
    "schema_version",
}
F64LE_MAGIC = b"safe-slate-matrix-f64le-v1\x00"


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value is forbidden: {value}")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_intake_spec(path: str | Path) -> dict[str, Any]:
    """Load one strict JSON intake specification without reading data rows."""

    spec = json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_pairs,
        parse_constant=_reject_constant,
    )
    if not isinstance(spec, dict):
        raise ValueError("intake specification must be one JSON object")
    return spec


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{name} keys differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _validate_matrix_file(path: object, schema_version: object) -> None:
    matrix_path = Path(path)  # type: ignore[arg-type]
    if schema_version == "safe-slate-matrix-json-v1":
        raw = matrix_path.read_bytes()
        matrix = json.loads(
            raw,
            parse_constant=_reject_constant,
        )
        if canonical_matrix_bytes(matrix) != raw:
            raise ValueError("matrix JSON input is not in canonical json-v1 encoding")
        return
    if schema_version != "safe-slate-matrix-f64le-v1":
        raise ValueError("unsupported matrix schema_version")
    with matrix_path.open("rb") as stream:
        if stream.read(len(F64LE_MAGIC)) != F64LE_MAGIC:
            raise ValueError("matrix f64le input has an invalid header")
        shape = stream.read(16)
        if len(shape) != 16:
            raise ValueError("matrix f64le input is missing its shape")
        rows, columns = struct.unpack("<QQ", shape)
        expected_bytes = len(F64LE_MAGIC) + 16 + rows * columns * 8
        if matrix_path.stat().st_size != expected_bytes:
            raise ValueError("matrix f64le byte size does not match its shape")
        for _ in range(rows * columns):
            value_bytes = stream.read(8)
            if len(value_bytes) != 8 or not isfinite(struct.unpack("<d", value_bytes)[0]):
                raise ValueError("matrix f64le input contains a non-finite value")


def freeze_external_input_manifest(spec: Mapping[str, Any]) -> ImmutableInputManifest:
    """Hash all ten required files into one immutable private manifest.

    This function validates identities and time boundaries only. It does not
    inspect row-level schema, prove privacy, establish propensity support, or
    authorize OPE, shadow claims, serving changes, or production rollout.
    """

    if not isinstance(spec, Mapping):
        raise ValueError("spec must be a mapping")
    _exact_keys(spec, SPEC_KEYS, "intake specification")
    if spec["schema_version"] != "safe-slate-external-intake-1.0":
        raise ValueError("unsupported intake schema_version")
    if spec["privacy_reviewed"] is not True:
        raise ValueError("privacy_reviewed must be explicitly true")
    if spec["synthetic_inputs"] is not False:
        raise ValueError("synthetic_inputs must be explicitly false")
    maximum_input_bytes = spec["maximum_input_bytes"]
    maximum_log_records = spec["maximum_log_records"]
    for name, value in (
        ("maximum_input_bytes", maximum_input_bytes),
        ("maximum_log_records", maximum_log_records),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    inputs_raw = spec["inputs"]
    if not isinstance(inputs_raw, list):
        raise ValueError("inputs must be a list")

    entries = []
    observed_roles: list[str] = []
    primary_logs = None
    service_completions = None
    for index, raw in enumerate(inputs_raw):
        if not isinstance(raw, Mapping):
            raise ValueError(f"inputs[{index}] must be an object")
        _exact_keys(raw, INPUT_KEYS, f"inputs[{index}]")
        role = raw["semantic_role"]
        if role not in EXTERNAL_INPUT_ROLES:
            raise ValueError(f"inputs[{index}] has an unknown semantic_role")
        observed_roles.append(role)
        path = Path(raw["path"])
        if path.stat().st_size > maximum_input_bytes:
            raise ValueError(f"{role} exceeds maximum_input_bytes")
        if role == "candidate_action_log":
            if raw["schema_version"] != "safe-slate-proposal-log-jsonl-1.3":
                raise ValueError("candidate_action_log schema_version is invalid")
            primary_logs = load_proposal_logs_jsonl(
                path,
                maximum_records=maximum_log_records,
                maximum_bytes=maximum_input_bytes,
            )
            if any(log.service_latency_trace is not None for log in primary_logs):
                raise ValueError(
                    "primary proposal logs must not embed post-commit service traces"
                )
        if role == "service_trace_completion":
            if (
                raw["schema_version"]
                != "safe-slate-service-trace-completion-jsonl-1.0"
            ):
                raise ValueError("service_trace_completion schema_version is invalid")
            service_completions = load_service_trace_completions_jsonl(
                path,
                maximum_records=maximum_log_records,
                maximum_bytes=maximum_input_bytes,
            )
        if role in {"linear_parameter_covariance", "request_contrast_features"}:
            if raw["schema_version"] not in {
                "safe-slate-matrix-f64le-v1",
                "safe-slate-matrix-json-v1",
            }:
                raise ValueError(
                    f"{role} requires an explicitly versioned matrix encoding"
                )
            _validate_matrix_file(raw["path"], raw["schema_version"])
        if role == "population_positivity_evidence":
            if raw["schema_version"] not in POSITIVITY_EVIDENCE_SCHEMA_VERSIONS:
                raise ValueError(
                    "population_positivity_evidence schema_version is invalid"
                )
            positivity = load_population_positivity_evidence_file(path)
            if raw["schema_version"] != positivity.schema_version:
                raise ValueError(
                    "population_positivity_evidence declared schema does not match bytes"
                )
            if raw["version"] != positivity.evidence_id:
                raise ValueError(
                    "population_positivity_evidence version must equal evidence_id"
                )
            if positivity.population_identity != spec["population_identity"]:
                raise ValueError(
                    "population positivity evidence population identity mismatch"
                )
            if raw["snapshot_time"] != positivity.audited_at:
                raise ValueError(
                    "population positivity evidence snapshot_time must equal audited_at"
                )
            if raw["temporal_scope"] != "atemporal":
                raise ValueError(
                    "population positivity evidence must use atemporal scope"
                )
        entries.append(
            immutable_input_from_local_file(
                raw["path"],
                input_id=raw["input_id"],
                semantic_role=role,
                version=raw["version"],
                snapshot_time=raw["snapshot_time"],
                temporal_scope=raw["temporal_scope"],
                schema_version=raw["schema_version"],
            )
        )
    if tuple(sorted(observed_roles)) != tuple(sorted(EXTERNAL_INPUT_ROLES)):
        missing = sorted(set(EXTERNAL_INPUT_ROLES) - set(observed_roles))
        extra_or_duplicate = sorted(
            role for role in set(observed_roles) if observed_roles.count(role) != 1
        )
        raise ValueError(
            "external input roles differ; "
            f"missing={missing}, duplicate={extra_or_duplicate}"
        )
    # Exact role equality above plus successful role-specific loaders guarantees
    # both values; retain this as defense against future role-table changes.
    if primary_logs is None or service_completions is None:  # pragma: no cover
        raise ValueError("primary production logs and service traces are required")
    join_service_trace_completions(primary_logs, service_completions)

    return create_input_manifest(
        manifest_id=spec["manifest_id"],
        population_identity=spec["population_identity"],
        created_at=spec["created_at"],
        feature_cutoff=spec["feature_cutoff"],
        label_cutoff=spec["label_cutoff"],
        required_roles=EXTERNAL_INPUT_ROLES,
        inputs=entries,
        schema_version="safe-slate-external-input-manifest-1.0",
    )


def write_external_input_manifest(
    spec_path: str | Path,
    *,
    output: str | Path,
    completion_marker: str | Path,
) -> ImmutableInputManifest:
    """Freeze one manifest and write its completion marker last."""

    output_path = Path(output)
    marker_path = Path(completion_marker)
    if output_path.exists() or marker_path.exists():
        raise FileExistsError("refusing to overwrite external input evidence")
    if not output_path.parent.is_dir() or marker_path.parent != output_path.parent:
        raise ValueError("output parent must exist and contain both outputs")
    manifest = freeze_external_input_manifest(load_intake_spec(spec_path))
    encoded = (
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    with output_path.open("xb") as stream:
        stream.write(encoded)
    marker = {
        "artifact": output_path.name,
        "byte_size": len(encoded),
        "completion_state": "external_inputs_frozen_private_audit_pending",
        "input_count": len(manifest.inputs),
        "manifest_digest": manifest.manifest_digest,
        "sha256": sha256(encoded).hexdigest(),
        "authorized_next_action": "run private row-schema and support audits",
        "production_rollout_authorized": False,
    }
    with marker_path.open("x", encoding="utf-8") as stream:
        json.dump(marker, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--completion-marker", required=True, type=Path)
    args = parser.parse_args()
    manifest = write_external_input_manifest(
        args.spec,
        output=args.output,
        completion_marker=args.completion_marker,
    )
    print(
        "Frozen ten private input identities; support/OPE/shadow audits remain pending. "
        f"manifest_digest={manifest.manifest_digest}"
    )


if __name__ == "__main__":
    main()
