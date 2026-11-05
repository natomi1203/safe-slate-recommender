"""Immutable identities for every data, model, query, and code input."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence
from urllib.parse import unquote, urlparse


TemporalScope = Literal["feature", "label", "atemporal"]

# Canonical semantic roles shared by the external freeze CLI and OPE preflight.
# Historical public external manifests already use these names.
OPE_REQUIRED_INPUT_ROLES = (
    "candidate_action_log",
    "propensity_or_interval",
    "settled_purchase_labels",
    "pseudonymous_user_blocks",
    "population_positivity_evidence",
)


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


def _nonempty(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _sha256(value: str, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _validate_uri(value: str) -> str:
    uri = _nonempty(value, "canonical_uri")
    parsed = urlparse(uri)
    if not parsed.scheme:
        raise ValueError("canonical_uri must include an explicit scheme")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(
            "canonical_uri may not contain credentials, query parameters, or fragments"
        )
    if parsed.scheme == "file" and (
        parsed.netloc not in {"", "localhost"} or not parsed.path.startswith("/")
    ):
        raise ValueError("file URI must contain an absolute local path")
    return uri


@dataclass(frozen=True, slots=True)
class ImmutableInput:
    input_id: str
    semantic_role: str
    canonical_uri: str
    version: str
    generation: str | None
    metageneration: str | None
    byte_size: int
    sha256: str
    snapshot_time: str
    temporal_scope: str
    schema_version: str
    contains_credentials: bool
    contains_direct_identifiers: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ImmutableInputManifest:
    schema_version: str
    manifest_id: str
    population_identity: str
    created_at: str
    feature_cutoff: str
    label_cutoff: str
    required_roles: tuple[str, ...]
    inputs: tuple[ImmutableInput, ...]
    manifest_digest: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def payload_dict(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload.pop("manifest_digest")
        return payload

    def verify_digest(self) -> bool:
        return self.manifest_digest == _manifest_digest(self.payload_dict())


@dataclass(frozen=True, slots=True)
class InputManifestVerification:
    valid: bool
    manifest_digest: str | None
    local_inputs_verified: int
    external_inputs_declared: int
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ObservedInputIdentity:
    """Storage identity observed immediately before an evaluation read."""

    input_id: str
    canonical_uri: str
    version: str
    generation: str | None
    metageneration: str | None
    byte_size: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ObservedInputVerification:
    """Exact comparison between a frozen manifest and observed resources."""

    valid: bool
    manifest_digest: str | None
    expected_input_count: int
    observed_input_count: int
    verified_input_ids: tuple[str, ...]
    missing_input_ids: tuple[str, ...]
    unexpected_input_ids: tuple[str, ...]
    mismatched_input_ids: tuple[str, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class InputObservationBatch:
    """Backend-independent observation attempt and its exact audit result."""

    valid: bool
    manifest_digest: str | None
    observations: tuple[ObservedInputIdentity, ...]
    verification: ObservedInputVerification
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def create_immutable_input(
    *,
    input_id: str,
    semantic_role: str,
    canonical_uri: str,
    version: str,
    byte_size: int,
    sha256_digest: str,
    snapshot_time: str,
    temporal_scope: TemporalScope,
    schema_version: str,
    generation: str | None = None,
    metageneration: str | None = None,
    contains_credentials: bool = False,
    contains_direct_identifiers: bool = False,
) -> ImmutableInput:
    """Validate one input identity without reading or copying its contents."""

    if temporal_scope not in {"feature", "label", "atemporal"}:
        raise ValueError("temporal_scope must be feature, label, or atemporal")
    if not isinstance(byte_size, int) or isinstance(byte_size, bool) or byte_size < 0:
        raise ValueError("byte_size must be a nonnegative integer")
    if not isinstance(contains_credentials, bool) or not isinstance(
        contains_direct_identifiers, bool
    ):
        raise ValueError("input safety flags must be boolean")
    if contains_credentials:
        raise ValueError("input manifests may not contain credentials")
    if contains_direct_identifiers:
        raise ValueError("input manifests may not contain direct identifiers")
    if generation is not None:
        generation = _nonempty(generation, "generation")
    if metageneration is not None:
        metageneration = _nonempty(metageneration, "metageneration")
    _parse_aware_timestamp(snapshot_time, "snapshot_time")
    return ImmutableInput(
        input_id=_nonempty(input_id, "input_id"),
        semantic_role=_nonempty(semantic_role, "semantic_role"),
        canonical_uri=_validate_uri(canonical_uri),
        version=_nonempty(version, "version"),
        generation=generation,
        metageneration=metageneration,
        byte_size=byte_size,
        sha256=_sha256(sha256_digest, "sha256_digest"),
        snapshot_time=snapshot_time,
        temporal_scope=temporal_scope,
        schema_version=_nonempty(schema_version, "schema_version"),
        contains_credentials=False,
        contains_direct_identifiers=False,
    )


def create_observed_input_identity(
    *,
    input_id: str,
    canonical_uri: str,
    version: str,
    byte_size: int,
    sha256_digest: str,
    generation: str | None = None,
    metageneration: str | None = None,
) -> ObservedInputIdentity:
    """Validate identity metadata returned by a storage or registry adapter."""

    if not isinstance(byte_size, int) or isinstance(byte_size, bool) or byte_size < 0:
        raise ValueError("byte_size must be a nonnegative integer")
    if generation is not None:
        generation = _nonempty(generation, "generation")
    if metageneration is not None:
        metageneration = _nonempty(metageneration, "metageneration")
    return ObservedInputIdentity(
        input_id=_nonempty(input_id, "input_id"),
        canonical_uri=_validate_uri(canonical_uri),
        version=_nonempty(version, "version"),
        generation=generation,
        metageneration=metageneration,
        byte_size=byte_size,
        sha256=_sha256(sha256_digest, "sha256_digest"),
    )


def _hash_local_file(path: str | Path) -> tuple[Path, int, str]:
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("local immutable input must be a regular file")
    digest = sha256()
    byte_size = 0
    with resolved.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            byte_size += len(chunk)
    return resolved, byte_size, digest.hexdigest()


def observed_input_from_local_file(
    path: str | Path,
    *,
    input_id: str,
    version: str,
    generation: str | None = None,
    metageneration: str | None = None,
) -> ObservedInputIdentity:
    """Re-hash one local file into a storage observation for preflight."""

    resolved, byte_size, digest = _hash_local_file(path)
    return create_observed_input_identity(
        input_id=input_id,
        canonical_uri=resolved.as_uri(),
        version=version,
        generation=generation,
        metageneration=metageneration,
        byte_size=byte_size,
        sha256_digest=digest,
    )


def observe_manifest_inputs(
    manifest: ImmutableInputManifest,
    observers: Mapping[str, Callable[[ImmutableInput], ObservedInputIdentity]]
    | None = None,
) -> InputObservationBatch:
    """Observe every manifest input through URI-scheme adapters.

    ``file:`` inputs use the built-in re-hashing adapter. Other schemes must
    have an explicitly supplied read-only observer. Adapter exception messages
    are not copied into results because they may contain credentials or paths.
    """

    if not isinstance(manifest, ImmutableInputManifest) or not manifest.verify_digest():
        verification = audit_input_manifest_observations(manifest, ())
        return InputObservationBatch(
            valid=False,
            manifest_digest=None,
            observations=(),
            verification=verification,
            reasons=verification.reasons,
        )
    if observers is None:
        observers = {}
    if not isinstance(observers, Mapping):
        raise ValueError("observers must map URI schemes to callables")
    normalized_observers: dict[
        str, Callable[[ImmutableInput], ObservedInputIdentity]
    ] = {}
    for scheme, observer in observers.items():
        normalized_scheme = _nonempty(scheme, "observer scheme").lower()
        if normalized_scheme == "file":
            raise ValueError("the built-in file observer may not be overridden")
        if not callable(observer):
            raise ValueError(f"observer must be callable: {normalized_scheme}")
        normalized_observers[normalized_scheme] = observer

    observations: list[ObservedInputIdentity] = []
    adapter_reasons: list[str] = []
    for entry in manifest.inputs:
        scheme = urlparse(entry.canonical_uri).scheme.lower()
        if scheme == "file":
            parsed = urlparse(entry.canonical_uri)
            path = Path(unquote(parsed.path))
            try:
                observation = observed_input_from_local_file(
                    path,
                    input_id=entry.input_id,
                    version=entry.version,
                    generation=entry.generation,
                    metageneration=entry.metageneration,
                )
            except (OSError, ValueError):
                adapter_reasons.append(f"input_observer_failed:{entry.input_id}:file")
                continue
        else:
            observer = normalized_observers.get(scheme)
            if observer is None:
                adapter_reasons.append(
                    f"input_observer_missing:{entry.input_id}:{scheme}"
                )
                continue
            try:
                observation = observer(entry)
            except Exception as error:
                adapter_reasons.append(
                    f"input_observer_failed:{entry.input_id}:{type(error).__name__}"
                )
                continue
        if not isinstance(observation, ObservedInputIdentity):
            adapter_reasons.append(
                f"input_observer_invalid_result:{entry.input_id}:{scheme}"
            )
            continue
        observations.append(observation)

    verification = audit_input_manifest_observations(manifest, observations)
    reasons = tuple((*adapter_reasons, *verification.reasons))
    return InputObservationBatch(
        valid=not reasons and verification.valid,
        manifest_digest=manifest.manifest_digest,
        observations=tuple(observations),
        verification=verification,
        reasons=reasons,
    )


def audit_input_manifest_observations(
    manifest: ImmutableInputManifest,
    observations: Sequence[ObservedInputIdentity],
) -> ObservedInputVerification:
    """Require exact, complete storage observations for every frozen input."""

    if not isinstance(manifest, ImmutableInputManifest) or not manifest.verify_digest():
        return ObservedInputVerification(
            valid=False,
            manifest_digest=None,
            expected_input_count=0,
            observed_input_count=0,
            verified_input_ids=(),
            missing_input_ids=(),
            unexpected_input_ids=(),
            mismatched_input_ids=(),
            reasons=("manifest_invalid_or_tampered",),
        )
    observed = tuple(observations)
    if any(not isinstance(value, ObservedInputIdentity) for value in observed):
        raise ValueError("observations must contain ObservedInputIdentity entries")
    reasons: list[str] = []
    by_id: dict[str, ObservedInputIdentity] = {}
    duplicate_ids: set[str] = set()
    for value in observed:
        if value.input_id in by_id:
            duplicate_ids.add(value.input_id)
        else:
            by_id[value.input_id] = value
    for input_id in sorted(duplicate_ids):
        reasons.append(f"duplicate_input_observation:{input_id}")

    expected = {value.input_id: value for value in manifest.inputs}
    missing = tuple(sorted(set(expected) - set(by_id)))
    unexpected = tuple(sorted(set(by_id) - set(expected)))
    reasons.extend(f"missing_input_observation:{input_id}" for input_id in missing)
    reasons.extend(
        f"unexpected_input_observation:{input_id}" for input_id in unexpected
    )

    verified: list[str] = []
    mismatched: list[str] = []
    identity_fields = (
        "canonical_uri",
        "version",
        "generation",
        "metageneration",
        "byte_size",
        "sha256",
    )
    for input_id in sorted(set(expected) & set(by_id)):
        if input_id in duplicate_ids:
            mismatched.append(input_id)
            continue
        frozen = expected[input_id]
        actual = by_id[input_id]
        changed_fields = [
            field
            for field in identity_fields
            if getattr(frozen, field) != getattr(actual, field)
        ]
        if changed_fields:
            mismatched.append(input_id)
            reasons.extend(
                f"input_identity_mismatch:{input_id}:{field}"
                for field in changed_fields
            )
        else:
            verified.append(input_id)

    mismatched_ids = tuple(sorted(set(mismatched)))
    return ObservedInputVerification(
        valid=(
            not reasons
            and len(verified) == len(manifest.inputs)
            and len(observed) == len(manifest.inputs)
        ),
        manifest_digest=manifest.manifest_digest,
        expected_input_count=len(manifest.inputs),
        observed_input_count=len(observed),
        verified_input_ids=tuple(verified),
        missing_input_ids=missing,
        unexpected_input_ids=unexpected,
        mismatched_input_ids=mismatched_ids,
        reasons=tuple(reasons),
    )


def immutable_input_from_local_file(
    path: str | Path,
    *,
    input_id: str,
    semantic_role: str,
    version: str,
    snapshot_time: str,
    temporal_scope: TemporalScope,
    schema_version: str,
) -> ImmutableInput:
    """Hash a local file without embedding its contents in the manifest."""

    resolved, byte_size, digest = _hash_local_file(path)
    return create_immutable_input(
        input_id=input_id,
        semantic_role=semantic_role,
        canonical_uri=resolved.as_uri(),
        version=version,
        byte_size=byte_size,
        sha256_digest=digest,
        snapshot_time=snapshot_time,
        temporal_scope=temporal_scope,
        schema_version=schema_version,
    )


def _manifest_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def create_input_manifest(
    *,
    manifest_id: str,
    population_identity: str,
    created_at: str,
    feature_cutoff: str,
    label_cutoff: str,
    required_roles: Sequence[str],
    inputs: Sequence[ImmutableInput],
    schema_version: str = "1.0.0",
) -> ImmutableInputManifest:
    """Freeze input identities and enforce feature/label time boundaries."""

    created = _parse_aware_timestamp(created_at, "created_at")
    feature_boundary = _parse_aware_timestamp(feature_cutoff, "feature_cutoff")
    label_boundary = _parse_aware_timestamp(label_cutoff, "label_cutoff")
    if feature_boundary > label_boundary or created < label_boundary:
        raise ValueError("feature, label, and manifest times are inconsistent")
    roles = tuple(_nonempty(value, "required_roles") for value in required_roles)
    if not roles or len(set(roles)) != len(roles):
        raise ValueError("required_roles must be non-empty and unique")
    entries = tuple(inputs)
    if not entries or any(not isinstance(value, ImmutableInput) for value in entries):
        raise ValueError("inputs must contain ImmutableInput entries")
    input_ids = [entry.input_id for entry in entries]
    if len(set(input_ids)) != len(input_ids):
        raise ValueError("input_id values must be unique")
    resource_identities = [
        (entry.canonical_uri, entry.version, entry.generation, entry.metageneration)
        for entry in entries
    ]
    if len(set(resource_identities)) != len(resource_identities):
        raise ValueError("immutable resource identities must be unique")
    observed_roles = {entry.semantic_role for entry in entries}
    missing_roles = sorted(set(roles) - observed_roles)
    if missing_roles:
        raise ValueError(
            "required input roles are missing: " + ", ".join(missing_roles)
        )
    for entry in entries:
        snapshot = _parse_aware_timestamp(
            entry.snapshot_time, f"inputs.{entry.input_id}.snapshot_time"
        )
        if entry.temporal_scope == "feature" and snapshot > feature_boundary:
            raise ValueError(f"feature input crosses cutoff: {entry.input_id}")
        if entry.temporal_scope == "label" and (
            snapshot < feature_boundary or snapshot > label_boundary
        ):
            raise ValueError(
                f"label input crosses settlement boundary: {entry.input_id}"
            )
        if entry.contains_credentials or entry.contains_direct_identifiers:
            raise ValueError(f"input safety flags violated: {entry.input_id}")
    ordered = tuple(sorted(entries, key=lambda value: value.input_id))
    payload = {
        "schema_version": _nonempty(schema_version, "schema_version"),
        "manifest_id": _nonempty(manifest_id, "manifest_id"),
        "population_identity": _nonempty(population_identity, "population_identity"),
        "created_at": created_at,
        "feature_cutoff": feature_cutoff,
        "label_cutoff": label_cutoff,
        "required_roles": roles,
        "inputs": tuple(entry.to_dict() for entry in ordered),
    }
    digest = _manifest_digest(payload)
    return ImmutableInputManifest(
        schema_version=payload["schema_version"],
        manifest_id=payload["manifest_id"],
        population_identity=payload["population_identity"],
        created_at=created_at,
        feature_cutoff=feature_cutoff,
        label_cutoff=label_cutoff,
        required_roles=roles,
        inputs=ordered,
        manifest_digest=digest,
    )


def load_input_manifest(payload: Mapping[str, Any]) -> ImmutableInputManifest:
    """Load and semantically revalidate a serialized immutable manifest."""

    if not isinstance(payload, Mapping):
        raise ValueError("manifest payload must be a mapping")
    expected = set(ImmutableInputManifest.__dataclass_fields__)
    if set(payload) != expected:
        raise ValueError("input manifest fields do not match canonical schema")
    entries = tuple(
        create_immutable_input(
            input_id=value["input_id"],
            semantic_role=value["semantic_role"],
            canonical_uri=value["canonical_uri"],
            version=value["version"],
            generation=value.get("generation"),
            metageneration=value.get("metageneration"),
            byte_size=value["byte_size"],
            sha256_digest=value["sha256"],
            snapshot_time=value["snapshot_time"],
            temporal_scope=value["temporal_scope"],
            schema_version=value["schema_version"],
            contains_credentials=value["contains_credentials"],
            contains_direct_identifiers=value["contains_direct_identifiers"],
        )
        for value in payload["inputs"]
    )
    recreated = create_input_manifest(
        schema_version=payload["schema_version"],
        manifest_id=payload["manifest_id"],
        population_identity=payload["population_identity"],
        created_at=payload["created_at"],
        feature_cutoff=payload["feature_cutoff"],
        label_cutoff=payload["label_cutoff"],
        required_roles=tuple(payload["required_roles"]),
        inputs=entries,
    )
    if recreated.manifest_digest != payload["manifest_digest"]:
        raise ValueError("input manifest digest mismatch")
    return recreated


def verify_local_manifest_inputs(
    manifest: ImmutableInputManifest,
) -> InputManifestVerification:
    """Re-hash local file inputs while leaving external identities explicit."""

    reasons: list[str] = []
    local_count = 0
    external_count = 0
    if not isinstance(manifest, ImmutableInputManifest) or not manifest.verify_digest():
        return InputManifestVerification(
            valid=False,
            manifest_digest=None,
            local_inputs_verified=0,
            external_inputs_declared=0,
            reasons=("manifest_invalid_or_tampered",),
        )
    for entry in manifest.inputs:
        parsed = urlparse(entry.canonical_uri)
        if parsed.scheme != "file":
            external_count += 1
            continue
        local_count += 1
        path = Path(unquote(parsed.path))
        try:
            stat = path.stat()
            if not path.is_file() or stat.st_size != entry.byte_size:
                reasons.append(f"local_input_size_mismatch:{entry.input_id}")
                continue
            digest = sha256()
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
            if digest.hexdigest() != entry.sha256:
                reasons.append(f"local_input_digest_mismatch:{entry.input_id}")
        except OSError:
            reasons.append(f"local_input_unreadable:{entry.input_id}")
    return InputManifestVerification(
        valid=not reasons,
        manifest_digest=manifest.manifest_digest,
        local_inputs_verified=local_count,
        external_inputs_declared=external_count,
        reasons=tuple(reasons),
    )
