# Changelog

## 0.3.0

- Qualified five platform wheels and one sdist on TestPyPI through a protected
  GitHub OIDC deployment, with manifest-bound SHA-256 verification and clean
  Python 3.15 reinstall smoke tests on every supported runner.
- Licensed the reusable software distribution under Apache-2.0 while retaining
  CC-BY-4.0 for repository-level research and publication artifacts.
- Added a documented seven-symbol stable API boundary, promoting the measured
  serving-gate assessor and candidate-pool facade alongside the existing core,
  plus a beginner quickstart, compatibility and wheel policy, research-
  integration classification, security guidance, and a no-authorization
  release checklist.
- Hardened production input validation so coercible strings, booleans used as
  integers, non-finite values, negative variances, and non-boolean raw serving
  gates cannot cross a fail-closed boundary.
- Added deterministic randomized oracle invariants and manifest-backed public
  API contract tests, plus locked Rust tests and warning-free Clippy to the
  software CI contract. The full Python package passes mypy, and native stubs
  are checked against the installed extension with `mypy.stubtest`.
- Included the package-local license in built distributions and expanded PyPI
  metadata for supported audiences, platforms, documentation, and issue links.
- Added strict bounded intake for the production Go service's
  `safe-slate-production-post-response-1.1` sidecar. It verifies nanosecond
  chronology and payload sizes, rejects duplicate identities and schema drift,
  and classifies the process-local gRPC envelope as partial observability
  rather than fabricating a full production trace.
- Added digest-bound post-commit service-trace sidecars. They solve the
  circularity of placing a proposal log's future durable-commit time inside
  that same immutable record, require complete candidate-generation,
  inference, queueing, RPC, guardrail, serialization, and log-write timings,
  and join one-to-one by request id, event time, W3C trace id, and primary-log
  SHA-256.
- Added a request-scoped monotonic `ServiceTraceRecorder` that measures local
  stages, accepts externally measured queue/RPC spans, and cannot finalize
  before all registered stages, response completion, and the primary durable
  log commit are observed.
- Added `safe-slate-freeze-inputs`, a strict no-clobber CLI that hashes all
  nine privacy-reviewed external evaluation roles, including post-commit
  service-trace sidecars, validates temporal and
  canonical matrix-encoding boundaries, writes no row data, and leaves
  support/OPE/shadow/rollout authorization explicitly pending.
- Added a no-clobber candidate-pool benchmark that measures deterministic
  operation construction, factorized uncertainty preparation, Rust search,
  fallback, deadlines, and final rechecks separately, including the
  600-item/560-protected synthetic serving shape without presenting it as
  production latency evidence.
- Added the versioned, shape-bound `f64le-v1` matrix identity for the
  latency-sensitive candidate-pool path, while retaining `json-v1` as the
  lower-level compatibility default and recording the encoding in every
  factorized proposal log.
- Combined f64-le projection and both matrix digests in one Rust crossing and
  reserved four milliseconds by default for synchronous candidate-pool result
  materialization and logging, avoiding the observed 10 ms all-baseline
  collapse without weakening final rechecks.
- Added `guarded_candidate_pool_rerank`, a request-level home-feed/search
  facade that accepts baseline and insertion-candidate arrays, compiles the
  replaceable suffix into balanced operations, preserves the protected core,
  and delegates request-specific cardinality and reward optimization to the
  Rust certified oracle.
- Added deterministic add-minus-remove reward, safety, and feature-contrast
  construction with native candidate conflict and partition-cap compilation.
  Duplicate baseline/candidate ids, non-finite values, excessive operation or
  conflict/feature expansion, and malformed feature shapes fail before search.
- Included candidate-pool preprocessing in the cooperative request budget and
  recorded the full budget handoff in proposal-log context. Runtime evidence
  gates remain fail-closed by default, and shadow mode never changes the
  served baseline.
- Added Rust-side projection of shared linear-model uncertainty,
  `Z parameter_covariance Z^T`, with finite-value, symmetry, shape, and
  positive-semidefinite validation.
- Added `guarded_linear_rerank`, which retains cross-operation correlation,
  references an immutable uncertainty snapshot, digests the projection inputs
  into the proposal log, and fails closed unless runtime gates are explicitly
  passed. Its cooperative timeout subtracts projection and digest time before
  assigning the remaining whole-millisecond budget to Rust search.
- Kept the linear serving path factorized as contrast rows and transformed
  contrast rows, avoiding quadratic operation-covariance logs and cubic
  operation-covariance validation while reproducing the same aggregate LCB.
- Added fail-closed binding from factorized proposal logs to an immutable,
  preflight-observed parameter-covariance snapshot, including digest, semantic
  role, model version, and request-time checks. Confirmatory shadow evidence
  also requires each request contrast payload to have a matching immutable
  input id, digest, role, model version, observation, and pre-request time.

## 0.2.0

- Added `guarded_rerank`, a production-facing home-feed/search re-ranking API.
- Added automatic conversion of a baseline metric and degradation tolerance to
  the baseline-relative safety threshold.
- Added fail-closed support, distribution-shift, uncertainty, and
  proposal-log completeness gates.
- Added `GuardedRerankResult` and `ProposalLog`, including selected and
  unselected proposal features for delayed-outcome joins.
- Added `serving_mode="shadow"`, separating the counterfactual proposed slate
  from the unchanged slate actually served.
- Added versioned shadow/serving log fields and eligibility flags that prevent
  baseline outcomes from being used as candidate-action feedback.
- Updated proposal-log schema to 1.2 and made timezone-aware event timestamps
  mandatory for shadow requests, so evaluation windows can be audited.
- Updated proposal-log schema to 1.3 with an optional, immutable service
  latency trace that separates response completion from proposal-log commit.
- Added complete service-stage timing validation and frozen shadow p95 gates;
  valid shadow-service latency evidence remains distinct from a universal
  production-latency claim.
- Shadow summaries now require an explicit half-open evaluation window and
  reject every record outside it.
- Added immutable `ShadowEvaluationContract` creation/loading with frozen log
  schema, window, operability gates, input identity, resource caps, output
  namespace, and explicit claim boundaries.
- Added no-clobber shadow evidence publication and source-recomputed
  verification. Ordered log digests, negative gate outcomes, insufficient
  sample conclusions, and integrity rejections are preserved separately.
- Added strict proposal-log dictionary and bounded JSONL loaders for replaying
  immutable shadow records without silently accepting schema drift, malformed
  arrays, non-finite values, or unbounded local reads.
- Added `assess_serving_gates`, which derives fail-closed support, shift,
  uncertainty, and proposal-completeness decisions from runtime evidence and
  stores the reason codes in proposal-log schema 1.1.
- Added `assess_feedback_coverage`, which separates complete direct feedback,
  request-level OPE prechecks, and descriptive-only logs; detects accepted-only
  feedback; and rejects unsettled, outcome-dependent, unsupported, heavy-tail,
  malformed, or shadow evidence fail-closed.
- Added `audit_ope_readiness`, which now separates 100% finite-log support
  from the additional population-positivity premise required for a
  full-population point estimand and retains 80% as the partial-support floor,
  user-blocked ESS, repeated-user and cross-fit-fold checks, missingness and
  event sufficiency, and raw weight-tail gates before OPE. Known propensities
  can authorize point OPE; covering intervals use a conservative ESS lower
  bound and authorize partial-identification bounds only.
- Added `summarize_shadow_run`, which audits baseline preservation, protected
  cores, posterior-frontier consistency, final rechecks, certificate margins,
  predicate failures, oracle latency, deadlines, and fallback rates over a
  frozen shadow window without authorizing production-latency or causal-lift
  claims.
- Added SHA-256 row-data binding to OPE readiness results and
  `estimate_cross_fitted_policy_value`, with doubly robust primary estimation,
  IPS/SNIPS diagnostics, frozen clipping, deterministic user-cluster bootstrap,
  request-weighted aggregation, and explicit identification attestations.
- Replaced the ambiguous automatic supported-subpopulation interpretation for
  80--<100% average support with `estimate_partial_identification_bounds`. It
  estimates the supported contribution by cross-fitted DR and bounds the
  unsupported target mass by a frozen outcome range, with a user-cluster
  bootstrap envelope for target value and lift.
- Added `estimate_propensity_interval_bounds`, which applies sign-aware
  worst-case importance weights under covering request-time propensity
  intervals, combines them with partial-support outcome bounds, and reports
  propensity and support contributions to identification width separately.
- Added outcome-observation propensities to readiness digests and combined
  action/observation weight gates. Missing outcomes now fail closed without
  these probabilities. `estimate_missingness_adjusted_policy_value` provides
  target and logging-policy DR corrections, user-cluster inference, and
  separate combined-weight and observation-weight clipping diagnostics.
- Added `estimate_joint_uncertainty_bounds` for partial support or propensity
  intervals combined with missing outcomes. It composes observation-weighted
  residuals, sign-aware propensity bounds, unsupported-mass outcome bounds,
  logging-policy DR correction, and user-cluster uncertainty without changing
  the original request-population estimand.
- Added immutable `OpeEvaluationContract` creation/loading with canonical
  SHA-256 identity, disjoint pilot/final windows, label-ready time, pinned input
  and code identities, frozen gates/inference settings, resource caps,
  no-clobber output identity, and explicit authorization boundaries. Contract
  digests now bind readiness and estimator results.
- Added `classify_evaluation_evidence`, keeping collection, settled labels,
  statistical sufficiency, independent final audit, and decision eligibility
  as separate fail-closed states rather than one success boolean.
- Added atomic `write_evidence_bundle` and byte-level
  `verify_evidence_bundle`. Bundles are no-clobber, completion-marker-last,
  resource-capped, contract/row-digest bound, and independently checked for
  exact files, artifact hashes, canonical semantics, execution identities,
  state consistency, safety flags, and authorization boundaries.
- Added canonical `ImmutableInputManifest` support for content-addressed data,
  model, query, and code identities with explicit feature/label cutoffs,
  required semantic roles, credential-safe URIs, and optional local re-hashing.
- Evidence bundles now embed the immutable input manifest and independently
  verify its digest, population, and temporal binding to the evaluation
  contract; a re-hashed but semantically altered manifest remains invalid.
- Added storage-observation identities and exact manifest auditing. Missing,
  extra, duplicate, generation-changed, size-changed, or digest-changed inputs
  fail closed with explicit reason codes.
- Added a local-file observation adapter that re-hashes mounted inputs
  immediately before manifest-bound evaluation.
- Added backend-independent URI-scheme observer dispatch for external object
  stores and registries, with fail-closed missing/invalid/error states and
  credential-safe error reporting.
- Added `audit_manifest_bound_ope_readiness`, which prevents row-level OPE
  auditing until every immutable input, population, final-window cutoff, and
  label-settlement boundary matches the frozen contract.
- Added `run_no_clobber_ope`, a fixed-stage local/backend-worker entrypoint that
  dispatches across all five registered OPE estimators and independently
  verifies the newly published bundle.
- Evidence bundles can now preserve a statistically valid
  `insufficient_evidence` conclusion without fabricating `estimate.json`.
- Added a deterministic production-shape benchmark that separates Python-facade
  and Rust-oracle latency, deadline, fallback, cardinality, and recheck metrics
  without mislabeling the process-local interval as service end-to-end latency.
- Added deadline checks between bounded predicate/materialization chunks,
  preventing a completed beam depth from causing large native timeout
  overshoot. An in-flight Python callback remains non-preemptible.
- Set the serving facade's default cooperative search budget to 10 ms while
  retaining the existing configurable low-level APIs.
- Preserved protected-core, certified-incumbent, posterior-frontier, and final
  Rust recheck behavior.

## 0.1.0

- Initial Rust/PyO3 certified slate and balanced-replacement oracle.
