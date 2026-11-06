# TommySlate

[![CI](https://github.com/natomi1203/tommyslate/actions/workflows/ci.yml/badge.svg)](https://github.com/natomi1203/tommyslate/actions/workflows/ci.yml)
[![Python 3.15](https://img.shields.io/badge/Python-3.15-3776AB.svg?logo=python&logoColor=white)](docs/compatibility.md)
[![Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/tommyslate.svg)](https://pypi.org/project/tommyslate/)

**Every recommendation, checked.**

Keep the part of a feed, search result, or recommendation slate that already
works. Explore better replacements only when they satisfy your registered
safety and business constraints. If evidence is missing, malformed, or fails
its final check, return the unchanged baseline and record why.

TommySlate is a safer slate re-ranker for Python, backed by a Rust search core.
It is built for teams that need more than a high-scoring list: they need a
bounded search, an auditable decision, and a safe fallback.

> **Release status:** TommySlate 0.4.1 is the Beta metadata and repository-name
> alignment candidate. Version 0.4.0 remains the current production release
> until the 0.4.1 artifact gates complete.

## Why use it?

| Your constraint | What the library provides |
| --- | --- |
| Part of the current ranking must never move | Protected-prefix replacement that preserves the registered baseline core |
| Search may stop before exploring every candidate | A rechecked incumbent with an explicit lower/upper cardinality certificate |
| Categories, conflicts, or whole-slate rules matter | Native group caps and conflicts, plus deterministic Python predicates |
| Candidate risks share model uncertainty | Diagonal, full-covariance, and factorized linear uncertainty paths |
| Missing evidence must never become permission | Fail-closed serving gates and exact baseline fallback |
| Decisions must be explainable later | Proposal-dominating logs, shadow evaluation, immutable inputs, and no-clobber OPE evidence |

The result is deliberately conservative: the library validates the decision
contract you give it, but it does not invent calibration, causal lift,
population support, or production authorization.

## Install

Python 3.15 is required. Add TommySlate to a uv-managed project:

```bash
uv add "tommyslate==0.4.1"
```

The stable Python import remains `safe_slate`; application imports do not
change when migrating from 0.3.1. Replace the old distribution rather than
installing both because they provide the same Python module:

```bash
uv remove safe-slate-recommender
uv add "tommyslate==0.4.1"
```

The legacy `safe-slate-freeze-inputs` command remains as an alias for the new
`tommyslate-freeze-inputs` command. The package has no runtime Python
dependencies. For a source build, follow the [quickstart](docs/quickstart.md)
from a trusted checkout.

## One-minute example

Protect the first two items in an existing slate and choose up to two safe
replacements for the remaining positions:

```python
from safe_slate import optimize_replacements

result = optimize_replacements(
    remove_positions=[2, 2, 3, 3],
    add_item_ids=[100, 101, 100, 102],
    reward_deltas=[0.8, 0.7, 0.6, 0.5],
    safety_means=[0.04, 0.03, 0.02, 0.01],
    safety_variances=[0.0001] * 4,
    baseline_size=4,
    protected_count=2,
    max_replacements=2,
    baseline_item_ids=[10, 11, 12, 13],
)

assert result.rechecked
print(result.materialized_slate)  # [10, 11, 100, 102]
print(result.k_lower, result.k_upper, result.status)  # 2 2 exact
```

The first two items remain untouched. The optimizer selects a compatible set
of remove/add operations, maximizes certified replacement count first and
reward second, and rechecks the final slate before returning it.

## Choose your path

- **Try the core optimizer:** follow the [quickstart](docs/quickstart.md).
- **Integrate a guarded re-ranker:** start with
  [`guarded_candidate_pool_rerank`](docs/api-reference.md#guarded_candidate_pool_rerank).
- **Evaluate before rollout:** use shadow mode, OPE readiness, and immutable
  evidence workflows described below.
- **Review the contract:** see the [stable API](docs/api-reference.md),
  [compatibility policy](docs/compatibility.md), and
  [safety boundaries](#safety-and-scope).

## Development

```bash
uv sync --frozen
uv run maturin develop --release
uv run pytest
```

## Production guarded re-ranking

For the simplest home-feed/search integration, use
`guarded_candidate_pool_rerank`. The caller supplies item-level arrays; the
library constructs every balanced replacement between the replaceable
baseline suffix and the insertion-candidate pool. Search, shared-uncertainty
evaluation, dynamic replacement count, timeout fallback, and the mandatory
final recheck remain in the Rust-backed path.

```python
from safe_slate import assess_serving_gates, guarded_candidate_pool_rerank

gates = assess_serving_gates(
    support_rate=0.997,
    distribution_shift_score=0.03,
    uncertainty_evidence_valid=True,
    expected_proposal_count=4,
    logged_proposal_count=4,
)

decision = guarded_candidate_pool_rerank(
    request_id="home-01J...",
    uncertainty_snapshot_id="ranker-v21/covariance/sha256:...",
    contrast_input_id="home-01J.../contrast-features",
    model_version="ranker-v21",
    baseline_item_ids=[10, 11, 12, 13],
    baseline_reward_scores=[0.90, 0.80, 0.20, 0.10],
    baseline_safety_means=[0.12, 0.11, 0.05, 0.04],
    baseline_features=[
        [0.3, 0.1], [0.2, 0.2], [0.1, 0.0], [0.0, 0.1]
    ],
    candidate_item_ids=[9001, 9002],
    candidate_reward_scores=[0.70, 0.60],
    candidate_safety_means=[0.06, 0.05],
    candidate_features=[[0.2, -0.1], [0.1, 0.3]],
    parameter_covariance=[[2e-5, 1e-6], [1e-6, 3e-5]],
    protected_count=2,
    baseline_metric=0.12,
    degradation_tolerance=0.01,
    max_replacements=2,
    gate_assessment=gates,
    timeout_ms=10,
    context={"surface": "home", "country": "JP"},
)

serve(decision.slate)
proposal_log_sink.write(decision.proposal_log.to_dict())
```

`selected_count` is the request-specific certified value of (K_t), not a
fixed exploration quota. The candidate-pool facade uses add-minus-remove
feature contrasts and keeps the shared parameter covariance factorized, so
correlation between replacement operations is retained. Candidate ids must be
new insertions not already present in the baseline. Reordering exactly the
same item set is deliberately rejected because it is a position-assignment
problem rather than balanced set replacement.

The Cartesian expansion is capped by `max_operations`, its materialized
contrast size by `max_feature_cells`, and compiled native conflicts by
`max_compiled_conflicts`; exceeding a cap raises before the request can be
served. Candidate-level exclusions can be supplied as
`candidate_conflicts=[(item_a, item_b)]`, and remove/add category limits as
baseline/candidate group arrays plus cap dictionaries. The 10 ms budget is
cooperative: candidate expansion and uncertainty projection reduce the Rust
search budget, while mandatory final checks may complete after it. Enforce a
hard service deadline outside the process and record measured end-to-end
latency separately.

The candidate-pool facade defaults to the versioned `f64le-v1` matrix digest,
which binds matrix shape and finite IEEE-754 values without JSON formatting on
the request path. External covariance and contrast manifests must hash
`canonical_matrix_f64le_bytes(...)` (or use `matrix_f64le_sha256(...)`) and
use input `schema_version="safe-slate-matrix-f64le-v1"` for both matrices,
while retaining the logged `matrix_digest_encoding`. The lower-level
`guarded_linear_rerank` keeps `json-v1` as its compatibility default. Callers
may select either encoding explicitly, but a manifest digest from one encoding
must never be compared with the other.

Its default `finalization_reserve_ms=4` is larger than the lower-level linear
facade's compatibility default because candidate-pool result materialization
and proposal-dominating logging are also on this synchronous Python path. A
larger reserve may return a shallower certified incumbent; it never relaxes
the final safety recheck.

`guarded_rerank` is the serving facade for a home feed or search re-ranker.
The Rust oracle still performs the search, timeout checks, incumbent retention,
and final safety/compatibility recheck. Python translates the business
degradation tolerance into the relative-safety threshold and emits an audit
record containing every proposed operation.

```python
from safe_slate import guarded_rerank

decision = guarded_rerank(
    request_id="home-01J...",
    model_version="ranker-2026-11-03",
    baseline_item_ids=list(range(600)),
    protected_count=560,
    remove_positions=[560, 560, 561, 562],
    add_item_ids=[9001, 9002, 9001, 9003],
    reward_deltas=[0.08, 0.07, 0.06, 0.05],
    # Expected purchase/CTR changes relative to the unchanged baseline.
    safety_mean_deltas=[0.004, 0.003, 0.002, 0.001],
    safety_variances=[2e-6, 2e-6, 1e-6, 1e-6],
    baseline_metric=0.12,
    degradation_tolerance=0.01,
    max_replacements=40,
    confidence_radius=1.96,
    support_ok=True,
    uncertainty_valid=True,
    distribution_shift_detected=False,
    proposal_log_complete=True,
    timeout_ms=10,
    context={"surface": "home", "country": "JP"},
)

serve(decision.slate)
proposal_log_sink.write(decision.proposal_log.to_dict())
```

When replacement contrasts share one fitted linear model, use
`guarded_linear_rerank`. The Rust projection preserves the cross-operation
correlation in (Z\Sigma Z^\top); treating the operation variances as
independent can otherwise over- or understate the slate-level uncertainty.

```python
from safe_slate import guarded_linear_rerank

decision = guarded_linear_rerank(
    request_id="home-01J...",
    uncertainty_snapshot_id="gs://model-registry/ranker-v17/covariance#sha256=...",
    contrast_input_id="home-01J...-contrast-features",
    model_version="ranker-v17",
    baseline_item_ids=list(range(600)),
    protected_count=560,
    remove_positions=[560, 561],
    add_item_ids=[9001, 9002],
    reward_deltas=[0.08, 0.05],
    safety_mean_deltas=[0.004, 0.003],
    # Each row is the add-minus-remove feature contrast for one operation.
    contrast_features=[[0.2, -0.1], [0.1, 0.3]],
    parameter_covariance=[[2e-5, 1e-6], [1e-6, 3e-5]],
    baseline_metric=0.12,
    degradation_tolerance=0.01,
    max_replacements=40,
    confidence_radius=1.96,
    # The linear facade is fail-closed by default. Supply measured gates or
    # pass an evidence-derived gate_assessment.
    support_ok=True,
    uncertainty_valid=True,
    distribution_shift_detected=False,
    proposal_log_complete=True,
    timeout_ms=10,
)
```

The snapshot identifier and SHA-256 digests of the contrast matrix and
parameter covariance are stored under
`proposal_log.context["safe_slate_uncertainty"]`. The serving path keeps the
uncertainty factorized: it logs operation-level marginal variances and the
selected aggregate certificate, but does not materialize or serialize the
quadratic-size operation covariance. Keep the source feature/covariance
snapshot immutable under the recorded identity and digests. The projection
only propagates supplied uncertainty. It does not prove that the confidence
radius has valid coverage, so an unvalidated or shifted snapshot must fail the
uncertainty/distribution-shift gate. Include projection time in the end-to-end
service trace.
For this linear facade, `timeout_ms` covers the projection/digest work plus the
cooperative Rust search: only the remaining whole milliseconds are passed to
the oracle after a configurable `finalization_reserve_ms` (default 2 ms).
Mandatory final rechecks and non-preemptible Python callbacks can finish later,
so this is not a hard service deadline.

Before treating a factorized shadow record as evaluation evidence, bind its
covariance snapshot to the frozen input manifest and to the storage identity
observed immediately before evaluation:

```python
from safe_slate import audit_linear_uncertainty_input_binding

binding = audit_linear_uncertainty_input_binding(
    decision.proposal_log,
    manifest=frozen_input_manifest,
    observed_inputs=preflight_observation.verification,
)
assert binding.valid
```

The covariance manifest input must use semantic role
`linear_parameter_covariance`, and the request matrix input must use
`request_contrast_features`. Their `input_id` values must match the logged
snapshot and contrast IDs, their `version` values must match the model version,
and their SHA-256 values must equal the canonical matrix digests returned by
`matrix_sha256`. Both resources must be observed during evaluation preflight
and must predate the request. Serving can omit `contrast_input_id`, but such a
factorized record is deliberately ineligible for confirmatory shadow evidence.

For a no-impact shadow rollout, add `serving_mode="shadow"`. The Rust oracle
still computes and rechecks its counterfactual plan, but `decision.slate`
remains the unchanged baseline:

```python
shadow = guarded_rerank(
    # The same request fields as above.
    request_id="shadow-01J...",
    event_timestamp="2026-10-01T12:00:00+09:00",
    baseline_item_ids=[10, 11, 12, 13],
    protected_count=2,
    remove_positions=[2, 3],
    add_item_ids=[100, 101],
    reward_deltas=[0.08, 0.05],
    safety_mean_deltas=[0.004, 0.003],
    safety_variances=[2e-6, 2e-6],
    baseline_metric=0.12,
    degradation_tolerance=0.01,
    max_replacements=2,
    serving_mode="shadow",
)

assert shadow.slate == (10, 11, 12, 13)          # actually served
assert shadow.proposed_slate == (10, 11, 100, 101)  # counterfactual

# Persist the primary proposal record first. Its own commit time cannot be
# placed inside the same immutable record without creating a cycle.
from safe_slate import (
    canonical_proposal_log_bytes,
    create_service_latency_trace,
    create_service_trace_completion,
)

primary_bytes = canonical_proposal_log_bytes(shadow.proposal_log)
proposal_log_sink.write_and_wait_for_commit(primary_bytes)

# After the primary commit, emit a digest-bound sidecar on the same W3C trace.
completion = create_service_trace_completion(
    shadow.proposal_log,
    trace_id=otel_trace_id,  # 32 lowercase hex characters
    completed_at=timezone_aware_completion_time,
    trace_source="opentelemetry",
    service_latency_trace=create_service_latency_trace(
        response_latency_ms=measured_ingress_to_response_ms,
        log_commit_latency_ms=measured_ingress_to_primary_log_commit_ms,
        candidate_generation_ms=measured_candidate_generation_ms,
        model_inference_ms=measured_model_inference_ms,
        queueing_ms=measured_queueing_ms,
        rpc_ms=measured_rpc_ms,
        guardrail_facade_ms=measured_guardrail_facade_ms,
        serialization_ms=measured_serialization_ms,
        proposal_log_write_ms=measured_primary_log_write_ms,
    ),
)
service_trace_sink.write(completion.to_dict())
```

The sidecar contains the primary proposal record's canonical SHA-256, request
id, event time, nonzero W3C trace id, completion time, source, and every stage
duration. Offline intake requires an exact one-to-one request/digest join:

`ServiceTraceRecorder` provides the request-side instrumentation. Create one at
service ingress, pass the current OpenTelemetry trace id, wrap local stages
with `measure(...)`, and use `record_stage_ms(...)` for upstream or remote
spans. `complete(...)` refuses to emit a sidecar until all seven stages,
response completion, and the primary log commit have been observed:

```python
from safe_slate import ServiceTraceRecorder, canonical_proposal_log_bytes

trace = ServiceTraceRecorder(
    trace_id=current_otel_trace_id,
    event_timestamp=request_event_timestamp,
)
trace.record_stage_ms("queueing", upstream_queue_ms)
trace.record_stage_ms("rpc", measured_rpc_ms)
with trace.measure("candidate_generation"):
    candidates = generate_candidates(request)
with trace.measure("model_inference"):
    scores = model(candidates)
with trace.measure("guardrail_facade"):
    decision = guarded_candidate_pool_rerank(...)
with trace.measure("serialization"):
    response_bytes = serialize(decision.slate)
send_response(response_bytes)
trace.mark_response()
with trace.measure("proposal_log_write"):
    proposal_log_sink.write_and_wait_for_commit(
        canonical_proposal_log_bytes(decision.proposal_log)
    )
trace.mark_primary_log_commit()
completion = trace.complete(
    decision.proposal_log,
    completed_at=timezone_aware_completion_time,
)
service_trace_sink.write(completion.to_dict())
```

The recorder is request-scoped and intentionally not thread-safe. It rejects
duplicate stages, unknown stages, negative/non-finite durations, backward
clocks, response/commit reordering, and completion with missing observations.

The current Go production integration emits a narrower, honest server-envelope
sidecar rather than the complete record above. Its schema is
`safe-slate-production-post-response-1.1`: gRPC begin, handler start, response
ready, actual `OutPayload` sent, and payload sizes. Intake is strict and
bounded:

```python
from safe_slate import (
    audit_observed_grpc_envelopes,
    load_observed_grpc_envelopes_jsonl,
)

records = load_observed_grpc_envelopes_jsonl(
    "/immutable/production-post-response.jsonl",
    maximum_records=1_000_000,
    maximum_bytes=2_000_000_000,
)
coverage = audit_observed_grpc_envelopes(records)
assert coverage.evidence_state == "partial_observability"
assert not coverage.full_service_trace_complete
```

This adapter never maps gRPC begin-to-handler time to upstream queueing and
never maps response-ready-to-payload-sent time to pure serialization. It also
does not invent the separate vendor RPC/internals, guardrail, proposal publish,
or durable log-commit acknowledgement. Those stages remain required before
the complete production service trace can be finalized.

```python
from safe_slate import (
    join_service_trace_completions,
    load_proposal_logs_jsonl,
    load_service_trace_completions_jsonl,
)

primary_logs = load_proposal_logs_jsonl(
    "/immutable/production-proposals.jsonl",
    maximum_records=1_000_000,
    maximum_bytes=2_000_000_000,
)
trace_completions = load_service_trace_completions_jsonl(
    "/immutable/service-trace-completions.jsonl",
    maximum_records=1_000_000,
    maximum_bytes=2_000_000_000,
)
shadow_logs = join_service_trace_completions(primary_logs, trace_completions)
```

Missing, extra, duplicate, digest-mismatched, non-finite, incomplete-stage, or
noncanonical records fail closed. The sidecar's own persistence is evidence
collection overhead; `log_commit_latency_ms` specifically measures durability
of the primary proposal record used by the join.

For actual `serving_mode="serve"` records, audit the frozen window directly
from both immutable streams:

```python
from safe_slate import summarize_production_service_run

service_summary = summarize_production_service_run(
    primary_logs,
    trace_completions,
    window_start="2026-10-01T00:00:00+09:00",
    window_end="2026-10-08T00:00:00+09:00",
    minimum_requests=10_000,
    maximum_response_p95_ms=50.0,
    maximum_log_commit_p95_ms=75.0,
    maximum_fallback_rate=0.01,
    maximum_invalid_record_rate=0.0,
    maximum_recheck_failure_rate=0.0,
    serving_mode="serve",
)
```

The summary reports p50/p95/p99/max for response, primary-log commit, candidate
generation, model inference, queueing, RPC, guardrail, serialization, and log
write, plus fallback, deadline, recheck, certificate, and invalid-record rates.
A pass authorizes only a descriptive statement for that exact frozen window;
the output deliberately keeps universal production-latency, future safety,
causal lift, and rollout claims false.

Shadow logs set `learning_update_eligible=False`, because observed outcomes
belong to the baseline slate, and separately expose
`shadow_evaluation_eligible` for latency, fallback, validity, and safety-margin
analysis.

Aggregate a frozen set of shadow logs with explicit operability thresholds:

```python
from safe_slate import summarize_shadow_run

shadow_summary = summarize_shadow_run(
    shadow_logs,
    window_start="2026-10-01T00:00:00+09:00",
    window_end="2026-10-08T00:00:00+09:00",
    minimum_requests=10_000,
    maximum_oracle_p95_ms=10.0,
    maximum_fallback_rate=0.01,
    maximum_invalid_output_rate=0.0,
    maximum_recheck_failure_rate=0.0,
    maximum_predicate_error_request_rate=0.0,
    require_service_latency_trace=True,
    maximum_service_response_p95_ms=50.0,
    maximum_log_commit_p95_ms=75.0,
)
```

The summary revalidates that every record is shadow-only, the actually served
slate equals the baseline, the protected core is unchanged, selected indices
and the posterior cardinality frontier are internally consistent, the final
recheck passed, and the certificate margin is nonnegative. It reports oracle
p50/p95/p99, fallback and gate-fallback rates, deadline and exact rates,
predicate errors, cardinality gaps, and safety margins. A deadline is reported
separately rather than treated as invalid when a certified incumbent survived
the final recheck. Every event must lie in the explicit half-open interval
`[window_start, window_end)`; the summary reports both that configured interval
and the minimum and maximum absolute event times actually observed.

Recorded latency covers only the in-process Rust oracle. It excludes upstream
model inference, RPC, queueing, serialization, and downstream logging, so
`production_latency_claim_authorized` remains false even when every operability
gate passes. A complete schema-1.3 service trace separately reports
ingress-to-response and ingress-to-log-commit p50/p95/p99, and can establish
`shadow_service_latency_evidence_valid` under frozen gates. It still does not
authorize a universal production-latency claim. Shadow logs contain baseline
outcomes and cannot identify target policy lift; `causal_lift_claim_authorized`
is therefore always false.

For confirmatory shadow evidence, freeze the window, thresholds, input
manifest, resource caps, output identity, and authorization boundary before
opening the final logs:

```python
from safe_slate import (
    create_shadow_evaluation_contract,
    load_proposal_logs_jsonl,
    run_no_clobber_shadow_evaluation,
    verify_shadow_evidence_bundle,
)

immutable_shadow_logs = load_proposal_logs_jsonl(
    "/read-only/mount/shadow-records.jsonl",
    maximum_records=1_000_000,
    maximum_bytes=2_000_000_000,
)

shadow_contract = create_shadow_evaluation_contract(
    evaluation_id="shadow-operability-v1",
    scientific_question="Does the guarded reranker operate safely in shadow?",
    population_identity="jp-home-feed-eligible-v3",
    window_start="2026-10-01T00:00:00+09:00",
    window_end="2026-10-08T00:00:00+09:00",
    input_manifest_digest=shadow_input_manifest.manifest_digest,
    code_revision=git_revision,
    dependency_lock_digest=uv_lock_sha256,
    output_namespace="evidence/shadow-operability-v1",
    minimum_requests=10_000,
    maximum_oracle_p95_ms=10.0,
    maximum_fallback_rate=0.01,
    maximum_invalid_output_rate=0.0,
    maximum_recheck_failure_rate=0.0,
    maximum_predicate_error_request_rate=0.0,
    require_service_latency_trace=True,
    maximum_service_response_p95_ms=50.0,
    maximum_log_commit_p95_ms=75.0,
    maximum_runtime_seconds=300.0,
    maximum_memory_mb=1_024,
    maximum_output_bytes=1_000_000,
)

shadow_run = run_no_clobber_shadow_evaluation(
    "/new/output/parent/shadow-operability-v1",
    output_identity=shadow_contract.output_namespace,
    contract=shadow_contract,
    input_manifest=shadow_input_manifest,
    observed_inputs=observed_shadow_inputs,
    logs=immutable_shadow_logs,
    created_at="2026-10-08T00:00:02+09:00",
    successful_execution_ids=["shadow-job-001"],
    peak_memory_mb=512,
    no_matching_execution_verified=True,
)

independent = verify_shadow_evidence_bundle(
    "/new/output/parent/shadow-operability-v1",
    logs=immutable_shadow_logs,
)
assert independent.valid and independent.source_recomputed
```

The bundle stores the contract, input manifest, input-observation audit,
summary, artifact hashes, and completion marker. It records a digest of the
ordered proposal logs but does not duplicate them. Independent verification
must therefore reopen the immutable source logs and recompute the complete
summary; verification without source logs fails. The JSONL loader accepts only
the exact proposal-log schema, finite JSON values, consistent operation/slate
shapes, timezone-aware shadow timestamps, and caller-supplied byte and record
caps. A small valid run is retained
as `insufficient_evidence`, an integrity failure as `evidence_rejected`, and an
operability-threshold miss as valid negative evidence. None authorizes serving,
production-latency claims, or causal purchase-lift claims.

## Production-shape benchmark

After a release build, run the deterministic synthetic benchmark:

```bash
uv run maturin develop --release
uv run python benchmarks/benchmark_guarded_rerank.py \
  --requests 1000 \
  --baseline-size 600 \
  --protected-count 560 \
  --operations 200 \
  --max-replacements 40 \
  --timeout-ms 10 \
  --serving-mode shadow
```

Add `--linear-feature-dimension 16` to exercise the factorized shared-model
uncertainty path without materializing an operation covariance matrix.

To include the new item-level candidate-pool expansion in the measured local
facade, run:

```bash
uv run python benchmarks/benchmark_candidate_pool.py \
  --requests 1000 \
  --baseline-size 600 \
  --protected-count 560 \
  --candidates 40 \
  --feature-dimension 8 \
  --max-replacements 40 \
  --timeout-ms 10 \
  --serving-mode shadow
```

This report separately exposes candidate-operation construction, uncertainty
projection/digesting, Rust-oracle, and complete process-local facade latency.
It also records the expanded operation count, deadline/fallback rates,
certified replacement counts, and final-recheck failures. It uses generated
scores and features and therefore remains development evidence only.

The JSON report separates Python-facade and Rust-oracle p50/p95/p99 latency,
deadline rate, fallback rate, selected cardinality, status counts, and final
recheck failures. The facade timer still excludes candidate generation, model
inference, RPC, queueing, and downstream logging. It is a process-local
synthetic benchmark, not production p95 or business-lift evidence. `--output`
is no-clobber and requires an existing parent directory.

The safety admission threshold is calculated as

```text
-degradation_tolerance * baseline_metric
```

so a one-percent tolerance on a baseline metric of `0.12` permits at most a
certified `-0.0012` aggregate change. The oracle first maximizes certified
replacement cardinality and then reward. `k_lower` is the rechecked served
cardinality; `k_upper` remains open when the search budget cannot close higher
depths.

Serving gates are fail-closed:

- Omitting runtime gate evidence returns the unchanged baseline. Callers must
  pass an evidence-derived `gate_assessment` or explicitly supply every gate.
- `support_ok=False` returns the unchanged baseline (`support_gate`).
- `distribution_shift_detected=True` returns the baseline
  (`distribution_shift_gate`).
- `uncertainty_valid=False` returns the baseline (`uncertainty_gate`).
- `proposal_log_complete=False` returns the baseline (`proposal_log_gate`) so
  exploration is not enabled without proposal-dominating feedback.

For evidence-derived gates, use `assess_serving_gates` instead of constructing
the booleans manually:

```python
from safe_slate import assess_serving_gates, guarded_rerank

gates = assess_serving_gates(
    support_rate=0.997,
    minimum_support_rate=0.99,
    distribution_shift_score=0.03,
    maximum_distribution_shift_score=0.10,
    uncertainty_evidence_valid=True,
    expected_proposal_count=200,
    logged_proposal_count=200,
)

decision = guarded_rerank(
    # baseline and replacement arrays omitted here
    gate_assessment=gates,
    # ...
)
```

Missing, non-finite, or out-of-range runtime evidence fails closed. Invalid
configured thresholds raise `ValueError`, because silently changing those
thresholds would change the deployment contract. The full assessment and its
reason codes are retained in log schema `1.1`.

After the outcome window settles, audit whether the proposal log can support a
direct update, an OPE precheck, or descriptive analysis only:

```python
from safe_slate import assess_feedback_coverage

feedback = assess_feedback_coverage(
    decision.proposal_log,
    # Operation indices with directly attributable settled outcomes.
    direct_observed_operation_indices=[0, 1, 2, 3],
    # Logging-policy probability for every proposed operation.
    propensities=[0.25, 0.25, 0.25, 0.25],
    labels_settled=True,
    missingness_independent=True,
    # This precheck does not replace manifest-bound population evidence.
    population_positivity_established=False,
    minimum_ope_support_rate=0.99,
    maximum_allowed_inverse_propensity_weight=100.0,
)

if feedback.direct_update_eligible:
    update_confidence_state(...)
elif feedback.ope_precheck_eligible:
    queue_for_user_level_ope(...)
else:
    retain_for_descriptive_audit_only(...)
```

`direct_update_eligible` requires settled, outcome-independent feedback for
every proposed operation, not only the accepted ones. If the observed set is
exactly the selected set while unselected proposals exist, the assessment
emits `accepted_only_feedback` and refuses a direct update. This prevents the
library from silently treating selection-biased feedback as proposal-dominating
feedback. The earlier `ProposalLog.learning_update_eligible` flag only records
that serving-time gates passed; it does not authorize a post-outcome update by
itself.

`ope_precheck_eligible` is intentionally only a request-level gate. It checks
positive propensity support and the largest inverse-propensity weight, but it
does not establish a causal estimand, user-level effective sample size,
independence across repeated users, or a valid final OPE. Those checks must be
performed by the evaluation pipeline. Shadow outcomes always belong to the
served baseline and therefore authorize neither direct candidate updates nor
candidate-policy OPE.

For a frozen evaluation window, aggregate request rows with
`audit_ope_readiness`. The non-support thresholds are required arguments so
they must be selected from a pilot or a preregistered contract before looking
at final outcomes:

```python
from safe_slate import audit_ope_readiness

readiness = audit_ope_readiness(
    request_ids=["r0", "r1", "r2", "r3"],
    user_ids=["u0", "u0", "u1", "u1"],
    # Per-request target-policy mass covered by known-positive logging support.
    guaranteed_target_support_mass=[1.0, 1.0, 1.0, 1.0],
    target_probability_of_logged_action=[0.5, 0.5, 0.5, 0.5],
    logging_propensities=[0.5, 0.5, 0.5, 0.5],
    purchase_events=[1, 0, 1, 0],
    # Every pseudonymous user must remain in one nuisance-model fold.
    fold_ids=[0, 0, 1, 1],
    propensity_mode="known",
    candidate_sets_reconstructable=True,
    labels_settled=True,
    missingness_independent=True,
    # Row-only audit cannot establish population positivity.
    population_positivity_established=False,
    minimum_user_effective_sample_size=2,
    minimum_purchase_events=2,
    maximum_missing_outcome_rate=0.0,
    maximum_allowed_p99_importance_weight=20.0,
    weight_clip=20.0,
)
```

The default support split follows the external-validation contract:

- exactly 100%: empirical overlap is complete on the finite logged requests;
- 80% to below 100%: point OPE stops; full-population bounds may proceed;
- below 80%: OPE stops and only descriptive replay remains.

Finite-log 100% support does not establish population positivity on unseen
contexts. A full-population point estimand additionally requires a separate
design-level or structural argument that the logging policy assigns positive
probability wherever the target policy has positive mass almost surely in the
registered target population. The row-only API never accepts that premise
from a boolean. Without a manifest-bound canonical positivity artifact,
readiness is limited to
`observed_finite_log_population_only` and standard population OPE remains
unauthorized.

An average support rate does not define a supported user subpopulation. For
example, 90% average support could mean that every request has 90% of its
target action mass covered, rather than that 90% of requests are fully covered.
Consequently, the library does not silently relabel the former as the latter.
A supported-subpopulation analysis requires a separately frozen population
rule and a fresh readiness audit over that population.

Known propensities produce an exact user-block importance-weight ESS. Covering
propensity intervals use the conservative lower bound
`(sum user_lower_weights)^2 / sum(user_upper_weights^2)` and can authorize only
partial-identification bounds, never point OPE. Missing propensities, users
crossing cross-fit folds, unsettled labels, outcome-dependent missingness,
insufficient purchase events, excessive missingness, a heavy p99 weight tail,
or low user ESS all fail closed. `production_rollout_authorized` is always
false: this audit controls offline analysis scope, not serving.

If any settled outcome is missing, the readiness audit additionally requires
one positive outcome-observation propensity per original request. ESS and the
p99 tail then use the combined potential weight
`pi_target / (pi_logging * rho_observe)`. Missing rows still enter the tail
audit, preventing a tiny observation probability from disappearing merely
because that row's outcome was not observed.

Only a full-population `pass` with known propensities can enter the packaged
point estimator:

```python
from safe_slate import estimate_cross_fitted_policy_value

estimate = estimate_cross_fitted_policy_value(
    readiness,
    request_ids=["r0", "r1", "r2", "r3"],
    user_ids=["u0", "u0", "u1", "u1"],
    guaranteed_target_support_mass=[1.0, 1.0, 1.0, 1.0],
    target_probability_of_logged_action=[0.5, 0.5, 0.5, 0.5],
    logging_propensities=[0.5, 0.5, 0.5, 0.5],
    purchase_events=[1, 0, 1, 0],
    fold_ids=[0, 0, 1, 1],
    # Predictions must come from models that excluded each user's fold.
    outcome_prediction_logged_action=[0.4, 0.4, 0.4, 0.4],
    outcome_prediction_target_policy=[0.6, 0.6, 0.6, 0.6],
    nuisance_predictions_cross_fitted=True,
    logging_propensities_are_request_time_probabilities=True,
    outcomes_match_logged_actions=True,
    bootstrap_repetitions=2_000,
    bootstrap_seed=16_630,
)
```

The readiness result carries a SHA-256 digest over request IDs, user blocks,
support mass, logged-action target probabilities, outcomes, folds, and
propensities. Estimation rejects any row-level mismatch, preventing a passing
audit from being reused for a different dataset. Doubly robust value and lift
are primary; IPS and SNIPS are diagnostics. Clipping is inherited from the
frozen readiness contract.

The percentile bootstrap resamples pseudonymous users as clusters and then
reconstructs every sampled user's requests. It recomputes a request-weighted
mean rather than equally averaging user means, so different request counts per
user do not silently change the purchases-per-request estimand. The
complete-outcome point estimator rejects propensity intervals, partial support,
and missing outcomes; the dedicated estimators below handle those cases
without silently changing the estimand. Its causal interpretation remains
conditional on the explicit identification attestations and never authorizes
rollout.

For known propensities and 80–<100% support, use full-population partial
identification instead of extrapolating a point estimate:

```python
from safe_slate import estimate_partial_identification_bounds

bounds = estimate_partial_identification_bounds(
    partial_readiness,
    request_ids=request_ids,
    user_ids=user_ids,
    guaranteed_target_support_mass=support_mass,
    target_probability_of_logged_action=target_probability,
    logging_propensities=logging_propensity,
    purchase_events=purchase_events,
    fold_ids=fold_ids,
    outcome_prediction_logged_action=q_logged,
    # Sum of pi_target(a|x) * q(a,x) over guaranteed-supported actions only.
    outcome_prediction_supported_target_contribution=q_supported_contribution,
    outcome_lower_bound=0.0,
    outcome_upper_bound=1.0,
    nuisance_predictions_cross_fitted=True,
    logging_propensities_are_request_time_probabilities=True,
    outcomes_match_logged_actions=True,
    bootstrap_repetitions=2_000,
    bootstrap_seed=16_630,
)
```

The supported contribution uses cross-fitted DR correction. The remaining
target-policy mass is assigned the registered outcome lower and upper bounds,
never a model prediction. The returned target-value and lift intervals
therefore widen in direct proportion to unsupported mass and outcome range.
The user-cluster percentile bootstrap provides a sampling-uncertainty envelope
around the estimated identification region; it is not a finite-sample coverage
theorem. Propensity-interval and missing-outcome cases remain separate
estimators rather than being forced through this known-propensity path.

When each logging propensity is known only to lie in a covering interval, use
the dedicated sign-aware estimator:

```python
from safe_slate import estimate_propensity_interval_bounds

interval_bounds = estimate_propensity_interval_bounds(
    interval_readiness,
    request_ids=request_ids,
    user_ids=user_ids,
    guaranteed_target_support_mass=support_mass,
    target_probability_of_logged_action=target_probability,
    logging_propensity_lower=propensity_lower,
    logging_propensity_upper=propensity_upper,
    purchase_events=purchase_events,
    fold_ids=fold_ids,
    outcome_prediction_logged_action=q_logged,
    outcome_prediction_supported_target_contribution=q_supported_contribution,
    outcome_lower_bound=0.0,
    outcome_upper_bound=1.0,
    propensity_intervals_cover_true_probabilities=True,
    nuisance_predictions_cross_fitted=True,
    propensity_intervals_are_request_time=True,
    outcomes_match_logged_actions=True,
    bootstrap_repetitions=2_000,
    bootstrap_seed=16_630,
)
```

For positive residuals, the lower bound uses the smallest feasible importance
weight and the upper bound the largest; for negative residuals this ordering is
reversed. The implementation never substitutes the interval midpoint.
`propensity_uncertainty_width` and `support_uncertainty_width` expose the two
sources of identification width separately. These bounds remain conditional on
the reported propensity intervals actually covering every true request-time
propensity; the library records that attestation but cannot prove it from the
interval endpoints alone.

For outcome-independent missing labels with validated request-time observation
propensities, use the missingness-adjusted point estimator:

```python
from safe_slate import estimate_missingness_adjusted_policy_value

adjusted = estimate_missingness_adjusted_policy_value(
    missingness_readiness,
    request_ids=request_ids,
    user_ids=user_ids,
    guaranteed_target_support_mass=support_mass,
    target_probability_of_logged_action=target_probability,
    logging_propensities=logging_propensity,
    outcome_observation_propensities=observation_propensity,
    purchase_events=purchase_events_with_none,
    fold_ids=fold_ids,
    outcome_prediction_logged_action=q_logged,
    outcome_prediction_target_policy=q_target,
    nuisance_predictions_cross_fitted=True,
    logging_propensities_are_request_time_probabilities=True,
    observation_propensities_are_validated=True,
    observation_propensities_are_request_time_probabilities=True,
    outcomes_match_logged_actions=True,
    conditional_missingness_independence=True,
    bootstrap_repetitions=2_000,
    bootstrap_seed=16_630,
)
```

Target-policy residuals use
`pi_target / (pi_logging * rho_observe)`. The logging comparator uses its own
observation-propensity DR correction, so target value and lift refer to the
same original request population. Combined and observation-only clipping rates
are reported separately. This path requires both observed and missing outcomes;
complete data use the ordinary estimator, while missing data without validated
observation propensities fail closed.

When missing outcomes occur together with partial support or propensity
intervals, use the joint-bound API rather than chaining point estimates:

```python
from safe_slate import estimate_joint_uncertainty_bounds

joint = estimate_joint_uncertainty_bounds(
    joint_readiness,
    request_ids=request_ids,
    user_ids=user_ids,
    guaranteed_target_support_mass=support_mass,
    target_probability_of_logged_action=target_probability,
    # Supply this array for known propensity mode.
    logging_propensities=logging_propensity,
    # For interval mode, omit logging_propensities and supply lower/upper.
    # logging_propensity_lower=propensity_lower,
    # logging_propensity_upper=propensity_upper,
    purchase_events=purchase_events_with_none,
    fold_ids=fold_ids,
    outcome_observation_propensities=observation_propensity,
    outcome_prediction_logged_action=q_logged,
    outcome_prediction_supported_target_contribution=q_supported_contribution,
    outcome_lower_bound=0.0,
    outcome_upper_bound=1.0,
    nuisance_predictions_cross_fitted=True,
    propensity_evidence_is_request_time=True,
    observation_propensities_are_validated=True,
    observation_propensities_are_request_time_probabilities=True,
    outcomes_match_logged_actions=True,
    conditional_missingness_independence=True,
    bootstrap_repetitions=2_000,
    bootstrap_seed=16_630,
)
```

This estimator applies observation-propensity correction only to observed
residuals, retains cross-fitted nuisance contributions for missing rows, uses
sign-aware action-propensity endpoints when intervals are supplied, and bounds
unsupported target mass last. The reported `propensity_uncertainty_width` and
`support_uncertainty_width` therefore remain interpretable when all three
uncertainty sources coexist. Known propensity plus full support uses the
missingness-adjusted point estimator instead; complete-outcome cases continue
to use the specialized estimators above.

Even on a gate failure, the proposal record contains selected and unselected
candidate scores, uncertainty inputs, gate state, frontier, latency, and final
recheck status. Realized clicks or purchases are deliberately excluded and
should be joined later by `request_id` after their observation window settles.
The current log schema is identified by `schema_version="1.3"`. Shadow records
require a timezone-aware `event_timestamp`, allowing a frozen evaluation window
to be audited rather than inferred from ingestion time.

## Immutable confirmatory evaluation contract

For confirmatory OPE, freeze scientific, temporal, statistical, execution, and
authorization boundaries before opening final labels:

For ten privacy-reviewed local exports, the no-clobber intake CLI hashes the
files without copying row data and writes the completion marker last:

Start from the tracked fail-closed template:

    cp external-inputs.template.json /private/evaluation/external-inputs.json

Replace every placeholder and set privacy_reviewed to true only after the
privacy review has completed.

```bash
uv run tommyslate-freeze-inputs \
  --spec /private/evaluation/external-inputs.json \
  --output /private/frozen/manifest.json \
  --completion-marker /private/frozen/manifest.complete.json
```

The strict JSON specification contains manifest identity/timestamps plus one
entry for each of `candidate_action_log`, `propensity_or_interval`,
`settled_purchase_labels`, `pseudonymous_user_blocks`,
`population_positivity_evidence`,
`production_constraint_trace`, `shadow_record`, `service_trace_completion`,
`linear_parameter_covariance`, and `request_contrast_features`. Each entry
provides `input_id`, `semantic_role`, local `path`, `version`,
`snapshot_time`, `temporal_scope`, and `schema_version`. The top level must
explicitly set `privacy_reviewed=true` and `synthetic_inputs=false`. Matrix
files are checked against their declared canonical `f64le-v1` or `json-v1`
encoding. `maximum_input_bytes` and `maximum_log_records` bound reads. Primary
proposal JSONL and post-commit trace JSONL are fully parsed and must complete
an exact one-to-one request/event/digest join before hashing. This step
establishes immutable identity and structural trace readiness only; row schema,
support, settlement, missingness, OPE, and shadow gates remain pending.
The tracked template deliberately keeps privacy_reviewed false, uses
nonexistent /private/replace-me/ paths, and contains no row data. Invoking the
CLI without completing those substitutions therefore fails closed. Private
rows and production paths must never be committed to this repository.
These semantic role names are also the canonical roles consumed by
`audit_manifest_bound_ope_readiness` and `run_no_clobber_ope`; the frozen CLI
manifest is passed directly to preflight without a role-renaming or re-freeze
step.

Population-positivity evidence has two immutable encodings. Legacy v1 retains
the original design-level artifact unchanged. Statistical v2 additionally
binds `guarantee_kind`, simultaneous failure probability, validity start/end,
registered epoch count, audit-design identity, and the SHA-256 of the source
certificate. The OPE preflight accepts either canonical schema, but for v2 it
requires the validity interval to cover the complete final evaluation window
and returns the failure probability and validity interval in its own result.
It never converts a finite-sample statement into an unqualified boolean.
For v2, readiness also reports the estimator's nominal failure
`alpha_ope=1-confidence_level` and the dependence-robust composition
`alpha_joint<=alpha_ope+alpha_positivity`. The resulting
`joint_confidence_level_lower_bound` is persisted in the readiness artifact;
the estimator's own interval remains labelled with its OPE-only confidence.
No independence between positivity auditing and OPE is assumed.

First, identify every input by immutable object identity and content digest.
Feature snapshots may not cross `feature_cutoff`; settled-label snapshots must
fall after that cutoff and no later than `label_cutoff`:

```python
from safe_slate import create_immutable_input, create_input_manifest

candidate_log = create_immutable_input(
    input_id="candidate-log",
    semantic_role="candidate_action_log",
    canonical_uri="gs://research-inputs/candidates.parquet",
    version="2026-09-final",
    generation="1725235200000000",
    metageneration="1",
    byte_size=candidate_log_size,
    sha256_digest=candidate_log_sha256,
    snapshot_time="2026-09-08T00:00:00+09:00",
    temporal_scope="feature",
    schema_version="candidate-log-v3",
)

# Add immutable entries for propensities, settled labels, pseudonymous user
# blocks, constraints, shadow records, trace completions, covariance, contrasts,
# model, query, code, and lockfile.
input_manifest = create_input_manifest(
    manifest_id="purchase-ope-final-inputs-v1",
    population_identity="jp-home-feed-eligible-v3",
    created_at="2026-10-08T00:00:00+09:00",
    feature_cutoff="2026-09-08T00:00:00+09:00",
    label_cutoff="2026-10-08T00:00:00+09:00",
    required_roles=(
        "candidate_action_log",
        "propensity_or_interval",
        "settled_purchase_labels",
        "pseudonymous_user_blocks",
        "population_positivity_evidence",
        "production_constraint_trace",
        "shadow_record",
        "service_trace_completion",
        "linear_parameter_covariance",
        "request_contrast_features",
    ),
    inputs=all_immutable_inputs,
)
```

Canonical URIs cannot contain credentials, query parameters, fragments, or
ambient aliases. `file:` inputs can be re-hashed locally with
`verify_local_manifest_inputs`, and `observed_input_from_local_file` produces a
fresh observation suitable for the manifest-bound preflight. External object
identities remain explicitly declared rather than being incorrectly reported
as byte-verified.

```python
from safe_slate import create_ope_evaluation_contract

contract = create_ope_evaluation_contract(
    evaluation_id="purchase-ope-final-v1",
    scientific_question="Does the target improve settled purchases per request?",
    baseline_policy="baseline-v1",
    target_policy="target-v2",
    population_identity="jp-home-feed-eligible-v3",
    pilot_window_start="2026-08-01T00:00:00+09:00",
    pilot_window_end="2026-08-08T00:00:00+09:00",
    final_window_start="2026-09-01T00:00:00+09:00",
    final_window_end="2026-09-08T00:00:00+09:00",
    earliest_analysis_time="2026-10-08T00:00:00+09:00",
    label_settlement_rule="30-day purchase window plus ingestion lag",
    primary_metric="settled purchases per request",
    primary_contrast="target policy value minus logging policy value",
    propensity_mode="known",
    input_manifest_digest=input_manifest.manifest_digest,
    code_revision=git_revision,
    dependency_lock_digest=uv_lock_sha256,
    output_namespace="evidence/purchase-ope-final-v1",
    minimum_support_rate_for_full_estimand=1.0,
    minimum_support_rate_for_partial_estimand=0.80,
    minimum_user_effective_sample_size=1_000,
    minimum_purchase_events=200,
    maximum_missing_outcome_rate=0.01,
    maximum_p99_importance_weight=20.0,
    weight_clip=20.0,
    bootstrap_repetitions=2_000,
    bootstrap_seed=16_630,
    confidence_level=0.95,
    outcome_lower_bound=0.0,
    outcome_upper_bound=1.0,
    minimum_raw_requests=10_000,
    maximum_runtime_seconds=3_600,
    maximum_memory_mb=4_096,
    maximum_output_bytes=10_000_000,
)
```

The contract is frozen, serializable, SHA-256 identified, timezone-aware, and
requires non-overlapping pilot and final windows. Its default permissions allow
offline research reporting only and explicitly prohibit rollout, serving
changes, online experiments, post-label threshold changes, and overwriting an
existing output identity. Loading a serialized contract verifies both its
digest and canonical semantics.

Immediately before reading evaluation data, the storage adapter must observe
the actual immutable identity of every resource. The manifest-bound entrypoint
refuses to inspect OPE rows until all declared inputs match exactly:

```python
from safe_slate import (
    audit_manifest_bound_ope_readiness,
    create_observed_input_identity,
)

observed_inputs = [
    create_observed_input_identity(
        input_id="candidate-log",
        canonical_uri=storage_metadata.canonical_uri,
        version=storage_metadata.version,
        generation=storage_metadata.generation,
        metageneration=storage_metadata.metageneration,
        byte_size=storage_metadata.byte_size,
        sha256_digest=storage_metadata.sha256,
    ),
    # One observation for every remaining manifest input.
]

# A backend-neutral alternative dispatches by URI scheme. `file:` is built in;
# adapters for gs:, s3:, model:, git:, and similar schemes remain read-only.
from safe_slate import observe_manifest_inputs

observation_batch = observe_manifest_inputs(
    input_manifest,
    observers={"gs": observe_gcs_object, "model": observe_model_registry},
)
if not observation_batch.valid:
    raise RuntimeError(observation_batch.reasons)
observed_inputs = observation_batch.observations

preflight = audit_manifest_bound_ope_readiness(
    contract,
    input_manifest,
    observed_inputs,
    analysis_timestamp="2026-10-08T00:00:00+09:00",
    # Canonical evidence whose identity and SHA-256 are registered in manifest.
    population_positivity_evidence=population_positivity_evidence,
    request_ids=request_ids,
    user_ids=user_ids,
    guaranteed_target_support_mass=support_mass,
    target_probability_of_logged_action=target_probability,
    logging_propensities=logging_propensity,
    purchase_events=settled_purchase_events,
    fold_ids=user_blocked_fold_ids,
    candidate_sets_reconstructable=True,
    labels_settled=True,
    missingness_independent=True,
)
if preflight.readiness is None:
    raise RuntimeError(preflight.reasons)
readiness = preflight.readiness
```

Missing, extra, duplicate, changed, or contract-inconsistent inputs yield
`decision="evidence_rejected"` and `readiness=None`. Before the frozen label
settlement time the result is `labels_pending`. Only an exact observation set
can reach the existing support, propensity-tail, missingness, event-count, and
user-blocked ESS gates. Missing adapters and adapter exceptions also fail
closed; exception messages are omitted so credentials cannot leak into audit
records. This result never authorizes production rollout.

Use `audit_manifest_bound_ope_readiness` for confirmatory runs after the
earliest label-ready time. `audit_ope_readiness_with_contract` remains the
lower-level diagnostic entrypoint, but it cannot authorize a population point
estimand without the manifest-bound positivity artifact. Contract-controlled
thresholds cannot be supplied or overridden at runtime. The resulting
readiness object and every estimator result carry the contract digest. A
contract-bound readiness object cannot be estimated without the matching
contract, and changing bootstrap repetitions, seed, confidence, clipping,
support gates, or outcome bounds is rejected.

For a local or backend-adapter worker, `run_no_clobber_ope` fixes the stage
order to input observation, readiness, estimator dispatch, bundle publication,
and byte-level verification:

```python
from safe_slate import run_no_clobber_ope

run = run_no_clobber_ope(
    "/new/output/parent/purchase-ope-final-v1",
    output_identity=contract.output_namespace,
    contract=contract,
    input_manifest=input_manifest,
    observed_inputs=observed_inputs,
    population_positivity_evidence=population_positivity_evidence,
    analysis_timestamp="2026-10-08T00:00:00+09:00",
    created_at="2026-10-08T00:00:01+09:00",
    row_evidence=row_evidence,
    estimation_evidence=estimation_evidence,
    collection_complete=True,
    label_snapshot_digest=settled_label_sha256,
    successful_execution_ids=["batch-evaluation-001"],
    peak_memory_mb=2_048,
    no_matching_execution_verified=True,
)
```

The runner selects point DR, partial-support bounds, propensity-interval
bounds, missingness-adjusted DR, or joint uncertainty bounds from the audited
readiness state. It never accepts runtime overrides for frozen bootstrap,
confidence, clipping, support, or outcome settings. A statistically
insufficient run publishes a valid bundle without `estimate.json`; an input
integrity rejection or unsettled label boundary stops before creating an
output. A successful run remains `artifact_complete` until a separate final
auditor reviews it and never authorizes production rollout.

`classify_evaluation_evidence` keeps collection completeness, label settlement,
statistical sufficiency, final audit, and decision eligibility separate. Its
states include `labels_pending`, `analysis_ready`, `insufficient_evidence`,
`artifact_complete`, `evidence_rejected`, and `decision_eligible`. A successful
process or completed artifact does not become valid evidence automatically;
even `decision_eligible` authorizes only the contract's offline actions and
retains every explicit prohibition.

Persist a completed offline run as a new evidence bundle:

```python
from safe_slate import write_evidence_bundle, verify_evidence_bundle

manifest = write_evidence_bundle(
    "/new/output/parent/purchase-ope-final-v1",
    output_identity=contract.output_namespace,
    contract=contract,
    input_manifest=input_manifest,
    input_verification=preflight.input_verification,
    readiness=readiness,
    estimate=estimate,
    evidence_status=evidence_status,
    created_at="2026-10-08T00:00:01+09:00",
    successful_execution_ids=["batch-evaluation-001"],
    failed_execution_ids=[],
    actual_runtime_seconds=125.4,
    peak_memory_mb=2_048,
    no_matching_execution_verified=True,
)

verification = verify_evidence_bundle(
    "/new/output/parent/purchase-ope-final-v1"
)
assert verification.valid
```

The writer refuses an existing destination, a mismatched output identity, an
unverified duplicate-execution check, resource-cap violations, contract/row
digest mismatches, or any production-write, serving-change, credential, or
personal-record safety flag. It stages files in a fresh sibling directory,
writes the completion marker last, and atomically publishes the directory. A
failure removes only that validated temporary directory; it never overwrites or
repairs an existing evidence identity.

The bundle contains the contract, immutable input manifest, input-observation
verification, readiness, optional estimate, evidence status, artifact
manifest, and completion marker JSON files. The estimate is omitted only for
a valid `insufficient_evidence` or pre-estimation `analysis_ready` state. The
independent verifier starts from persisted bytes, rechecks every
size and SHA-256 digest, reloads the canonical contract and input manifest,
and verifies their digest, population, and temporal binding together with
frozen readiness and inference settings, resource caps, state booleans,
execution identities, and authorization boundaries. Rehashing a modified
artifact and manifest does not make a semantic contract violation valid.
Corrections and retries require a fresh evaluation and output identity; the
failed bundle is preserved as non-evidence.

The caller must still check the execution service for a matching job identity;
an empty local destination alone is insufficient. Direct, unbound audit and
estimator APIs remain available for exploratory work, but should not be used as
confirmatory evidence.

## Research basis and citation

This library extracts reusable software contracts from the author's
[`purchase-aware-retrieval-research`](https://github.com/natomi1203/purchase-aware-retrieval-research)
repository. The [research integration boundary](docs/research-integration.md)
separates those repository-authored contracts from third-party theoretical
background and from research that is not yet eligible for the stable API. The
same classification is available as
[machine-readable research lineage](docs/research-lineage.json), pinned to the
frozen source and publication-registry revisions used for the assessment.

Use [`CITATION.cff`](CITATION.cff) to cite the software. Citing the software
does not imply that a manuscript's empirical, theoretical, or production
claims have been independently validated.

## Safety and scope

- `safety_means` are item-operation contrasts relative to the unchanged
  baseline. The baseline has contrast zero, so `safety_threshold` must be at
  most zero.
- `guarded_rerank` expects the same operation-level contrasts and computes
  `safety_threshold = -degradation_tolerance * baseline_metric`. It does not
  infer causal purchase lift from prediction scores.
- `guaranteed_target_support_mass` must be derived from the complete
  request-level logging and target action distributions. It cannot be inferred
  from the single logged action alone.
- `outcome_prediction_supported_target_contribution` is a probability-weighted
  contribution over guaranteed-supported target actions, not a conditional
  mean normalized by support mass.
- The library can verify fold separation, row digests, numerical contracts,
  and declared propensities, but it cannot independently prove that nuisance
  models excluded each fold or that logged propensities were generated at
  request time. Those remain explicit, auditable attestations.
- A propensity-interval result has no coverage meaning if the true propensity
  falls outside its reported interval. Interval coverage must be validated by
  the logging pipeline or a separate calibration audit.
- Missingness adjustment requires observation propensities defined for every
  original request, including rows whose labels are missing. Estimating them
  only on observed rows does not identify the original request population.
- Joint bounds do not make missing-at-random or propensity-coverage assumptions
  testable. They compose the declared assumptions without converting them into
  empirical facts.
- Slate safety is checked as
  `sum(safety_means) - confidence_radius * sqrt(slate_variance)`.
- `covariance`, when supplied, must be finite, symmetric, positive
  semidefinite, and have the same diagonal as `safety_variances`.
- Native group caps and pair conflicts are downward-closed and can prune the
  search. A custom Python predicate is not assumed monotone: a rejected slate
  is not used to prune its supersets.
- Calling Python for every safety-eligible candidate is flexible but can
  dominate runtime. `custom_predicate_batch` receives every safety-eligible
  candidate at one completed beam depth in one Python call. Stable high-volume
  rules should still be promoted to native Rust constraints.
- `optimize_replacements` treats each candidate as a balanced `(remove,
  add)` operation. It prevents duplicate removals and additions, rejects every
  removal from the protected prefix, and can enforce independent remove-side
  and add-side group-cap partitions.
- Supplying `baseline_item_ids` materializes the final fixed-length slate.
  Protected positions must remain unchanged and the final item IDs must be
  unique. A temporarily colliding partial replacement is retained because a
  later operation can remove the original copy.
- Python predicate exceptions and non-boolean returns reject that candidate.
  The returned incumbent is checked again; final failure falls back to the
  registered unchanged baseline.
- A custom predicate must be deterministic and side-effect free for an
  `exact` status to have its ordinary mathematical meaning. Any observed
  predicate exception keeps the posterior interval open.
- `timeout_ms` is cooperative. Rust checks it between expansions and callback
  calls; it cannot safely interrupt a Python callback that does not return.
- The production facade defaults to `timeout_ms=10`. This is a configured
  budget, not evidence that a particular production workload meets a 10 ms
  p95. Benchmark the actual candidate count, constraints, callbacks, and
  hardware before setting an SLA.
- `k_lower` is the returned feasible cardinality. `k_upper == k_lower` only
  after exhaustive search; otherwise the current skeleton returns the declared
  cap as a conservative upper bound.
- Finite-cell calibration requires direct bounded contrasts independent of
  training. It does not validate support, stationarity, repeated-user
  dependence, adaptive calibration reuse, or distribution shift.
- `support_ok`, `uncertainty_valid`, and `distribution_shift_detected` are
  externally supplied evidence gates. The library enforces their fail-closed
  semantics but does not manufacture support or shift evidence.
- The returned `first_unresolved_depth` is the first cardinality above the
  incumbent that has not been globally closed. In incomplete search,
  `k_upper` uses remove/add uniqueness, mutable-region size, and independent
  group-cap capacities. A greedy clique cover of the registered operation
  conflict graph adds a certified independent-set upper bound. The cover need
  not be minimum, so the interval can remain conservative.
- A stronger proof-stage relaxation and asynchronous/out-of-process Python
  predicate execution remain subsequent modules.
