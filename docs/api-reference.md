# Public API reference

The stable API for the 0.3 series is the seven-symbol surface recorded in
`library.json`. Other root-level exports support advanced evidence workflows
but are provisional until they are added to that manifest.

## `optimize_slate`

Construct a slate from item-level reward and safety inputs. It supports
diagonal variances, a full covariance matrix, or factorized linear
uncertainty; group caps; conflicts; and scalar or batched whole-slate
predicates.

Returns `CertifiedOracleResult`. The safety-relevant fields are:

- `selected_indices`: the rechecked incumbent;
- `safety_lcb`: its aggregate lower confidence bound;
- `rechecked`: whether the final safety and compatibility check completed;
- `exact`: whether search closed the frontier;
- `k_lower`, `k_upper`: certified cardinality interval;
- `stop_reason`, `deadline_reached`: budget outcome.

## `optimize_replacements`

Choose balanced remove/add operations against an existing baseline. The
protected prefix is immutable. Returns `CertifiedReplacementResult`, including
the selected operations and `materialized_slate` when baseline item ids are
provided.

## `guarded_rerank`

Production facade around `optimize_replacements`. Runtime evidence gates are
closed by default. Supply either a `ServingGateAssessment` or all four strict
boolean gate values. Failed evidence gates return the unchanged baseline with
`fallback=True`; malformed configuration raises `ValueError`.

The timeout is cooperative. Search stops accepting new work at its deadline,
but the mandatory final recheck may finish after it. Enforce a hard service
deadline outside the process and log measured end-to-end latency.

## `assess_serving_gates`

Convert measured support, shift, uncertainty, and proposal-log evidence into
the four decisions consumed by the serving facades. Missing, malformed, or
insufficient runtime evidence closes its gate and records a reason.

## `guarded_candidate_pool_rerank`

Beginner-facing production facade that expands a baseline suffix and insertion
candidate pool into balanced operations, retains shared linear-model
uncertainty, and delegates search and final rechecks to Rust. Expansion,
feature-cell, and compiled-conflict caps bound preprocessing before search.

## `audit_ope_readiness`

Audit logged-policy support, population positivity, user blocking, propensity
quality, outcome observation, effective sample size, and weight tails before
off-policy estimation. Missing or insufficient runtime evidence produces a
negative assessment; malformed frozen policy thresholds raise `ValueError`.
An assessment does not itself authorize rollout.

## `run_no_clobber_ope`

Run the contract-bound OPE workflow and publish a verified evidence bundle.
The destination must not exist. Identity or contract mismatch stops before
estimation, and an existing destination is never overwritten.

## Common exception contract

- `ValueError`: malformed, non-finite, inconsistent, or unsafe configuration;
- `FileExistsError`: a no-clobber output already exists;
- `RuntimeError`: an environmental or verification failure that is not an
  input-domain error.

Serving callers should treat every exception as a no-change decision. A search
budget exhaustion is not an exception: it returns a rechecked incumbent and an
open certificate interval.
