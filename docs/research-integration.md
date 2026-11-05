# Research integration boundary

This classification compares the repository's research registry and frozen
research artifacts with the public library surface. It is intentionally about
reusable implementation contracts, not publication claims.

## 1. Generalizable results included in the library

The following abstractions are mature enough to expose without importing a
paper-specific theorem or fixed result:

- cardinality-first anytime slate construction with a rechecked incumbent and
  an explicit lower/upper cardinality certificate;
- protected-core balanced replacement, partition caps, pair conflicts, and
  non-monotone whole-slate predicates;
- diagonal, full-covariance, and factorized shared-linear uncertainty with
  finite, shape, symmetry, and positive-semidefinite checks;
- fail-closed runtime gates, baseline fallback, shadow serving, proposal-
  dominating logs, and deadline/budget audit fields;
- immutable input identities, no-clobber evidence bundles, deterministic
  digests, contract-bound OPE readiness, partial-identification bounds, and
  explicit authorization boundaries;
- implementation invariants derived from the research: never prune all
  descendants after a non-monotone predicate rejection, never serve without a
  final recheck, never silently replace missing evidence with a favorable
  default, and never overwrite an evidence destination.

These capabilities support the registry entries `certified-reranking-systems`
and `certified-slate-oracle`. Their external-evidence status limits empirical
claims; it does not weaken the fail-closed software contracts above.

## 2. Publication-only results referenced but not copied

The library does not contain manuscript text, paper-owned theorem statements
or proofs, fixed experiment tables, registered audit outcomes, submission
language, dataset-specific conclusions, or claims of production lift,
latency, calibration, regret, or external validity. Those remain in their
Publication PRs and frozen evidence directories.

Synthetic stress profiles, exact numeric thresholds chosen for one study, and
paper-specific ablations are test or benchmark inspiration only. They are not
library defaults unless an independent software contract justifies the value.

## 3. Results excluded from the public API

Registry work marked `research_in_progress` or `draft_unreviewed` is excluded:
evidence-preserving representation updates, irreversible-deadline asymptotics,
scenario-tree certification, and adaptive-testability characterizations.
Human-review-pending neural calibration and full-minimax research is also not
promoted as a new stable API solely from its publication status.

Protocol templates, unreconciled legacy results, incomplete external
validation, production-only assumptions, and any method that lacks a frozen
software invariant remain research inputs. They may be reconsidered only in a
later Software PR after independent validation, API review, and a migration
plan.
