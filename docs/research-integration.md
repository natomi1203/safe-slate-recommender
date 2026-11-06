# Research integration boundary

This classification compares the repository's research registry and frozen
research artifacts with the public library surface. It is intentionally about
reusable implementation contracts, not publication claims. The exact mapping
is also published as [`research-lineage.json`](research-lineage.json), and the
software citation is recorded in [`../CITATION.cff`](../CITATION.cff).

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

### Adopted contract map

| Repository-authored research | Stable software contract | Public interfaces | Claim boundary |
| --- | --- | --- | --- |
| `certified-reranking-systems` | protected baseline, fail-closed serving, proposal-dominating observability | `optimize_replacements`, `guarded_rerank`, `guarded_candidate_pool_rerank` | no production lift, latency, calibration, or external-validity claim |
| `certified-reranking-systems` | immutable inputs, frozen OPE gates, no-clobber evidence | `audit_ope_readiness`, `run_no_clobber_ope` | readiness and artifact completeness do not authorize rollout |
| `certified-slate-oracle` | anytime incumbent with lower/upper cardinality certificate and final recheck | `optimize_slate`, `optimize_replacements` | an open interval is not an exact-optimality claim |
| `certified-slate-oracle` | non-monotone predicate correctness and shared covariance validation | `optimize_slate`, `guarded_candidate_pool_rerank` | validation does not prove deployment calibration or arbitrary-constraint tractability |

The source implementation is frozen at
`1a8b4290755f00c1fb78a69602af336ebe7e7742`. Publication statuses are a
separate snapshot pinned in the machine-readable lineage. A later paper status
does not silently add or remove a public software contract.

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

In particular, the paper-specific `required_clusters` normal approximation,
the `certified-slate-oracle` external-validation intake validator, and the
development-only representation-revision certificate checker are not copied
into the stable package.

## 4. Third-party theoretical background

The research repository cites third-party work that helps place the software
contracts in context. The library did not import code from these papers and
does not present their algorithms as repository-authored contributions.

- Conservative decisions and baseline protection: [Wu et al. (2016)](https://proceedings.mlr.press/v48/wu16.html), [Kazerouni et al. (2017)](https://proceedings.neurips.cc/paper/2017/hash/bdc4626aa1d1df8e14d80d345b2a442d-Abstract.html), and [Laroche et al. (2019)](https://proceedings.mlr.press/v97/laroche19a.html).
- Slate off-policy evaluation and sequential inference: [Swaminathan et al. (2017)](https://proceedings.neurips.cc/paper/2017/hash/5352696a9ca3397beb79f116f3a33991-Abstract.html) and [Karampatziakis et al. (2021)](https://proceedings.mlr.press/v139/karampatziakis21a.html).
- Linear confidence bounds: [Abbasi-Yadkori et al. (2011)](https://proceedings.neurips.cc/paper/2011/hash/e1d5be1c7f2f456670de3d53c7b54f4a-Abstract.html).
- Uncertainty and dataset shift: [Ovadia et al. (2019)](https://proceedings.neurips.cc/paper_files/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html) and [Rabanser et al. (2019)](https://proceedings.neurips.cc/paper_files/paper/2019/hash/846c260d715e5b854ffad5f70a516c88-Abstract.html).

The full bibliographic metadata and the exact `background_only` disposition
are recorded in `research-lineage.json`.
