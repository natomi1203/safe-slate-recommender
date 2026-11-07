# TommySlate

[![CI](https://github.com/natomi1203/tommyslate/actions/workflows/ci.yml/badge.svg)](https://github.com/natomi1203/tommyslate/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/tommyslate.svg)](https://pypi.org/project/tommyslate/)
[![Python 3.15](https://img.shields.io/badge/Python-3.15-3776AB.svg?logo=python&logoColor=white)](docs/compatibility.md)
[![Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

## Improve a ranked list without risking the one that already works.

TommySlate sits between your ranking model and the final list. Give it the
current ranking, candidates, business rules, and uncertainty estimates; it
returns an approved update—or keeps the original list and records why.

It is a decision layer, not another recommendation model.

```text
your ranking model ──▶ current list ──▶ TommySlate ──▶ final list
                                          │
                          rules + uncertainty + audit record
```

## See the decision

| Input | Example |
| --- | --- |
| Current list | `[A, B, C, D]` |
| Protect | Keep `[A, B]` in place |
| Candidates | `[X, Y]` |
| Rules | Change at most 2 positions; satisfy category and safety conditions |

If every check passes, TommySlate can return `[A, B, X, Y]`.
If a rule or evidence check fails, it returns `[A, B, C, D]` and the reason.

## Three jobs

### Control change

Keep positions fixed, limit how many items may change, and enforce conflicts,
category caps, and whole-list rules.

### Decide under uncertainty

Use variance or covariance estimates, respect a time budget, and recheck the
complete result before it can be returned.

### Operate responsibly

Keep the original ranking when approval is not justified, and record the
decision for audits, shadow evaluation, and offline policy evaluation (OPE).

## Install

TommySlate requires Python 3.15. Add it to a uv-managed project:

```bash
uv add tommyslate
```

The distribution is `tommyslate`; the stable Python import is `safe_slate`.

## Try one decision

The example approves two replacements, then shows the same request falling
back to the original list when its serving evidence is not supplied.

```python
from safe_slate import guarded_rerank

request = dict(
    request_id="readme-example",
    baseline_item_ids=[10, 11, 12, 13],       # [A, B, C, D]
    protected_count=2,                        # protect [A, B]
    remove_positions=[2, 3],
    add_item_ids=[100, 101],                  # [X, Y]
    reward_deltas=[0.8, 0.7],
    safety_mean_deltas=[0.04, 0.03],
    safety_variances=[0.0001, 0.0001],
    baseline_metric=0.5,
    degradation_tolerance=0.01,
    max_replacements=2,
)

approved = guarded_rerank(
    **request,
    support_ok=True,
    uncertainty_valid=True,
    distribution_shift_detected=False,
    proposal_log_complete=True,
)
assert approved.slate == (10, 11, 100, 101)
assert approved.slate[:2] == (10, 11)
assert approved.oracle.rechecked

kept = guarded_rerank(**request)
assert kept.slate == (10, 11, 12, 13)
assert kept.fallback_reason == "support_gate"
assert kept.oracle.rechecked
```

## What TommySlate does not do

TommySlate does not generate candidates, train or replace your prediction
model, calibrate uncertainty, prove causal or revenue lift, or authorize a
production rollout. Your system remains responsible for those inputs and
claims.

## For engineers

TommySlate is a fail-closed ranked-list optimizer with a Rust search core and
a typed Python API. The unchanged input list is the baseline; an accepted
result is a final-rechecked slate with a complete proposal log.

| Goal | Start with |
| --- | --- |
| Optimize a new list | `optimize_slate` |
| Protect part of an existing list | `optimize_replacements` |
| Apply serving gates and audit logging | `guarded_rerank` |
| Convert candidate arrays into replacement operations | `guarded_candidate_pool_rerank` |
| Validate serving evidence | `assess_serving_gates` |
| Check OPE inputs | `audit_ope_readiness` |
| Write immutable evaluation evidence | `run_no_clobber_ope` |

Fail-closed means a failed gate returns the registered baseline with a reason.
Invalid configuration raises before a decision can be served. A deadline
returns only a rechecked incumbent—or the baseline. Evidence writers refuse
to overwrite an existing destination.

## Documentation

- [Quickstart](docs/quickstart.md) — inputs, decisions, outputs, and a first run.
- [API reference](docs/api-reference.md) — stable functions and contracts.
- [Full guide](docs/full-guide.md) — uncertainty, shadow mode, OPE, and evidence.
- [gRPC sidecar](docs/sidecar.md) — loopback deployment and model contract.
- [Compatibility](docs/compatibility.md) — Python, ABI, platform, and migration policy.
- [Research boundary](docs/research-integration.md) — adopted contracts and non-claims.

## Release and development

TommySlate provides CPython stable-ABI wheels for Python 3.15 on Linux x86-64
and ARM64, macOS x86-64 and ARM64, and Windows x86-64. Releases are qualified
on TestPyPI, matched by SHA-256, and reinstalled on all five runners before
production promotion.

```bash
uv sync --frozen
uv run maturin develop --release
uv run pytest
```

Python and Rust coverage gates are 100%. The software is Apache-2.0 licensed.
See [CITATION.cff](CITATION.cff) and [LICENSING.md](LICENSING.md) for research
lineage, citation metadata, and licensing boundaries.
