# TommySlate

[![CI](https://github.com/natomi1203/tommyslate/actions/workflows/ci.yml/badge.svg)](https://github.com/natomi1203/tommyslate/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/tommyslate.svg)](https://pypi.org/project/tommyslate/)
[![Python 3.15](https://img.shields.io/badge/Python-3.15-3776AB.svg?logo=python&logoColor=white)](docs/compatibility.md)
[![Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**Change the ranking—not the safety contract.**

TommySlate is a fail-closed slate re-ranker for Python, powered by a Rust
search core. It helps recommendation, feed, and search teams improve part of a
ranked list while protecting the baseline, respecting business constraints,
and leaving an audit trail.

When evidence is missing or a final check fails, TommySlate does not guess. It
returns the unchanged baseline and records why.

```text
                         gates fail
baseline + candidates ───────────────▶ unchanged baseline + reason
          │
          └─ gates pass ─▶ bounded search ─▶ final recheck ─▶ safer slate
                                                   │
                                                   └─ fails ─▶ baseline
```

## Why TommySlate?

| You need to… | TommySlate gives you… |
| --- | --- |
| Keep trusted results fixed | Protected prefixes and balanced replacements |
| Search inside a latency budget | An anytime incumbent with explicit lower/upper certificates |
| Enforce real slate rules | Group caps, conflicts, and whole-slate predicates |
| Account for shared model risk | Diagonal, covariance, and factorized uncertainty paths |
| Fail safely | Baseline fallback, strict validation, and a mandatory final recheck |
| Explain the decision later | Proposal logs, shadow evidence, and no-clobber OPE artifacts |

## Install

Python 3.15 is required. Add TommySlate with uv:

```bash
uv add tommyslate
```

The package has no runtime Python dependencies. Its stable import is
`safe_slate`.

Migrating from the old distribution name? Replace it rather than installing
both packages, because both provide the same Python module:

```bash
uv remove safe-slate-recommender
uv add tommyslate
```

## See it work

Protect the first two results and safely improve only the remaining positions:

```python
from safe_slate import optimize_replacements

result = optimize_replacements(
    baseline_item_ids=[10, 11, 12, 13],
    baseline_size=4,
    protected_count=2,
    remove_positions=[2, 2, 3, 3],
    add_item_ids=[100, 101, 100, 102],
    reward_deltas=[0.8, 0.7, 0.6, 0.5],
    safety_means=[0.04, 0.03, 0.02, 0.01],
    safety_variances=[0.0001] * 4,
    max_replacements=2,
)

assert result.rechecked
assert result.materialized_slate == [10, 11, 100, 102]
print(result.status, result.k_lower, result.k_upper)  # exact 2 2
```

The protected prefix stays untouched. TommySlate chooses a compatible set of
remove/add operations, maximizes certified replacement count first and reward
second, then rechecks the materialized slate before returning it.

## Pick the right entry point

| Goal | Start with |
| --- | --- |
| Optimize a new slate | `optimize_slate` |
| Protect part of an existing slate | `optimize_replacements` |
| Add fail-closed serving gates | `guarded_rerank` |
| Expand baseline/candidate arrays into operations | `guarded_candidate_pool_rerank` |
| Decide whether serving evidence is ready | `assess_serving_gates` |
| Check OPE inputs before evaluation | `audit_ope_readiness` |
| Produce immutable evaluation evidence | `run_no_clobber_ope` |

The [API reference](docs/api-reference.md) defines the stable surface. For a
guided first run, use the [quickstart](docs/quickstart.md).

## Designed to fail closed

- A failed serving gate returns the registered baseline with a reason.
- Invalid configuration raises before a decision can be served.
- A deadline returns only a final-rechecked incumbent—or the baseline.
- Missing uncertainty, support, or logging evidence never becomes permission.
- Evidence writers refuse to overwrite an existing destination.

TommySlate validates the decision contract you provide. It does **not** invent
calibration, prove causal lift, establish population support, or authorize a
production rollout. Those claims still require evidence from your system.

## Documentation

- [Quickstart](docs/quickstart.md) — install and make a first certified choice.
- [API reference](docs/api-reference.md) — stable functions and contracts.
- [Full guide](docs/full-guide.md) — guarded serving, uncertainty, shadow mode,
  OPE, evidence bundles, and benchmarks.
- [gRPC sidecar](docs/sidecar.md) — loopback deployment and model contract.
- [Compatibility](docs/compatibility.md) — versions, ABI, platforms, and
  migration rules.
- [Research boundary](docs/research-integration.md) — what comes from the
  author's research, third-party background, and excluded work.

## Supported release targets

TommySlate ships CPython stable-ABI wheels for Python 3.15 on Linux x86-64 and
ARM64, macOS x86-64 and ARM64, and Windows x86-64. Every public release is
built from one source revision, checked on TestPyPI, matched by SHA-256, and
reinstalled on all five runners before production promotion.

## Development

```bash
uv sync --frozen
uv run maturin develop --release
uv run pytest
```

The Python and Rust coverage gates are 100%. See the
[full guide](docs/full-guide.md#production-shape-benchmark) for synthetic
benchmark commands and their claim boundaries.

## Research and license

TommySlate extracts reusable software contracts from
[`purchase-aware-retrieval-research`](https://github.com/natomi1203/purchase-aware-retrieval-research).
The [research lineage](docs/research-lineage.json) distinguishes the author's
adopted contracts from third-party theoretical background and work that is not
part of the stable API.

The software is Apache-2.0 licensed. Research papers and publication artifacts
in the source research repository remain CC-BY-4.0. See
[`CITATION.cff`](CITATION.cff) for citation metadata and
[`LICENSING.md`](LICENSING.md) for the boundary.
