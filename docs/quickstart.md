# Quickstart

`safe-slate-recommender` requires Python 3.15 or newer. It ships a compiled
Rust extension, so install a wheel that matches your operating system and CPU.

```bash
python3.15 -m pip install safe-slate-recommender
```

The package has no runtime Python dependencies. Until a release is published,
build and install the wheel from a trusted checkout as described in the
[release checklist](../RELEASE_CHECKLIST.md).

## Choose a certified slate

The unchanged baseline is represented by the empty selection. A candidate is
admissible only when its lower confidence bound is at least
`safety_threshold`. The optimizer maximizes certified cardinality first and
reward second.

```python
from safe_slate import optimize_slate

result = optimize_slate(
    rewards=[0.9, 0.7, 0.4],
    safety_means=[0.08, 0.05, 0.01],
    safety_variances=[0.0004, 0.0004, 0.0001],
    max_items=2,
    group_ids=[0, 0, 1],
    group_caps={0: 1, 1: 1},
    timeout_ms=50,
)

assert result.rechecked
print(result.selected_indices)
print(result.k_lower, result.k_upper, result.status)
```

Always inspect `rechecked`. `exact=False` means the returned incumbent is safe
but the search did not close the cardinality interval before its budget ended.
`k_lower` is certified; `k_upper` is the remaining structural upper bound.

## Protect an existing slate

Use `optimize_replacements` when the first part of a baseline slate must never
change. Each operation pairs one removable position with one new item. The
native constraints prevent two selected operations from removing the same
position or inserting the same item.

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

print(result.materialized_slate)
```

## Production rule

The low-level optimizers validate mathematical inputs and always perform a
final recheck, but they do not establish that your uncertainty model or
logging process is valid. For serving, use `guarded_rerank`; its runtime gates
default to closed. A failed gate returns the unchanged baseline and records a
reason. Invalid configuration raises `ValueError` before a decision is served.

See [API reference](api-reference.md), [compatibility policy](compatibility.md),
and [research integration boundary](research-integration.md).
