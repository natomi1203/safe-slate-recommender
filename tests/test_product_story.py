from __future__ import annotations

from pathlib import Path

from safe_slate import guarded_rerank


ROOT = Path(__file__).resolve().parents[1]


def _readme_request() -> dict[str, object]:
    return {
        "request_id": "readme-example",
        "baseline_item_ids": [10, 11, 12, 13],
        "protected_count": 2,
        "remove_positions": [2, 3],
        "add_item_ids": [100, 101],
        "reward_deltas": [0.8, 0.7],
        "safety_mean_deltas": [0.04, 0.03],
        "safety_variances": [0.0001, 0.0001],
        "baseline_metric": 0.5,
        "degradation_tolerance": 0.01,
        "max_replacements": 2,
    }


def test_readme_leads_with_the_product_decision_before_engineering_terms() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    lines = readme.splitlines()

    assert 120 <= len(lines) <= 160
    assert "Improve a ranked list without risking the one that already works." in readme
    assert "It is a decision layer, not another recommendation model." in readme
    assert all(
        value in readme for value in ("[A, B, C, D]", "[X, Y]", "[A, B, X, Y]")
    )
    assert all(
        title in readme
        for title in (
            "Control change",
            "Decide under uncertainty",
            "Operate responsibly",
        )
    )
    assert "uv add tommyslate" in readme
    assert "## What TommySlate does not do" in readme
    assert readme.index("## For engineers") < readme.index("| Goal | Start with |")
    assert readme.count("```python") == 1


def test_readme_example_approves_or_keeps_the_same_ranked_list() -> None:
    request = _readme_request()
    approved = guarded_rerank(
        **request,
        support_ok=True,
        uncertainty_valid=True,
        distribution_shift_detected=False,
        proposal_log_complete=True,
    )
    kept = guarded_rerank(**request)

    assert approved.slate == (10, 11, 100, 101)
    assert approved.slate[:2] == (10, 11)
    assert approved.oracle.rechecked
    assert not approved.fallback
    assert kept.slate == (10, 11, 12, 13)
    assert kept.fallback_reason == "support_gate"
    assert kept.oracle.rechecked


def test_quickstart_uses_the_same_input_decision_output_model() -> None:
    quickstart = (ROOT / "docs" / "quickstart.md").read_text(encoding="utf-8")
    opening = quickstart.split("## Choose a certified slate", 1)[0]

    assert "current ranking, candidates, business rules, and uncertainty" in opening
    assert "approved update—or keeps the original list and records why" in opening
    assert "decision layer" in opening
    assert "uv add tommyslate" in opening
