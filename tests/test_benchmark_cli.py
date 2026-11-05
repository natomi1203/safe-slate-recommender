from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).parents[1] / "benchmarks" / "benchmark_guarded_rerank.py"


def _command(*extra: str) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT),
        "--requests",
        "2",
        "--warmup",
        "1",
        "--baseline-size",
        "8",
        "--protected-count",
        "4",
        "--operations",
        "4",
        "--max-replacements",
        "2",
        "--timeout-ms",
        "10",
        "--beam-width",
        "8",
        "--max-expansions",
        "1000",
        "--serving-mode",
        "shadow",
        *extra,
    ]


def test_benchmark_runs_with_current_shadow_schema_and_honest_scope() -> None:
    completed = subprocess.run(
        _command(),
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

    report = json.loads(completed.stdout)
    assert report["schema_version"] == "1.2"
    assert "python_facade_latency_ms" in report
    assert "latency_ms" not in report
    assert "oracle_latency_ms" in report
    assert "excludes candidate generation" in report["claim_boundary"]
    assert report["recheck_failures"] == 0


def test_benchmark_runs_factorized_linear_uncertainty_path() -> None:
    completed = subprocess.run(
        _command("--linear-feature-dimension", "2"),
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )

    report = json.loads(completed.stdout)
    assert report["config"]["uncertainty_mode"] == "factorized_linear"
    assert report["config"]["linear_feature_dimension"] == 2
    assert "includes in-process linear uncertainty" in report["claim_boundary"]
    assert report["recheck_failures"] == 0


def test_benchmark_output_is_no_clobber(tmp_path: Path) -> None:
    output = tmp_path / "benchmark.json"
    subprocess.run(
        _command("--output", str(output)),
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    before = output.read_bytes()

    repeated = subprocess.run(
        _command("--output", str(output)),
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert repeated.returncode != 0
    assert output.read_bytes() == before
