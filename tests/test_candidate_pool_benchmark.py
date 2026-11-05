from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


SCRIPT = Path(__file__).parents[1] / "benchmarks" / "benchmark_candidate_pool.py"


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
        "--candidates",
        "3",
        "--feature-dimension",
        "2",
        "--max-replacements",
        "2",
        "--timeout-ms",
        "20",
        "--beam-width",
        "8",
        "--max-expansions",
        "1000",
        "--serving-mode",
        "shadow",
        *extra,
    ]


def test_candidate_pool_benchmark_measures_full_local_facade() -> None:
    completed = subprocess.run(
        _command(),
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    report = json.loads(completed.stdout)

    assert report["schema_version"] == "1.0"
    assert report["config"]["expected_operations"] == 12
    assert report["operation_count"] == {"min": 12, "max": 12}
    assert "candidate_pool_facade_latency_ms" in report
    assert "operation_build_latency_ms" in report
    assert "projection_and_digest_latency_ms" in report
    assert "oracle_latency_ms" in report
    assert report["recheck_failures"] == 0
    assert "not production latency" in report["claim_boundary"]


def test_candidate_pool_benchmark_output_is_no_clobber(tmp_path: Path) -> None:
    output = tmp_path / "candidate-pool-benchmark.json"
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


def test_candidate_pool_benchmark_runs_600_by_560_shape() -> None:
    completed = subprocess.run(
        _command(
            "--requests",
            "1",
            "--warmup",
            "0",
            "--baseline-size",
            "600",
            "--protected-count",
            "560",
            "--candidates",
            "40",
            "--feature-dimension",
            "4",
            "--max-replacements",
            "40",
            "--max-expansions",
            "64",
            "--timeout-ms",
            "10",
        ),
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    report = json.loads(completed.stdout)
    assert report["config"]["baseline_size"] == 600
    assert report["config"]["protected_count"] == 560
    assert report["config"]["expected_operations"] == 1600
    assert report["operation_count"] == {"min": 1600, "max": 1600}
    assert report["recheck_failures"] == 0
