#!/usr/bin/env python3
"""Production-shape latency/fallback benchmark for guarded re-ranking."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
import json
import math
import platform
import random
import sys
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import Any, Sequence

from safe_slate import guarded_linear_rerank, guarded_rerank


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    ordered = sorted(values)
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return ordered[index]


def make_request(
    rng: random.Random,
    *,
    request_index: int,
    baseline_size: int,
    protected_count: int,
    operation_count: int,
    max_replacements: int,
    timeout_ms: int,
    beam_width: int,
    max_expansions: int,
    serving_mode: str,
    linear_feature_dimension: int,
) -> dict[str, Any]:
    mutable_count = baseline_size - protected_count
    if mutable_count <= 0:
        raise ValueError("benchmark requires at least one mutable baseline position")
    remove_positions = [
        protected_count + (index % mutable_count) for index in range(operation_count)
    ]
    # Every add ID is unique, leaving remove-side uniqueness as the active
    # balanced-operation cap.
    add_item_ids = [
        1_000_000 + request_index * operation_count + index
        for index in range(operation_count)
    ]
    reward_deltas = [rng.uniform(-0.02, 0.10) for _ in range(operation_count)]
    safety_mean_deltas = [rng.uniform(0.001, 0.004) for _ in range(operation_count)]
    request = {
        "request_id": f"benchmark-{request_index}",
        "event_timestamp": (
            datetime(2026, 1, 1, tzinfo=UTC) + timedelta(microseconds=request_index)
        ).isoformat(),
        "model_version": "benchmark-v1",
        "baseline_item_ids": list(range(baseline_size)),
        "protected_count": protected_count,
        "remove_positions": remove_positions,
        "add_item_ids": add_item_ids,
        "reward_deltas": reward_deltas,
        "safety_mean_deltas": safety_mean_deltas,
        "baseline_metric": 0.10,
        "degradation_tolerance": 0.01,
        "max_replacements": max_replacements,
        "serving_mode": serving_mode,
        "confidence_radius": 1.96,
        "beam_width": beam_width,
        "max_expansions": max_expansions,
        "timeout_ms": timeout_ms,
        "context": {"surface": "synthetic_benchmark"},
    }
    if linear_feature_dimension > 0:
        contrast_features = []
        for operation in range(operation_count):
            row = [0.0] * linear_feature_dimension
            row[operation % linear_feature_dimension] = rng.uniform(0.005, 0.015)
            contrast_features.append(row)
        request.update(
            {
                "uncertainty_snapshot_id": "synthetic-factorized-linear-v1",
                "contrast_features": contrast_features,
                "parameter_covariance": [
                    [
                        1e-4 if row == column else 0.0
                        for column in range(linear_feature_dimension)
                    ]
                    for row in range(linear_feature_dimension)
                ],
                "support_ok": True,
                "uncertainty_valid": True,
                "distribution_shift_detected": False,
                "proposal_log_complete": True,
            }
        )
    else:
        request["safety_variances"] = [
            rng.uniform(0.2e-6, 1.0e-6) for _ in range(operation_count)
        ]
    return request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=1_000)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--baseline-size", type=int, default=600)
    parser.add_argument("--protected-count", type=int, default=560)
    parser.add_argument("--operations", type=int, default=200)
    parser.add_argument("--max-replacements", type=int, default=40)
    parser.add_argument("--timeout-ms", type=int, default=10)
    parser.add_argument("--beam-width", type=int, default=128)
    parser.add_argument("--max-expansions", type=int, default=100_000)
    parser.add_argument("--linear-feature-dimension", type=int, default=0)
    parser.add_argument("--serving-mode", choices=("serve", "shadow"), default="shadow")
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    for name in (
        "requests",
        "baseline_size",
        "operations",
        "beam_width",
        "max_expansions",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if (
        args.warmup < 0
        or args.timeout_ms < 0
        or args.max_replacements < 0
        or args.linear_feature_dimension < 0
    ):
        parser.error("warmup, timeout, and replacement limits must be nonnegative")
    if not 0 <= args.protected_count < args.baseline_size:
        parser.error("protected count must leave at least one mutable position")
    return args


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    latencies_ms: list[float] = []
    oracle_latencies_ms: list[float] = []
    selected_counts: list[int] = []
    status_counts: Counter[str] = Counter()
    fallback_count = 0
    deadline_count = 0
    recheck_failures = 0

    total = args.warmup + args.requests
    for request_index in range(total):
        request = make_request(
            rng,
            request_index=request_index,
            baseline_size=args.baseline_size,
            protected_count=args.protected_count,
            operation_count=args.operations,
            max_replacements=args.max_replacements,
            timeout_ms=args.timeout_ms,
            beam_width=args.beam_width,
            max_expansions=args.max_expansions,
            serving_mode=args.serving_mode,
            linear_feature_dimension=args.linear_feature_dimension,
        )
        started = time.perf_counter()
        result = (
            guarded_linear_rerank(**request)
            if args.linear_feature_dimension > 0
            else guarded_rerank(**request)
        )
        elapsed_ms = (time.perf_counter() - started) * 1_000.0
        if request_index < args.warmup:
            continue
        latencies_ms.append(elapsed_ms)
        oracle_latencies_ms.append(result.oracle.elapsed_seconds * 1_000.0)
        selected_counts.append(result.selected_count)
        status_counts[result.status] += 1
        fallback_count += int(result.fallback)
        deadline_count += int(result.oracle.deadline_reached)
        recheck_failures += int(not result.oracle.rechecked)

    report = {
        "schema_version": "1.2",
        "library_version": version("tommyslate"),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "config": {
            "requests": args.requests,
            "warmup": args.warmup,
            "baseline_size": args.baseline_size,
            "protected_count": args.protected_count,
            "operations": args.operations,
            "max_replacements": args.max_replacements,
            "timeout_ms": args.timeout_ms,
            "beam_width": args.beam_width,
            "max_expansions": args.max_expansions,
            "linear_feature_dimension": args.linear_feature_dimension,
            "uncertainty_mode": (
                "factorized_linear"
                if args.linear_feature_dimension > 0
                else "diagonal_operation_variance"
            ),
            "serving_mode": args.serving_mode,
            "seed": args.seed,
        },
        "python_facade_latency_ms": {
            "p50": percentile(latencies_ms, 0.50),
            "p95": percentile(latencies_ms, 0.95),
            "p99": percentile(latencies_ms, 0.99),
            "max": max(latencies_ms),
        },
        "oracle_latency_ms": {
            "p50": percentile(oracle_latencies_ms, 0.50),
            "p95": percentile(oracle_latencies_ms, 0.95),
            "p99": percentile(oracle_latencies_ms, 0.99),
            "max": max(oracle_latencies_ms),
        },
        "fallback_rate": fallback_count / args.requests,
        "deadline_rate": deadline_count / args.requests,
        "recheck_failures": recheck_failures,
        "selected_count": {
            "mean": sum(selected_counts) / args.requests,
            "min": min(selected_counts),
            "max": max(selected_counts),
        },
        "status_counts": dict(sorted(status_counts.items())),
        "claim_boundary": (
            "Synthetic process-local Python facade plus Rust oracle only; excludes "
            "candidate generation, model inference, RPC, queueing, and downstream "
            "logging. Factorized mode includes in-process linear uncertainty "
            "preparation; neither mode is production p95, business lift, or "
            "calibration evidence."
        ),
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        if not args.output.parent.is_dir():
            raise ValueError("benchmark output parent must already exist")
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
