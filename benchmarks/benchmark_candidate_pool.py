#!/usr/bin/env python3
"""Synthetic candidate-pool latency and fallback frontier."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import UTC, datetime, timedelta
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import random
import sys
import time
from typing import Any, Sequence

from safe_slate import guarded_candidate_pool_rerank


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
    candidate_count: int,
    feature_dimension: int,
    max_replacements: int,
    timeout_ms: int,
    beam_width: int,
    max_expansions: int,
    serving_mode: str,
) -> dict[str, Any]:
    baseline = list(range(baseline_size))
    candidates = [
        1_000_000 + request_index * candidate_count + index
        for index in range(candidate_count)
    ]
    baseline_rewards = [
        1.0 - 0.8 * index / max(1, baseline_size - 1)
        for index in range(baseline_size)
    ]
    candidate_rewards = [
        rng.uniform(0.25, 0.65) for _ in range(candidate_count)
    ]
    baseline_safety = [rng.uniform(0.045, 0.055) for _ in baseline]
    candidate_safety = [rng.uniform(0.048, 0.060) for _ in candidates]

    def feature_row() -> list[float]:
        return [rng.uniform(-0.02, 0.02) for _ in range(feature_dimension)]

    return {
        "request_id": f"candidate-pool-benchmark-{request_index}",
        "event_timestamp": (
            datetime(2026, 1, 1, tzinfo=UTC) + timedelta(microseconds=request_index)
        ).isoformat(),
        "uncertainty_snapshot_id": "synthetic-candidate-pool-covariance-v1",
        "contrast_input_id": f"synthetic-request-{request_index}-contrasts",
        "model_version": "synthetic-candidate-pool-v1",
        "baseline_item_ids": baseline,
        "baseline_reward_scores": baseline_rewards,
        "baseline_safety_means": baseline_safety,
        "baseline_features": [feature_row() for _ in baseline],
        "candidate_item_ids": candidates,
        "candidate_reward_scores": candidate_rewards,
        "candidate_safety_means": candidate_safety,
        "candidate_features": [feature_row() for _ in candidates],
        "parameter_covariance": [
            [1e-4 if row == column else 0.0 for column in range(feature_dimension)]
            for row in range(feature_dimension)
        ],
        "protected_count": protected_count,
        "baseline_metric": 0.10,
        "degradation_tolerance": 0.01,
        "max_replacements": max_replacements,
        "serving_mode": serving_mode,
        "confidence_radius": 1.96,
        "support_ok": True,
        "uncertainty_valid": True,
        "distribution_shift_detected": False,
        "proposal_log_complete": True,
        "beam_width": beam_width,
        "max_expansions": max_expansions,
        "timeout_ms": timeout_ms,
        "context": {"surface": "synthetic_candidate_pool_benchmark"},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=1_000)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--baseline-size", type=int, default=600)
    parser.add_argument("--protected-count", type=int, default=560)
    parser.add_argument("--candidates", type=int, default=40)
    parser.add_argument("--feature-dimension", type=int, default=8)
    parser.add_argument("--max-replacements", type=int, default=40)
    parser.add_argument("--timeout-ms", type=int, default=10)
    parser.add_argument("--beam-width", type=int, default=128)
    parser.add_argument("--max-expansions", type=int, default=100_000)
    parser.add_argument("--serving-mode", choices=("serve", "shadow"), default="shadow")
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    for name in (
        "requests",
        "baseline_size",
        "candidates",
        "feature_dimension",
        "beam_width",
        "max_expansions",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.warmup < 0 or args.timeout_ms < 0 or args.max_replacements < 0:
        parser.error("warmup, timeout, and replacement limits must be nonnegative")
    if not 0 <= args.protected_count < args.baseline_size:
        parser.error("protected count must leave at least one mutable position")
    return args


def _latency_summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    facade_latencies: list[float] = []
    build_latencies: list[float] = []
    projection_digest_latencies: list[float] = []
    oracle_latencies: list[float] = []
    selected_counts: list[int] = []
    operation_counts: list[int] = []
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
            candidate_count=args.candidates,
            feature_dimension=args.feature_dimension,
            max_replacements=args.max_replacements,
            timeout_ms=args.timeout_ms,
            beam_width=args.beam_width,
            max_expansions=args.max_expansions,
            serving_mode=args.serving_mode,
        )
        started = time.perf_counter()
        result = guarded_candidate_pool_rerank(**request)
        elapsed_ms = (time.perf_counter() - started) * 1_000.0
        if request_index < args.warmup:
            continue
        candidate_metadata = result.proposal_log.context["safe_slate_candidate_pool"]
        uncertainty_metadata = result.proposal_log.context["safe_slate_uncertainty"]
        facade_latencies.append(elapsed_ms)
        build_latencies.append(candidate_metadata["operation_build_elapsed_ms"])
        projection_digest_latencies.append(uncertainty_metadata["pre_oracle_elapsed_ms"])
        oracle_latencies.append(result.oracle.elapsed_seconds * 1_000.0)
        selected_counts.append(result.selected_count)
        operation_counts.append(candidate_metadata["operation_count"])
        status_counts[result.status] += 1
        fallback_count += int(result.fallback)
        deadline_count += int(result.oracle.deadline_reached)
        recheck_failures += int(not result.oracle.rechecked)

    report = {
        "schema_version": "1.0",
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
            "replaceable_positions": args.baseline_size - args.protected_count,
            "candidates": args.candidates,
            "expected_operations": (
                (args.baseline_size - args.protected_count) * args.candidates
            ),
            "feature_dimension": args.feature_dimension,
            "matrix_digest_encoding": "f64le-v1",
            "max_replacements": args.max_replacements,
            "timeout_ms": args.timeout_ms,
            "beam_width": args.beam_width,
            "max_expansions": args.max_expansions,
            "serving_mode": args.serving_mode,
            "seed": args.seed,
        },
        "candidate_pool_facade_latency_ms": _latency_summary(facade_latencies),
        "operation_build_latency_ms": _latency_summary(build_latencies),
        "projection_and_digest_latency_ms": _latency_summary(
            projection_digest_latencies
        ),
        "oracle_latency_ms": _latency_summary(oracle_latencies),
        "fallback_rate": fallback_count / args.requests,
        "deadline_rate": deadline_count / args.requests,
        "recheck_failures": recheck_failures,
        "selected_count": {
            "mean": sum(selected_counts) / args.requests,
            "min": min(selected_counts),
            "max": max(selected_counts),
        },
        "operation_count": {
            "min": min(operation_counts),
            "max": max(operation_counts),
        },
        "status_counts": dict(sorted(status_counts.items())),
        "claim_boundary": (
            "Synthetic process-local candidate-pool construction, linear "
            "uncertainty preparation, Python facade, and Rust oracle only. It "
            "excludes upstream retrieval/model inference, RPC, queueing, hard "
            "service cancellation, durable log writes, calibrated production "
            "residuals, user traffic, and business outcomes. These measurements "
            "are an environment-specific development frontier, not production "
            "latency, production safety, or purchase-lift evidence."
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
