#!/usr/bin/env python3
"""Fail closed when Python or Rust coverage reports miss their contracts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


PYTHON_STATEMENT_MIN = 90.0
PYTHON_BRANCH_MIN = 80.0
PYTHON_FILE_MIN = 80.0
RUST_LINE_MIN = 90.0
RUST_FUNCTION_MIN = 90.0
RUST_REGION_MIN = 90.0


class CoverageGateError(ValueError):
    """Raised when a coverage report is invalid or below its contract."""


def _read_report(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise CoverageGateError(f"cannot read coverage report {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise CoverageGateError(f"invalid coverage JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise CoverageGateError(f"coverage report {path} must contain a JSON object")
    return value


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CoverageGateError(f"{label} must be an object")
    return value


def _percentage(mapping: dict[str, Any], field: str, label: str) -> float:
    value = mapping.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CoverageGateError(f"{label}.{field} must be a finite number")
    percentage = float(value)
    if not math.isfinite(percentage) or not 0.0 <= percentage <= 100.0:
        raise CoverageGateError(f"{label}.{field} must be between 0 and 100")
    return percentage


def _require_minimum(
    failures: list[str], label: str, actual: float, minimum: float
) -> None:
    if actual < minimum:
        failures.append(f"{label}: {actual:.2f}% < {minimum:.2f}%")


def evaluate_python_report(
    report: dict[str, Any], expected_files: set[str]
) -> tuple[list[str], dict[str, float]]:
    meta = _mapping(report.get("meta"), "python.meta")
    if meta.get("format") != 3 or meta.get("branch_coverage") is not True:
        raise CoverageGateError(
            "Python coverage report must use format 3 with branch coverage enabled"
        )
    totals = _mapping(report.get("totals"), "python.totals")
    statements = _percentage(
        totals, "percent_statements_covered", "python.totals"
    )
    branches = _percentage(totals, "percent_branches_covered", "python.totals")
    files = _mapping(report.get("files"), "python.files")
    actual_files = set(files)
    missing = sorted(expected_files - actual_files)
    unexpected = sorted(actual_files - expected_files)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unexpected:
            details.append(f"unexpected={unexpected}")
        raise CoverageGateError("Python coverage file set differs: " + ", ".join(details))

    failures: list[str] = []
    _require_minimum(failures, "Python statements", statements, PYTHON_STATEMENT_MIN)
    _require_minimum(failures, "Python branches", branches, PYTHON_BRANCH_MIN)
    for filename in sorted(expected_files):
        file_report = _mapping(files[filename], f"python.files[{filename!r}]")
        summary = _mapping(
            file_report.get("summary"), f"python.files[{filename!r}].summary"
        )
        combined = _percentage(
            summary, "percent_covered", f"python.files[{filename!r}].summary"
        )
        _require_minimum(failures, f"Python file {filename}", combined, PYTHON_FILE_MIN)
    return failures, {"statements": statements, "branches": branches}


def evaluate_rust_report(
    report: dict[str, Any]
) -> tuple[list[str], dict[str, float]]:
    if report.get("type") != "llvm.coverage.json.export":
        raise CoverageGateError("Rust coverage report has an unknown type")
    version = report.get("version")
    if not isinstance(version, str) or not version.startswith("3."):
        raise CoverageGateError("Rust coverage report must use LLVM JSON schema 3.x")
    data = report.get("data")
    if not isinstance(data, list) or len(data) != 1:
        raise CoverageGateError("Rust coverage report must contain exactly one data set")
    totals = _mapping(_mapping(data[0], "rust.data[0]").get("totals"), "rust.totals")
    values: dict[str, float] = {}
    for metric in ("lines", "functions", "regions"):
        metric_data = _mapping(totals.get(metric), f"rust.totals.{metric}")
        values[metric] = _percentage(metric_data, "percent", f"rust.totals.{metric}")

    failures: list[str] = []
    _require_minimum(failures, "Rust lines", values["lines"], RUST_LINE_MIN)
    _require_minimum(
        failures, "Rust functions", values["functions"], RUST_FUNCTION_MIN
    )
    _require_minimum(failures, "Rust regions", values["regions"], RUST_REGION_MIN)
    return failures, values


def expected_python_files(source_root: Path) -> set[str]:
    files = {path.as_posix() for path in source_root.rglob("*.py") if path.is_file()}
    if not files:
        raise CoverageGateError(f"no Python source files found under {source_root}")
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-json", type=Path, required=True)
    parser.add_argument("--rust-json", type=Path, required=True)
    parser.add_argument("--python-source", type=Path, default=Path("python/safe_slate"))
    args = parser.parse_args()

    try:
        python_failures, python_values = evaluate_python_report(
            _read_report(args.python_json), expected_python_files(args.python_source)
        )
        rust_failures, rust_values = evaluate_rust_report(_read_report(args.rust_json))
    except CoverageGateError as error:
        parser.exit(1, f"coverage gate error: {error}\n")

    print(
        "Python coverage: "
        f"statements={python_values['statements']:.2f}% "
        f"branches={python_values['branches']:.2f}%"
    )
    print(
        "Rust coverage: "
        f"lines={rust_values['lines']:.2f}% "
        f"functions={rust_values['functions']:.2f}% "
        f"regions={rust_values['regions']:.2f}%"
    )
    failures = [*python_failures, *rust_failures]
    if failures:
        print("Coverage contract failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Coverage contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
