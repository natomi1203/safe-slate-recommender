from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_coverage", ROOT / "scripts" / "check_coverage.py"
)
assert SPEC is not None and SPEC.loader is not None
coverage_gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(coverage_gate)

PYTHON_FILE = "python/safe_slate/example.py"


def _python_report(
    *, statements: float = 90.0, branches: float = 80.0, file_total: float = 80.0
) -> dict[str, object]:
    return {
        "meta": {"format": 3, "branch_coverage": True},
        "totals": {
            "percent_statements_covered": statements,
            "percent_branches_covered": branches,
        },
        "files": {
            PYTHON_FILE: {"summary": {"percent_covered": file_total}},
        },
    }


def _rust_report(
    *, lines: float = 100.0, functions: float = 100.0, regions: float = 100.0
) -> dict[str, object]:
    return {
        "type": "llvm.coverage.json.export",
        "version": "3.1.0",
        "data": [
            {
                "totals": {
                    "lines": {"percent": lines},
                    "functions": {"percent": functions},
                    "regions": {"percent": regions},
                }
            }
        ],
    }


def test_exact_thresholds_pass() -> None:
    python_failures, _ = coverage_gate.evaluate_python_report(
        _python_report(), {PYTHON_FILE}
    )
    rust_failures, _ = coverage_gate.evaluate_rust_report(_rust_report())
    assert python_failures == []
    assert rust_failures == []


def test_one_hundredth_below_global_thresholds_fails() -> None:
    python_failures, _ = coverage_gate.evaluate_python_report(
        _python_report(statements=89.99, branches=79.99), {PYTHON_FILE}
    )
    rust_failures, _ = coverage_gate.evaluate_rust_report(
        _rust_report(lines=99.99, functions=99.99, regions=99.99)
    )
    failures = [*python_failures, *rust_failures]
    assert len(failures) == 5
    assert any("Python statements" in failure for failure in failures)
    assert any("Rust functions" in failure for failure in failures)


def test_python_file_floor_is_enforced() -> None:
    failures, _ = coverage_gate.evaluate_python_report(
        _python_report(file_total=79.99), {PYTHON_FILE}
    )
    assert failures == [f"Python file {PYTHON_FILE}: 79.99% < 80.00%"]


def test_python_report_requires_the_exact_source_file_set() -> None:
    with pytest.raises(coverage_gate.CoverageGateError, match="file set differs"):
        coverage_gate.evaluate_python_report(
            _python_report(), {PYTHON_FILE, "python/safe_slate/missing.py"}
        )


@pytest.mark.parametrize(
    "report, message",
    [
        (
            {**_python_report(), "meta": {"format": 2, "branch_coverage": True}},
            "format 3",
        ),
        ({**_rust_report(), "version": "2.0.0"}, "schema 3.x"),
    ],
)
def test_unknown_report_schema_is_rejected(
    report: dict[str, object], message: str
) -> None:
    with pytest.raises(coverage_gate.CoverageGateError, match=message):
        if "meta" in report:
            coverage_gate.evaluate_python_report(report, {PYTHON_FILE})
        else:
            coverage_gate.evaluate_rust_report(report)


def test_missing_and_corrupt_json_fail_closed(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(coverage_gate.CoverageGateError, match="cannot read"):
        coverage_gate._read_report(missing)

    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not-json", encoding="utf-8")
    with pytest.raises(coverage_gate.CoverageGateError, match="invalid coverage JSON"):
        coverage_gate._read_report(corrupt)


def test_expected_python_files_discovers_only_python_sources(tmp_path: Path) -> None:
    package = tmp_path / "python" / "safe_slate"
    package.mkdir(parents=True)
    (package / "module.py").write_text("value = 1\n", encoding="utf-8")
    (package / "module.pyi").write_text("value: int\n", encoding="utf-8")
    assert coverage_gate.expected_python_files(tmp_path / "python" / "safe_slate") == {
        (package / "module.py").as_posix()
    }
