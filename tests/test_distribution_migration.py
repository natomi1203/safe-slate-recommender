from __future__ import annotations

import json
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_tommyslate_distribution_preserves_python_import_contract() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    cargo = tomllib.loads((ROOT / "Cargo.toml").read_text(encoding="utf-8"))
    library = json.loads((ROOT / "library.json").read_text(encoding="utf-8"))

    assert pyproject["project"]["name"] == "tommyslate"
    assert pyproject["project"]["version"] == "0.4.1"
    assert cargo["package"]["name"] == "tommyslate"
    assert cargo["package"]["version"] == "0.4.1"
    assert library["software_id"] == "tommyslate"
    assert library["distribution_name"] == "tommyslate"
    assert library["import_modules"] == ["safe_slate"]
    assert pyproject["tool"]["maturin"]["module-name"] == "safe_slate._core"
    assert "Development Status :: 4 - Beta" in pyproject["project"]["classifiers"]
    assert "Development Status :: 3 - Alpha" not in pyproject["project"][
        "classifiers"
    ]
    assert pyproject["project"]["urls"]["Repository"] == (
        "https://github.com/natomi1203/tommyslate"
    )
    assert cargo["package"]["repository"] == (
        "https://github.com/natomi1203/tommyslate"
    )


def test_old_and_new_cli_names_resolve_to_the_same_entry_point() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = pyproject["project"]["scripts"]

    assert scripts["tommyslate-freeze-inputs"] == "safe_slate.external_intake:main"
    assert scripts["safe-slate-freeze-inputs"] == scripts[
        "tommyslate-freeze-inputs"
    ]


def test_release_workflow_targets_only_the_new_distribution() -> None:
    workflow = (ROOT / ".github/workflows/safe-slate-release.yml").read_text(
        encoding="utf-8"
    )

    assert 'PACKAGE_NAME: "tommyslate"' in workflow
    assert 'PACKAGE_VERSION: "0.4.1"' in workflow
    assert "/p/tommyslate/" in workflow
    assert "pypi/safe-slate-recommender" not in workflow
    assert 'f"safe-slate-recommender==' not in workflow


def test_production_promotion_reuses_testpypi_qualified_artifacts() -> None:
    workflow = (ROOT / ".github/workflows/safe-slate-release.yml").read_text(
        encoding="utf-8"
    )

    assert "qualified_run_id:" in workflow
    assert "Recover exact TestPyPI-qualified artifacts" in workflow
    assert "actions: read" in workflow
    assert "TestPyPI differs from recovered manifest" in workflow
    assert "run-id: ${{ inputs.qualified_run_id }}" in workflow
    assert "inputs.qualified_run_id == ''" in workflow
    assert "needs.recover-qualified-release.result == 'success'" in workflow
    assert "needs.pypi-preflight.result == 'success'" in workflow
    assert "needs.publish-pypi.result == 'success'" in workflow
    assert "needs.verify-pypi.result == 'success'" in workflow
