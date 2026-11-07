from __future__ import annotations

import json
from pathlib import Path
import re
import tomllib


ROOT = Path(__file__).resolve().parents[1]
LINEAGE_PATH = ROOT / "docs" / "research-lineage.json"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
LOCAL_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _root_cff_scalars() -> dict[str, str]:
    result: dict[str, str] = {}
    for line in (ROOT / "CITATION.cff").read_text(encoding="utf-8").splitlines():
        if not line or line[0].isspace() or ":" not in line:
            continue
        key, raw = line.split(":", 1)
        value = raw.strip().strip('"').strip("'")
        if value and value not in {">", ">-", "|", "|-"}:
            result[key] = value
    return result


def test_research_lineage_matches_registered_public_contract() -> None:
    lineage = _load_json(LINEAGE_PATH)
    library = _load_json(ROOT / "library.json")

    assert lineage["schema_version"] == "1.0.0"
    source = lineage["source"]
    assert isinstance(source, dict)
    assert source["repository"] == (
        "https://github.com/natomi1203/purchase-aware-retrieval-research"
    )
    assert isinstance(source["frozen_revision"], str)
    assert HEX40.fullmatch(source["frozen_revision"])
    assert isinstance(source["registry_snapshot_revision"], str)
    assert HEX40.fullmatch(source["registry_snapshot_revision"])
    assert source["frozen_revision"] == library["source_revision"]
    assert library["research_lineage"] == "docs/research-lineage.json"
    assert library["citation_file"] == "CITATION.cff"

    adopted = lineage["adopted_research"]
    assert isinstance(adopted, list)
    adopted_ids = [entry["paper_id"] for entry in adopted]
    assert len(adopted_ids) == len(set(adopted_ids))
    assert set(adopted_ids) == set(library["supports_papers"])

    public_api = set(library["public_api"])
    capability_ids: list[str] = []
    for paper in adopted:
        assert paper["adoption_state"] == "stable_contracts_adopted"
        assert paper["publication_status_at_snapshot"] in {
            "blocked_external_evidence",
            "human_review_pending",
            "published",
        }
        assert paper["capabilities"]
        for capability in paper["capabilities"]:
            capability_ids.append(capability["capability_id"])
            assert set(capability["interfaces"]) <= public_api
            assert capability["adopted_contract"]
            assert capability["non_claim"]
    assert len(capability_ids) == len(set(capability_ids))


def test_background_and_excluded_research_cannot_enter_the_stable_surface() -> None:
    lineage = _load_json(LINEAGE_PATH)
    library = _load_json(ROOT / "library.json")

    foundations = lineage["external_foundations"]
    assert isinstance(foundations, list)
    reference_ids = [entry["reference_id"] for entry in foundations]
    assert len(reference_ids) == len(set(reference_ids))
    for entry in foundations:
        assert entry["role"] == "background_only"
        assert entry["direct_code_import"] is False
        assert entry["url"].startswith("https://")
        assert entry["relevance"]

    excluded = lineage["excluded_research"]
    assert isinstance(excluded, list)
    excluded_ids = [entry["paper_id"] for entry in excluded]
    assert len(excluded_ids) == len(set(excluded_ids))
    assert set(excluded_ids).isdisjoint(library["supports_papers"])
    for entry in excluded:
        assert entry["status_at_snapshot"] in {
            "human_review_pending",
            "research_in_progress",
            "draft_unreviewed",
        }
        assert entry["adoption_state"] == "excluded_pending_independent_review"

    excluded_implementations = lineage["excluded_implementations"]
    assert isinstance(excluded_implementations, list)
    implementation_ids = [
        entry["implementation_id"] for entry in excluded_implementations
    ]
    assert len(implementation_ids) == len(set(implementation_ids))
    assert all(
        entry["disposition"] == "research_only"
        for entry in excluded_implementations
    )


def test_citation_metadata_matches_package_metadata() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    citation = _root_cff_scalars()

    assert citation["cff-version"] == "1.2.0"
    library = _load_json(ROOT / "library.json")

    assert citation["title"] == library["display_name"]
    assert library["distribution_name"] == project["name"]
    assert citation["version"] == project["version"]
    assert citation["license"] == project["license"]
    assert citation["repository-code"] == project["urls"]["Repository"]
    assert citation["url"] == project["urls"]["Repository"]


def test_research_documentation_local_links_resolve() -> None:
    documents = (
        ROOT / "README.md",
        ROOT / "SOFTWARE_PROVENANCE.md",
        ROOT / "docs" / "quickstart.md",
        ROOT / "docs" / "full-guide.md",
        ROOT / "docs" / "research-integration.md",
    )
    for document in documents:
        text = document.read_text(encoding="utf-8")
        for target in LOCAL_LINK.findall(text):
            if "://" in target or target.startswith("#"):
                continue
            path_text = target.split("#", 1)[0]
            assert (document.parent / path_text).exists(), (
                f"broken local link in {document.relative_to(ROOT)}: {target}"
            )
