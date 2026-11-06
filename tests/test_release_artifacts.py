from __future__ import annotations

import importlib.util
import io
import json
import tarfile
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
VERSION = "0.3.1"
SOURCE_SHA = "1" * 40

SPEC = importlib.util.spec_from_file_location(
    "verify_release_artifacts", ROOT / "scripts" / "verify_release_artifacts.py"
)
assert SPEC is not None and SPEC.loader is not None
release_artifacts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release_artifacts)
EXPECTED_WHEEL_PLATFORMS = release_artifacts.EXPECTED_WHEEL_PLATFORMS
ArtifactError = release_artifacts.ArtifactError
verify_release = release_artifacts.verify_release


def _metadata() -> bytes:
    return (
        "Metadata-Version: 2.4\n"
        "Name: safe-slate-recommender\n"
        f"Version: {VERSION}\n"
        "Requires-Python: >=3.15\n"
        "License-Expression: Apache-2.0\n\n"
    ).encode()


def _write_wheel(path: Path, license_text: bytes) -> None:
    dist_info = f"safe_slate_recommender-{VERSION}.dist-info"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{dist_info}/METADATA", _metadata())
        archive.writestr(f"{dist_info}/licenses/LICENSE", license_text)
        archive.writestr(f"{dist_info}/RECORD", "")
        archive.writestr(
            f"{dist_info}/sboms/safe-slate-recommender.cyclonedx.json", "{}"
        )
        archive.writestr("safe_slate/_core.pyi", "")
        archive.writestr("safe_slate/_core.abi3.so", b"native")
        archive.writestr("safe_slate/py.typed", "")


def _write_sdist(path: Path, license_text: bytes) -> None:
    root = f"safe_slate_recommender-{VERSION}"
    with tarfile.open(path, "w:gz") as archive:
        for name, content in (
            (f"{root}/PKG-INFO", _metadata()),
            (f"{root}/LICENSE", license_text),
        ):
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))


def _write_release_set(directory: Path) -> None:
    license_text = (ROOT / "LICENSE").read_bytes()
    for platform in EXPECTED_WHEEL_PLATFORMS:
        _write_wheel(
            directory
            / f"safe_slate_recommender-{VERSION}-cp310-abi3-{platform}.whl",
            license_text,
        )
    _write_sdist(
        directory / f"safe_slate_recommender-{VERSION}.tar.gz", license_text
    )


def test_release_inventory_binds_six_artifacts_to_source_sha(tmp_path: Path) -> None:
    _write_release_set(tmp_path)

    manifest = verify_release(
        tmp_path,
        source_sha=SOURCE_SHA,
        version=VERSION,
        output_dir=tmp_path,
    )

    assert manifest["source_sha"] == SOURCE_SHA
    assert len(manifest["files"]) == 6
    written = json.loads((tmp_path / "release-manifest.json").read_text())
    assert written == manifest
    assert len((tmp_path / "SHA256SUMS").read_text().splitlines()) == 6


def test_release_inventory_rejects_non_apache_wheel(tmp_path: Path) -> None:
    _write_release_set(tmp_path)
    wheel = next(tmp_path.glob("*manylinux_2_28_x86_64.whl"))
    _write_wheel(wheel, b"Creative Commons Attribution 4.0")

    with pytest.raises(ArtifactError, match="Apache-2.0 LICENSE"):
        verify_release(
            tmp_path,
            source_sha=SOURCE_SHA,
            version=VERSION,
            output_dir=tmp_path,
        )
