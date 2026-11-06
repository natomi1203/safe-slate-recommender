#!/usr/bin/env python3
"""Verify and inventory a complete safe-slate-recommender release set."""

from __future__ import annotations

import argparse
import email
import hashlib
import json
import re
import tarfile
import zipfile
from pathlib import Path


PROJECT = "safe-slate-recommender"
DIST_PREFIX = "safe_slate_recommender"
EXPECTED_WHEEL_PLATFORMS = {
    "manylinux_2_28_x86_64",
    "manylinux_2_28_aarch64",
    "macosx_11_0_x86_64",
    "macosx_11_0_arm64",
    "win_amd64",
}
SHA_RE = re.compile(r"[0-9a-f]{40}")


class ArtifactError(ValueError):
    """Raised when a release artifact violates the distribution contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_metadata(raw: bytes, *, version: str, artifact: str) -> None:
    metadata = email.message_from_bytes(raw)
    expected = {
        "Name": PROJECT,
        "Version": version,
        "Requires-Python": ">=3.15",
        "License-Expression": "Apache-2.0",
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise ArtifactError(
                f"{artifact}: {field} must be {value!r}, got {metadata.get(field)!r}"
            )


def _check_license(raw: bytes, *, artifact: str) -> None:
    text = raw.decode("utf-8")
    required = ("Apache License", "Version 2.0, January 2004", "END OF TERMS AND CONDITIONS")
    if any(marker not in text for marker in required):
        raise ArtifactError(f"{artifact}: complete Apache-2.0 LICENSE is missing")
    if "CC-BY-4.0" in text:
        raise ArtifactError(f"{artifact}: software LICENSE contains the research license")


def _only_name(names: list[str], suffix: str, *, artifact: str) -> str:
    matches = [name for name in names if name.endswith(suffix)]
    if len(matches) != 1:
        raise ArtifactError(
            f"{artifact}: expected one {suffix!r} member, found {len(matches)}"
        )
    return matches[0]


def _check_wheel(path: Path, *, version: str, platform: str) -> None:
    expected_name = f"{DIST_PREFIX}-{version}-cp310-abi3-{platform}.whl"
    if path.name != expected_name:
        raise ArtifactError(f"unexpected wheel filename: {path.name}")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ArtifactError(f"{path.name}: duplicate archive members")
        metadata_name = _only_name(names, ".dist-info/METADATA", artifact=path.name)
        license_name = _only_name(
            names, ".dist-info/licenses/LICENSE", artifact=path.name
        )
        _check_metadata(archive.read(metadata_name), version=version, artifact=path.name)
        _check_license(archive.read(license_name), artifact=path.name)
        required_suffixes = ("/py.typed", "/_core.pyi", ".dist-info/RECORD")
        for suffix in required_suffixes:
            _only_name(names, suffix, artifact=path.name)
        native_suffixes = (".so", ".pyd", ".dylib")
        if not any(name.endswith(native_suffixes) for name in names):
            raise ArtifactError(f"{path.name}: native extension is missing")
        sboms = [
            name
            for name in names
            if ".dist-info/sboms/" in name and name.endswith(".cyclonedx.json")
        ]
        if len(sboms) != 1:
            raise ArtifactError(f"{path.name}: expected exactly one CycloneDX SBOM")


def _check_sdist(path: Path, *, version: str) -> None:
    expected_name = f"{DIST_PREFIX}-{version}.tar.gz"
    if path.name != expected_name:
        raise ArtifactError(f"unexpected sdist filename: {path.name}")
    with tarfile.open(path, "r:gz") as archive:
        names = archive.getnames()
        if len(names) != len(set(names)):
            raise ArtifactError(f"{path.name}: duplicate archive members")
        metadata_name = _only_name(names, "/PKG-INFO", artifact=path.name)
        license_name = _only_name(names, "/LICENSE", artifact=path.name)
        metadata_member = archive.extractfile(metadata_name)
        license_member = archive.extractfile(license_name)
        if metadata_member is None or license_member is None:
            raise ArtifactError(f"{path.name}: required member is not a regular file")
        _check_metadata(metadata_member.read(), version=version, artifact=path.name)
        _check_license(license_member.read(), artifact=path.name)


def verify_release(
    artifact_dir: Path,
    *,
    source_sha: str,
    version: str,
    output_dir: Path,
) -> dict[str, object]:
    if not SHA_RE.fullmatch(source_sha):
        raise ArtifactError("source_sha must be a lowercase 40-character commit SHA")

    all_archives = sorted(
        [*artifact_dir.glob("*.whl"), *artifact_dir.glob("*.tar.gz")]
    )
    if len(all_archives) != 6:
        raise ArtifactError(
            f"expected exactly six distribution archives, found {len(all_archives)}"
        )
    wheels = sorted(artifact_dir.glob(f"{DIST_PREFIX}-{version}-*.whl"))
    sdists = sorted(artifact_dir.glob(f"{DIST_PREFIX}-{version}.tar.gz"))
    archives = [*wheels, *sdists]
    if len(wheels) != 5 or len(sdists) != 1:
        raise ArtifactError(
            f"expected five wheels and one sdist, found {len(wheels)} and {len(sdists)}"
        )
    if len({path.name for path in archives}) != 6:
        raise ArtifactError("release artifact filenames are not unique")

    platforms: set[str] = set()
    wheel_prefix = f"{DIST_PREFIX}-{version}-cp310-abi3-"
    for wheel in wheels:
        if not wheel.name.startswith(wheel_prefix) or not wheel.name.endswith(".whl"):
            raise ArtifactError(f"unexpected wheel tag: {wheel.name}")
        platform = wheel.name[len(wheel_prefix) : -len(".whl")]
        if platform in platforms:
            raise ArtifactError(f"duplicate wheel platform: {platform}")
        platforms.add(platform)
        _check_wheel(wheel, version=version, platform=platform)
    if platforms != EXPECTED_WHEEL_PLATFORMS:
        raise ArtifactError(
            f"wheel platforms differ: expected {sorted(EXPECTED_WHEEL_PLATFORMS)}, "
            f"got {sorted(platforms)}"
        )
    _check_sdist(sdists[0], version=version)

    files = [
        {
            "filename": path.name,
            "sha256": _sha256(path),
            "size": path.stat().st_size,
        }
        for path in sorted(archives, key=lambda item: item.name)
    ]
    manifest: dict[str, object] = {
        "schema_version": "1.0.0",
        "project": PROJECT,
        "version": version,
        "source_sha": source_sha,
        "files": files,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    sums_path = output_dir / "SHA256SUMS"
    manifest_path = output_dir / "release-manifest.json"
    if sums_path.exists() or manifest_path.exists():
        raise ArtifactError("refusing to overwrite an existing release inventory")
    sums_path.write_text(
        "".join(f"{item['sha256']}  {item['filename']}\n" for item in files),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--version", default="0.3.1")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.artifact_dir
    manifest = verify_release(
        args.artifact_dir,
        source_sha=args.source_sha,
        version=args.version,
        output_dir=output_dir,
    )
    print(
        f"verified {len(manifest['files'])} artifacts for "
        f"{manifest['project']} {manifest['version']} at {manifest['source_sha']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
