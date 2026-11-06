# PyPI readiness

Assessment date: 2026-11-06 (Asia/Tokyo).

## Version 0.3.1 production candidate

Version 0.3.1 is the production candidate because `main` contains software
changes made after the TestPyPI-only 0.3.0 qualification. Reusing 0.3.0 would
produce different immutable artifacts under an existing version, so the
release workflow must build 0.3.1 from one exact `main` commit.

The workflow first publishes and verifies the six 0.3.1 artifacts on TestPyPI,
including installation on all five supported runners. Only then can the
protected `pypi` environment release the same downloaded artifact set to
production through OIDC. It rebuilds nothing between indexes and rejects any
remote filename whose SHA-256 differs from the release manifest.

Production PyPI publication is approved. Tags, GitHub Releases, containers,
deployments, and announcements are not part of this approval.

## Verified locally and on TestPyPI

- `safe-slate-recommender` was not present in the public PyPI JSON API at the
  time of this check. This is not a reservation; availability must be checked
  again immediately before an approved production upload.
- The rights holder selected Apache-2.0 for this reusable software
  distribution. Repository-level research remains CC-BY-4.0.
- Wheel and source distribution build from the locked checkout and pass
  `twine check`.
- The wheel contains the native extension, `_core.pyi`, `py.typed`, license,
  RECORD, metadata, and a CycloneDX SBOM.
- The package has no runtime Python dependency. Locked Python development/build
  dependencies and Rust dependencies had no known advisories in the local
  `pip-audit` and RustSec scans.
- The dedicated public repository uses a protected `testpypi` GitHub
  Environment restricted to `main`, with Naoyuki Tomita as the required
  reviewer. Publication uses a matching TestPyPI Trusted Publisher and no
  long-lived API token.
- The reviewed default-branch workflow built one sdist and five wheels from
  source commit `12b491f734961833a181438a51100d391d00c4f5` for Linux
  x86-64/ARM64, macOS x86-64/ARM64, and Windows x86-64.
- The approved OIDC qualification run published version `0.3.0` to
  [TestPyPI](https://test.pypi.org/project/safe-slate-recommender/0.3.0/),
  matched every remote SHA-256 to the release manifest, and reinstalled the
  exact version successfully on all five target runners. See
  [workflow run 42144510365](https://github.com/natomi1203/safe-slate-recommender/actions/runs/42144510365).

## TestPyPI release manifest

| Artifact | SHA-256 |
| --- | --- |
| `safe_slate_recommender-0.3.0-cp310-abi3-macosx_11_0_arm64.whl` | `61a1d06f57f4916d072c082ba494dc1937b57b6cca6366e5328bfa1f026f9681` |
| `safe_slate_recommender-0.3.0-cp310-abi3-macosx_11_0_x86_64.whl` | `f8777a7088e333758426d6d8a4a22d9ab80594950f1d2e746880aace400c55a8` |
| `safe_slate_recommender-0.3.0-cp310-abi3-manylinux_2_28_aarch64.whl` | `f7874913ef4dd3c7528e9e42ec4a630bca043711108970ef121792e212099bfa` |
| `safe_slate_recommender-0.3.0-cp310-abi3-manylinux_2_28_x86_64.whl` | `394e7a362c5a5cc17d361e3c863fae5243fcf79e1a8c68c136426354c5569c65` |
| `safe_slate_recommender-0.3.0-cp310-abi3-win_amd64.whl` | `cd4fa725f3b7a3e0a4ec7eb05a09dc2cbc698978f668060e78fd2fadacc7173d` |
| `safe_slate_recommender-0.3.0.tar.gz` | `0541324c7f1c95167c653c09d7139cc74d00ddb5d139cb4e1996ad0ac566b0c1` |

## Remaining 0.3.1 production release gates

- Merge the reviewed 0.3.1 release workflow and freeze its exact source SHA.
- Configure the protected `pypi` GitHub Environment and matching pending PyPI
  Trusted Publisher without creating a long-lived token.
- Recheck the production PyPI name and immutable artifact set immediately
  before upload.
- Complete TestPyPI and production PyPI hash matching plus five-platform
  installation checks in one workflow run.

The historical 0.3.0 TestPyPI qualification remains valid for those exact
artifacts. No production upload, tag, GitHub Release, or announcement has yet
been performed.
