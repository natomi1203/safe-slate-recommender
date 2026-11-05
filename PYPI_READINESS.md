# PyPI readiness

Assessment date: 2026-11-06 (Asia/Tokyo).

## Verified locally

- `safe-slate-recommender` was not present in the public PyPI JSON API at the
  time of this check. This is not a reservation; availability must be checked
  again immediately before an approved upload.
- Wheel and source distribution build from the locked checkout.
- `twine check` passes for both artifacts.
- Both artifacts install into separate empty Python 3.15 environments and
  import successfully outside the repository.
- The wheel contains the native extension, `_core.pyi`, `py.typed`, license,
  RECORD, metadata, and a CycloneDX SBOM.
- The package has no runtime Python dependency. Locked Python development/build
  dependencies and Rust dependencies had no known advisories in the local
  `pip-audit` and RustSec scans.

## Remaining release gates

- Build and smoke-test the documented Linux, macOS, and Windows wheel matrix
  from one source revision. Current repository CI verifies Python 3.15 on
  Linux, and the local clean artifact check covered macOS ARM64 only.
- Confirm the software license choice with the owner.
- Test the immutable artifacts on TestPyPI and record their SHA-256 values.
- Obtain explicit approval for upload, tag, GitHub Release, and announcement.

The package is metadata- and source-ready but not yet authorized or fully
platform-qualified for a public release. No upload or release was performed.
