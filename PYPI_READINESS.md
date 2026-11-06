# PyPI readiness

Assessment date: 2026-11-06 (Asia/Tokyo).

## Verified locally

- `safe-slate-recommender` was not present in the public PyPI JSON API at the
  time of this check. This is not a reservation; availability must be checked
  again immediately before an approved upload.
- The same project name was not present in the TestPyPI JSON API at the time of
  this check. A pending Trusted Publisher does not reserve either name.
- The rights holder selected Apache-2.0 for this reusable software
  distribution. Repository-level research remains CC-BY-4.0.
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

- Merge the separately governed release workflow, then build and smoke-test
  Linux x86-64/ARM64, macOS x86-64/ARM64, and Windows x86-64 wheels from one
  exact source revision.
- Configure a protected `testpypi` GitHub Environment and the matching pending
  TestPyPI Trusted Publisher; no long-lived API token is used.
- Publish the frozen artifacts to TestPyPI, verify every remote SHA-256, and
  repeat the smoke test from the package index on all five runners.
- Obtain separate explicit approval before any PyPI upload, tag, GitHub
  Release, or announcement.

The package is authorized for a TestPyPI qualification run only. It is not yet
platform-qualified or authorized for a production PyPI release. No upload or
release was performed as of this assessment.
