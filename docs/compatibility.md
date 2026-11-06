# Compatibility and versioning

## Version policy

The distribution uses Semantic Versioning. While the major version is `0`, a
minor release may change provisional exports, but the seven symbols listed in
`library.json` remain source-compatible within the 0.4 series. Patch releases
may tighten validation when previously accepted inputs were ambiguous,
non-finite, silently coerced, or capable of bypassing a fail-closed boundary.

Schema versions embedded in logs, contracts, and evidence are independent of
the package version. Readers may add support for a new schema without changing
the meaning of an old one. Existing canonical bytes and digests are immutable.

## Python and native ABI

- Minimum and CI interpreter: Python 3.15.
- Native implementation: Rust 2024 edition, minimum Rust 1.85.
- The extension uses CPython's stable ABI. Wheel metadata still enforces
  `Requires-Python >=3.15`; the lower ABI tag does not expand supported Python
  versions.
- Supported wheel targets for a public release: Linux x86-64 and ARM64,
  macOS x86-64 and ARM64, and Windows x86-64. Each artifact must be built from
  the same signed source revision and pass isolated import and smoke tests.

Until the release workflow builds and verifies that complete matrix, source
installs require a compatible Rust toolchain and are not the preferred public
installation path.

## Migration rules

Version 0.4.0 changes the PyPI distribution name from
`safe-slate-recommender` to `tommyslate`. It does not rename the Python package:
applications continue to import `safe_slate`. The preferred intake command is
`tommyslate-freeze-inputs`; `safe-slate-freeze-inputs` remains an equivalent
compatibility alias throughout the 0.4 series. Do not install both
distributions in one environment because they provide the same Python module.

- Never reinterpret an existing safety threshold, digest encoding, or evidence
  schema in place.
- Add a new versioned encoding or schema, retain the old reader, and record the
  selected version in every audit artifact.
- Deprecations must appear in the changelog for at least one minor release
  before removal.
- A change that can alter a previously certified decision requires a minor
  version, explicit migration notes, and native/Python parity tests.
