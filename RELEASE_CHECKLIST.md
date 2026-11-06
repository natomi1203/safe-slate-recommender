# Release checklist

No step in this file authorizes publication. PyPI upload, GitHub Release
creation, tag signing, and external announcements require explicit owner
approval.

## Source and scope

- [x] Release source is commit
      `12b491f734961833a181438a51100d391d00c4f5` on the dedicated repository's
      `main` branch.
- [x] `SOFTWARE_PROVENANCE.md` and `library.json` identify the frozen source
      revision; frozen research and publication evidence are unchanged.
- [x] The diff contains no paper, theorem ownership, fixed result, credentials,
      private URI, or row-level research data.
- [x] Version agrees in `pyproject.toml`, `Cargo.toml`, and `CHANGELOG.md`.
- [x] The rights holder confirms Apache-2.0 for this software distribution;
      repository-level research and publication artifacts remain CC-BY-4.0.

## Verification

- [x] `uv sync --frozen` succeeds on the latest Python 3.15 maintenance release.
- [x] `uv run maturin develop --release` succeeds.
- [x] `uv run pytest` passes, including native parity, randomized invariants,
      fail-closed boundaries, and no-clobber tests.
- [x] Full-package mypy, `mypy.stubtest`, and Python bytecode compilation pass.
- [x] `cargo test --all-targets` and `cargo clippy --all-targets -- -D warnings`
      pass with the locked Rust dependency graph.
- [x] A wheel and sdist build from a clean checkout; both install into empty
      environments and import `safe_slate` without the repository on
      `PYTHONPATH`.
- [x] Wheel contents include `py.typed`, `_core.pyi`, the native extension,
      license, metadata, RECORD, and the generated CycloneDX SBOM.
- [x] Linux x86-64/ARM64, macOS x86-64/ARM64, and Windows x86-64 wheels are
      built from one revision and smoke-tested on Python 3.15.
- [x] Dependency vulnerability review covers `Cargo.lock`, `uv.lock`, and build
      dependencies; unresolved findings are recorded with an owner and date.
- [x] Candidate-pool and guarded-rerank benchmarks are run in no-clobber mode;
      results are labeled synthetic and are not used as production claims.

## Package index readiness

- [x] The distribution name is rechecked on PyPI and TestPyPI immediately
      before upload.
- [x] `twine check` or an equivalent metadata validator passes for every wheel
      and the sdist.
- [x] The release workflow runs from its reviewed default-branch definition,
      builds an exact source SHA, and publishes through the protected
      `testpypi` environment with OIDC Trusted Publishing.
- [x] TestPyPI exposes exactly five platform wheels and one sdist whose
      filenames and SHA-256 values match the frozen release manifest.
- [x] TestPyPI installation is smoke-tested on every supported runner.
- [x] The owner approves the final artifact hashes and TestPyPI command through
      the protected deployment review.

## Production publication, only after separate approval

- [ ] Upload immutable artifacts without rebuilding.
- [ ] Create and sign the tag and GitHub Release from the approved revision.
- [ ] Verify install from PyPI in a new Python 3.15 environment.
- [ ] Record artifact hashes, index URLs, and any residual platform gaps in the
      PR and changelog.
