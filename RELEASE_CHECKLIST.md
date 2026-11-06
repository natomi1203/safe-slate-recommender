# Release checklist — TommySlate 0.4.0

Version 0.4.0 migrates the PyPI distribution name from
`safe-slate-recommender` to `tommyslate`. It preserves the `safe_slate` import,
the stable public API, and the legacy intake CLI alias. Production publication
requires a separate owner approval after TestPyPI qualification.

## Source and migration contract

- [ ] Freeze the exact 40-character merge commit selected from `main`.
- [x] `pyproject.toml`, `Cargo.toml`, `CITATION.cff`, and `library.json` agree on
      version `0.4.0`; package metadata identifies distribution `tommyslate`.
- [x] The native module remains `safe_slate._core`; all seven registered public
      interfaces retain their existing import paths.
- [x] `tommyslate-freeze-inputs` and `safe-slate-freeze-inputs` resolve to the
      same implementation.
- [x] Migration documentation warns against installing both distributions in
      one environment because they provide the same Python module.
- [x] The rights holder confirms Apache-2.0 for the software distribution;
      repository-level research remains CC-BY-4.0.

## Verification for the frozen source SHA

- [ ] `uv sync --frozen` and `uv run maturin develop --release` succeed.
- [ ] Python tests, mypy, `mypy.stubtest`, and bytecode compilation pass.
- [ ] Rust tests and warning-free Clippy pass with the locked graph.
- [ ] Python and Rust coverage gates remain at 100%.
- [ ] A local wheel and sdist identify `tommyslate==0.4.0`, contain
      `safe_slate`, and pass `twine check`.
- [ ] Five platform wheels and one sdist are built from one source SHA.
- [ ] Every wheel installs into an empty Python 3.15 environment, passes
      `pip check`, imports `safe_slate`, and completes the deterministic smoke
      test.

## Index and OIDC readiness

- [ ] Recheck `tommyslate` on PyPI and TestPyPI immediately before upload.
- [ ] Register pending Trusted Publishers for project `tommyslate`, repository
      `natomi1203/safe-slate-recommender`, workflow
      `safe-slate-release.yml`, and the matching `testpypi`/`pypi` environment.
- [ ] Confirm both GitHub Environments still require owner review and remain
      restricted to `main`; do not downgrade protection.
- [ ] Run the workflow once with both publication inputs disabled and freeze
      the six-artifact manifest.
- [ ] Publish and verify the exact manifest on TestPyPI through OIDC.
- [ ] Reinstall `tommyslate==0.4.0` from TestPyPI on all five target runners.
- [ ] Obtain explicit owner approval for the production artifact hashes.

## Production publication

- [ ] Upload the exact TestPyPI-qualified files to PyPI without rebuilding.
- [ ] Match all production filenames and SHA-256 values to the frozen manifest.
- [ ] Reinstall `tommyslate==0.4.0` from PyPI on all five target runners.
- [ ] Replace candidate wording and the legacy badge with the production
      TommySlate URL and immutable release evidence.

Do not delete or overwrite `safe-slate-recommender==0.3.1`. Repository rename,
compatibility-shim publication under the old distribution name, tags, GitHub
Release creation, containers, deployments, and announcements are separate
decisions.
