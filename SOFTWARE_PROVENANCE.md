# Software split provenance

This reusable distribution is extracted from the
[`purchase-aware-retrieval-research`](https://github.com/natomi1203/purchase-aware-retrieval-research)
PR #8 research freeze `1a8b4290755f00c1fb78a69602af336ebe7e7742`
under tag `pr8-research-freeze-2026-11-06`.

It contains the stable Python API, typed package marker, Rust backend, API and
integration tests, compatibility checks, and reusable evidence/serving
interfaces.  It does not contain a paper, theorem ownership, or fixed
paper-specific result.  Publication PRs pin a future merged software revision;
their paper-specific simulations and immutable aggregate results remain with
the paper.

The three-way extraction decision and the excluded research topics are
recorded in [`docs/research-integration.md`](docs/research-integration.md).
The corresponding machine-readable map is
[`docs/research-lineage.json`](docs/research-lineage.json); it pins the
publication-registry snapshot separately from the frozen implementation
revision so that later paper-status changes cannot silently rewrite the
software's extraction history. These files are software scope maps, not
publication results.

Software citation metadata is recorded in [`CITATION.cff`](CITATION.cff).
Third-party papers listed in the lineage are theoretical background only; no
code was imported from them, and their citation is not an originality claim
for either the library or the repository-authored manuscripts.

The reusable distribution is licensed under Apache-2.0 by its rights holder.
Repository-level papers, evidence, protocols, and publication artifacts remain
under CC-BY-4.0. See `LICENSING.md` for the exact boundary; this licensing
decision does not alter the frozen source revision named above.
