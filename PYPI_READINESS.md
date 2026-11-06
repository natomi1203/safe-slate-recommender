# PyPI release evidence

Assessment date: 2026-11-06 (Asia/Tokyo).

## TommySlate 0.4.0 migration candidate

Version 0.4.0 changes only the distribution identity and migration-facing
metadata: users will install `tommyslate`, while Python continues to import
`safe_slate`. The new preferred CLI is `tommyslate-freeze-inputs`; the existing
`safe-slate-freeze-inputs` entry point remains an alias. The stable API and
runtime behavior are unchanged from 0.3.1.

The release workflow is retargeted to build five `tommyslate` wheels and one
sdist from an exact source SHA, verify metadata and hashes, qualify the same
files on TestPyPI, and only then permit production publication. New pending
Trusted Publishers are required because `tommyslate` is a distinct index
project. They must bind repository `natomi1203/safe-slate-recommender`, workflow
`safe-slate-release.yml`, and the protected `testpypi` or `pypi` environment.

The frozen source is `9594cc0fd96a667cc323a3a8e99b42b04d3100db`. A
[publication-disabled run](https://github.com/natomi1203/safe-slate-recommender/actions/runs/42222485659)
first completed all five wheel builds, local installs, smoke tests, sdist build,
metadata checks, and manifest generation.

The approved
[TestPyPI qualification run](https://github.com/natomi1203/safe-slate-recommender/actions/runs/42224307122)
then published version 0.4.0 through OIDC, matched all six remote hashes to its
manifest, and reinstalled `tommyslate==0.4.0` successfully on Linux x86-64 and
ARM64, macOS x86-64 and ARM64, and Windows x86-64. The project is available at
[TestPyPI](https://test.pypi.org/project/tommyslate/0.4.0/).

## TommySlate 0.4.0 TestPyPI manifest

| Artifact | SHA-256 |
| --- | --- |
| `tommyslate-0.4.0-cp310-abi3-macosx_11_0_arm64.whl` | `04017dd6015a91bd6aca241983123d366c000d5990a0eb6b5d0db03ed2f27a24` |
| `tommyslate-0.4.0-cp310-abi3-macosx_11_0_x86_64.whl` | `aacc49de42fa0c80c2d053c22215b68628e9c65e856e5c50b4d9c0816a0a665e` |
| `tommyslate-0.4.0-cp310-abi3-manylinux_2_28_aarch64.whl` | `e7966677c35c63ba33737c3d0dfac3eaa744eb58976d92d32c4b7d5ce06a66b8` |
| `tommyslate-0.4.0-cp310-abi3-manylinux_2_28_x86_64.whl` | `974379266174ed985a3ecc2aa535b051ee319c0fa40ec4f424d46ef41a4934a5` |
| `tommyslate-0.4.0-cp310-abi3-win_amd64.whl` | `dffd79c9f1e0c52a99ccc960d0782b0a25e34640c00fa160e6dca48f85d6b7ac` |
| `tommyslate-0.4.0.tar.gz` | `c413975ffe09e7df341cac81986944f6285ea2a20ccf5024db372e6349fdcb5d` |

Wheel archives rebuilt in a later run can have different bytes even at the
same source SHA. Production promotion therefore recovers the artifact bundle
from qualification run `42224307122`, verifies the run identity, source SHA,
manifest, local hashes and TestPyPI hashes, and uploads those exact files. It
does not rebuild the distribution for production.

## Version 0.3.1 production release

Version 0.3.1 is published on
[PyPI](https://pypi.org/project/safe-slate-recommender/0.3.1/) from source
commit `342a016d644d72da15f9fc2a8d59a70f6ffda3c2`. The approved
[release workflow](https://github.com/natomi1203/safe-slate-recommender/actions/runs/42215206667)
built one source distribution and five wheels, qualified the exact artifacts
on TestPyPI, then uploaded the same files to production PyPI without rebuilding.

Both indexes matched every filename and SHA-256 value in the frozen manifest.
The production package was reinstalled and smoke-tested successfully on Linux
x86-64/ARM64, macOS x86-64/ARM64, and Windows x86-64.

Publication used protected `testpypi` and `pypi` GitHub Environments with OIDC
Trusted Publishing. No long-lived package-index token was created. The first
two Linux x86-64 TestPyPI reinstall attempts observed stale Simple API cache
data; the unchanged workflow succeeded after the cache refreshed, with no
artifact or hash mismatch.

## Production release manifest

| Artifact | SHA-256 |
| --- | --- |
| `safe_slate_recommender-0.3.1-cp310-abi3-macosx_11_0_arm64.whl` | `c94312c517162f0f35819ec81547ac4bf69bae2e9cb1af8edf00429e33e002a7` |
| `safe_slate_recommender-0.3.1-cp310-abi3-macosx_11_0_x86_64.whl` | `58ca4bfb47cff21cfada335e3d32dfd3b1a0e9ff895d897f8ad5cd81abf8d990` |
| `safe_slate_recommender-0.3.1-cp310-abi3-manylinux_2_28_aarch64.whl` | `1f40aca49499a126cc92f56d27a12652592ac58f897bc93d546a6f6ef4e3c2ab` |
| `safe_slate_recommender-0.3.1-cp310-abi3-manylinux_2_28_x86_64.whl` | `3a645a143e005f9cbcc7dfd0cd03df9486c07736f761ce29738b1752026175f9` |
| `safe_slate_recommender-0.3.1-cp310-abi3-win_amd64.whl` | `62f3130bc5d9009bab347246e4ca0a10cac145018633ad3515ae3292dfd05352` |
| `safe_slate_recommender-0.3.1.tar.gz` | `ae6d13274ccf686472050d488245d80d2eca8f49bdd28bd25c801147a23f447b` |

The rights holder selected Apache-2.0 for this reusable software distribution;
repository-level research remains CC-BY-4.0. Tags, GitHub Releases, containers,
deployments, and announcements were not part of this publication.

The historical 0.3.0 TestPyPI qualification remains valid only for its exact
artifacts and is not the recommended installation target.
