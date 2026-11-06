# PyPI release evidence

Assessment date: 2026-11-06 (Asia/Tokyo).

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
