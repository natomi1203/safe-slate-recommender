# Security policy

Please report suspected vulnerabilities privately to the repository owner
through GitHub's private vulnerability reporting feature. Do not include user
data, credentials, private storage locations, or production traces in an issue.

The library treats malformed inputs, missing runtime evidence, uncertainty
validation failures, predicate errors, and incomplete search as no-change or
rechecked-incumbent outcomes. This is not a substitute for a process-level
service timeout, resource isolation, authentication, authorization, or privacy
review in the embedding system.

Release artifacts must be built from locked dependencies, include their SBOM,
and be matched to an approved source revision by SHA-256. See
`RELEASE_CHECKLIST.md`.
