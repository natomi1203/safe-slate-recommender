# Localhost gRPC sidecar

`safe-slate-sidecar` exposes the Rust replacement oracle to a co-located application container. It refuses non-loopback listen addresses and does not include a production model.

Required environment:

- `SAFE_SLATE_CONTRACT_SHA256`: lowercase SHA-256 of the frozen online experiment contract
- `SAFE_SLATE_MODEL_PATH`: mounted immutable model bundle matching `schemas/model_bundle.schema.json`
- `SAFE_SLATE_LISTEN_ADDR`: optional loopback address; default `127.0.0.1:50051`

The model identity returned by the service is the SHA-256 of the exact model file bytes. Every request must repeat that digest and the expected contract digest. Identity mismatch, malformed candidates, duplicate items, changed guardrails, no certified improvement, or final-recheck failure returns the unchanged baseline with `fallback=true`.

The request feature vector must follow `feature_names` in the model bundle. The sidecar computes expected 30-day settled item revenue with the frozen reward coefficients. It computes seven-day purchase probability and a delta-method contrast covariance from the frozen safety coefficients and covariance. Replacement contrasts require a 95% lower bound above the frozen 1% relative non-inferiority threshold.

The caller must apply an end-to-end 10 ms deadline. The sidecar also limits the oracle budget to the request deadline, but model scoring and serialization remain inside the caller's deadline and are not claimed as separately isolated latency.

Build the local image without publishing it:

```bash
docker build -f Dockerfile.sidecar -t safe-slate-sidecar:local .
```

The process binds only to loopback. A separate application container can
reach it only when both containers intentionally share a network namespace,
as Kubernetes containers in one Pod do. Mount the frozen model file read-only,
pass its exact contract digest, and keep the caller's unchanged-baseline
fallback outside the sidecar. Do not expose this service through a host,
cluster, or public listener.
