from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from safe_slate import (
    attach_service_latency,
    create_service_latency_trace,
    load_proposal_log,
    load_proposal_logs_jsonl,
)
from test_shadow import _shadow_log


def test_proposal_log_round_trips_from_canonical_dictionary() -> None:
    original = _shadow_log("r0")

    loaded = load_proposal_log(original.to_dict())

    assert loaded == original
    assert loaded.to_dict() == original.to_dict()


def test_complete_service_latency_trace_round_trips() -> None:
    original = attach_service_latency(
        _shadow_log("r0"),
        create_service_latency_trace(
            response_latency_ms=20.0,
            log_commit_latency_ms=25.0,
            candidate_generation_ms=3.0,
            model_inference_ms=5.0,
            queueing_ms=1.0,
            rpc_ms=2.0,
            guardrail_facade_ms=10.0,
            serialization_ms=1.0,
            proposal_log_write_ms=2.0,
        ),
    )

    loaded = load_proposal_log(original.to_dict())

    assert loaded == original
    assert loaded.service_latency_trace is not None
    assert loaded.service_latency_trace.complete


def test_noncanonical_service_latency_trace_is_rejected() -> None:
    payload = attach_service_latency(
        _shadow_log("r0"),
        create_service_latency_trace(
            response_latency_ms=20.0,
            log_commit_latency_ms=25.0,
        ),
    ).to_dict()
    payload["service_latency_trace"]["complete"] = True

    with pytest.raises(ValueError, match="completeness"):
        load_proposal_log(payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(schema_version="1.1"), "schema"),
        (lambda value: value.pop("request_id"), "fields"),
        (lambda value: value.update(extra="field"), "fields"),
        (lambda value: value.update(event_timestamp="2026-10-01T12:00:00"), "timezone"),
        (lambda value: value.update(elapsed_seconds=math.inf), "finite"),
        (lambda value: value.update(selected_flags=[False, False]), "selected flags"),
        (lambda value: value.update(reward_deltas=[0.1]), "inconsistent"),
        (lambda value: value.update(covariance=[[1.0]]), "covariance shape"),
        (lambda value: value.update(context={"bad": math.nan}), "JSON object"),
    ],
)
def test_malformed_proposal_log_is_rejected(mutation, message: str) -> None:
    payload = _shadow_log("r0").to_dict()
    mutation(payload)

    with pytest.raises(ValueError, match=message):
        load_proposal_log(payload)


def test_jsonl_loader_is_bounded_and_preserves_order(tmp_path: Path) -> None:
    records = [_shadow_log("r0"), _shadow_log("r1")]
    source = tmp_path / "shadow.jsonl"
    source.write_text(
        "".join(json.dumps(record.to_dict()) + "\n" for record in records),
        encoding="utf-8",
    )

    loaded = load_proposal_logs_jsonl(
        source,
        maximum_records=2,
        maximum_bytes=source.stat().st_size,
    )

    assert loaded == tuple(records)

    with pytest.raises(ValueError, match="maximum_records"):
        load_proposal_logs_jsonl(
            source,
            maximum_records=1,
            maximum_bytes=source.stat().st_size,
        )
    with pytest.raises(ValueError, match="maximum_bytes"):
        load_proposal_logs_jsonl(
            source,
            maximum_records=2,
            maximum_bytes=source.stat().st_size - 1,
        )


@pytest.mark.parametrize("contents", ["", "\n", "{bad json}\n"])
def test_jsonl_loader_rejects_empty_blank_or_invalid_sources(
    tmp_path: Path, contents: str
) -> None:
    source = tmp_path / "invalid.jsonl"
    source.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError):
        load_proposal_logs_jsonl(source, maximum_records=10, maximum_bytes=1_000)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("maximum_records", 0),
        ("maximum_records", 1.5),
        ("maximum_bytes", 0),
        ("maximum_bytes", True),
    ],
)
def test_jsonl_loader_rejects_invalid_resource_caps(
    tmp_path: Path, field: str, value: object
) -> None:
    source = tmp_path / "shadow.jsonl"
    source.write_text(json.dumps(_shadow_log("r0").to_dict()) + "\n", encoding="utf-8")
    values = {"maximum_records": 1, "maximum_bytes": source.stat().st_size}
    values[field] = value

    with pytest.raises(ValueError):
        load_proposal_logs_jsonl(source, **values)  # type: ignore[arg-type]
