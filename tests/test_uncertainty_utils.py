from __future__ import annotations

import math

import pytest

from safe_slate import (
    canonical_matrix_bytes,
    canonical_matrix_f64le_bytes,
    matrix_f64le_sha256,
    matrix_sha256,
    matrix_sha256_for_encoding,
)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("matrix", "sequence of rows"),
        (["row"], "numeric sequences"),
        ([[True]], "finite numeric"),
        ([[math.nan]], "finite numeric"),
        ([[1.0], [1.0, 2.0]], "equal length"),
    ],
)
def test_canonical_matrix_encodings_reject_malformed_values(value, message: str) -> None:
    for encoder in (canonical_matrix_bytes, canonical_matrix_f64le_bytes):
        with pytest.raises(ValueError, match=message):
            encoder(value)


def test_matrix_encodings_cover_empty_zero_width_and_hash_dispatch() -> None:
    assert canonical_matrix_bytes([]) == b"[]"
    assert canonical_matrix_f64le_bytes([]).endswith(b"\x00" * 16)
    assert canonical_matrix_f64le_bytes([[], []]).endswith(
        b"\x02\x00\x00\x00\x00\x00\x00\x00" + b"\x00" * 8
    )
    matrix = [[1.0, 2.0], [3.0, 4.0]]
    assert matrix_sha256_for_encoding(matrix, "json-v1") == matrix_sha256(matrix)
    assert matrix_sha256_for_encoding(matrix, "f64le-v1") == matrix_f64le_sha256(matrix)
    with pytest.raises(ValueError, match="matrix digest encoding"):
        matrix_sha256_for_encoding(matrix, "unknown")
