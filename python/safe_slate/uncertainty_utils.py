"""Canonical finite-matrix identities shared by serving and evidence audits."""

from __future__ import annotations

from hashlib import sha256
import json
from math import isfinite
from numbers import Real
import struct
from typing import Sequence


MATRIX_DIGEST_JSON_V1 = "json-v1"
MATRIX_DIGEST_F64LE_V1 = "f64le-v1"


def canonical_matrix_bytes(matrix: Sequence[Sequence[float]]) -> bytes:
    """Serialize a rectangular finite matrix with one canonical JSON encoding."""

    if not isinstance(matrix, Sequence) or isinstance(matrix, (str, bytes)):
        raise ValueError("matrix must be a sequence of rows")
    normalized: list[list[float]] = []
    width: int | None = None
    for row in matrix:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise ValueError("matrix rows must be numeric sequences")
        values: list[float] = []
        for value in row:
            if (
                not isinstance(value, Real)
                or isinstance(value, bool)
                or not isfinite(value)
            ):
                raise ValueError("matrix values must be finite numeric values")
            values.append(float(value))
        if width is None:
            width = len(values)
        elif len(values) != width:
            raise ValueError("matrix rows must have equal length")
        normalized.append(values)
    return json.dumps(
        normalized,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def matrix_sha256(matrix: Sequence[Sequence[float]]) -> str:
    """Return the canonical lowercase SHA-256 identity of a finite matrix."""

    return sha256(canonical_matrix_bytes(matrix)).hexdigest()


def canonical_matrix_f64le_bytes(matrix: Sequence[Sequence[float]]) -> bytes:
    """Serialize a finite matrix as shape-bound little-endian IEEE-754 data."""

    if not isinstance(matrix, Sequence) or isinstance(matrix, (str, bytes)):
        raise ValueError("matrix must be a sequence of rows")
    row_count = len(matrix)
    width: int | None = None
    body = bytearray()
    pack_row = None
    for row in matrix:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise ValueError("matrix rows must be numeric sequences")
        values: list[float] = []
        for value in row:
            if (
                not isinstance(value, Real)
                or isinstance(value, bool)
                or not isfinite(value)
            ):
                raise ValueError("matrix values must be finite numeric values")
            values.append(float(value))
        if width is None:
            width = len(values)
            pack_row = struct.Struct(f"<{width}d").pack if width else None
        elif len(values) != width:
            raise ValueError("matrix rows must have equal length")
        if pack_row is not None:
            body.extend(pack_row(*values))
    column_count = width or 0
    encoded = bytearray(b"safe-slate-matrix-f64le-v1\x00")
    encoded.extend(struct.pack("<QQ", row_count, column_count))
    encoded.extend(body)
    return bytes(encoded)


def matrix_f64le_sha256(matrix: Sequence[Sequence[float]]) -> str:
    """Return the fast shape-bound f64-le matrix identity."""

    return sha256(canonical_matrix_f64le_bytes(matrix)).hexdigest()


def matrix_sha256_for_encoding(
    matrix: Sequence[Sequence[float]],
    encoding: str,
) -> str:
    """Hash a matrix using one explicitly versioned canonical encoding."""

    if encoding == MATRIX_DIGEST_JSON_V1:
        return matrix_sha256(matrix)
    if encoding == MATRIX_DIGEST_F64LE_V1:
        return matrix_f64le_sha256(matrix)
    raise ValueError(
        "matrix digest encoding must be 'json-v1' or 'f64le-v1'"
    )
