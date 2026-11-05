from __future__ import annotations

from hashlib import sha256

import pytest

from safe_slate import (
    canonical_matrix_bytes,
    canonical_matrix_f64le_bytes,
    matrix_f64le_sha256,
    matrix_sha256,
    matrix_sha256_for_encoding,
    prepare_linear_contrast_marginals_f64le,
)


def test_f64le_identity_is_deterministic_shape_bound_and_finite() -> None:
    matrix = [[1.0, -0.0], [2.5, 3.25]]
    encoded = canonical_matrix_f64le_bytes(matrix)

    assert encoded.startswith(b"safe-slate-matrix-f64le-v1\x00")
    assert matrix_f64le_sha256(matrix) == sha256(encoded).hexdigest()
    assert matrix_sha256_for_encoding(matrix, "f64le-v1") == matrix_f64le_sha256(
        matrix
    )
    assert matrix_sha256_for_encoding(matrix, "json-v1") == matrix_sha256(matrix)
    assert canonical_matrix_bytes(matrix) != encoded
    assert matrix_f64le_sha256([[1.0, 2.0]]) != matrix_f64le_sha256(
        [[1.0], [2.0]]
    )


@pytest.mark.parametrize(
    "matrix",
    [
        [[float("nan")]],
        [[float("inf")]],
        [[1.0], [1.0, 2.0]],
        [[True]],
    ],
)
def test_f64le_identity_rejects_malformed_matrices(matrix) -> None:
    with pytest.raises(ValueError):
        canonical_matrix_f64le_bytes(matrix)


def test_matrix_digest_rejects_unknown_encoding() -> None:
    with pytest.raises(ValueError, match="encoding"):
        matrix_sha256_for_encoding([[1.0]], "unknown")


def test_rust_preparation_matches_python_f64le_identities() -> None:
    contrasts = [[1.0, -0.0], [0.5, 2.0]]
    covariance = [[0.04, 0.01], [0.01, 0.09]]
    prepared = prepare_linear_contrast_marginals_f64le(contrasts, covariance)

    assert prepared.matrix_digest_encoding == "f64le-v1"
    assert prepared.feature_dimension == 2
    assert prepared.safety_variances == pytest.approx([0.04, 0.39])
    assert prepared.contrast_features_sha256 == matrix_f64le_sha256(contrasts)
    assert prepared.parameter_covariance_sha256 == matrix_f64le_sha256(covariance)
