import pytest
import numpy as np
from svv.domain.core.m_matrix import m00, m01, m11, m_matrix


def test_m00_basic():
    """
    Test m00 function for basic 2D points.
    """
    points = np.array([[0, 0], [1, 0], [0, 1]])
    rbf_degree = 3
    result = m00(points, rbf_degree)
    expected_shape = (3, 3)  # N x N matrix
    assert result.shape == expected_shape, f"Expected shape {expected_shape}, got {result.shape}"
    # Verify diagonal elements (distance to self is zero)
    assert np.allclose(np.diag(result), 0.0), "Diagonal elements of m00 should be zero."


def test_m01_basic():
    """
    Test m01 function for basic 2D points.
    """
    points = np.array([[0, 0], [1, 0], [0, 1]])
    rbf_degree = 3
    result = m01(points, rbf_degree)
    expected_shape = (3, 6)  # N x (D*N) matrix
    assert result.shape == expected_shape, f"Expected shape {expected_shape}, got {result.shape}"
    # Verify elements related to self (should be zero due to derivative constraint)
    assert np.allclose(result[:, :2], 0.0), "Self-related elements in m01 should be zero."


def test_m11_basic():
    """
    Test m11 function for basic 2D points.
    """
    points = np.array([[0, 0], [1, 0], [0, 1]])
    rbf_degree = 3
    result = m11(points, rbf_degree)
    expected_shape = (6, 6)  # (D*N) x (D*N) matrix
    assert result.shape == expected_shape, f"Expected shape {expected_shape}, got {result.shape}"
    # Verify symmetry for indices
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            assert np.isclose(result[i, j], result[j, i]), "m11 matrix should be symmetric."


def test_m_matrix_basic():
    """
    Test m_matrix function for basic 2D points.
    """
    points = np.array([[0, 0], [1, 0], [0, 1]])
    rbf_degree = 3
    result = m_matrix(points, rbf_degree)
    n = points.shape[0]
    d = points.shape[1]
    expected_shape = (n * (d + 1), n * (d + 1))  # N*(D+1) x N*(D+1) matrix
    assert result.shape == expected_shape, f"Expected shape {expected_shape}, got {result.shape}"


def test_m00_edge_case_empty_points():
    """
    Test m00 function for an empty input.
    """
    points = np.empty((0, 2))
    rbf_degree = 3
    result = m00(points, rbf_degree)
    assert result.shape == (0, 0), "m00 with empty input should return an empty matrix."


def test_m01_edge_case_empty_points():
    """
    Test m01 function for an empty input.
    """
    points = np.empty((0, 2))
    rbf_degree = 3
    result = m01(points, rbf_degree)
    assert result.shape == (0, 0), "m01 with empty input should return an empty matrix."


def test_m11_edge_case_empty_points():
    """
    Test m11 function for an empty input.
    """
    points = np.empty((0, 2))
    rbf_degree = 3
    result = m11(points, rbf_degree)
    assert result.shape == (0, 0), "m11 with empty input should return an empty matrix."


def test_m_matrix_edge_case_empty_points():
    """
    Test m_matrix function for an empty input.
    """
    points = np.empty((0, 2))
    rbf_degree = 3
    result = m_matrix(points, rbf_degree)
    assert result.shape == (0, 0), "m_matrix with empty input should return an empty matrix."


def test_m_matrix_high_dimensional_points():
    """
    Test m_matrix function for 3D points.
    """
    points = np.random.rand(4, 3)  # 4 points in 3D
    rbf_degree = 3
    result = m_matrix(points, rbf_degree)
    n = points.shape[0]
    d = points.shape[1]
    expected_shape = (n * (d + 1), n * (d + 1))  # N*(D+1) x N*(D+1) matrix
    assert result.shape == expected_shape, f"Expected shape {expected_shape}, got {result.shape}"


def test_m_matrix_symmetry():
    """
    Verify symmetry of the full M matrix.
    """
    points = np.random.rand(4, 2)  # 4 points in 2D
    rbf_degree = 3
    result = m_matrix(points, rbf_degree)
    assert np.allclose(result, result.T), "Full M matrix should be symmetric."
