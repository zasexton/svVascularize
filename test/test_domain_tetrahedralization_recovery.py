import importlib
from types import SimpleNamespace

import numpy as np
import pyvista as pv
import pytest

from svv.domain.domain import Domain
from svv.domain.routines.tetrahedralize import _symmetric_surface_distance
from svv.tree.tree import Tree


domain_mod = importlib.import_module("svv.domain.domain")


def _domain_with_boundary(surface):
    domain = Domain(surface)
    domain.boundary = surface.copy(deep=True)
    domain.set_random_seed(102)
    domain.set_random_generator()
    return domain


def test_domain_installs_recovered_surface_and_sampling_state(
    closed_self_intersecting_surface,
):
    source = closed_self_intersecting_surface
    source_points = np.asarray(source.points).copy()
    source_faces = np.asarray(source.faces).copy()
    domain = _domain_with_boundary(source)

    mesh = domain.get_interior(repair_max_distance_ratio=0.4)

    assert domain.mesh_build_report.selected_strategy in {"original", "meshfix"}
    if domain.mesh_build_report.attempts[0].status == "failed":
        assert domain.mesh_build_report.selected_strategy == "meshfix"
        assert [
            attempt.strategy for attempt in domain.mesh_build_report.attempts[:2]
        ] == ["original", "meshfix"]
    else:
        assert domain.mesh_build_report.attempts[0].strategy == "original"
        assert domain.mesh_build_report.attempts[0].status == "succeeded"
    assert np.array_equal(domain.original_boundary.points, source_points)
    assert np.array_equal(domain.original_boundary.faces, source_faces)
    assert np.array_equal(source.points, source_points)
    assert np.array_equal(source.faces, source_faces)

    assert domain.boundary.is_manifold
    assert domain.boundary.n_open_edges == 0
    assert domain.boundary.n_points == domain.mesh_build_report.selected_surface.n_points
    assert domain.boundary.n_cells == domain.mesh_build_report.selected_surface.n_cells
    assert np.array_equal(domain.boundary_nodes, domain.boundary.points)
    assert np.array_equal(
        domain.boundary_vertices,
        domain.boundary.faces.reshape(-1, 4)[:, 1:],
    )
    boundary_weights = np.asarray(domain.boundary.cell_data["Normalized_Area"])
    assert np.isfinite(boundary_weights).all()
    assert (boundary_weights >= 0).all()
    assert boundary_weights.sum() == pytest.approx(1.0)

    assert mesh is domain.mesh
    assert mesh.n_cells > 0
    assert set(np.unique(mesh.celltypes)).issubset(
        {int(pv.CellType.TETRA), int(pv.CellType.QUADRATIC_TETRA)}
    )
    assert np.isfinite(domain.mesh_nodes).all()
    assert domain.mesh_vertices.min() >= 0
    assert domain.mesh_vertices.max() < domain.mesh_nodes.shape[0]
    volume_weights = np.asarray(mesh.cell_data["Normalized_Volume"])
    assert np.isfinite(volume_weights).all()
    assert (volume_weights >= 0).all()
    assert volume_weights.sum() == pytest.approx(1.0)
    assert domain.cumulative_probability[-1] == pytest.approx(1.0)
    assert domain.mesh_tree is not None
    assert domain.mesh_tree_2 is not None

    extracted = mesh.extract_surface().triangulate()
    diagonal = np.linalg.norm(
        np.asarray(domain.boundary.bounds)[1::2]
        - np.asarray(domain.boundary.bounds)[::2]
    )
    assert _symmetric_surface_distance(domain.boundary, extracted) <= 0.01 * diagonal

    boundary_point = domain.get_boundary_points(1)
    assert boundary_point.shape == (1, 3)
    assert np.isfinite(boundary_point).all()

    domain.evaluate_fast = lambda points, **kwargs: -0.5 * np.ones(
        (np.asarray(points).shape[0], 1),
        dtype=float,
    )
    tree = Tree()
    tree.set_domain(domain)
    tree.set_root(max_attempts=5, attempts=20)
    assert tree.data.shape[0] == 1
    assert np.isfinite(np.asarray(tree.data)[0, :6]).all()


def test_domain_clean_surface_keeps_original_strategy_and_geometry():
    source = pv.Cube().triangulate()
    domain = _domain_with_boundary(source)

    domain.get_interior()

    assert domain.mesh_build_report.selected_strategy == "original"
    assert domain.boundary.n_points == source.n_points
    assert domain.boundary.n_cells == source.n_cells
    assert _symmetric_surface_distance(domain.boundary, source) == pytest.approx(0.0)


def test_domain_failure_does_not_install_partial_volume_state(monkeypatch):
    source = pv.Cube().triangulate()
    domain = _domain_with_boundary(source)
    previous_boundary = domain.boundary.copy(deep=True)
    previous_mesh = object()
    previous_tree = object()
    previous_tree_2 = object()
    previous_report = object()
    previous_nodes = np.array([[9.0, 8.0, 7.0]])
    previous_vertices = np.array([[0, 0, 0, 0]])
    domain.mesh = previous_mesh
    domain.mesh_tree = previous_tree
    domain.mesh_tree_2 = previous_tree_2
    domain.mesh_build_report = previous_report
    domain.mesh_nodes = previous_nodes
    domain.mesh_vertices = previous_vertices

    invalid_result = SimpleNamespace(
        grid=pv.UnstructuredGrid(),
        nodes=np.empty((0, 3)),
        elements=np.empty((0, 4), dtype=np.int64),
        surface=pv.Sphere(),
        report=object(),
    )
    monkeypatch.setattr(domain_mod, "tetrahedralize", lambda *args, **kwargs: invalid_result)

    with pytest.raises(ValueError, match="empty|non-empty"):
        domain.get_interior()

    assert domain.mesh is previous_mesh
    assert domain.mesh_tree is previous_tree
    assert domain.mesh_tree_2 is previous_tree_2
    assert domain.mesh_build_report is previous_report
    assert domain.mesh_nodes is previous_nodes
    assert domain.mesh_vertices is previous_vertices
    assert np.array_equal(domain.boundary.points, previous_boundary.points)
    assert np.array_equal(domain.boundary.faces, previous_boundary.faces)


def test_domain_rejects_worker_arrays_that_do_not_match_grid(monkeypatch):
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    cells = np.array([4, 0, 1, 2, 3], dtype=np.int64)
    grid = pv.UnstructuredGrid(
        cells,
        np.array([pv.CellType.TETRA], dtype=np.uint8),
        points,
    )
    surface = grid.extract_surface().triangulate()
    domain = _domain_with_boundary(surface)
    invalid_result = SimpleNamespace(
        grid=grid,
        nodes=np.vstack((points, [[2.0, 2.0, 2.0]])),
        elements=np.array([[0, 1, 2, 3]], dtype=np.int64),
        surface=surface,
        report=object(),
    )
    monkeypatch.setattr(domain_mod, "tetrahedralize", lambda *args, **kwargs: invalid_result)

    with pytest.raises(ValueError, match="node count"):
        domain.get_interior()

    assert domain.mesh is None
    assert domain.mesh_tree is None
    assert domain.mesh_tree_2 is None
    assert domain.mesh_build_report is None


def test_boundary_install_rejects_zero_measure_without_partial_state():
    domain = Domain(np.zeros((1, 3)))
    previous = pv.Cube().triangulate()
    domain.boundary = previous
    degenerate = pv.PolyData(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        np.array([3, 0, 1, 2]),
    )

    with pytest.raises(ValueError, match="positive finite"):
        domain._set_boundary_mesh(degenerate)

    assert domain.boundary is previous
