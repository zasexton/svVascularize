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


def test_domain_rejects_quadratic_tetrahedra_from_first_order_worker(monkeypatch):
    corner_points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    edge_points = np.array(
        [
            [0.5, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ]
    )
    points = np.vstack((corner_points, edge_points))
    elements = np.arange(10, dtype=np.int64).reshape(1, 10)
    grid = pv.UnstructuredGrid(
        np.concatenate(([10], elements[0])),
        np.array([pv.CellType.QUADRATIC_TETRA], dtype=np.uint8),
        points,
    )
    surface = grid.extract_surface().triangulate()
    domain = _domain_with_boundary(surface)
    result = SimpleNamespace(
        grid=grid,
        nodes=points,
        elements=elements,
        surface=surface,
        report=object(),
    )
    monkeypatch.setattr(domain_mod, "tetrahedralize", lambda *args, **kwargs: result)

    with pytest.raises(ValueError, match="first-order.*M, 4"):
        domain.get_interior()

    assert domain.mesh is None
    assert domain.mesh_build_report is None


def test_domain_rejects_raw_tetgen_switches_before_worker(monkeypatch):
    domain = _domain_with_boundary(pv.Cube().triangulate())

    def unexpected_worker(*args, **kwargs):
        raise AssertionError("Raw switches must not reach the Domain TetGen worker")

    monkeypatch.setattr(domain_mod, "tetrahedralize", unexpected_worker)

    with pytest.raises(ValueError, match="switches.*order=1.*nobisect"):
        domain.get_interior(switches="pq1.2")


def test_boundary_rebuild_failure_cannot_leave_stale_volume_state(monkeypatch):
    domain = _domain_with_boundary(pv.Cube().triangulate())
    stale = object()
    domain.mesh = stale
    domain.mesh_nodes = stale
    domain.mesh_vertices = stale
    domain.mesh_tree = stale
    domain.mesh_tree_2 = stale
    domain.all_mesh_cells = stale
    domain.cumulative_probability = stale
    domain.characteristic_length = 2.0
    domain.area = 3.0
    domain.volume = 4.0
    domain.convexity = 0.5
    domain.mesh_build_report = stale
    domain.random_points = stale

    replacement = pv.Sphere(theta_resolution=8, phi_resolution=8).triangulate()
    domain.original_boundary = replacement
    replacement_grid = object()
    monkeypatch.setattr(
        domain_mod,
        "contour",
        lambda *args, **kwargs: (replacement.copy(deep=True), replacement_grid),
    )

    boundary, grid = domain.get_boundary(25)

    assert grid is replacement_grid
    assert boundary.n_points == replacement.n_points
    assert domain.mesh is None
    assert domain.mesh_nodes is None
    assert domain.mesh_vertices is None
    assert domain.mesh_tree is None
    assert domain.mesh_tree_2 is None
    assert domain.all_mesh_cells is None
    assert domain.cumulative_probability is None
    assert domain.characteristic_length is None
    assert domain.area is None
    assert domain.volume is None
    assert domain.convexity is None
    assert domain.mesh_build_report is None
    assert domain.random_points is None

    monkeypatch.setattr(
        domain_mod,
        "tetrahedralize",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("worker failed")),
    )
    with pytest.raises(RuntimeError, match="worker failed"):
        domain.get_interior()

    assert domain.mesh is None
    assert domain.mesh_build_report is None
    assert domain.boundary.n_points == replacement.n_points


def test_cached_build_rechecks_mesh_after_boundary_replacement(monkeypatch):
    domain = _domain_with_boundary(pv.Cube().triangulate())
    domain.patches = []
    domain.PTS = np.zeros((1, 1, 1, 1, 1, 3))
    domain.function_tree = object()
    domain.random_generator = object()
    domain.boundary = None
    domain.mesh = object()
    domain.mesh_tree = object()
    domain.mesh_tree_2 = object()
    calls = []
    replacement = pv.Sphere(theta_resolution=8, phi_resolution=8).triangulate()

    def rebuild_boundary(resolution):
        domain._set_boundary_mesh(replacement)
        domain.grid = object()
        return domain.boundary, domain.grid

    def rebuild_interior(**kwargs):
        calls.append("interior")
        domain.mesh = object()
        domain.mesh_tree = object()
        domain.mesh_tree_2 = object()
        return domain.mesh

    monkeypatch.setattr(domain, "get_boundary", rebuild_boundary)
    monkeypatch.setattr(domain, "get_interior", rebuild_interior)

    domain.build(resolution=25)

    assert calls == ["interior"]
    assert domain.mesh is not None
    assert domain.mesh_tree is not None


def test_domain_rejects_zero_volume_selected_surface(monkeypatch):
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
    grid = pv.UnstructuredGrid(
        np.array([4, 0, 1, 2, 3], dtype=np.int64),
        np.array([pv.CellType.TETRA], dtype=np.uint8),
        points,
    )
    domain = _domain_with_boundary(pv.Cube().triangulate())
    result = SimpleNamespace(
        grid=grid,
        nodes=points,
        elements=elements,
        surface=pv.Plane().triangulate(),
        report=object(),
    )
    monkeypatch.setattr(domain_mod, "tetrahedralize", lambda *args, **kwargs: result)
    monkeypatch.setattr(domain_mod, "validate_recovery_surface", lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match="selected surface volume.*positive.*finite"):
        domain.get_interior()

    assert domain.mesh is None
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
