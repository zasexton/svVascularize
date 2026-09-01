import numpy as np
import pyvista as pv
import pytest

from svv.domain.domain import Domain
from svv.domain.routines.tetrahedralize import _symmetric_surface_distance


def test_issue_102_attachment_completes_domain_build(issue_102_stl_path):
    surface = pv.read(issue_102_stl_path)
    source_points = np.asarray(surface.points).copy()
    source_faces = np.asarray(surface.faces).copy()
    source_diagonal = np.linalg.norm(
        np.asarray(surface.bounds)[1::2] - np.asarray(surface.bounds)[::2]
    )

    domain = Domain(surface)
    domain.create()
    domain.solve()
    domain.build(resolution=25)

    assert domain.mesh is not None
    assert domain.mesh.n_cells > 0
    assert domain.mesh_tree is not None
    assert domain.mesh_tree_2 is not None
    assert domain.mesh_build_report.selected_strategy == "meshfix"
    assert np.isfinite(domain.mesh_nodes).all()
    assert domain.mesh_vertices.min() >= 0
    assert domain.mesh_vertices.max() < domain.mesh_nodes.shape[0]

    probabilities = np.asarray(domain.mesh.cell_data["Normalized_Volume"])
    assert np.isfinite(probabilities).all()
    assert (probabilities >= 0).all()
    assert probabilities.sum() == pytest.approx(1.0)
    assert domain.cumulative_probability[-1] == pytest.approx(1.0)

    cell_volumes = np.asarray(domain.mesh.cell_data["Volume"])
    total_volume = float(cell_volumes.sum())
    assert np.isfinite(total_volume)
    assert total_volume > 0
    selected_volume = abs(float(domain.boundary.volume))
    assert abs(total_volume - selected_volume) / selected_volume <= 0.005

    assert domain.boundary.is_manifold
    assert domain.boundary.n_open_edges == 0
    extracted = domain.mesh.extract_surface().triangulate()
    assert _symmetric_surface_distance(domain.boundary, extracted) <= 0.01 * source_diagonal

    assert np.array_equal(domain.original_boundary.points, source_points)
    assert np.array_equal(domain.original_boundary.faces, source_faces)
    assert np.array_equal(surface.points, source_points)
    assert np.array_equal(surface.faces, source_faces)

    boundary_point = domain.get_boundary_points(1)
    assert boundary_point.shape == (1, 3)
    assert np.isfinite(boundary_point).all()
