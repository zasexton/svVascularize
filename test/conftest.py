import hashlib
import os
import sys
from pathlib import Path

import pyvista as pv
import pytest

# Add the parent directory to PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def closed_self_intersecting_surface():
    """Small closed manifold whose folded cap intersects the lower surface."""

    surface = pv.Sphere(theta_resolution=20, phi_resolution=20)
    surface.points[surface.points[:, 2] > 0.2, 2] -= 0.8
    regions = surface.connectivity().cell_data["RegionId"]

    assert surface.is_all_triangles
    assert int(regions.max()) + 1 == 1
    assert surface.is_manifold
    assert surface.n_open_edges == 0
    return surface.copy(deep=True)


@pytest.fixture
def issue_102_stl_path():
    """Return the verified issue attachment path when explicitly provided."""

    configured = os.environ.get("SVV_ISSUE_102_STL")
    if not configured:
        pytest.skip("Set SVV_ISSUE_102_STL to run the issue #102 attachment regression")

    path = Path(configured).expanduser().resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)

    assert digest.hexdigest() == (
        "ec3d4e23757659604c939e7d2f418587bfedc2a067479e4964f0ab40ee637275"
    )
    return path
