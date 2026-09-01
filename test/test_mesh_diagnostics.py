import math
import json

import pyvista as pv
import pytest

from svv.domain.routines.mesh_diagnostics import (
    TetGenAttemptReport,
    TetrahedralizationError,
    TetrahedralizationReport,
    summarize_surface,
    summarize_tetgen_output,
)


def test_summarize_surface_reports_closed_cube_geometry():
    summary = summarize_surface(pv.Cube().triangulate())

    assert summary.n_points == 8
    assert summary.n_triangles == 12
    assert summary.n_components == 1
    assert summary.is_all_triangles is True
    assert summary.points_finite is True
    assert summary.is_manifold is True
    assert summary.n_open_edges == 0
    assert summary.bounds == (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)
    assert summary.diagonal == math.sqrt(3.0)
    assert summary.area == 6.0
    assert summary.volume == pytest.approx(1.0)


def test_summarize_tetgen_output_identifies_intersections_and_native_abort():
    stdout = """\
Warning:  A segment and a facet intersect.
  segment: [38341,38340] tag(-1).
  facet triangle: [1033,1035,1238] tag(-1)
Warning:  A segment and a facet intersect.
  segment: [38311,38310] tag(-1).
  facet triangle: [37032,74417,74416] tag(-1)
Warning:  Two facets exactly intersect.
  1st facet triangle: [1608,1810,75715] tag(-1).
  2nd facet triangle: [1605,1608,75721] tag(-1).
  151404 (12) subfaces are recovered (missing).
"""

    summary = summarize_tetgen_output(
        stdout,
        "free(): invalid next size (normal)\n",
        -6,
    )

    assert summary.segment_facet_intersections == 2
    assert summary.facet_facet_intersections == 1
    assert summary.missing_subfaces == 12
    assert summary.native_abort is True
    assert summary.return_code == -6
    assert summary.signal_name == "SIGABRT"
    assert any("segment: [38341,38340]" in line for line in summary.examples)


def test_failure_report_provides_actionable_summary_details_and_json():
    surface = summarize_surface(pv.Cube().triangulate())
    diagnostic = summarize_tetgen_output(
        "Warning: A segment and a facet intersect.\n"
        "  segment: [4,5] tag(-1).\n"
        "  facet triangle: [1,2,3] tag(-1).\n",
        "free(): invalid next size (normal)\n",
        -6,
    )
    attempt = TetGenAttemptReport(
        strategy="original",
        status="failed",
        surface=surface,
        duration_seconds=1.25,
        recoverable=True,
        tetgen_args=(),
        tetgen_kwargs={"order": 1, "nobisect": True},
        diagnostics=diagnostic,
        message="TetGen rejected the input surface.",
    )
    report = TetrahedralizationReport(
        source=surface,
        attempts=[attempt],
        selected_strategy=None,
        selected_surface=None,
        versions={"tetgen": "0.6.4"},
    )

    summary = report.user_summary()
    details = report.detailed_text()
    payload = report.to_dict()

    assert "intersecting surface facets" in summary
    assert "repair the source surface" in summary.lower()
    assert "original" in details
    assert "SIGABRT" in details
    assert "nobisect=True" in details
    assert "8 points" in details
    assert "segment: [4,5]" in details
    assert payload["source"]["n_points"] == 8
    assert payload["attempts"][0]["diagnostics"]["return_code"] == -6
    json.dumps(payload)
    assert str(TetrahedralizationError(report)) == summary


def test_tetgen_output_is_bounded_without_losing_python_failure_classification():
    summary = summarize_tetgen_output(
        "x" * 40000,
        "Traceback (most recent call last):\nRuntimeError: Unknown exception\n",
        1,
    )

    assert summary.python_exception is True
    assert summary.native_abort is False
    assert summary.signal_name is None
    assert len(summary.stdout) < 33000
    assert "[truncated" in summary.stdout
