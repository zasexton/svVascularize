import importlib
import os
import sys
from dataclasses import replace

import numpy as np
import pyvista as pv
import pytest

from svv.domain.routines.mesh_diagnostics import (
    TetGenAttemptReport,
    TetGenWorkerError,
    TetrahedralizationError,
    summarize_surface,
    summarize_tetgen_output,
)


tetrahedralize_mod = importlib.import_module("svv.domain.routines.tetrahedralize")


def _geometry_worker_error(surface, strategy, tet_kwargs):
    return TetGenWorkerError(
        TetGenAttemptReport(
            strategy=strategy,
            status="failed",
            surface=summarize_surface(surface),
            duration_seconds=0.01,
            recoverable=True,
            tetgen_args=(),
            tetgen_kwargs=dict(tet_kwargs),
            diagnostics=summarize_tetgen_output(
                "Warning: A segment and a facet intersect.\n", "", 1
            ),
            message="TetGen rejected the {} surface.".format(strategy),
        )
    )


def test_closed_manifold_self_intersection_recovers_with_meshfix(
    closed_self_intersecting_surface,
):
    result = tetrahedralize_mod.tetrahedralize(
        closed_self_intersecting_surface,
        order=1,
        nobisect=True,
        repair_max_distance_ratio=0.4,
        remesh_on_failure=False,
        return_result=True,
    )

    assert result.report.selected_strategy in {"original", "meshfix"}
    if result.report.attempts[0].status == "failed":
        assert result.report.selected_strategy == "meshfix"
    else:
        assert result.report.attempts[0].strategy == "original"
        assert result.report.attempts[0].status == "succeeded"
    assert result.grid.n_cells > 0
    assert np.isfinite(result.nodes).all()
    assert result.elements.shape[1] == 4
    assert result.surface.is_manifold
    assert result.surface.n_open_edges == 0
    assert set(np.unique(result.grid.celltypes)) == {pv.CellType.TETRA}


def test_clean_surface_keeps_original_fast_path_and_legacy_tuple(monkeypatch):
    def unexpected_recovery(*args, **kwargs):
        raise AssertionError("Recovery must not run after the original surface succeeds")

    monkeypatch.setattr(
        tetrahedralize_mod,
        "repair_surface_with_meshfix",
        unexpected_recovery,
    )
    monkeypatch.setattr(
        tetrahedralize_mod,
        "uniform_remesh_surface",
        unexpected_recovery,
    )
    surface = pv.Cube().triangulate()

    result = tetrahedralize_mod.tetrahedralize(surface, return_result=True)
    legacy = tetrahedralize_mod.tetrahedralize(surface)

    assert result.report.selected_strategy == "original"
    assert len(result.report.attempts) == 1
    assert result.report.attempts[0].status == "succeeded"
    assert isinstance(legacy, tuple)
    assert len(legacy) == 3
    assert legacy[0].n_cells > 0


def test_recovery_does_not_mutate_or_share_storage_with_input(
    closed_self_intersecting_surface,
):
    original_points = closed_self_intersecting_surface.points.copy()
    original_faces = closed_self_intersecting_surface.faces.copy()

    result = tetrahedralize_mod.tetrahedralize(
        closed_self_intersecting_surface,
        repair_max_distance_ratio=0.4,
        remesh_on_failure=False,
        return_result=True,
    )
    result.surface.points[0] += 1.0

    assert np.array_equal(closed_self_intersecting_surface.points, original_points)
    assert np.array_equal(closed_self_intersecting_surface.faces, original_faces)
    assert not np.shares_memory(result.surface.points, closed_self_intersecting_surface.points)


def test_default_repair_bound_rejects_large_surface_change(
    closed_self_intersecting_surface,
):
    with pytest.raises(ValueError, match="bounds|displacement"):
        tetrahedralize_mod.repair_surface_with_meshfix(
            closed_self_intersecting_surface,
            max_distance_ratio=0.01,
        )


def test_recovery_validation_rejects_internal_displacement_with_unchanged_bounds():
    source = pv.Sphere(theta_resolution=20, phi_resolution=20)
    candidate = source.copy(deep=True)
    interior_index = int(np.argmin(np.abs(candidate.points[:, 2])))
    candidate.points[interior_index] *= 0.5

    with pytest.raises(ValueError, match="displacement"):
        tetrahedralize_mod.validate_recovery_surface(
            source,
            candidate,
            max_distance_ratio=0.01,
        )


def test_recovery_validation_rejects_nonfinite_surface_distance(monkeypatch):
    source = pv.Cube().triangulate()
    candidate = source.copy(deep=True)
    monkeypatch.setattr(
        tetrahedralize_mod,
        "_symmetric_surface_distance",
        lambda *args: np.nan,
    )

    with pytest.raises(ValueError, match="distance.*finite"):
        tetrahedralize_mod.validate_recovery_surface(
            source,
            candidate,
            max_distance_ratio=0.01,
        )


@pytest.mark.parametrize("distances", [np.array([]), np.array([np.nan]), np.array([np.inf])])
def test_symmetric_surface_distance_rejects_invalid_arrays(distances):
    class FakeSurface:
        def compute_implicit_distance(self, other):
            return {"implicit_distance": distances}

    with pytest.raises(ValueError, match="distance arrays.*finite.*non-empty"):
        tetrahedralize_mod._symmetric_surface_distance(
            FakeSurface(),
            FakeSurface(),
        )


def test_recovery_validation_rejects_nonfinite_bounds(monkeypatch):
    source = pv.Cube().triangulate()
    candidate = source.copy(deep=True)
    real_summary = tetrahedralize_mod.summarize_surface

    def summary(surface):
        result = real_summary(surface)
        if surface is candidate:
            return replace(result, bounds=(np.nan,) + result.bounds[1:])
        return result

    monkeypatch.setattr(tetrahedralize_mod, "summarize_surface", summary)

    with pytest.raises(ValueError, match="bounds.*finite"):
        tetrahedralize_mod.validate_recovery_surface(
            source,
            candidate,
            max_distance_ratio=0.01,
        )


def test_recovery_validation_rejects_nonfinite_allowed_distance():
    source = pv.Cube().triangulate()

    with pytest.raises(ValueError, match="allowed recovery distance.*finite"):
        tetrahedralize_mod.validate_recovery_surface(
            source,
            source.copy(deep=True),
            max_distance_ratio=np.finfo(float).max,
        )


def test_invalid_repair_candidate_is_rejected_before_tetgen(monkeypatch):
    source = pv.Cube().triangulate()
    invalid_candidate = pv.Plane().triangulate()
    calls = []

    def worker(surface, tet_args, tet_kwargs, worker_script, python_exe, *, strategy):
        calls.append(strategy)
        if strategy != "original":
            raise AssertionError("Invalid recovery candidate reached TetGen")
        raise TetGenWorkerError(
            TetGenAttemptReport(
                strategy="original",
                status="failed",
                surface=summarize_surface(surface),
                duration_seconds=0.01,
                recoverable=True,
                tetgen_args=(),
                tetgen_kwargs=dict(tet_kwargs),
                diagnostics=summarize_tetgen_output(
                    "Warning: A segment and a facet intersect.\n", "", 1
                ),
                message="TetGen rejected the original surface.",
            )
        )

    monkeypatch.setattr(tetrahedralize_mod, "_tetgen_worker_tetrahedralize", worker)
    monkeypatch.setattr(
        tetrahedralize_mod,
        "repair_surface_with_meshfix",
        lambda *args, **kwargs: invalid_candidate.copy(deep=True),
    )

    with pytest.raises(TetrahedralizationError) as error_info:
        tetrahedralize_mod.tetrahedralize(
            source,
            remesh_on_failure=False,
            return_result=True,
        )

    assert calls == ["original"]
    assert [attempt.status for attempt in error_info.value.report.attempts] == [
        "failed",
        "rejected",
    ]
    rejected = error_info.value.report.attempts[1]
    assert rejected.surface.n_points == invalid_candidate.n_points
    assert rejected.surface.n_cells == invalid_candidate.n_cells


def test_meshfix_rejection_report_describes_the_repaired_candidate(monkeypatch):
    source = pv.Cube().triangulate()

    class DistortingMeshFix:
        def __init__(self, points, faces):
            self.v = np.asarray(points).copy()
            self.f = np.asarray(faces).copy()

        def repair(self, **kwargs):
            self.v[:, 0] += 10.0

    def worker(surface, tet_args, tet_kwargs, worker_script, python_exe, *, strategy):
        raise _geometry_worker_error(surface, strategy, tet_kwargs)

    monkeypatch.setattr(tetrahedralize_mod.pymeshfix, "MeshFix", DistortingMeshFix)
    monkeypatch.setattr(tetrahedralize_mod, "_tetgen_worker_tetrahedralize", worker)

    with pytest.raises(TetrahedralizationError) as error_info:
        tetrahedralize_mod.tetrahedralize(source, remesh_on_failure=False)

    rejected = error_info.value.report.attempts[1]
    assert rejected.status == "rejected"
    assert rejected.surface.bounds[0] > 9.0


def test_unexpected_meshfix_error_is_not_relabelled_as_geometry_rejection(monkeypatch):
    source = pv.Cube().triangulate()

    def worker(surface, tet_args, tet_kwargs, worker_script, python_exe, *, strategy):
        raise _geometry_worker_error(surface, strategy, tet_kwargs)

    def broken_repair(*args, **kwargs):
        raise ValueError("meshfix implementation failed")

    monkeypatch.setattr(tetrahedralize_mod, "_tetgen_worker_tetrahedralize", worker)
    monkeypatch.setattr(tetrahedralize_mod, "repair_surface_with_meshfix", broken_repair)

    with pytest.raises(ValueError, match="meshfix implementation failed"):
        tetrahedralize_mod.tetrahedralize(source, remesh_on_failure=False)


def test_unexpected_pyacvd_error_is_not_relabelled_as_geometry_rejection(monkeypatch):
    source = pv.Cube().triangulate()

    def worker(surface, tet_args, tet_kwargs, worker_script, python_exe, *, strategy):
        raise _geometry_worker_error(surface, strategy, tet_kwargs)

    def broken_remesh(*args, **kwargs):
        raise RuntimeError("remesher implementation failed")

    monkeypatch.setattr(tetrahedralize_mod, "_tetgen_worker_tetrahedralize", worker)
    monkeypatch.setattr(tetrahedralize_mod, "uniform_remesh_surface", broken_remesh)

    with pytest.raises(RuntimeError, match="remesher implementation failed"):
        tetrahedralize_mod.tetrahedralize(source, repair_on_failure=False)


def test_open_pyacvd_candidate_is_repaired_before_tetgen(monkeypatch):
    source = pv.Cube().triangulate()
    face_rows = source.faces.reshape(-1, 4)
    open_surface = pv.PolyData(source.points.copy(), face_rows[:-1].copy())
    worker_calls = []
    repair_calls = []

    def worker(surface, tet_args, tet_kwargs, worker_script, python_exe, *, strategy):
        worker_calls.append(strategy)
        if strategy == "original":
            raise TetGenWorkerError(
                TetGenAttemptReport(
                    strategy=strategy,
                    status="failed",
                    surface=summarize_surface(surface),
                    duration_seconds=0.01,
                    recoverable=True,
                    tetgen_args=(),
                    tetgen_kwargs=dict(tet_kwargs),
                    diagnostics=summarize_tetgen_output(
                        "Warning: A segment and a facet intersect.\n", "", 1
                    ),
                    message="TetGen rejected the original surface.",
                )
            )
        if strategy != "pyacvd_meshfix":
            raise AssertionError("Open PyACVD surface reached TetGen")
        nodes = np.array(
            [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.0, 0.25, 0.0], [0.0, 0.0, 0.25]]
        )
        return nodes, np.array([[0, 1, 2, 3]], dtype=np.int64)

    def repair(surface, **kwargs):
        repair_calls.append(surface.n_open_edges)
        if len(repair_calls) == 1:
            return open_surface.copy(deep=True)
        return source.copy(deep=True)

    monkeypatch.setattr(tetrahedralize_mod, "_tetgen_worker_tetrahedralize", worker)
    monkeypatch.setattr(tetrahedralize_mod, "repair_surface_with_meshfix", repair)
    monkeypatch.setattr(
        tetrahedralize_mod,
        "uniform_remesh_surface",
        lambda *args, **kwargs: open_surface.copy(deep=True),
    )

    result = tetrahedralize_mod.tetrahedralize(source, return_result=True)

    assert result.report.selected_strategy == "pyacvd_meshfix"
    assert worker_calls == ["original", "pyacvd_meshfix"]
    assert repair_calls == [0, open_surface.n_open_edges]


@pytest.mark.parametrize("switches", [None, "pqQ"])
def test_quiet_unknown_failure_runs_isolated_geometry_diagnostic(
    tmp_path,
    monkeypatch,
    switches,
):
    worker = tmp_path / "diagnostic_worker.py"
    worker.write_text(
        """\
import json
import pathlib
import sys

with open(sys.argv[3], 'r') as stream:
    options = json.load(stream)['kwargs']
switches = options.get('switches')
if switches:
    diagnostic_enabled = 'd' in switches and 'V' in switches and 'Q' not in switches
else:
    diagnostic_enabled = options.get('diagnose') and options.get('quiet') is False
if diagnostic_enabled:
    pathlib.Path('_skipped.node').write_text('diagnostic')
    print('Warning: A segment and a facet intersect.')
    print('  segment: [4,5] tag(-1).')
    print('  facet triangle: [1,2,3] tag(-1).')
    raise SystemExit(3)
print('RuntimeError: Failed to tetrahedralize: Unknown exception', file=sys.stderr)
raise SystemExit(1)
"""
    )
    monkeypatch.chdir(tmp_path)

    tetgen_options = {"switches": switches} if switches is not None else {}
    with pytest.raises(TetrahedralizationError) as error_info:
        tetrahedralize_mod.tetrahedralize(
            pv.Cube().triangulate(),
            worker_script=str(worker),
            python_exe=sys.executable,
            repair_on_failure=False,
            remesh_on_failure=False,
            return_result=True,
            **tetgen_options,
        )

    report = error_info.value.report
    assert len(report.attempts) == 1
    assert report.attempts[0].diagnostics.segment_facet_intersections == 1
    assert "intersecting surface facets" in str(error_info.value)
    assert not list(tmp_path.glob("_skipped.*"))


def test_relative_worker_path_is_resolved_before_temporary_chdir(tmp_path, monkeypatch):
    worker = tmp_path / "relative_worker.py"
    worker.write_text(
        """\
import numpy as np
import sys

nodes = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
)
elems = np.array([[0, 1, 2, 3]], dtype=np.int64)
np.savez(sys.argv[2], nodes=nodes, elems=elems)
"""
    )
    monkeypatch.chdir(tmp_path)

    result = tetrahedralize_mod.tetrahedralize(
        pv.Cube().triangulate(),
        worker_script="relative_worker.py",
        python_exe=sys.executable,
        repair_on_failure=False,
        remesh_on_failure=False,
        return_result=True,
    )

    assert result.grid.n_cells == 1


def test_worker_launch_path_resolution_preserves_path_lookup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    worker, interpreter = tetrahedralize_mod._resolve_worker_launch_paths(
        "worker.py",
        os.path.join("env", "python"),
    )
    _, path_interpreter = tetrahedralize_mod._resolve_worker_launch_paths(
        "worker.py",
        "python",
    )

    assert worker == str(tmp_path / "worker.py")
    assert interpreter == str(tmp_path / "env" / "python")
    assert path_interpreter == "python"


def test_opaque_native_abort_runs_isolated_geometry_diagnostic(tmp_path):
    worker = tmp_path / "native_diagnostic_worker.py"
    worker.write_text(
        """\
import json
import sys

with open(sys.argv[3], 'r') as stream:
    options = json.load(stream)['kwargs']
if options.get('diagnose'):
    print('Warning: Two facets exactly intersect.')
    print('  1st facet triangle: [1,2,3] tag(-1).')
    print('  2nd facet triangle: [4,5,6] tag(-1).')
    raise SystemExit(3)
print('free(): invalid next size (normal)', file=sys.stderr)
raise SystemExit(1)
"""
    )

    with pytest.raises(TetrahedralizationError) as error_info:
        tetrahedralize_mod.tetrahedralize(
            pv.Cube().triangulate(),
            worker_script=str(worker),
            python_exe=sys.executable,
            repair_on_failure=False,
            remesh_on_failure=False,
        )

    diagnostic = error_info.value.report.attempts[0].diagnostics
    assert diagnostic.facet_facet_intersections == 1
    assert "intersecting surface facets" in str(error_info.value)


def test_disabling_repair_and_remesh_returns_structured_direct_failure(monkeypatch):
    def worker(surface, tet_args, tet_kwargs, worker_script, python_exe, *, strategy):
        raise _geometry_worker_error(surface, strategy, tet_kwargs)

    monkeypatch.setattr(tetrahedralize_mod, "_tetgen_worker_tetrahedralize", worker)

    with pytest.raises(TetrahedralizationError) as error_info:
        tetrahedralize_mod.tetrahedralize(
            pv.Cube().triangulate(),
            repair_on_failure=False,
            remesh_on_failure=False,
        )

    assert [attempt.strategy for attempt in error_info.value.report.attempts] == [
        "original"
    ]


def test_nongeometry_worker_failure_does_not_start_recovery(monkeypatch):
    source = pv.Cube().triangulate()
    infrastructure_attempt = TetGenAttemptReport(
        strategy="original",
        status="failed",
        surface=summarize_surface(source),
        duration_seconds=0.01,
        recoverable=False,
        tetgen_args=(),
        tetgen_kwargs={"verbose": 0},
        diagnostics=summarize_tetgen_output(
            "", "ModuleNotFoundError: missing dependency\n", 1
        ),
        message="TetGen worker infrastructure failed for the original surface.",
    )

    def worker(*args, **kwargs):
        raise TetGenWorkerError(infrastructure_attempt)

    def unexpected_recovery(*args, **kwargs):
        raise AssertionError("Infrastructure failures must not start geometry recovery")

    monkeypatch.setattr(tetrahedralize_mod, "_tetgen_worker_tetrahedralize", worker)
    monkeypatch.setattr(
        tetrahedralize_mod, "repair_surface_with_meshfix", unexpected_recovery
    )
    monkeypatch.setattr(tetrahedralize_mod, "uniform_remesh_surface", unexpected_recovery)

    with pytest.raises(TetGenWorkerError) as error_info:
        tetrahedralize_mod.tetrahedralize(source)

    assert error_info.value.recoverable is False


def test_all_geometry_attempts_fail_with_ordered_aggregate_report(monkeypatch):
    source = pv.Cube().triangulate()

    def worker(surface, tet_args, tet_kwargs, worker_script, python_exe, *, strategy):
        raise _geometry_worker_error(surface, strategy, tet_kwargs)

    monkeypatch.setattr(tetrahedralize_mod, "_tetgen_worker_tetrahedralize", worker)
    monkeypatch.setattr(
        tetrahedralize_mod,
        "repair_surface_with_meshfix",
        lambda *args, **kwargs: source.copy(deep=True),
    )
    monkeypatch.setattr(
        tetrahedralize_mod,
        "uniform_remesh_surface",
        lambda *args, **kwargs: source.copy(deep=True),
    )

    with pytest.raises(TetrahedralizationError) as error_info:
        tetrahedralize_mod.tetrahedralize(source)

    report = error_info.value.report
    assert [attempt.strategy for attempt in report.attempts] == [
        "original",
        "meshfix",
        "pyacvd",
    ]
    assert report.selected_strategy is None
    assert "intersecting surface facets" in report.user_summary()


def test_meshfix_repair_preserves_components_by_policy(monkeypatch):
    source = pv.Cube().triangulate()
    captured = {}

    class FakeMeshFix:
        def __init__(self, points, faces):
            self.v = np.asarray(points).copy()
            self.f = np.asarray(faces).copy()

        def repair(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(tetrahedralize_mod.pymeshfix, "MeshFix", FakeMeshFix)

    repaired = tetrahedralize_mod.repair_surface_with_meshfix(source)

    assert repaired.n_points == source.n_points
    assert captured["joincomp"] is False
    assert captured["remove_smallest_components"] is False


def test_recovery_validation_rejects_component_loss():
    first = pv.Cube(center=(0.0, 0.0, 0.0)).triangulate()
    second = pv.Cube(center=(3.0, 0.0, 0.0)).triangulate()
    source = first.merge(second, merge_points=False)

    with pytest.raises(ValueError, match="connected-component count from 2 to 1"):
        tetrahedralize_mod.validate_recovery_surface(
            source,
            first,
            max_distance_ratio=1.0,
        )


@pytest.mark.parametrize("value", [0.0, -0.1, np.nan, np.inf])
def test_repair_distance_ratio_must_be_finite_and_positive(value):
    with pytest.raises(ValueError, match="finite and positive"):
        tetrahedralize_mod.tetrahedralize(
            pv.Cube().triangulate(),
            repair_max_distance_ratio=value,
        )


@pytest.mark.parametrize(
    "value",
    [True, np.bool_(True), np.array(0.1), np.array([0.1]), "0.1", None],
)
def test_repair_distance_ratio_must_be_a_real_scalar(value):
    with pytest.raises(ValueError, match="finite and positive scalar"):
        tetrahedralize_mod.tetrahedralize(
            pv.Cube().triangulate(),
            repair_max_distance_ratio=value,
        )


@pytest.mark.parametrize("value", [-1.0, np.nan, np.inf])
def test_remesh_clean_tolerance_must_be_finite_and_nonnegative(value):
    with pytest.raises(ValueError, match="finite and non-negative"):
        tetrahedralize_mod.tetrahedralize(
            pv.Cube().triangulate(),
            remesh_clean_tolerance=value,
        )


@pytest.mark.parametrize(
    "value",
    [True, np.bool_(True), np.array(0.1), np.array([0.1]), "0.1"],
)
def test_remesh_clean_tolerance_must_be_a_real_scalar(value):
    with pytest.raises(ValueError, match="finite and non-negative scalar"):
        tetrahedralize_mod.tetrahedralize(
            pv.Cube().triangulate(),
            remesh_clean_tolerance=value,
        )


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("remesh_subdivisions", np.nan, "non-negative integer"),
        ("remesh_subdivisions", 1.5, "non-negative integer"),
        ("remesh_subdivisions", True, "non-negative integer"),
        ("remesh_clusters", np.nan, "positive integer"),
        ("remesh_clusters", 2.5, "positive integer"),
        ("remesh_clusters", True, "positive integer"),
    ],
)
def test_remesh_integer_controls_are_validated_before_tetgen(
    monkeypatch,
    option,
    value,
    message,
):
    def unexpected_worker(*args, **kwargs):
        raise AssertionError("Invalid recovery controls must fail before TetGen")

    monkeypatch.setattr(
        tetrahedralize_mod,
        "_tetgen_worker_tetrahedralize",
        unexpected_worker,
    )

    with pytest.raises(ValueError, match=message):
        tetrahedralize_mod.tetrahedralize(
            pv.Cube().triangulate(),
            **{option: value},
        )
