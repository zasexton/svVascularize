import tetgen
import pymeshfix
from dataclasses import dataclass, replace
import subprocess
import tempfile
import os
import sys
from tqdm.auto import tqdm
from itertools import cycle
from time import sleep
import time
import numpy as np
import pyvista as pv
from svv.utils.remeshing import remesh
from svv.domain.routines.mesh_diagnostics import (
    TetGenAttemptReport,
    TetGenWorkerError,
    TetrahedralizationError,
    TetrahedralizationReport,
    summarize_surface,
    summarize_tetgen_output,
)
import shutil
import json
from concurrent.futures import ThreadPoolExecutor

filepath = os.path.abspath(__file__)
dirpath = os.path.dirname(filepath)

def format_elapsed(seconds: float) -> str:
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    else:
        return f"{m:02d}:{s:02d}"


def _spinner_cycle():
    """
    Return a spinner cycle that is safe for the current stdout encoding.

    On Windows runners the default code page may not support the braille
    characters used by common Unicode spinners, which can raise a
    UnicodeEncodeError when writing to sys.stdout. To avoid this, we
    fall back to a simple ASCII spinner if the encoding cannot handle
    the Unicode characters.
    """
    ascii_spinner = ["-", "\\", "|", "/"]

    encoding = getattr(sys.stdout, "encoding", None)
    if not encoding:
        return cycle(ascii_spinner)

    fancy_spinner = ["⠋", "⠙", "⠹", "⠸", "⠼",
                     "⠴", "⠦", "⠧", "⠇", "⠏"]
    try:
        "".join(fancy_spinner).encode(encoding)
    except Exception:
        return cycle(ascii_spinner)

    return cycle(fancy_spinner)

def triangulate(curve, verbose=False, **kwargs):
    """
    Triangulate a curve using VTK.

    Parameters
    ----------
    curve : Pyvista.PolyData PolyLine object
        The boundary curve within which the triangulation will
        be performed.
    verbose : bool
        A flag to indicate if mesh fixing should be verbose.
    kwargs : dict
        A dictionary of keyword arguments to be passed to VTK.

    Returns
    -------
    mesh : PyMesh mesh object
        A triangular mesh representing the triangulated region bounded by
        the curve.
    nodes : ndarray
        An array of node coordinates for the triangular mesh.
    vertices : ndarray
        An array of vertex indices for the triangular mesh.
    """
    mesh = curve.delaunay_2d(**kwargs)
    mesh = remesh.remesh_surface(mesh)
    nodes = mesh.points
    vertices = mesh.faces.reshape(-1, 4)[:, 1:]
    return mesh, nodes, vertices

def _run_tetgen(surface_mesh):
    tgen = tetgen.TetGen(surface_mesh)
    nodes, elems = tgen.tetrahedralize(verbose=0)
    return nodes, elems


def prepare_surface(surface: pv.DataSet) -> pv.PolyData:
    """Return a finite, non-empty triangular deep copy of ``surface``."""

    if isinstance(surface, pv.PolyData):
        prepared = surface.copy(deep=True)
    else:
        prepared = surface.extract_surface().copy(deep=True)
    if not prepared.is_all_triangles:
        prepared = prepared.triangulate()
    prepared = prepared.clean(tolerance=0.0, absolute=True)
    if prepared.n_points == 0 or prepared.n_cells == 0:
        raise ValueError("Cannot tetrahedralize an empty surface")
    if not np.isfinite(np.asarray(prepared.points)).all():
        raise ValueError("Surface points must contain only finite coordinates")
    if not prepared.is_all_triangles:
        raise ValueError("Surface preparation did not produce only triangles")
    return prepared


def _symmetric_surface_distance(first: pv.PolyData, second: pv.PolyData) -> float:
    first_distances = np.abs(
        np.asarray(first.compute_implicit_distance(second)["implicit_distance"])
    )
    second_distances = np.abs(
        np.asarray(second.compute_implicit_distance(first)["implicit_distance"])
    )
    return float(max(first_distances.max(), second_distances.max()))


def validate_recovery_surface(
    source: pv.PolyData,
    candidate: pv.PolyData,
    *,
    max_distance_ratio: float,
):
    """Validate topology and geometric displacement of a recovery candidate."""

    if not np.isfinite(max_distance_ratio) or max_distance_ratio <= 0:
        raise ValueError("max_distance_ratio must be finite and positive")
    source_summary = summarize_surface(source)
    candidate_summary = summarize_surface(candidate)
    if not candidate_summary.points_finite:
        raise ValueError("Recovery surface points must be finite")
    if not candidate_summary.is_all_triangles:
        raise ValueError("Recovery surface must contain only triangles")
    if not candidate_summary.is_manifold:
        raise ValueError("Recovery surface is non-manifold")
    if candidate_summary.n_open_edges != 0:
        raise ValueError(
            "Recovery surface has {} open edges".format(candidate_summary.n_open_edges)
        )
    if candidate_summary.n_components != source_summary.n_components:
        raise ValueError(
            "Recovery changed connected-component count from {} to {}".format(
                source_summary.n_components,
                candidate_summary.n_components,
            )
        )

    allowed_distance = source_summary.diagonal * max_distance_ratio
    bounds_delta = float(
        np.max(
            np.abs(
                np.asarray(candidate_summary.bounds, dtype=float)
                - np.asarray(source_summary.bounds, dtype=float)
            )
        )
    )
    if bounds_delta > allowed_distance:
        raise ValueError(
            "Recovery surface bounds changed by {:.6g}, exceeding {:.6g}".format(
                bounds_delta,
                allowed_distance,
            )
        )
    distance = _symmetric_surface_distance(source, candidate)
    if distance > allowed_distance:
        raise ValueError(
            "Recovery surface displacement {:.6g} exceeds {:.6g} "
            "({:.3%} of the source diagonal)".format(
                distance,
                allowed_distance,
                max_distance_ratio,
            )
        )
    return candidate_summary


def repair_surface_with_meshfix(
    surface: pv.PolyData,
    *,
    max_distance_ratio: float = 0.01,
) -> pv.PolyData:
    """Repair a copy of ``surface`` without joining or dropping components."""

    prepared = prepare_surface(surface)
    faces = np.asarray(prepared.faces).reshape(-1, 4)[:, 1:]
    meshfix = pymeshfix.MeshFix(np.asarray(prepared.points), faces)
    meshfix.repair(
        verbose=False,
        joincomp=False,
        remove_smallest_components=False,
    )
    repaired_faces = np.column_stack(
        (
            np.full(len(meshfix.f), 3, dtype=np.int64),
            np.asarray(meshfix.f, dtype=np.int64),
        )
    )
    repaired = pv.PolyData(np.asarray(meshfix.v).copy(), repaired_faces)
    repaired = prepare_surface(repaired)
    validate_recovery_surface(
        prepared,
        repaired,
        max_distance_ratio=max_distance_ratio,
    )
    return repaired

def uniform_remesh_surface(surface: pv.PolyData,
                           *,
                           subdivisions: int = 3,
                           clusters: int = 20000,
                           clean_tolerance: float = 1e-5) -> pv.PolyData:
    """
    Generate a uniform, isotropic triangle surface for TetGen retry attempts.

    PyACVD is imported lazily so callers that do not need the retry path do not
    pay the import cost until TetGen actually fails.
    """
    try:
        import pyacvd
    except ImportError as exc:
        raise RuntimeError(
            "PyACVD is required for TetGen uniform remeshing fallback. "
            "Install pyacvd or call tetrahedralize(..., remesh_on_failure=False)."
        ) from exc

    if subdivisions < 0:
        raise ValueError("subdivisions must be non-negative")
    if clusters <= 0:
        raise ValueError("clusters must be positive")

    if not isinstance(surface, pv.PolyData):
        surface = surface.extract_surface()
    base_mesh = pv.PolyData(surface.points, surface.faces)
    if clean_tolerance is not None:
        base_mesh = base_mesh.clean(tolerance=clean_tolerance)
    if not base_mesh.is_all_triangles:
        base_mesh = base_mesh.triangulate()

    if base_mesh.n_cells == 0:
        raise ValueError("Cannot remesh an empty surface")

    clustering = pyacvd.Clustering(base_mesh)
    if subdivisions:
        clustering.subdivide(int(subdivisions))
    clustering.cluster(int(clusters))
    remeshed = clustering.create_mesh()
    if clean_tolerance is not None:
        remeshed = remeshed.clean(tolerance=clean_tolerance)
    if not remeshed.is_all_triangles:
        remeshed = remeshed.triangulate()
    return remeshed


def _tetgen_worker_tetrahedralize(surface: pv.PolyData,
                                  tet_args,
                                  tet_kwargs,
                                  worker_script: str,
                                  python_exe: str,
                                  *,
                                  strategy: str = "original"):
    attempt_start = time.perf_counter()
    surface_summary = summarize_surface(surface)

    def infrastructure_error(message, exc):
        details = "{}: {}".format(type(exc).__name__, exc)
        return TetGenWorkerError(
            TetGenAttemptReport(
                strategy=strategy,
                status="infrastructure-error",
                surface=surface_summary,
                duration_seconds=time.perf_counter() - attempt_start,
                recoverable=False,
                tetgen_args=tuple(tet_args),
                tetgen_kwargs=dict(tet_kwargs),
                diagnostics=summarize_tetgen_output("", details, 1),
                message="{}: {}".format(message, details),
            )
        )

    # On Windows, `tempfile` honors TMPDIR, which may be set to a POSIX-style
    # path such as '/tmp' and is not a valid directory there. Prefer the
    # standard TEMP/TMP locations when available to avoid spurious
    # "[WinError 267] The directory name is invalid" errors.
    tmp_root = None
    if os.name == "nt":
        for env_var in ("TEMP", "TMP"):
            candidate = os.environ.get(env_var)
            if candidate and os.path.isdir(candidate):
                tmp_root = candidate
                break

    with tempfile.TemporaryDirectory(dir=tmp_root) as tmpdir:
        surface_path = os.path.join(tmpdir, "surface.vtp")
        out_path = os.path.join(tmpdir, "tet.npz")
        config_path = os.path.join(tmpdir, "config.json")

        cfg = {
            "args": list(tet_args),
            "kwargs": tet_kwargs,
        }
        try:
            with open(config_path, "w") as f:
                json.dump(cfg, f)

            # Save the surface mesh so the worker can read it.
            surface.save(surface_path)
        except Exception as exc:
            raise infrastructure_error(
                "TetGen worker input preparation failed",
                exc,
            ) from exc

        # Command: call the worker script as a separate Python process
        cmd = [python_exe, worker_script, surface_path, out_path, config_path]

        # Start the worker process
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,   # decode to strings
                cwd=tmpdir,
            )
        except Exception as exc:
            raise infrastructure_error(
                "TetGen worker process could not be launched",
                exc,
            ) from exc

        show_spinner = sys.stdout.isatty()
        if show_spinner:
            spinner = _spinner_cycle()
            start_time = time.time()

            # Print label once
            sys.stdout.write("TetGen meshing| ")
            sys.stdout.flush()

            # Drain both pipes in a background thread while updating the spinner.
            # Waiting for process exit before reading can deadlock when TetGen emits
            # more output than an OS pipe can buffer.
            with ThreadPoolExecutor(max_workers=1) as executor:
                communication = executor.submit(proc.communicate)
                while not communication.done():
                    elapsed = time.time() - start_time
                    elapsed_str = format_elapsed(elapsed)
                    spin_char = next(spinner)
                    left = f"TetGen meshing| {spin_char}"

                    try:
                        width = shutil.get_terminal_size(fallback=(80, 20)).columns
                    except Exception:
                        width = 80

                    min_gap = 1
                    total_len = len(left) + min_gap + len(elapsed_str)
                    if total_len <= width:
                        spaces = width - len(left) - len(elapsed_str)
                    else:
                        spaces = min_gap

                    line = f"{left}{' ' * spaces}{elapsed_str}"
                    sys.stdout.write("\r" + line)
                    sys.stdout.flush()
                    time.sleep(0.1)
                stdout, stderr = communication.result()

            # Finish line
            sys.stdout.write("\n")
            sys.stdout.flush()
        else:
            # communicate() drains both pipes while waiting for completion.
            stdout, stderr = proc.communicate()

        if proc.returncode != 0:
            diagnostics = summarize_tetgen_output(stdout, stderr, proc.returncode)
            lower_output = (stdout + "\n" + stderr).lower()
            recoverable = bool(
                diagnostics.segment_facet_intersections
                or diagnostics.facet_facet_intersections
                or diagnostics.missing_segments
                or diagnostics.missing_subfaces
                or diagnostics.native_abort
                or "failed to tetrahedralize" in lower_output
                or "unknown exception" in lower_output
            )
            if recoverable:
                message = "TetGen rejected the {} surface.".format(strategy)
            else:
                message = "TetGen worker infrastructure failed for the {} surface.".format(
                    strategy
                )
            raise TetGenWorkerError(
                TetGenAttemptReport(
                    strategy=strategy,
                    status="failed",
                    surface=surface_summary,
                    duration_seconds=time.perf_counter() - attempt_start,
                    recoverable=recoverable,
                    tetgen_args=tuple(tet_args),
                    tetgen_kwargs=dict(tet_kwargs),
                    diagnostics=diagnostics,
                    message=message,
                )
            )

        # Load and validate results before cleaning the temporary directory.
        try:
            with np.load(out_path) as data:
                nodes = np.asarray(data["nodes"])
                elems = np.asarray(data["elems"])
            elems = _validate_tetgen_arrays(nodes, elems)
        except Exception as exc:
            diagnostics = summarize_tetgen_output(stdout, stderr, proc.returncode)
            raise TetGenWorkerError(
                TetGenAttemptReport(
                    strategy=strategy,
                    status="invalid-output",
                    surface=surface_summary,
                    duration_seconds=time.perf_counter() - attempt_start,
                    recoverable=False,
                    tetgen_args=tuple(tet_args),
                    tetgen_kwargs=dict(tet_kwargs),
                    diagnostics=diagnostics,
                    message="TetGen worker returned invalid output: {}".format(exc),
                )
            ) from exc

    return nodes, elems


def _validate_tetgen_arrays(nodes, elems):
    """Validate node coordinates and tetrahedral connectivity from a worker."""

    if nodes.ndim != 2 or nodes.shape[1] != 3 or nodes.shape[0] == 0:
        raise ValueError("TetGen nodes must have non-empty shape (N, 3)")
    if not np.isfinite(nodes).all():
        raise ValueError("TetGen nodes must contain only finite coordinates")
    if elems.ndim != 2 or elems.shape[0] == 0 or elems.shape[1] not in (4, 10):
        raise ValueError("TetGen elements must have non-empty shape (M, 4) or (M, 10)")
    if not np.issubdtype(elems.dtype, np.integer):
        raise ValueError("TetGen element connectivity must use an integer dtype")

    minimum = int(elems.min())
    maximum = int(elems.max())
    if minimum < 0 or maximum > nodes.shape[0]:
        raise ValueError("TetGen element connectivity contains out-of-range node indices")
    if maximum == nodes.shape[0]:
        if minimum < 1:
            raise ValueError(
                "TetGen element connectivity contains an out-of-range or mixed-base "
                "node index"
            )
        elems = elems - 1
    elif maximum >= nodes.shape[0]:
        raise ValueError("TetGen element connectivity contains out-of-range node indices")
    return elems


def _tetgen_grid_from_arrays(nodes, elems):
    """
    Convert TetGen node/connectivity arrays into a PyVista unstructured grid.
    """
    nodes = np.asarray(nodes)
    elems = np.asarray(elems)
    n_cells, n_vertices_per_cell = elems.shape
    cells = np.hstack(
        [
            np.full((n_cells, 1), n_vertices_per_cell, dtype=np.int64),
            elems.astype(np.int64),
        ]
    ).ravel()
    if n_vertices_per_cell == 4:
        celltypes = np.full(n_cells, pv.CellType.TETRA, dtype=np.uint8)
    elif n_vertices_per_cell == 10:
        celltypes = np.full(n_cells, pv.CellType.QUADRATIC_TETRA, dtype=np.uint8)
    else:
        raise ValueError(f"Unexpected number of vertices per cell: {n_vertices_per_cell}")

    grid = pv.UnstructuredGrid(cells, celltypes, nodes)

    return grid, nodes, elems


@dataclass
class TetrahedralizationResult:
    """Rich tetrahedralization result for callers that need provenance."""

    grid: pv.UnstructuredGrid
    nodes: np.ndarray
    elements: np.ndarray
    surface: pv.PolyData
    report: TetrahedralizationReport


def _dependency_versions():
    return {
        "tetgen": str(getattr(tetgen, "__version__", "unknown")),
        "pyvista": str(getattr(pv, "__version__", "unknown")),
        "pymeshfix": str(getattr(pymeshfix, "__version__", "unknown")),
    }


def _success_result(
    candidate,
    strategy,
    nodes,
    elems,
    report,
    duration,
    tet_args,
    tet_kwargs,
):
    grid, nodes, elems = _tetgen_grid_from_arrays(nodes, elems)
    candidate_summary = summarize_surface(candidate)
    report.attempts.append(
        TetGenAttemptReport(
            strategy=strategy,
            status="succeeded",
            surface=candidate_summary,
            duration_seconds=duration,
            recoverable=True,
            tetgen_args=tuple(tet_args),
            tetgen_kwargs=dict(tet_kwargs),
            diagnostics=summarize_tetgen_output("", "", 0),
            message="TetGen accepted the {} surface.".format(strategy),
        )
    )
    report.selected_strategy = strategy
    report.selected_surface = candidate_summary
    return TetrahedralizationResult(
        grid=grid,
        nodes=nodes,
        elements=elems,
        surface=candidate.copy(deep=True),
        report=report,
    )


def _rejected_attempt(strategy, surface, report, tet_args, tet_kwargs, message, duration):
    report.attempts.append(
        TetGenAttemptReport(
            strategy=strategy,
            status="rejected",
            surface=summarize_surface(surface),
            duration_seconds=duration,
            recoverable=True,
            tetgen_args=tuple(tet_args),
            tetgen_kwargs=dict(tet_kwargs),
            diagnostics=None,
            message=message,
        )
    )


def tetrahedralize(surface: pv.PolyData,
                   *tet_args,
                   worker_script: str = dirpath+os.sep+"tetgen_worker.py",
                   python_exe: str = sys.executable,
                   repair_on_failure: bool = True,
                   repair_max_distance_ratio: float = 0.01,
                   remesh_on_failure: bool = True,
                   remesh_subdivisions: int = 3,
                   remesh_clusters: int = 20000,
                   remesh_clean_tolerance: float = 1e-5,
                   return_result: bool = False,
                   **tet_kwargs):
    """
    Tetrahedralize a surface mesh using isolated TetGen worker processes.

    The unchanged, prepared surface is tried first. Geometry rejections then
    use a component-preserving MeshFix repair, followed by a validated PyACVD
    candidate as the final optional fallback. Recovery candidates must be
    closed, manifold, triangular, component-preserving, and within the
    configured displacement and bounds envelope. The caller's TetGen options
    are unchanged across attempts, and the input surface is never mutated.

    Parameters
    ----------
    surface : pyvista.DataSet
        Surface mesh to tetrahedralize. A deep triangular copy is prepared.
    *tet_args
        Positional arguments forwarded unchanged to TetGen.
    worker_script : str
        Worker entry point used to isolate native TetGen calls.
    python_exe : str
        Python interpreter used for the worker process.
    repair_on_failure : bool
        If True, retry a geometry-related TetGen failure after a
        component-preserving PyMeshFix repair.
    repair_max_distance_ratio : float
        Maximum symmetric repair displacement as a fraction of the source
        bounding-box diagonal.
    remesh_on_failure : bool
        If True, retain a validated PyACVD remesh as the final recovery path.
    remesh_subdivisions : int
        Number of PyACVD subdivision passes used by the retry path.
    remesh_clusters : int
        Number of PyACVD clusters used by the retry path.
    remesh_clean_tolerance : float
        PyVista clean tolerance applied before and after PyACVD remeshing.
    return_result : bool
        Return a ``TetrahedralizationResult`` containing the selected surface
        and structured report instead of the historical three-value tuple.
    **tet_kwargs
        Keyword arguments forwarded unchanged to every TetGen meshing attempt.

    Returns
    -------
    tuple or TetrahedralizationResult
        The historical ``(grid, nodes, elements)`` tuple, or a rich result
        when ``return_result=True``.

    Raises
    ------
    TetGenWorkerError
        If worker launch, dependencies, serialization, or result validation
        fails. Infrastructure failures do not start geometry recovery.
    TetrahedralizationError
        If every enabled safe geometry strategy fails. The exception carries
        the ordered ``report`` used by the GUI and troubleshooting tools.
    """
    if not isinstance(repair_on_failure, bool):
        raise ValueError("repair_on_failure must be a boolean")
    if not isinstance(remesh_on_failure, bool):
        raise ValueError("remesh_on_failure must be a boolean")
    if not isinstance(return_result, bool):
        raise ValueError("return_result must be a boolean")
    if not np.isfinite(repair_max_distance_ratio) or repair_max_distance_ratio <= 0:
        raise ValueError("repair_max_distance_ratio must be finite and positive")
    if remesh_subdivisions < 0:
        raise ValueError("remesh_subdivisions must be non-negative")
    if remesh_clusters <= 0:
        raise ValueError("remesh_clusters must be positive")
    if remesh_clean_tolerance is not None:
        try:
            valid_clean_tolerance = bool(
                np.isfinite(remesh_clean_tolerance) and remesh_clean_tolerance >= 0
            )
        except TypeError:
            valid_clean_tolerance = False
        if not valid_clean_tolerance:
            raise ValueError("remesh_clean_tolerance must be finite and non-negative")

    tet_kwargs.setdefault("verbose", 0)
    source = prepare_surface(surface)
    report = TetrahedralizationReport(
        source=summarize_surface(source),
        attempts=[],
        selected_strategy=None,
        selected_surface=None,
        versions=_dependency_versions(),
    )

    def diagnose_opaque_failure(candidate, error):
        diagnostic = error.attempt.diagnostics
        if diagnostic is None:
            return error.attempt
        has_geometry_details = bool(
            diagnostic.segment_facet_intersections
            or diagnostic.facet_facet_intersections
            or diagnostic.missing_segments
            or diagnostic.missing_subfaces
        )
        if has_geometry_details:
            return error.attempt

        diagnostic_kwargs = dict(tet_kwargs)
        diagnostic_kwargs["diagnose"] = 1
        diagnostic_kwargs["verbose"] = 1
        try:
            _tetgen_worker_tetrahedralize(
                candidate,
                tet_args,
                diagnostic_kwargs,
                worker_script,
                python_exe,
                strategy="{}-diagnostic".format(error.attempt.strategy),
            )
        except TetGenWorkerError as diagnostic_error:
            extra = diagnostic_error.attempt.diagnostics
            if extra is None:
                return error.attempt
            merged = summarize_tetgen_output(
                diagnostic.stdout + "\n" + extra.stdout,
                diagnostic.stderr + "\n" + extra.stderr,
                extra.return_code,
            )
            return replace(
                error.attempt,
                duration_seconds=(
                    error.attempt.duration_seconds
                    + diagnostic_error.attempt.duration_seconds
                ),
                diagnostics=merged,
                message=(
                    error.attempt.message
                    + " A diagnostic TetGen pass captured additional geometry details."
                ),
            )
        return error.attempt

    def attempt(candidate, strategy):
        started = time.perf_counter()
        try:
            nodes, elems = _tetgen_worker_tetrahedralize(
                candidate,
                tet_args,
                tet_kwargs,
                worker_script,
                python_exe,
                strategy=strategy,
            )
        except TetGenWorkerError as exc:
            attempt_report = (
                diagnose_opaque_failure(candidate, exc)
                if exc.recoverable
                else exc.attempt
            )
            report.attempts.append(attempt_report)
            if not exc.recoverable:
                raise
            return None
        return _success_result(
            candidate,
            strategy,
            nodes,
            elems,
            report,
            time.perf_counter() - started,
            tet_args,
            tet_kwargs,
        )

    result = attempt(source, "original")
    if result is not None:
        return result if return_result else (result.grid, result.nodes, result.elements)

    if repair_on_failure:
        started = time.perf_counter()
        try:
            repaired = repair_surface_with_meshfix(
                source,
                max_distance_ratio=repair_max_distance_ratio,
            )
            repaired = prepare_surface(repaired)
            validate_recovery_surface(
                source,
                repaired,
                max_distance_ratio=repair_max_distance_ratio,
            )
        except Exception as exc:
            _rejected_attempt(
                "meshfix",
                source,
                report,
                tet_args,
                tet_kwargs,
                "PyMeshFix recovery was rejected: {}".format(exc),
                time.perf_counter() - started,
            )
        else:
            result = attempt(repaired, "meshfix")
            if result is not None:
                return result if return_result else (result.grid, result.nodes, result.elements)

    if remesh_on_failure:
        started = time.perf_counter()
        try:
            remeshed = prepare_surface(
                uniform_remesh_surface(
                    source,
                    subdivisions=remesh_subdivisions,
                    clusters=remesh_clusters,
                    clean_tolerance=remesh_clean_tolerance,
                )
            )
            validate_recovery_surface(
                source,
                remeshed,
                max_distance_ratio=repair_max_distance_ratio,
            )
            remesh_strategy = "pyacvd"
        except Exception as remesh_error:
            if repair_on_failure and "remeshed" in locals():
                try:
                    remeshed = repair_surface_with_meshfix(
                        remeshed,
                        max_distance_ratio=repair_max_distance_ratio,
                    )
                    validate_recovery_surface(
                        source,
                        remeshed,
                        max_distance_ratio=repair_max_distance_ratio,
                    )
                    remesh_strategy = "pyacvd_meshfix"
                except Exception as repair_error:
                    _rejected_attempt(
                        "pyacvd_meshfix",
                        source,
                        report,
                        tet_args,
                        tet_kwargs,
                        "PyACVD recovery was rejected: {}; repair failed: {}".format(
                            remesh_error,
                            repair_error,
                        ),
                        time.perf_counter() - started,
                    )
                    remeshed = None
            else:
                _rejected_attempt(
                    "pyacvd",
                    source,
                    report,
                    tet_args,
                    tet_kwargs,
                    "PyACVD recovery was rejected: {}".format(remesh_error),
                    time.perf_counter() - started,
                )
                remeshed = None

        if remeshed is not None:
            result = attempt(remeshed, remesh_strategy)
            if result is not None:
                return result if return_result else (result.grid, result.nodes, result.elements)

    raise TetrahedralizationError(report)
