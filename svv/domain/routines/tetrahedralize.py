import tetgen
import pymeshfix
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
import shutil
import json

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

def tetrahedralize(surface: pv.PolyData,
                   *tet_args,
                   worker_script: str = dirpath+os.sep+"tetgen_worker.py",
                   python_exe: str = sys.executable,
                   **tet_kwargs):
    """
    Tetrahedralize a surface mesh using TetGen.

    Parameters
    ----------
    surface_mesh : PyMesh mesh object
        The surface mesh to tetrahedralize.
    verbose : bool
        A flag to indicate if mesh fixing should be verbose.
    kwargs : dict
        A dictionary of keyword arguments to be passed to TetGen.

    Returns
    -------
    mesh : PyMesh mesh object
        An unstructured grid mesh representing the tetrahedralized
        volume enclosed by the surface mesh manifold.
    """
    tet_kwargs.setdefault("verbose", 0)

    with tempfile.TemporaryDirectory() as tmpdir:
        surface_path = os.path.join(tmpdir, "surface.vtp")
        out_path = os.path.join(tmpdir, "tet.npz")
        config_path = os.path.join(tmpdir, "config.json")

        cfg = {
            "args": list(tet_args),
            "kwargs": tet_kwargs,
        }
        with open(config_path, "w") as f:
            json.dump(cfg, f)

        # Save the surface mesh so the worker can read it
        surface.save(surface_path)

        # Command: call the worker script as a separate Python process
        cmd = [python_exe, worker_script, surface_path, out_path, config_path]

        # Start the worker process
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,   # decode to strings
        )

        spinner = cycle(["⠋", "⠙", "⠹", "⠸", "⠼",
                         "⠴", "⠦", "⠧", "⠇", "⠏"])
        start_time = time.time()

        # Print label once
        sys.stdout.write("TetGen meshing| ")
        sys.stdout.flush()

        # Live spinner loop
        while proc.poll() is None:
            # Compute elapsed time
            elapsed = time.time() - start_time
            elapsed_str = format_elapsed(elapsed)

            # Build left side message
            spin_char = next(spinner)
            left = f"TetGen meshing| {spin_char}"

            # Get terminal width (fallback if IDE doesn't report it)
            try:
                width = shutil.get_terminal_size(fallback=(80, 20)).columns
            except Exception:
                width = 80

            # Compute spacing so elapsed time is right-aligned
            # We'll always keep at least one space between left and right
            min_gap = 1
            total_len = len(left) + min_gap + len(elapsed_str)
            if total_len <= width:
                spaces = width - len(left) - len(elapsed_str)
            else:
                # If line is longer than terminal, don't try to be clever; just put a single space
                spaces = min_gap

            line = f"{left}{' ' * spaces}{elapsed_str}"

            # '\r' to return to the start of the same line and overwrite
            sys.stdout.write("\r" + line)
            sys.stdout.flush()

            time.sleep(0.1)

        # Finish line
        sys.stdout.write("\n")
        sys.stdout.flush()

        # Collect output (so the pipes don't hang)
        stdout, stderr = proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError(
                f"TetGen worker failed with code {proc.returncode}\n"
                f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
            )

        # Load results
        data = np.load(out_path)
        nodes = data["nodes"]
        elems = data["elems"]

    if elems.min() == 1:
        elems = elems - 1

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
