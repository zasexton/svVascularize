import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import numpy as np
import pyvista as pv
import pytest

from svv.domain.routines.mesh_diagnostics import TetGenWorkerError


tetrahedralize_mod = importlib.import_module("svv.domain.routines.tetrahedralize")
tetgen_worker_mod = importlib.import_module("svv.domain.routines.tetgen_worker")


def _write_array_worker(tmp_path, nodes_expression, elements_expression):
    worker = tmp_path / "array_worker.py"
    worker.write_text(
        "import sys\n"
        "import numpy as np\n"
        "nodes = {}\n".format(nodes_expression)
        + "elements = {}\n".format(elements_expression)
        + "np.savez(sys.argv[2], nodes=nodes, elems=elements)\n"
    )
    return worker


def test_worker_confines_tetgen_artifacts_to_its_temporary_directory(tmp_path, monkeypatch):
    caller_dir = tmp_path / "caller"
    caller_dir.mkdir()
    monkeypatch.chdir(caller_dir)
    worker = tmp_path / "artifact_worker.py"
    worker.write_text(
        """\
import pathlib
import sys
import numpy as np

pathlib.Path('_skipped.node').write_text('diagnostic')
pathlib.Path('_skipped.face').write_text('diagnostic')
nodes = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
np.savez(sys.argv[2], nodes=nodes, elems=elements)
"""
    )

    nodes, elements = tetrahedralize_mod._tetgen_worker_tetrahedralize(
        pv.Tetrahedron().extract_surface(),
        (),
        {"verbose": 0},
        str(worker),
        sys.executable,
    )
    leaked = list(caller_dir.glob("_skipped.*"))
    for path in leaked:
        path.unlink()

    assert nodes.shape == (4, 3)
    assert elements.shape == (1, 4)
    assert leaked == []


def test_worker_drains_large_output_without_deadlocking(tmp_path):
    worker = tmp_path / "verbose_worker.py"
    worker.write_text(
        """\
import sys
import numpy as np

print('x' * 200000, flush=True)
nodes = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
np.savez(sys.argv[2], nodes=nodes, elems=elements)
"""
    )
    driver = tmp_path / "driver.py"
    driver.write_text(
        """\
import importlib
import sys
import pyvista as pv

module = importlib.import_module('svv.domain.routines.tetrahedralize')
module._tetgen_worker_tetrahedralize(
    pv.Tetrahedron().extract_surface(), (), {'verbose': 0}, sys.argv[1], sys.executable
)
"""
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(tetrahedralize_mod.__file__).resolve().parents[3])

    completed = subprocess.run(
        [sys.executable, str(driver), str(worker)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_worker_raises_typed_recoverable_error_for_geometry_rejection(tmp_path):
    worker = tmp_path / "geometry_failure_worker.py"
    worker.write_text(
        """\
import sys

print('Warning: A segment and a facet intersect.')
print('  segment: [4,5] tag(-1).')
print('  facet triangle: [1,2,3] tag(-1).')
raise SystemExit(3)
"""
    )

    with pytest.raises(TetGenWorkerError) as error_info:
        tetrahedralize_mod._tetgen_worker_tetrahedralize(
            pv.Tetrahedron().extract_surface(),
            (),
            {"order": 1, "nobisect": True, "verbose": 0},
            str(worker),
            sys.executable,
            strategy="original",
        )

    error = error_info.value
    assert error.recoverable is True
    assert error.attempt.strategy == "original"
    assert error.attempt.status == "failed"
    assert error.attempt.diagnostics.return_code == 3
    assert error.attempt.diagnostics.segment_facet_intersections == 1


def test_worker_treats_internal_tetgen_error_as_geometry_rejection(tmp_path):
    worker = tmp_path / "internal_tetgen_failure_worker.py"
    worker.write_text(
        "raise RuntimeError('Internal TetGen error within `recoversubfaces`.')\n"
    )

    with pytest.raises(TetGenWorkerError) as error_info:
        tetrahedralize_mod._tetgen_worker_tetrahedralize(
            pv.Tetrahedron().extract_surface(),
            (),
            {"order": 1, "nobisect": True, "verbose": 0},
            str(worker),
            sys.executable,
            strategy="original",
        )

    error = error_info.value
    assert error.recoverable is True
    assert error.attempt.status == "failed"
    assert error.attempt.diagnostics.python_exception is True
    assert "Internal TetGen error" in error.attempt.diagnostics.stderr


def test_worker_treats_tetgen_self_intersection_error_as_geometry_rejection(
    tmp_path,
):
    worker = tmp_path / "self_intersection_failure_worker.py"
    worker.write_text(
        "raise RuntimeError('The input surface mesh contain self-intersections.')\n"
    )

    with pytest.raises(TetGenWorkerError) as error_info:
        tetrahedralize_mod._tetgen_worker_tetrahedralize(
            pv.Tetrahedron().extract_surface(),
            (),
            {"order": 1, "nobisect": True, "verbose": 0},
            str(worker),
            sys.executable,
            strategy="original",
        )

    error = error_info.value
    assert error.recoverable is True
    assert error.attempt.status == "failed"
    assert "self-intersections" in error.attempt.diagnostics.stderr


def test_worker_classifies_missing_dependency_as_nonrecoverable(tmp_path):
    worker = tmp_path / "dependency_failure_worker.py"
    worker.write_text("import missing_issue_102_dependency\n")

    with pytest.raises(TetGenWorkerError) as error_info:
        tetrahedralize_mod._tetgen_worker_tetrahedralize(
            pv.Tetrahedron().extract_surface(),
            (),
            {"verbose": 0},
            str(worker),
            sys.executable,
            strategy="original",
        )

    error = error_info.value
    assert error.recoverable is False
    assert "infrastructure" in str(error).lower()
    assert error.attempt.diagnostics.python_exception is True
    assert "ModuleNotFoundError" in error.attempt.diagnostics.stderr


def test_self_intersection_in_worker_path_does_not_hide_dependency_failure(tmp_path):
    worker = tmp_path / "self-intersection-dependency-worker.py"
    worker.write_text("import missing_issue_102_dependency\n")

    with pytest.raises(TetGenWorkerError) as error_info:
        tetrahedralize_mod._tetgen_worker_tetrahedralize(
            pv.Tetrahedron().extract_surface(),
            (),
            {"verbose": 0},
            str(worker),
            sys.executable,
            strategy="original",
        )

    error = error_info.value
    assert error.recoverable is False
    assert "infrastructure" in str(error).lower()
    assert error.attempt.diagnostics.python_exception is True
    assert "ModuleNotFoundError" in error.attempt.diagnostics.stderr


def test_worker_rejects_nonfinite_result_arrays_as_infrastructure_failure(tmp_path):
    worker = tmp_path / "invalid_result_worker.py"
    worker.write_text(
        """\
import sys
import numpy as np

nodes = np.array([[np.nan, 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
elements = np.array([[0, 1, 2, 3]], dtype=np.int64)
np.savez(sys.argv[2], nodes=nodes, elems=elements)
"""
    )

    with pytest.raises(TetGenWorkerError) as error_info:
        tetrahedralize_mod._tetgen_worker_tetrahedralize(
            pv.Tetrahedron().extract_surface(),
            (),
            {"verbose": 0},
            str(worker),
            sys.executable,
        )

    error = error_info.value
    assert error.recoverable is False
    assert error.attempt.status == "invalid-output"
    assert "finite" in str(error).lower()


def test_worker_normalizes_one_based_connectivity_once(tmp_path):
    worker = _write_array_worker(
        tmp_path,
        "np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])",
        "np.array([[1, 2, 3, 4]], dtype=np.int64)",
    )

    _, elements = tetrahedralize_mod._tetgen_worker_tetrahedralize(
        pv.Tetrahedron().extract_surface(),
        (),
        {"verbose": 0},
        str(worker),
        sys.executable,
    )

    assert np.array_equal(elements, [[0, 1, 2, 3]])


@pytest.mark.parametrize(
    ("nodes_expression", "elements_expression", "message"),
    [
        (
            "np.empty((0, 3))",
            "np.array([[0, 1, 2, 3]], dtype=np.int64)",
            "nodes",
        ),
        (
            "np.zeros((4, 2))",
            "np.array([[0, 1, 2, 3]], dtype=np.int64)",
            "nodes",
        ),
        (
            "np.zeros((4, 3))",
            "np.empty((0, 4), dtype=np.int64)",
            "elements",
        ),
        (
            "np.zeros((4, 3))",
            "np.array([[0, 1, 2]], dtype=np.int64)",
            "elements",
        ),
        (
            "np.zeros((4, 3))",
            "np.array([[0., 1., 2., 3.]])",
            "integer",
        ),
        (
            "np.zeros((4, 3))",
            "np.array([[0, 1, 2, 4]], dtype=np.int64)",
            "range",
        ),
        (
            "np.zeros((4, 3))",
            "np.array([[-1, 1, 2, 3]], dtype=np.int64)",
            "range",
        ),
    ],
)
def test_worker_rejects_malformed_result_arrays(
    tmp_path,
    nodes_expression,
    elements_expression,
    message,
):
    worker = _write_array_worker(tmp_path, nodes_expression, elements_expression)

    with pytest.raises(TetGenWorkerError) as error_info:
        tetrahedralize_mod._tetgen_worker_tetrahedralize(
            pv.Tetrahedron().extract_surface(),
            (),
            {"verbose": 0},
            str(worker),
            sys.executable,
        )

    assert error_info.value.recoverable is False
    assert error_info.value.attempt.status == "invalid-output"
    assert message in str(error_info.value).lower()


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal return codes are unavailable")
def test_worker_reports_native_signal_as_recoverable(tmp_path):
    worker = tmp_path / "signal_worker.py"
    worker.write_text(
        "import os\n"
        "import signal\n"
        "os.kill(os.getpid(), signal.SIGABRT)\n"
    )

    with pytest.raises(TetGenWorkerError) as error_info:
        tetrahedralize_mod._tetgen_worker_tetrahedralize(
            pv.Tetrahedron().extract_surface(),
            (),
            {"verbose": 0},
            str(worker),
            sys.executable,
        )

    diagnostics = error_info.value.attempt.diagnostics
    assert error_info.value.recoverable is True
    assert diagnostics.return_code == -signal.SIGABRT
    assert diagnostics.signal_name == "SIGABRT"


def test_worker_wraps_launch_failure_as_infrastructure_error(tmp_path):
    missing_worker = tmp_path / "missing_worker.py"

    with pytest.raises(TetGenWorkerError) as error_info:
        tetrahedralize_mod._tetgen_worker_tetrahedralize(
            pv.Tetrahedron().extract_surface(),
            (),
            {"verbose": 0},
            str(missing_worker),
            str(tmp_path / "missing-python"),
        )

    assert error_info.value.recoverable is False
    assert error_info.value.attempt.status == "infrastructure-error"


def test_worker_wraps_unserializable_configuration_as_infrastructure_error(tmp_path):
    worker = tmp_path / "unused_worker.py"
    worker.write_text("pass\n")

    with pytest.raises(TetGenWorkerError) as error_info:
        tetrahedralize_mod._tetgen_worker_tetrahedralize(
            pv.Tetrahedron().extract_surface(),
            (),
            {"unsupported": object()},
            str(worker),
            sys.executable,
        )

    assert error_info.value.recoverable is False
    assert error_info.value.attempt.status == "infrastructure-error"


def test_worker_removes_temporary_directory_after_failure(tmp_path, monkeypatch):
    temp_root = tmp_path / "worker-temp"
    temp_root.mkdir()
    monkeypatch.setattr(tetrahedralize_mod.tempfile, "tempdir", str(temp_root))
    worker = tmp_path / "failing_worker.py"
    worker.write_text("raise SystemExit(7)\n")

    with pytest.raises(TetGenWorkerError):
        tetrahedralize_mod._tetgen_worker_tetrahedralize(
            pv.Tetrahedron().extract_surface(),
            (),
            {"verbose": 0},
            str(worker),
            sys.executable,
        )

    assert list(temp_root.iterdir()) == []


def test_worker_wraps_missing_result_archive_as_infrastructure_failure(tmp_path):
    worker = tmp_path / "missing_result_worker.py"
    worker.write_text("pass\n")

    with pytest.raises(TetGenWorkerError) as error_info:
        tetrahedralize_mod._tetgen_worker_tetrahedralize(
            pv.Tetrahedron().extract_surface(),
            (),
            {"verbose": 0},
            str(worker),
            sys.executable,
        )

    error = error_info.value
    assert error.recoverable is False
    assert error.attempt.status == "invalid-output"
    assert "invalid output" in str(error).lower()


@pytest.mark.parametrize("return_tuple", [True, False])
def test_tetgen_worker_main_saves_supported_result_formats(
    tmp_path,
    monkeypatch,
    return_tuple,
):
    surface_path = tmp_path / "surface.vtp"
    output_path = tmp_path / "tet.npz"
    config_path = tmp_path / "config.json"
    pv.Tetrahedron().extract_surface().save(surface_path)
    config_path.write_text(json.dumps({"args": [], "kwargs": {"verbose": 0}}))
    expected_nodes = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    expected_elements = np.array([[0, 1, 2, 3]], dtype=np.int64)

    class TetGenResult:
        def __init__(self, surface):
            self.node = expected_nodes
            self.elem = expected_elements

        def tetrahedralize(self, *args, **kwargs):
            if return_tuple:
                return self.node, self.elem
            return pv.UnstructuredGrid()

    monkeypatch.setattr(tetgen_worker_mod.tetgen, "TetGen", TetGenResult)

    tetgen_worker_mod.main(str(surface_path), str(output_path), str(config_path))

    with np.load(output_path) as result:
        assert np.array_equal(result["nodes"], expected_nodes)
        assert np.array_equal(result["elems"], expected_elements)
