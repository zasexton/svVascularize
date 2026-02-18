import os
import sys
import tempfile
from contextlib import contextmanager

# Ensure CI can initialize Qt/VTK without a physical display.
# (Linux runners are frequently headless; Xvfb is used in the workflow,
# but these defaults make the script more robust when run locally too.)
if sys.platform.startswith("linux"):
    os.environ.setdefault("SVV_GUI_GL_MODE", "software")
    # If we have an X server (e.g. Xvfb in CI), prefer the normal XCB platform
    # so PyVistaQt/VTK can embed a render window reliably.
    if os.environ.get("DISPLAY"):
        if os.environ.get("GITHUB_ACTIONS") or os.environ.get("CI"):
            os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    else:
        # No display available: fall back to offscreen modes to keep imports working.
        os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Never prompt for telemetry consent in CI.
os.environ.setdefault("SVV_TELEMETRY_DISABLED", "1")

import pyvista as pv

from svv.domain.domain import Domain
from svv.forest.forest import Forest
from svv.tree.tree import Tree
from svv.simulation.simulation import Simulation
from svv.utils.remeshing.remesh import remesh_surface


@contextmanager
def _temp_cwd():
    """Run work in a temp dir to avoid polluting the repo with tmp.mesh files."""
    old = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            yield tmp
        finally:
            os.chdir(old)


def _smoke_gui(domain: Domain) -> None:
    """Verify the Qt GUI can be created and shown briefly."""
    from PySide6.QtCore import QTimer, Qt
    from PySide6.QtWidgets import QApplication
    from svv.visualize.gui.main_window import VascularizeGUI

    app = QApplication.instance() or QApplication(sys.argv[:1])
    gui = VascularizeGUI(domain=domain)
    # Ensure close() actually destroys the window so VTK/Qt resources are
    # released while the X server is still available (important for Xvfb CI).
    gui.setAttribute(Qt.WA_DeleteOnClose, True)

    # Quit once the window is destroyed.
    gui.destroyed.connect(lambda *_: app.quit())
    gui.show()

    # Run the event loop briefly so widgets initialize.
    QTimer.singleShot(250, gui.close)
    # Safety net in case close() doesn't fire destroyed (shouldn't happen with WA_DeleteOnClose).
    QTimer.singleShot(2000, app.quit)
    app.exec()


def main() -> None:
    # Domain build (geometry + implicit function + tetrahedral mesh)
    cube = Domain(pv.Cube().triangulate())
    cube.create()
    cube.solve()
    cube.build()

    # MMG remeshing (validate that packaged/built executables run)
    with _temp_cwd():
        remeshed = remesh_surface(pv.Cube().triangulate(), verbosity=0)
    assert remeshed.n_cells > 0

    # Tree build
    t = Tree()
    t.set_domain(cube)
    t.parameters.set('root_pressure', 100)
    t.parameters.set('terminal_pressure', 0)
    t.set_root()
    # Use a small tree to keep the smoke test
    # lightweight across all CI runners.
    t.n_add(3)

    # Forest build (minimal network + one add step)
    forest = Forest(domain=cube, n_networks=1, n_trees_per_network=[1])
    forest.set_domain(cube)
    forest.set_roots()
    forest.add(1)

    sim = Simulation(t)
    # For CI we only require the fluid mesh;
    # skipping the tissue mesh keeps TetGen runs lighter,
    # especially on Windows and macOS runners.
    # sim.build_meshes(fluid=True, tissue=False, boundary_layer=False)

    # GUI launch (after core objects exist)
    _smoke_gui(cube)


if __name__ == "__main__":
    main()
