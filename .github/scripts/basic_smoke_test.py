import pyvista as pv

from svv.domain.domain import Domain
from svv.tree.tree import Tree
from svv.simulation.simulation import Simulation


def main() -> None:
    cube = Domain(pv.Cube())
    cube.create()
    cube.solve()
    cube.build()

    t = Tree()
    t.set_domain(cube)
    t.parameters.set('root_pressure', 100)
    t.parameters.set('terminal_pressure', 0)
    t.set_root()
    # Use a small tree to keep the smoke test
    # lightweight across all CI runners.
    t.n_add(3)

    sim = Simulation(t)
    # For CI we only require the fluid mesh;
    # skipping the tissue mesh keeps TetGen runs lighter,
    # especially on Windows and macOS runners.
    # sim.build_meshes(fluid=True, tissue=False, boundary_layer=False)


if __name__ == "__main__":
    main()
