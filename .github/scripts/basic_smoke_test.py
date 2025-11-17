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
    t.set_root()
    t.n_add(10)

    sim = Simulation(t)
    sim.build_meshes(fluid=True, tissue=True, boundary_layer=False)


if __name__ == "__main__":
    main()

