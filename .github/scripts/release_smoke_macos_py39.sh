#!/usr/bin/env bash
set -euo pipefail

python -m venv .venv-smoke-py39
source .venv-smoke-py39/bin/activate

# universal2 wheel selection on Apple Silicon requires a reasonably recent pip
python -m pip install --upgrade "pip>=21.0.1" setuptools wheel

wheel=$(ls dist/svv-*.whl | head -n 1)
if [ -z "${wheel}" ]; then
  echo "No svv wheel found in dist/"
  exit 1
fi

# Force TetGen to resolve from a wheel during the release smoke test.
python -m pip install --only-binary=tetgen "${wheel}"

python - <<'PY'
import importlib.metadata as md
import os
import tempfile

tetgen_version = md.version("tetgen")
print("tetgen version:", tetgen_version, flush=True)
if tetgen_version != "0.6.4":
    raise SystemExit(f"Expected tetgen 0.6.4 on macOS Python 3.9, got {tetgen_version}")

os.chdir(tempfile.mkdtemp(prefix="svv-release-smoke-"))

import pyvista as pv
import tetgen
from svv.domain.domain import Domain
from svv.simulation.simulation import Simulation

surface = pv.Cube().triangulate()
tgen = tetgen.TetGen(surface)
nodes, elems = tgen.tetrahedralize(verbose=0)
print("tetgen nodes:", len(nodes), "elems:", len(elems), flush=True)
if len(nodes) == 0 or len(elems) == 0:
    raise SystemExit("TetGen smoke test produced no tetrahedral output")

domain = Domain(surface)
domain.create()
domain.solve()

sim = Simulation.__name__
print("Imported Domain and Simulation successfully:", sim, flush=True)
PY
