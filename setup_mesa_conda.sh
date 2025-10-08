#!/bin/bash
# Setup Mesa drivers completely within conda environment to avoid system conflicts

set -e

echo "========================================="
echo "svVascularize Mesa Setup for Conda"
echo "========================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Miniconda or Anaconda first."
    exit 1
fi

# Detect or create conda environment
ENV_NAME="${1:-svv}"
echo "Setting up Mesa in conda environment: $ENV_NAME"

# Check if environment exists
if conda env list | grep -q "^$ENV_NAME "; then
    echo "Environment '$ENV_NAME' exists."
else
    echo "Creating new conda environment '$ENV_NAME' with Python 3.9..."
    conda create -n "$ENV_NAME" python=3.9 -y
fi

# Activate environment and install packages
echo "Installing Mesa and dependencies in $ENV_NAME environment..."

# Use conda run to execute in the environment
conda run -n "$ENV_NAME" conda install -y -c conda-forge \
    mesalib \
    mesa-libgl-devel-cos7-x86_64 \
    libglu \
    xorg-libx11 \
    xorg-libxext \
    xorg-libxrender \
    xorg-libxau \
    xorg-libxdmcp \
    libxcb \
    libffi \
    pyside6 \
    pyvista \
    pyvistaqt \
    vtk \
    numpy || {
    echo "Warning: Some packages failed to install, trying alternative packages..."

    # Try alternative mesa packages
    conda run -n "$ENV_NAME" conda install -y -c conda-forge \
        mesa \
        mesa-libgl-cos6-x86_64 \
        libglu \
        libffi \
        pyside6 \
        pyvista \
        pyvistaqt || true
}

# Install OSMesa for fallback
conda run -n "$ENV_NAME" conda install -y -c conda-forge osmesa || {
    echo "OSMesa not available, trying mesalib..."
    conda run -n "$ENV_NAME" conda install -y -c conda-forge mesalib || true
}

echo ""
echo "Checking installation..."

# Verify installation
conda run -n "$ENV_NAME" python -c "
import sys
print(f'Python: {sys.version}')
try:
    import PySide6
    print(f'PySide6: OK - {PySide6.__version__}')
except ImportError as e:
    print(f'PySide6: FAILED - {e}')
try:
    import pyvista
    print(f'PyVista: OK - {pyvista.__version__}')
except ImportError as e:
    print(f'PyVista: FAILED - {e}')
try:
    import pyvistaqt
    print('PyVistaQt: OK')
except ImportError as e:
    print(f'PyVistaQt: FAILED - {e}')
"

echo ""
echo "========================================="
echo "Setup complete!"
echo ""
echo "To use the GUI, run:"
echo "  conda activate $ENV_NAME"
echo "  ./launch_gui_conda.sh"
echo ""
echo "Or directly:"
echo "  ./launch_gui_conda.sh $ENV_NAME"
echo "========================================="