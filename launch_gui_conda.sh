#!/bin/bash
# Launch svVascularize GUI using conda environment with proper Mesa setup

# Default environment name
ENV_NAME="${1:-svv}"

# Remove env name from args if provided
if [ "$#" -ge 1 ]; then
    shift
fi

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please ensure conda is installed."
    exit 1
fi

# Check if environment exists
if ! conda env list | grep -q "^$ENV_NAME "; then
    echo "Error: Conda environment '$ENV_NAME' not found."
    echo "Please run: ./setup_mesa_conda.sh $ENV_NAME"
    exit 1
fi

echo "Using conda environment: $ENV_NAME"

# Get conda base directory
CONDA_BASE=$(conda info --base)

# Set up environment for conda-only Mesa
export CONDA_ENV_PATH="$CONDA_BASE/envs/$ENV_NAME"

# Create a wrapper script that will be executed in the conda environment
cat << 'EOF' > /tmp/svv_gui_launcher.py
#!/usr/bin/env python
import os
import sys

# Ensure we're using conda's Mesa
conda_env = os.environ.get('CONDA_ENV_PATH', '')
if conda_env:
    # Set Mesa environment variables
    os.environ['LIBGL_DRIVERS_PATH'] = os.path.join(conda_env, 'lib', 'dri')
    os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
    os.environ['GALLIUM_DRIVER'] = 'llvmpipe'
    os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'

    # Ensure conda's libraries are used
    lib_path = os.path.join(conda_env, 'lib')
    if 'LD_LIBRARY_PATH' in os.environ:
        os.environ['LD_LIBRARY_PATH'] = f"{lib_path}:{os.environ['LD_LIBRARY_PATH']}"
    else:
        os.environ['LD_LIBRARY_PATH'] = lib_path

# Add svVascularize to path
sys.path.insert(0, os.getcwd())

# Try to import and run GUI
try:
    # Try primary import path
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt

    # Set software OpenGL
    try:
        QApplication.setAttribute(Qt.AA_UseSoftwareOpenGL)
    except:
        pass

    # Import GUI
    try:
        from svv.visualize.gui import VascularizeGUI
    except ImportError:
        # Try alternative import
        from svv.visualize.gui.main_window import VascularizeGUI

    # Create and run application
    app = QApplication(sys.argv)
    gui = VascularizeGUI()
    gui.show()

    try:
        sys.exit(app.exec())
    except AttributeError:
        sys.exit(app.exec_())

except ImportError as e:
    print(f"Import error: {e}")
    print("\nPlease ensure all dependencies are installed:")
    print("  ./setup_mesa_conda.sh", os.environ.get('ENV_NAME', 'svv'))
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

# Parse command line options
DEBUG=0
for arg in "$@"; do
    case $arg in
        --debug)
            DEBUG=1
            ;;
        --help)
            echo "Usage: $0 [ENV_NAME] [OPTIONS]"
            echo ""
            echo "Arguments:"
            echo "  ENV_NAME       Conda environment name (default: svv)"
            echo ""
            echo "Options:"
            echo "  --debug        Enable debug output"
            echo "  --help         Show this help message"
            exit 0
            ;;
    esac
done

# Set debug environment if requested
if [ "$DEBUG" -eq 1 ]; then
    export MESA_DEBUG=1
    export LIBGL_DEBUG=verbose
    export SVV_GUI_DEBUG_GL=1
    echo "Debug mode enabled"
fi

# Add current directory to Python path
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Use conda run to execute in the proper environment with isolated libraries
echo "Launching svVascularize GUI..."

# First, try to set up the environment properly
conda run -n "$ENV_NAME" --no-capture-output \
    env CONDA_ENV_PATH="$CONDA_ENV_PATH" \
    PYTHONPATH="$PYTHONPATH" \
    LD_LIBRARY_PATH="$CONDA_ENV_PATH/lib:$LD_LIBRARY_PATH" \
    LIBGL_DRIVERS_PATH="$CONDA_ENV_PATH/lib/dri" \
    LIBGL_ALWAYS_SOFTWARE=1 \
    GALLIUM_DRIVER=llvmpipe \
    MESA_GL_VERSION_OVERRIDE=3.3 \
    python /tmp/svv_gui_launcher.py || {

    echo ""
    echo "GUI launch failed. Trying fallback with OSMesa..."

    # Try with OSMesa
    conda run -n "$ENV_NAME" --no-capture-output \
        env CONDA_ENV_PATH="$CONDA_ENV_PATH" \
        PYTHONPATH="$PYTHONPATH" \
        PYVISTA_USE_OSMESA=true \
        python /tmp/svv_gui_launcher.py || {

        echo ""
        echo "========================================="
        echo "Failed to launch GUI."
        echo ""
        echo "Troubleshooting steps:"
        echo "1. Ensure Mesa is properly installed:"
        echo "   ./setup_mesa_conda.sh $ENV_NAME"
        echo ""
        echo "2. Check your display:"
        echo "   echo \$DISPLAY"
        echo ""
        echo "3. Try debug mode:"
        echo "   $0 $ENV_NAME --debug"
        echo ""
        echo "4. For SSH connections, ensure X11 forwarding:"
        echo "   ssh -X user@host"
        echo "========================================="
        exit 1
    }
}

# Clean up
rm -f /tmp/svv_gui_launcher.py