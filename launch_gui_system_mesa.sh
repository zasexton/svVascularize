#!/bin/bash
# Launch GUI using system Mesa libraries instead of conda's broken ones

echo "Launching svVascularize GUI with system Mesa..."
echo "==============================================="
echo ""

# Unset conda's Mesa paths and use system libraries
unset LIBGL_DRIVERS_PATH
export SVV_GUI_GL_MODE=system

# Use system OpenGL libraries
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Enable software rendering if no GPU
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_LOADER_DRIVER_OVERRIDE=swrast

echo "Using system Mesa libraries from /usr/lib/x86_64-linux-gnu"
echo "OpenGL mode: software rendering (swrast)"
echo ""

# Check if system Mesa is installed
if [ ! -f /usr/lib/x86_64-linux-gnu/dri/swrast_dri.so ]; then
    echo "ERROR: System Mesa not found!"
    echo ""
    echo "Please install system Mesa libraries:"
    echo "  sudo apt-get update"
    echo "  sudo apt-get install -y libgl1-mesa-glx libgl1-mesa-dri mesa-utils"
    echo ""
    exit 1
fi

echo "System Mesa found: OK"
echo "Launching GUI..."
echo ""

# Launch GUI
python -m svv.visualize.gui "$@"
