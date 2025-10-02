#!/bin/bash
# Launcher script for svVascularize GUI with software rendering

# Set environment for software rendering
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe
export MESA_GL_VERSION_OVERRIDE=3.3

# Set DRI driver path to conda's Mesa drivers
if [ -n "$CONDA_PREFIX" ]; then
    export LIBGL_DRIVERS_PATH="$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot/usr/lib64/dri"
fi

# Add svVascularize to Python path
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Launch the GUI
python -c "
import sys
from qtpy.QtWidgets import QApplication
from svv.visualize.gui import VascularizeGUI

app = QApplication(sys.argv)
gui = VascularizeGUI()
gui.show()
sys.exit(app.exec_())
"
