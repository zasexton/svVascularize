#!/bin/bash
# Launch GUI using Xvfb (virtual framebuffer) to avoid OpenGL issues

# Check if Xvfb is installed
if ! command -v Xvfb &> /dev/null; then
    echo "Xvfb not found. Installing..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y xvfb
    elif command -v yum &> /dev/null; then
        sudo yum install -y xorg-x11-server-Xvfb
    else
        echo "Please install Xvfb manually:"
        echo "  Ubuntu/Debian: sudo apt-get install xvfb"
        echo "  RedHat/CentOS: sudo yum install xorg-x11-server-Xvfb"
        exit 1
    fi
fi

# Find an available display number
DISPLAY_NUM=99
while [ -e /tmp/.X11-unix/X$DISPLAY_NUM ]; do
    DISPLAY_NUM=$((DISPLAY_NUM + 1))
done

echo "Starting virtual display :$DISPLAY_NUM"

# Start Xvfb in the background
Xvfb :$DISPLAY_NUM -screen 0 1024x768x24 &
XVFB_PID=$!

# Give Xvfb time to start
sleep 2

# Set display
export DISPLAY=:$DISPLAY_NUM

# Configure software rendering
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe
export MESA_GL_VERSION_OVERRIDE=3.3

# Add svVascularize to Python path
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Launch the GUI
echo "Launching GUI on virtual display..."
python -c "
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import sys
try:
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt
    QApplication.setAttribute(Qt.AA_UseSoftwareOpenGL)
except:
    pass

from svv.visualize.gui import VascularizeGUI

app = QApplication(sys.argv)
gui = VascularizeGUI()

# Take a screenshot for verification
gui.show()
app.processEvents()

print('GUI launched successfully in virtual display')
print('Note: This is running in a virtual framebuffer.')
print('To interact with the GUI, you need a VNC viewer or similar.')

try:
    app.exec()
except AttributeError:
    app.exec_()
" || echo "Failed to launch GUI"

# Cleanup
echo "Shutting down virtual display..."
kill $XVFB_PID 2>/dev/null