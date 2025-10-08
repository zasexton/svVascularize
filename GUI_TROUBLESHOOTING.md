# svVascularize GUI Troubleshooting Guide

## Problem: libGL/Mesa Driver Errors

The GUI requires OpenGL for 3D visualization. On Linux systems, especially in conda environments, there can be conflicts between system and conda libraries.

## Quick Solutions

### Solution 1: Use the Updated Launch Script
```bash
./launch_gui.sh
```

This script automatically:
- Detects and configures Mesa drivers
- Fixes libffi version conflicts
- Sets up software rendering

### Solution 2: Use System OpenGL (if you have a GPU)
```bash
./launch_gui.sh --system-gl
```

### Solution 3: Debug Mode
```bash
./launch_gui.sh --debug-gl
```

### Solution 4: Python Fallback Script
```bash
python launch_gui_fallback.py
```

Options:
- `--system-gl` - Use system OpenGL drivers
- `--osmesa` - Use OSMesa for off-screen rendering
- `--debug-gl` - Enable verbose debugging

## Manual Fixes

### Fix 1: Install Mesa Drivers in Conda
```bash
# Run the installation script
./install_mesa.sh

# Or manually install:
conda install -c conda-forge mesa-libgl-cos7-x86_64 mesa-dri-drivers-cos7-x86_64 mesalib libffi
```

### Fix 2: Set Environment Variables Manually
```bash
# Force software rendering
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe
export MESA_GL_VERSION_OVERRIDE=3.3

# Set DRI drivers path (adjust to your system)
export LIBGL_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri

# Fix libffi conflicts (if using conda)
export LD_PRELOAD=$CONDA_PREFIX/lib/libffi.so.7

# Launch
python -m svv.visualize.gui
```

### Fix 3: Install System Dependencies
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    mesa-utils \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libglu1-mesa \
    libosmesa6 \
    libffi8  # or libffi7 for older Ubuntu

# Fedora/RHEL
sudo dnf install -y \
    mesa-libGL \
    mesa-dri-drivers \
    mesa-libOSMesa \
    libffi
```

## Common Error Messages and Solutions

### Error: "Cannot create GLX context"
**Cause:** OpenGL context creation failed
**Solution:** Use software rendering with `./launch_gui.sh`

### Error: "undefined symbol: ffi_type_sint32"
**Cause:** libffi version mismatch
**Solution:** The updated launch_gui.sh script handles this automatically

### Error: "failed to load driver: swrast"
**Cause:** Mesa drivers not found or incompatible
**Solution:**
1. Run `./install_mesa.sh` to install conda mesa drivers
2. Or use system drivers with `./launch_gui.sh --system-gl`

### Error: "libOSMesa not found"
**Cause:** OSMesa library missing
**Solution:**
1. Install OSMesa: `conda install -c conda-forge mesalib`
2. Or system: `sudo apt-get install libosmesa6`

## Environment Variables

You can customize the GUI behavior with these environment variables:

- `SVV_GUI_GL_MODE`: Set to 'system' to use system OpenGL, 'software' for Mesa software rendering
- `SVV_GUI_SOFTWARE_DRIVER`: Choose driver ('llvmpipe', 'swr', 'softpipe')
- `SVV_LIBGL_DRIVERS_PATH`: Override the DRI drivers path
- `SVV_GUI_DEBUG_GL`: Set to '1' for debug output

Example:
```bash
export SVV_GUI_GL_MODE=software
export SVV_GUI_SOFTWARE_DRIVER=llvmpipe
./launch_gui.sh
```

## Testing OpenGL

Test your OpenGL setup:
```bash
# Check OpenGL info
glxinfo | grep "OpenGL"

# Test with glxgears
glxgears

# Check Mesa version
glxinfo | grep "Mesa"
```

## Remote Access Issues

If accessing via SSH or remote desktop:

1. **SSH with X11 Forwarding:**
   ```bash
   ssh -X user@host
   export LIBGL_ALWAYS_INDIRECT=0
   ./launch_gui.sh
   ```

2. **VNC/Remote Desktop:**
   - Software rendering is usually required
   - Use `./launch_gui.sh` (automatically uses software rendering)

3. **Docker/Container:**
   - Mount X11 socket
   - Use software rendering
   - May need to install mesa drivers in container

## Still Having Issues?

1. Collect debug information:
   ```bash
   ./launch_gui.sh --debug-gl > debug.log 2>&1
   ```

2. Check your environment:
   ```bash
   echo "Conda: $CONDA_PREFIX"
   echo "Python: $(which python)"
   python -c "import sys; print(sys.version)"
   conda list | grep -E "(pyside|pyqt|vtk|mesa|libffi)"
   ls -la /usr/lib/x86_64-linux-gnu/dri/ | head -5
   ```

3. Try the minimal test:
   ```bash
   python -c "from PySide6.QtWidgets import QApplication; app = QApplication([]); print('Qt OK')"
   python -c "import pyvista; print('PyVista OK')"
   ```

## Alternative: Use Jupyter Notebook

If the GUI continues to have issues, you can use the visualization tools in a Jupyter notebook:

```python
import pyvista as pv
from svv.domain import Domain

# Load your domain
domain = Domain.from_file('your_domain.dmn')

# Visualize in notebook
plotter = pv.Plotter(notebook=True)
plotter.add_mesh(domain.boundary)
plotter.show()
```