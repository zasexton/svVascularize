# OpenGL Troubleshooting for svVascularize GUI

## Problem

When launching the GUI, you may see errors like:

```
libGL error: MESA-LOADER: failed to open swrast: libLLVM-7-rhel.so: cannot open shared object file
libGL error: failed to load driver: swrast
vtkXOpenGLRenderWindow: Cannot create GLX context
vtkOpenGLRenderWindow: Failed to initialize OpenGL functions!
Segmentation fault (core dumped)
```

This indicates that VTK/PyVista cannot initialize OpenGL for 3D rendering.

## Root Cause

The issue occurs because:
1. Missing Mesa software rendering libraries (libLLVM, swrast driver)
2. VTK trying to use hardware OpenGL but no GPU available
3. Incorrect library paths for Mesa drivers

## Solutions

### Solution 1: Install Mesa Libraries (Recommended)

Run the provided installation script:

```bash
cd /path/to/svVascularize
chmod +x install_mesa.sh
./install_mesa.sh
```

Or manually install with conda:

```bash
conda install -c conda-forge mesalib libllvm7
```

### Solution 2: Use System Mesa

If you have Mesa installed system-wide:

```bash
# Find your Mesa DRI drivers
find /usr/lib -name "swrast_dri.so" 2>/dev/null

# Set environment variable (example path)
export LIBGL_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri

# Then launch GUI
python -m svv.visualize.gui
```

### Solution 3: Check Conda Installation

Ensure your conda environment has the necessary packages:

```bash
conda list | grep -E "mesa|llvm|vtk|pyvista"

# Should show:
# mesalib
# libllvm7 or libllvm11
# vtk
# pyvista
# pyvistaqt
```

If any are missing:

```bash
conda install -c conda-forge mesalib libllvm11 vtk pyvista pyvistaqt
```

### Solution 4: Manual Library Check

Check if Mesa libraries exist:

```bash
# Check conda environment
ls $CONDA_PREFIX/lib/dri/

# Should contain: swrast_dri.so

# Check for LLVM
ls $CONDA_PREFIX/lib/libLLVM*.so

# Should show LLVM libraries
```

If `swrast_dri.so` is missing:

```bash
conda install -c conda-forge mesa-libgl-devel-cos7-x86_64
```

## Verification

After installing, verify the setup:

```bash
# Check environment
echo $CONDA_PREFIX
ls $CONDA_PREFIX/lib/dri/
ls $CONDA_PREFIX/lib/libLLVM*.so

# Test Python imports
python -c "import vtk; print('VTK:', vtk.VTK_VERSION)"
python -c "import pyvista; print('PyVista:', pyvista.__version__)"
python -c "from pyvistaqt import QtInteractor; print('PyVistaQt: OK')"
```

All should succeed without errors.

## Environment Variables

The GUI automatically sets these for software rendering:

```bash
LIBGL_ALWAYS_SOFTWARE=1
GALLIUM_DRIVER=llvmpipe
MESA_GL_VERSION_OVERRIDE=3.3
MESA_LOADER_DRIVER_OVERRIDE=llvmpipe
QT_OPENGL=software
```

You can override them if needed:

```bash
# Use system OpenGL instead of software
export SVV_GUI_GL_MODE=system

# Specify custom DRI path
export SVV_LIBGL_DRIVERS_PATH=/custom/path/to/dri

# Enable debug output
export SVV_GUI_DEBUG_GL=1
```

## Testing the GUI

After fixing OpenGL, test the GUI:

```bash
# Launch GUI
python -c "from svv.visualize.gui import launch_gui; launch_gui(style='cad')"
```

If successful:
- Window opens without errors
- 3D viewport is visible (dark gray for CAD style, light gray for modern style)
- No segmentation faults

If errors persist, check the error handling section below.

## Error Handling in Code

The GUI now includes fallback handling:

1. **If OpenGL fails**: Shows warning message in viewport area
2. **GUI still functional**: Can use all controls except 3D visualization
3. **Error message displays**: Instructions for fixing OpenGL issues

The GUI will display:
```
3D Visualization unavailable: [error details]

This may be due to missing OpenGL libraries.
The GUI will function with limited 3D visualization.

To fix, install Mesa libraries:
conda install -c conda-forge mesalib
```

## Platform-Specific Notes

### Linux (Ubuntu/Debian)
```bash
# System-wide Mesa
sudo apt-get install libgl1-mesa-glx libgl1-mesa-dri

# Or use conda (recommended)
conda install -c conda-forge mesalib
```

### Linux (RedHat/CentOS)
```bash
# System-wide Mesa
sudo yum install mesa-libGL mesa-dri-drivers

# Or use conda (recommended)
conda install -c conda-forge mesalib
```

### WSL (Windows Subsystem for Linux)
```bash
# WSL2 with WSLg (Windows 11)
# Should work automatically with system OpenGL

# WSL1 or older
# Must use software rendering
export LIBGL_ALWAYS_SOFTWARE=1
conda install -c conda-forge mesalib
```

### macOS
Generally works out of the box. If issues:
```bash
conda install -c conda-forge vtk pyvista pyvistaqt
```

## Common Error Messages and Fixes

### Error: "libLLVM-7-rhel.so: cannot open shared object file"
**Fix**: Install libllvm
```bash
conda install -c conda-forge libllvm11
```

### Error: "failed to open swrast"
**Fix**: Install Mesa DRI drivers
```bash
conda install -c conda-forge mesa-libgl-devel-cos7-x86_64
```

### Error: "Cannot create GLX context"
**Fix**: Use software rendering
```bash
export LIBGL_ALWAYS_SOFTWARE=1
conda install -c conda-forge mesalib
```

### Error: "Segmentation fault (core dumped)"
**Fix**: Install all Mesa dependencies
```bash
conda install -c conda-forge mesalib libllvm11 mesa-libgl-devel-cos7-x86_64
```

### Error: "Unable to find a valid OpenGL 3.2 or later implementation"
**Fix**: Update Mesa or use software rendering
```bash
conda install -c conda-forge mesalib>=18.3
export MESA_GL_VERSION_OVERRIDE=3.3
```

## Advanced Debugging

### Enable VTK Debug Output
```python
import vtk
vtk.vtkObject.GlobalWarningDisplayOn()

from svv.visualize.gui import launch_gui
launch_gui(style='cad')
```

### Check OpenGL Info
```python
import vtk
render_window = vtk.vtkRenderWindow()
render_window.SetOffScreenRendering(1)
try:
    render_window.Render()
    print("OpenGL Version:", render_window.GetOpenGLVersion())
except:
    print("OpenGL initialization failed")
```

### Test PyVista Directly
```python
import pyvista as pv
pv.set_plot_theme('document')  # Use software rendering theme

# Create simple plot
sphere = pv.Sphere()
try:
    plotter = pv.Plotter()
    plotter.add_mesh(sphere)
    plotter.show()
    print("PyVista rendering works!")
except Exception as e:
    print(f"PyVista error: {e}")
```

## Alternative: Use launch_gui_safe.py

The repository includes a safe launcher that handles OpenGL issues:

```bash
python launch_gui_safe.py
```

This script:
- Automatically configures software rendering
- Falls back to minimal GUI if 3D fails
- Provides detailed error messages
- Suggests fixes

## Still Having Issues?

1. **Check Mesa version**:
   ```bash
   conda list mesa
   # Should be >= 18.3
   ```

2. **Reinstall packages**:
   ```bash
   conda remove --force vtk pyvista pyvistaqt
   conda install -c conda-forge vtk pyvista pyvistaqt mesalib
   ```

3. **Create fresh environment**:
   ```bash
   conda create -n svv-fresh python=3.9
   conda activate svv-fresh
   conda install -c conda-forge vtk pyvista pyvistaqt mesalib libllvm11
   pip install -e .
   ```

4. **Report the issue**:
   Include output from:
   ```bash
   python -c "import vtk; print(vtk.VTK_VERSION)"
   python -c "import pyvista; print(pyvista.__version__)"
   echo $CONDA_PREFIX
   ls $CONDA_PREFIX/lib/dri/
   ls $CONDA_PREFIX/lib/libLLVM*.so
   ```

## Summary

**Quick Fix for Most Cases:**
```bash
conda install -c conda-forge mesalib libllvm11
python -m svv.visualize.gui
```

**If that doesn't work:**
```bash
# Reinstall everything
conda install -c conda-forge --force-reinstall mesalib libllvm11 vtk pyvista pyvistaqt

# Launch with debug
export SVV_GUI_DEBUG_GL=1
python -m svv.visualize.gui
```

**Last resort:**
```bash
# Use the safe launcher
python launch_gui_safe.py --safe
```

This will run the GUI without 3D visualization, allowing you to at least configure parameters.
