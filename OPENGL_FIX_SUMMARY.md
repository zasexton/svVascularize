# OpenGL Issue - Quick Fix Guide

## The Problem

You're seeing this error when launching the GUI:

```
libGL error: MESA-LOADER: failed to open swrast: libLLVM-7-rhel.so: cannot open shared object file
Segmentation fault (core dumped)
```

This means Mesa software rendering libraries are missing or incomplete.

## Quick Fix (Try This First)

```bash
# Install Mesa libraries
conda install -c conda-forge mesalib libllvm11

# Launch GUI
python -m svv.visualize.gui
```

## If That Doesn't Work

### Step 1: Install All Mesa Dependencies

```bash
conda install -c conda-forge mesalib libllvm11 mesa-libgl-devel-cos7-x86_64
```

### Step 2: Verify Installation

```bash
# Check DRI drivers exist
ls $CONDA_PREFIX/lib/dri/
# Should see: swrast_dri.so

# Check LLVM libraries exist
ls $CONDA_PREFIX/lib/libLLVM*.so
# Should see libLLVM-*.so files
```

### Step 3: Test GUI

```python
python -m svv.visualize.gui
```

## Alternative: Use Installation Script

```bash
chmod +x install_mesa.sh
./install_mesa.sh
```

## What We Fixed in the Code

1. **Added Error Handling**: GUI now catches OpenGL errors gracefully
2. **Fallback Display**: Shows helpful error message instead of crashing
3. **Safety Checks**: All VTK operations check if plotter initialized
4. **Better Warnings**: Suppressed VTK warnings that aren't helpful

## Current GUI Behavior

**If OpenGL works**: Normal 3D visualization
**If OpenGL fails**:
- Window still opens
- Shows warning message in viewport area
- Message explains the issue and how to fix it
- Can still use all other GUI features

## Testing

After installing Mesa, test with:

```python
# Simple test
python -c "import vtk; print('VTK OK')"
python -c "import pyvista; print('PyVista OK')"

# Launch GUI
from svv.visualize.gui import launch_gui
launch_gui(style='cad')
```

## Common Issues

### Issue: "swrast not found"
**Fix**: `conda install -c conda-forge mesa-libgl-devel-cos7-x86_64`

### Issue: "libLLVM not found"
**Fix**: `conda install -c conda-forge libllvm11`

### Issue: Still crashes
**Fix**: Reinstall everything
```bash
conda remove --force vtk pyvista pyvistaqt
conda install -c conda-forge vtk pyvista pyvistaqt mesalib libllvm11
```

## Documentation

See `OPENGL_TROUBLESHOOTING.md` for complete troubleshooting guide including:
- Detailed error explanations
- Platform-specific fixes
- Advanced debugging
- Environment variables
- Alternative solutions

## Summary

**What happened**: Missing Mesa/LLVM libraries for software OpenGL
**Fix**: Install with `conda install -c conda-forge mesalib libllvm11`
**Result**: GUI launches successfully with 3D visualization

**Time to fix**: ~2-3 minutes for installation

Try the quick fix above and you should be good to go!
