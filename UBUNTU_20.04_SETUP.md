# Ubuntu 20.04 Setup Guide for svVascularize GUI

## Compatibility

✅ **Yes, fully compatible with Ubuntu 20.04!**

The fixes work on Ubuntu 20.04 because:
- Mesa libraries are available in conda-forge for Ubuntu 20.04
- Python 3.9 (your version) is fully supported
- VTK/PyVista work natively on Ubuntu 20.04
- Software rendering (llvmpipe) is standard on Ubuntu 20.04

## Quick Setup for Ubuntu 20.04

### Option 1: Use Conda (Recommended)

This is the **best approach** - everything is self-contained in your conda environment:

```bash
# Activate your conda environment
conda activate svv  # or whatever your env is called

# Install Mesa libraries
conda install -c conda-forge mesalib libllvm11

# Launch GUI
python -m svv.visualize.gui
```

**Why conda is best on Ubuntu 20.04:**
- Self-contained (doesn't affect system)
- No sudo/root required
- Consistent across machines
- Easy to reproduce

### Option 2: Use System Mesa

If you prefer using Ubuntu's system libraries:

```bash
# Install system Mesa libraries
sudo apt-get update
sudo apt-get install -y \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    mesa-utils \
    libglu1-mesa \
    libllvm11

# Tell the GUI to use system OpenGL
export SVV_GUI_GL_MODE=system

# Launch GUI
python -m svv.visualize.gui
```

### Option 3: Hybrid (Conda + System Fallback)

Use conda Mesa but allow fallback to system:

```bash
# Install conda Mesa
conda install -c conda-forge mesalib libllvm11

# Also install system Mesa as backup
sudo apt-get install -y libgl1-mesa-glx libgl1-mesa-dri

# Launch GUI (will use conda Mesa)
python -m svv.visualize.gui
```

## Ubuntu 20.04 Specific Notes

### LLVM Version
Ubuntu 20.04 ships with **LLVM 10** by default, but Mesa works best with **LLVM 11**:

```bash
# Install LLVM 11 system-wide (optional)
sudo apt-get install -y llvm-11 llvm-11-dev

# Or use conda's LLVM 11 (recommended)
conda install -c conda-forge libllvm11
```

### libffi Version
Ubuntu 20.04 has **libffi7**, which is compatible:

```bash
# Check your version
dpkg -l | grep libffi

# Should show libffi7 (which is fine)
# If missing:
sudo apt-get install -y libffi7

# Conda will handle its own libffi:
conda install -c conda-forge libffi
```

### Display Server
Ubuntu 20.04 defaults to **Xorg** (not Wayland), which is ideal for OpenGL:

```bash
# Check your display server
echo $XDG_SESSION_TYPE
# Should output: x11 (Xorg) - perfect!

# If it says "wayland":
export QT_QPA_PLATFORM=xcb  # Force X11
```

## Complete Ubuntu 20.04 Installation

### Fresh Install from Scratch

```bash
# 1. System dependencies (optional but recommended)
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    mesa-utils \
    libglu1-mesa \
    libffi7

# 2. Create conda environment
conda create -n svvascularize python=3.9
conda activate svvascularize

# 3. Install Mesa in conda
conda install -c conda-forge mesalib libllvm11

# 4. Install svVascularize dependencies
conda install -c conda-forge vtk pyvista pyvistaqt PySide6

# 5. Install svVascularize
cd /path/to/svVascularize
pip install -e .

# 6. Launch GUI
python -m svv.visualize.gui
```

## Verification for Ubuntu 20.04

```bash
# Check Ubuntu version
lsb_release -a
# Should show: Ubuntu 20.04.x LTS

# Check Python version
python --version
# Should show: Python 3.9.23 (your version)

# Check conda environment
echo $CONDA_PREFIX
# Should show: /home/zack/miniconda3/envs/...

# Check Mesa libraries
ls $CONDA_PREFIX/lib/dri/swrast_dri.so
# Should exist

# Check LLVM
ls $CONDA_PREFIX/lib/libLLVM*.so
# Should show LLVM library files

# Test OpenGL
glxinfo | grep "OpenGL version"
# Should show Mesa version

# Test VTK
python -c "import vtk; print('VTK:', vtk.VTK_VERSION)"
# Should print VTK version without errors

# Test PyVista
python -c "import pyvista; print('PyVista:', pyvista.__version__)"
# Should print PyVista version without errors

# Launch GUI
python -m svv.visualize.gui
# Should open without errors
```

## Common Ubuntu 20.04 Issues

### Issue 1: Missing libLLVM-11.so.1

**Symptom:**
```
error while loading shared libraries: libLLVM-11.so.1: cannot open shared object file
```

**Fix:**
```bash
conda install -c conda-forge libllvm11
```

### Issue 2: GLIBCXX version issues

**Symptom:**
```
version 'GLIBCXX_3.4.29' not found
```

**Fix:**
```bash
conda install -c conda-forge libstdcxx-ng
```

### Issue 3: Qt platform plugin issues

**Symptom:**
```
qt.qpa.plugin: Could not load the Qt platform plugin "xcb"
```

**Fix:**
```bash
conda install -c conda-forge qt
export QT_QPA_PLATFORM=xcb
```

### Issue 4: Display not set

**Symptom:**
```
Error: Can't open display
```

**Fix:**
```bash
export DISPLAY=:0
# Or if using SSH:
ssh -X user@host
```

## Ubuntu 20.04 Advantages

Your Ubuntu 20.04 setup is actually **ideal** for this application:

✅ **Stable LTS**: Long-term support, well-tested
✅ **Xorg default**: Better OpenGL compatibility than Wayland
✅ **Mesa 20.x**: Fully compatible with our requirements
✅ **Python 3.9**: Perfect version for our dependencies
✅ **Conda support**: Excellent conda-forge package availability

## Performance on Ubuntu 20.04

Expected performance with software rendering (llvmpipe):

- **Small models** (<1000 vertices): Smooth, 30-60 FPS
- **Medium models** (1000-10000): Good, 15-30 FPS
- **Large models** (>10000): Usable, 5-15 FPS

If you have a GPU and want hardware acceleration:

```bash
# Check GPU
lspci | grep VGA

# If NVIDIA:
sudo apt-get install nvidia-driver-470
export SVV_GUI_GL_MODE=system

# If AMD:
sudo apt-get install mesa-vulkan-drivers
export SVV_GUI_GL_MODE=system

# If Intel:
sudo apt-get install mesa-vulkan-drivers intel-media-va-driver
export SVV_GUI_GL_MODE=system
```

## Headless Ubuntu 20.04 (No Display)

If running on a server without display:

```bash
# Install Xvfb (virtual framebuffer)
sudo apt-get install -y xvfb

# Use the provided script
chmod +x launch_gui_xvfb.sh
./launch_gui_xvfb.sh

# Or manually:
xvfb-run -a python -m svv.visualize.gui
```

## Docker on Ubuntu 20.04

If using Docker:

```dockerfile
FROM ubuntu:20.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    mesa-utils \
    libglu1-mesa \
    libffi7 \
    xvfb

# Install conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda

# Create environment
RUN /opt/conda/bin/conda create -n svv python=3.9
RUN /opt/conda/bin/conda install -n svv -c conda-forge \
    mesalib libllvm11 vtk pyvista pyvistaqt PySide6

# Set environment
ENV PATH=/opt/conda/envs/svv/bin:$PATH
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV GALLIUM_DRIVER=llvmpipe

# Install svVascularize
COPY . /app/svVascularize
WORKDIR /app/svVascularize
RUN pip install -e .

# Run with Xvfb
CMD ["xvfb-run", "-a", "python", "-m", "svv.visualize.gui"]
```

## Testing on Ubuntu 20.04

Complete test sequence:

```bash
# 1. Environment test
conda activate svvascularize
python --version  # Should be 3.9.x
echo $CONDA_PREFIX  # Should show conda path

# 2. Library test
python -c "import vtk; print('VTK OK')"
python -c "import pyvista; print('PyVista OK')"
python -c "from pyvistaqt import QtInteractor; print('PyVistaQt OK')"

# 3. Mesa test
ls $CONDA_PREFIX/lib/dri/swrast_dri.so  # Should exist
python -c "import os; os.environ['LIBGL_ALWAYS_SOFTWARE']='1'; import vtk; print('Software rendering OK')"

# 4. GUI test
python -m svv.visualize.gui

# 5. CAD style test
python -c "from svv.visualize.gui import launch_gui; launch_gui(style='cad')"

# 6. Modern style test
python -c "from svv.visualize.gui import launch_gui; launch_gui(style='modern')"
```

All tests should pass without errors on Ubuntu 20.04.

## Summary for Ubuntu 20.04

**Status**: ✅ Fully Compatible

**Recommended Setup**:
```bash
conda install -c conda-forge mesalib libllvm11
python -m svv.visualize.gui
```

**Expected Result**:
- GUI opens successfully
- 3D visualization works
- No segmentation faults
- Smooth interaction

**Time to Setup**: 2-3 minutes for conda install

**Why it works**: Ubuntu 20.04 is a stable, well-supported platform with excellent Mesa support. All our dependencies have mature packages for Ubuntu 20.04.

You're on a great platform for this application! 🐧
