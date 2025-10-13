#!/bin/bash
# Fix LLVM version mismatch for Mesa drivers

echo "Fixing LLVM version mismatch for Mesa..."
echo "========================================="
echo ""

# The old mesa-dri-drivers-cos7 packages need LLVM 7 specifically
# We need to either:
# 1. Install LLVM 7 to match the drivers, OR
# 2. Install newer Mesa packages that work with LLVM 11+

echo "Option 1: Install LLVM 7 (matches your current Mesa drivers)"
echo "-----------------------------------------------------------"
echo "conda install -c conda-forge llvmdev=7"
echo ""
echo "Option 2: Update to newer Mesa (recommended)"
echo "---------------------------------------------"
echo "conda install -c conda-forge 'mesalib>=21' 'mesa-libgl-cos6-x86_64>=17'"
echo ""

read -p "Choose option (1 or 2, default=2): " choice
choice=${choice:-2}

if [ "$choice" = "1" ]; then
    echo ""
    echo "Installing LLVM 7..."
    conda install -y -c conda-forge llvmdev=7

elif [ "$choice" = "2" ]; then
    echo ""
    echo "Updating Mesa packages..."
    # Remove old mesa packages
    conda remove --force -y \
        mesa-dri-drivers-cos7-x86_64 \
        mesa-libgl-cos7-x86_64 \
        mesa-libgl-devel-cos7-x86_64 \
        mesalib

    # Install newer Mesa
    conda install -y -c conda-forge \
        'mesalib>=21' \
        libllvm11 \
        mesa-libgl-devel-cos6-x86_64
fi

echo ""
echo "Fix applied! Now test the GUI:"
echo "  python -m svv.visualize.gui"
