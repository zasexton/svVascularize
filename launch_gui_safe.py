#!/usr/bin/env python
"""
Safe GUI launcher that handles OpenGL issues gracefully.
Uses VTK's null rendering when hardware acceleration fails.
"""

import os
import sys
import warnings

# Add current directory to path
sys.path.insert(0, os.getcwd())

def configure_safe_rendering():
    """Configure the safest possible rendering setup."""

    # Set environment for software rendering
    os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
    os.environ['GALLIUM_DRIVER'] = 'llvmpipe'
    os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'

    # Try to use conda's Mesa if available
    conda_prefix = os.environ.get('CONDA_PREFIX', '')
    if conda_prefix:
        lib_path = os.path.join(conda_prefix, 'lib')
        dri_path = os.path.join(lib_path, 'dri')
        if os.path.exists(dri_path):
            os.environ['LIBGL_DRIVERS_PATH'] = dri_path
            # Prepend conda lib to LD_LIBRARY_PATH
            ld_path = os.environ.get('LD_LIBRARY_PATH', '')
            os.environ['LD_LIBRARY_PATH'] = f"{lib_path}:{ld_path}" if ld_path else lib_path

    # Configure VTK for software rendering
    os.environ['VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN'] = '1'

    # Suppress VTK warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='vtk')

def create_minimal_gui():
    """Create a minimal GUI that works even without OpenGL."""

    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout,
        QLabel, QPushButton, QTextEdit, QMessageBox
    )
    from PySide6.QtCore import Qt

    class MinimalGUI(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("svVascularize - Safe Mode")
            self.setGeometry(100, 100, 800, 600)

            # Central widget
            central = QWidget()
            self.setCentralWidget(central)
            layout = QVBoxLayout(central)

            # Info label
            info = QLabel(
                "Running in Safe Mode due to OpenGL issues.\n"
                "3D visualization is limited, but core functionality is available."
            )
            info.setAlignment(Qt.AlignCenter)
            layout.addWidget(info)

            # Text area for logs
            self.log_area = QTextEdit()
            self.log_area.setReadOnly(True)
            layout.addWidget(self.log_area)

            # Buttons
            load_btn = QPushButton("Load Domain")
            load_btn.clicked.connect(self.load_domain)
            layout.addWidget(load_btn)

            self.log("Safe Mode GUI initialized")
            self.log("To fix OpenGL issues, run: ./setup_mesa_conda.sh")

        def log(self, message):
            self.log_area.append(message)

        def load_domain(self):
            self.log("Load domain functionality would go here...")
            QMessageBox.information(
                self,
                "Info",
                "In safe mode, 3D visualization is limited.\n"
                "Please fix OpenGL issues for full functionality."
            )

    return MinimalGUI

def try_full_gui():
    """Try to load the full GUI with 3D visualization."""

    try:
        # Import VTK early to test if it works
        import vtk

        # Test VTK rendering
        render_window = vtk.vtkRenderWindow()
        render_window.SetOffScreenRendering(1)
        render_window.Render()
        render_window.Finalize()

        # If we get here, VTK works
        from svv.visualize.gui import VascularizeGUI
        return VascularizeGUI

    except Exception as e:
        print(f"Cannot use full GUI due to: {e}")
        print("Falling back to safe mode...")
        return None

def main():
    """Main entry point."""

    import argparse
    parser = argparse.ArgumentParser(description='Launch svVascularize GUI safely')
    parser.add_argument('--safe', action='store_true', help='Force safe mode (no 3D)')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    args = parser.parse_args()

    if args.debug:
        os.environ['SVV_GUI_DEBUG_GL'] = '1'
        print("Debug mode enabled")
        print(f"Python: {sys.version}")
        print(f"DISPLAY: {os.environ.get('DISPLAY', 'Not set')}")
        print(f"CONDA_PREFIX: {os.environ.get('CONDA_PREFIX', 'Not set')}")

    # Configure rendering
    configure_safe_rendering()

    # Import Qt
    try:
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import Qt
    except ImportError:
        print("PySide6 not found. Please install:")
        print("  conda install -c conda-forge pyside6")
        sys.exit(1)

    # Try to use software OpenGL
    try:
        QApplication.setAttribute(Qt.AA_UseSoftwareOpenGL)
    except:
        pass

    # Create application
    app = QApplication(sys.argv)

    # Determine which GUI to use
    if args.safe:
        print("Using safe mode (no 3D visualization)")
        GuiClass = create_minimal_gui()
    else:
        # Try full GUI first
        GuiClass = try_full_gui()
        if GuiClass is None:
            print("Using safe mode GUI")
            GuiClass = create_minimal_gui()

    # Create and show GUI
    try:
        if callable(GuiClass):
            gui = GuiClass()
        else:
            gui = GuiClass  # Already instantiated

        gui.show()

        # Run event loop
        try:
            sys.exit(app.exec())
        except AttributeError:
            sys.exit(app.exec_())

    except Exception as e:
        print(f"Failed to launch GUI: {e}")

        if args.debug:
            import traceback
            traceback.print_exc()

        print("\n" + "=" * 50)
        print("Troubleshooting:")
        print("1. Run setup: ./setup_mesa_conda.sh")
        print("2. Use conda launcher: ./launch_gui_conda.sh")
        print("3. Force safe mode: python launch_gui_safe.py --safe")
        print("4. Use virtual display: ./launch_gui_xvfb.sh")
        print("=" * 50)

        sys.exit(1)

if __name__ == '__main__':
    main()