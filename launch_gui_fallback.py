#!/usr/bin/env python
"""
Fallback GUI launcher with better error handling and rendering options.
"""
import os
import sys
import warnings


def setup_software_rendering():
    """Configure software rendering for Linux systems."""
    if not sys.platform.startswith('linux'):
        return

    # Configure software rendering
    os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE', '1')
    os.environ.setdefault('GALLIUM_DRIVER', 'llvmpipe')
    os.environ.setdefault('MESA_GL_VERSION_OVERRIDE', '3.3')
    os.environ.setdefault('MESA_LOADER_DRIVER_OVERRIDE', 'llvmpipe')
    os.environ.setdefault('QT_OPENGL', 'software')

    # Try to find DRI drivers
    conda_prefix = os.environ.get('CONDA_PREFIX', '')
    candidates = []

    if conda_prefix:
        candidates.extend([
            os.path.join(conda_prefix, 'lib', 'dri'),
            os.path.join(conda_prefix, 'x86_64-conda-linux-gnu', 'sysroot', 'usr', 'lib64', 'dri'),
            os.path.join(conda_prefix, 'x86_64-conda-linux-gnu', 'sysroot', 'usr', 'lib', 'dri'),
        ])

        # Check base conda if in an environment
        if '/envs/' in conda_prefix:
            base_prefix = conda_prefix.split('/envs/')[0]
            candidates.append(
                os.path.join(base_prefix, 'x86_64-conda-linux-gnu', 'sysroot', 'usr', 'lib64', 'dri')
            )

    # Add system locations
    candidates.extend([
        '/usr/lib/x86_64-linux-gnu/dri',
        '/usr/lib64/dri',
        '/usr/lib/dri',
    ])

    # Find first valid DRI directory
    for dri_path in candidates:
        if os.path.isdir(dri_path):
            dri_files = os.listdir(dri_path)
            if any('swrast' in f or 'llvmpipe' in f for f in dri_files):
                os.environ['LIBGL_DRIVERS_PATH'] = dri_path
                print(f"Using Mesa drivers from: {dri_path}")
                break


def try_import_gui():
    """Try to import and configure the GUI with proper error handling."""
    try:
        # Try importing with Qt attributes set
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication

        # Set software OpenGL on Linux
        if sys.platform.startswith('linux'):
            try:
                QApplication.setAttribute(Qt.AA_UseSoftwareOpenGL)
            except Exception:
                pass

        # Import GUI
        from svv.visualize.gui import VascularizeGUI
        return QApplication, VascularizeGUI

    except ImportError as e:
        print(f"Import error: {e}")

        # Try alternative imports
        try:
            # Try with PyQt6 instead
            from PyQt6.QtCore import Qt
            from PyQt6.QtWidgets import QApplication
            if sys.platform.startswith('linux'):
                try:
                    QApplication.setAttribute(Qt.AA_UseSoftwareOpenGL)
                except Exception:
                    pass

            from svv.visualize.gui import VascularizeGUI
            return QApplication, VascularizeGUI
        except ImportError:
            pass

        # Provide helpful error message
        missing = []
        try:
            import PySide6
        except ImportError:
            missing.append('PySide6')

        try:
            import pyvista
        except ImportError:
            missing.append('pyvista')

        try:
            import pyvistaqt
        except ImportError:
            missing.append('pyvistaqt')

        if missing:
            print("\nMissing dependencies detected!")
            print(f"Please install: {', '.join(missing)}")
            print("\nRun: conda install -c conda-forge " + ' '.join(missing))
        else:
            print("\nCould not import GUI module. Check that svv package is properly installed.")

        sys.exit(1)


def main():
    """Main entry point for the GUI launcher."""
    # Parse arguments
    use_system_gl = '--system-gl' in sys.argv
    debug_gl = '--debug-gl' in sys.argv
    use_osmesa = '--osmesa' in sys.argv

    if '--help' in sys.argv:
        print("Usage: python launch_gui_fallback.py [OPTIONS]")
        print("")
        print("Options:")
        print("  --system-gl    Use system OpenGL instead of software rendering")
        print("  --debug-gl     Enable OpenGL debugging output")
        print("  --osmesa       Try OSMesa backend (off-screen rendering)")
        print("  --help         Show this help message")
        sys.exit(0)

    # Configure rendering
    if not use_system_gl:
        setup_software_rendering()

    if debug_gl:
        os.environ['SVV_GUI_DEBUG_GL'] = '1'
        os.environ['MESA_DEBUG'] = '1'
        os.environ['LIBGL_DEBUG'] = 'verbose'

    if use_osmesa:
        # Try to use OSMesa backend
        os.environ['PYVISTA_USE_OSMESA'] = 'true'
        print("Using OSMesa backend for off-screen rendering")

    # Add current directory to Python path
    sys.path.insert(0, os.getcwd())

    # Import and launch GUI
    QApplication, VascularizeGUI = try_import_gui()

    # Create application
    app = QApplication(sys.argv)

    # Create and show GUI
    try:
        gui = VascularizeGUI()
        gui.show()

        # Start event loop
        try:
            sys.exit(app.exec())
        except AttributeError:
            # Fallback for older versions
            sys.exit(app.exec_())

    except Exception as e:
        print(f"\nError launching GUI: {e}")

        # Provide debugging info
        if debug_gl or 'libGL' in str(e) or 'GLX' in str(e):
            print("\nOpenGL Information:")
            print(f"  LIBGL_DRIVERS_PATH: {os.environ.get('LIBGL_DRIVERS_PATH', 'Not set')}")
            print(f"  LIBGL_ALWAYS_SOFTWARE: {os.environ.get('LIBGL_ALWAYS_SOFTWARE', 'Not set')}")
            print(f"  GALLIUM_DRIVER: {os.environ.get('GALLIUM_DRIVER', 'Not set')}")

            # Try to get more GL info
            try:
                import subprocess
                result = subprocess.run(['glxinfo'], capture_output=True, text=True)
                if 'OpenGL' in result.stdout:
                    for line in result.stdout.split('\n'):
                        if 'OpenGL' in line or 'GLX' in line:
                            print(f"  {line.strip()}")
            except Exception:
                pass

        print("\nTroubleshooting:")
        print("1. Install Mesa drivers: ./install_mesa.sh")
        print("2. Try system OpenGL: python launch_gui_fallback.py --system-gl")
        print("3. Try OSMesa backend: python launch_gui_fallback.py --osmesa")
        print("4. Enable debugging: python launch_gui_fallback.py --debug-gl")

        sys.exit(1)


if __name__ == '__main__':
    main()