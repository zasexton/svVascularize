import os
import sys

# CRITICAL: Set GL environment BEFORE any Qt/VTK imports
# This must happen at module import time
if sys.platform.startswith('linux'):
    gl_mode = os.environ.get('SVV_GUI_GL_MODE', 'software')
    if gl_mode != 'system':
        os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE', '1')
        os.environ.setdefault('GALLIUM_DRIVER', 'llvmpipe')
        os.environ.setdefault('MESA_GL_VERSION_OVERRIDE', '3.3')
        os.environ.setdefault('MESA_LOADER_DRIVER_OVERRIDE', 'llvmpipe')
        os.environ.setdefault('QT_OPENGL', 'software')

        # Set DRI drivers path
        override_dri = os.environ.get('SVV_LIBGL_DRIVERS_PATH')
        if override_dri and os.path.isdir(override_dri):
            os.environ.setdefault('LIBGL_DRIVERS_PATH', override_dri)
        else:
            conda_prefix = os.environ.get('CONDA_PREFIX', '')
            if conda_prefix:
                # Try environment-specific paths first
                candidates = [
                    os.path.join(conda_prefix, 'lib', 'dri'),
                    os.path.join(conda_prefix, 'x86_64-conda-linux-gnu', 'sysroot', 'usr', 'lib64', 'dri'),
                ]

                # Also check base conda path (for cos7 packages installed at root)
                # Extract base path from environment path
                if '/envs/' in conda_prefix:
                    base_prefix = conda_prefix.split('/envs/')[0]
                    candidates.append(os.path.join(base_prefix, 'x86_64-conda-linux-gnu', 'sysroot', 'usr', 'lib64', 'dri'))

                for dri_path in candidates:
                    if os.path.isdir(dri_path):
                        os.environ.setdefault('LIBGL_DRIVERS_PATH', dri_path)
                        break

        if 'WAYLAND_DISPLAY' in os.environ:
            os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')

# Import both GUI styles
from svv.visualize.gui.main_window import VascularizeGUI  # Original modern style
from svv.visualize.gui.main_window_cad import VascularizeCADGUI  # CAD style (default)


def launch_gui(domain=None, block=True, style='cad'):
    """
    Launch the Vascularize GUI from a Python interpreter or script.

    Parameters
    ----------
    domain : optional
        Optional domain object to preload into the GUI.
    block : bool
        If True, start the Qt event loop and block until exit.
        If False, return ``(app, gui)`` without starting the loop.
    style : str
        GUI style: 'cad' for CAD/engineering interface (default),
        'modern' for consumer/mobile-style interface.

    Returns
    -------
    tuple | None
        ``(app, gui)`` when ``block=False``, otherwise ``None``.
    """
    import sys
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt

    # On Linux, prefer Qt's software OpenGL to avoid GPU/driver issues
    # unless explicitly opting into system GL.
    if sys.platform.startswith('linux') and os.environ.get('SVV_GUI_GL_MODE', 'software') != 'system':
        try:
            QApplication.setAttribute(Qt.AA_UseSoftwareOpenGL)
        except Exception:
            pass

    app = QApplication.instance() or QApplication(sys.argv)

    # Choose GUI style
    if style == 'cad':
        gui = VascularizeCADGUI(domain=domain)
    else:
        import warnings
        warnings.warn(
            "The 'modern' GUI style is deprecated and will be removed in a future version. "
            "Please use 'cad' style (the default) instead.",
            DeprecationWarning,
            stacklevel=2
        )
        gui = VascularizeGUI(domain=domain)

    gui.show()

    if not block:
        return app, gui

    try:
        app.exec()
    except AttributeError:
        app.exec_()
    return None


__all__ = ['VascularizeGUI', 'VascularizeCADGUI', 'launch_gui']
