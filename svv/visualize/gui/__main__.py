"""
Entry point for launching the svVascularize GUI as a module.

Usage:
    python -m svv.visualize.gui
    python -m svv.visualize.gui --domain path/to/domain.dmn
"""
import sys
import os
import argparse

# CRITICAL: Set GL environment BEFORE any imports
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

from svv.visualize.gui import launch_gui


def main():
    """Main entry point for the GUI."""
    parser = argparse.ArgumentParser(
        description='svVascularize - Interactive Domain Visualization'
    )
    parser.add_argument(
        '--domain',
        type=str,
        help='Path to a .dmn domain file to load on startup',
        default=None
    )

    args = parser.parse_args()

    # Load domain if specified
    domain = None
    if args.domain:
        try:
            from svv.domain.domain import Domain
            print(f"Loading domain from {args.domain}...")
            domain = Domain.load(args.domain)
            print("Domain loaded successfully.")
        except Exception as e:
            print(f"Error loading domain: {e}")
            print("Starting with empty GUI...")

    # Delegate GUI creation and telemetry setup to the shared launcher
    launch_gui(domain=domain, block=True)


if __name__ == '__main__':
    main()
