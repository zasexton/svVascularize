"""
Unified CAD theme for svVascularize GUI.

This module provides the token-driven CAD theme with SVG icon support.
Inspired by FreeCAD's modern dark theme.
"""

from pathlib import Path
from svv.visualize.gui.theme_generator import ThemeGenerator
from PySide6.QtGui import QIcon, QPixmap, QPainter, QColor
from PySide6.QtCore import Qt, QSize
from PySide6.QtSvg import QSvgRenderer


class CADTheme:
    """
    CAD-style theme for svVascularize.

    Uses design tokens from JSON to generate WCAG AA compliant stylesheets.
    Inspired by FreeCAD's modern interface design.
    """

    _generator = None
    _stylesheet = None

    @classmethod
    def _ensure_generator(cls):
        """Ensure theme generator is initialized."""
        if cls._generator is None:
            token_file = Path(__file__).parent / 'design_tokens.json'
            cls._generator = ThemeGenerator(token_file)

    @classmethod
    def get_stylesheet(cls) -> str:
        """
        Get the complete QSS stylesheet.

        Returns:
            Complete QSS stylesheet string
        """
        cls._ensure_generator()

        if cls._stylesheet is None:
            cls._stylesheet = cls._generator.generate_qss()

        return cls._stylesheet

    @classmethod
    def reload_stylesheet(cls) -> str:
        """
        Force reload the stylesheet from tokens.

        Returns:
            Complete QSS stylesheet string
        """
        cls._generator = None
        cls._stylesheet = None
        return cls.get_stylesheet()

    @classmethod
    def get_color(cls, *keys: str) -> str:
        """
        Get a color value from the design tokens.

        Args:
            *keys: Nested keys to access color (e.g., 'text', 'primary')

        Returns:
            Hex color string

        Examples:
            >>> CADTheme.get_color('text', 'primary')
            '#CCCCCC'
            >>> CADTheme.get_color('action', 'primary')
            '#0078D4'
        """
        cls._ensure_generator()

        value = cls._generator.colors
        for key in keys:
            value = value[key]
        return value

    @classmethod
    def get_spacing(cls, size: str) -> str:
        """
        Get a spacing value from design tokens.

        Args:
            size: Size key ('xs', 'sm', 'md', 'lg', 'xl', '2xl', '3xl')

        Returns:
            Spacing value with units (e.g., '8px')
        """
        cls._ensure_generator()
        return cls._generator.spacing[size]

    @classmethod
    def get_size(cls, category: str, key: str) -> str:
        """
        Get a size value from design tokens.

        Args:
            category: Size category ('icon', 'button', 'input', 'toolbar')
            key: Size key within category

        Returns:
            Size value with units (e.g., '16px')
        """
        cls._ensure_generator()
        return cls._generator.size[category][key]

    @classmethod
    def validate_contrast(cls) -> dict:
        """
        Validate WCAG contrast ratios.

        Returns:
            Dictionary of contrast validation results
        """
        cls._ensure_generator()
        return cls._generator.validate_contrast()


class CADIcons:
    """
    SVG Icon provider for CAD-style GUI.

    Loads SVG icons from the icons directory and colorizes them
    based on the current theme.
    """

    # Map logical icon names to SVG file names
    _icon_files = {
        "open": "folder-open.svg",
        "save": "device-floppy.svg",
        "export": "file-export.svg",
        "view": "view-360.svg",
        "fit": "maximize.svg",
        "iso": "box.svg",
        "top": "arrow-up.svg",
        "front": "arrow-right.svg",
        "right": "rotate-clockwise.svg",
        "visible": "eye.svg",
        "hidden": "eye-off.svg",
        "point": "point.svg",
        "vector": "vector.svg",
        "tree": "binary-tree.svg",
        "forest": "trees.svg",
        "generate": "player-play.svg",
        "mesh": "3d-cube-sphere.svg",
        "surface": "artboard.svg",
        "success": "check.svg",
        "warning": "alert-triangle.svg",
        "info": "info-circle.svg",
        "error": "x.svg",
        "settings": "settings.svg",
        "measure": "ruler.svg",
        "ruler": "ruler.svg",
        "grid": "grid.svg",
        "play": "player-play.svg",
        "pause": "player-pause.svg",
        "stop": "player-stop.svg",
        "loop": "repeat.svg",
        "repeat": "repeat.svg",
        "adjustments": "adjustments.svg",
        "statistics": "chart-bar.svg",
        "chart": "chart-bar.svg",
        "slice": "cut.svg",
        "cut": "cut.svg",
        "probe": "crosshair.svg",
        "crosshair": "crosshair.svg",
        "download": "download.svg",
        "filter": "filter.svg",
        "threshold": "filter.svg",
        "movie": "movie.svg",
        "animation": "movie.svg",
        "arrows-vertical": "arrows-vertical.svg",
        "plus": "plus.svg",
        "minus": "minus.svg",
        "eye": "eye.svg",
        "expand": "arrows-diagonal.svg",
    }

    _icon_dir = Path(__file__).parent / "icons"
    _cache: dict[str, dict[str, QIcon]] = {}  # key -> {color -> QIcon}

    @classmethod
    def _colorize_svg(cls, svg_path: Path, color: str, size: int = 20) -> QPixmap:
        """
        Load an SVG and colorize it with the specified color.

        Args:
            svg_path: Path to SVG file
            color: Hex color string
            size: Icon size in pixels

        Returns:
            Colorized QPixmap
        """
        # Read SVG content
        with open(svg_path, 'r') as f:
            svg_content = f.read()

        # Replace currentColor with the specified color
        svg_content = svg_content.replace('currentColor', color)
        svg_content = svg_content.replace('stroke-width="1.5"', 'stroke-width="1.8"')

        # Create renderer from modified SVG
        renderer = QSvgRenderer(svg_content.encode('utf-8'))

        # Create pixmap and render
        pixmap = QPixmap(size, size)
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHints(
            QPainter.Antialiasing |
            QPainter.SmoothPixmapTransform |
            QPainter.TextAntialiasing
        )
        renderer.render(painter)
        painter.end()

        return pixmap

    @classmethod
    def get_icon(cls, key: str, color: str = None, size: int = 20) -> QIcon:
        """
        Get a themed QIcon for the given key.

        Args:
            key: Icon name (e.g., 'open', 'save', 'tree')
            color: Optional hex color override. If None, uses theme text color.
            size: Icon size in pixels

        Returns:
            QIcon instance
        """
        if color is None:
            color = CADTheme.get_color('text', 'primary')

        # Check cache
        cache_key = f"{key}_{color}_{size}"
        if cache_key in cls._cache:
            return cls._cache[cache_key]

        # Get SVG file
        svg_file = cls._icon_files.get(key)
        if svg_file is None:
            # Return empty icon for unknown keys
            return QIcon()

        svg_path = cls._icon_dir / svg_file
        if not svg_path.exists():
            # Return empty icon if file doesn't exist
            return QIcon()

        # Create icon with multiple states
        icon = QIcon()

        # Normal state
        normal_pixmap = cls._colorize_svg(svg_path, color, size)
        icon.addPixmap(normal_pixmap, QIcon.Normal, QIcon.Off)

        # Hover/Active state (slightly brighter)
        hover_color = CADTheme.get_color('text', 'heading')
        hover_pixmap = cls._colorize_svg(svg_path, hover_color, size)
        icon.addPixmap(hover_pixmap, QIcon.Active, QIcon.Off)

        # Disabled state
        disabled_color = CADTheme.get_color('text', 'disabled')
        disabled_pixmap = cls._colorize_svg(svg_path, disabled_color, size)
        icon.addPixmap(disabled_pixmap, QIcon.Disabled, QIcon.Off)

        # Selected/Checked state
        selected_color = CADTheme.get_color('action', 'primary')
        selected_pixmap = cls._colorize_svg(svg_path, selected_color, size)
        icon.addPixmap(selected_pixmap, QIcon.Normal, QIcon.On)
        icon.addPixmap(selected_pixmap, QIcon.Selected, QIcon.Off)

        cls._cache[cache_key] = icon
        return icon

    @classmethod
    def get_colored_icon(cls, key: str, color: str, size: int = 20) -> QIcon:
        """
        Get an icon with a specific color.

        Args:
            key: Icon name
            color: Hex color string
            size: Icon size in pixels

        Returns:
            QIcon instance
        """
        return cls.get_icon(key, color=color, size=size)

    @classmethod
    def get_accent_icon(cls, key: str, size: int = 20) -> QIcon:
        """
        Get an icon in the primary accent color.

        Args:
            key: Icon name
            size: Icon size in pixels

        Returns:
            QIcon instance
        """
        return cls.get_icon(key, color=CADTheme.get_color('action', 'primary'), size=size)

    @classmethod
    def get_success_icon(cls, key: str = "success", size: int = 20) -> QIcon:
        """Get success-colored icon."""
        return cls.get_icon(key, color=CADTheme.get_color('status', 'success'), size=size)

    @classmethod
    def get_warning_icon(cls, key: str = "warning", size: int = 20) -> QIcon:
        """Get warning-colored icon."""
        return cls.get_icon(key, color=CADTheme.get_color('status', 'warning'), size=size)

    @classmethod
    def get_error_icon(cls, key: str = "error", size: int = 20) -> QIcon:
        """Get error-colored icon."""
        return cls.get_icon(key, color=CADTheme.get_color('status', 'error'), size=size)

    @classmethod
    def get_info_icon(cls, key: str = "info", size: int = 20) -> QIcon:
        """Get info-colored icon."""
        return cls.get_icon(key, color=CADTheme.get_color('status', 'info'), size=size)

    @classmethod
    def clear_cache(cls):
        """Clear the icon cache. Call after theme changes."""
        cls._cache.clear()

    @classmethod
    def available_icons(cls) -> list[str]:
        """
        Get list of available icon names.

        Returns:
            List of icon key names
        """
        return list(cls._icon_files.keys())


# Backwards compatibility - will be removed in future version
class ModernTheme:
    """
    DEPRECATED: Modern theme is deprecated.

    Use CADTheme instead. This class will be removed in a future version.
    """

    @staticmethod
    def get_stylesheet() -> str:
        """
        DEPRECATED: Use CADTheme.get_stylesheet() instead.
        """
        import warnings
        warnings.warn(
            "ModernTheme is deprecated and will be removed in a future version. "
            "Use CADTheme instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return CADTheme.get_stylesheet()


def apply_theme(app):
    """
    Apply the CAD theme to a QApplication.

    Args:
        app: QApplication instance

    Example:
        >>> from PySide6.QtWidgets import QApplication
        >>> app = QApplication([])
        >>> apply_theme(app)
    """
    app.setStyleSheet(CADTheme.get_stylesheet())


if __name__ == '__main__':
    # Test the theme
    print("=== CAD Theme Test ===\n")

    # Test color access
    print("Colors:")
    print(f"  Primary Text: {CADTheme.get_color('text', 'primary')}")
    print(f"  Background: {CADTheme.get_color('background', 'primary')}")
    print(f"  Action Primary: {CADTheme.get_color('action', 'primary')}")
    print()

    # Test spacing access
    print("Spacing:")
    print(f"  Small: {CADTheme.get_spacing('sm')}")
    print(f"  Medium: {CADTheme.get_spacing('md')}")
    print(f"  Large: {CADTheme.get_spacing('lg')}")
    print()

    # Test size access
    print("Sizes:")
    print(f"  Icon Medium: {CADTheme.get_size('icon', 'medium')}")
    print(f"  Button Height: {CADTheme.get_size('button', 'height')}")
    print()

    # Test contrast validation
    print("Contrast Validation:")
    results = CADTheme.validate_contrast()
    for name, result in results.items():
        status = "PASS" if result['passes'] else "FAIL"
        print(f"  {status} {name}: {result['ratio']}:1")
    print()

    # Test available icons
    print("Available Icons:")
    print(f"  {', '.join(CADIcons.available_icons())}")
    print()

    # Test stylesheet generation
    stylesheet = CADTheme.get_stylesheet()
    print(f"Stylesheet length: {len(stylesheet)} characters")
    print("Theme test completed successfully!")
