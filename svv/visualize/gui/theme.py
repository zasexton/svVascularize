"""
Unified CAD theme for svVascularize GUI.

This module provides the token-driven CAD theme with SVG icon support.
Inspired by FreeCAD's modern dark theme.
"""

import json
import sys
from importlib import resources as importlib_resources
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
    _warned_missing_assets = False

    _fallback_tokens = {
        "color": {
            "background": {
                "primary": "#1E1E1E",
                "secondary": "#252526",
                "tertiary": "#2D2D2D",
                "surface": "#333333",
                "elevated": "#3C3C3C",
            },
            "text": {
                "primary": "#CCCCCC",
                "secondary": "#8C8C8C",
                "accent": "#569CD6",
                "disabled": "#6B6B6B",
                "inverse": "#1E1E1E",
                "heading": "#FFFFFF",
            },
            "action": {
                "primary": "#0078D4",
                "primary-hover": "#1C8AE6",
                "primary-pressed": "#005A9E",
                "primary-text": "#FFFFFF",
                "secondary": "#3C3C3C",
                "secondary-hover": "#505050",
                "secondary-pressed": "#2D2D2D",
                "danger": "#D32F2F",
                "danger-hover": "#E57373",
                "danger-pressed": "#B71C1C",
                "success": "#388E3C",
                "success-hover": "#4CAF50",
            },
            "status": {
                "success": "#4CAF50",
                "warning": "#FF9800",
                "error": "#F44336",
                "info": "#2196F3",
            },
            "border": {
                "subtle": "#6E6E6E",
                "strong": "#808080",
                "focus": "#0078D4",
                "divider": "#404040",
                "hover": "#909090",
            },
            "viewport": {
                "background-top": "#B8D4E8",
                "background-bottom": "#5A8DB8",
                "grid": "#7BA3C4",
                "selection": "#FF9800",
                "hover": "#0078D4",
                "axis-x": "#F44336",
                "axis-y": "#4CAF50",
                "axis-z": "#2196F3",
                "domain-surface": "#E8C4A2",
                "domain-edge": "#C49A6C",
            },
            "selection": {
                "background": "#094771",
                "text": "#FFFFFF",
                "border": "#0078D4",
            },
        },
        "spacing": {
            "xs": "2px",
            "sm": "4px",
            "md": "6px",
            "lg": "8px",
            "xl": "12px",
            "2xl": "16px",
            "3xl": "24px",
        },
        "size": {
            "icon": {
                "small": "14px",
                "medium": "16px",
                "large": "20px",
                "xlarge": "24px",
            },
            "button": {
                "height": "24px",
                "min-width": "64px",
            },
            "input": {
                "height": "22px",
            },
            "toolbar": {
                "height": "32px",
                "icon-size": "16px",
            },
            "dock-title": {
                "height": "24px",
            },
        },
    }

    @classmethod
    def _ensure_generator(cls):
        """Ensure theme generator is initialized."""
        if cls._generator is not None:
            return

        token_file = Path(__file__).with_name("design_tokens.json")
        try:
            if token_file.is_file():
                cls._generator = ThemeGenerator(token_file)
                return
        except Exception as exc:
            cls._warn_missing_assets(exc)

        # Fall back to package resources (works for zip imports) if available.
        try:
            tokens_text = (
                importlib_resources.files(__package__)
                .joinpath("design_tokens.json")
                .read_text(encoding="utf-8")
            )
            tokens = json.loads(tokens_text)
            cls._generator = ThemeGenerator.from_tokens(tokens, token_name="design_tokens.json")
            return
        except Exception as exc:
            cls._warn_missing_assets(exc)

    @classmethod
    def get_stylesheet(cls) -> str:
        """
        Get the complete QSS stylesheet.

        Returns:
            Complete QSS stylesheet string
        """
        if cls._stylesheet is not None:
            return cls._stylesheet

        cls._ensure_generator()
        if cls._generator is not None:
            try:
                cls._stylesheet = cls._generator.generate_qss()
                return cls._stylesheet
            except Exception as exc:
                cls._warn_missing_assets(exc)

        cls._stylesheet = cls._load_fallback_stylesheet()
        return cls._stylesheet

    @classmethod
    def _load_fallback_stylesheet(cls) -> str:
        """
        Load a conservative fallback stylesheet.

        Prefer the bundled theme.qss when available; otherwise return an empty
        stylesheet so the GUI can still start with Qt defaults.
        """
        qss_file = Path(__file__).with_name("theme.qss")
        try:
            if qss_file.is_file():
                return qss_file.read_text(encoding="utf-8")
        except Exception:
            pass

        try:
            return (
                importlib_resources.files(__package__)
                .joinpath("theme.qss")
                .read_text(encoding="utf-8")
            )
        except Exception:
            return ""

    @classmethod
    def _warn_missing_assets(cls, error: Exception) -> None:
        if cls._warned_missing_assets:
            return
        cls._warned_missing_assets = True

        message = (
            "svVascularize GUI theme resources could not be loaded.\n"
            "The GUI will continue with a fallback theme.\n\n"
            "This can happen if the package was installed without its GUI asset files.\n"
            "Try upgrading/reinstalling the 'svv' package."
        )
        print(f"{message}\n\nDetails: {error}", file=sys.stderr)

        # If running inside a Qt application, also show a one-time warning dialog.
        try:
            from PySide6.QtWidgets import QApplication, QMessageBox

            if QApplication.instance() is None:
                return
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("svVascularize Theme Warning")
            msg.setText(message)
            msg.setDetailedText(str(error))
            msg.exec()
        except Exception:
            pass

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

        if cls._generator is not None:
            try:
                value = cls._generator.colors
                for key in keys:
                    value = value[key]
                return value
            except Exception:
                pass

        value = cls._fallback_tokens.get("color", {})
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                return "#CCCCCC"
            value = value[key]
        return value if isinstance(value, str) else "#CCCCCC"

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
        if cls._generator is not None:
            try:
                return cls._generator.spacing[size]
            except Exception:
                pass
        return cls._fallback_tokens.get("spacing", {}).get(size, "0px")

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
        if cls._generator is not None:
            try:
                return cls._generator.size[category][key]
            except Exception:
                pass
        return cls._fallback_tokens.get("size", {}).get(category, {}).get(key, "0px")

    @classmethod
    def validate_contrast(cls) -> dict:
        """
        Validate WCAG contrast ratios.

        Returns:
            Dictionary of contrast validation results
        """
        cls._ensure_generator()
        if cls._generator is None:
            return {}
        try:
            return cls._generator.validate_contrast()
        except Exception:
            return {}


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
