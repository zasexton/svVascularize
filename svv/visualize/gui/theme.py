"""
Unified CAD theme for svVascularize GUI.

This module provides the token-driven CAD theme and replaces the old
cad_styles.py and styles.py modules.
"""

from pathlib import Path
from svv.visualize.gui.theme_generator import ThemeGenerator


class CADTheme:
    """
    CAD-style theme for svVascularize.

    Uses design tokens from JSON to generate WCAG AA compliant stylesheets.
    """

    _generator = None
    _stylesheet = None

    @classmethod
    def _ensure_generator(cls):
        """Ensure theme generator is initialized."""
        if cls._generator is None:
            token_file = Path(__file__).parent / 'design_tokens_cad.json'
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
    def get_color(cls, *keys: str) -> str:
        """
        Get a color value from the design tokens.

        Args:
            *keys: Nested keys to access color (e.g., 'text', 'primary')

        Returns:
            Hex color string

        Examples:
            >>> CADTheme.get_color('text', 'primary')
            '#E0E0E0'
            >>> CADTheme.get_color('action', 'primary')
            '#4A6A8F'
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
    Icon constants for CAD-style GUI.

    Currently uses Unicode emoji. These will be replaced with SVG icons
    in Phase 3 of the GUI improvements.
    """

    # File operations
    OPEN = "\U0001F4C2"      # 📂 Folder
    FOLDER_OPEN = "\U0001F4C2"  # 📂 Folder (alias)
    SAVE = "\U0001F4BE"      # 💾 Floppy
    EXPORT = "\U0001F4E4"    # 📤 Outbox
    IMPORT = "\U0001F4E5"    # 📥 Inbox

    # View controls
    VIEW_FIT = "\U0001F50D"   # 🔍 Magnifying Glass
    VIEW_ISO = "\U0001F4D0"   # 📐 Triangular Ruler
    VIEW_TOP = "\U00002B06"   # ⬆ Up Arrow
    VIEW_FRONT = "\U000027A1"  # ➡ Right Arrow
    VIEW_RIGHT = "\U000021AA"  # ↪ Right Arrow Curved
    VISIBLE = "\U0001F441"    # 👁 Eye
    EYE = "\U0001F441"        # 👁 Eye (alias)
    HIDDEN = "\U0001F576"     # 🕶 Sunglasses
    CAMERA = "\U0001F4F7"     # 📷 Camera

    # Generation tools
    POINT = "\U0001F534"      # 🔴 Red Circle
    VECTOR = "\U000027A1"     # ➡ Arrow
    ARROW = "\U000027A1"      # ➡ Arrow (alias)
    TREE = "\U0001F333"       # 🌳 Tree
    FOREST = "\U0001F332"     # 🌲 Evergreen
    GENERATE = "\U000026A1"   # ⚡ Lightning

    # Objects
    MESH = "\U0001F3D7"       # 🏗 Building Construction
    SURFACE = "\U0001F4CB"    # 📋 Clipboard

    # Status
    SUCCESS = "\U00002705"    # ✅ Check Mark
    CHECK = "\U00002705"      # ✅ Check Mark (alias)
    ERROR = "\U0000274C"      # ❌ Cross Mark
    CROSS = "\U0000274C"      # ❌ Cross Mark (alias)
    WARNING = "\U000026A0"    # ⚠ Warning
    INFO = "\U00002139"       # ℹ Information

    # Tools
    SETTINGS = "\U00002699"   # ⚙ Gear
    MEASURE = "\U0001F4CF"    # 📏 Ruler

    # Actions
    PLAY = "\U000025B6"       # ▶ Play
    PLUS = "\U00002795"       # ➕ Plus
    MINUS = "\U00002796"      # ➖ Minus


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

    # Test contrast validation
    print("Contrast Validation:")
    results = CADTheme.validate_contrast()
    for name, result in results.items():
        status = "✓" if result['passes'] else "✗"
        print(f"  {status} {name}: {result['ratio']}:1")
    print()

    # Test stylesheet generation
    stylesheet = CADTheme.get_stylesheet()
    print(f"Stylesheet length: {len(stylesheet)} characters")
    print("Theme test completed successfully!")
