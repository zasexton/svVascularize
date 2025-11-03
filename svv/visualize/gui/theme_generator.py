"""
Theme generator for svVascularize CAD GUI.

This module loads design tokens from JSON and generates QSS stylesheets
with WCAG AA accessibility compliance.
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple
import colorsys


class ThemeGenerator:
    """Generates QSS stylesheets from design token JSON files."""

    def __init__(self, token_file: Path):
        """
        Initialize theme generator with token file.

        Args:
            token_file: Path to design tokens JSON file
        """
        self.token_file = Path(token_file)
        with open(self.token_file, 'r') as f:
            self.tokens: Dict[str, Any] = json.load(f)

        # Shortcuts for easier access
        self.colors = self.tokens['color']
        self.spacing = self.tokens['spacing']
        self.typography = self.tokens['typography']
        self.border_radius = self.tokens['borderRadius']
        self.elevation = self.tokens['elevation']
        self.size = self.tokens['size']

    def generate_qss(self) -> str:
        """
        Generate complete QSS stylesheet from tokens.

        Returns:
            Complete QSS stylesheet as string
        """
        return f"""
/* ============================================================================
   svVascularize CAD Theme
   Auto-generated from {self.token_file.name} - DO NOT EDIT MANUALLY
   Version: {self.tokens.get('version', '1.0.0')}
   ============================================================================ */

/* === Global Styles === */
QMainWindow {{
    background-color: {self.colors['background']['primary']};
    color: {self.colors['text']['primary']};
}}

QWidget {{
    background-color: {self.colors['background']['primary']};
    color: {self.colors['text']['primary']};
    font-family: {self.typography['family']['primary']};
    font-size: {self.typography['size']['body']};
}}

/* === QPushButton === */
QPushButton {{
    background-color: {self.colors['action']['primary']};
    color: {self.colors['action']['primary-text']};
    border: 1px solid {self.colors['border']['subtle']};
    border-radius: {self.border_radius['sm']};
    padding: {self.spacing['md']} {self.spacing['lg']};
    font-size: {self.typography['size']['body']};
    min-width: {self.size['button']['min-width']};
    min-height: {self.size['button']['height']};
}}

QPushButton:hover {{
    background-color: {self.colors['action']['primary-hover']};
    border-color: {self.colors['border']['focus']};
}}

QPushButton:pressed {{
    background-color: {self.colors['action']['primary-pressed']};
}}

QPushButton:focus {{
    outline: 2px solid {self.colors['border']['focus']};
    outline-offset: 1px;
}}

QPushButton:disabled {{
    background-color: {self.colors['background']['secondary']};
    color: {self.colors['text']['disabled']};
    border-color: {self.colors['border']['subtle']};
}}

QPushButton[danger="true"] {{
    background-color: {self.colors['action']['danger']};
}}

QPushButton[danger="true"]:hover {{
    background-color: {self.colors['action']['danger-hover']};
}}

QPushButton[success="true"] {{
    background-color: {self.colors['action']['success']};
}}

QPushButton[success="true"]:hover {{
    background-color: {self.colors['action']['success-hover']};
}}

/* === QLineEdit, QSpinBox, QDoubleSpinBox === */
QLineEdit, QSpinBox, QDoubleSpinBox {{
    background-color: {self.colors['background']['secondary']};
    color: {self.colors['text']['primary']};
    border: 1px solid {self.colors['border']['subtle']};
    border-radius: {self.border_radius['sm']};
    padding: {self.spacing['sm']} {self.spacing['md']};
    min-height: {self.size['input']['height']};
    selection-background-color: {self.colors['action']['primary']};
}}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border: 2px solid {self.colors['border']['focus']};
    padding: {self.spacing['sm']} {self.spacing['md']};
}}

QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
    background-color: {self.colors['background']['primary']};
    color: {self.colors['text']['disabled']};
}}

/* === QComboBox === */
QComboBox {{
    background-color: {self.colors['background']['secondary']};
    color: {self.colors['text']['primary']};
    border: 1px solid {self.colors['border']['subtle']};
    border-radius: {self.border_radius['sm']};
    padding: {self.spacing['sm']} {self.spacing['md']};
    min-height: {self.size['input']['height']};
}}

QComboBox:focus {{
    border: 2px solid {self.colors['border']['focus']};
}}

QComboBox::drop-down {{
    border: none;
    width: 20px;
}}

QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 6px solid {self.colors['text']['primary']};
    margin-right: {self.spacing['sm']};
}}

QComboBox QAbstractItemView {{
    background-color: {self.colors['background']['secondary']};
    color: {self.colors['text']['primary']};
    border: 1px solid {self.colors['border']['subtle']};
    selection-background-color: {self.colors['action']['primary']};
    selection-color: {self.colors['text']['primary']};
}}

/* === QLabel === */
QLabel {{
    background-color: transparent;
    color: {self.colors['text']['primary']};
    border: none;
}}

QLabel[secondary="true"] {{
    color: {self.colors['text']['secondary']};
}}

QLabel[heading="true"] {{
    font-size: {self.typography['size']['heading']};
    font-weight: {self.typography['weight']['bold']};
}}

/* === QGroupBox === */
QGroupBox {{
    background-color: {self.colors['background']['surface']};
    border: 1px solid {self.colors['border']['subtle']};
    border-radius: {self.border_radius['md']};
    margin-top: {self.spacing['xl']};
    padding: {self.spacing['lg']};
    font-weight: {self.typography['weight']['medium']};
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: {self.spacing['lg']};
    padding: 0 {self.spacing['sm']};
    color: {self.colors['text']['primary']};
    background-color: {self.colors['background']['surface']};
}}

/* === QTreeWidget === */
QTreeWidget {{
    background-color: {self.colors['background']['secondary']};
    color: {self.colors['text']['primary']};
    border: 1px solid {self.colors['border']['subtle']};
    border-radius: {self.border_radius['sm']};
    outline: none;
    show-decoration-selected: 1;
}}

QTreeWidget::item {{
    padding: {self.spacing['sm']} {self.spacing['md']};
    border: none;
}}

QTreeWidget::item:selected {{
    background-color: {self.colors['action']['primary']};
    color: {self.colors['text']['primary']};
}}

QTreeWidget::item:hover {{
    background-color: {self.colors['background']['tertiary']};
}}

QTreeWidget::branch {{
    background-color: {self.colors['background']['secondary']};
}}

QTreeWidget::branch:has-children:!has-siblings:closed,
QTreeWidget::branch:closed:has-children:has-siblings {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 6px solid {self.colors['text']['secondary']};
}}

QTreeWidget::branch:open:has-children:!has-siblings,
QTreeWidget::branch:open:has-children:has-siblings {{
    image: none;
    border-left: 6px solid transparent;
    border-right: 6px solid transparent;
    border-bottom: 4px solid {self.colors['text']['secondary']};
}}

/* === QListWidget === */
QListWidget {{
    background-color: {self.colors['background']['secondary']};
    color: {self.colors['text']['primary']};
    border: 1px solid {self.colors['border']['subtle']};
    border-radius: {self.border_radius['sm']};
    outline: none;
}}

QListWidget::item {{
    padding: {self.spacing['md']};
    border: none;
}}

QListWidget::item:selected {{
    background-color: {self.colors['action']['primary']};
    color: {self.colors['text']['primary']};
}}

QListWidget::item:hover {{
    background-color: {self.colors['background']['tertiary']};
}}

/* === QTabWidget === */
QTabWidget::pane {{
    background-color: {self.colors['background']['surface']};
    border: 1px solid {self.colors['border']['subtle']};
    border-radius: {self.border_radius['sm']};
    top: -1px;
}}

QTabBar::tab {{
    background-color: {self.colors['background']['secondary']};
    color: {self.colors['text']['secondary']};
    border: 1px solid {self.colors['border']['subtle']};
    border-bottom: none;
    border-top-left-radius: {self.border_radius['sm']};
    border-top-right-radius: {self.border_radius['sm']};
    padding: {self.spacing['md']} {self.spacing['lg']};
    margin-right: {self.spacing['xs']};
}}

QTabBar::tab:selected {{
    background-color: {self.colors['background']['surface']};
    color: {self.colors['text']['primary']};
    font-weight: {self.typography['weight']['medium']};
}}

QTabBar::tab:hover:!selected {{
    background-color: {self.colors['background']['tertiary']};
}}

/* === QToolBar === */
QToolBar {{
    background-color: {self.colors['background']['secondary']};
    border: none;
    border-bottom: 1px solid {self.colors['border']['subtle']};
    spacing: {self.spacing['sm']};
    padding: {self.spacing['sm']};
}}

QToolBar::separator {{
    background-color: {self.colors['border']['divider']};
    width: 1px;
    margin: {self.spacing['sm']} {self.spacing['md']};
}}

QToolButton {{
    background-color: transparent;
    color: {self.colors['text']['primary']};
    border: 1px solid transparent;
    border-radius: {self.border_radius['sm']};
    padding: {self.spacing['sm']};
}}

QToolButton:hover {{
    background-color: {self.colors['background']['tertiary']};
    border-color: {self.colors['border']['subtle']};
}}

QToolButton:pressed {{
    background-color: {self.colors['background']['primary']};
}}

QToolButton:checked {{
    background-color: {self.colors['action']['primary']};
    border-color: {self.colors['border']['focus']};
}}

/* === QMenuBar === */
QMenuBar {{
    background-color: {self.colors['background']['secondary']};
    color: {self.colors['text']['primary']};
    border-bottom: 1px solid {self.colors['border']['subtle']};
    padding: {self.spacing['xs']};
}}

QMenuBar::item {{
    background-color: transparent;
    padding: {self.spacing['sm']} {self.spacing['lg']};
    border-radius: {self.border_radius['sm']};
}}

QMenuBar::item:selected {{
    background-color: {self.colors['background']['tertiary']};
}}

QMenuBar::item:pressed {{
    background-color: {self.colors['action']['primary']};
}}

/* === QMenu === */
QMenu {{
    background-color: {self.colors['background']['secondary']};
    color: {self.colors['text']['primary']};
    border: 1px solid {self.colors['border']['subtle']};
    border-radius: {self.border_radius['sm']};
    padding: {self.spacing['sm']};
}}

QMenu::item {{
    padding: {self.spacing['md']} {self.spacing['2xl']};
    border-radius: {self.border_radius['sm']};
}}

QMenu::item:selected {{
    background-color: {self.colors['action']['primary']};
}}

QMenu::separator {{
    height: 1px;
    background-color: {self.colors['border']['divider']};
    margin: {self.spacing['sm']} {self.spacing['md']};
}}

/* === QStatusBar === */
QStatusBar {{
    background-color: {self.colors['background']['secondary']};
    color: {self.colors['text']['secondary']};
    border-top: 1px solid {self.colors['border']['subtle']};
}}

QStatusBar::item {{
    border: none;
}}

QStatusBar QLabel {{
    padding: {self.spacing['xs']} {self.spacing['md']};
}}

/* === QDockWidget === */
QDockWidget {{
    color: {self.colors['text']['primary']};
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
}}

QDockWidget::title {{
    background-color: {self.colors['background']['secondary']};
    border: 1px solid {self.colors['border']['subtle']};
    border-bottom: none;
    padding: {self.spacing['md']};
    font-weight: {self.typography['weight']['medium']};
}}

QDockWidget::close-button, QDockWidget::float-button {{
    background-color: transparent;
    border: none;
    padding: {self.spacing['xs']};
}}

QDockWidget::close-button:hover, QDockWidget::float-button:hover {{
    background-color: {self.colors['background']['tertiary']};
}}

/* === QScrollBar === */
QScrollBar:vertical {{
    background-color: {self.colors['background']['primary']};
    width: 12px;
    border: none;
}}

QScrollBar::handle:vertical {{
    background-color: {self.colors['border']['subtle']};
    border-radius: {self.border_radius['sm']};
    min-height: 20px;
    margin: 2px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: {self.colors['text']['disabled']};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}

QScrollBar:horizontal {{
    background-color: {self.colors['background']['primary']};
    height: 12px;
    border: none;
}}

QScrollBar::handle:horizontal {{
    background-color: {self.colors['border']['subtle']};
    border-radius: {self.border_radius['sm']};
    min-width: 20px;
    margin: 2px;
}}

QScrollBar::handle:horizontal:hover {{
    background-color: {self.colors['text']['disabled']};
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0px;
}}

/* === QProgressBar === */
QProgressBar {{
    background-color: {self.colors['background']['secondary']};
    border: 1px solid {self.colors['border']['subtle']};
    border-radius: {self.border_radius['sm']};
    text-align: center;
    color: {self.colors['text']['primary']};
}}

QProgressBar::chunk {{
    background-color: {self.colors['action']['primary']};
    border-radius: {self.border_radius['sm']};
}}

/* === QCheckBox === */
QCheckBox {{
    color: {self.colors['text']['primary']};
    spacing: {self.spacing['md']};
}}

QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {self.colors['border']['subtle']};
    border-radius: {self.border_radius['sm']};
    background-color: {self.colors['background']['secondary']};
}}

QCheckBox::indicator:hover {{
    border-color: {self.colors['border']['focus']};
}}

QCheckBox::indicator:checked {{
    background-color: {self.colors['action']['primary']};
    border-color: {self.colors['action']['primary']};
}}

QCheckBox:focus {{
    outline: 2px solid {self.colors['border']['focus']};
    outline-offset: 2px;
}}

/* === QRadioButton === */
QRadioButton {{
    color: {self.colors['text']['primary']};
    spacing: {self.spacing['md']};
}}

QRadioButton::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {self.colors['border']['subtle']};
    border-radius: 8px;
    background-color: {self.colors['background']['secondary']};
}}

QRadioButton::indicator:hover {{
    border-color: {self.colors['border']['focus']};
}}

QRadioButton::indicator:checked {{
    background-color: {self.colors['action']['primary']};
    border: 4px solid {self.colors['background']['secondary']};
}}

/* === QSlider === */
QSlider::groove:horizontal {{
    background-color: {self.colors['background']['secondary']};
    height: 4px;
    border-radius: 2px;
}}

QSlider::handle:horizontal {{
    background-color: {self.colors['action']['primary']};
    border: 1px solid {self.colors['border']['subtle']};
    width: 14px;
    margin: -6px 0;
    border-radius: 7px;
}}

QSlider::handle:horizontal:hover {{
    background-color: {self.colors['action']['primary-hover']};
}}

/* === QToolTip === */
QToolTip {{
    background-color: {self.colors['background']['tertiary']};
    color: {self.colors['text']['primary']};
    border: 1px solid {self.colors['border']['subtle']};
    border-radius: {self.border_radius['sm']};
    padding: {self.spacing['sm']} {self.spacing['md']};
}}

/* === QDialog === */
QDialog {{
    background-color: {self.colors['background']['primary']};
}}

/* === Custom Property Classes === */
.success-text {{
    color: {self.colors['status']['success']};
}}

.warning-text {{
    color: {self.colors['status']['warning']};
}}

.error-text {{
    color: {self.colors['status']['error']};
}}

.info-text {{
    color: {self.colors['status']['info']};
}}
"""

    @staticmethod
    def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    @staticmethod
    def relative_luminance(rgb: Tuple[int, int, int]) -> float:
        """
        Calculate relative luminance according to WCAG 2.1.

        Args:
            rgb: RGB color tuple (0-255)

        Returns:
            Relative luminance (0-1)
        """
        r, g, b = [x / 255.0 for x in rgb]

        # Apply gamma correction
        r = r / 12.92 if r <= 0.03928 else ((r + 0.055) / 1.055) ** 2.4
        g = g / 12.92 if g <= 0.03928 else ((g + 0.055) / 1.055) ** 2.4
        b = b / 12.92 if b <= 0.03928 else ((b + 0.055) / 1.055) ** 2.4

        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    def contrast_ratio(self, color1: str, color2: str) -> float:
        """
        Calculate contrast ratio between two colors according to WCAG 2.1.

        Args:
            color1: First color (hex)
            color2: Second color (hex)

        Returns:
            Contrast ratio (1-21)
        """
        rgb1 = self.hex_to_rgb(color1)
        rgb2 = self.hex_to_rgb(color2)

        l1 = self.relative_luminance(rgb1)
        l2 = self.relative_luminance(rgb2)

        lighter = max(l1, l2)
        darker = min(l1, l2)

        return (lighter + 0.05) / (darker + 0.05)

    def validate_contrast(self) -> Dict[str, Dict[str, Any]]:
        """
        Validate contrast ratios for all text/background combinations.

        Returns:
            Dictionary with validation results for each combination
        """
        results = {}

        # Define critical text/background pairs to validate
        pairs = [
            ("Text Primary on Background",
             self.colors['text']['primary'],
             self.colors['background']['primary'],
             4.5),
            ("Text Secondary on Background",
             self.colors['text']['secondary'],
             self.colors['background']['primary'],
             4.5),
            ("Text Primary on Surface",
             self.colors['text']['primary'],
             self.colors['background']['surface'],
             4.5),
            ("Action Primary Text on Action Primary",
             self.colors['action']['primary-text'],
             self.colors['action']['primary'],
             4.5),
            ("Border Subtle on Background",
             self.colors['border']['subtle'],
             self.colors['background']['primary'],
             3.0),  # Lower requirement for UI components
            ("Text Disabled on Background",
             self.colors['text']['disabled'],
             self.colors['background']['primary'],
             4.5),
        ]

        for name, fg, bg, required_ratio in pairs:
            ratio = self.contrast_ratio(fg, bg)
            passes = ratio >= required_ratio

            results[name] = {
                'foreground': fg,
                'background': bg,
                'ratio': round(ratio, 2),
                'required': required_ratio,
                'passes': passes,
                'wcag_level': 'AA' if ratio >= 4.5 else ('AA Large' if ratio >= 3.0 else 'FAIL')
            }

        return results

    def save_qss(self, output_path: Path) -> None:
        """
        Generate and save QSS stylesheet to file.

        Args:
            output_path: Path where QSS file should be saved
        """
        qss = self.generate_qss()
        with open(output_path, 'w') as f:
            f.write(qss)


if __name__ == '__main__':
    # Example usage
    import sys
    from pathlib import Path

    # Get the directory of this file
    gui_dir = Path(__file__).parent
    token_file = gui_dir / 'design_tokens_cad.json'

    # Generate theme
    generator = ThemeGenerator(token_file)

    # Validate contrast
    print("=== WCAG Contrast Validation ===")
    results = generator.validate_contrast()

    all_pass = True
    for name, result in results.items():
        status = "✓ PASS" if result['passes'] else "✗ FAIL"
        print(f"{status} {name}:")
        print(f"  {result['foreground']} on {result['background']}")
        print(f"  Ratio: {result['ratio']}:1 (Required: {result['required']}:1)")
        print(f"  WCAG Level: {result['wcag_level']}")
        print()

        if not result['passes']:
            all_pass = False

    if all_pass:
        print("✓ All contrast ratios meet WCAG AA standards!")
    else:
        print("✗ Some contrast ratios fail WCAG AA standards.")
        sys.exit(1)

    # Generate QSS
    output_file = gui_dir / 'theme_cad.qss'
    generator.save_qss(output_file)
    print(f"\nGenerated stylesheet: {output_file}")
