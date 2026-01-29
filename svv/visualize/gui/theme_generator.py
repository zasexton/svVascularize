"""
Theme generator for svVascularize CAD GUI.

This module loads design tokens from JSON and generates QSS stylesheets
with WCAG AA accessibility compliance and FreeCAD-inspired flat styling.
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple


class ThemeGenerator:
    """Generates QSS stylesheets from design token JSON files."""

    def __init__(self, token_file: Path):
        """
        Initialize theme generator with token file.

        Args:
            token_file: Path to design tokens JSON file
        """
        self.token_file = Path(token_file)
        with open(self.token_file, 'r', encoding='utf-8') as f:
            tokens: Dict[str, Any] = json.load(f)
        self._load_tokens(tokens)

    @classmethod
    def from_tokens(cls, tokens: Dict[str, Any], token_name: str = "design_tokens.json") -> "ThemeGenerator":
        """
        Construct a theme generator from an in-memory token dict.

        This is useful when loading tokens from package resources where a real
        filesystem path may not exist (e.g., zipped imports).

        Args:
            tokens: Design token dictionary
            token_name: Human-readable name used in generated QSS headers
        """
        generator = cls.__new__(cls)
        generator.token_file = Path(token_name)
        generator._load_tokens(tokens)
        return generator

    def _load_tokens(self, tokens: Dict[str, Any]) -> None:
        self.tokens = tokens

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
   svVascularize CAD Theme - FreeCAD Inspired
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
    outline: none;
}}

/* === QPushButton - Flat CAD Style === */
QPushButton {{
    background-color: {self.colors['action']['secondary']};
    color: {self.colors['text']['primary']};
    border: 1px solid {self.colors['border']['subtle']};
    border-radius: {self.border_radius['xs']};
    padding: {self.spacing['sm']} {self.spacing['xl']};
    font-size: {self.typography['size']['body']};
    font-weight: {self.typography['weight']['normal']};
    min-width: {self.size['button']['min-width']};
    min-height: {self.size['button']['height']};
}}

QPushButton:hover {{
    background-color: {self.colors['action']['secondary-hover']};
    border-color: {self.colors['border']['hover']};
}}

QPushButton:pressed {{
    background-color: {self.colors['action']['secondary-pressed']};
}}

QPushButton:focus {{
    border: 1px solid {self.colors['border']['focus']};
}}

QPushButton:disabled {{
    background-color: {self.colors['background']['tertiary']};
    color: {self.colors['text']['disabled']};
    border-color: {self.colors['border']['divider']};
}}

/* Primary action buttons - accent color */
QPushButton#primaryButton {{
    background-color: {self.colors['action']['primary']};
    color: {self.colors['action']['primary-text']};
    border: none;
    font-weight: {self.typography['weight']['medium']};
}}

QPushButton#primaryButton:hover {{
    background-color: {self.colors['action']['primary-hover']};
}}

QPushButton#primaryButton:pressed {{
    background-color: {self.colors['action']['primary-pressed']};
}}

/* Secondary buttons - subtle */
QPushButton#secondaryButton {{
    background-color: transparent;
    color: {self.colors['text']['primary']};
    border: 1px solid {self.colors['border']['subtle']};
}}

QPushButton#secondaryButton:hover {{
    background-color: {self.colors['background']['tertiary']};
    border-color: {self.colors['border']['hover']};
}}

QPushButton#secondaryButton:pressed {{
    background-color: {self.colors['background']['secondary']};
}}

/* Danger buttons */
QPushButton[danger="true"] {{
    background-color: {self.colors['action']['danger']};
    color: {self.colors['action']['primary-text']};
    border: none;
}}

QPushButton[danger="true"]:hover {{
    background-color: {self.colors['action']['danger-hover']};
}}

QPushButton[danger="true"]:pressed {{
    background-color: {self.colors['action']['danger-pressed']};
}}

/* Success buttons */
QPushButton[success="true"] {{
    background-color: {self.colors['action']['success']};
    color: {self.colors['action']['primary-text']};
    border: none;
}}

QPushButton[success="true"]:hover {{
    background-color: {self.colors['action']['success-hover']};
}}

/* === QLineEdit, QSpinBox, QDoubleSpinBox - Compact Inputs === */
QLineEdit, QSpinBox, QDoubleSpinBox {{
    background-color: {self.colors['background']['tertiary']};
    color: {self.colors['text']['primary']};
    border: 1px solid {self.colors['border']['subtle']};
    border-radius: {self.border_radius['xs']};
    padding: {self.spacing['xs']} {self.spacing['md']};
    min-height: {self.size['input']['height']};
    selection-background-color: {self.colors['selection']['background']};
    selection-color: {self.colors['selection']['text']};
}}

QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
    border-color: {self.colors['border']['hover']};
}}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {self.colors['border']['focus']};
    background-color: {self.colors['background']['secondary']};
}}

QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
    background-color: {self.colors['background']['secondary']};
    color: {self.colors['text']['disabled']};
    border-color: {self.colors['border']['divider']};
}}

QSpinBox::up-button, QDoubleSpinBox::up-button {{
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 16px;
    border: none;
    background-color: transparent;
}}

QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 16px;
    border: none;
    background-color: transparent;
}}

QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-bottom: 5px solid {self.colors['text']['secondary']};
    width: 0;
    height: 0;
}}

QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {self.colors['text']['secondary']};
    width: 0;
    height: 0;
}}

QSpinBox::up-arrow:hover, QDoubleSpinBox::up-arrow:hover,
QSpinBox::down-arrow:hover, QDoubleSpinBox::down-arrow:hover {{
    border-bottom-color: {self.colors['text']['primary']};
    border-top-color: {self.colors['text']['primary']};
}}

/* === QComboBox - Flat Dropdown === */
QComboBox {{
    background-color: {self.colors['background']['tertiary']};
    color: {self.colors['text']['primary']};
    border: 1px solid {self.colors['border']['subtle']};
    border-radius: {self.border_radius['xs']};
    padding: {self.spacing['xs']} {self.spacing['md']};
    padding-right: 20px;
    min-height: {self.size['input']['height']};
}}

QComboBox:hover {{
    border-color: {self.colors['border']['hover']};
}}

QComboBox:focus {{
    border-color: {self.colors['border']['focus']};
}}

QComboBox::drop-down {{
    border: none;
    width: 18px;
    subcontrol-position: right center;
}}

QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {self.colors['text']['secondary']};
    margin-right: {self.spacing['sm']};
}}

QComboBox::down-arrow:hover {{
    border-top-color: {self.colors['text']['primary']};
}}

QComboBox QAbstractItemView {{
    background-color: {self.colors['background']['secondary']};
    color: {self.colors['text']['primary']};
    border: 1px solid {self.colors['border']['subtle']};
    selection-background-color: {self.colors['selection']['background']};
    selection-color: {self.colors['selection']['text']};
    outline: none;
    padding: {self.spacing['xs']};
}}

QComboBox QAbstractItemView::item {{
    padding: {self.spacing['sm']} {self.spacing['md']};
    min-height: 20px;
}}

QComboBox QAbstractItemView::item:hover {{
    background-color: {self.colors['background']['tertiary']};
}}

/* === QLabel === */
QLabel {{
    background-color: transparent;
    color: {self.colors['text']['primary']};
    border: none;
    padding: 0;
}}

QLabel[secondary="true"] {{
    color: {self.colors['text']['secondary']};
}}

QLabel[heading="true"] {{
    font-size: {self.typography['size']['heading']};
    font-weight: {self.typography['weight']['semibold']};
    color: {self.colors['text']['heading']};
}}

QLabel[caption="true"] {{
    font-size: {self.typography['size']['caption']};
    color: {self.colors['text']['secondary']};
}}

/* === QGroupBox - Minimal Border === */
QGroupBox {{
    background-color: transparent;
    border: 1px solid {self.colors['border']['subtle']};
    border-radius: {self.border_radius['xs']};
    margin-top: {self.spacing['xl']};
    padding: {self.spacing['lg']};
    padding-top: {self.spacing['2xl']};
    font-weight: {self.typography['weight']['normal']};
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: {self.spacing['lg']};
    padding: 0 {self.spacing['sm']};
    color: {self.colors['text']['secondary']};
    background-color: {self.colors['background']['primary']};
    font-size: {self.typography['size']['body-small']};
    font-weight: {self.typography['weight']['medium']};
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}

/* === QTreeWidget - Compact Tree === */
QTreeWidget {{
    background-color: {self.colors['background']['secondary']};
    alternate-background-color: {self.colors['background']['tertiary']};
    color: {self.colors['text']['primary']};
    border: none;
    border-radius: 0;
    outline: none;
    show-decoration-selected: 1;
}}

QTreeWidget::item {{
    padding: {self.spacing['xs']} {self.spacing['sm']};
    min-height: 20px;
    border: none;
}}

QTreeWidget::item:selected {{
    background-color: {self.colors['selection']['background']};
    color: {self.colors['selection']['text']};
}}

QTreeWidget::item:hover:!selected {{
    background-color: {self.colors['background']['tertiary']};
}}

QTreeWidget::branch {{
    background-color: transparent;
}}

QTreeWidget::branch:has-children:closed {{
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid {self.colors['text']['secondary']};
    margin-left: 4px;
}}

QTreeWidget::branch:has-children:open {{
    image: none;
    border-left: 6px solid transparent;
    border-right: 6px solid transparent;
    border-bottom: 5px solid {self.colors['text']['secondary']};
    margin-left: 4px;
}}

QTreeWidget QHeaderView::section {{
    background-color: {self.colors['background']['tertiary']};
    color: {self.colors['text']['secondary']};
    border: none;
    border-bottom: 1px solid {self.colors['border']['subtle']};
    padding: {self.spacing['sm']} {self.spacing['md']};
    font-weight: {self.typography['weight']['medium']};
    font-size: {self.typography['size']['body-small']};
    text-transform: uppercase;
}}

/* === QListWidget - Compact List === */
QListWidget {{
    background-color: {self.colors['background']['secondary']};
    color: {self.colors['text']['primary']};
    border: 1px solid {self.colors['border']['subtle']};
    border-radius: {self.border_radius['xs']};
    outline: none;
    padding: {self.spacing['xs']};
}}

QListWidget::item {{
    padding: {self.spacing['sm']} {self.spacing['md']};
    border: none;
    border-radius: {self.border_radius['xs']};
}}

QListWidget::item:selected {{
    background-color: {self.colors['selection']['background']};
    color: {self.colors['selection']['text']};
}}

QListWidget::item:hover:!selected {{
    background-color: {self.colors['background']['tertiary']};
}}

/* === QTabWidget - Clean Tabs === */
QTabWidget::pane {{
    background-color: {self.colors['background']['secondary']};
    border: 1px solid {self.colors['border']['subtle']};
    border-top: none;
    border-radius: 0;
}}

QTabBar {{
    background-color: transparent;
}}

QTabBar::tab {{
    background-color: {self.colors['background']['tertiary']};
    color: {self.colors['text']['secondary']};
    border: 1px solid {self.colors['border']['subtle']};
    border-bottom: none;
    padding: {self.spacing['sm']} {self.spacing['xl']};
    margin-right: 1px;
    min-width: 60px;
}}

QTabBar::tab:selected {{
    background-color: {self.colors['background']['secondary']};
    color: {self.colors['text']['primary']};
    border-bottom: 2px solid {self.colors['action']['primary']};
    margin-bottom: -1px;
}}

QTabBar::tab:hover:!selected {{
    background-color: {self.colors['background']['surface']};
    color: {self.colors['text']['primary']};
}}

QTabBar::tab:first {{
    border-top-left-radius: {self.border_radius['xs']};
}}

QTabBar::tab:last {{
    border-top-right-radius: {self.border_radius['xs']};
}}

/* === QToolBar - Integrated Toolbar === */
QToolBar {{
    background-color: {self.colors['background']['secondary']};
    border: none;
    border-bottom: 1px solid {self.colors['border']['subtle']};
    spacing: {self.spacing['xs']};
    padding: {self.spacing['xs']} {self.spacing['sm']};
}}

QToolBar::separator {{
    background-color: {self.colors['border']['subtle']};
    width: 1px;
    margin: {self.spacing['sm']} {self.spacing['sm']};
}}

QToolButton {{
    background-color: transparent;
    color: {self.colors['text']['primary']};
    border: none;
    border-radius: {self.border_radius['xs']};
    padding: {self.spacing['sm']};
    margin: 1px;
}}

QToolButton:hover {{
    background-color: {self.colors['background']['tertiary']};
}}

QToolButton:pressed {{
    background-color: {self.colors['background']['primary']};
}}

QToolButton:checked {{
    background-color: {self.colors['selection']['background']};
    border: 1px solid {self.colors['selection']['border']};
}}

QToolButton[popupMode="1"] {{
    padding-right: 16px;
}}

QToolButton::menu-indicator {{
    image: none;
    subcontrol-origin: padding;
    subcontrol-position: right center;
    right: 4px;
}}

/* === QMenuBar - Clean Menu === */
QMenuBar {{
    background-color: {self.colors['background']['secondary']};
    color: {self.colors['text']['primary']};
    border-bottom: 1px solid {self.colors['border']['subtle']};
    padding: 0;
    spacing: 0;
}}

QMenuBar::item {{
    background-color: transparent;
    padding: {self.spacing['sm']} {self.spacing['lg']};
}}

QMenuBar::item:selected {{
    background-color: {self.colors['background']['tertiary']};
}}

QMenuBar::item:pressed {{
    background-color: {self.colors['selection']['background']};
    color: {self.colors['selection']['text']};
}}

/* === QMenu - Dropdown Menu === */
QMenu {{
    background-color: {self.colors['background']['secondary']};
    color: {self.colors['text']['primary']};
    border: 1px solid {self.colors['border']['subtle']};
    padding: {self.spacing['xs']};
}}

QMenu::item {{
    padding: {self.spacing['sm']} {self.spacing['2xl']};
    padding-right: {self.spacing['3xl']};
}}

QMenu::item:selected {{
    background-color: {self.colors['selection']['background']};
    color: {self.colors['selection']['text']};
}}

QMenu::item:disabled {{
    color: {self.colors['text']['disabled']};
}}

QMenu::separator {{
    height: 1px;
    background-color: {self.colors['border']['divider']};
    margin: {self.spacing['xs']} {self.spacing['md']};
}}

QMenu::icon {{
    margin-left: {self.spacing['md']};
}}

QMenu::indicator {{
    width: 16px;
    height: 16px;
    margin-left: {self.spacing['sm']};
}}

/* === QStatusBar - Minimal Status === */
QStatusBar {{
    background-color: {self.colors['background']['secondary']};
    color: {self.colors['text']['secondary']};
    border-top: 1px solid {self.colors['border']['subtle']};
    font-size: {self.typography['size']['body-small']};
}}

QStatusBar::item {{
    border: none;
}}

QStatusBar QLabel {{
    padding: {self.spacing['xs']} {self.spacing['md']};
    color: {self.colors['text']['secondary']};
}}

/* === QDockWidget - FreeCAD Panel Style === */
QDockWidget {{
    color: {self.colors['text']['primary']};
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
}}

QDockWidget::title {{
    background-color: {self.colors['background']['secondary']};
    border: none;
    border-bottom: 1px solid {self.colors['border']['subtle']};
    padding: {self.spacing['sm']} {self.spacing['md']};
    font-size: {self.typography['size']['body-small']};
    font-weight: {self.typography['weight']['medium']};
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}

QDockWidget::close-button, QDockWidget::float-button {{
    background-color: transparent;
    border: none;
    padding: {self.spacing['xs']};
    icon-size: 12px;
}}

QDockWidget::close-button:hover, QDockWidget::float-button:hover {{
    background-color: {self.colors['background']['tertiary']};
}}

/* === QScrollBar - Minimal Scrollbar === */
QScrollBar:vertical {{
    background-color: {self.colors['background']['primary']};
    width: 10px;
    border: none;
    margin: 0;
}}

QScrollBar::handle:vertical {{
    background-color: {self.colors['border']['subtle']};
    border-radius: {self.border_radius['xs']};
    min-height: 24px;
    margin: 2px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: {self.colors['border']['hover']};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}

QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: none;
}}

QScrollBar:horizontal {{
    background-color: {self.colors['background']['primary']};
    height: 10px;
    border: none;
    margin: 0;
}}

QScrollBar::handle:horizontal {{
    background-color: {self.colors['border']['subtle']};
    border-radius: {self.border_radius['xs']};
    min-width: 24px;
    margin: 2px;
}}

QScrollBar::handle:horizontal:hover {{
    background-color: {self.colors['border']['hover']};
}}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0px;
}}

QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
    background: none;
}}

/* === QProgressBar - Slim Progress === */
QProgressBar {{
    background-color: {self.colors['background']['tertiary']};
    border: none;
    border-radius: {self.border_radius['xs']};
    text-align: center;
    color: {self.colors['text']['primary']};
    font-size: {self.typography['size']['body-small']};
    height: 6px;
}}

QProgressBar::chunk {{
    background-color: {self.colors['action']['primary']};
    border-radius: {self.border_radius['xs']};
}}

/* === QCheckBox - Compact Checkbox === */
QCheckBox {{
    color: {self.colors['text']['primary']};
    spacing: {self.spacing['sm']};
}}

QCheckBox::indicator {{
    width: 14px;
    height: 14px;
    border: 1px solid {self.colors['border']['subtle']};
    border-radius: {self.border_radius['xs']};
    background-color: {self.colors['background']['tertiary']};
}}

QCheckBox::indicator:hover {{
    border-color: {self.colors['border']['hover']};
    background-color: {self.colors['background']['surface']};
}}

QCheckBox::indicator:checked {{
    background-color: {self.colors['action']['primary']};
    border-color: {self.colors['action']['primary']};
}}

QCheckBox::indicator:checked:hover {{
    background-color: {self.colors['action']['primary-hover']};
    border-color: {self.colors['action']['primary-hover']};
}}

QCheckBox::indicator:disabled {{
    background-color: {self.colors['background']['secondary']};
    border-color: {self.colors['border']['divider']};
}}

/* === QRadioButton - Compact Radio === */
QRadioButton {{
    color: {self.colors['text']['primary']};
    spacing: {self.spacing['sm']};
}}

QRadioButton::indicator {{
    width: 14px;
    height: 14px;
    border: 1px solid {self.colors['border']['subtle']};
    border-radius: 7px;
    background-color: {self.colors['background']['tertiary']};
}}

QRadioButton::indicator:hover {{
    border-color: {self.colors['border']['hover']};
}}

QRadioButton::indicator:checked {{
    background-color: {self.colors['action']['primary']};
    border: 4px solid {self.colors['background']['tertiary']};
}}

/* === QSlider - Minimal Slider === */
QSlider::groove:horizontal {{
    background-color: {self.colors['background']['tertiary']};
    height: 4px;
    border-radius: 2px;
}}

QSlider::handle:horizontal {{
    background-color: {self.colors['action']['primary']};
    border: none;
    width: 12px;
    height: 12px;
    margin: -4px 0;
    border-radius: 6px;
}}

QSlider::handle:horizontal:hover {{
    background-color: {self.colors['action']['primary-hover']};
}}

QSlider::sub-page:horizontal {{
    background-color: {self.colors['action']['primary']};
    border-radius: 2px;
}}

/* === QToolTip - Clean Tooltip === */
QToolTip {{
    background-color: {self.colors['background']['surface']};
    color: {self.colors['text']['primary']};
    border: 1px solid {self.colors['border']['subtle']};
    padding: {self.spacing['sm']} {self.spacing['md']};
    font-size: {self.typography['size']['body-small']};
}}

/* === QDialog === */
QDialog {{
    background-color: {self.colors['background']['primary']};
}}

/* === QSplitter - Subtle Splitter === */
QSplitter::handle {{
    background-color: {self.colors['border']['subtle']};
}}

QSplitter::handle:horizontal {{
    width: 2px;
}}

QSplitter::handle:vertical {{
    height: 2px;
}}

QSplitter::handle:hover {{
    background-color: {self.colors['action']['primary']};
}}

/* === QHeaderView - Table Headers === */
QHeaderView::section {{
    background-color: {self.colors['background']['tertiary']};
    color: {self.colors['text']['secondary']};
    border: none;
    border-right: 1px solid {self.colors['border']['divider']};
    border-bottom: 1px solid {self.colors['border']['subtle']};
    padding: {self.spacing['sm']} {self.spacing['md']};
    font-weight: {self.typography['weight']['medium']};
    font-size: {self.typography['size']['body-small']};
}}

QHeaderView::section:hover {{
    background-color: {self.colors['background']['surface']};
}}

/* === QScrollArea === */
QScrollArea {{
    background-color: transparent;
    border: none;
}}

QScrollArea > QWidget > QWidget {{
    background-color: transparent;
}}

/* === QPlainTextEdit - Console/Output === */
QPlainTextEdit {{
    background-color: {self.colors['background']['primary']};
    color: {self.colors['text']['primary']};
    border: 1px solid {self.colors['border']['subtle']};
    border-radius: {self.border_radius['xs']};
    font-family: {self.typography['family']['monospace']};
    font-size: {self.typography['size']['body']};
    selection-background-color: {self.colors['selection']['background']};
    selection-color: {self.colors['selection']['text']};
}}

/* === QTextEdit === */
QTextEdit {{
    background-color: {self.colors['background']['primary']};
    color: {self.colors['text']['primary']};
    border: 1px solid {self.colors['border']['subtle']};
    border-radius: {self.border_radius['xs']};
    selection-background-color: {self.colors['selection']['background']};
    selection-color: {self.colors['selection']['text']};
}}

/* === QProgressDialog === */
QProgressDialog {{
    background-color: {self.colors['background']['secondary']};
}}

/* === QMessageBox === */
QMessageBox {{
    background-color: {self.colors['background']['secondary']};
}}

QMessageBox QLabel {{
    color: {self.colors['text']['primary']};
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

/* === Form Layout === */
QFormLayout {{
    spacing: {self.spacing['md']};
}}

/* === Disabled state for all widgets === */
*:disabled {{
    color: {self.colors['text']['disabled']};
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
             3.0),  # Lower requirement for disabled text
            ("Selection Text on Selection Background",
             self.colors['selection']['text'],
             self.colors['selection']['background'],
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
    token_file = gui_dir / 'design_tokens.json'

    # Generate theme
    generator = ThemeGenerator(token_file)

    # Validate contrast
    print("=== WCAG Contrast Validation ===")
    results = generator.validate_contrast()

    all_pass = True
    for name, result in results.items():
        status = "PASS" if result['passes'] else "FAIL"
        print(f"{status} {name}:")
        print(f"  {result['foreground']} on {result['background']}")
        print(f"  Ratio: {result['ratio']}:1 (Required: {result['required']}:1)")
        print(f"  WCAG Level: {result['wcag_level']}")
        print()

        if not result['passes']:
            all_pass = False

    if all_pass:
        print("All contrast ratios meet WCAG AA standards!")
    else:
        print("Some contrast ratios fail WCAG AA standards.")
        sys.exit(1)

    # Generate QSS
    output_file = gui_dir / 'theme.qss'
    generator.save_qss(output_file)
    print(f"\nGenerated stylesheet: {output_file}")
