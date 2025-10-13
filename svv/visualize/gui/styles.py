"""
Modern styling and theme system for svVascularize GUI.
"""

class ModernTheme:
    """
    Modern color scheme and styling for the application.
    """

    # Color palette
    PRIMARY = "#2196F3"  # Blue
    PRIMARY_DARK = "#1976D2"
    PRIMARY_LIGHT = "#BBDEFB"
    SECONDARY = "#FF5722"  # Deep Orange
    SUCCESS = "#4CAF50"  # Green
    WARNING = "#FF9800"  # Orange
    ERROR = "#F44336"  # Red

    # Neutral colors
    BACKGROUND = "#FAFAFA"
    SURFACE = "#FFFFFF"
    TEXT_PRIMARY = "#212121"
    TEXT_SECONDARY = "#757575"
    DIVIDER = "#E0E0E0"

    # Shadows
    SHADOW_LIGHT = "rgba(0, 0, 0, 0.1)"
    SHADOW_MEDIUM = "rgba(0, 0, 0, 0.2)"

    @staticmethod
    def get_stylesheet():
        """
        Get the complete stylesheet for the application.

        Returns
        -------
        str
            Qt stylesheet string
        """
        return f"""
        /* Main Window */
        QMainWindow {{
            background-color: {ModernTheme.BACKGROUND};
        }}

        /* Group Boxes */
        QGroupBox {{
            font-weight: bold;
            font-size: 13px;
            border: 2px solid {ModernTheme.DIVIDER};
            border-radius: 8px;
            margin-top: 12px;
            padding-top: 16px;
            background-color: {ModernTheme.SURFACE};
        }}

        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 12px;
            padding: 0 5px;
            color: {ModernTheme.PRIMARY};
        }}

        /* Buttons */
        QPushButton {{
            background-color: {ModernTheme.PRIMARY};
            color: white;
            border: none;
            border-radius: 6px;
            padding: 8px 16px;
            font-size: 12px;
            font-weight: 500;
            min-height: 28px;
        }}

        QPushButton:hover {{
            background-color: {ModernTheme.PRIMARY_DARK};
        }}

        QPushButton:pressed {{
            background-color: {ModernTheme.PRIMARY_DARK};
            padding-top: 9px;
            padding-bottom: 7px;
        }}

        QPushButton:disabled {{
            background-color: {ModernTheme.DIVIDER};
            color: {ModernTheme.TEXT_SECONDARY};
        }}

        /* Primary Action Button */
        QPushButton#primaryButton {{
            background-color: {ModernTheme.SECONDARY};
            font-size: 13px;
            font-weight: bold;
            padding: 12px 20px;
            min-height: 36px;
        }}

        QPushButton#primaryButton:hover {{
            background-color: #E64A19;
        }}

        /* Secondary Button */
        QPushButton#secondaryButton {{
            background-color: {ModernTheme.SURFACE};
            color: {ModernTheme.PRIMARY};
            border: 2px solid {ModernTheme.PRIMARY};
        }}

        QPushButton#secondaryButton:hover {{
            background-color: {ModernTheme.PRIMARY_LIGHT};
        }}

        /* Danger Button */
        QPushButton#dangerButton {{
            background-color: {ModernTheme.ERROR};
        }}

        QPushButton#dangerButton:hover {{
            background-color: #D32F2F;
        }}

        /* Checkable Buttons (Toggle) */
        QPushButton:checked {{
            background-color: {ModernTheme.SUCCESS};
        }}

        /* Spin Boxes and Line Edits */
        QSpinBox, QDoubleSpinBox, QLineEdit {{
            border: 2px solid {ModernTheme.DIVIDER};
            border-radius: 4px;
            padding: 6px 8px;
            background-color: {ModernTheme.SURFACE};
            selection-background-color: {ModernTheme.PRIMARY_LIGHT};
            min-height: 24px;
        }}

        QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{
            border-color: {ModernTheme.PRIMARY};
        }}

        QSpinBox:disabled, QDoubleSpinBox:disabled, QLineEdit:disabled {{
            background-color: {ModernTheme.BACKGROUND};
            color: {ModernTheme.TEXT_SECONDARY};
        }}

        /* Combo Boxes */
        QComboBox {{
            border: 2px solid {ModernTheme.DIVIDER};
            border-radius: 4px;
            padding: 6px 8px;
            background-color: {ModernTheme.SURFACE};
            min-height: 24px;
        }}

        QComboBox:focus {{
            border-color: {ModernTheme.PRIMARY};
        }}

        QComboBox::drop-down {{
            border: none;
            width: 24px;
        }}

        QComboBox::down-arrow {{
            image: none;
            border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-top: 6px solid {ModernTheme.TEXT_SECONDARY};
            margin-right: 8px;
        }}

        QComboBox QAbstractItemView {{
            border: 2px solid {ModernTheme.DIVIDER};
            border-radius: 4px;
            background-color: {ModernTheme.SURFACE};
            selection-background-color: {ModernTheme.PRIMARY_LIGHT};
            selection-color: {ModernTheme.TEXT_PRIMARY};
        }}

        /* List Widget */
        QListWidget {{
            border: 2px solid {ModernTheme.DIVIDER};
            border-radius: 4px;
            background-color: {ModernTheme.SURFACE};
            padding: 4px;
        }}

        QListWidget::item {{
            padding: 8px;
            border-radius: 4px;
            margin: 2px;
        }}

        QListWidget::item:selected {{
            background-color: {ModernTheme.PRIMARY_LIGHT};
            color: {ModernTheme.TEXT_PRIMARY};
        }}

        QListWidget::item:hover {{
            background-color: {ModernTheme.BACKGROUND};
        }}

        /* Check Boxes */
        QCheckBox {{
            spacing: 8px;
        }}

        QCheckBox::indicator {{
            width: 18px;
            height: 18px;
            border: 2px solid {ModernTheme.DIVIDER};
            border-radius: 4px;
            background-color: {ModernTheme.SURFACE};
        }}

        QCheckBox::indicator:hover {{
            border-color: {ModernTheme.PRIMARY};
        }}

        QCheckBox::indicator:checked {{
            background-color: {ModernTheme.PRIMARY};
            border-color: {ModernTheme.PRIMARY};
            image: none;
        }}

        /* Labels */
        QLabel {{
            color: {ModernTheme.TEXT_PRIMARY};
        }}

        QLabel#headerLabel {{
            font-size: 14px;
            font-weight: bold;
            color: {ModernTheme.PRIMARY};
        }}

        QLabel#subtitleLabel {{
            font-size: 11px;
            color: {ModernTheme.TEXT_SECONDARY};
        }}

        /* Status Bar */
        QStatusBar {{
            background-color: {ModernTheme.SURFACE};
            color: {ModernTheme.TEXT_SECONDARY};
            border-top: 1px solid {ModernTheme.DIVIDER};
            padding: 4px;
        }}

        /* Menu Bar */
        QMenuBar {{
            background-color: {ModernTheme.SURFACE};
            border-bottom: 1px solid {ModernTheme.DIVIDER};
            padding: 4px;
        }}

        QMenuBar::item {{
            padding: 6px 12px;
            border-radius: 4px;
        }}

        QMenuBar::item:selected {{
            background-color: {ModernTheme.PRIMARY_LIGHT};
        }}

        QMenu {{
            background-color: {ModernTheme.SURFACE};
            border: 2px solid {ModernTheme.DIVIDER};
            border-radius: 6px;
            padding: 4px;
        }}

        QMenu::item {{
            padding: 8px 24px;
            border-radius: 4px;
        }}

        QMenu::item:selected {{
            background-color: {ModernTheme.PRIMARY_LIGHT};
        }}

        /* Scroll Bar */
        QScrollBar:vertical {{
            border: none;
            background-color: {ModernTheme.BACKGROUND};
            width: 12px;
            margin: 0;
        }}

        QScrollBar::handle:vertical {{
            background-color: {ModernTheme.DIVIDER};
            border-radius: 6px;
            min-height: 24px;
        }}

        QScrollBar::handle:vertical:hover {{
            background-color: {ModernTheme.TEXT_SECONDARY};
        }}

        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            border: none;
            background: none;
            height: 0px;
        }}

        /* Progress Dialog */
        QProgressDialog {{
            background-color: {ModernTheme.SURFACE};
        }}

        QProgressBar {{
            border: 2px solid {ModernTheme.DIVIDER};
            border-radius: 6px;
            text-align: center;
            background-color: {ModernTheme.BACKGROUND};
        }}

        QProgressBar::chunk {{
            background-color: {ModernTheme.PRIMARY};
            border-radius: 4px;
        }}

        /* Splitter */
        QSplitter::handle {{
            background-color: {ModernTheme.DIVIDER};
        }}

        QSplitter::handle:horizontal {{
            width: 2px;
        }}

        QSplitter::handle:vertical {{
            height: 2px;
        }}

        /* Tool Tips */
        QToolTip {{
            background-color: {ModernTheme.TEXT_PRIMARY};
            color: white;
            border: none;
            border-radius: 4px;
            padding: 6px 8px;
            font-size: 11px;
        }}

        /* Scroll Area */
        QScrollArea {{
            border: none;
            background-color: transparent;
        }}
        """

    @staticmethod
    def get_button_style(button_type="primary"):
        """
        Get a specific button style.

        Parameters
        ----------
        button_type : str
            Type of button: "primary", "secondary", or "danger"

        Returns
        -------
        str
            Object name for Qt stylesheet selector
        """
        type_map = {
            "primary": "primaryButton",
            "secondary": "secondaryButton",
            "danger": "dangerButton"
        }
        return type_map.get(button_type, "")


class Icons:
    """
    Unicode icons for use in the GUI (works without external icon files).
    """

    # Common icons using Unicode
    FOLDER_OPEN = "\U0001F4C2"  # 📂
    SAVE = "\U0001F4BE"  # 💾
    SETTINGS = "\u2699"  # ⚙
    TREE = "\U0001F333"  # 🌳
    FOREST = "\U0001F332"  # 🌲
    POINT = "\u2022"  # •
    ARROW = "\u2192"  # →
    PLUS = "\u002B"  # +
    MINUS = "\u2212"  # −
    CHECK = "\u2714"  # ✔
    CROSS = "\u2716"  # ✖
    INFO = "\u2139"  # ℹ
    WARNING = "\u26A0"  # ⚠
    PLAY = "\u25B6"  # ▶
    CAMERA = "\U0001F4F7"  # 📷
    EYE = "\U0001F441"  # 👁
    REFRESH = "\u21BB"  # ↻
    HELP = "\u003F"  # ?
