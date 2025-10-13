"""
CAD-style theme for svVascularize GUI, inspired by FreeCAD, SolidWorks, and AutoCAD.
Professional engineering/fabrication-focused design.
"""

class CADTheme:
    """
    Professional CAD-style theme with neutral colors and technical appearance.
    """

    # CAD Color palette - neutral, professional
    PRIMARY = "#5A7FA8"  # Muted blue
    PRIMARY_DARK = "#3D5A7A"
    PRIMARY_LIGHT = "#7A9BC0"

    ACCENT = "#E67E22"  # Engineering orange
    ACCENT_HOVER = "#D35400"

    # Neutral professional colors
    BACKGROUND = "#353535"  # Dark gray background (CAD standard)
    SURFACE = "#2C2C2C"  # Darker panels
    SURFACE_LIGHT = "#404040"  # Lighter panels

    PANEL_BG = "#2A2A2A"  # Side panel background
    TOOLBAR_BG = "#3C3C3C"  # Toolbar background

    TEXT_PRIMARY = "#E0E0E0"  # Light text on dark
    TEXT_SECONDARY = "#B0B0B0"  # Muted text
    TEXT_ACCENT = "#4CAF50"  # Green for active states

    BORDER = "#1A1A1A"  # Dark borders
    DIVIDER = "#505050"  # Subtle dividers

    # 3D viewport
    VIEWPORT_BG = "#353535"  # Standard CAD viewport gray

    # Status colors
    SUCCESS = "#4CAF50"  # Green
    WARNING = "#FF9800"  # Orange
    ERROR = "#E74C3C"  # Red
    INFO = "#3498DB"  # Blue

    @staticmethod
    def get_stylesheet():
        """
        Get the complete CAD-style stylesheet.

        Returns
        -------
        str
            Qt stylesheet string
        """
        return f"""
        /* Main Window - Dark CAD Theme */
        QMainWindow {{
            background-color: {CADTheme.BACKGROUND};
            color: {CADTheme.TEXT_PRIMARY};
        }}

        QWidget {{
            background-color: {CADTheme.BACKGROUND};
            color: {CADTheme.TEXT_PRIMARY};
            font-family: "Segoe UI", "Arial", sans-serif;
            font-size: 9pt;
        }}

        /* Toolbars */
        QToolBar {{
            background-color: {CADTheme.TOOLBAR_BG};
            border: none;
            border-bottom: 1px solid {CADTheme.BORDER};
            spacing: 3px;
            padding: 2px;
        }}

        QToolBar::separator {{
            background-color: {CADTheme.DIVIDER};
            width: 1px;
            margin: 4px 2px;
        }}

        QToolButton {{
            background-color: transparent;
            border: 1px solid transparent;
            border-radius: 3px;
            padding: 5px;
            margin: 1px;
            color: {CADTheme.TEXT_PRIMARY};
        }}

        QToolButton:hover {{
            background-color: {CADTheme.SURFACE_LIGHT};
            border: 1px solid {CADTheme.PRIMARY};
        }}

        QToolButton:pressed {{
            background-color: {CADTheme.PRIMARY_DARK};
        }}

        QToolButton:checked {{
            background-color: {CADTheme.PRIMARY};
            border: 1px solid {CADTheme.PRIMARY_LIGHT};
        }}

        /* Dock Widgets */
        QDockWidget {{
            titlebar-close-icon: url(close.png);
            titlebar-normal-icon: url(float.png);
            border: 1px solid {CADTheme.BORDER};
            color: {CADTheme.TEXT_PRIMARY};
        }}

        QDockWidget::title {{
            background-color: {CADTheme.PANEL_BG};
            text-align: left;
            padding: 6px 8px;
            border-bottom: 1px solid {CADTheme.DIVIDER};
            font-weight: bold;
            font-size: 9pt;
        }}

        QDockWidget::close-button, QDockWidget::float-button {{
            border: none;
            background-color: transparent;
            padding: 2px;
        }}

        QDockWidget::close-button:hover, QDockWidget::float-button:hover {{
            background-color: {CADTheme.SURFACE_LIGHT};
        }}

        /* Tree/List Widgets */
        QTreeWidget, QListWidget {{
            background-color: {CADTheme.PANEL_BG};
            border: 1px solid {CADTheme.BORDER};
            border-radius: 0px;
            color: {CADTheme.TEXT_PRIMARY};
            outline: none;
        }}

        QTreeWidget::item, QListWidget::item {{
            padding: 4px;
            border: none;
        }}

        QTreeWidget::item:selected, QListWidget::item:selected {{
            background-color: {CADTheme.PRIMARY};
            color: white;
        }}

        QTreeWidget::item:hover, QListWidget::item:hover {{
            background-color: {CADTheme.SURFACE_LIGHT};
        }}

        QTreeWidget::branch {{
            background-color: {CADTheme.PANEL_BG};
        }}

        QTreeWidget::branch:has-children:closed {{
            image: none;
            border: none;
        }}

        QTreeWidget::branch:has-children:open {{
            image: none;
            border: none;
        }}

        /* Group Boxes */
        QGroupBox {{
            font-weight: bold;
            font-size: 9pt;
            border: 1px solid {CADTheme.DIVIDER};
            border-radius: 2px;
            margin-top: 10px;
            padding-top: 12px;
            background-color: {CADTheme.SURFACE};
            color: {CADTheme.TEXT_PRIMARY};
        }}

        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 8px;
            padding: 0 4px;
            color: {CADTheme.TEXT_ACCENT};
            background-color: {CADTheme.SURFACE};
        }}

        /* Buttons */
        QPushButton {{
            background-color: {CADTheme.SURFACE_LIGHT};
            color: {CADTheme.TEXT_PRIMARY};
            border: 1px solid {CADTheme.DIVIDER};
            border-radius: 2px;
            padding: 6px 12px;
            font-size: 9pt;
            min-height: 24px;
        }}

        QPushButton:hover {{
            background-color: {CADTheme.PRIMARY};
            border-color: {CADTheme.PRIMARY_LIGHT};
        }}

        QPushButton:pressed {{
            background-color: {CADTheme.PRIMARY_DARK};
        }}

        QPushButton:disabled {{
            background-color: {CADTheme.SURFACE};
            color: {CADTheme.TEXT_SECONDARY};
            border-color: {CADTheme.BORDER};
        }}

        /* Primary Action Button */
        QPushButton#primaryAction {{
            background-color: {CADTheme.ACCENT};
            color: white;
            border: 1px solid {CADTheme.ACCENT};
            font-weight: bold;
        }}

        QPushButton#primaryAction:hover {{
            background-color: {CADTheme.ACCENT_HOVER};
        }}

        /* Danger Button */
        QPushButton#dangerAction {{
            background-color: transparent;
            color: {CADTheme.ERROR};
            border: 1px solid {CADTheme.ERROR};
        }}

        QPushButton#dangerAction:hover {{
            background-color: {CADTheme.ERROR};
            color: white;
        }}

        /* Input Fields */
        QSpinBox, QDoubleSpinBox, QLineEdit {{
            background-color: {CADTheme.PANEL_BG};
            border: 1px solid {CADTheme.BORDER};
            border-radius: 2px;
            padding: 4px 6px;
            color: {CADTheme.TEXT_PRIMARY};
            selection-background-color: {CADTheme.PRIMARY};
            min-height: 20px;
        }}

        QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{
            border-color: {CADTheme.PRIMARY};
            background-color: {CADTheme.SURFACE};
        }}

        QSpinBox:disabled, QDoubleSpinBox:disabled, QLineEdit:disabled {{
            background-color: {CADTheme.SURFACE};
            color: {CADTheme.TEXT_SECONDARY};
        }}

        QSpinBox::up-button, QDoubleSpinBox::up-button {{
            background-color: {CADTheme.SURFACE_LIGHT};
            border-left: 1px solid {CADTheme.BORDER};
            border-bottom: 1px solid {CADTheme.BORDER};
            width: 16px;
        }}

        QSpinBox::down-button, QDoubleSpinBox::down-button {{
            background-color: {CADTheme.SURFACE_LIGHT};
            border-left: 1px solid {CADTheme.BORDER};
            width: 16px;
        }}

        QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
        QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
            background-color: {CADTheme.PRIMARY};
        }}

        /* Combo Boxes */
        QComboBox {{
            background-color: {CADTheme.PANEL_BG};
            border: 1px solid {CADTheme.BORDER};
            border-radius: 2px;
            padding: 4px 6px;
            color: {CADTheme.TEXT_PRIMARY};
            min-height: 20px;
        }}

        QComboBox:focus {{
            border-color: {CADTheme.PRIMARY};
        }}

        QComboBox::drop-down {{
            border: none;
            width: 20px;
        }}

        QComboBox::down-arrow {{
            image: none;
            border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-top: 5px solid {CADTheme.TEXT_SECONDARY};
            margin-right: 6px;
        }}

        QComboBox QAbstractItemView {{
            background-color: {CADTheme.PANEL_BG};
            border: 1px solid {CADTheme.BORDER};
            selection-background-color: {CADTheme.PRIMARY};
            selection-color: white;
            color: {CADTheme.TEXT_PRIMARY};
        }}

        /* Check Boxes */
        QCheckBox {{
            spacing: 6px;
            color: {CADTheme.TEXT_PRIMARY};
        }}

        QCheckBox::indicator {{
            width: 16px;
            height: 16px;
            border: 1px solid {CADTheme.DIVIDER};
            border-radius: 2px;
            background-color: {CADTheme.PANEL_BG};
        }}

        QCheckBox::indicator:hover {{
            border-color: {CADTheme.PRIMARY};
        }}

        QCheckBox::indicator:checked {{
            background-color: {CADTheme.TEXT_ACCENT};
            border-color: {CADTheme.TEXT_ACCENT};
        }}

        /* Labels */
        QLabel {{
            color: {CADTheme.TEXT_PRIMARY};
            background-color: transparent;
        }}

        /* Status Bar */
        QStatusBar {{
            background-color: {CADTheme.SURFACE};
            color: {CADTheme.TEXT_PRIMARY};
            border-top: 1px solid {CADTheme.BORDER};
            font-size: 8pt;
        }}

        QStatusBar::item {{
            border: none;
        }}

        QStatusBar QLabel {{
            padding: 2px 8px;
        }}

        /* Menu Bar */
        QMenuBar {{
            background-color: {CADTheme.TOOLBAR_BG};
            color: {CADTheme.TEXT_PRIMARY};
            border-bottom: 1px solid {CADTheme.BORDER};
            font-size: 9pt;
        }}

        QMenuBar::item {{
            padding: 6px 12px;
            background-color: transparent;
        }}

        QMenuBar::item:selected {{
            background-color: {CADTheme.PRIMARY};
        }}

        QMenu {{
            background-color: {CADTheme.SURFACE};
            border: 1px solid {CADTheme.BORDER};
            color: {CADTheme.TEXT_PRIMARY};
        }}

        QMenu::item {{
            padding: 6px 24px 6px 8px;
        }}

        QMenu::item:selected {{
            background-color: {CADTheme.PRIMARY};
        }}

        QMenu::separator {{
            height: 1px;
            background-color: {CADTheme.DIVIDER};
            margin: 4px 0px;
        }}

        /* Scroll Bars */
        QScrollBar:vertical {{
            background-color: {CADTheme.PANEL_BG};
            width: 14px;
            border: none;
        }}

        QScrollBar::handle:vertical {{
            background-color: {CADTheme.DIVIDER};
            min-height: 24px;
            border-radius: 2px;
            margin: 2px;
        }}

        QScrollBar::handle:vertical:hover {{
            background-color: {CADTheme.PRIMARY};
        }}

        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            border: none;
            background: none;
            height: 0px;
        }}

        QScrollBar:horizontal {{
            background-color: {CADTheme.PANEL_BG};
            height: 14px;
            border: none;
        }}

        QScrollBar::handle:horizontal {{
            background-color: {CADTheme.DIVIDER};
            min-width: 24px;
            border-radius: 2px;
            margin: 2px;
        }}

        QScrollBar::handle:horizontal:hover {{
            background-color: {CADTheme.PRIMARY};
        }}

        /* Progress Bar */
        QProgressBar {{
            border: 1px solid {CADTheme.BORDER};
            border-radius: 2px;
            text-align: center;
            background-color: {CADTheme.PANEL_BG};
            color: {CADTheme.TEXT_PRIMARY};
        }}

        QProgressBar::chunk {{
            background-color: {CADTheme.TEXT_ACCENT};
        }}

        /* Splitter */
        QSplitter::handle {{
            background-color: {CADTheme.BORDER};
        }}

        QSplitter::handle:horizontal {{
            width: 2px;
        }}

        QSplitter::handle:vertical {{
            height: 2px;
        }}

        /* Tab Widget */
        QTabWidget::pane {{
            border: 1px solid {CADTheme.BORDER};
            background-color: {CADTheme.SURFACE};
        }}

        QTabBar::tab {{
            background-color: {CADTheme.SURFACE};
            border: 1px solid {CADTheme.BORDER};
            padding: 6px 12px;
            color: {CADTheme.TEXT_PRIMARY};
        }}

        QTabBar::tab:selected {{
            background-color: {CADTheme.SURFACE_LIGHT};
            border-bottom-color: {CADTheme.SURFACE_LIGHT};
        }}

        QTabBar::tab:hover {{
            background-color: {CADTheme.PRIMARY};
        }}

        /* Tool Tips */
        QToolTip {{
            background-color: {CADTheme.PANEL_BG};
            color: {CADTheme.TEXT_PRIMARY};
            border: 1px solid {CADTheme.PRIMARY};
            padding: 4px;
            font-size: 8pt;
        }}

        /* Scroll Area */
        QScrollArea {{
            border: none;
            background-color: transparent;
        }}
        """


class CADIcons:
    """
    CAD-standard icons using Unicode characters.
    """

    # File operations
    NEW = "\u2795"  # ➕ Heavy Plus
    OPEN = "\U0001F4C2"  # 📂 Folder
    SAVE = "\U0001F4BE"  # 💾 Floppy
    EXPORT = "\u2934"  # ⤴ Arrow pointing up

    # View operations
    VIEW_FIT = "\u26F6"  # ⛶ Square with perspective
    VIEW_TOP = "\u2B1C"  # ⬜ White square
    VIEW_FRONT = "\u25A1"  # ▢ White square outline
    VIEW_RIGHT = "\u25A2"  # ▣ White square with lines
    VIEW_ISO = "\u25A6"  # ▦ Square with fill
    ZOOM_IN = "\U0001F50D"  # 🔍 Magnifier
    ZOOM_OUT = "\u2315"  # ⌕ Circle with minus

    # Object operations
    TREE = "\U0001F333"  # 🌳 Tree
    FOREST = "\u2663"  # ♣ Club (multiple trees)
    MESH = "\u25A6"  # ▦ Grid
    POINT = "\u25CF"  # ● Black circle
    VECTOR = "\u2192"  # → Arrow

    # Edit operations
    ADD = "\u002B"  # + Plus
    REMOVE = "\u2212"  # − Minus
    DELETE = "\u2716"  # ✖ Heavy X
    CLEAR = "\u267B"  # ♻ Recycle

    # Generation
    GENERATE = "\u25B6"  # ▶ Play
    STOP = "\u25A0"  # ■ Square
    SETTINGS = "\u2699"  # ⚙ Gear

    # Status
    SUCCESS = "\u2714"  # ✔ Check
    ERROR = "\u2716"  # ✖ X
    WARNING = "\u26A0"  # ⚠ Warning
    INFO = "\u2139"  # ℹ Info

    # Tools
    SELECT = "\u25C9"  # ◉ Circle with dot
    MEASURE = "\u29D0"  # ⧐ Ruler
    CAMERA = "\U0001F4F7"  # 📷 Camera
    VISIBLE = "\u25CE"  # ◎ Eye
    HIDDEN = "\u25CC"  # ◌ Empty circle
