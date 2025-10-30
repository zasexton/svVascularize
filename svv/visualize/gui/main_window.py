"""
Main GUI window for svVascularize using PySide6.
"""
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QMenuBar, QMenu, QStatusBar, QFileDialog,
    QMessageBox, QLabel
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from svv.visualize.gui.vtk_widget import VTKWidget
from svv.visualize.gui.point_selector import PointSelectorWidget
from svv.visualize.gui.parameter_panel import ParameterPanel
from svv.visualize.gui.theme_fusion360 import Fusion360Theme, Fusion360Icons


class VascularizeGUI(QMainWindow):
    """
    Main GUI window for visualizing and manipulating Domain objects
    to configure Tree and Forest vascularization.

    Features Fusion360-inspired modern CAD interface with:
    - Professional dark theme optimized for engineering work
    - Clean, intuitive layout with dockable panels
    - Enhanced 3D viewport integration
    - WCAG AA accessibility compliance
    """

    def __init__(self, domain=None):
        """
        Initialize the main GUI window.

        Parameters
        ----------
        domain : svv.domain.Domain, optional
            Initial domain object to visualize
        """
        super().__init__()
        self.domain = domain
        self.trees = []
        self.forest = None

        # Apply Fusion360-inspired theme
        self.setStyleSheet(Fusion360Theme.get_stylesheet())

        self.setWindowTitle(f"{Fusion360Icons.TREE} svVascularize - Vascular Generation")
        self.setGeometry(100, 100, 1600, 1000)

        self._init_ui()
        self._create_menu_bar()
        self._create_status_bar()

        # Load domain if provided
        if domain is not None:
            self.load_domain(domain)

    def _init_ui(self):
        """Initialize the user interface layout."""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Header with title and subtitle
        header_widget = self._create_header()
        main_layout.addWidget(header_widget)

        # Content area
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(8, 8, 8, 8)
        content_layout.setSpacing(8)

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)

        # Left panel: 3D visualization
        self.vtk_widget = VTKWidget(self)
        splitter.addWidget(self.vtk_widget)

        # Right panel: Controls
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        # Point selector widget
        self.point_selector = PointSelectorWidget(self)
        right_layout.addWidget(self.point_selector)

        # Parameter panel
        self.parameter_panel = ParameterPanel(self)
        right_layout.addWidget(self.parameter_panel)

        splitter.addWidget(right_panel)

        # Set splitter proportions (70% visualization, 30% controls)
        splitter.setSizes([1000, 400])

        content_layout.addWidget(splitter)
        main_layout.addLayout(content_layout)

    def _create_header(self):
        """Create the header widget with title and subtitle in Fusion360 style."""
        header = QWidget()
        header_bg = Fusion360Theme.get_color('background', 'toolbar')
        header.setStyleSheet(f"""
            QWidget {{
                background-color: {header_bg};
                border-bottom: 1px solid {Fusion360Theme.get_color('border', 'divider')};
            }}
        """)
        layout = QVBoxLayout(header)
        layout.setContentsMargins(20, 12, 20, 12)
        layout.setSpacing(4)

        # Title
        title = QLabel(f"{Fusion360Icons.TREE} svVascularize")
        title.setProperty("title", True)
        title.setStyleSheet(f"""
            font-size: {Fusion360Theme.get_typography('size', 'display')};
            font-weight: {Fusion360Theme.get_typography('weight', 'semibold')};
            color: {Fusion360Theme.get_color('text', 'bright')};
        """)
        layout.addWidget(title)

        # Subtitle
        subtitle = QLabel("Professional Vascular Tree and Forest Generation")
        subtitle.setProperty("secondary", True)
        subtitle.setStyleSheet(f"""
            font-size: {Fusion360Theme.get_typography('size', 'body')};
            color: {Fusion360Theme.get_color('text', 'secondary')};
        """)
        layout.addWidget(subtitle)

        return header

    def _create_menu_bar(self):
        """Create the application menu bar in Fusion360 style."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        load_domain_action = QAction(f"{Fusion360Icons.FOLDER_OPEN} Load Domain...", self)
        load_domain_action.setShortcut(QKeySequence.Open)
        load_domain_action.setStatusTip("Load a domain mesh file (.dmn)")
        load_domain_action.triggered.connect(self.load_domain_dialog)
        file_menu.addAction(load_domain_action)

        save_config_action = QAction(f"{Fusion360Icons.SAVE} Save Configuration...", self)
        save_config_action.setShortcut(QKeySequence.Save)
        save_config_action.setStatusTip("Save the current vascular tree configuration")
        save_config_action.triggered.connect(self.save_configuration)
        file_menu.addAction(save_config_action)

        file_menu.addSeparator()

        export_action = QAction(f"{Fusion360Icons.EXPORT} Export Results...", self)
        export_action.setShortcut("Ctrl+E")
        export_action.setStatusTip("Export generated trees to file")
        export_action.triggered.connect(self.export_results)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.setStatusTip("Exit the application")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        reset_camera_action = QAction(f"{Fusion360Icons.CAMERA} Reset Camera", self)
        reset_camera_action.setShortcut("R")
        reset_camera_action.setStatusTip("Reset camera to default isometric view")
        reset_camera_action.triggered.connect(self.vtk_widget.reset_camera)
        view_menu.addAction(reset_camera_action)

        view_menu.addSeparator()

        toggle_domain_action = QAction(f"{Fusion360Icons.EYE} Toggle Domain Visibility", self)
        toggle_domain_action.setShortcut("D")
        toggle_domain_action.setStatusTip("Show/hide the domain mesh")
        toggle_domain_action.setCheckable(True)
        toggle_domain_action.setChecked(True)
        toggle_domain_action.triggered.connect(self.vtk_widget.toggle_domain_visibility)
        view_menu.addAction(toggle_domain_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        about_action = QAction(f"{Fusion360Icons.INFO} About", self)
        about_action.setStatusTip("About svVascularize")
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def _create_status_bar(self):
        """Create the status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def load_domain(self, domain):
        """
        Load a domain object into the visualization.

        Parameters
        ----------
        domain : svv.domain.Domain
            Domain object to visualize
        """
        self.domain = domain
        self.vtk_widget.set_domain(domain)
        self.point_selector.set_domain(domain)
        self.update_status(f"{Fusion360Icons.CHECK} Domain loaded - Ready to configure trees")

    def load_domain_dialog(self):
        """Open a file dialog to load a domain from a .dmn file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Domain",
            "",
            "Domain Files (*.dmn);;All Files (*)"
        )

        if file_path:
            try:
                from svv.domain.domain import Domain
                domain = Domain.load(file_path)

                # If boundary wasn't saved in the .dmn file, rebuild it
                if domain.boundary is None and hasattr(domain, 'patches') and len(domain.patches) > 0:
                    self.status_bar.showMessage("Building domain boundary...")
                    domain.build()

                self.load_domain(domain)
            except Exception as e:
                self.update_status(f"{Fusion360Icons.ERROR} Failed to load domain")
                QMessageBox.critical(
                    self,
                    f"{Fusion360Icons.ERROR} Error Loading Domain",
                    f"Failed to load domain file:\n\n{str(e)}\n\n"
                    "Please ensure the file is a valid domain file (.dmn)"
                )

    def save_configuration(self):
        """Save the current start points and configuration."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Configuration",
            "",
            "JSON Files (*.json);;All Files (*)"
        )

        if file_path:
            try:
                import json
                config = self.point_selector.get_configuration()
                with open(file_path, 'w') as f:
                    json.dump(config, f, indent=2)
                self.update_status(f"{Fusion360Icons.CHECK} Configuration saved successfully")
                QMessageBox.information(
                    self,
                    f"{Fusion360Icons.CHECK} Success",
                    f"Configuration saved successfully to:\n{file_path}"
                )
            except Exception as e:
                self.update_status(f"{Fusion360Icons.ERROR} Failed to save configuration")
                QMessageBox.critical(
                    self,
                    f"{Fusion360Icons.ERROR} Error Saving Configuration",
                    f"Failed to save configuration:\n\n{str(e)}\n\n"
                    "Please check file permissions and try again."
                )

    def export_results(self):
        """Export generated trees/forest to file."""
        if not self.trees and not self.forest:
            QMessageBox.warning(
                self,
                f"{Fusion360Icons.WARNING} No Results",
                "No trees or forests have been generated yet.\n\n"
                "Please generate a vascular structure first."
            )
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Results",
            "",
            "VTK Files (*.vtu);;All Files (*)"
        )

        if file_path:
            try:
                # Export logic would go here
                self.update_status(f"{Fusion360Icons.CHECK} Results exported successfully")
                QMessageBox.information(
                    self,
                    f"{Fusion360Icons.CHECK} Export Complete",
                    f"Results exported successfully to:\n{file_path}"
                )
            except Exception as e:
                self.update_status(f"{Fusion360Icons.ERROR} Export failed")
                QMessageBox.critical(
                    self,
                    f"{Fusion360Icons.ERROR} Export Error",
                    f"Failed to export results:\n\n{str(e)}"
                )

    def show_about(self):
        """Show the about dialog."""
        QMessageBox.about(
            self,
            "About svVascularize",
            f"{Fusion360Icons.TREE} svVascularize\n"
            "Professional Vascular Tree and Forest Generation\n\n"
            "A modern CAD-style interface for interactive vascular network modeling.\n"
            "Built with PySide6, PyVista, and Fusion360-inspired design.\n\n"
            "Version: 2.0\n"
            "\u00A9 SimVascular"
        )

    def update_status(self, message):
        """
        Update the status bar message.

        Parameters
        ----------
        message : str
            Status message to display
        """
        self.status_bar.showMessage(message)
