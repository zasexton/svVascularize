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
from svv.visualize.gui.styles import ModernTheme, Icons


class VascularizeGUI(QMainWindow):
    """
    Main GUI window for visualizing and manipulating Domain objects
    to configure Tree and Forest vascularization.
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

        # Apply modern theme
        self.setStyleSheet(ModernTheme.get_stylesheet())

        self.setWindowTitle(f"{Icons.TREE} svVascularize - Domain Visualization")
        self.setGeometry(100, 100, 1400, 900)

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
        """Create the header widget with title and subtitle."""
        header = QWidget()
        header.setStyleSheet(f"""
            QWidget {{
                background-color: {ModernTheme.PRIMARY};
                padding: 16px;
            }}
        """)
        layout = QVBoxLayout(header)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(4)

        # Title
        title = QLabel(f"{Icons.TREE} svVascularize")
        title.setStyleSheet("""
            font-size: 20px;
            font-weight: bold;
            color: white;
        """)
        layout.addWidget(title)

        # Subtitle
        subtitle = QLabel("Interactive vascular tree and forest generation")
        subtitle.setStyleSheet("""
            font-size: 12px;
            color: rgba(255, 255, 255, 0.9);
        """)
        layout.addWidget(subtitle)

        return header

    def _create_menu_bar(self):
        """Create the application menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        load_domain_action = QAction(f"{Icons.FOLDER_OPEN} Load Domain...", self)
        load_domain_action.setShortcut(QKeySequence.Open)
        load_domain_action.setStatusTip("Load a domain mesh file")
        load_domain_action.triggered.connect(self.load_domain_dialog)
        file_menu.addAction(load_domain_action)

        save_config_action = QAction(f"{Icons.SAVE} Save Configuration...", self)
        save_config_action.setShortcut(QKeySequence.Save)
        save_config_action.setStatusTip("Save the current configuration")
        save_config_action.triggered.connect(self.save_configuration)
        file_menu.addAction(save_config_action)

        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.setStatusTip("Exit the application")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        reset_camera_action = QAction(f"{Icons.CAMERA} Reset Camera", self)
        reset_camera_action.setShortcut("R")
        reset_camera_action.setStatusTip("Reset camera to default view")
        reset_camera_action.triggered.connect(self.vtk_widget.reset_camera)
        view_menu.addAction(reset_camera_action)

        toggle_domain_action = QAction(f"{Icons.EYE} Toggle Domain Visibility", self)
        toggle_domain_action.setShortcut("D")
        toggle_domain_action.setStatusTip("Show/hide the domain mesh")
        toggle_domain_action.triggered.connect(self.vtk_widget.toggle_domain_visibility)
        view_menu.addAction(toggle_domain_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        about_action = QAction(f"{Icons.INFO} About", self)
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
        self.update_status(f"{Icons.CHECK} Domain loaded - Ready to configure trees")

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
                self.update_status(f"{Icons.ERROR} Failed to load domain")
                QMessageBox.critical(
                    self,
                    f"{Icons.ERROR} Error Loading Domain",
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
                self.update_status(f"{Icons.CHECK} Configuration saved successfully")
                QMessageBox.information(
                    self,
                    f"{Icons.CHECK} Success",
                    f"Configuration saved successfully to:\n{file_path}"
                )
            except Exception as e:
                self.update_status(f"{Icons.ERROR} Failed to save configuration")
                QMessageBox.critical(
                    self,
                    f"{Icons.ERROR} Error Saving Configuration",
                    f"Failed to save configuration:\n\n{str(e)}\n\n"
                    "Please check file permissions and try again."
                )

    def show_about(self):
        """Show the about dialog."""
        QMessageBox.about(
            self,
            "About svVascularize",
            f"{Icons.TREE} svVascularize Domain Visualization Tool\n\n"
            "Interactive GUI for configuring vascular tree and forest generation.\n\n"
            "Built with PySide6 and PyVista\n\n"
            "Version: 1.0\n"
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
