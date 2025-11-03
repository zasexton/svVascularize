"""
CAD-style main GUI window for svVascularize.
Professional engineering/fabrication interface similar to FreeCAD.
"""
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QDockWidget, QToolBar, QStatusBar, QFileDialog,
    QMessageBox, QLabel, QTreeWidget, QTreeWidgetItem,
    QSplitter, QTabWidget
)
from PySide6.QtCore import Qt, QSize, QSettings
from PySide6.QtGui import QAction, QKeySequence, QIcon
from svv.visualize.gui.vtk_widget import VTKWidget
from svv.visualize.gui.point_selector import PointSelectorWidget
from svv.visualize.gui.parameter_panel import ParameterPanel
from svv.visualize.gui.theme import CADTheme, CADIcons


class ObjectBrowserWidget(QTreeWidget):
    """
    Object/model tree browser similar to FreeCAD's tree view.
    Shows domain, trees, forests, and their properties.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent

        self.setHeaderLabels(["Model Tree"])
        self.setAlternatingRowColors(True)

        # Root items
        self.domain_item = None
        self.trees_item = None
        self.forests_item = None
        self.points_item = None

        self._init_tree()

    def _init_tree(self):
        """Initialize the tree structure."""
        # Scene root
        self.scene_root = QTreeWidgetItem(self, ["Scene"])
        self.scene_root.setExpanded(True)

        # Domain
        self.domain_item = QTreeWidgetItem(self.scene_root, [f"{CADIcons.MESH} Domain"])
        self.domain_item.setCheckState(0, Qt.Checked)

        # Start Points
        self.points_item = QTreeWidgetItem(self.scene_root, [f"{CADIcons.POINT} Start Points"])
        self.points_item.setExpanded(True)

        # Trees
        self.trees_item = QTreeWidgetItem(self.scene_root, [f"{CADIcons.TREE} Trees"])
        self.trees_item.setExpanded(True)

        # Forests
        self.forests_item = QTreeWidgetItem(self.scene_root, [f"{CADIcons.FOREST} Forests"])
        self.forests_item.setExpanded(True)

        # Connect item changes
        self.itemChanged.connect(self._on_item_changed)

    def _on_item_changed(self, item, column):
        """Handle item visibility changes."""
        if item == self.domain_item and self.main_window:
            checked = item.checkState(0) == Qt.Checked
            if hasattr(self.main_window, 'vtk_widget'):
                if checked:
                    self.main_window.vtk_widget.domain_actor.SetVisibility(True)
                else:
                    self.main_window.vtk_widget.domain_actor.SetVisibility(False)
                self.main_window.vtk_widget.plotter.render()

    def add_point(self, point_id, point_data):
        """Add a point to the tree."""
        point_item = QTreeWidgetItem(
            self.points_item,
            [f"{CADIcons.POINT} Point {point_id}"]
        )
        point_item.setCheckState(0, Qt.Checked)
        return point_item

    def add_tree(self, tree_id, n_vessels):
        """Add a tree to the tree view."""
        tree_item = QTreeWidgetItem(
            self.trees_item,
            [f"{CADIcons.TREE} Tree {tree_id} ({n_vessels} vessels)"]
        )
        tree_item.setCheckState(0, Qt.Checked)
        return tree_item

    def add_forest(self, forest_id, n_networks):
        """Add a forest to the tree view."""
        forest_item = QTreeWidgetItem(
            self.forests_item,
            [f"{CADIcons.FOREST} Forest {forest_id} ({n_networks} networks)"]
        )
        forest_item.setCheckState(0, Qt.Checked)
        return forest_item

    def clear_trees(self):
        """Clear all trees from view."""
        self.trees_item.takeChildren()

    def clear_forests(self):
        """Clear all forests from view."""
        self.forests_item.takeChildren()

    def clear_points(self):
        """Clear all points from view."""
        self.points_item.takeChildren()


class VascularizeCADGUI(QMainWindow):
    """
    CAD-style main GUI window for visualizing and manipulating Domain objects.
    Professional interface for fabrication/engineering users.
    """

    def __init__(self, domain=None):
        """
        Initialize the CAD-style GUI window.

        Parameters
        ----------
        domain : svv.domain.Domain, optional
            Initial domain object to visualize
        """
        super().__init__()
        self.domain = domain
        self.trees = []
        self.forest = None

        # Initialize QSettings for persistent layout
        self.settings = QSettings("SimVascular", "svVascularize")

        # Apply CAD theme
        self.setStyleSheet(CADTheme.get_stylesheet())

        self.setWindowTitle("svVascularize - Vascular CAD")
        self.setGeometry(100, 100, 1600, 1000)

        # Create central 3D viewport first
        self._init_viewport()

        # Create toolbars
        self._create_toolbars()

        # Create dockable panels
        self._create_dock_widgets()

        # Create menu bar
        self._create_menu_bar()

        # Create status bar
        self._create_status_bar()

        # Restore window geometry and dock layout from settings
        self._restore_layout()

        # Load domain if provided
        if domain is not None:
            self.load_domain(domain)

    def _init_viewport(self):
        """Initialize the central 3D viewport."""
        self.vtk_widget = VTKWidget(self)
        self.setCentralWidget(self.vtk_widget)

    def _create_toolbars(self):
        """Create CAD-standard toolbars."""
        # File Toolbar
        self.file_toolbar = QToolBar("File")
        self.file_toolbar.setIconSize(QSize(24, 24))
        self.file_toolbar.setMovable(True)
        self.file_toolbar.setFloatable(False)
        self.addToolBar(Qt.TopToolBarArea, self.file_toolbar)

        # Open action
        self.action_open = QAction(f"{CADIcons.OPEN}", self)
        self.action_open.setShortcut(QKeySequence.Open)
        self.action_open.setStatusTip("Load domain mesh file")
        self.action_open.setToolTip("Load Domain (Ctrl+O)")
        self.action_open.triggered.connect(self.load_domain_dialog)
        self.file_toolbar.addAction(self.action_open)

        # Save action
        self.action_save = QAction(f"{CADIcons.SAVE}", self)
        self.action_save.setShortcut(QKeySequence.Save)
        self.action_save.setStatusTip("Save configuration")
        self.action_save.setToolTip("Save Configuration (Ctrl+S)")
        self.action_save.triggered.connect(self.save_configuration)
        self.file_toolbar.addAction(self.action_save)

        # Export action
        self.action_export = QAction(f"{CADIcons.EXPORT}", self)
        self.action_export.setStatusTip("Export generated vasculature")
        self.action_export.setToolTip("Export Results")
        self.file_toolbar.addAction(self.action_export)

        self.file_toolbar.addSeparator()

        # View Toolbar
        self.view_toolbar = QToolBar("View")
        self.view_toolbar.setIconSize(QSize(24, 24))
        self.view_toolbar.setMovable(True)
        self.view_toolbar.setFloatable(False)
        self.addToolBar(Qt.TopToolBarArea, self.view_toolbar)

        # Fit view
        self.action_fit = QAction(f"{CADIcons.VIEW_FIT}", self)
        self.action_fit.setShortcut("V, F")
        self.action_fit.setStatusTip("Fit all objects in view")
        self.action_fit.setToolTip("Fit View (V, F)")
        self.action_fit.triggered.connect(self.vtk_widget.reset_camera)
        self.view_toolbar.addAction(self.action_fit)

        # Isometric view
        self.action_iso = QAction(f"{CADIcons.VIEW_ISO}", self)
        self.action_iso.setShortcut("V, I")
        self.action_iso.setStatusTip("Isometric view")
        self.action_iso.setToolTip("Isometric View (V, I)")
        self.view_toolbar.addAction(self.action_iso)

        # Top view
        self.action_top = QAction(f"{CADIcons.VIEW_TOP}", self)
        self.action_top.setShortcut("V, T")
        self.action_top.setStatusTip("Top view")
        self.action_top.setToolTip("Top View (V, T)")
        self.view_toolbar.addAction(self.action_top)

        # Front view
        self.action_front = QAction(f"{CADIcons.VIEW_FRONT}", self)
        self.action_front.setShortcut("V, 1")
        self.action_front.setStatusTip("Front view")
        self.action_front.setToolTip("Front View (V, 1)")
        self.view_toolbar.addAction(self.action_front)

        # Right view
        self.action_right = QAction(f"{CADIcons.VIEW_RIGHT}", self)
        self.action_right.setShortcut("V, 3")
        self.action_right.setStatusTip("Right view")
        self.action_right.setToolTip("Right View (V, 3)")
        self.view_toolbar.addAction(self.action_right)

        self.view_toolbar.addSeparator()

        # Toggle domain visibility
        self.action_toggle_domain = QAction(f"{CADIcons.VISIBLE}", self)
        self.action_toggle_domain.setCheckable(True)
        self.action_toggle_domain.setChecked(True)
        self.action_toggle_domain.setShortcut("D")
        self.action_toggle_domain.setStatusTip("Toggle domain visibility")
        self.action_toggle_domain.setToolTip("Toggle Domain (D)")
        self.action_toggle_domain.triggered.connect(self.vtk_widget.toggle_domain_visibility)
        self.view_toolbar.addAction(self.action_toggle_domain)

        # Generation Toolbar
        self.gen_toolbar = QToolBar("Generation")
        self.gen_toolbar.setIconSize(QSize(24, 24))
        self.gen_toolbar.setMovable(True)
        self.gen_toolbar.setFloatable(False)
        self.addToolBar(Qt.TopToolBarArea, self.gen_toolbar)

        # Add point tool
        self.action_add_point = QAction(f"{CADIcons.POINT}", self)
        self.action_add_point.setCheckable(True)
        self.action_add_point.setShortcut("P")
        self.action_add_point.setStatusTip("Add start point")
        self.action_add_point.setToolTip("Add Start Point (P)")
        self.gen_toolbar.addAction(self.action_add_point)

        # Add direction tool
        self.action_add_vector = QAction(f"{CADIcons.VECTOR}", self)
        self.action_add_vector.setStatusTip("Add direction vector")
        self.action_add_vector.setToolTip("Add Direction Vector")
        self.gen_toolbar.addAction(self.action_add_vector)

        self.gen_toolbar.addSeparator()

        # Generate action
        self.action_generate = QAction(f"{CADIcons.GENERATE}", self)
        self.action_generate.setShortcut("G")
        self.action_generate.setStatusTip("Generate vascular tree/forest")
        self.action_generate.setToolTip("Generate (G)")
        self.gen_toolbar.addAction(self.action_generate)

    def _create_dock_widgets(self):
        """Create dockable panels (CAD-style)."""
        # Model Tree / Object Browser (Left)
        self.tree_dock = QDockWidget("Model Tree", self)
        self.tree_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        self.object_browser = ObjectBrowserWidget(self)
        self.tree_dock.setWidget(self.object_browser)

        self.addDockWidget(Qt.LeftDockWidgetArea, self.tree_dock)

        # Properties / Parameters Panel (Right)
        self.properties_dock = QDockWidget("Properties", self)
        self.properties_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        # Create tabbed interface for properties
        self.properties_tabs = QTabWidget()

        # Generation parameters tab
        self.parameter_panel = ParameterPanel(self)
        self.properties_tabs.addTab(self.parameter_panel, f"{CADIcons.SETTINGS} Generation")

        # Point selector tab
        self.point_selector = PointSelectorWidget(self)
        self.properties_tabs.addTab(self.point_selector, f"{CADIcons.POINT} Start Points")

        self.properties_dock.setWidget(self.properties_tabs)
        self.addDockWidget(Qt.RightDockWidgetArea, self.properties_dock)

        # Info Panel (Bottom - initially hidden)
        self.info_dock = QDockWidget("Information", self)
        self.info_dock.setAllowedAreas(Qt.BottomDockWidgetArea)

        self.info_widget = QLabel("Domain statistics and generation info will appear here.")
        self.info_widget.setWordWrap(True)
        self.info_widget.setStyleSheet("padding: 8px;")
        self.info_dock.setWidget(self.info_widget)

        self.addDockWidget(Qt.BottomDockWidgetArea, self.info_dock)
        self.info_dock.hide()  # Hidden by default

    def _create_menu_bar(self):
        """Create the menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        file_menu.addAction(self.action_open)
        file_menu.addAction(self.action_save)
        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.setStatusTip("Exit application")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        view_menu.addAction(self.action_fit)
        view_menu.addAction(self.action_iso)
        view_menu.addAction(self.action_top)
        view_menu.addAction(self.action_front)
        view_menu.addAction(self.action_right)
        view_menu.addSeparator()
        view_menu.addAction(self.action_toggle_domain)
        view_menu.addSeparator()

        # Panel visibility submenu
        panels_menu = view_menu.addMenu("&Panels")
        panels_menu.addAction(self.tree_dock.toggleViewAction())
        panels_menu.addAction(self.properties_dock.toggleViewAction())
        panels_menu.addAction(self.info_dock.toggleViewAction())

        view_menu.addSeparator()

        # Toolbar visibility submenu
        toolbars_menu = view_menu.addMenu("&Toolbars")
        toolbars_menu.addAction(self.file_toolbar.toggleViewAction())
        toolbars_menu.addAction(self.view_toolbar.toggleViewAction())
        toolbars_menu.addAction(self.gen_toolbar.toggleViewAction())

        view_menu.addSeparator()

        # Layout management
        save_layout_action = QAction("Save Layout", self)
        save_layout_action.setStatusTip("Save current window layout")
        save_layout_action.triggered.connect(self._save_layout)
        view_menu.addAction(save_layout_action)

        reset_layout_action = QAction("Reset Layout", self)
        reset_layout_action.setStatusTip("Reset window layout to defaults")
        reset_layout_action.triggered.connect(self._reset_layout)
        view_menu.addAction(reset_layout_action)

        # Generation menu
        gen_menu = menubar.addMenu("&Generate")
        gen_menu.addAction(self.action_add_point)
        gen_menu.addAction(self.action_add_vector)
        gen_menu.addSeparator()
        gen_menu.addAction(self.action_generate)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        about_action = QAction(f"{CADIcons.INFO} About", self)
        about_action.setStatusTip("About svVascularize")
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def _create_status_bar(self):
        """Create CAD-style status bar with multiple sections."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Main status message
        self.status_message = QLabel("Ready")
        self.status_bar.addWidget(self.status_message, 1)

        # Viewport info
        self.status_viewport = QLabel("View: Perspective")
        self.status_bar.addPermanentWidget(self.status_viewport)

        # Object count
        self.status_objects = QLabel("Objects: 0")
        self.status_bar.addPermanentWidget(self.status_objects)

        # Vertex/element count
        self.status_elements = QLabel("Vessels: 0")
        self.status_bar.addPermanentWidget(self.status_elements)

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

        # Update object browser
        if self.object_browser.domain_item:
            self.object_browser.domain_item.setText(0, f"{CADIcons.MESH} Domain (loaded)")

        # Update info panel
        self.info_widget.setText(
            f"<b>Domain Information</b><br>"
            f"File: {getattr(domain, 'filename', 'Unknown')}<br>"
            f"Characteristic Length: {getattr(domain, 'characteristic_length', 'N/A')}<br>"
            f"Status: Ready for vascular generation"
        )
        self.info_dock.show()

        self.update_status(f"{CADIcons.SUCCESS} Domain loaded successfully")
        self.update_object_count()

    def load_domain_dialog(self):
        """Open file dialog to load domain."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Domain",
            "",
            "Domain Files (*.dmn *.vtu);;All Files (*)"
        )

        if file_path:
            try:
                from svv.domain.domain import Domain
                domain = Domain.load(file_path)

                if domain.boundary is None and hasattr(domain, 'patches') and len(domain.patches) > 0:
                    self.update_status("Building domain boundary...")
                    domain.build()

                domain.filename = file_path
                self.load_domain(domain)
            except Exception as e:
                self.update_status(f"{CADIcons.ERROR} Failed to load domain")
                QMessageBox.critical(
                    self,
                    "Error Loading Domain",
                    f"Failed to load domain:\n\n{str(e)}"
                )

    def save_configuration(self):
        """Save current configuration."""
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
                self.update_status(f"{CADIcons.SUCCESS} Configuration saved")
            except Exception as e:
                self.update_status(f"{CADIcons.ERROR} Save failed")
                QMessageBox.critical(
                    self,
                    "Error Saving",
                    f"Failed to save:\n\n{str(e)}"
                )

    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About svVascularize",
            f"{CADIcons.TREE} <b>svVascularize - Vascular CAD</b><br><br>"
            "Professional vascular tree and forest generation tool<br>"
            "CAD-style interface for engineering and fabrication<br><br>"
            "Built with PySide6 and PyVista<br>"
            "Version 1.0<br><br>"
            "© SimVascular"
        )

    def update_status(self, message):
        """Update status bar message."""
        self.status_message.setText(message)

    def update_object_count(self):
        """Update object count in status bar."""
        count = 0
        if self.domain:
            count += 1
        count += len(self.trees)
        if self.forest:
            count += 1

        self.status_objects.setText(f"Objects: {count}")

    def update_vessel_count(self, count):
        """Update vessel count in status bar."""
        self.status_elements.setText(f"Vessels: {count}")

    def _restore_layout(self):
        """Restore window geometry and dock widget layout from QSettings."""
        # Restore window geometry
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)

        # Restore window state (dock positions, toolbar positions, etc.)
        window_state = self.settings.value("windowState")
        if window_state:
            self.restoreState(window_state)

    def _save_layout(self):
        """Save current window geometry and dock widget layout to QSettings."""
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
        self.update_status(f"{CADIcons.SUCCESS} Layout saved")

    def _reset_layout(self):
        """Reset window layout to default configuration."""
        # Clear saved settings
        self.settings.remove("geometry")
        self.settings.remove("windowState")

        # Reset to default geometry
        self.setGeometry(100, 100, 1600, 1000)

        # Reset dock positions
        self.addDockWidget(Qt.LeftDockWidgetArea, self.tree_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.properties_dock)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.info_dock)

        # Show/hide default panels
        self.tree_dock.show()
        self.properties_dock.show()
        self.info_dock.hide()

        # Reset toolbar positions
        self.addToolBar(Qt.TopToolBarArea, self.file_toolbar)
        self.addToolBar(Qt.TopToolBarArea, self.view_toolbar)
        self.addToolBar(Qt.TopToolBarArea, self.gen_toolbar)

        self.update_status(f"{CADIcons.SUCCESS} Layout reset to defaults")

    def closeEvent(self, event):
        """
        Handle window close event - save layout before closing.

        Parameters
        ----------
        event : QCloseEvent
            The close event
        """
        # Save current layout
        self._save_layout()

        # Accept the close event
        event.accept()
