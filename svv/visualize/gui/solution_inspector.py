from __future__ import annotations

import os
from typing import Optional, Sequence

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QGroupBox,
    QFormLayout,
    QLineEdit,
    QLabel,
    QPushButton,
    QSlider,
    QComboBox,
    QDoubleSpinBox,
    QSpinBox,
    QFileDialog,
    QCheckBox,
    QDockWidget,
    QMenuBar,
    QToolBar,
    QToolButton,
    QFrame,
    QScrollArea,
    QSizePolicy,
    QButtonGroup,
    QGridLayout,
    QMessageBox,
)
from PySide6.QtGui import QAction, QIcon

from svv.visualize.gui.vtk_widget import VTKWidget
from svv.visualize.gui.theme import CADTheme, CADIcons
from svv.telemetry import capture_exception, capture_message


class CollapsibleGroupBox(QGroupBox):
    """
    A QGroupBox that can be collapsed/expanded by clicking on its title.
    """

    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(title, parent)
        self.setCheckable(True)
        self.setChecked(True)
        self.toggled.connect(self._on_toggled)
        self._content_height = 0

    def _on_toggled(self, checked: bool) -> None:
        """Handle toggle to show/hide contents."""
        for i in range(self.layout().count()) if self.layout() else []:
            widget = self.layout().itemAt(i).widget()
            if widget:
                widget.setVisible(checked)


class SolutionInspectorWidget(QWidget):
    """
    Dockable widget for inspecting 0D/ROM solution time series.

    Provides an embedded 3D viewport and a side panel for loading a
    ``.pvd`` file, selecting time steps, and adjusting scalar display
    options (field, range, and label).

    Features:
    - Animation playback with speed control
    - Mesh display options (opacity, representation, edges)
    - Data probing and statistics
    - Slicing and threshold filtering
    - Camera presets
    - Vector field visualization
    - Export capabilities
    """

    # Signals
    probed_value_changed = Signal(float, float, float, float)  # x, y, z, value

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._pv = None
        self._reader = None
        self._current_mesh = None
        self._current_scalar_name: Optional[str] = None
        self._time_values: Sequence[float] = []
        self._current_time_index: int = 0
        self._current_cmap: str = "coolwarm"
        self._global_ranges = {}
        self._colorbar_visible: bool = True

        # Animation state
        self._animation_timer: Optional[QTimer] = None
        self._animation_playing: bool = False
        self._animation_loop: bool = True
        self._animation_fps: float = 10.0

        # Mesh display state
        self._mesh_opacity: float = 1.0
        self._mesh_representation: str = "Surface"
        self._mesh_show_edges: bool = False

        # Probing state
        self._probe_enabled: bool = False
        self._probed_point: Optional[tuple] = None
        self._probe_actor = None
        self._time_series_data: dict = {}  # point_key -> [(time, value), ...]

        # Slicing state
        self._slice_enabled: bool = False
        self._slice_actor = None
        self._slice_widget = None

        # Threshold state
        self._threshold_enabled: bool = False
        self._threshold_actor = None

        # Vector field state
        self._vector_field_name: Optional[str] = None
        self._glyph_actor = None
        self._glyph_scale: float = 1.0

        self._try_import_pyvista()
        self._build_ui()

    def _record_telemetry(self, exc=None, message: str | None = None, level: str = "error", **tags) -> None:
        """
        Send errors or warnings to telemetry without impacting the UI.

        This mirrors the lightweight helpers used elsewhere in the GUI so
        that any error which results in a popup can also be surfaced to
        Sentry when telemetry is enabled.
        """
        try:
            if exc is not None:
                try:
                    import sentry_sdk  # type: ignore[import]

                    with sentry_sdk.push_scope() as scope:
                        for key, value in tags.items():
                            scope.set_tag(key, value)
                        sentry_sdk.capture_exception(exc)
                except Exception:
                    capture_exception(exc)
                return
            if message:
                capture_message(message, level=level, **tags)
        except Exception:
            # Telemetry must never break the inspector UI
            pass

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------
    def _try_import_pyvista(self) -> None:
        try:
            import pyvista as pv  # type: ignore[import]

            self._pv = pv
        except Exception:
            self._pv = None

    def _build_ui(self) -> None:
        # Outer layout: menu-style toolbar + main content
        outer_layout = QVBoxLayout()
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)
        self.setLayout(outer_layout)

        # --- Menu toolbar: File / View / Tools + close button ---
        title_bar = QHBoxLayout()
        title_bar.setContentsMargins(4, 4, 4, 4)
        title_bar.setSpacing(6)

        menu_bar = QMenuBar(self)

        # File menu
        file_menu = menu_bar.addMenu("File")
        open_action = QAction(CADIcons.get_icon('open'), "Open...", self)
        open_action.setStatusTip("Open a time-varying solution (.pvd) file")
        open_action.triggered.connect(self._browse_for_file)
        file_menu.addAction(open_action)
        file_menu.addSeparator()

        screenshot_action = QAction(CADIcons.get_icon('save'), "Screenshot...", self)
        screenshot_action.setStatusTip("Save a screenshot of the current 3D view")
        screenshot_action.triggered.connect(self._take_screenshot)
        file_menu.addAction(screenshot_action)

        file_menu.addSeparator()

        # Export submenu
        export_menu = file_menu.addMenu("Export")
        export_mesh_action = QAction(CADIcons.get_icon('mesh'), "Export Mesh (VTP/VTU)...", self)
        export_mesh_action.triggered.connect(self._export_mesh)
        export_menu.addAction(export_mesh_action)

        export_csv_action = QAction(CADIcons.get_icon('download'), "Export Scalars (CSV)...", self)
        export_csv_action.triggered.connect(self._export_csv)
        export_menu.addAction(export_csv_action)

        export_animation_action = QAction(CADIcons.get_icon('movie'), "Export Animation (GIF)...", self)
        export_animation_action.triggered.connect(self._export_animation)
        export_menu.addAction(export_animation_action)

        export_timeseries_action = QAction(CADIcons.get_icon('chart'), "Export Time Series (CSV)...", self)
        export_timeseries_action.triggered.connect(self._export_time_series)
        export_menu.addAction(export_timeseries_action)

        # View menu
        view_menu = menu_bar.addMenu("View")
        reset_view_action = QAction(CADIcons.get_icon('fit'), "Reset View", self)
        reset_view_action.setStatusTip("Reset camera to show entire solution")
        reset_view_action.triggered.connect(self._reset_camera_view)
        view_menu.addAction(reset_view_action)

        iso_view_action = QAction(CADIcons.get_icon('iso'), "Isometric View", self)
        iso_view_action.setStatusTip("Isometric 3D view")
        iso_view_action.triggered.connect(self._view_iso)
        view_menu.addAction(iso_view_action)

        view_menu.addSeparator()

        # Camera preset submenu
        camera_menu = view_menu.addMenu("Camera Presets")
        for label, method in [
            ("+X View", self._view_plus_x),
            ("-X View", self._view_minus_x),
            ("+Y View", self._view_plus_y),
            ("-Y View", self._view_minus_y),
            ("+Z View", self._view_plus_z),
            ("-Z View", self._view_minus_z),
        ]:
            action = QAction(label, self)
            action.triggered.connect(method)
            camera_menu.addAction(action)

        view_menu.addSeparator()

        self.scale_bar_action = QAction(CADIcons.get_icon('ruler'), "Scale Bar", self)
        self.scale_bar_action.setCheckable(True)
        self.scale_bar_action.setChecked(True)
        self.scale_bar_action.setStatusTip("Toggle visibility of the scale bar in this viewport")
        self.scale_bar_action.triggered.connect(self._toggle_scale_bar)
        view_menu.addAction(self.scale_bar_action)

        self.color_bar_action = QAction("Color Bar", self)
        self.color_bar_action.setCheckable(True)
        self.color_bar_action.setChecked(True)
        self.color_bar_action.setStatusTip("Toggle visibility of the scalar color bar")
        self.color_bar_action.triggered.connect(self._toggle_color_bar)
        view_menu.addAction(self.color_bar_action)

        view_menu.addSeparator()

        self.parallel_proj_action = QAction("Parallel Projection", self)
        self.parallel_proj_action.setCheckable(True)
        self.parallel_proj_action.setChecked(False)
        self.parallel_proj_action.setStatusTip("Toggle between perspective and parallel projection")
        self.parallel_proj_action.triggered.connect(self._toggle_parallel_projection)
        view_menu.addAction(self.parallel_proj_action)

        # Tools menu
        tools_menu = menu_bar.addMenu("Tools")

        self.probe_action = QAction(CADIcons.get_icon('probe'), "Probe Mode", self)
        self.probe_action.setCheckable(True)
        self.probe_action.setStatusTip("Enable point probing to inspect values")
        self.probe_action.triggered.connect(self._toggle_probe_mode)
        tools_menu.addAction(self.probe_action)

        tools_menu.addSeparator()

        self.slice_action = QAction(CADIcons.get_icon('slice'), "Plane Slice...", self)
        self.slice_action.setCheckable(True)
        self.slice_action.setStatusTip("Add a cutting plane to slice the mesh")
        self.slice_action.triggered.connect(self._toggle_slice)
        tools_menu.addAction(self.slice_action)

        self.threshold_action = QAction(CADIcons.get_icon('filter'), "Threshold Filter...", self)
        self.threshold_action.setCheckable(True)
        self.threshold_action.setStatusTip("Filter mesh by scalar value range")
        self.threshold_action.triggered.connect(self._toggle_threshold)
        tools_menu.addAction(self.threshold_action)

        title_bar.addWidget(menu_bar, 1)

        close_btn = QPushButton("X")
        close_btn.setFixedSize(20, 20)
        close_btn.setToolTip("Hide Solution Inspector")
        close_btn.clicked.connect(self._request_close)
        title_bar.addWidget(close_btn, 0, alignment=Qt.AlignRight | Qt.AlignVCenter)

        outer_layout.addLayout(title_bar, 0)

        # --- Main content: 3D viewport + properties panel ---
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        outer_layout.addLayout(main_layout, 1)

        # Left: 3D viewport
        self.vtk_widget = VTKWidget(self)
        main_layout.addWidget(self.vtk_widget, stretch=3)

        # Right: scrollable properties panel
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setMinimumWidth(280)
        scroll_area.setMaximumWidth(350)

        side_panel = QWidget(scroll_area)
        side_layout = QVBoxLayout()
        side_layout.setContentsMargins(8, 8, 8, 8)
        side_layout.setSpacing(6)
        side_panel.setLayout(side_layout)
        scroll_area.setWidget(side_panel)

        # --- Solution file display ---
        file_group = QGroupBox("Solution File", side_panel)
        file_layout = QVBoxLayout()
        file_layout.setSpacing(4)
        file_group.setLayout(file_layout)

        self.file_edit = QLineEdit()
        self.file_edit.setReadOnly(True)
        self.file_edit.setPlaceholderText("No solution loaded")
        file_layout.addWidget(self.file_edit)

        side_layout.addWidget(file_group)

        # --- Animation / Playback controls ---
        anim_group = QGroupBox("Playback", side_panel)
        anim_layout = QVBoxLayout()
        anim_layout.setSpacing(4)
        anim_group.setLayout(anim_layout)

        # Time slider
        self.time_label = QLabel("Time: --")
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setRange(0, 0)
        self.time_slider.setEnabled(False)
        self.time_slider.valueChanged.connect(self._on_time_index_changed)
        anim_layout.addWidget(self.time_label)
        anim_layout.addWidget(self.time_slider)

        # Playback buttons
        playback_layout = QHBoxLayout()
        playback_layout.setSpacing(2)

        self.play_btn = QToolButton()
        self.play_btn.setIcon(CADIcons.get_icon('play'))
        self.play_btn.setToolTip("Play animation")
        self.play_btn.clicked.connect(self._toggle_animation)
        playback_layout.addWidget(self.play_btn)

        self.stop_btn = QToolButton()
        self.stop_btn.setIcon(CADIcons.get_icon('stop'))
        self.stop_btn.setToolTip("Stop and reset to start")
        self.stop_btn.clicked.connect(self._stop_animation)
        playback_layout.addWidget(self.stop_btn)

        self.loop_btn = QToolButton()
        self.loop_btn.setIcon(CADIcons.get_icon('loop'))
        self.loop_btn.setCheckable(True)
        self.loop_btn.setChecked(True)
        self.loop_btn.setToolTip("Loop animation")
        self.loop_btn.toggled.connect(self._on_loop_toggled)
        playback_layout.addWidget(self.loop_btn)

        playback_layout.addStretch()

        # Speed control
        playback_layout.addWidget(QLabel("FPS:"))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(10)
        self.fps_spin.setToolTip("Animation frames per second")
        self.fps_spin.valueChanged.connect(self._on_fps_changed)
        playback_layout.addWidget(self.fps_spin)

        anim_layout.addLayout(playback_layout)
        side_layout.addWidget(anim_group)

        # --- Mesh Display options ---
        display_group = QGroupBox("Mesh Display", side_panel)
        display_layout = QFormLayout()
        display_layout.setSpacing(4)
        display_group.setLayout(display_layout)

        # Opacity
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.setToolTip("Mesh opacity (0-100%)")
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        display_layout.addRow("Opacity:", self.opacity_slider)

        # Representation
        self.repr_combo = QComboBox()
        self.repr_combo.addItems(["Surface", "Wireframe", "Points", "Surface + Edges"])
        self.repr_combo.currentTextChanged.connect(self._on_representation_changed)
        display_layout.addRow("Style:", self.repr_combo)

        side_layout.addWidget(display_group)

        # --- Scalar Field controls ---
        scalar_group = QGroupBox("Scalar Field", side_panel)
        scalar_layout = QFormLayout()
        scalar_layout.setSpacing(4)
        scalar_group.setLayout(scalar_layout)

        self.scalar_combo = QComboBox()
        self.scalar_combo.currentTextChanged.connect(self._on_scalar_changed)
        scalar_layout.addRow("Field:", self.scalar_combo)

        # Vector component selector (for vector fields)
        self.component_combo = QComboBox()
        self.component_combo.addItems(["Magnitude", "X", "Y", "Z"])
        self.component_combo.setEnabled(False)
        self.component_combo.currentTextChanged.connect(self._on_component_changed)
        scalar_layout.addRow("Component:", self.component_combo)

        self.auto_range_cb = QCheckBox("Auto range")
        self.auto_range_cb.setChecked(True)
        self.auto_range_cb.stateChanged.connect(self._on_auto_range_toggled)
        scalar_layout.addRow("", self.auto_range_cb)

        self.scalar_min_spin = QDoubleSpinBox()
        self.scalar_min_spin.setDecimals(4)
        self.scalar_min_spin.setRange(-1e12, 1e12)
        self.scalar_min_spin.valueChanged.connect(self._on_clim_changed)

        self.scalar_max_spin = QDoubleSpinBox()
        self.scalar_max_spin.setDecimals(4)
        self.scalar_max_spin.setRange(-1e12, 1e12)
        self.scalar_max_spin.valueChanged.connect(self._on_clim_changed)

        scalar_layout.addRow("Min:", self.scalar_min_spin)
        scalar_layout.addRow("Max:", self.scalar_max_spin)

        self.scalar_label_edit = QLineEdit()
        self.scalar_label_edit.setPlaceholderText("Colorbar label (optional)")
        self.scalar_label_edit.editingFinished.connect(self._on_scalar_label_changed)
        scalar_layout.addRow("Label:", self.scalar_label_edit)

        side_layout.addWidget(scalar_group)

        # --- Colorbar controls ---
        colorbar_group = QGroupBox("Colorbar", side_panel)
        colorbar_layout = QFormLayout()
        colorbar_layout.setSpacing(4)
        colorbar_group.setLayout(colorbar_layout)

        # Colormap selection
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems([
            "coolwarm", "viridis", "plasma", "magma", "inferno",
            "cividis", "RdBu", "Spectral", "Greys", "jet", "rainbow",
        ])
        self.cmap_combo.setCurrentText(self._current_cmap)
        self.cmap_combo.currentTextChanged.connect(self._on_cmap_changed)
        colorbar_layout.addRow("Colormap:", self.cmap_combo)

        # Global normalization over time
        self.global_range_cb = QCheckBox("Normalize over time")
        self.global_range_cb.setChecked(False)
        self.global_range_cb.setToolTip("Use global min/max across all time steps")
        self.global_range_cb.stateChanged.connect(self._on_global_range_toggled)
        colorbar_layout.addRow("", self.global_range_cb)

        # Orientation
        self.orientation_combo = QComboBox()
        self.orientation_combo.addItems(["Vertical", "Horizontal"])
        self.orientation_combo.setCurrentText("Vertical")
        self.orientation_combo.currentTextChanged.connect(self._on_orientation_changed)
        colorbar_layout.addRow("Orientation:", self.orientation_combo)

        # Width
        self.colorbar_width_spin = QDoubleSpinBox()
        self.colorbar_width_spin.setRange(0.01, 0.5)
        self.colorbar_width_spin.setValue(0.05)
        self.colorbar_width_spin.setSingleStep(0.01)
        self.colorbar_width_spin.setDecimals(2)
        self.colorbar_width_spin.setToolTip("Colorbar width (fraction of window)")
        self.colorbar_width_spin.valueChanged.connect(self._on_colorbar_layout_changed)
        colorbar_layout.addRow("Width:", self.colorbar_width_spin)

        # Height
        self.colorbar_height_spin = QDoubleSpinBox()
        self.colorbar_height_spin.setRange(0.1, 1.0)
        self.colorbar_height_spin.setValue(0.8)
        self.colorbar_height_spin.setSingleStep(0.05)
        self.colorbar_height_spin.setDecimals(2)
        self.colorbar_height_spin.setToolTip("Colorbar height (fraction of window)")
        self.colorbar_height_spin.valueChanged.connect(self._on_colorbar_layout_changed)
        colorbar_layout.addRow("Height:", self.colorbar_height_spin)

        # Number of tick labels
        self.colorbar_nlabels_spin = QSpinBox()
        self.colorbar_nlabels_spin.setRange(2, 20)
        self.colorbar_nlabels_spin.setValue(5)
        self.colorbar_nlabels_spin.setToolTip("Number of tick labels on colorbar")
        self.colorbar_nlabels_spin.valueChanged.connect(self._on_colorbar_layout_changed)
        colorbar_layout.addRow("Tick Labels:", self.colorbar_nlabels_spin)

        # Number format
        self.colorbar_fmt_combo = QComboBox()
        self.colorbar_fmt_combo.addItems([
            "Auto (%.3g)",
            "Decimal (%.2f)",
            "Decimal (%.4f)",
            "Scientific (%.2e)",
            "Scientific (%.3e)",
            "Integer (%d)",
            "Percent (%.1f%%)",
        ])
        self.colorbar_fmt_combo.setCurrentIndex(0)
        self.colorbar_fmt_combo.setToolTip("Number format for tick labels")
        self.colorbar_fmt_combo.currentIndexChanged.connect(self._on_colorbar_layout_changed)
        colorbar_layout.addRow("Format:", self.colorbar_fmt_combo)

        # Normalize display to [0, 1]
        self.colorbar_normalize_cb = QCheckBox("Normalize to [0, 1]")
        self.colorbar_normalize_cb.setChecked(False)
        self.colorbar_normalize_cb.setToolTip("Display colorbar scale as normalized [0, 1] range")
        self.colorbar_normalize_cb.stateChanged.connect(self._on_colorbar_layout_changed)
        colorbar_layout.addRow("", self.colorbar_normalize_cb)

        # Position X
        self.colorbar_pos_x_spin = QDoubleSpinBox()
        self.colorbar_pos_x_spin.setRange(0.0, 0.95)
        self.colorbar_pos_x_spin.setValue(0.85)
        self.colorbar_pos_x_spin.setSingleStep(0.05)
        self.colorbar_pos_x_spin.setDecimals(2)
        self.colorbar_pos_x_spin.setToolTip("Colorbar X position (0=left, 1=right)")
        self.colorbar_pos_x_spin.valueChanged.connect(self._on_colorbar_layout_changed)
        colorbar_layout.addRow("Position X:", self.colorbar_pos_x_spin)

        # Position Y
        self.colorbar_pos_y_spin = QDoubleSpinBox()
        self.colorbar_pos_y_spin.setRange(0.0, 0.95)
        self.colorbar_pos_y_spin.setValue(0.1)
        self.colorbar_pos_y_spin.setSingleStep(0.05)
        self.colorbar_pos_y_spin.setDecimals(2)
        self.colorbar_pos_y_spin.setToolTip("Colorbar Y position (0=bottom, 1=top)")
        self.colorbar_pos_y_spin.valueChanged.connect(self._on_colorbar_layout_changed)
        colorbar_layout.addRow("Position Y:", self.colorbar_pos_y_spin)

        # Font family
        self.colorbar_font_combo = QComboBox()
        self.colorbar_font_combo.addItems([
            "Arial",
            "Courier",
            "Times",
        ])
        self.colorbar_font_combo.setCurrentText("Arial")
        self.colorbar_font_combo.setToolTip("Font family for colorbar text")
        self.colorbar_font_combo.currentTextChanged.connect(self._on_colorbar_layout_changed)
        colorbar_layout.addRow("Font:", self.colorbar_font_combo)

        # Title font size
        self.colorbar_title_size_spin = QSpinBox()
        self.colorbar_title_size_spin.setRange(8, 48)
        self.colorbar_title_size_spin.setValue(16)
        self.colorbar_title_size_spin.setToolTip("Font size for colorbar title")
        self.colorbar_title_size_spin.valueChanged.connect(self._on_colorbar_layout_changed)
        colorbar_layout.addRow("Title Size:", self.colorbar_title_size_spin)

        # Label font size
        self.colorbar_label_size_spin = QSpinBox()
        self.colorbar_label_size_spin.setRange(8, 48)
        self.colorbar_label_size_spin.setValue(14)
        self.colorbar_label_size_spin.setToolTip("Font size for colorbar tick labels")
        self.colorbar_label_size_spin.valueChanged.connect(self._on_colorbar_layout_changed)
        colorbar_layout.addRow("Label Size:", self.colorbar_label_size_spin)

        # Bold font option
        self.colorbar_bold_cb = QCheckBox("Bold text")
        self.colorbar_bold_cb.setChecked(False)
        self.colorbar_bold_cb.setToolTip("Use bold font for colorbar text")
        self.colorbar_bold_cb.stateChanged.connect(self._on_colorbar_layout_changed)
        colorbar_layout.addRow("", self.colorbar_bold_cb)

        # Shadow for better readability
        self.colorbar_shadow_cb = QCheckBox("Text shadow")
        self.colorbar_shadow_cb.setChecked(True)
        self.colorbar_shadow_cb.setToolTip("Add shadow behind text for better visibility")
        self.colorbar_shadow_cb.stateChanged.connect(self._on_colorbar_layout_changed)
        colorbar_layout.addRow("", self.colorbar_shadow_cb)

        side_layout.addWidget(colorbar_group)

        # --- Statistics Panel ---
        stats_group = QGroupBox("Statistics", side_panel)
        stats_layout = QVBoxLayout()
        stats_layout.setSpacing(2)
        stats_group.setLayout(stats_layout)

        self.stats_label = QLabel("Load a solution to view statistics")
        self.stats_label.setWordWrap(True)
        self.stats_label.setStyleSheet("font-family: monospace; font-size: 10px;")
        stats_layout.addWidget(self.stats_label)

        refresh_stats_btn = QPushButton("Refresh Statistics")
        refresh_stats_btn.setIcon(CADIcons.get_icon('chart'))
        refresh_stats_btn.clicked.connect(self._update_statistics)
        stats_layout.addWidget(refresh_stats_btn)

        side_layout.addWidget(stats_group)

        # --- Probing Panel ---
        probe_group = QGroupBox("Probe Data", side_panel)
        probe_layout = QVBoxLayout()
        probe_layout.setSpacing(2)
        probe_group.setLayout(probe_layout)

        self.probe_label = QLabel("Enable probe mode to inspect values.\nClick on the mesh to probe.")
        self.probe_label.setWordWrap(True)
        self.probe_label.setStyleSheet("font-family: monospace; font-size: 10px;")
        probe_layout.addWidget(self.probe_label)

        self.probe_mode_btn = QPushButton("Enable Probe Mode")
        self.probe_mode_btn.setIcon(CADIcons.get_icon('probe'))
        self.probe_mode_btn.setCheckable(True)
        self.probe_mode_btn.toggled.connect(self._toggle_probe_mode)
        probe_layout.addWidget(self.probe_mode_btn)

        side_layout.addWidget(probe_group)

        # --- Vector Field / Glyphs ---
        vector_group = QGroupBox("Vector Field", side_panel)
        vector_layout = QFormLayout()
        vector_layout.setSpacing(4)
        vector_group.setLayout(vector_layout)

        self.vector_field_combo = QComboBox()
        self.vector_field_combo.addItem("(None)")
        self.vector_field_combo.currentTextChanged.connect(self._on_vector_field_changed)
        vector_layout.addRow("Field:", self.vector_field_combo)

        self.glyph_scale_spin = QDoubleSpinBox()
        self.glyph_scale_spin.setRange(0.01, 100.0)
        self.glyph_scale_spin.setValue(1.0)
        self.glyph_scale_spin.setSingleStep(0.1)
        self.glyph_scale_spin.setToolTip("Scale factor for vector glyphs")
        self.glyph_scale_spin.valueChanged.connect(self._on_glyph_scale_changed)
        vector_layout.addRow("Scale:", self.glyph_scale_spin)

        self.glyph_density_spin = QSpinBox()
        self.glyph_density_spin.setRange(1, 100)
        self.glyph_density_spin.setValue(20)
        self.glyph_density_spin.setToolTip("Percentage of points to show glyphs on")
        self.glyph_density_spin.valueChanged.connect(self._on_glyph_density_changed)
        vector_layout.addRow("Density %:", self.glyph_density_spin)

        side_layout.addWidget(vector_group)

        # --- Camera Presets ---
        camera_group = QGroupBox("Camera Presets", side_panel)
        camera_layout = QGridLayout()
        camera_layout.setSpacing(2)
        camera_group.setLayout(camera_layout)

        preset_buttons = [
            ("+X", self._view_plus_x, 0, 0),
            ("-X", self._view_minus_x, 0, 1),
            ("+Y", self._view_plus_y, 0, 2),
            ("-Y", self._view_minus_y, 1, 0),
            ("+Z", self._view_plus_z, 1, 1),
            ("-Z", self._view_minus_z, 1, 2),
            ("Iso", self._view_iso, 2, 0),
            ("Fit", self._reset_camera_view, 2, 1),
        ]

        for label, callback, row, col in preset_buttons:
            btn = QPushButton(label)
            btn.setFixedHeight(24)
            btn.clicked.connect(callback)
            camera_layout.addWidget(btn, row, col)

        # Parallel projection checkbox
        self.parallel_cb = QCheckBox("Parallel projection")
        self.parallel_cb.toggled.connect(self._toggle_parallel_projection)
        camera_layout.addWidget(self.parallel_cb, 2, 2)

        side_layout.addWidget(camera_group)

        # --- Slice Controls (initially hidden) ---
        self.slice_group = QGroupBox("Plane Slice", side_panel)
        slice_layout = QFormLayout()
        slice_layout.setSpacing(4)
        self.slice_group.setLayout(slice_layout)

        self.slice_origin_x = QDoubleSpinBox()
        self.slice_origin_x.setRange(-1e6, 1e6)
        self.slice_origin_x.setDecimals(3)
        self.slice_origin_x.valueChanged.connect(self._update_slice)
        slice_layout.addRow("Origin X:", self.slice_origin_x)

        self.slice_origin_y = QDoubleSpinBox()
        self.slice_origin_y.setRange(-1e6, 1e6)
        self.slice_origin_y.setDecimals(3)
        self.slice_origin_y.valueChanged.connect(self._update_slice)
        slice_layout.addRow("Origin Y:", self.slice_origin_y)

        self.slice_origin_z = QDoubleSpinBox()
        self.slice_origin_z.setRange(-1e6, 1e6)
        self.slice_origin_z.setDecimals(3)
        self.slice_origin_z.valueChanged.connect(self._update_slice)
        slice_layout.addRow("Origin Z:", self.slice_origin_z)

        self.slice_normal_combo = QComboBox()
        self.slice_normal_combo.addItems(["X", "Y", "Z", "Custom"])
        self.slice_normal_combo.currentTextChanged.connect(self._update_slice)
        slice_layout.addRow("Normal:", self.slice_normal_combo)

        self.slice_group.setVisible(False)
        side_layout.addWidget(self.slice_group)

        # --- Threshold Controls (initially hidden) ---
        self.threshold_group = QGroupBox("Threshold Filter", side_panel)
        threshold_layout = QFormLayout()
        threshold_layout.setSpacing(4)
        self.threshold_group.setLayout(threshold_layout)

        self.threshold_min_spin = QDoubleSpinBox()
        self.threshold_min_spin.setRange(-1e12, 1e12)
        self.threshold_min_spin.setDecimals(4)
        self.threshold_min_spin.valueChanged.connect(self._update_threshold)
        threshold_layout.addRow("Min:", self.threshold_min_spin)

        self.threshold_max_spin = QDoubleSpinBox()
        self.threshold_max_spin.setRange(-1e12, 1e12)
        self.threshold_max_spin.setDecimals(4)
        self.threshold_max_spin.valueChanged.connect(self._update_threshold)
        threshold_layout.addRow("Max:", self.threshold_max_spin)

        self.threshold_invert_cb = QCheckBox("Invert selection")
        self.threshold_invert_cb.toggled.connect(self._update_threshold)
        threshold_layout.addRow("", self.threshold_invert_cb)

        self.threshold_group.setVisible(False)
        side_layout.addWidget(self.threshold_group)

        side_layout.addStretch(1)
        main_layout.addWidget(scroll_area, stretch=0)

        self._set_scalar_controls_enabled(False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        """Release heavy resources when the application closes."""
        # Stop animation timer
        if self._animation_timer is not None:
            self._animation_timer.stop()
            self._animation_timer = None

        if getattr(self, "vtk_widget", None) is not None:
            try:
                self.vtk_widget.shutdown()
            except Exception:
                pass
        self._reader = None
        self._current_mesh = None

    def _request_close(self) -> None:
        """Request closing of the parent dock widget that hosts this inspector."""
        parent = self.parent()
        while parent is not None and not isinstance(parent, QDockWidget):
            parent = parent.parent()
        if isinstance(parent, QDockWidget):
            parent.hide()

    # ------------------------------------------------------------------
    # Animation controls
    # ------------------------------------------------------------------
    def _toggle_animation(self) -> None:
        """Toggle play/pause animation."""
        if self._animation_playing:
            self._pause_animation()
        else:
            self._play_animation()

    def _play_animation(self) -> None:
        """Start animation playback."""
        if not self._time_values or len(self._time_values) < 2:
            return

        self._animation_playing = True
        self.play_btn.setIcon(CADIcons.get_icon('pause'))
        self.play_btn.setToolTip("Pause animation")

        if self._animation_timer is None:
            self._animation_timer = QTimer(self)
            self._animation_timer.timeout.connect(self._advance_frame)

        interval_ms = int(1000.0 / self._animation_fps)
        self._animation_timer.start(interval_ms)

    def _pause_animation(self) -> None:
        """Pause animation playback."""
        self._animation_playing = False
        self.play_btn.setIcon(CADIcons.get_icon('play'))
        self.play_btn.setToolTip("Play animation")

        if self._animation_timer is not None:
            self._animation_timer.stop()

    def _stop_animation(self) -> None:
        """Stop animation and reset to first frame."""
        self._pause_animation()
        self.time_slider.setValue(0)

    def _advance_frame(self) -> None:
        """Advance to next frame in animation."""
        if not self._time_values:
            return

        next_idx = self._current_time_index + 1
        if next_idx >= len(self._time_values):
            if self._animation_loop:
                next_idx = 0
            else:
                self._pause_animation()
                return

        self.time_slider.setValue(next_idx)

    def _on_loop_toggled(self, checked: bool) -> None:
        """Handle loop toggle."""
        self._animation_loop = checked

    def _on_fps_changed(self, value: int) -> None:
        """Handle FPS change."""
        self._animation_fps = float(value)
        if self._animation_playing and self._animation_timer is not None:
            interval_ms = int(1000.0 / self._animation_fps)
            self._animation_timer.setInterval(interval_ms)

    # ------------------------------------------------------------------
    # Mesh display controls
    # ------------------------------------------------------------------
    def _on_opacity_changed(self, value: int) -> None:
        """Handle opacity slider change."""
        self._mesh_opacity = value / 100.0
        self._render_current_mesh()

    def _on_representation_changed(self, text: str) -> None:
        """Handle representation style change."""
        self._mesh_representation = text
        self._mesh_show_edges = (text == "Surface + Edges")
        self._render_current_mesh()

    # ------------------------------------------------------------------
    # Camera controls
    # ------------------------------------------------------------------
    def _reset_camera_view(self) -> None:
        """Reset the 3D view camera."""
        if hasattr(self, "vtk_widget") and self.vtk_widget is not None:
            try:
                self.vtk_widget.reset_camera()
            except Exception:
                pass

    def _view_iso(self) -> None:
        """Switch to isometric view."""
        if hasattr(self, "vtk_widget") and self.vtk_widget is not None:
            try:
                self.vtk_widget.view_iso()
            except Exception:
                pass

    def _view_plus_x(self) -> None:
        """View from +X direction."""
        plotter = getattr(self.vtk_widget, "plotter", None)
        if plotter:
            plotter.view_yz()
            plotter.camera.azimuth = 0
            plotter.render()

    def _view_minus_x(self) -> None:
        """View from -X direction."""
        plotter = getattr(self.vtk_widget, "plotter", None)
        if plotter:
            plotter.view_yz()
            plotter.camera.azimuth = 180
            plotter.render()

    def _view_plus_y(self) -> None:
        """View from +Y direction."""
        plotter = getattr(self.vtk_widget, "plotter", None)
        if plotter:
            plotter.view_xz()
            plotter.camera.azimuth = 90
            plotter.render()

    def _view_minus_y(self) -> None:
        """View from -Y direction."""
        plotter = getattr(self.vtk_widget, "plotter", None)
        if plotter:
            plotter.view_xz()
            plotter.camera.azimuth = -90
            plotter.render()

    def _view_plus_z(self) -> None:
        """View from +Z direction (top)."""
        plotter = getattr(self.vtk_widget, "plotter", None)
        if plotter:
            plotter.view_xy()
            plotter.render()

    def _view_minus_z(self) -> None:
        """View from -Z direction (bottom)."""
        plotter = getattr(self.vtk_widget, "plotter", None)
        if plotter:
            plotter.view_xy()
            plotter.camera.elevation = -90
            plotter.render()

    def _toggle_parallel_projection(self, checked: bool = None) -> None:
        """Toggle between perspective and parallel projection."""
        plotter = getattr(self.vtk_widget, "plotter", None)
        if plotter is None:
            return

        if checked is None:
            checked = not plotter.camera.GetParallelProjection()

        plotter.camera.SetParallelProjection(checked)
        plotter.render()

        # Sync UI
        self.parallel_proj_action.setChecked(checked)
        self.parallel_cb.setChecked(checked)

    # ------------------------------------------------------------------
    # Probing
    # ------------------------------------------------------------------
    def _toggle_probe_mode(self, enabled: bool = None) -> None:
        """Toggle probe mode for inspecting values."""
        if enabled is None:
            enabled = not self._probe_enabled

        self._probe_enabled = enabled
        self.probe_action.setChecked(enabled)
        self.probe_mode_btn.setChecked(enabled)

        plotter = getattr(self.vtk_widget, "plotter", None)
        if plotter is None:
            return

        # Always disable any existing picking first to avoid conflicts
        try:
            plotter.disable_picking()
        except Exception:
            pass

        if enabled:
            self.probe_label.setText("Click on mesh to probe values...")
            try:
                plotter.enable_point_picking(
                    callback=self._on_probe_point,
                    show_message=False,
                    color='yellow',
                    point_size=16,
                    show_point=True,
                )
            except Exception as e:
                self.probe_label.setText(f"Probe mode error: {e}")
                self._probe_enabled = False
                self.probe_action.setChecked(False)
                self.probe_mode_btn.setChecked(False)
        else:
            self.probe_label.setText("Probe mode disabled")
            # Clear probe marker if exists
            if self._probe_actor is not None:
                try:
                    plotter.remove_actor(self._probe_actor)
                except Exception:
                    pass
                self._probe_actor = None

    def _on_probe_point(self, point) -> None:
        """Handle probed point."""
        if not self._probe_enabled or self._current_mesh is None:
            return

        import numpy as np
        point = np.asarray(point)

        # Find closest point on mesh
        mesh = self._current_mesh
        try:
            closest_idx = mesh.find_closest_point(point)
            closest_pt = mesh.points[closest_idx]
        except Exception:
            return

        # Get scalar value at this point
        name = self._current_scalar_name
        value = None
        if name and name in mesh.point_data:
            value = mesh.point_data[name][closest_idx]
        elif name and name in mesh.cell_data:
            # For cell data, find cell containing point
            try:
                cell_id = mesh.find_containing_cell(closest_pt)
                if cell_id >= 0:
                    value = mesh.cell_data[name][cell_id]
            except Exception:
                pass

        # Update probe display
        if value is not None:
            self.probe_label.setText(
                f"Position: ({closest_pt[0]:.4g}, {closest_pt[1]:.4g}, {closest_pt[2]:.4g})\n"
                f"Value: {value:.6g}"
            )
            self.probed_value_changed.emit(closest_pt[0], closest_pt[1], closest_pt[2], float(value))

            # Store for time series
            point_key = (round(closest_pt[0], 4), round(closest_pt[1], 4), round(closest_pt[2], 4))
            if point_key not in self._time_series_data:
                self._time_series_data[point_key] = []

            if self._time_values:
                time = self._time_values[self._current_time_index]
                self._time_series_data[point_key].append((time, float(value)))
        else:
            self.probe_label.setText(
                f"Position: ({closest_pt[0]:.4g}, {closest_pt[1]:.4g}, {closest_pt[2]:.4g})\n"
                f"No scalar value available"
            )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def _update_statistics(self) -> None:
        """Update statistics display for current scalar field."""
        mesh = self._current_mesh
        name = self._current_scalar_name

        if mesh is None or name is None:
            self.stats_label.setText("No data loaded")
            return

        try:
            import numpy as np

            values = None
            if name in mesh.point_data:
                values = mesh.point_data[name]
            elif name in mesh.cell_data:
                values = mesh.cell_data[name]

            if values is None or len(values) == 0:
                self.stats_label.setText("No values for selected field")
                return

            # Compute statistics
            vmin = float(np.nanmin(values))
            vmax = float(np.nanmax(values))
            vmean = float(np.nanmean(values))
            vstd = float(np.nanstd(values))
            vmedian = float(np.nanmedian(values))
            count = len(values)

            stats_text = (
                f"Field: {name}\n"
                f"Count: {count:,}\n"
                f"Min:   {vmin:.6g}\n"
                f"Max:   {vmax:.6g}\n"
                f"Mean:  {vmean:.6g}\n"
                f"Std:   {vstd:.6g}\n"
                f"Median:{vmedian:.6g}"
            )
            self.stats_label.setText(stats_text)

        except Exception as e:
            self.stats_label.setText(f"Error computing stats:\n{e}")

    # ------------------------------------------------------------------
    # Slicing
    # ------------------------------------------------------------------
    def _toggle_slice(self, enabled: bool = None) -> None:
        """Toggle plane slice visualization."""
        if enabled is None:
            enabled = not self._slice_enabled

        self._slice_enabled = enabled
        self.slice_action.setChecked(enabled)
        self.slice_group.setVisible(enabled)

        if enabled and self._current_mesh is not None:
            # Initialize slice origin to mesh center
            center = self._current_mesh.center
            self.slice_origin_x.setValue(center[0])
            self.slice_origin_y.setValue(center[1])
            self.slice_origin_z.setValue(center[2])
            self._update_slice()
        else:
            # Remove slice actor
            plotter = getattr(self.vtk_widget, "plotter", None)
            if plotter and self._slice_actor is not None:
                try:
                    plotter.remove_actor(self._slice_actor)
                except Exception:
                    pass
                self._slice_actor = None
            plotter.render() if plotter else None

    def _update_slice(self) -> None:
        """Update the slice plane."""
        if not self._slice_enabled or self._current_mesh is None:
            return

        plotter = getattr(self.vtk_widget, "plotter", None)
        if plotter is None:
            return

        # Remove old slice
        if self._slice_actor is not None:
            try:
                plotter.remove_actor(self._slice_actor)
            except Exception:
                pass

        origin = (
            self.slice_origin_x.value(),
            self.slice_origin_y.value(),
            self.slice_origin_z.value(),
        )

        normal_map = {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1), "Custom": (1, 0, 0)}
        normal = normal_map.get(self.slice_normal_combo.currentText(), (1, 0, 0))

        try:
            sliced = self._current_mesh.slice(normal=normal, origin=origin)
            if sliced.n_points > 0:
                name = self._current_scalar_name
                self._slice_actor = plotter.add_mesh(
                    sliced,
                    scalars=name if name else None,
                    cmap=self._current_cmap,
                    line_width=3,
                    name="slice_plane",
                )
        except Exception:
            pass

        plotter.render()

    # ------------------------------------------------------------------
    # Threshold filtering
    # ------------------------------------------------------------------
    def _toggle_threshold(self, enabled: bool = None) -> None:
        """Toggle threshold filter visualization."""
        if enabled is None:
            enabled = not self._threshold_enabled

        self._threshold_enabled = enabled
        self.threshold_action.setChecked(enabled)
        self.threshold_group.setVisible(enabled)

        if enabled and self._current_mesh is not None:
            # Initialize with current scalar range
            rng = self._compute_local_range(self._current_scalar_name)
            if rng:
                self.threshold_min_spin.setValue(rng[0])
                self.threshold_max_spin.setValue(rng[1])
            self._update_threshold()
        else:
            # Restore original mesh visibility
            self._render_current_mesh()

    def _update_threshold(self) -> None:
        """Update threshold filter."""
        if not self._threshold_enabled or self._current_mesh is None:
            return

        plotter = getattr(self.vtk_widget, "plotter", None)
        if plotter is None:
            return

        name = self._current_scalar_name
        if name is None:
            return

        vmin = self.threshold_min_spin.value()
        vmax = self.threshold_max_spin.value()
        invert = self.threshold_invert_cb.isChecked()

        try:
            # Remove current mesh and add thresholded version
            plotter.clear()

            thresholded = self._current_mesh.threshold(
                value=[vmin, vmax],
                scalars=name,
                invert=invert,
            )

            if thresholded.n_points > 0:
                self._add_mesh_to_plotter(plotter, thresholded)

            plotter.render()
        except Exception:
            # Fall back to normal rendering
            self._render_current_mesh()

    # ------------------------------------------------------------------
    # Vector field / glyphs
    # ------------------------------------------------------------------
    def _on_vector_field_changed(self, name: str) -> None:
        """Handle vector field selection change."""
        if name == "(None)" or not name:
            self._vector_field_name = None
            self._remove_glyphs()
            return

        self._vector_field_name = name
        self._update_glyphs()

    def _on_glyph_scale_changed(self, value: float) -> None:
        """Handle glyph scale change."""
        self._glyph_scale = value
        self._update_glyphs()

    def _on_glyph_density_changed(self, value: int) -> None:
        """Handle glyph density change."""
        self._update_glyphs()

    def _update_glyphs(self) -> None:
        """Update vector field glyph visualization."""
        plotter = getattr(self.vtk_widget, "plotter", None)
        if plotter is None or self._current_mesh is None:
            return

        self._remove_glyphs()

        name = self._vector_field_name
        if name is None:
            return

        try:
            import numpy as np

            # Get vector data
            vectors = None
            if name in self._current_mesh.point_data:
                vectors = self._current_mesh.point_data[name]
            elif name in self._current_mesh.cell_data:
                # Convert to point data for glyphs
                self._current_mesh = self._current_mesh.cell_data_to_point_data()
                vectors = self._current_mesh.point_data.get(name)

            if vectors is None or vectors.ndim != 2 or vectors.shape[1] != 3:
                return

            # Subsample for density
            density = self.glyph_density_spin.value() / 100.0
            n_points = self._current_mesh.n_points
            n_show = max(1, int(n_points * density))

            if n_show < n_points:
                indices = np.random.choice(n_points, n_show, replace=False)
                points = self._current_mesh.points[indices]
                vectors = vectors[indices]
            else:
                points = self._current_mesh.points

            # Create arrow glyphs
            arrow = self._pv.Arrow()
            glyphs = self._pv.PolyData(points)
            glyphs["vectors"] = vectors
            glyphs.set_active_vectors("vectors")

            glyph_mesh = glyphs.glyph(
                orient="vectors",
                scale="vectors",
                factor=self._glyph_scale,
                geom=arrow,
            )

            self._glyph_actor = plotter.add_mesh(
                glyph_mesh,
                scalars=np.linalg.norm(vectors, axis=1),
                cmap=self._current_cmap,
                name="vector_glyphs",
            )

            plotter.render()

        except Exception:
            pass

    def _remove_glyphs(self) -> None:
        """Remove vector glyphs from visualization."""
        plotter = getattr(self.vtk_widget, "plotter", None)
        if plotter and self._glyph_actor is not None:
            try:
                plotter.remove_actor(self._glyph_actor)
            except Exception:
                pass
            self._glyph_actor = None

    # ------------------------------------------------------------------
    # Export functions
    # ------------------------------------------------------------------
    def _take_screenshot(self) -> None:
        """Capture a screenshot of the current 3D view."""
        if not hasattr(self, "vtk_widget") or self.vtk_widget is None:
            return
        plotter = getattr(self.vtk_widget, "plotter", None)
        if plotter is None:
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Screenshot",
            "",
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;All Files (*)",
        )
        if not path:
            return

        try:
            plotter.screenshot(path)
        except Exception:
            pass

    def _export_mesh(self) -> None:
        """Export current mesh to VTP/VTU file."""
        if self._current_mesh is None:
            QMessageBox.warning(self, "No Data", "No mesh loaded to export.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Mesh",
            "",
            "VTK PolyData (*.vtp);;VTK Unstructured Grid (*.vtu);;All Files (*)",
        )
        if not path:
            return

        try:
            self._current_mesh.save(path)
        except Exception as e:
            self._record_telemetry(e, action="solution_export_mesh")
            QMessageBox.critical(self, "Export Failed", f"Failed to export mesh:\n{e}")

    def _export_csv(self) -> None:
        """Export current scalar data to CSV."""
        if self._current_mesh is None or self._current_scalar_name is None:
            QMessageBox.warning(self, "No Data", "No scalar data loaded to export.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Scalars to CSV",
            "",
            "CSV Files (*.csv);;All Files (*)",
        )
        if not path:
            return

        try:
            import numpy as np

            name = self._current_scalar_name
            mesh = self._current_mesh

            if name in mesh.point_data:
                values = mesh.point_data[name]
                points = mesh.points
            elif name in mesh.cell_data:
                values = mesh.cell_data[name]
                points = mesh.cell_centers().points
            else:
                raise ValueError("Scalar field not found")

            # Write CSV
            with open(path, 'w') as f:
                f.write(f"x,y,z,{name}\n")
                for i, (pt, val) in enumerate(zip(points, values)):
                    if np.isscalar(val):
                        f.write(f"{pt[0]},{pt[1]},{pt[2]},{val}\n")
                    else:
                        # Vector data
                        f.write(f"{pt[0]},{pt[1]},{pt[2]},{','.join(map(str, val))}\n")

        except Exception as e:
            self._record_telemetry(e, action="solution_export_csv")
            QMessageBox.critical(self, "Export Failed", f"Failed to export CSV:\n{e}")

    def _export_animation(self) -> None:
        """Export animation as GIF or MP4."""
        if not self._time_values or len(self._time_values) < 2:
            QMessageBox.warning(self, "No Animation", "Need multiple time steps for animation.")
            return

        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Animation",
            "",
            "GIF Image (*.gif);;MP4 Video (*.mp4);;All Files (*)",
        )
        if not path:
            return

        # Determine format and ensure extension
        is_mp4 = "mp4" in selected_filter.lower() or path.lower().endswith('.mp4')
        if is_mp4:
            if not path.lower().endswith('.mp4'):
                path = path + '.mp4'
        else:
            if not path.lower().endswith('.gif'):
                path = path + '.gif'

        plotter = getattr(self.vtk_widget, "plotter", None)
        if plotter is None:
            return

        try:
            from PySide6.QtWidgets import QApplication, QProgressDialog
            from PySide6.QtCore import Qt
            import numpy as np

            # Create progress dialog
            progress = QProgressDialog("Exporting animation...", "Cancel", 0, len(self._time_values), self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)

            # Save ALL current plotter state
            saved_state = self._save_plotter_state(plotter)
            original_idx = self._current_time_index

            # Collect frames
            frames = []

            for idx in range(len(self._time_values)):
                if progress.wasCanceled():
                    break

                progress.setValue(idx)
                progress.setLabelText(f"Rendering frame {idx + 1} of {len(self._time_values)}...")
                QApplication.processEvents()

                # Update mesh for this time step with full state restoration
                self._update_mesh_for_animation(idx, saved_state)

                # Force multiple render passes for complete rendering
                plotter.render()
                QApplication.processEvents()
                plotter.render()

                # Get high-quality screenshot as RGB
                frame = plotter.screenshot(return_img=True)

                # Ensure frame is uint8
                if frame is not None:
                    if frame.dtype != np.uint8:
                        frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
                    # Handle RGBA -> RGB
                    if frame.ndim == 3 and frame.shape[2] == 4:
                        frame = frame[:, :, :3]
                    frames.append(frame.copy())

            progress.setValue(len(self._time_values))
            progress.setLabelText("Encoding video...")
            QApplication.processEvents()

            # Restore original state
            self._on_time_index_changed(original_idx)
            self._restore_plotter_state(plotter, saved_state)
            plotter.render()

            if progress.wasCanceled() or not frames:
                return

            # Save animation
            if is_mp4:
                self._save_as_mp4(path, frames, progress)
            else:
                self._save_as_gif(path, frames, progress)

        except Exception as e:
            import traceback
            # Record the failure (including full traceback) so export issues
            # are visible to developers in telemetry as well as in the UI.
            self._record_telemetry(e, action="solution_export_animation")
            QMessageBox.critical(
                self,
                "Export Failed",
                f"Failed to export animation:\n{e}\n\n{traceback.format_exc()}"
            )

    def _save_plotter_state(self, plotter) -> dict:
        """Save all relevant plotter state for animation consistency."""
        from svv.visualize.gui.theme import CADTheme

        state = {}

        # Camera settings - save everything
        try:
            state['camera_position'] = plotter.camera_position
            state['camera_focal_point'] = tuple(plotter.camera.focal_point)
            state['camera_view_up'] = tuple(plotter.camera.up)
            state['camera_parallel_projection'] = plotter.camera.GetParallelProjection()
            state['camera_parallel_scale'] = plotter.camera.GetParallelScale()
            state['camera_clipping_range'] = plotter.camera.clipping_range
            state['camera_view_angle'] = plotter.camera.view_angle
        except Exception:
            state['camera_position'] = None

        # Background colors from theme
        try:
            state['bg_bottom'] = CADTheme.get_color('viewport', 'background-bottom')
            state['bg_top'] = CADTheme.get_color('viewport', 'background-top')
        except Exception:
            state['bg_bottom'] = '#1e1e1e'
            state['bg_top'] = '#2d2d30'

        # Display settings from internal state
        state['cmap'] = self._current_cmap
        state['scalar_name'] = self._current_scalar_name
        state['opacity'] = self._mesh_opacity
        state['representation'] = self._mesh_representation
        state['show_edges'] = self._mesh_show_edges
        state['colorbar_visible'] = self._colorbar_visible

        # Scalar range settings
        state['auto_range'] = self.auto_range_cb.isChecked()
        state['global_range'] = self.global_range_cb.isChecked()
        state['scalar_min'] = self.scalar_min_spin.value()
        state['scalar_max'] = self.scalar_max_spin.value()
        state['scalar_label'] = self.scalar_label_edit.text()
        state['colorbar_orientation'] = self.orientation_combo.currentText()

        # Component selection for vectors
        state['component'] = self.component_combo.currentText()

        return state

    def _restore_plotter_state(self, plotter, state: dict) -> None:
        """Restore plotter state from saved state."""
        # Restore camera
        if state.get('camera_position') is not None:
            try:
                plotter.camera_position = state['camera_position']
                plotter.camera.focal_point = state['camera_focal_point']
                plotter.camera.up = state['camera_view_up']
                plotter.camera.SetParallelProjection(state['camera_parallel_projection'])
                plotter.camera.SetParallelScale(state['camera_parallel_scale'])
                plotter.camera.clipping_range = state['camera_clipping_range']
                plotter.camera.view_angle = state['camera_view_angle']
            except Exception:
                pass

        # Restore background
        try:
            plotter.set_background(state['bg_bottom'], top=state['bg_top'])
        except Exception:
            pass

    def _update_mesh_for_animation(self, time_index: int, saved_state: dict) -> None:
        """Update mesh for animation frame with full state restoration."""
        if self._reader is None or self._pv is None:
            return

        plotter = getattr(self.vtk_widget, "plotter", None)
        if plotter is None:
            return

        # Load mesh for this time index
        try:
            if hasattr(self._reader, "set_active_time_index"):
                self._reader.set_active_time_index(int(time_index))
            mesh = self._reader.read()
        except Exception:
            return

        if isinstance(mesh, self._pv.MultiBlock):
            try:
                mesh = mesh.combine()
            except Exception:
                if len(mesh) > 0:
                    mesh = mesh[0]

        self._current_mesh = mesh
        self._current_time_index = time_index

        # Clear plotter
        plotter.clear()

        # Restore background gradient
        try:
            plotter.set_background(saved_state['bg_bottom'], top=saved_state['bg_top'])
        except Exception:
            pass

        # Add mesh using the SAME function that works for normal display
        # This ensures consistent colormap rendering
        if mesh is not None:
            self._add_mesh_to_plotter(plotter, mesh)

        # Restore camera position AFTER adding mesh
        if saved_state.get('camera_position') is not None:
            try:
                plotter.camera_position = saved_state['camera_position']
                plotter.camera.focal_point = saved_state['camera_focal_point']
                plotter.camera.up = saved_state['camera_view_up']
                plotter.camera.SetParallelProjection(saved_state['camera_parallel_projection'])
                plotter.camera.SetParallelScale(saved_state['camera_parallel_scale'])
                plotter.camera.clipping_range = saved_state['camera_clipping_range']
            except Exception:
                pass

    def _add_mesh_to_plotter_with_state(self, plotter, mesh, state: dict) -> None:
        """Add mesh to plotter using saved state for consistent rendering."""
        import numpy as np

        # Get scalar name and handle vector components
        scalar_name = state.get('scalar_name')
        display_scalar_name = scalar_name

        if scalar_name and mesh is not None:
            arr = None
            if scalar_name in mesh.point_data:
                arr = mesh.point_data[scalar_name]
            elif scalar_name in mesh.cell_data:
                arr = mesh.cell_data[scalar_name]

            # Handle vector fields - compute component
            if arr is not None and arr.ndim == 2 and arr.shape[1] == 3:
                component = state.get('component', 'Magnitude')
                if component == "X":
                    computed = arr[:, 0]
                    display_scalar_name = f"{scalar_name}_X"
                elif component == "Y":
                    computed = arr[:, 1]
                    display_scalar_name = f"{scalar_name}_Y"
                elif component == "Z":
                    computed = arr[:, 2]
                    display_scalar_name = f"{scalar_name}_Z"
                else:  # Magnitude
                    computed = np.linalg.norm(arr, axis=1)
                    display_scalar_name = f"{scalar_name}_mag"

                # Add computed array to mesh
                if scalar_name in mesh.point_data:
                    mesh.point_data[display_scalar_name] = computed
                else:
                    mesh.cell_data[display_scalar_name] = computed

        # Determine color limits
        clim = None
        if display_scalar_name:
            if state.get('auto_range', True):
                # Compute range from current mesh
                try:
                    values = mesh[display_scalar_name]
                    if values is not None and len(values) > 0:
                        clim = (float(np.nanmin(values)), float(np.nanmax(values)))
                except Exception:
                    pass

                # Use global range if enabled
                if state.get('global_range', False):
                    global_clim = self._global_ranges.get(f"{scalar_name}_{state.get('component', 'Magnitude')}")
                    if global_clim:
                        clim = global_clim
            else:
                clim = (state.get('scalar_min', 0), state.get('scalar_max', 1))

        # Scalar bar configuration
        scalar_bar_args = {
            "title": state.get('scalar_label', '') or display_scalar_name or '',
            "vertical": state.get('colorbar_orientation', 'Vertical') == 'Vertical',
            "title_font_size": 12,
            "label_font_size": 10,
            "n_labels": 5,
            "fmt": "%.3g",
        }

        # Representation style
        style = "surface"
        show_edges = state.get('show_edges', False)
        repr_mode = state.get('representation', 'Surface')
        if repr_mode == "Wireframe":
            style = "wireframe"
        elif repr_mode == "Points":
            style = "points"
        elif repr_mode == "Surface + Edges":
            show_edges = True

        # Add mesh with full settings
        try:
            cmap = state.get('cmap', 'coolwarm')
            opacity = state.get('opacity', 1.0)
            show_colorbar = bool(display_scalar_name and state.get('colorbar_visible', True))

            kwargs = {
                "scalars": display_scalar_name if display_scalar_name else None,
                "clim": clim,
                "cmap": cmap,
                "opacity": opacity,
                "style": style,
                "show_edges": show_edges,
                "lighting": True,
                "smooth_shading": True,
            }

            if show_colorbar and display_scalar_name:
                kwargs["scalar_bar_args"] = scalar_bar_args
            else:
                kwargs["show_scalar_bar"] = False

            plotter.add_mesh(mesh, **kwargs)
        except Exception:
            # Fallback: add mesh without scalars
            try:
                plotter.add_mesh(mesh, opacity=state.get('opacity', 1.0))
            except Exception:
                pass

    def _save_as_gif(self, path: str, frames: list, progress) -> None:
        """Save frames as GIF using imageio (best compatibility)."""
        from PySide6.QtWidgets import QApplication

        progress.setLabelText("Saving GIF...")
        QApplication.processEvents()

        # Use imageio directly - it handles GIF encoding well
        try:
            import imageio.v3 as iio
            duration_ms = int(1000.0 / self._animation_fps)
            iio.imwrite(path, frames, extension=".gif", duration=duration_ms, loop=0)
            QMessageBox.information(self, "Export Complete", f"Animation saved to:\n{path}")
        except (ImportError, TypeError):
            try:
                import imageio
                duration = 1.0 / self._animation_fps
                imageio.mimsave(path, frames, format='GIF', duration=duration, loop=0)
                QMessageBox.information(self, "Export Complete", f"Animation saved to:\n{path}")
            except ImportError:
                QMessageBox.warning(
                    self, "Missing Dependency",
                    "imageio is required for GIF export.\n"
                    "Install with: pip install imageio"
                )
            except Exception as e:
                self._record_telemetry(e, action="solution_export_gif")
                QMessageBox.critical(self, "GIF Export Failed", f"Error: {e}")

    def _get_colormap_colors(self, cmap_name: str, n_colors: int = 64) -> list:
        """Get RGB colors from a matplotlib/pyvista colormap."""
        colors = []
        try:
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm

            # Get the colormap
            try:
                cmap = plt.get_cmap(cmap_name)
            except ValueError:
                cmap = plt.get_cmap('coolwarm')  # fallback

            # Sample colors from the colormap
            for i in range(n_colors):
                t = i / (n_colors - 1)
                rgba = cmap(t)
                r = int(rgba[0] * 255)
                g = int(rgba[1] * 255)
                b = int(rgba[2] * 255)
                colors.append((r, g, b))

        except ImportError:
            # Fallback: manually define coolwarm-like colors
            # Blue to white to red
            for i in range(n_colors):
                t = i / (n_colors - 1)
                if t < 0.5:
                    # Blue to white
                    t2 = t * 2
                    r = int(59 + t2 * (255 - 59))
                    g = int(76 + t2 * (255 - 76))
                    b = int(192 + t2 * (255 - 192))
                else:
                    # White to red
                    t2 = (t - 0.5) * 2
                    r = int(255)
                    g = int(255 - t2 * (255 - 58))
                    b = int(255 - t2 * (255 - 68))
                colors.append((r, g, b))

        return colors

    def _save_as_mp4(self, path: str, frames: list, progress) -> None:
        """Save frames as MP4 video for better color quality."""
        from PySide6.QtWidgets import QApplication

        try:
            import imageio.v3 as iio

            progress.setLabelText("Encoding MP4 video...")
            QApplication.processEvents()

            # Use ffmpeg plugin for MP4
            fps = int(self._animation_fps)
            iio.imwrite(
                path,
                frames,
                extension=".mp4",
                fps=fps,
                codec='libx264',
                quality=8,  # Higher quality (1-10 scale, higher is better)
            )

            QMessageBox.information(self, "Export Complete", f"Video saved to:\n{path}")

        except ImportError:
            try:
                import imageio

                fps = int(self._animation_fps)
                writer = imageio.get_writer(path, fps=fps, codec='libx264', quality=8)
                for frame in frames:
                    writer.append_data(frame)
                writer.close()

                QMessageBox.information(self, "Export Complete", f"Video saved to:\n{path}")

            except ImportError:
                QMessageBox.warning(
                    self, "Missing Dependency",
                    "imageio with ffmpeg is required for MP4 export.\n"
                    "Install with: pip install imageio imageio-ffmpeg"
                )
            except Exception as e:
                self._record_telemetry(e, action="solution_export_mp4")
                QMessageBox.critical(self, "Export Failed", f"Failed to save MP4:\n{e}")

    def _export_time_series(self) -> None:
        """Export probed time series data to CSV."""
        if not self._time_series_data:
            QMessageBox.warning(
                self, "No Data",
                "No time series data collected.\nEnable probe mode and click on points to collect data."
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Time Series",
            "",
            "CSV Files (*.csv);;All Files (*)",
        )
        if not path:
            return

        try:
            with open(path, 'w') as f:
                f.write("point_x,point_y,point_z,time,value\n")
                for (px, py, pz), data in self._time_series_data.items():
                    for time, value in data:
                        f.write(f"{px},{py},{pz},{time},{value}\n")
        except Exception as e:
            self._record_telemetry(e, action="solution_export_time_series")
            QMessageBox.critical(self, "Export Failed", f"Failed to export time series:\n{e}")

    # ------------------------------------------------------------------
    # File handling
    # ------------------------------------------------------------------
    def _browse_for_file(self) -> None:
        if self._pv is None:
            return

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Solution (.pvd)",
            "",
            "VTK Collection (*.pvd);;VTK Files (*.vtp *.vtu);;All Files (*)",
        )
        if not path:
            return

        self.file_edit.setText(path)
        self._load_solution(path)

    def _load_solution(self, path: str) -> None:
        if self._pv is None:
            return
        if not os.path.isfile(path):
            return

        try:
            reader = self._pv.get_reader(path)
        except Exception:
            return

        self._reader = reader
        self._time_values = list(getattr(reader, "time_values", []) or [])
        self._global_ranges.clear()
        self._time_series_data.clear()

        if self._time_values:
            self.time_slider.setEnabled(True)
            self.time_slider.setRange(0, len(self._time_values) - 1)
            self.time_slider.setValue(0)
            self._current_time_index = 0
            self._update_time_label(0)
        else:
            self.time_slider.setEnabled(False)
            self.time_slider.setRange(0, 0)
            self.time_label.setText("Time: --")

        self._update_mesh_from_reader(time_index=0)
        self._populate_scalar_fields()
        self._populate_vector_fields()
        self._update_statistics()

    # ------------------------------------------------------------------
    # Time handling
    # ------------------------------------------------------------------
    def _update_time_label(self, index: int) -> None:
        if not self._time_values or index < 0 or index >= len(self._time_values):
            self.time_label.setText("Time: --")
            return
        t = self._time_values[index]
        total = len(self._time_values)
        try:
            self.time_label.setText(f"Time: {t:.4g} ({index + 1}/{total})")
        except Exception:
            self.time_label.setText(f"Time: {t}")

    def _on_time_index_changed(self, index: int) -> None:
        if self._reader is None:
            return
        self._current_time_index = index
        self._update_time_label(index)
        self._update_mesh_from_reader(time_index=index)

    def _update_mesh_from_reader(self, time_index: Optional[int] = None) -> None:
        if self._reader is None or self._pv is None:
            return

        try:
            if time_index is not None and hasattr(self._reader, "set_active_time_index"):
                self._reader.set_active_time_index(int(time_index))
            mesh = self._reader.read()
        except Exception:
            return

        if isinstance(mesh, self._pv.MultiBlock):
            try:
                mesh = mesh.combine()
            except Exception:
                if len(mesh) > 0:
                    mesh = mesh[0]

        self._current_mesh = mesh
        self._update_scalar_range()
        self._render_current_mesh()

        # Update slice if active
        if self._slice_enabled:
            self._update_slice()

    # ------------------------------------------------------------------
    # Scalar controls
    # ------------------------------------------------------------------
    def _set_scalar_controls_enabled(self, enabled: bool) -> None:
        self.scalar_combo.setEnabled(enabled)
        self.auto_range_cb.setEnabled(enabled)
        self.scalar_min_spin.setEnabled(enabled and not self.auto_range_cb.isChecked())
        self.scalar_max_spin.setEnabled(enabled and not self.auto_range_cb.isChecked())
        self.scalar_label_edit.setEnabled(enabled)

    def _populate_scalar_fields(self) -> None:
        self._current_scalar_name = None
        self.scalar_combo.blockSignals(True)
        self.scalar_combo.clear()
        self.scalar_combo.blockSignals(False)

        mesh = self._current_mesh
        if self._pv is None or mesh is None:
            self._set_scalar_controls_enabled(False)
            return

        names = []
        try:
            names.extend(list(mesh.point_data.keys()))
        except Exception:
            pass
        try:
            names.extend(list(mesh.cell_data.keys()))
        except Exception:
            pass

        unique_names = sorted(set(names))
        if not unique_names:
            self._set_scalar_controls_enabled(False)
            return

        self.scalar_combo.blockSignals(True)
        for name in unique_names:
            self.scalar_combo.addItem(name)
        self.scalar_combo.blockSignals(False)

        self._current_scalar_name = unique_names[0]
        self.scalar_combo.setCurrentText(self._current_scalar_name)
        self.scalar_label_edit.setText(self._current_scalar_name)

        # Check if selected field is a vector (3 components)
        self._check_vector_field()

        self._set_scalar_controls_enabled(True)
        self._update_scalar_range()
        self._render_current_mesh()

    def _populate_vector_fields(self) -> None:
        """Populate vector field combo with 3-component arrays."""
        self.vector_field_combo.blockSignals(True)
        self.vector_field_combo.clear()
        self.vector_field_combo.addItem("(None)")

        mesh = self._current_mesh
        if mesh is None:
            self.vector_field_combo.blockSignals(False)
            return

        # Find 3-component arrays (vectors)
        for name in mesh.point_data.keys():
            arr = mesh.point_data[name]
            if arr.ndim == 2 and arr.shape[1] == 3:
                self.vector_field_combo.addItem(name)

        for name in mesh.cell_data.keys():
            arr = mesh.cell_data[name]
            if arr.ndim == 2 and arr.shape[1] == 3:
                if name not in [self.vector_field_combo.itemText(i)
                               for i in range(self.vector_field_combo.count())]:
                    self.vector_field_combo.addItem(name)

        self.vector_field_combo.blockSignals(False)

    def _check_vector_field(self) -> None:
        """Check if current scalar field is a vector and enable component selector."""
        name = self._current_scalar_name
        mesh = self._current_mesh

        if name is None or mesh is None:
            self.component_combo.setEnabled(False)
            return

        arr = None
        if name in mesh.point_data:
            arr = mesh.point_data[name]
        elif name in mesh.cell_data:
            arr = mesh.cell_data[name]

        if arr is not None and arr.ndim == 2 and arr.shape[1] == 3:
            self.component_combo.setEnabled(True)
        else:
            self.component_combo.setEnabled(False)
            self.component_combo.setCurrentText("Magnitude")

    def _on_component_changed(self, component: str) -> None:
        """Handle vector component selection change."""
        self._update_scalar_range()
        self._render_current_mesh()

    def _compute_local_range(self, name: str) -> Optional[tuple[float, float]]:
        mesh = self._current_mesh
        if mesh is None or name is None:
            return None
        try:
            values = mesh[name]
        except Exception:
            return None
        if values is None or len(values) == 0:
            return None

        import numpy as np

        # Handle vector fields
        if values.ndim == 2 and values.shape[1] == 3:
            component = self.component_combo.currentText()
            if component == "X":
                values = values[:, 0]
            elif component == "Y":
                values = values[:, 1]
            elif component == "Z":
                values = values[:, 2]
            else:  # Magnitude
                values = np.linalg.norm(values, axis=1)

        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
        return vmin, vmax

    def _compute_global_range(self, name: str) -> Optional[tuple[float, float]]:
        if self._reader is None or self._pv is None:
            return None
        if not self._time_values:
            return self._compute_local_range(name)

        cache_key = f"{name}_{self.component_combo.currentText()}"
        if cache_key in self._global_ranges:
            return self._global_ranges[cache_key]

        import numpy as np

        mins = []
        maxs = []
        original_index = self._current_time_index

        for idx in range(len(self._time_values)):
            try:
                if hasattr(self._reader, "set_active_time_index"):
                    self._reader.set_active_time_index(idx)
                mesh = self._reader.read()
            except Exception:
                continue

            if isinstance(mesh, self._pv.MultiBlock):
                try:
                    mesh = mesh.combine()
                except Exception:
                    if len(mesh) > 0:
                        mesh = mesh[0]
                    else:
                        mesh = None
            if mesh is None:
                continue
            try:
                values = mesh[name]
            except Exception:
                continue
            if values is None or len(values) == 0:
                continue

            # Handle vector fields
            if values.ndim == 2 and values.shape[1] == 3:
                component = self.component_combo.currentText()
                if component == "X":
                    values = values[:, 0]
                elif component == "Y":
                    values = values[:, 1]
                elif component == "Z":
                    values = values[:, 2]
                else:
                    values = np.linalg.norm(values, axis=1)

            mins.append(float(np.nanmin(values)))
            maxs.append(float(np.nanmax(values)))

        # Restore original time index
        try:
            if hasattr(self._reader, "set_active_time_index"):
                self._reader.set_active_time_index(original_index)
        except Exception:
            pass

        if not mins:
            return None
        vmin = min(mins)
        vmax = max(maxs)
        self._global_ranges[cache_key] = (vmin, vmax)
        return vmin, vmax

    def _update_scalar_range(self) -> None:
        mesh = self._current_mesh
        name = self._current_scalar_name
        if mesh is None or name is None:
            return
        if getattr(self, "global_range_cb", None) and self.global_range_cb.isChecked():
            rng = self._compute_global_range(name)
        else:
            rng = self._compute_local_range(name)
        if rng is None:
            return
        vmin, vmax = rng
        if not self.auto_range_cb.isChecked():
            self.scalar_min_spin.blockSignals(True)
            self.scalar_max_spin.blockSignals(True)
            self.scalar_min_spin.setValue(vmin)
            self.scalar_max_spin.setValue(vmax)
            self.scalar_min_spin.blockSignals(False)
            self.scalar_max_spin.blockSignals(False)

    def _on_scalar_changed(self, name: str) -> None:
        if not name:
            return
        self._current_scalar_name = name
        if not self.scalar_label_edit.text().strip():
            self.scalar_label_edit.setText(name)
        self._check_vector_field()
        self._update_scalar_range()
        self._render_current_mesh()
        self._update_statistics()

    def _on_auto_range_toggled(self, state: int) -> None:
        auto = state == Qt.Checked
        self.scalar_min_spin.setEnabled(not auto)
        self.scalar_max_spin.setEnabled(not auto)
        if auto:
            self._update_scalar_range()
        self._render_current_mesh()

    def _on_clim_changed(self) -> None:
        if self.auto_range_cb.isChecked():
            return
        vmin = self.scalar_min_spin.value()
        vmax = self.scalar_max_spin.value()
        if vmax <= vmin:
            return
        self._render_current_mesh()

    def _on_scalar_label_changed(self) -> None:
        self._render_current_mesh()

    def _on_cmap_changed(self, cmap: str) -> None:
        if not cmap:
            return
        self._current_cmap = cmap
        self._render_current_mesh()

    def _on_global_range_toggled(self, state: int) -> None:
        self._update_scalar_range()
        self._render_current_mesh()

    def _on_colorbar_layout_changed(self, *args) -> None:
        self._render_current_mesh()

    def _on_orientation_changed(self, orientation: str) -> None:
        """Handle colorbar orientation change - adjust defaults appropriately."""
        # Block signals to prevent multiple re-renders
        self.colorbar_pos_x_spin.blockSignals(True)
        self.colorbar_pos_y_spin.blockSignals(True)
        self.colorbar_width_spin.blockSignals(True)
        self.colorbar_height_spin.blockSignals(True)

        if orientation == "Horizontal":
            # Horizontal colorbar: wide and short, at bottom
            self.colorbar_pos_x_spin.setValue(0.25)
            self.colorbar_pos_y_spin.setValue(0.05)
            self.colorbar_width_spin.setValue(0.5)
            self.colorbar_height_spin.setValue(0.08)
        else:
            # Vertical colorbar: narrow and tall, at right
            self.colorbar_pos_x_spin.setValue(0.85)
            self.colorbar_pos_y_spin.setValue(0.1)
            self.colorbar_width_spin.setValue(0.05)
            self.colorbar_height_spin.setValue(0.8)

        self.colorbar_pos_x_spin.blockSignals(False)
        self.colorbar_pos_y_spin.blockSignals(False)
        self.colorbar_width_spin.blockSignals(False)
        self.colorbar_height_spin.blockSignals(False)

        self._render_current_mesh()

    def _toggle_scale_bar(self, checked: bool) -> None:
        """Toggle the scale bar visibility in this solution viewport."""
        if hasattr(self, "vtk_widget") and self.vtk_widget is not None:
            try:
                self.vtk_widget.set_scale_bar_visible(bool(checked))
            except Exception:
                pass

    def _toggle_color_bar(self, checked: bool) -> None:
        """Toggle the scalar color bar visibility."""
        self._colorbar_visible = bool(checked)
        self._render_current_mesh()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _get_display_scalars(self):
        """Get scalar values for display, handling vector components."""
        name = self._current_scalar_name
        mesh = self._current_mesh

        if name is None or mesh is None:
            return None, name

        arr = None
        if name in mesh.point_data:
            arr = mesh.point_data[name]
        elif name in mesh.cell_data:
            arr = mesh.cell_data[name]

        if arr is None:
            return None, name

        # Handle vector fields
        if arr.ndim == 2 and arr.shape[1] == 3:
            import numpy as np
            component = self.component_combo.currentText()

            if component == "X":
                computed = arr[:, 0]
                display_name = f"{name}_X"
            elif component == "Y":
                computed = arr[:, 1]
                display_name = f"{name}_Y"
            elif component == "Z":
                computed = arr[:, 2]
                display_name = f"{name}_Z"
            else:  # Magnitude
                computed = np.linalg.norm(arr, axis=1)
                display_name = f"{name}_mag"

            # Add computed array to mesh
            if name in mesh.point_data:
                mesh.point_data[display_name] = computed
            else:
                mesh.cell_data[display_name] = computed

            return display_name, display_name

        return name, name

    def _add_mesh_to_plotter(self, plotter, mesh) -> None:
        """Add mesh to plotter with current display settings."""
        scalar_name, _ = self._get_display_scalars()

        # Determine color limits
        clim = None
        if scalar_name:
            if self.auto_range_cb.isChecked():
                if getattr(self, "global_range_cb", None) and self.global_range_cb.isChecked():
                    rng = self._compute_global_range(self._current_scalar_name)
                else:
                    rng = self._compute_local_range(self._current_scalar_name)
                if rng is not None and rng[1] > rng[0]:
                    clim = rng
            else:
                vmin = self.scalar_min_spin.value()
                vmax = self.scalar_max_spin.value()
                if vmax > vmin:
                    clim = (vmin, vmax)

        # Scalar bar configuration
        scalar_bar_args = self._build_scalar_bar_args(clim)

        cmap = getattr(self, "_current_cmap", "coolwarm")

        # Representation style
        style = "surface"
        show_edges = self._mesh_show_edges
        if self._mesh_representation == "Wireframe":
            style = "wireframe"
        elif self._mesh_representation == "Points":
            style = "points"

        try:
            kwargs = {
                "scalars": scalar_name if scalar_name else None,
                "clim": clim,
                "cmap": cmap,
                "opacity": self._mesh_opacity,
                "style": style,
                "show_edges": show_edges,
            }
            show_colorbar = bool(scalar_name and getattr(self, "_colorbar_visible", True))
            if show_colorbar:
                kwargs["scalar_bar_args"] = scalar_bar_args
            else:
                kwargs["show_scalar_bar"] = False

            plotter.add_mesh(mesh, **kwargs)
        except Exception:
            pass

    def _build_scalar_bar_args(self, clim=None) -> dict:
        """Build scalar bar arguments dictionary from current UI settings."""
        scalar_bar_args = {}

        # Title
        label = self.scalar_label_edit.text().strip()
        if label:
            scalar_bar_args["title"] = label

        # Orientation
        vertical = True
        if hasattr(self, "orientation_combo"):
            vertical = self.orientation_combo.currentText() == "Vertical"
        scalar_bar_args["vertical"] = vertical

        # Width and height
        if hasattr(self, "colorbar_width_spin"):
            scalar_bar_args["width"] = self.colorbar_width_spin.value()
        if hasattr(self, "colorbar_height_spin"):
            scalar_bar_args["height"] = self.colorbar_height_spin.value()

        # Position
        if hasattr(self, "colorbar_pos_x_spin"):
            scalar_bar_args["position_x"] = self.colorbar_pos_x_spin.value()
        if hasattr(self, "colorbar_pos_y_spin"):
            scalar_bar_args["position_y"] = self.colorbar_pos_y_spin.value()

        # Number of labels
        if hasattr(self, "colorbar_nlabels_spin"):
            scalar_bar_args["n_labels"] = self.colorbar_nlabels_spin.value()

        # Number format
        if hasattr(self, "colorbar_fmt_combo"):
            fmt_text = self.colorbar_fmt_combo.currentText()
            # Extract format string from the combo text
            if "%.3g" in fmt_text:
                fmt = "%.3g"
            elif "%.2f" in fmt_text:
                fmt = "%.2f"
            elif "%.4f" in fmt_text:
                fmt = "%.4f"
            elif "%.2e" in fmt_text:
                fmt = "%.2e"
            elif "%.3e" in fmt_text:
                fmt = "%.3e"
            elif "%d" in fmt_text:
                fmt = "%d"
            elif "%.1f%%" in fmt_text:
                fmt = "%.1f%%"
            else:
                fmt = "%.3g"
            scalar_bar_args["fmt"] = fmt

        # Font family
        if hasattr(self, "colorbar_font_combo"):
            scalar_bar_args["font_family"] = self.colorbar_font_combo.currentText().lower()

        # Font sizes - use larger values for better resolution
        if hasattr(self, "colorbar_title_size_spin"):
            scalar_bar_args["title_font_size"] = self.colorbar_title_size_spin.value()
        if hasattr(self, "colorbar_label_size_spin"):
            scalar_bar_args["label_font_size"] = self.colorbar_label_size_spin.value()

        # Bold text
        if hasattr(self, "colorbar_bold_cb") and self.colorbar_bold_cb.isChecked():
            scalar_bar_args["bold"] = True

        # Shadow for better readability
        if hasattr(self, "colorbar_shadow_cb") and self.colorbar_shadow_cb.isChecked():
            scalar_bar_args["shadow"] = True

        # Italic off by default for cleaner look
        scalar_bar_args["italic"] = False

        # Handle normalized [0, 1] display
        if hasattr(self, "colorbar_normalize_cb") and self.colorbar_normalize_cb.isChecked():
            if clim is not None:
                vmin, vmax = clim
                if vmax > vmin:
                    # Create custom labels for [0, 1] range
                    n_labels = scalar_bar_args.get("n_labels", 5)
                    # PyVista doesn't directly support custom label mapping,
                    # but we can use annotations or adjust the title to indicate normalization
                    original_title = scalar_bar_args.get("title", "")
                    if original_title:
                        scalar_bar_args["title"] = f"{original_title} (normalized)"
                    else:
                        scalar_bar_args["title"] = "Normalized [0, 1]"

        return scalar_bar_args

    def _render_current_mesh(self) -> None:
        plotter = getattr(self.vtk_widget, "plotter", None)
        mesh = self._current_mesh

        if plotter is None:
            return

        plotter.clear()
        if mesh is None:
            plotter.render()
            return

        self._add_mesh_to_plotter(plotter, mesh)

        plotter.reset_camera()
        plotter.render()

        # Re-add glyphs if active
        if self._vector_field_name:
            self._update_glyphs()
