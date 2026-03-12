import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from svv.visualize.gui.vtk_widget import VTKWidget


def test_vtk_widget_disables_vtk_for_offscreen_qt(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.delenv("SVV_GUI_DISABLE_VTK", raising=False)

    app = QApplication.instance() or QApplication([])
    widget = VTKWidget()

    widget._ensure_plotter_initialized()

    assert widget.plotter is None
    assert widget._plotter_init_failed is True
    assert widget._plotter_init_done is False
    assert widget._vtk_placeholder is not None
    assert "offscreen" in widget._vtk_placeholder.text().lower()

    widget.deleteLater()
    app.processEvents()
