import importlib
import json
from queue import Queue
import sys
import threading
from types import SimpleNamespace

import pyvista as pv
import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

from svv.domain.routines.mesh_diagnostics import (
    TetGenAttemptReport,
    TetrahedralizationError,
    TetrahedralizationReport,
    summarize_surface,
    summarize_tetgen_output,
)
from svv.visualize.gui.domain_build_feedback import (
    apply_feedback_to_message_box,
    build_domain_feedback,
    report_for_telemetry,
    sanitize_local_paths,
)


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


def _attempt(
    *,
    strategy="original",
    status="failed",
    recoverable=True,
    message="TetGen rejected the surface.",
    diagnostics=None,
):
    return TetGenAttemptReport(
        strategy=strategy,
        status=status,
        surface=summarize_surface(pv.Cube().triangulate()),
        duration_seconds=1.25,
        recoverable=recoverable,
        tetgen_args=(),
        tetgen_kwargs={"order": 1, "nobisect": True},
        diagnostics=diagnostics,
        message=message,
    )


def _report(attempts, selected_strategy=None):
    surface = summarize_surface(pv.Cube().triangulate())
    return TetrahedralizationReport(
        source=surface,
        attempts=list(attempts),
        selected_strategy=selected_strategy,
        selected_surface=surface if selected_strategy else None,
        versions={"tetgen": "0.6.4", "pyvista": "0.46.5"},
    )


def test_recovered_success_feedback_names_automatic_strategy():
    report = _report(
        [
            _attempt(),
            _attempt(strategy="meshfix", status="succeeded", recoverable=False),
        ],
        selected_strategy="meshfix",
    )

    feedback = build_domain_feedback(report=report, success=True)

    assert feedback.recovered is True
    assert "successfully after surface repair" in feedback.status.lower()
    assert "meshfix" in feedback.status.lower()
    assert "automatic surface recovery" in feedback.informative_text.lower()


def test_intersection_failure_names_cause_and_action():
    diagnostics = summarize_tetgen_output(
        "Warning: Two facets exactly intersect.\n"
        "  1st facet triangle: [1,2,3] tag(-1).\n"
        "  2nd facet triangle: [4,5,6] tag(-1).\n",
        "free(): invalid next size (normal)\n",
        -6,
    )
    report = _report([_attempt(diagnostics=diagnostics)])

    feedback = build_domain_feedback(
        exception=TetrahedralizationError(report),
        report=report,
        success=False,
    )

    assert "intersecting surface facets" in feedback.informative_text.lower()
    assert "repair the source surface" in feedback.informative_text.lower()
    assert "facet triangle: [1,2,3]" in feedback.detailed_text


def test_open_nonmanifold_fallback_names_rejected_geometry():
    report = _report(
        [
            _attempt(),
            _attempt(
                strategy="pyacvd",
                status="rejected",
                message="Recovery surface is non-manifold and has 16 open edges.",
                diagnostics=None,
            ),
        ]
    )

    feedback = build_domain_feedback(report=report, success=False)

    assert "open or non-manifold" in feedback.informative_text.lower()
    assert "repair the source surface" in feedback.informative_text.lower()


def test_infrastructure_failure_does_not_blame_source_geometry():
    diagnostics = summarize_tetgen_output(
        "",
        "ModuleNotFoundError: No module named 'tetgen'\n",
        1,
    )
    report = _report(
        [
            _attempt(
                status="infrastructure-error",
                recoverable=False,
                message="TetGen worker dependency failed.",
                diagnostics=diagnostics,
            )
        ]
    )

    feedback = build_domain_feedback(report=report, success=False)

    assert "worker or its environment failed" in feedback.informative_text.lower()
    assert "verify the installation" in feedback.informative_text.lower()
    assert "intersecting" not in feedback.informative_text.lower()


def test_details_are_bounded_path_sanitized_and_deduplicate_tracebacks():
    traceback_text = (
        "Traceback (most recent call last):\n"
        "  File \"/home/person/private/project/worker.py\", line 9, in <module>\n"
        "RuntimeError: Failed to tetrahedralize\n"
    )
    diagnostics = summarize_tetgen_output("x" * 40000, traceback_text, 1)
    report = _report(
        [
            _attempt(strategy="original", diagnostics=diagnostics),
            _attempt(strategy="meshfix", diagnostics=diagnostics),
        ]
    )

    feedback = build_domain_feedback(report=report, success=False)

    assert "Attempt 1: original" in feedback.detailed_text
    assert "Attempt 2: meshfix" in feedback.detailed_text
    assert len(feedback.detailed_text) < 70000
    assert feedback.detailed_text.count("Traceback (most recent call last)") == 1
    assert "/home/person/private" not in feedback.detailed_text
    assert "<local-path>" in feedback.detailed_text
    assert "array([" not in feedback.detailed_text


def test_message_box_receives_short_informative_and_detailed_fields(qt_app):
    report = _report([_attempt()])
    feedback = build_domain_feedback(report=report, success=False)
    box = QMessageBox()

    apply_feedback_to_message_box(box, feedback)

    assert box.windowTitle() == "Domain Build Warning"
    assert box.text() == "The domain loaded, but its interior mesh could not be built."
    assert box.informativeText() == feedback.informative_text
    assert box.detailedText() == feedback.detailed_text
    assert box.standardButtons() == QMessageBox.Ok


def test_telemetry_report_is_json_safe_and_removes_local_paths():
    diagnostics = summarize_tetgen_output(
        "",
        "Failure while reading /home/person/private/surface.stl\n",
        1,
    )
    report = _report([_attempt(diagnostics=diagnostics)])

    payload = report_for_telemetry(report)

    json.dumps(payload)
    serialized = json.dumps(payload)
    assert "/home/person/private" not in serialized
    assert "<local-path>" in serialized
    assert "points" not in payload
    assert "faces" not in payload


@pytest.mark.parametrize(
    "local_path",
    [
        "/private.stl",
        r"C:\private.stl",
        "C:/private.stl",
        r"\\server\share\private.stl",
        r"\\?\C:\private.stl",
        r"\\?\UNC\server\share\private.stl",
    ],
)
def test_path_sanitizer_removes_root_drive_and_unc_paths(local_path):
    sanitized = sanitize_local_paths("Failure while reading {}".format(local_path))

    assert sanitized == "Failure while reading <local-path>"


@pytest.mark.parametrize(
    "local_path",
    [
        "/home/person/My Files/private.stl",
        r"C:\Users\Person Name\private.stl",
        r"\\server\share\Private Files\private.stl",
    ],
)
def test_path_sanitizer_removes_quoted_paths_containing_spaces(local_path):
    sanitized = sanitize_local_paths('Failure while reading "{}"'.format(local_path))

    assert sanitized == 'Failure while reading "<local-path>"'


def test_structured_telemetry_honors_disabled_gate(monkeypatch):
    main_window_mod = importlib.import_module("svv.visualize.gui.main_window")
    sentry_calls = []

    class FakeScope:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def set_tag(self, *args):
            sentry_calls.append(("tag", args))

        def set_extra(self, *args):
            sentry_calls.append(("extra", args))

        def set_context(self, *args):
            sentry_calls.append(("context", args))

    fake_sentry = SimpleNamespace(
        push_scope=lambda: FakeScope(),
        capture_exception=lambda *args, **kwargs: sentry_calls.append(
            ("exception", args, kwargs)
        ),
        capture_message=lambda *args, **kwargs: sentry_calls.append(
            ("message", args, kwargs)
        ),
        flush=lambda *args, **kwargs: sentry_calls.append(("flush", args, kwargs)),
    )
    monkeypatch.setitem(sys.modules, "sentry_sdk", fake_sentry)
    monkeypatch.setattr(main_window_mod, "telemetry_enabled", lambda: False, raising=False)
    monkeypatch.setattr(
        main_window_mod,
        "capture_exception",
        lambda *args, **kwargs: sentry_calls.append(("wrapped-exception", args, kwargs)),
    )
    monkeypatch.setattr(
        main_window_mod,
        "capture_message",
        lambda *args, **kwargs: sentry_calls.append(("wrapped-message", args, kwargs)),
    )

    main_window_mod.VascularizeGUI._record_telemetry(
        object(),
        RuntimeError("disabled"),
        telemetry_context={"tetrahedralization": {"selected_strategy": "meshfix"}},
    )
    main_window_mod.VascularizeGUI._record_telemetry(
        object(),
        message="disabled",
        telemetry_context={"tetrahedralization": {"selected_strategy": "meshfix"}},
    )

    assert sentry_calls == []


class _LoaderHarness:
    def __init__(self):
        self.telemetry = []

    def _record_telemetry(self, *args, **kwargs):
        self.telemetry.append((args, kwargs))


def _progress_labels(progress_queue):
    labels = []
    while not progress_queue.empty():
        item = progress_queue.get_nowait()
        if isinstance(item, dict) and item.get("label"):
            labels.append(item["label"])
    return labels


def test_mesh_background_load_preserves_success_report(monkeypatch):
    main_window_mod = importlib.import_module("svv.visualize.gui.main_window")
    domain_module = importlib.import_module("svv.domain.domain")
    report = _report(
        [_attempt(strategy="meshfix", status="succeeded", recoverable=False)],
        selected_strategy="meshfix",
    )

    class FakeDomain:
        def __init__(self, mesh):
            self.mesh = None
            self.mesh_tree = None
            self.boundary = None
            self.patches = []
            self.mesh_build_report = None

        def create(self, progress_callback=None):
            return None

        def solve(self, progress_callback=None):
            return None

        def build(self, resolution=25, progress_callback=None):
            self.mesh_build_report = report
            self.boundary = object()

    monkeypatch.setattr(domain_module, "Domain", FakeDomain)
    monkeypatch.setattr(pv, "read", lambda path: object())
    harness = _LoaderHarness()
    progress = Queue()

    result = main_window_mod.VascularizeGUI._load_domain_file(
        harness,
        "/tmp/surface.stl",
        threading.Event(),
        progress,
        25,
    )

    assert result._build_failed is False
    assert result._build_error is None
    assert result._build_exception is None
    assert result._build_report is report
    assert any("successfully after surface repair" in label.lower() for label in _progress_labels(progress))
    assert harness.telemetry[0][1]["telemetry_context"]["tetrahedralization"]["selected_strategy"] == "meshfix"


def test_dmn_background_load_preserves_structured_failure(monkeypatch):
    main_window_mod = importlib.import_module("svv.visualize.gui.main_window")
    domain_module = importlib.import_module("svv.domain.domain")
    diagnostics = summarize_tetgen_output(
        "Warning: A segment and a facet intersect.\n",
        "free(): invalid next size (normal)\n",
        -6,
    )
    report = _report([_attempt(diagnostics=diagnostics)])
    build_exception = TetrahedralizationError(report)

    class FakeDomain:
        def __init__(self):
            self.mesh = None
            self.mesh_tree = None
            self.boundary = None
            self.patches = []
            self.mesh_build_report = None

        @classmethod
        def load(cls, path):
            return cls()

        def build(self, resolution=25, progress_callback=None):
            raise build_exception

    monkeypatch.setattr(domain_module, "Domain", FakeDomain)
    harness = _LoaderHarness()

    result = main_window_mod.VascularizeGUI._load_domain_file(
        harness,
        "/tmp/domain.dmn",
        threading.Event(),
        Queue(),
        25,
    )

    assert result._build_failed is True
    assert "intersecting surface facets" in result._build_error.lower()
    assert result._build_exception is build_exception
    assert result._build_report is report
    telemetry = harness.telemetry[0][1]
    assert telemetry["action"] == "load_domain_build"
    assert telemetry["telemetry_context"]["tetrahedralization"]["attempts"][0]["strategy"] == "original"
