"""User-facing feedback for Domain tetrahedralization outcomes."""

from dataclasses import dataclass, replace
import re
from typing import Any, Dict, Optional

from svv.domain.routines.mesh_diagnostics import TetrahedralizationReport


_QUOTED_WINDOWS_PATH = re.compile(
    r'''(?i)(?P<quote>["'])(?:[a-z]:[\\/]|\\\\)[^"']+(?P=quote)'''
)
_QUOTED_UNIX_PATH = re.compile(
    r'''(?P<quote>["'])/(?!/)[^"']+(?P=quote)'''
)
_WINDOWS_EXTENDED_PATH = re.compile(
    r'''(?i)(?<!\\)\\\\\?\\(?:UNC\\[^\\\s:"']+\\[^\\\s:"']+|[a-z]:\\[^\\\s:"']+)(?:\\[^\\\s:"']+)*'''
)
_WINDOWS_UNC_PATH = re.compile(
    r'''(?i)(?<!\\)\\\\[^\\\s:"']+\\[^\\\s:"']+(?:\\[^\\\s:"']+)*'''
)
_WINDOWS_DRIVE_PATH = re.compile(
    r'''(?i)(?<![\w])(?:[a-z]:[\\/])(?:[^\\/\s:"']+[\\/])*[^\\/\s:"']+'''
)
_UNIX_PATH = re.compile(
    r'''(?<![\w.])/(?!/)(?:[^/\s:"']+/)*[^/\s:"']+'''
)


@dataclass(frozen=True)
class DomainBuildFeedback:
    """Concise and detailed text for one Domain build outcome."""

    text: str
    informative_text: str
    detailed_text: str
    status: str
    recovered: bool


def sanitize_local_paths(value: str) -> str:
    """Replace absolute local filesystem paths in diagnostic text."""

    def replace_quoted(match):
        quote = match.group("quote")
        return "{}<local-path>{}".format(quote, quote)

    sanitized = _QUOTED_WINDOWS_PATH.sub(replace_quoted, str(value))
    sanitized = _QUOTED_UNIX_PATH.sub(replace_quoted, sanitized)
    sanitized = _WINDOWS_EXTENDED_PATH.sub("<local-path>", sanitized)
    sanitized = _WINDOWS_UNC_PATH.sub("<local-path>", sanitized)
    sanitized = _WINDOWS_DRIVE_PATH.sub("<local-path>", sanitized)
    return _UNIX_PATH.sub("<local-path>", sanitized)


def extract_build_report(exception=None, domain=None) -> Optional[TetrahedralizationReport]:
    """Find a structured report on an exception or completed Domain."""

    report = getattr(exception, "report", None)
    if report is not None:
        return report
    attempt = getattr(exception, "attempt", None)
    if attempt is not None:
        return TetrahedralizationReport(
            source=attempt.surface,
            attempts=[attempt],
            selected_strategy=None,
            selected_surface=None,
            versions={},
        )
    return getattr(domain, "mesh_build_report", None)


def _report_without_duplicate_tracebacks(report):
    seen_tracebacks = set()
    attempts = []
    for attempt in report.attempts:
        diagnostic = attempt.diagnostics
        if diagnostic is None:
            attempts.append(attempt)
            continue
        replacements = {}
        for stream_name in ("stdout", "stderr"):
            stream = getattr(diagnostic, stream_name)
            if "traceback (most recent call last)" not in stream.lower():
                continue
            signature = stream.strip()
            if signature in seen_tracebacks:
                replacements[stream_name] = "[duplicate Python traceback omitted]"
            else:
                seen_tracebacks.add(signature)
        if replacements:
            diagnostic = replace(diagnostic, **replacements)
            attempt = replace(attempt, diagnostics=diagnostic)
        attempts.append(attempt)
    return TetrahedralizationReport(
        source=report.source,
        attempts=attempts,
        selected_strategy=report.selected_strategy,
        selected_surface=report.selected_surface,
        versions=dict(report.versions),
    )


def build_domain_feedback(
    *,
    report: Optional[TetrahedralizationReport] = None,
    exception=None,
    success: bool,
) -> DomainBuildFeedback:
    """Format a Domain build result for status and optional warning display."""

    if report is None:
        report = extract_build_report(exception=exception)

    if success:
        recovered = bool(
            report is not None
            and report.selected_strategy
            and report.selected_strategy != "original"
        )
        if recovered:
            strategy = report.selected_strategy
            status = "Domain built successfully after surface repair ({})".format(
                strategy
            )
            informative = report.user_summary()
        else:
            status = "Domain built successfully"
            informative = (
                report.user_summary()
                if report is not None
                else "The interior mesh was built successfully."
            )
        details = report.detailed_text() if report is not None else ""
        return DomainBuildFeedback(
            text="The domain and its interior mesh were built successfully.",
            informative_text=sanitize_local_paths(informative),
            detailed_text=sanitize_local_paths(details),
            status=status,
            recovered=recovered,
        )

    if report is not None:
        display_report = _report_without_duplicate_tracebacks(report)
        informative = display_report.user_summary()
        details = display_report.detailed_text()
    else:
        informative = (
            "Volume meshing failed. Review the technical details, verify the "
            "meshing installation, and inspect the source surface."
        )
        details = "{}: {}".format(
            type(exception).__name__ if exception is not None else "Error",
            exception if exception is not None else "No additional details were recorded.",
        )
    return DomainBuildFeedback(
        text="The domain loaded, but its interior mesh could not be built.",
        informative_text=sanitize_local_paths(informative),
        detailed_text=sanitize_local_paths(details),
        status="Domain loaded without an interior mesh",
        recovered=False,
    )


def apply_feedback_to_message_box(message_box, feedback: DomainBuildFeedback) -> None:
    """Populate a QMessageBox-compatible object with structured feedback."""

    box_type = type(message_box)
    message_box.setWindowTitle("Domain Build Warning")
    message_box.setIcon(box_type.Warning)
    message_box.setText(feedback.text)
    message_box.setInformativeText(feedback.informative_text)
    message_box.setDetailedText(feedback.detailed_text)
    message_box.setStandardButtons(box_type.Ok)


def _sanitize_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _sanitize_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_payload(item) for item in value]
    if isinstance(value, str):
        return sanitize_local_paths(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def report_for_telemetry(report: Optional[TetrahedralizationReport]) -> Optional[Dict[str, Any]]:
    """Return a path-sanitized, JSON-safe report without mesh arrays."""

    if report is None:
        return None
    return _sanitize_payload(report.to_dict())
