"""Structured diagnostics for surface preparation and TetGen attempts."""

from dataclasses import asdict, dataclass
import re
import signal
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyvista as pv


MAX_CAPTURE_CHARS = 32 * 1024
MAX_DIAGNOSTIC_EXAMPLES = 12


@dataclass(frozen=True)
class SurfaceMeshSummary:
    """Small, serializable description of a candidate surface mesh."""

    n_points: int
    n_cells: int
    n_triangles: int
    n_components: int
    is_all_triangles: bool
    points_finite: bool
    is_manifold: bool
    n_open_edges: int
    bounds: Tuple[float, float, float, float, float, float]
    diagonal: float
    area: float
    volume: Optional[float]


@dataclass(frozen=True)
class TetGenDiagnosticSummary:
    """Parsed, bounded output from one TetGen subprocess."""

    segment_facet_intersections: int
    facet_facet_intersections: int
    missing_segments: int
    missing_subfaces: int
    python_exception: bool
    native_abort: bool
    return_code: int
    signal_name: Optional[str]
    examples: Tuple[str, ...]
    stdout: str
    stderr: str


@dataclass(frozen=True)
class TetGenAttemptReport:
    """Outcome and bounded diagnostics for one surface strategy."""

    strategy: str
    status: str
    surface: SurfaceMeshSummary
    duration_seconds: float
    recoverable: bool
    tetgen_args: Tuple[Any, ...]
    tetgen_kwargs: Dict[str, Any]
    diagnostics: Optional[TetGenDiagnosticSummary]
    message: str


@dataclass
class TetrahedralizationReport:
    """Ordered record of candidate preparation and TetGen attempts."""

    source: SurfaceMeshSummary
    attempts: List[TetGenAttemptReport]
    selected_strategy: Optional[str]
    selected_surface: Optional[SurfaceMeshSummary]
    versions: Dict[str, str]

    def user_summary(self) -> str:
        """Return a concise cause and next action suitable for a dialog."""

        if self.selected_strategy == "original":
            return "TetGen built the interior mesh from the original surface."
        if self.selected_strategy:
            return (
                "TetGen built the interior mesh after automatic surface recovery "
                "using {}.".format(self.selected_strategy)
            )

        diagnostics = [
            attempt.diagnostics
            for attempt in self.attempts
            if attempt.diagnostics is not None
        ]
        if any(
            item.segment_facet_intersections or item.facet_facet_intersections
            for item in diagnostics
        ):
            return (
                "Volume meshing failed because TetGen detected intersecting surface "
                "facets. Inspect the facet and segment identifiers in the technical "
                "details, repair the source surface, and retry."
            )

        messages = " ".join(attempt.message.lower() for attempt in self.attempts)
        if "open edge" in messages or "non-manifold" in messages:
            return (
                "Volume meshing failed because a recovery surface was open or "
                "non-manifold. Repair the source surface and retry."
            )
        if any(not attempt.recoverable for attempt in self.attempts):
            return (
                "Volume meshing failed because the TetGen worker or its environment "
                "failed. Review the technical details and verify the installation."
            )
        return (
            "Volume meshing failed after all safe surface recovery attempts. "
            "Review the technical details, repair the source surface, and retry."
        )

    def detailed_text(self) -> str:
        """Render a bounded attempt report for troubleshooting."""

        lines = ["Tetrahedralization report"]
        if self.versions:
            versions = ", ".join(
                "{}={}".format(key, value) for key, value in sorted(self.versions.items())
            )
            lines.append("Versions: {}".format(versions))
        lines.append("Source: {}".format(_format_surface(self.source)))
        lines.append("Selected strategy: {}".format(self.selected_strategy or "none"))

        for index, attempt in enumerate(self.attempts, start=1):
            lines.append("")
            lines.append(
                "Attempt {}: {} [{}] ({:.3f}s)".format(
                    index,
                    attempt.strategy,
                    attempt.status,
                    attempt.duration_seconds,
                )
            )
            lines.append("Surface: {}".format(_format_surface(attempt.surface)))
            if attempt.tetgen_args:
                lines.append("TetGen args: {}".format(repr(attempt.tetgen_args)))
            if attempt.tetgen_kwargs:
                kwargs = ", ".join(
                    "{}={}".format(key, repr(value))
                    for key, value in sorted(attempt.tetgen_kwargs.items())
                )
                lines.append("TetGen options: {}".format(kwargs))
            lines.append("Message: {}".format(attempt.message))
            diagnostic = attempt.diagnostics
            if diagnostic is None:
                continue
            return_label = str(diagnostic.return_code)
            if diagnostic.signal_name:
                return_label += " ({})".format(diagnostic.signal_name)
            lines.append("Worker return: {}".format(return_label))
            lines.append(
                "Intersections: segment-facet={}, facet-facet={}".format(
                    diagnostic.segment_facet_intersections,
                    diagnostic.facet_facet_intersections,
                )
            )
            if diagnostic.examples:
                lines.append("Representative TetGen diagnostics:")
                lines.extend("  {}".format(line) for line in diagnostic.examples)
            if diagnostic.stdout.strip():
                lines.append("STDOUT:\n{}".format(diagnostic.stdout.rstrip()))
            if diagnostic.stderr.strip():
                lines.append("STDERR:\n{}".format(diagnostic.stderr.rstrip()))

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe report containing no mesh coordinate arrays."""

        return _json_safe(asdict(self))


class TetGenWorkerError(RuntimeError):
    """A typed worker failure carrying a single attempt report."""

    def __init__(self, attempt: TetGenAttemptReport):
        self.attempt = attempt
        super().__init__(attempt.message)

    @property
    def recoverable(self) -> bool:
        return self.attempt.recoverable


class TetrahedralizationError(RuntimeError):
    """Failure of all enabled, safe tetrahedralization attempts."""

    def __init__(self, report: TetrahedralizationReport):
        self.report = report
        super().__init__(report.user_summary())


def summarize_surface(surface: pv.DataSet) -> SurfaceMeshSummary:
    """Return topology and scale information without mutating ``surface``."""

    poly = surface if isinstance(surface, pv.PolyData) else surface.extract_surface()
    bounds = tuple(float(value) for value in poly.bounds)
    lengths = np.asarray(bounds[1::2]) - np.asarray(bounds[::2])
    connected = poly.connectivity()
    region_ids = connected.cell_data.get("RegionId")
    n_components = int(np.unique(region_ids).size) if region_ids is not None else 0
    is_all_triangles = bool(poly.is_all_triangles)

    if is_all_triangles:
        n_triangles = int(poly.n_cells)
    else:
        faces = np.asarray(poly.faces)
        n_triangles = 0
        offset = 0
        while offset < faces.size:
            cell_size = int(faces[offset])
            n_triangles += int(cell_size == 3)
            offset += cell_size + 1

    try:
        volume = float(poly.volume)
    except Exception:
        volume = None

    return SurfaceMeshSummary(
        n_points=int(poly.n_points),
        n_cells=int(poly.n_cells),
        n_triangles=n_triangles,
        n_components=n_components,
        is_all_triangles=is_all_triangles,
        points_finite=bool(np.isfinite(np.asarray(poly.points)).all()),
        is_manifold=bool(poly.is_manifold),
        n_open_edges=int(poly.n_open_edges),
        bounds=bounds,
        diagonal=float(np.linalg.norm(lengths)),
        area=float(poly.area),
        volume=volume,
    )


def _bounded_text(value: str, limit: int = MAX_CAPTURE_CHARS) -> str:
    if len(value) <= limit:
        return value
    omitted = len(value) - limit
    return value[:limit] + "\n...[truncated {} characters]".format(omitted)


def _format_surface(summary: SurfaceMeshSummary) -> str:
    return (
        "{} points, {} cells, {} triangles, {} components, manifold={}, "
        "open_edges={}, diagonal={:.6g}".format(
            summary.n_points,
            summary.n_cells,
            summary.n_triangles,
            summary.n_components,
            summary.is_manifold,
            summary.n_open_edges,
            summary.diagonal,
        )
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _missing_count(text: str, noun: str) -> int:
    pattern = r"\((\d+)\)\s+{}\b[^\n]*\(missing\)".format(re.escape(noun))
    return sum(int(match) for match in re.findall(pattern, text, flags=re.IGNORECASE))


def summarize_tetgen_output(
    stdout: str,
    stderr: str,
    return_code: int,
) -> TetGenDiagnosticSummary:
    """Classify TetGen output while retaining only bounded diagnostic text."""

    stdout = stdout or ""
    stderr = stderr or ""
    combined = stdout + "\n" + stderr
    lower = combined.lower()
    segment_facet = lower.count("a segment and a facet intersect")
    facet_facet = lower.count("two facets exactly intersect")

    signal_name = None
    windows_native_status = 0x80000000 <= return_code <= 0xFFFFFFFF
    if windows_native_status:
        signal_name = "NTSTATUS_0x{:08X}".format(return_code)
    elif return_code < 0:
        try:
            signal_name = signal.Signals(-return_code).name
        except (ValueError, OSError):
            signal_name = "SIGNAL_{}".format(-return_code)

    native_markers = (
        "free():",
        "segmentation fault",
        "core dumped",
        "access violation",
        "malloc():",
    )
    native_abort = (
        return_code < 0
        or windows_native_status
        or any(marker in lower for marker in native_markers)
    )
    python_exception = "traceback (most recent call last)" in lower or "runtimeerror:" in lower

    examples = []
    for line in combined.splitlines():
        normalized = line.strip()
        lowered = normalized.lower()
        if (
            "intersect" in lowered
            or lowered.startswith("segment:")
            or "facet triangle:" in lowered
            or "recovered (missing)" in lowered
        ):
            examples.append(normalized)
        if len(examples) >= MAX_DIAGNOSTIC_EXAMPLES:
            break

    return TetGenDiagnosticSummary(
        segment_facet_intersections=segment_facet,
        facet_facet_intersections=facet_facet,
        missing_segments=_missing_count(combined, "segments"),
        missing_subfaces=_missing_count(combined, "subfaces"),
        python_exception=python_exception,
        native_abort=native_abort,
        return_code=int(return_code),
        signal_name=signal_name,
        examples=tuple(examples),
        stdout=_bounded_text(stdout),
        stderr=_bounded_text(stderr),
    )
