"""
svg_loader.py -- Parse an SVG file into color-grouped polylines.

Uses svgelements, which handles nested transforms, shape primitives
(rect/circle/ellipse/line/polyline/polygon/path) and CSS-style colors.
Curves (cubic/quadratic Beziers, arcs) are flattened into short line
segments based on a length tolerance, since GRBL plotting works best
with plain G1 moves.

Text elements are NOT rendered -- convert text to paths in your vector
editor (Inkscape: Path > Object to Path) before loading.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from svgelements import SVG, Shape, Path, Move, Close, Line

from gcode_gen import simplify_polyline

# Any curve is subdivided so no sub-segment is longer than this many
# SVG user-units.  It is re-scaled at load time so that flattening
# quality is roughly constant in *output* millimetres.
DEFAULT_FLATNESS_MM = 0.2


@dataclass
class ColorLayer:
    """All polylines that share one pen color."""
    color: str                                   # '#rrggbb'
    polylines: list = field(default_factory=list)  # list[list[(x, y)]]
    enabled: bool = True

    def total_length(self) -> float:
        total = 0.0
        for line in self.polylines:
            for (x0, y0), (x1, y1) in zip(line, line[1:]):
                total += math.hypot(x1 - x0, y1 - y0)
        return total


@dataclass
class Artwork:
    """A parsed SVG: layers keyed by color, plus its bounding box."""
    layers: list = field(default_factory=list)   # list[ColorLayer]
    min_x: float = 0.0
    min_y: float = 0.0
    max_x: float = 0.0
    max_y: float = 0.0

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_y - self.min_y


def _element_color(element) -> str:
    """Pick the pen color for a shape: stroke first, then fill, then black."""
    for attr in (element.stroke, element.fill):
        if attr is not None and attr.value is not None:
            hexv = attr.hexrgb.lower()
            if hexv != "#ffffff" or attr is element.stroke:
                return hexv
    return "#000000"


def _approx_seg_length(seg) -> float:
    """Cheap upper-bound length via the control polygon (no integration)."""
    pts = []
    for attr in ("start", "control1", "control2", "control", "end"):
        p = getattr(seg, attr, None)
        if p is not None:
            pts.append((p.x, p.y))
    if len(pts) >= 2:
        return sum(math.hypot(x1 - x0, y1 - y0)
                   for (x0, y0), (x1, y1) in zip(pts, pts[1:]))
    try:
        return seg.length(error=1e-2)
    except TypeError:
        return seg.length()


def _flatten_path(path: Path, flatness: float) -> list:
    """Convert one svgelements Path into a list of polylines.

    Curvature-aware: every curve is oversampled well below the target
    tolerance (with a floor of 8 samples so small tight arcs cannot be
    under-sampled into visible facets), then Douglas-Peucker keeps
    points only where curvature demands them. The result is dense on
    tight curves, sparse on gentle ones, with chord error <= flatness.
    """
    polylines = []
    current = []

    def emit():
        nonlocal current
        if len(current) >= 2:
            polylines.append(current)
        current = []

    for seg in path:
        if isinstance(seg, Move):
            emit()
            if seg.end is not None:
                current = [(seg.end.x, seg.end.y)]
        elif isinstance(seg, (Line, Close)):
            if seg.end is not None:
                if not current and seg.start is not None:
                    current = [(seg.start.x, seg.start.y)]
                current.append((seg.end.x, seg.end.y))
        else:
            # CubicBezier / QuadraticBezier / Arc
            if not current and seg.start is not None:
                current = [(seg.start.x, seg.start.y)]
            seg_len = _approx_seg_length(seg)
            steps = int(math.ceil(math.sqrt(seg_len / max(0.5 * flatness,
                                                          1e-9))))
            steps = min(max(steps, 8), 4000)
            ts = np.linspace(0.0, 1.0, steps + 1)[1:]
            pts = seg.npoint(ts)
            current.extend(zip(pts[:, 0].tolist(), pts[:, 1].tolist()))
    emit()
    return [simplify_polyline(pl, flatness) for pl in polylines]


def load_svg(filepath: str, flatness_mm: float = DEFAULT_FLATNESS_MM,
             expected_output_mm: float = 300.0) -> Artwork:
    """
    Parse *filepath* into an Artwork.

    flatness_mm is the desired curve tolerance in output millimetres;
    it is converted into user-units using a first pass over the bbox.
    """
    svg = SVG.parse(filepath, reify=True)

    shapes = [el for el in svg.elements()
              if isinstance(el, Shape) and not isinstance(el, SVG)]

    # First pass: rough bbox to estimate units-per-mm for flattening.
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    paths = []
    for el in shapes:
        try:
            path = Path(el)
        except Exception:
            continue
        if len(path) == 0:
            continue
        bbox = path.bbox()
        if bbox is None:
            continue
        x0, y0, x1, y1 = bbox
        min_x, min_y = min(min_x, x0), min(min_y, y0)
        max_x, max_y = max(max_x, x1), max(max_y, y1)
        paths.append((path, _element_color(el)))

    if not paths:
        raise ValueError("No drawable vector shapes found in this SVG. "
                         "(Text must be converted to paths first.)")

    span = max(max_x - min_x, max_y - min_y, 1e-6)
    units_per_mm = span / max(expected_output_mm, 1.0)
    flatness_units = max(flatness_mm * units_per_mm, span * 1e-5)

    # Second pass: flatten and group by color.
    by_color: dict = {}
    for path, color in paths:
        for polyline in _flatten_path(path, flatness_units):
            by_color.setdefault(color, []).append(polyline)

    layers = [ColorLayer(color=c, polylines=pls)
              for c, pls in sorted(by_color.items(),
                                   key=lambda kv: -len(kv[1]))]
    return Artwork(layers=layers, min_x=min_x, min_y=min_y,
                   max_x=max_x, max_y=max_y)
