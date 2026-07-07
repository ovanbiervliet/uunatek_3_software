"""
gcode_gen.py -- Turn color layers into GRBL-flavoured G-code.

Coordinate conventions for the UUNA TEK / iDraw DrawCore boards:
  * Firmware is GRBL 1.1 over USB serial (115200 baud).
  * The pen lift is a real Z axis (0..~10 mm). Z0 is the pen RAISED
    and larger Z lowers the pen to the paper. Both are configurable.
  * The machine homes ($H) to a corner; users typically set the
    front-left corner of the paper as X0 Y0 with G92.
  * SVG Y grows downward, machine Y grows away from the operator,
    so Y is flipped by default.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class PlotSettings:
    out_width_mm: float = 304.8     # artwork width on paper (12 in)
    out_height_mm: float = 304.8    # artwork height on paper
    offset_x_mm: float = 0.0        # margin from machine X0
    offset_y_mm: float = 0.0        # margin from machine Y0
    flip_x: bool = False            # mirror machine X
    flip_y: bool = False            # mirror machine Y
    pen_up_z: float = 0.0           # mm (Z0 = raised on iDraw)
    pen_down_z: float = 10.0        # mm (pen on paper)
    draw_feed: float = 10000.0      # mm/min for drawing moves
    travel_feed: float = 8000.0     # mm/min for pen-up moves (G1-limited)
    z_feed: float = 2500.0          # mm/min for pen lift/drop
    use_rapids: bool = True         # G0 for travel instead of G1
    simplify_mm: float = 0.05       # merge near-collinear points (0 = off)


def simplify_polyline(points: list, epsilon: float) -> list:
    """
    Ramer-Douglas-Peucker simplification (iterative, numpy).

    Uniform curve sampling leaves many nearly-collinear points; merging
    them into longer segments matters for speed: GRBL only accelerates
    to what it can brake from within its ~15-move lookahead buffer, so
    longer segments let the planner actually reach the commanded feed.
    Max deviation from the original path is bounded by *epsilon*.
    """
    n = len(points)
    if epsilon <= 0 or n < 3:
        return points
    pts = np.asarray(points, dtype=float)
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        seg = pts[a + 1:b]
        d = pts[b] - pts[a]
        norm = math.hypot(d[0], d[1])
        if norm < 1e-12:
            dist = np.hypot(seg[:, 0] - pts[a][0], seg[:, 1] - pts[a][1])
        else:
            dist = np.abs((seg[:, 0] - pts[a][0]) * d[1]
                          - (seg[:, 1] - pts[a][1]) * d[0]) / norm
        i = int(np.argmax(dist))
        if dist[i] > epsilon:
            idx = a + 1 + i
            keep[idx] = True
            stack.append((a, idx))
            stack.append((idx, b))
    return [tuple(p) for p in pts[keep]]


def make_transform(artwork, s: PlotSettings):
    """
    Return f(x, y) -> (mx, my) mapping SVG user units to machine mm.
    flip_x / flip_y mirror the output to match the machine's homed
    orientation; the on-screen preview is unaffected by them.
    """
    sx = s.out_width_mm / max(artwork.width, 1e-9)
    sy = s.out_height_mm / max(artwork.height, 1e-9)

    def transform(x: float, y: float):
        if s.flip_x:
            mx = (artwork.max_x - x) * sx + s.offset_x_mm
        else:
            mx = (x - artwork.min_x) * sx + s.offset_x_mm
        if s.flip_y:
            my = (artwork.max_y - y) * sy + s.offset_y_mm
        else:
            my = (y - artwork.min_y) * sy + s.offset_y_mm
        return mx, my

    return transform


def order_polylines(polylines: list) -> list:
    """
    Greedy nearest-neighbour ordering to cut pen-up travel time.
    Also reverses polylines when their far end is closer.
    """
    remaining = list(polylines)
    ordered = []
    cx, cy = 0.0, 0.0
    while remaining:
        best_i, best_rev, best_d = 0, False, float("inf")
        for i, line in enumerate(remaining):
            hx, hy = line[0]
            tx, ty = line[-1]
            dh = (hx - cx) ** 2 + (hy - cy) ** 2
            dt = (tx - cx) ** 2 + (ty - cy) ** 2
            if dh < best_d:
                best_i, best_rev, best_d = i, False, dh
            if dt < best_d:
                best_i, best_rev, best_d = i, True, dt
        line = remaining.pop(best_i)
        if best_rev:
            line = line[::-1]
        ordered.append(line)
        cx, cy = line[-1]
    return ordered


def _fmt(v: float) -> str:
    return f"{v:.3f}".rstrip("0").rstrip(".")


def generate_layer_gcode(layer, artwork, s: PlotSettings) -> list:
    """G-code lines for one color layer (no header/footer)."""
    tf = make_transform(artwork, s)
    lines = []
    travel_cmd = "G0" if s.use_rapids else f"G1 F{_fmt(s.travel_feed)}"

    machine_polylines = []
    for pl in layer.polylines:
        mp = [tf(x, y) for x, y in pl]
        machine_polylines.append(simplify_polyline(mp, s.simplify_mm))

    for pl in order_polylines(machine_polylines):
        x0, y0 = pl[0]
        lines.append(f"{travel_cmd} X{_fmt(x0)} Y{_fmt(y0)}")
        lines.append(f"G1 Z{_fmt(s.pen_down_z)} F{_fmt(s.z_feed)}")
        first = True
        for x, y in pl[1:]:
            feed = f" F{_fmt(s.draw_feed)}" if first else ""
            lines.append(f"G1 X{_fmt(x)} Y{_fmt(y)}{feed}")
            first = False
        lines.append(f"G1 Z{_fmt(s.pen_up_z)} F{_fmt(s.z_feed)}")
    return lines


def header_gcode(s: PlotSettings) -> list:
    return [
        "G21",                       # millimetres
        "G90",                       # absolute positioning
        f"G1 Z{_fmt(s.pen_up_z)} F{_fmt(s.z_feed)}",   # make sure pen is up
    ]


def footer_gcode(s: PlotSettings) -> list:
    return [
        f"G1 Z{_fmt(s.pen_up_z)} F{_fmt(s.z_feed)}",
        "G0 X0 Y0",                  # park at work origin
    ]


def segment_stats(gcode_lines: list) -> tuple:
    """(count, avg_len_mm) of drawing segments in generated G-code."""
    import re as _re
    count, total = 0, 0.0
    last = None
    for line in gcode_lines:
        if line.startswith("G1 X") or line.startswith("G0 X"):
            m = _re.match(r"G[01] X(-?[\d.]+) Y(-?[\d.]+)", line)
            if not m:
                continue
            pt = (float(m.group(1)), float(m.group(2)))
            if line.startswith("G1 X") and last is not None:
                count += 1
                total += math.hypot(pt[0] - last[0], pt[1] - last[1])
            last = pt
    return count, (total / count if count else 0.0)


def estimate_time_minutes(layer, artwork, s: PlotSettings) -> float:
    """Very rough plot-time estimate for one layer."""
    tf = make_transform(artwork, s)
    draw = 0.0
    for pl in layer.polylines:
        pts = [tf(x, y) for x, y in pl]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            draw += math.hypot(x1 - x0, y1 - y0)
    lifts = len(layer.polylines)
    z_travel = lifts * 2 * abs(s.pen_up_z - s.pen_down_z)
    return draw / max(s.draw_feed, 1) + z_travel / max(s.z_feed, 1)
