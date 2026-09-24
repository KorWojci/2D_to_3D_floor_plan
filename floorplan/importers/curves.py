"""Helpers for recognising circular arcs in sampled curves (SVG/PDF Béziers)."""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np


def fit_arc(pts: Sequence[tuple[float, float]], rel_tol: float = 0.01) -> Optional[tuple[float, float, float, float, float]]:
    """Least-squares circle fit.  Returns (cx, cy, r, a0, a1) with CCW angles in degrees."""
    if len(pts) < 5:
        return None
    a = np.asarray(pts, dtype=float)
    chord = np.linalg.norm(a[-1] - a[0])
    x, y = a[:, 0], a[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones(len(a))])
    b = x * x + y * y
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy, c = sol
    r2 = c + cx * cx + cy * cy
    if r2 <= 0:
        return None
    r = math.sqrt(r2)
    resid = np.abs(np.hypot(x - cx, y - cy) - r)
    if resid.max() > rel_tol * r or chord < 1e-9:
        return None
    angs = np.degrees(np.arctan2(y - cy, x - cx))
    # determine traversal direction
    d = np.diff(np.unwrap(np.radians(angs)))
    ccw = d.sum() > 0
    sweep = abs(math.degrees(d.sum()))
    if sweep < 10 or sweep > 359:
        return None
    if ccw:
        return cx, cy, r, angs[0] % 360, angs[-1] % 360
    return cx, cy, r, angs[-1] % 360, angs[0] % 360


def polyline_arc(pts: Sequence[tuple[float, float]]) -> Optional[tuple[float, float, float, float, float]]:
    """Recognise a flattened circular arc (many short, evenly turning segments)."""
    if len(pts) < 7:
        return None
    arc = fit_arc(pts, rel_tol=0.02)
    if arc is None:
        return None
    a = np.asarray(pts, dtype=float)
    seg = np.hypot(*np.diff(a, axis=0).T)
    if seg.max() > 0.4 * arc[2]:
        return None
    return arc
